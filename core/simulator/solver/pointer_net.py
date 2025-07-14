from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .base import BaseSolver

import torch.optim as optim
import torch.nn as nn
import numpy as np
import simulator
import torch


class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        # create actor/critic model
        self.create_model()

    def create_model(self):
        args = self.args
        self.actor  = simulator.pointer_net.Actor(args).to(device=args.device)
        self.critic = simulator.pointer_net.Critic(args).to(device=args.device)
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = \
            optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()),
                       lr=args.lr)
        self.actor_scheduler = \
            optim.lr_scheduler.StepLR(self.actor_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)
        self.critic_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                          lr=args.lr)
        self.critic_scheduler = \
            optim.lr_scheduler.StepLR(self.critic_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)

    def select_action(self, z, h, c, ref, mask):
        # extract args
        args = self.args
        assert mask.shape[0] == z.shape[0]
        assert mask.shape[0] == h.shape[1]
        assert mask.shape[0] == c.shape[1]
        assert mask.shape[0] == ref.shape[0]
        # decoder step
        d, (h, c) = self.actor.decoder(z, h, c) # d: n_subproblem, 1, n_embed
        q = h.squeeze(0) # q:= n_subproblem, n_embed # TODO: note original implementation h.squeeze(0)
        # glimpse step
        for _ in range(args.n_glimpse):
            q = self.actor.glimpse(q, ref, mask)
        # pointing using attention
        logits = self.actor.pointer(q, ref, mask) # logits: n_subproblem, n_node
        log_prob = torch.log_softmax(logits, dim=-1) # log_prob: n_subproblem, n_node
        # select next node
        action = torch.argmax(log_prob, dim=1).long() # next_node: n_subproblem
        # compute log_likelihood
        log_likelihood = self.actor.get_log_likelihood(log_prob, action)
        return log_likelihood, action, (d, h, c)

    def reset(self):
        # extract args
        args = self.args
        env = self.env
        critic = self.critic
        actor = self.actor
        # reset environment
        observation, self.edge_index, self.edge_weight = env.reset()
        # input
        x = observation.to(args.device) # n_subproblem, n_node, n_input
        # estimate value:
        ew = self.edge_weight.to(args.device)
        ei = self.edge_index.to(args.device)
        l_hat = critic(x, ei, ew)
        # embed
        e = self.actor.embedding(x, ei, ew) # n_subproblem, n_node, n_embed
        # encode
        ref, (h, c) = self.actor.encoder(e) # ref: n_subproblem, n_node, n_embed
        # decode loop
        z = self.actor.decoder.get_z0(x) # n_subproblem, 1, n_embed
        return x, e, ref, h, c, z, l_hat

    def step(self, x, e, ref, h, c, z):
        # extract args
        args = self.args
        env = self.env
        # extract mask
        mask = x[:, :, -1] # 1' entries are non selectable
        # print(f'{mask.shape=} {z.shape=} {h.shape=} {c.shape=} {x.shape=} {ref.shape=}')
        # select action
        log_likelihood, action, (d, h, c) = self.select_action(z, h, c, ref, mask)
        # TODO: HERE, update ref depending on action
        # and split subproblems
        action = action.detach()
        next_observation, reward, done, info = env.step(action.cpu().tolist())
        indices = info['subproblem_indices']
        # split z:
        e = simulator.util.split_subproblem(e, indices)
        action = simulator.util.split_subproblem(action, indices)
        z = self.actor.decoder.gather_z(e, action)
        h = simulator.util.split_subproblem(h.transpose(0, 1), indices).transpose(0, 1)
        c = simulator.util.split_subproblem(c.transpose(0, 1), indices).transpose(0, 1)
        ref = simulator.util.split_subproblem(ref, indices)
        #
        n_subproblem = len(next_observation)
        info.update({
            'avg_reward': reward.mean().item(),
            'n_subproblem': n_subproblem,
            'done': done,
        })
        x = next_observation.to(args.device)
        return x, e, ref, h, c, z, log_likelihood, reward, info

    def test(self):
        # extract args
        monitor = self.monitor
        args = self.args
        env  = self.env
        self.actor.eval()
        # procedure before test
        print(f'[+] testing {monitor.label}')
        # TODO: env.reset(), change to another model
        x, e, ref, h, c, z, l_hat = self.reset()
        # initial
        done = False
        step = 0
        total_reward = 0
        while not done:
            x, e, ref, h, c, z, log_likelihood, reward, info = self.step(x, e, ref, h, c, z)
            monitor.step(info)
            step += 1
            total_reward += reward.sum().item()
            info['step'] = step
            info['total_reward'] = total_reward
            if step >= args.ns_n_step:
                break
        monitor.export_csv()

    def train(self):
        # extract args
        monitor = self.monitor
        args = self.args
        env  = self.env
        self.actor.train()
        self.critic.train()
        # procedure before test
        print(f'[+] training {monitor.label}')
        for episode in range(args.n_train_episode):
            # TODO: env.reset(), change to another model
            x, e, ref, h, c, z, l_hat = self.reset()
            # initial
            done = False
            step = 0
            total_reward = 0
            log_likelihood = 0
            n_subproblem = 0
            while not done:
                x, e, ref, h, c, z, ll, reward, info = self.step(x, e, ref, h, c, z)
                log_likelihood += ll.sum()
                step += 1
                total_reward += reward.sum().item()
                n_subproblem += x.shape[0]
                info['step'] = step
                info['total_reward'] = total_reward
                if step >= args.ns_n_step:
                    break
            # optimize critic
            l = torch.tensor(total_reward, requires_grad=False, device=args.device)
            # reward scaling in [-1, 0] to improve learning
            l /= args.reward_scale
            l_hat /= args.reward_scale
            log_likelihood /= args.reward_scale
            critic_loss = self.critic_loss_fn(l_hat, l)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                     max_norm=1., norm_type=2)
            self.critic_optimizer.step()
            self.critic_scheduler.step()
            # optimize actor
            advantage = l.detach() - l_hat.detach()
            actor_loss = (advantage * log_likelihood).mean() # log_likelihood?
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),
                                     max_norm=1., norm_type=2)
            self.actor_optimizer.step()
            self.actor_scheduler.step()
            # export csv
            monitor.export_csv()
            episode_info = {
                'episode': episode,
                'value': l.item(),
                'estimate_value': l_hat.item(),
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'log_likelihood': log_likelihood.item(),
                'n_step': step,
                'n_subproblem': n_subproblem,
                'total_reward': total_reward,
            }
            monitor.step(episode_info)
            print()
