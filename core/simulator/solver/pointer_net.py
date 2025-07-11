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
        # decoder step
        d, (h, c) = self.actor.decoder(z, h, c) # d: n_subproblem, 1, n_embed
        q = h.squeeze(1) # q:= n_subproblem, n_embed # TODO: note original implementation h.squeeze(0)
        # glimpse step
        for _ in range(args.n_glimpse):
            q = self.actor.glimpse(q, ref, mask)
        # pointing using attention
        logits = self.actor.pointer(q, ref, mask) # logits: n_subproblem, n_node
        log_prob = torch.log_softmax(logits, dim=-1) # log_prob: n_subproblem, n_node
        # select next node
        action = torch.argmax(log_prob, dim=1).long() # next_node: n_subproblem
        return action, (d, h, c)

    def test(self):
        # extract args
        monitor = self.monitor
        args = self.args
        env  = self.env
        # procedure before test
        print(f'[+] testing {monitor.label}')
        # TODO: env.reset(), change to another model
        # reset environment
        observation, self.edge_index, self.edge_weight = env.reset()
        ew = self.edge_weight.to(args.device)
        ei  = self.edge_index.to(args.device)
        # input
        x = observation.to(args.device) # n_subproblem, n_node, n_input
        # embed
        e = self.actor.embedding(x, ei, ew) # n_subproblem, n_node, n_embed
        # encode
        ref, (h, c) = self.actor.encoder(e) # ref: n_subproblem, n_node, n_embed
        # decode loop
        z = self.actor.decoder.get_z0(x) # n_subproblem, 1, n_embed
        # init mask
        mask = torch.zeros(*x.shape[:-1], device=args.device) # n_subproblem, n_node
        # initial
        done = False
        step = 0
        total_reward = 0
        while not done:
            # select action
            action, (d, h, c) = self.select_action(z, h, c, ref, mask)
            # update z
            z = self.actor.decoder.gather_z(e, action)
            # update mask
            mask += torch.zeros(mask.shape, device=args.device).\
                    scatter_(dim=1, index=action.unsqueeze(1), value=1)
            # TODO: HERE, update ref depending on action
            # and split subproblems
            action = action.detach().cpu().tolist()
            next_observation, reward, done, info = env.step(action)
            n_subproblem = len(next_observation)
            total_reward += reward.sum().item()
            info.update({
                'step': step,
                'total_reward': total_reward,
                'avg_reward': reward.mean().item(),
                'n_subproblem': n_subproblem,
                'done': done,
            })
            monitor.step(info)
            step += 1
            observation = next_observation
            if step >= args.ns_n_step:
                break
        monitor.export_csv()
