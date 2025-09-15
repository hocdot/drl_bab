import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os

from .replay_buffer import ReplayBuffer, TrainingData
from ..base import BaseSolver
from model.sac import Actor, SoftQNetwork

class SACSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.create_model()
        self.global_step = 0

    def create_model(self):
        args = self.args
        self.actor           = Actor(args).to(args.device)
        self.qf1             = SoftQNetwork(args).to(args.device)
        self.qf2             = SoftQNetwork(args).to(args.device)
        self.qf1_target      = SoftQNetwork(args).to(args.device)
        self.qf2_target      = SoftQNetwork(args).to(args.device)
        self.q_optimizer     = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.actor_lr, eps=1e-4)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        if args.autotune_entropy:
            self.target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(args.n_var))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.actor_lr, eps=1e-4)
        else:
            self.alpha = args.alpha
        self.replay_buffer = ReplayBuffer(args.n_buffer)

    def transform(self, x):
        args = self.args
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, list):
            if len(x) > 0:
                if isinstance(x[0], int) or isinstance(x[0], np.int64):
                    x = torch.tensor(x, device=args.device, dtype=torch.long)
                    return x
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif isinstance(x, np.ndarray):
            if x.dtype == 'float64':
                x = torch.tensor(x, device=args.device, dtype=torch.float)
                return x
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def inverse_transform(self, x):
        args = self.args
        if isinstance(x, torch.Tensor):
            x = x.tolist()
            return x
        else:
            raise NotImplementedError

    def init_hook(self, observation):
        x = self.transform(observation)
        self.decoder_state = self.actor.decoder.get_z0(x)

    def step_hook(self, next_observation):
        x = self.transform(next_observation)
        a = self.env.expanded_action
        if len(a) == 0:
            return
        a = self.transform(a)
        e = self.actor.embedding(x)
        z = self.decoder_state
        z = self.actor.decoder.gather_z(e, a)
        self.decoder_state = z

    def select_action(self, observation):
        args = self.args
        if args.mode == 'train':
            x = self.transform(observation)
            z = self.decoder_state
            action, _, _ = self.actor.get_action(x, z)
            action = self.inverse_transform(action)
        elif args.mode == 'test':
            x = self.transform(observation)
            z = self.decoder_state
            logits = self.actor(x, z)
            action = torch.argmax(logits, dim=1)
            action = self.inverse_transform(action)
        else:
            raise NotImplementedError
        return action

    def select_random_action(self, observation):
        n_sub, n_var, n_feature = observation.shape
        active = observation[:, :, -1]
        action = []
        for i in range(n_sub):
            indices = np.where(active[i] == 1)[0]
            j = np.random.choice(indices)
            action.append(j)
        return action

    def select_cost_based_action(self, observation):
        n_sub, n_var, n_feature = observation.shape
        active = observation[:, :, -1]
        score = observation[:, :, 1]
        action = []
        for i in range(n_sub):
            indices = np.where(active[i] == 1)[0]
            j = np.argmax(score[i][indices])
            k = indices[j]
            # j = np.random.choice(indices)
            action.append(k)
        return action

    def select_rounding_error_action(self, observation):
        n_sub, n_var, n_feature = observation.shape
        active = observation[:, :, -1]
        score = observation[:, :, 0]
        action = []
        for i in range(n_sub):
            indices = np.where(active[i] == 1)[0]
            j = np.argmax(score[i][indices])
            k = indices[j]
            # j = np.random.choice(indices)
            action.append(k)
        return action

    def train_episode(self):
        # extract args
        replay_buffer = self.replay_buffer
        args = self.args
        env = self.env
        # reset environment
        observation, done = env.reset()
        info = {'episode': env.episode, 'step': self.global_step, 'value': 0.0, 'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0}
        self.init_hook(observation)
        episode_step = 0
        # step loop
        while not done:
            # select action
            with torch.no_grad():
                if self.global_step > args.n_start_learning:
                    action = self.select_action(observation)
                else:
                    i = self.global_step % 3
                    if i == 0:
                        action = self.select_random_action(observation)
                    elif i == 1:
                        action = self.select_cost_based_action(observation)
                    elif i == 2:
                        action = self.select_rounding_error_action(observation)
            # store previous action
            pa = np.array(env.expanded_action)
            # send to environment
            next_observation, reward, done, step_info = env.step(action)
            # extract training data
            o = observation
            a = np.array(env.train_action)
            no = np.array(env.train_next_observation)
            nd = np.array(env.train_done)
            nr = np.array(env.train_reward)
            # push data to replay buffer
            # print('[+] push to replay_buffer')
            # print(f'    - {pa.shape=} {o.shape=} {a.shape=} {no.shape=} {nd.shape=} {nr.shape=}')
            for i in range(len(pa)):
                replay_buffer.push(pa[i], o[i], a[i], no[i], nd[i], nr[i])
            # print()
            # optimize model
            if self.global_step > args.n_start_learning:
                if self.global_step % args.n_optimize == 0:
                    actor_loss, critic_loss, alpha_loss = self.optimize()
                    info['actor_loss'] += actor_loss
                    info['critic_loss'] += critic_loss
                    info['alpha_loss'] += alpha_loss
                    episode_step += 1
            # update
            self.step_hook(next_observation)
            info['value'] += reward
            observation = next_observation
            self.global_step += 1
        if episode_step > 0:
            info['actor_loss'] /= episode_step
            info['critic_loss'] /= episode_step
            info['alpha_loss'] /= episode_step
        info['n_buffer'] = len(replay_buffer)
        # exit()
        return info

    def optimize_qf(self, data):
        # extract args
        actor, qf1, qf2, qf1_target, qf2_target = self.actor, self.qf1, self.qf2, self.qf1_target, self.qf2_target
        q_optimizer = self.q_optimizer
        alpha = self.alpha
        args = self.args
        # train
        with torch.no_grad():
            # 1st branch
            _, next_log_prob, next_action_probs = actor.decode_and_get_action(data.next_observation[:, 0], data.action[:, 0])
            qv1_target      = qf1_target.decode_and_forward(data.next_observation[:, 0], data.action[:, 0])
            qv2_target      = qf2_target.decode_and_forward(data.next_observation[:, 0], data.action[:, 0])
            # min_qv_target_1 = next_action_probs * (torch.min(qv1_target, qv2_target) - alpha * next_log_prob)
            min_qv_target_1 = next_action_probs * (torch.min(qv1_target, qv2_target))
            min_qv_target_1 = min_qv_target_1.sum(dim=1)
            # 2nd branch
            _, next_log_prob, next_action_probs = actor.decode_and_get_action(data.next_observation[:, 1], data.action[:, 1])
            qv1_target      = qf1_target.decode_and_forward(data.next_observation[:, 1], data.action[:, 1])
            qv2_target      = qf2_target.decode_and_forward(data.next_observation[:, 1], data.action[:, 1])
            # min_qv_target_2 = next_action_probs * (torch.min(qv1_target, qv2_target) - alpha * next_log_prob)
            min_qv_target_2 = next_action_probs * (torch.min(qv1_target, qv2_target))
            min_qv_target_2 = min_qv_target_2.sum(dim=1)
            # add to next qv
            next_qv = data.reward + (1 - data.done[:, 0]) * args.gamma * (min_qv_target_1) + (1 - data.done[:, 1]) * args.gamma * (min_qv_target_2)
        # print(f'{data.observation.shape=} {data.prev_action.shape=}')
        qv1 = qf1.decode_and_forward(data.observation, data.prev_action)
        qv2 = qf2.decode_and_forward(data.observation, data.prev_action)
        # print(f'{qv1.shape=} {qv2.shape=} {data.action.shape=}')
        qv1_a = qv1.gather(1, data.action[:, 0][:, None]).squeeze()
        qv2_a = qv2.gather(1, data.action[:, 0][:, None]).squeeze()
        # begin print debug
        r_ = data.reward[0].item()
        done_ = data.done[0]
        next_qv_ = next_qv[0].item()
        qv1_a_ = qv1_a[0].item()
        qv2_a_ = qv2_a[0].item()
        # print(f'{r_=} {done_=} {next_qv_=} {qv1_a_=} {qv2_a_=}')
        # print()
        # end print debug
        qf1_loss = F.mse_loss(qv1_a, next_qv)
        qf2_loss = F.mse_loss(qv2_a, next_qv)
        qf_loss = qf1_loss + qf2_loss
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()
        return qf_loss.item()

    def optimize_actor(self, data):
        # extract args
        args = self.args
        if args.autotune_entropy:
            alpha, log_alpha, target_entropy = self.alpha, self.log_alpha, self.target_entropy
            actor_optimizer, alpha_optimizer = self.actor_optimizer, self.alpha_optimizer
        else:
            actor_optimizer = self.actor_optimizer
            alpha = self.alpha
        actor, qf1, qf2 = self.actor, self.qf1, self.qf2
        alpha_loss = 0.0
        # train
        _, log_prob, action_probs = actor.decode_and_get_action(data.observation, data.prev_action)
        with torch.no_grad():
            qv1 = qf1.decode_and_forward(data.observation, data.prev_action)
            qv2 = qf2.decode_and_forward(data.observation, data.prev_action)
            min_qv = torch.min(qv1, qv2)
        # actor_loss = (action_probs * ((alpha * log_prob) - min_qv)).mean()
        # print(f'{action_probs.shape=} {min_qv.shape=}')
        actor_loss = (action_probs * (- min_qv)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        actor_loss = actor_loss.item()
        # extract args
        args = self.args
        # train
        if args.autotune_entropy:
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_prob + target_entropy).detach())).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            self.alpha = log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        return actor_loss, alpha_loss

    def update_target(self):
        # extract args
        qf1, qf2, qf1_target, qf2_target = self.qf1, self.qf2, self.qf1_target, self.qf2_target
        args = self.args
        # update
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    def optimize(self):
        # extract args
        args          = self.args
        replay_buffer = self.replay_buffer
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        # extract batch from buffer
        transitions = replay_buffer.sample(args.batch_size)
        data = TrainingData(transitions, args)
        qf_loss = 0
        # print(f'[+] training data')
        # print(f'    - {data.prev_action.shape=}')      # n_subproblem
        # print(f'    - {data.observation.shape=}')      # n_subproblem, n_node, n_feature
        # print(f'    - {data.action.shape=}')           # n_subproblem, 2
        # print(f'    - {data.next_observation.shape=}') # n_subproblem, 2, n_node, n_feature
        # print(f'    - {data.done.shape=}')             # n_subproblem, 2
        # print(f'    - {data.reward.shape=}')           # n_subproblem
        with torch.autograd.set_detect_anomaly(True):
            qf_loss = self.optimize_qf(data)
            actor_loss, alpha_loss = self.optimize_actor(data)
        if self.global_step % args.n_target_update == 0:
            self.update_target()
        return actor_loss, qf_loss, alpha_loss

    def save(self):
        args = self.args
        state_dict = self.actor.state_dict()
        # save model
        path = os.path.join(args.checkpoint_dir, f'{args.dataset}_{args.solver}.pt')
        data = {
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict()
        }
        torch.save(data, path)
        print(f'[+] saved {path}')

    def load(self):
        args = self.args
        path = os.path.join(args.checkpoint_dir, f'{args.dataset}_{args.solver}.pt')
        if os.path.exists(path):
            print(f'[+] loading {path}')
            data = torch.load(path)
            self.actor.load_state_dict(data['actor'])
            self.qf1.load_state_dict(data['qf1'])
            self.qf2.load_state_dict(data['qf2'])
            self.qf1_target.load_state_dict(data['qf1'])
            self.qf2_target.load_state_dict(data['qf2'])
            self.actor.eval()
            self.qf1.eval()
            self.qf2.eval()
            self.qf1_target.eval()
            self.qf2_target.eval()

    def __repr__(self):
        return f'SACSolver(mode={self.args.mode}, dataset={self.args.dataset})'