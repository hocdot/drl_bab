import torch.optim as optim
import numpy as np
import torch
import os

from ..milp_baseline import select_random_action, select_cost_based_action, select_rounding_error_action
from .replay_buffer import ReplayBuffer
from .model import Actor, SoftQNetwork
from ..base import Base

class SACDecider(Base):

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
        self.q_optimizer     = optim.AdamW(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.AdamW(list(self.actor.parameters()), lr=args.actor_lr)
        self.actor_scaler    = torch.amp.GradScaler(args.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        if args.autotune_entropy:
            self.target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(args.n_var))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.actor_lr)
        else:
            self.alpha = args.alpha
        self.replay_buffer = ReplayBuffer(args.n_buffer)

    def transform(self, x):
        if isinstance(x, torch.Tensor):
            return x
        
        if isinstance(x, list):
            if len(x) > 0:
                if isinstance(x[0], int) or isinstance(x[0], np.int64):
                    x = torch.tensor(x, device=self.args.device, dtype=torch.long)
                    return x
                raise NotImplementedError
            raise NotImplementedError
        
        if isinstance(x, np.ndarray):
            if x.dtype == 'float64':
                x = torch.tensor(x, device=self.args.device, dtype=torch.float)
                return x
            raise NotImplementedError
        raise NotImplementedError

    def inverse_transform(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
            return x
        raise NotImplementedError

    def init_hook(self, observation):
        x = self.transform(observation)
        self.decoder_state = self.actor.decoder.get_z0(x)

    def step_hook(self, next_observation):
        x = self.transform(next_observation)
        a = self.env.prev_actions
        if len(a) == 0:
            return
        a = self.transform(a)
        e = self.actor.embedding(x)
        z = self.decoder_state
        z = self.actor.decoder.gather_z(e, a)
        self.decoder_state = z

    def select_action(self, observation):
        x = self.transform(observation)
        z = self.decoder_state
        if self.args.mode == 'train':
            action, _, _ = self.actor.get_action(x, z)
        elif self.args.mode == 'test':
            logits = self.actor(x, z)
            action = torch.argmax(logits, dim=1)
        else:
            raise NotImplementedError
        action = self.inverse_transform(action)
        return action
    
    def select_heuristic_action(self, observation):
        i = self.global_step % 3
        if i == 0:
            action = select_random_action(observation)
        elif i == 1:
            action = select_cost_based_action(observation)
        elif i == 2:
            action = select_rounding_error_action(observation)
        return action

    def train_episode(self):
        # reset environment
        self.args.mode = 'train'
        observation, done = self.env.reset()
        info = {
            'episode': self.env.episode, 
            'value': 0.0, 
            'actor_loss': 0.0, 
            'critic_loss': 0.0, 
            'alpha_loss': 0.0,
        }
        self.init_hook(observation)
        episode_step = 0
        # step loop
        while not done:
            # select action
            with torch.no_grad():
                if self.global_step > self.args.n_start_learning:
                    action = self.select_action(observation)
                else:
                    action = self.select_heuristic_action(observation)
            # store previous action
            pa = np.array(self.env.prev_actions)
            # send to environment
            next_observation, reward, done, step_info, early_stop = self.env.step(action)
            if early_stop:
                break
            # extract training data
            o = observation
            a = np.array(self.env.train_action)
            no = np.array(self.env.train_next_observation)
            nd = np.array(self.env.train_done)
            nr = np.array(self.env.train_reward)
            # push data to replay buffer
            # print('[+] push to replay_buffer')
            # print(f'    - {pa.shape=} {o.shape=} {a.shape=} {no.shape=} {nd.shape=} {nr.shape=}')
            for i in range(len(pa)):
                self.replay_buffer.push(pa[i], o[i], a[i], no[i], nd[i], nr[i])
            # print()
            # optimize model
            if self.global_step > self.args.n_start_learning:
                if self.global_step % self.args.n_optimize == 0:
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
        info['done'] = done
        info['global_step'] = self.global_step
        info['early_stop'] = early_stop
        # exit()
        return info

    @property
    def checkpoint_path(self):
        return os.path.join(self.args.checkpoint_dir, f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.pt')

    def save(self):
        # save model
        path = self.checkpoint_path
        data = {
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict()
        }
        torch.save(data, path)
        print(f'[+] saved {path}')

    def load(self):
        path = self.checkpoint_path
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
            return True
        return False


    def __repr__(self):
        return f'SACDecider(mode={self.args.mode}, dataset={self.args.dataset})'
    
    from .optimize import (optimize_qf, optimize_actor, update_target, optimize, print_gradients)