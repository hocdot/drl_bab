from beartype import beartype
import torch.optim as optim
import torch
import os

from ..milp_baseline import select_cost_based_action, select_rounding_error_action, select_random_action
from .replay_buffer import ReplayBuffer
from model import Actor, SoftQNetwork
from .scaler import StandardScaler
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
        self.reward_scaler = StandardScaler()
        

    @beartype
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.args.device)

    @beartype
    def init_hook(self, observation: torch.Tensor):
        x = self.transform(observation)
        self.decoder_state = self.actor.decoder.get_z0(x)


    @beartype
    def step_hook(self, next_observation: torch.Tensor):
        x = self.transform(next_observation)
        a = self.env.prev_actions
        if len(a) == 0:
            return
        a = self.transform(a)
        e = self.actor.embedding(x)
        z = self.decoder_state
        z = self.actor.decoder.gather_z(e, a)
        self.decoder_state = z

    @beartype
    def select_action(self, observation: torch.Tensor) -> list[int]:
        x = self.transform(observation)
        z = self.decoder_state
        if self.args.use_scale_action:
            _, _, action_probs = self.actor.get_action(x, z)
            new_observation = x.clone()
            new_observation[:, :, 1] *= action_probs
            action = select_cost_based_action(new_observation.cpu().numpy())
            return action
        else:
            if self.args.mode == 'train':
                action, _, _ = self.actor.get_action(x, z)
            elif self.args.mode == 'test':
                logits = self.actor(x, z)
                action = torch.argmax(logits, dim=1)
            else:
                raise NotImplementedError
            return action.tolist()
    
    @beartype
    def select_heuristic_action(self, observation: torch.Tensor) -> list[int]:
        n = self.global_step % 3
        if self.args.dataset == 'milp':
            if n == 0:
                action = select_rounding_error_action(observation)
            elif n == 1:
                action = select_cost_based_action(observation)
            elif n == 2:
                action = select_random_action(observation)
        else:
            # action in active neuron index format
            if n == 0:
                action = select_random_action(observation)
            else:
                action = self.env._get_fsb_action()
        return action

    def train_instance(self):
        # reset environment
        self.args.mode = 'train'
        observation, done = self.env.reset()
        info = {
            'value': 0.0, 
            'actor_loss': 0.0, 
            'critic_loss': 0.0, 
            'alpha_loss': 0.0,
        }
        self.init_hook(observation) if not done else None
        episode_step = 0
        early_stop = False
        # step loop
        while not done:
            # select action
            with torch.no_grad():
                if (self.global_step <= self.args.n_start_learning) or ((self.global_step % 10) < 3):
                    try:
                        action = self.select_heuristic_action(observation)
                    except ValueError: # all neurons are split
                        early_stop = True
                        break
                else:
                    # if torch.any(observation[:, :, -1].sum(dim=1) == 0): # all neurons are split
                    #     early_stop = True
                    #     break
                    action = self.select_action(observation)

            # send to environment
            remaining_next_observation, reward, done, early_stop, train_data = self.env.step(action)
            self.step_hook(remaining_next_observation)
            observation = remaining_next_observation
            
            # extract training data
            for data in train_data:
                self.replay_buffer.push(data)

            # save offline data
            if self.args.save_offline_data:
                if (self.global_step > self.args.n_start_save_offline_data) and (self.global_step % 10 == 0):
                    self.sample_batch()
                
            # optimize model
            if (self.global_step > self.args.n_start_learning) and (self.global_step % self.args.n_optimize == 0):
                actor_loss, critic_loss, alpha_loss = self.optimize(data=None,n_qf_optimize=self.args.n_qf_optimize)
                info['actor_loss'] += actor_loss
                info['critic_loss'] += critic_loss
                info['alpha_loss'] += alpha_loss
                episode_step += 1
                
            # update
            info['value'] += reward
            self.global_step += 1
            if early_stop:
                break
            
        if episode_step > 0:
            info['actor_loss'] /= episode_step
            info['critic_loss'] /= episode_step
            info['alpha_loss'] /= episode_step
        info['done'] = done
        info['early_stop'] = early_stop
        return info

    @property
    def checkpoint_path(self):
        return os.path.join(self.args.checkpoint_dir, self.args.dataset, f'{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.pt')

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
        print(f'[-] not found {path}')
        return False


    def __repr__(self):
        return f'SACDecider(mode={self.args.mode}, dataset={self.args.dataset})'
    
    from .optimize import (
        optimize_qf, optimize_actor, update_target, optimize, 
        print_gradients, sample_batch)
    from .train_offline import train_offline