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
        self._one_shot_raw_scale = None

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
        self._one_shot_raw_scale = None
        x = self.transform(observation)
        self.decoder_state = self.actor.decoder.get_z0(x)


    @beartype
    def step_hook(self, next_observation: torch.Tensor):
        if getattr(self.args, 'use_one_shot_inference', False):
            return  # scores are frozen; no need to update decoder state
        x = self.transform(next_observation)
        a = self.env.prev_actions
        if len(a) == 0:
            return
        a = self.transform(a)
        e = self.actor.embedding(x)
        self.decoder_state = self.actor.decoder.gather_z(e, a)
        
    @beartype
    def _convert_observation(self, x: torch.Tensor) -> torch.Tensor:
        if self.env.reduced_active_neuron_index is not None:
            return x[:, self.env.reduced_active_neuron_index]
        return x

    @beartype
    def _convert_action_probs(self, action_probs: torch.Tensor) -> torch.Tensor:
        if self.env.reduced_active_neuron_index is not None:
            # TODO: initializing with 0.0 might not be the best idea
            new_action_probs = torch.zeros(action_probs.shape[0], len(self.env.active_neuron_index), device=action_probs.device)
            batched_index = torch.tensor(self.env.reduced_active_neuron_index, device=action_probs.device).unsqueeze(0).expand(action_probs.shape[0], -1)
            new_action_probs.scatter_(1, batched_index, action_probs)
            return new_action_probs
        return action_probs

    def _expand_raw_scale(self, raw_scale: torch.Tensor, n_batch: int) -> torch.Tensor:
        if raw_scale.shape[0] == n_batch:
            return raw_scale
        return raw_scale[:1].expand(n_batch, -1).contiguous()
    
    def _get_baseline_action(self, scale=None, raw_scale=None) -> list[int]:
        heuristic = getattr(self.args, 'decision_heuristic', 'fsb')
        getters = {
            'fsb': self.env._get_fsb_action,
            'babsr': self.env._get_babsr_action,
            'upb': self.env._get_upb_action,
        }
        if heuristic not in getters:
            raise ValueError(f'Unknown decision_heuristic: {heuristic}')
        return getters[heuristic](scale=scale, raw_scale=raw_scale)

    @beartype
    def select_action(self, observation: torch.Tensor) -> list[int]:
        x = self.transform(observation)
        x = self._convert_observation(x) # reduce number of active neurons
        z = self.decoder_state
        if self.args.use_scale_action:
            use_one_shot = getattr(self.args, 'use_one_shot_inference', False)
            if use_one_shot and self.args.dataset != 'milp':
                if self._one_shot_raw_scale is None:
                    _, _, action_probs = self.actor.get_action(x, z)
                    action_probs = self._convert_action_probs(action_probs)
                    self._one_shot_raw_scale = self.env._convert_scale_to_raw(action_probs).detach()
                raw_scale = self._expand_raw_scale(self._one_shot_raw_scale, observation.shape[0])
                return self._get_baseline_action(raw_scale=raw_scale)

            if self.args.n_test_fallback > 0 and self.args.mode == 'test' and self.env.n_step % self.args.n_test_fallback == 0:
                action_probs = None
            else:
                _, _, action_probs = self.actor.get_action(x, z)
                action_probs = self._convert_action_probs(action_probs) # add back to number of active neurons
            if self.args.dataset == 'milp':
                action = select_cost_based_action(observation, scale=action_probs)
            else:
                action = self._get_baseline_action(scale=action_probs)
        else:
            if self.args.mode == 'train':
                action, _, _ = self.actor.get_action(x, z)
            elif self.args.mode == 'test':
                logits = self.actor(x, z)
                action = torch.argmax(logits, dim=1)
            else:
                raise NotImplementedError
            action = action.tolist()
        return action
    
    @beartype
    def select_heuristic_action(self, observation: torch.Tensor) -> list[int]:
        n = self.global_step % 5
        if self.args.dataset == 'milp':
            if n == 0:
                action = select_random_action(observation)
            elif n == 1:
                action = select_cost_based_action(observation)
            else:
                action = select_rounding_error_action(observation)
        else:
            # action in active neuron index format
            if n == 0:
                action = select_random_action(observation)
            else:
                action = self._get_baseline_action()
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
                if (self.global_step <= self.args.n_start_learning) or ((self.global_step % 10) < 5):
                    try:
                        action = self.select_heuristic_action(observation)
                    except (ValueError, AssertionError): # all neurons are split
                        print(f'[-] error when selecting heuristic action')
                        early_stop = True
                        break
                else:
                    if torch.any(observation[:, :, -1].sum(dim=1) == 0): # all neurons are split
                        print(f'[!] all neurons are split')
                        early_stop = True
                        break
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
            if (self.global_step > self.args.n_start_learning) or (len(self.replay_buffer) > 0.5 * self.args.n_buffer):
                actor_loss, critic_loss, alpha_loss = self.optimize(n_qf_optimize=self.args.n_qf_optimize)
                info['actor_loss'] += actor_loss
                info['critic_loss'] += critic_loss
                info['alpha_loss'] += alpha_loss
                episode_step += 1
                
            # update
            info['value'] += reward
            self.global_step += 1
            
            # save checkpoint
            if self.global_step % self.args.n_save_checkpoint == 0:
                self.save()
                
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
        if self.args.checkpoint_postfix:
            postfix = f'_{self.args.checkpoint_postfix}'
        else:
            postfix = ''
        return os.path.join(self.args.checkpoint_dir, self.args.dataset, f'{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}{postfix}.pt')

    def save(self):
        # save model
        path = self.checkpoint_path
        
        count = 0
        while os.path.exists(f'{path[:-3]}_{count}.pt'):
            count += 1
        
        backup_path = f'{path[:-3]}_{count}.pt'
        os.system(f'cp "{path}" "{backup_path}"')
        print(f'[+] backup {backup_path}')
        
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
            data = torch.load(path, map_location=self.args.device)
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