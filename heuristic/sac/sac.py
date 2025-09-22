import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os

from .replay_buffer import ReplayBuffer, TrainingData
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
        # reset environment
        self.args.mode = 'train'
        observation, done = self.env.reset()
        info = {'episode': self.env.episode, 'step': self.global_step, 'value': 0.0, 'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0}
        self.init_hook(observation)
        episode_step = 0
        # step loop
        while not done:
            # select action
            with torch.no_grad():
                if self.global_step > self.args.n_start_learning:
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
            pa = np.array(self.env.prev_actions)
            # send to environment
            next_observation, reward, done, step_info = self.env.step(action)
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
        info['n_buffer'] = len(self.replay_buffer)
        info['done'] = done
        info['global_step'] = self.global_step
        info['mode'] = self.args.mode
        # exit()
        return info

    def optimize_qf(self, data):
        # train
        with torch.no_grad():
            # 1st branch
            _, next_log_prob, next_action_probs = self.actor.decode_and_get_action(data.next_observation[:, 0], data.action[:, 0])
            qv1_target      = self.qf1_target.decode_and_forward(data.next_observation[:, 0], data.action[:, 0])
            qv2_target      = self.qf2_target.decode_and_forward(data.next_observation[:, 0], data.action[:, 0])
            # min_qv_target_1 = next_action_probs * (torch.min(qv1_target, qv2_target) - alpha * next_log_prob)
            min_qv_target_1 = next_action_probs * (torch.min(qv1_target, qv2_target))
            min_qv_target_1 = min_qv_target_1.sum(dim=1)
            # 2nd branch
            _, next_log_prob, next_action_probs = self.actor.decode_and_get_action(data.next_observation[:, 1], data.action[:, 1])
            qv1_target      = self.qf1_target.decode_and_forward(data.next_observation[:, 1], data.action[:, 1])
            qv2_target      = self.qf2_target.decode_and_forward(data.next_observation[:, 1], data.action[:, 1])
            # min_qv_target_2 = next_action_probs * (torch.min(qv1_target, qv2_target) - alpha * next_log_prob)
            min_qv_target_2 = next_action_probs * (torch.min(qv1_target, qv2_target))
            min_qv_target_2 = min_qv_target_2.sum(dim=1)
            # add to next qv
            next_qv = data.reward + (1 - data.done[:, 0]) * self.args.gamma * (min_qv_target_1) + (1 - data.done[:, 1]) * self.args.gamma * (min_qv_target_2)
        # print(f'{data.observation.shape=} {data.prev_action.shape=}')
        qv1 = self.qf1.decode_and_forward(data.observation, data.prev_action)
        qv2 = self.qf2.decode_and_forward(data.observation, data.prev_action)
        # print(f'{qv1.shape=} {qv2.shape=} {data.action.shape=}')
        qv1_a = qv1.gather(1, data.action[:, 0][:, None]).squeeze()
        qv2_a = qv2.gather(1, data.action[:, 0][:, None]).squeeze()

        # compute loss
        qf1_loss = F.mse_loss(qv1_a, next_qv)
        qf2_loss = F.mse_loss(qv2_a, next_qv)
        qf_loss = qf1_loss + qf2_loss
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        # print grad to check gradient vanishing/exploding
        if self.global_step % 100 == 0:  # Print every 100 steps to avoid spam
            self._print_gradients("Q-Networks", [self.qf1, self.qf2])
        
        self.q_optimizer.step()
        return qf_loss.item()

    def optimize_actor(self, data):
        alpha_loss = 0.0
        # train
        _, log_prob, action_probs = self.actor.decode_and_get_action(data.observation, data.prev_action)
        with torch.no_grad():
            qv1 = self.qf1.decode_and_forward(data.observation, data.prev_action)
            qv2 = self.qf2.decode_and_forward(data.observation, data.prev_action)
            min_qv = torch.min(qv1, qv2)
        # actor_loss = (action_probs * ((alpha * log_prob) - min_qv)).mean()
        # print(f'{action_probs.shape=} {min_qv.shape=}')
        actor_loss = (action_probs * (- min_qv)).mean()
        self.actor_optimizer.zero_grad()
        self.actor_scaler.scale(actor_loss).backward()
        self.actor_scaler.step(self.actor_optimizer)
        self.actor_scaler.update()
        
        # print grad to check gradient vanishing/exploding
        if self.global_step % 100 == 0:  # Print every 100 steps to avoid spam
            self._print_gradients("Actor", [self.actor])
            
        actor_loss = actor_loss.item()
        # train
        if self.args.autotune_entropy:
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        return actor_loss, alpha_loss

    def update_target(self):
        # update
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def optimize(self):
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        # extract batch from buffer
        transitions = self.replay_buffer.sample(self.args.batch_size)
        data = TrainingData(transitions, self.args)
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
        if self.global_step % self.args.n_target_update == 0:
            self.update_target()
        return actor_loss, qf_loss, alpha_loss

    def save(self):
        # save model
        path = os.path.join(self.args.checkpoint_dir, f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.pt')
        data = {
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict()
        }
        torch.save(data, path)
        print(f'[+] saved {path}')

    def load(self):
        path = os.path.join(self.args.checkpoint_dir, f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.pt')
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

    def _print_gradients(self, network_name, networks):
        """Print gradient statistics to check for vanishing/exploding gradients."""
        print(f"\n=== {network_name} Gradient Stats (Step {self.global_step}) ===")
        for i, network in enumerate(networks):
            network_suffix = f"_{i+1}" if len(networks) > 1 else ""
            print(f"{network_name}{network_suffix}:")
            
            grad_norms = []
            for name, param in network.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if grad_norm < 1e-6:
                        print(f"  {name}: {grad_norm:.2e} (VANISHING)")
                    elif grad_norm > 10.0:
                        print(f"  {name}: {grad_norm:.2e} (EXPLODING)")
                    else:
                        print(f"  {name}: {grad_norm:.6f}")
                else:
                    print(f"  {name}: None (no gradient)")
            
            if grad_norms:
                mean_grad = sum(grad_norms) / len(grad_norms)
                max_grad = max(grad_norms)
                min_grad = min(grad_norms)
                print(f"  Summary - Mean: {mean_grad:.6f}, Max: {max_grad:.6f}, Min: {min_grad:.6f}")
            print()

    def __repr__(self):
        return f'SACDecider(mode={self.args.mode}, dataset={self.args.dataset})'