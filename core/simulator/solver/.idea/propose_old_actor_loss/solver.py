import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import torch
import tqdm
import os

from .buffer import PretrainBuffer, PretrainTransition
from .critic import Critic
from .actor import Actor

from ..base import BaseSolver
import simulator

class Solver(BaseSolver):

    def __init__(self, args): # share
        super().__init__(args)
        # initialize model
        self.initialize_model()

    def initialize_model(self):
        # extract args
        args = self.args
        env  = self.env
        # initialize actor
        self.actor = Actor(args).to(args.device)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()),
                                          lr=args.actor_lr)
        # initialize critic
        self.critic1 = Critic(args).to(args.device)
        self.critic2 = Critic(args).to(args.device)
        self.critic1_target = Critic(args).to(args.device)
        self.critic2_target = Critic(args).to(args.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # initialize optimizer
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + \
                                           list(self.critic2.parameters()),
                                           lr=args.critic_lr)
        #
        self.edge_indices, self.edge_weights = [], []
        for i, (_, onnx) in pd.read_csv(os.path.join(args.onnx_dir, 'onnx.csv')).iterrows():
            self.edge_indices.append(torch.load(os.path.join(args.onnx_dir, f'{onnx}_index.pth')))
            self.edge_weights.append(torch.load(os.path.join(args.onnx_dir, f'{onnx}_weight.pth')))

    # =========================================================
    # PRETRAIN
    # =========================================================
    def pretrain_add_info(self, actor_loss, critic_loss):
        info = {
            'step': self.step,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
        }
        return info

    def pretrain_load_buffer(self):
        # extract args
        args    = self.args
        #
        self.pretrain_buffers = []
        n_step = 0
        for name in tqdm.tqdm(os.listdir(args.data_dir), desc='pretrain load buffer'):
            path = os.path.join(args.data_dir, name)
            with open(path, 'rb') as fp:
                buffer = pickle.load(fp)
                n_step += len(buffer)
                self.pretrain_buffers.append(buffer)
        n_buffer = len(self.pretrain_buffers)
        print(f'[+] loaded {n_buffer=} {n_step=}')

    def pretrain_step(self):
        # extract args
        args             = self.args
        env              = self.env
        buffer           = np.random.choice(self.pretrain_buffers)
        actor            = self.actor
        actor_optimizer  = self.actor_optimizer
        critic1          = self.critic1
        critic2          = self.critic2
        critic1_target   = self.critic1_target
        critic2_target   = self.critic2_target
        critic_optimizer = self.critic_optimizer
        # extract batch of data
        transitions         = buffer.sample(args.batch_size)
        batch               = PretrainTransition(*zip(*transitions))
        observation         = torch.stack(batch.observation).to(args.device)
        mask                = torch.stack(batch.mask).to(args.device)
        min_reward          = torch.stack(batch.min_reward).to(args.device)
        max_reward          = torch.stack(batch.max_reward).to(args.device)
        iwb                 = batch.iwb[0].to(args.device)
        onnx_id             = batch.onnx_id[0].to(args.device)
        reward              = min_reward
        indices             = torch.where(mask == 0)
        reward[indices]     = -25
        min_reward[indices] = -25
        max_reward[indices] = -25
        next_observation = torch.stack(batch.next_observation).to(args.device)
        edge_index, edge_weight = self.edge_indices[onnx_id], self.edge_weights[onnx_id]
        # compute target
        with torch.no_grad():
            target_q_value1        = critic1_target(next_observation, edge_index, edge_weight).amax(dim=(1, 2), keepdims=True)
            target_q_value2        = critic2_target(next_observation, edge_index, edge_weight).amax(dim=(1, 2), keepdims=True)
            min_max_target_q_value = torch.min(target_q_value1, target_q_value2)
            next_q_value           = reward + args.gamma * min_max_target_q_value
        # compute estimate
        q_value1     = critic1(observation, edge_index, edge_weight)
        q_value2     = critic2(observation, edge_index, edge_weight)
        critic1_loss = F.mse_loss(q_value1, next_q_value)
        critic2_loss = F.mse_loss(q_value2, next_q_value)
        critic_loss  = (critic1_loss + critic2_loss) / args.batch_size
        # optimize the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        # tinh m
        m = ((min_reward + max_reward - 2 * iwb) / (- 2 * iwb)).flatten(1)
        g = actor(observation, edge_index, edge_weight).flatten(1)
        z = g[:, :, None] - g[:, None, :]
        theta = (0.3 - z).clamp(min=0)
        Y = ((m[:, :, None] - m[:, None, :]) > 0).to(torch.int32)
        K = Y.sum()
        actor_loss = (theta * Y).sum() / K
        # optimize actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # report data
        return actor_loss.item(), critic_loss.item()

    def pretrain(self):
        # extract args
        args    = self.args
        monitor = self.monitor
        actor   = self.actor
        critic  = self.critic1
        # load full feedback buffer
        self.pretrain_load_buffer()
        # optimization loop
        try:
            self.step = 0
            for step in range(args.n_pretrain_step):
                actor_loss, critic_loss = self.pretrain_step()
                # add more info
                info = self.pretrain_add_info(actor_loss, critic_loss)
                # logging
                monitor.step(info)
                #
                self.step += 1
            # export csv
            monitor.export_csv()
        except KeyboardInterrupt:
            pass
        finally:
            # save pretrain actor/critic model
            path = os.path.join(args.model_dir, f'pretrain_critic_{args.solver}_{args.gnn_class}.pth')
            print(f'[+] saved {path=}')
            torch.save(critic.state_dict(), path)
            path = os.path.join(args.model_dir, f'pretrain_actor_{args.solver}_{args.gnn_class}.pth')
            print(f'[+] saved {path=}')
            torch.save(actor.state_dict(), path)
    # =========================================================

    # =========================================================
    # TEST
    # =========================================================
    def select_action(self, observation):
        assert not torch.isnan(observation).any(), f'{torch.where(observation.isnan())=}'
        # observation: [bs, n_layer, n_neuron, n_feature]
        # extract data
        # extract args
        args   = self.args
        critic = self.critic1
        actor  = self.actor
        env    = self.env
        edge_index, edge_weight = env.edge_index, env.edge_weight
        # extract bs
        bs   = observation.shape[0]
        mask = observation[..., -1]
        # combined score
        def combined(observation, edge_index, edge_weight):
            # compute score via actor
            actor_mask = torch.zeros_like(mask)
            actor_score = actor(observation) # bs, n_layer, n_neuron -> mask: bs, n_layer, n_neuron
            # compute mask by actor
            for i in range(bs):
                threshold = torch.sort(actor_score[i].flatten(), descending=True).values[args.top_k_actor]
                actor_mask[i, ...] = (actor_score[i] > threshold).to(actor_mask)
            # compute critic score for only mask*actor_mask==1
            critic_score = critic(observation, edge_index, edge_weight)
            critic_score = (actor_mask * mask) * critic_score + (1 - actor_mask * mask) * -25
            return critic_score
        # compute score
        score = eval(args.score)(observation, edge_index, edge_weight)
        score = mask * score + (1 - mask) * -25
        # initialize action
        action = []
        for i in range(bs):
            # get indices of active neurons
            indices = torch.where(score[i] == torch.max(score[i]))
            # choose random candidate if there is more than neuron has max score
            choice = np.random.choice(np.arange(len(indices[0])))
            a = [indices[_][choice].item() for _ in range(len(indices))]
            action.append(a)
        if args.verbose:
            print(f'{action=}')
        return np.array(action)

    def test(self):
        # extract args
        args = self.args
        # load pretrain actor/critic model
        path = os.path.join(args.model_dir, f'pretrain_critic_{args.solver}_{args.gnn_class}.pth')
        self.critic1.load_state_dict(torch.load(path, weights_only=True))
        self.critic1.eval()
        print(f'loaded {path=}')
        path = os.path.join(args.model_dir, f'pretrain_actor_{args.solver}_{args.gnn_class}.pth')
        self.actor.load_state_dict(torch.load(path, weights_only=True))
        self.actor.eval()
        print(f'loaded {path=}')
        super().test()
