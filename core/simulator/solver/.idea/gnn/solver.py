import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import torch
import tqdm
import os

from .buffer import FullFeedbackBuffer, FullFeedbackTransition
from .critic import Critic

import simulator

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # create env
        self.env = simulator.Env(args)
        # load monitor
        self.monitor = simulator.Monitor(args)
        # initialize model
        self.initialize_model()

    def initialize_model(self):
        # extract args
        args = self.args
        # initialize critic
        self.critic1 = Critic(args)
        self.critic2 = Critic(args)
        self.critic1_target = Critic(args)
        self.critic2_target = Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # initialize optimizer
        self.critic_optimizer   = optim.Adam(list(self.critic1.parameters()) + \
                                             list(self.critic2.parameters()),
                                             lr=args.critic_lr)
        # initialize buffer
        self.full_feedback_buffer = FullFeedbackBuffer(args.full_feedback_buffer_size)
        #
        self.edge_index, self.edge_weight = simulator.prepare_graph(env.net, args)

    def gen_data_select_action(self, observation, step=None):
        # extract data
        env  = self.env
        args = self.args
        # compute optimal randomized top-k action
        rewards, _, actions = env.full_feedback()
        bs, _               = rewards.shape
        _actions = []
        for i in range(bs):
            indices = torch.argsort(rewards[i], descending=True)
            n_top_k = 2 # int(args.top_percentage * len(indices))
            # assert n_top_k >= 1
            indices = indices.detach().cpu().numpy()
            indices = indices[:n_top_k]
            idx     = np.random.choice(indices)
            _actions.append(actions[i, idx])
        action = np.array(_actions)
        # convert reward to desired format
        # actions.shape=(bs, n_active_neuron, 2) rewards.shape=(bs, n_active_neuron)
        # want rewards.shape=(bs,n_layer,n_hidden), inactive neuron's reward = 0
        rewards_ = torch.zeros(list(observation.shape[:-1]))
        n_active_neuron = rewards.shape[-1]
        for i in range(bs):
            for j in range(n_active_neuron):
                layer  = actions[i, j, 0]
                neuron = actions[i, j, 1]
                rewards_[i, layer, neuron] = rewards[i, j]
        # extract mask
        mask = observation[..., -1]
        return action, rewards_, mask

    def gen_data_episode(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        # reset environment
        observation = env.reset()
        done        = False
        step        = 0
        #
        episode_info = {}
        # iteratively run environment until done
        for step in range(args.n_step):
            # extract all rewards
            action, rewards, mask = self.gen_data_select_action(observation, step=step)
            # send to env
            next_observation, reward, done, info = env.step(action)
            next_observation_train = info['next_observation_train']
            # store data to buffer
            assert len(observation) == len(rewards) == len(mask), \
                f'{len(observation)=} == {len(rewards)=} == {len(mask)=} == {len(next_observation_train)=}'
            bs = len(observation)
            for i in range(bs):
                self.full_feedback_buffer.push(observation[i], rewards[i], mask[i], next_observation_train[i])
            # update observation
            observation = next_observation
            step       += 1
            # update info
            for k in info:
                if k == 'reward':
                    info[k] = torch.min(info[k])
                if k not in episode_info:
                    episode_info[k] = [info[k]]
                else:
                    episode_info[k].append(info[k])
            # display
            if args.verbose:
                print(f'{step=} {reward=} buffer={len(self.full_feedback_buffer)}')
            assert rewards.shape == mask.shape
            # stopping condition
            if done:
                break
        return episode_info

    def gen_data(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        # initialize testing variables
        self.step = 0
        #
        for episode in range(args.n_gen_data_episode):
            # test
            info = self.gen_data_episode()
            # add more info
            info = self.add_gen_data_info(episode, info)
            # logging
            monitor.step(info)
        # export csv
        monitor.export_csv()
        # save full feedback buffer
        path = os.path.join(args.data_dir, f'full_feedback_buffer.pkl')
        with open(path, 'wb') as fp:
            pickle.dump(self.full_feedback_buffer, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def pretrain_step(self):
        # extract args
        args             = self.args
        buffer           = self.full_feedback_buffer
        critic1          = self.critic1
        critic2          = self.critic2
        critic1_target   = self.critic1_target
        critic2_target   = self.critic2_target
        critic_optimizer = self.critic_optimizer
        # extract batch of data
        transitions      = buffer.sample(args.batch_size)
        batch            = FullFeedbackTransition(*zip(*transitions))
        observation      = torch.stack(batch.observation).to(args.device)
        mask             = torch.stack(batch.mask).to(args.device)
        reward           = torch.stack(batch.reward).to(args.device)
        next_observation = torch.stack(batch.next_observation).to(args.device)
        edge_index, edge_weight = self.edge_index, self.edge_weight
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
        # report data
        return critic_loss.item()

    def add_pretrain_info(self, critic_loss):
        info = {
            'step': self.step,
            'critic_loss': critic_loss,
        }
        return info

    def pretrain(self):
        # extract args
        args    = self.args
        monitor = self.monitor
        critic  = self.critic1
        # load full feedback buffer
        path = os.path.join(args.data_dir, f'full_feedback_buffer.pkl')
        with open(path, 'rb') as fp:
            self.full_feedback_buffer = pickle.load(fp)
        # optimization loop
        self.step = 0
        for step in range(args.n_pretrain_step):
            critic_loss = self.pretrain_step()
            # add more info
            info = self.add_pretrain_info(critic_loss)
            # logging
            monitor.step(info)
        # export csv
        monitor.export_csv()
        # save pretrain critic model
        path = os.path.join(args.model_dir, f'pretrain_critic_{args.solver}.pth')
        torch.save(critic.state_dict(), path)

    def select_action(self, observation):
        raise NotImplementedError

    def test_episode(self):
        raise NotImplementedError

    def add_info(self):
        raise NotImplementedError

    def add_gen_data_info(self, episode, episode_info):
        info = {
            'episode': episode,
            'step': self.step,
            'total_reward': np.sum(episode_info['reward']),
            'avg_reward': np.mean(episode_info['reward']),
            'avg_remaining': np.mean(episode_info['remaining']),
            'avg_worst_bound': np.mean(episode_info['worst_bound']),
            'n_visited': episode_info['visited'][-1],
            'n_step': len(episode_info['visited']),
            'n_buffer': len(self.full_feedback_buffer),
        }
        return info

    def add_test_info(self, episode, episode_info):
        info = {
            'episode': episode,
            'total_reward': np.sum(episode_info['reward']),
            'avg_remaining': np.mean(episode_info['remaining']),
            'avg_worst_bound': np.mean(episode_info['worst_bound']),
            'n_visited': episode_info['visited'][-1],
            'n_step': len(episode_info['visited']),
        }
        return info

    def select_action(self, observation):
        # observation: [bs, n_layer, n_neuron, n_feature]
        # extract data
        # extract args
        args   = self.args
        critic = self.critic1
        edge_index, edge_weight = self.edge_index, self.edge_weight
        # compute score
        mask   = observation[..., -1]
        score  = critic(observation, edge_index, edge_weight)
        bs     = observation.shape[0]
        # initialize action
        action = []
        for i in range(bs):
            # get indices of active neurons
            a = [_.item() for _ in torch.where(score[i] == torch.max(score[i]))]
            action.append(a)
        if args.verbose:
            print(f'{action=}')
        return np.array(action)

    def test_episode(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        critic  = self.critic1
        # reset environment
        observation = env.reset()
        done        = False
        step        = 0
        # load pretrain critic model
        path = os.path.join(args.model_dir, f'pretrain_critic_{args.solver}.pth')
        critic.load_state_dict(torch.load(path, weights_only=True))
        critic.eval()
        #
        episode_info = {}
        # iteratively run environment until done
        for step in range(args.n_step):
            # select greedy action
            action = self.select_action(observation)
            # send to env
            next_observation, reward, done, info = env.step(action)
            # update observation
            observation = next_observation
            step       += 1
            # update info
            for k in info:
                if k == 'reward':
                    info[k] = torch.min(info[k])
                if k not in episode_info:
                    episode_info[k] = [info[k]]
                else:
                    episode_info[k].append(info[k])

            # display
            if args.verbose:
                print(f'{step=} {reward=}')
                print()
            # stopping condition
            if done:
                break
        return episode_info

    def test(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        #
        for episode in range(args.n_test_episode):
            # test
            info = self.test_episode()
            # add more info
            info = self.add_test_info(episode, info)
            # logging
            monitor.step(info)
        # export csv
        monitor.export_csv()
