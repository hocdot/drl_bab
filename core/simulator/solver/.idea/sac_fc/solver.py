import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools
import torch

import simulator


from .normalizer import RunningMeanStd, RunningMinMax
from .buffer import ReplayBuffer, Transition
from .entropy import Entropy
from .critic import Critic
from .actor import Actor

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
        # initialize actor
        self.actor = Actor(args)
        # initialize critic
        self.critic1 = Critic(args)
        self.critic2 = Critic(args)
        self.critic1_target = Critic(args)
        self.critic2_target = Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # automatic entropy tuning
        self.entropy = Entropy(args)
        # initialize optimizer
        self.critic_optimizer   = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                             lr=args.critic_lr)
        self.actor_optimizer    = optim.Adam(list(self.actor.parameters()),
                                             lr=args.actor_lr)
        self.entropy_optimizer  = optim.Adam(list(self.entropy.parameters()),
                                             lr=args.entropy_lr)
        # initialize buffer
        self.buffer = ReplayBuffer(args.replay_buffer_size)
        # initialize observation normalizer
        self.observation_normalizer = RunningMinMax(shape=[args.n_observation])
        self.reward_normalizer      = RunningMeanStd(shape=[1])

    def compute_score(self, observation):
        # extract args
        args = self.args
        actor = self.actor
        # extract data
        mask = observation[..., -1]
        # compute score using Actor
        score, log_prob, _ = self.actor.select_action(observation)
        score = score * mask
        return score, log_prob

    def sample_greedy_score(self, observation):
        bs    = observation.shape[0]
        mask  = observation[..., -1]
        score = observation[..., 0] * mask
        return score

    def score_to_action_with_critic(self, observation, score):
        # extract args
        args         = self.args
        env          = self.env
        critic1      = self.critic1
        critic2      = self.critic1
        bs           = score.shape[0]
        n_top_action = int(args.top_action_ratio * args.n_action)
        # initialize action
        final_action = []
        final_score  = []
        # for each subproblem
        for i in range(bs):
            # select top actions from actor's score
            actor_score, indices = torch.topk(score[i, ...].view(-1), n_top_action)
            # recalculate the score for top actions
            # 1's in the location of the selected action, otherwise fill with zeros
            top_score = torch.eye(args.n_action)[indices].view(*tuple([n_top_action] + env.action_shape))
            #
            top_observation = observation[i, ...].repeat(n_top_action, 1, 1, 1)
            # re-evaluate them using critic
            q_value1    = critic1(top_observation, top_score)
            q_value2    = critic2(top_observation, top_score)
            min_q_value = torch.min(q_value1, q_value2)
            # choose one among top actions with highest q_value
            idx = torch.argmax(min_q_value)
            action = torch.cat(torch.where(top_score[idx] == 1)).detach().cpu().numpy()
            # return final action and score
            final_action.append(action)
            final_score.append(top_score[idx])
        return np.array(final_action), torch.stack(final_score)

    def score_to_action(self, score, log_prob):
        #
        bs = score.shape[0]
        # initialize action
        action = []
        for i in range(bs):
            # get indices of active neurons having max score
            indices = torch.where(score[i] == torch.max(score[i]))
            # select a random neuron with top score
            n_top_neuron = len(indices[0])
            j = np.random.randint(low=0, high=n_top_neuron)
            a = [_[j].item() for _ in indices]
            action.append(a)
        return np.array(action), log_prob

    def select_action(self, observation):
        score, _  = self.compute_score(observation)
        action, _ = self.score_to_action(score, _)
        return action

    def test_episode(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        on      = self.observation_normalizer
        # reset environment
        observation = env.reset()
        done        = False
        # update normalizer
        # TODO: disable observation normalizer update on testing
        on.update(observation)
        norm_observation = on.normalize(observation)
        # initialize episode info
        episode_info = {}
        # iteratively run environment until done
        for step in range(args.n_step):
            # select action using actor
            action = self.select_action(norm_observation)
            # send to env
            next_observation, reward, done, info = env.step(action)
            # update observation
            observation = next_observation
            on.update(observation)
            norm_observation = on.normalize(observation)
            self.step  += 1
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
                print(f'{self.step=} {action[0]=} {reward[0]=} {len(reward)=}')
            # stopping condition
            if done:
                break
        return episode_info

    def train_episode(self): # to be modified
        # extract args
        args     = self.args
        env      = self.env
        actor    = self.actor
        monitor  = self.monitor
        buffer   = self.buffer
        rn       = self.reward_normalizer
        on       = self.observation_normalizer
        # reset environment
        observation = env.reset()
        done        = False
        # update normalizer
        on.update(observation)
        norm_observation = on.normalize(observation)
        # initialize episode info
        episode_info = {'actor_loss': [], 'critic_loss': [], 'entropy_loss': []}
        # iteratively run environment until done
        step = 0
        for step in range(args.n_step):
            # agent select action
            with torch.no_grad():
                if self.step < args.start_learning:
                    score     = self.sample_greedy_score(norm_observation)
                    action, _ = self.score_to_action(score, None)
                else:
                    score, _  = self.compute_score(norm_observation)
                    action, _ = self.score_to_action(score, _)
            # send action to environment
            next_observation, reward, done, info = env.step(action)
            # update normalizer
            if not done:
                on.update(next_observation)
                rn.update(reward[:, None])
            # buffer management
            if not done:
                next_observation_train = info['next_observation_train']
                n_subproblem = len(reward)
                for i in range(n_subproblem):
                    buffer.push(observation[i], torch.tensor(action[i]), next_observation_train[i], reward[i])
                # update observation
                observation = next_observation
                norm_observation = on.normalize(observation)
            self.step += 1
            # optimize network
            actor_loss, critic_loss, entropy_loss = self.optimize()
            # update info
            if actor_loss != 0: episode_info['actor_loss'].append(actor_loss)
            if critic_loss != 0: episode_info['critic_loss'].append(critic_loss)
            if entropy_loss != 0: episode_info['entropy_loss'].append(entropy_loss)
            for k in info:
                if k == 'reward':
                    info[k] = torch.min(info[k])
                if k not in episode_info:
                    episode_info[k] = [info[k]]
                else:
                    episode_info[k].append(info[k])
            # display
            if args.verbose:
                # print(f'{self.step=} {reward=:0.3f} {actor_loss=:0.6f}')
                pass
            # stopping condition
            if done:
                break
        return episode_info

    def optimize(self):
        # extract args
        args              = self.args
        buffer            = self.buffer
        actor             = self.actor
        critic1           = self.critic1
        critic2           = self.critic2
        entropy           = self.entropy
        critic1_target    = self.critic1_target
        critic2_target    = self.critic2_target
        actor_optimizer   = self.actor_optimizer
        critic_optimizer  = self.critic_optimizer
        entropy_optimizer = self.entropy_optimizer
        on                = self.observation_normalizer
        rn                = self.reward_normalizer
        # optimization loop
        actor_losses, critic_losses, entropy_losses = [], [], []
        # training when enough data
        if self.step > args.start_learning:
            # sample data
            transitions = buffer.sample(args.batch_size)
            batch       = Transition(*zip(*transitions))
            # ============
            # train critic
            # ============
            # extract data
            next_observation = on.normalize(torch.stack(batch.next_observation))
            observation      = on.normalize(torch.stack(batch.observation))
            reward           = torch.stack(batch.reward)
            action           = torch.stack(batch.action)
            # compute target
            with torch.no_grad():
                next_score, log_prob  = self.compute_score(next_observation)
                next_action, log_prob = self.score_to_action(next_score, log_prob)
                target_q_value1       = critic1_target(next_observation, next_action) # return all reward
                target_q_value2       = critic2_target(next_observation, next_action) # then argmax_action
                min_target_q_value    = torch.min(target_q_value1, target_q_value2).view(-1)
                next_q_value          = reward + args.gamma * min_target_q_value
                next_q_value          = next_q_value.mean() # - entropy.alpha * log_prob
            # compute estimate
            q_value1 = critic1(observation, action).mean()
            q_value2 = critic2(observation, action).mean()
            critic1_loss = F.mse_loss(q_value1, next_q_value)
            critic2_loss = F.mse_loss(q_value2, next_q_value)
            critic_loss  = (critic1_loss + critic2_loss) / args.batch_size
            # optimize the critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            # report data
            critic_losses.append(float(critic_loss.item()))
            # =======================
            # train actor and entropy
            # =======================
            if self.step % args.actor_update_interval == 0:
                for _ in range(args.actor_update_interval * args.actor_iter):
                    # actor inference
                    score, log_prob  = self.compute_score(observation)
                    action, log_prob = self.score_to_action(score, log_prob)
                    if self.step < args.end_guiding: # guided on heuristic
                        actor_loss  = F.mse_loss(score, observation[..., 0])
                    else: # gradient ascending on critic
                        q_value1    = critic1(observation, action)
                        q_value2    = critic2(observation, action)
                        min_q_value = torch.min(q_value1, q_value2)
                        actor_loss  = (- min_q_value).mean() / args.batch_size
                        # actor_loss     += (entropy.alpha * log_prob - min_q_value).mean() / args.batch_size
                        # entropy inference
                        # with torch.no_grad():
                        #     _, log_prob = self.compute_score(observation, edge_index, edge_weight)
                        # entropy_loss += (-entropy.log_alpha.exp() * (log_prob + entropy.target_entropy)).mean() / args.batch_size
                        entropy_loss = 0
                    # optimize actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    # optimize entropy
                    # entropy_optimizer.zero_grad()
                    # entropy_loss.backward()
                    # entropy_optimizer.step()
                    # entropy.alpha = entropy.log_alpha.exp().item()
                    # report data
                    actor_losses.append(float(actor_loss.item()))
                    # entropy_losses.append(float(entropy_loss.item()))
            # ==================================
            # update the target_nework of critic
            # ==================================
            if self.step % args.target_update_interval == 0:
                for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        # compute average
        avg_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
        avg_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
        avg_entropy_loss = np.mean(entropy_losses) if len(entropy_losses) > 0 else 0
        return avg_actor_loss, avg_critic_loss, avg_entropy_loss

    def add_test_info(self, episode, episode_info):
        info = {
            'episode': episode,
            'step': self.step,
            'total_reward': np.sum(episode_info['reward']),
            'avg_reward': np.mean(episode_info['reward']),
            'avg_remaining': np.mean(episode_info['remaining']),
            'avg_worst_bound': np.mean(episode_info['worst_bound']),
            'n_visited': episode_info['visited'][-1],
            'n_step': len(episode_info['visited']),
        }
        return info

    def add_train_info(self, episode, episode_info):
        info = self.add_test_info(episode, episode_info)
        for key in episode_info:
            if 'loss' in key:
                info[key] = np.mean(episode_info[key])
        return info

    def test(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        # initialize testing variables
        self.step = 0
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

    def train(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        # initialize training variables
        self.patience = 0
        self.step     = 0
        #
        for episode in range(args.n_train_episode):
            # test
            info = self.train_episode()
            # add more info
            info = self.add_train_info(episode, info)
            # logging
            monitor.step(info)
            # TODO: stopping condition
            # export csv
            monitor.export_csv()

    def save(self):
        pass

    def load(self):
        pass
