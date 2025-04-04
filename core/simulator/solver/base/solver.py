import numpy as np
import torch
import time

import simulator

class BaseSolver:

    def __init__(self, args): # share
        # save args
        self.args = args
        # create env
        self.env = simulator.Env(args)
        # load monitor
        self.monitor = simulator.Monitor(args)

    def test_add_info(self, episode, episode_info): # share
        info = {
            'episode': episode,
            'step': self.step,
            'total_reward': np.sum(episode_info['reward']),
            'avg_remaining': np.mean(episode_info['remaining']),
            'awb': np.mean(episode_info['worst_bound']),
            'runtime': np.mean(episode_info['runtime']),
            'n_visited': episode_info['visited'][-1],
            'n_step': len(episode_info['visited']),
        }
        return info

    def select_action(self, observation):
        # observation: [bs, n_layer, n_neuron, n_feature]
        # return: <np.array> : [bs, 2]
        raise NotImplementedError

    def test_episode(self): # share
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        # reset environment
        observation, done = env.reset()
        assert not done, 'cho hai gen data ban the nhi'

        # initialize episode info
        episode_info = {'runtime': []}
        # iteratively run environment until done
        for step in range(args.n_step):
            # stopping condition
            if done or observation is None:
                break
            # select greedy action
            tic    = time.monotonic()
            action = self.select_action(observation)
            toc    = time.monotonic()
            episode_info['runtime'].append(toc - tic)
            # send to env
            next_observation, reward, done, info = env.step(action)
            # update observation
            observation = next_observation
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
                if observation is not None:
                    mask = observation[..., -1]
                    n_active_neuron = float(mask.sum(dim=[1, 2]).detach().cpu().numpy().mean())
                    print(f'{self.step=} {reward.mean()=:0.2f} {n_active_neuron=:0.2f}')
                # print(f'{self.step=} {reward=} {n_active_neuron=:0.2f}')
                # print()

        return episode_info

    def test(self): # share
        # extract args
        args      = self.args
        env       = self.env
        monitor   = self.monitor
        self.step = 0
        #
        try:
            for episode in range(args.n_test_episode):
                # test
                info = self.test_episode()
                if info is not None:
                    # add more info
                    info = self.test_add_info(episode, info)
                    # logging
                    monitor.step(info)
        except KeyboardInterrupt:
            pass
        finally:
            # export csv
            monitor.export_csv()
