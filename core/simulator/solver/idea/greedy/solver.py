import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from ..base import BaseSolver
import simulator

class Solver(BaseSolver):

    def __init__(self, args): # share
        super().__init__(args)

    def select_action(self, observation):
        # extract data
        env  = self.env
        args = self.args
        # compute optimal myopic action
        min_rewards, max_rewards, actions = env.full_feedback()
        rewards = 0.5 * (min_rewards + max_rewards)
        bs, _ = rewards.shape
        _actions = []
        for i in range(bs):
            idx = torch.argmax(rewards[i]).detach().cpu().numpy()
            _action = actions[i, idx]
            _reward = rewards[i, idx]
            _actions.append(_action)
            if args.verbose and i == 0:
                layer, neuron = _action
                print(f'{layer=} {neuron=} {observation[i, layer, neuron]=} {_reward=}')
                plt.hist(rewards[i].detach().cpu().numpy())
                step = self.step
                path = os.path.join(args.figure_dir, f'reward_hist_{step=}.pdf')
                plt.savefig(path)
                plt.cla()
                plt.clf()
            # DEBUG
        action = np.array(_actions)
        if args.verbose:
            print(action)
        return action
