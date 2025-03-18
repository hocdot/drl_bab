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
        observation_mask = observation[..., -1].bool()
        bs, n_layer, n_neuron, _ = observation.shape
        # convert min_rewards, max_reward to (bs, n_layer, n_neuron)
        min_rewards_ = torch.zeros(bs, n_layer, n_neuron)
        max_rewards_ = torch.zeros_like(min_rewards_)
        for i in range(bs):
            for j, a in enumerate(actions[i]):
                min_rewards_[i, a[0], a[1]] = min_rewards[i, j]
                max_rewards_[i, a[0], a[1]] = max_rewards[i, j]
        min_rewards = min_rewards_
        max_rewards = max_rewards_
        _actions    = []
        #
        for i in range(bs):
            # filter node with both min_reward and max_reward positive
            mask = torch.logical_and(min_rewards[i] >= 0, max_rewards[i] >= 0).int() * observation_mask[i].int()
            heuristic_score = observation[i, :, :, 0]
            if mask.sum() > 0:
                score = heuristic_score * mask + -10000 * (1 - mask)
            else:
                # filter node with one among min_reward and max_reward positive
                mask = torch.logical_or(min_rewards[i] >= 0, max_rewards[i] >= 0).int() * observation_mask[i].int()
                if mask.sum() > 0:
                    score = heuristic_score * mask + -10000 * (1 - mask)
                # filter node with both min_reward and max_reward positive
                else:
                    mask = observation_mask[i].int()
                    score = heuristic_score * mask + -10000 * (1 - mask)
            candidates = torch.where(score == score.max())
            index      = np.random.choice(np.arange(len(candidates[0])))
            a          = [_[index].item() for _ in candidates]
            # print(f'{a=} {score.shape=}')
            _actions.append(a)
        action = np.array(_actions)
        if args.verbose:
            print(action)
        return action
