import numpy as np
import torch

from ..base import BaseSolver

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
            # filter node with both min_reward and max_reward positive
            mask = torch.logical_and(min_rewards[i] >= 0, max_rewards[i] >= 0).int()
            if mask.sum() > 0:
                score = min_rewards[i] * mask + args.iwb_threshold * (1 - mask)
            else:
                # filter node with one among min_reward and max_reward positive
                mask = torch.logical_or(min_rewards[i] >= 0, max_rewards[i] >= 0).int()
                if mask.sum() > 0:
                    score = min_rewards[i] * mask + args.iwb_threshold * (1 - mask)
                # filter node with both min_reward and max_reward positive
                else:
                    score = max_rewards[i]
            # print(score.shape, observation.shape)
            idx = torch.argmax(score).detach().cpu().numpy()
            _action = actions[i, idx]
            # _reward = rewards[i, idx]
            _actions.append(_action)
        action = np.array(_actions)
        if args.verbose:
            print(action)
        return action
