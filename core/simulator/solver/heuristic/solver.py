import numpy as np
import torch

from ..base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args): # share
        super().__init__(args)

    def select_action(self, observation):
        # observation: [bs, n_layer, n_neuron, n_feature]
        # extract data
        mask  = observation[..., -1]
        score = observation[..., 0] * mask
        bs    = observation.shape[0]
        args  = self.args
        # initialize action
        action = []
        for i in range(bs):
            candidates = torch.where(score[i] == score[i].max()) # .tolist()
            index      = np.random.choice(np.arange(len(candidates[0])))
            a          = [_[index].item() for _ in candidates]
            action.append(a)
        if args.verbose:
            print(f'{action=}')
        return np.array(action)
