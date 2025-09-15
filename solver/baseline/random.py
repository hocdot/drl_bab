import numpy as np

from ..base import BaseSolver

class RandomSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        n_sub, n_var, n_feature = observation.shape
        active = observation[:, :, -1]
        action = []
        for i in range(n_sub):
            indices = np.where(active[i] == 1)[0]
            j = np.random.choice(indices)
            action.append(j)
        return action

    def __repr__(self):
        return f'RandomSolver(mode={self.args.mode}, dataset={self.args.dataset})'