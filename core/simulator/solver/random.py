import numpy as np
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        n_subproblem = len(observation)
        action = []
        for i in range(n_subproblem):
            mask = observation[i, :, -1]
            active_neurons = torch.where(mask == 1)[0].detach().cpu().numpy()
            next_node = int(np.random.choice(active_neurons))
            observation[i, next_node, -1] = 0
            action.append(next_node)
        return action
