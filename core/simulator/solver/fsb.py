import numpy as np
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, embed, observation):
        n_subproblem = len(observation)
        action = self.env.get_fsb_action()
        print(f'{action=}')
        return action
