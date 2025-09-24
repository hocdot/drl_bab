import numpy as np

from ..base import Base

def select_cost_based_action(observation):
    n_sub, n_var, n_feature = observation.shape
    active = observation[:, :, -1]
    score = observation[:, :, 1]
    action = []
    for i in range(n_sub):
        indices = np.where(active[i] == 1)[0]
        j = np.argmax(score[i][indices])
        k = indices[j]
        # j = np.random.choice(indices)
        action.append(k)
    return action

class CostBasedDecider(Base):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        return select_cost_based_action(observation)

    def __repr__(self):
        return f'CostBasedDecider(mode={self.args.mode}, dataset={self.args.dataset})'