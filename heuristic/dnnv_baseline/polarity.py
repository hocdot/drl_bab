import numpy as np

from ..base import Base

def select_polarity_action(observation):
    n_sub, n_var, n_feature = observation.shape
    active = observation[:, :, -1]
    lbs = observation[:, :, 0]
    ubs = observation[:, :, 1]
    score = (lbs + ubs) / (ubs - lbs + 1e-8)
    action = []
    for i in range(n_sub):
        indices = np.where(active[i] == 1)[0]
        j = np.argmax(score[i][indices])
        k = indices[j]
        action.append(int(k))
    return action

class PolarityDecider(Base):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        return select_polarity_action(observation)

    def __repr__(self):
        return f'PolarityDecider(mode={self.args.mode}, dataset={self.args.dataset})'