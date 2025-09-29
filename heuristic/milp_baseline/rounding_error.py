import numpy as np

from ..base import Base

def select_rounding_error_action(observation):
    n_sub, n_var, n_feature = observation.shape
    active = observation[:, :, -1]
    score = observation[:, :, 0]
    action = []
    for i in range(n_sub):
        indices = np.where(active[i] == 1)[0]
        j = np.argmax(score[i][indices])
        k = indices[j]
        # j = np.random.choice(indices)
        action.append(int(k))
    return action


class RoundingErrorDecider(Base):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        return select_rounding_error_action(observation)

    def __repr__(self):
        return f'RoundingErrorDecider(mode={self.args.mode}, dataset={self.args.dataset})'