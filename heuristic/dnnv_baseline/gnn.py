from ..base import Base

class GnnDecider(Base):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        # action in active neuron index format
        action = self.env._get_gnn_action()
        assert len(action) == len(observation), f'{len(action)=} {[o.shape for o in observation]=}'
        assert all(0 <= a <= observation.shape[-2] for a in action), f'{observation.shape=} {action=}'
        return action

    def __repr__(self):
        return f'GnnDecider(mode={self.args.mode}, dataset={self.args.dataset})'
    