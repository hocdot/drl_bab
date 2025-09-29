from ..base import Base


class FsbDecider(Base):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        action = self.env.solver.get_fsb_action()
        action = [self.env.reverse_active_neuron_index[int(a)] for a in action]
        assert len(action) == len(observation), f'{len(action)=} {[o.shape for o in observation]=}'
        assert all(0 <= a <= observation.shape[-2] for a in action), f'{observation.shape=} {action=}'
        return action

    def __repr__(self):
        return f'FsbDecider(mode={self.args.mode}, dataset={self.args.dataset})'