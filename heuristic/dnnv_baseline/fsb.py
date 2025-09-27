from ..base import Base


class FsbDecider(Base):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        action = self.env.solver.get_fsb_action()
        assert len(action) == len(observation), f'{len(action)=} {[o.shape for o in observation]=}'
        print(f'{action=}')
        return action

    def __repr__(self):
        return f'FsbDecider(mode={self.args.mode}, dataset={self.args.dataset})'