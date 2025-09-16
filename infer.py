from heuristic.baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


if __name__ == '__main__':
    args = get_args()
    for solver_class in [CostBasedDecider, RandomDecider, RoundingErrorDecider, SACDecider]:
        solver = solver_class(args)
        solver.test()
        print()
