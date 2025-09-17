from heuristic.baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    for solver_class in [CostBasedDecider, RandomDecider, RoundingErrorDecider, SACDecider][-1:]:
        solver = solver_class(args)
        solver.test()
        
