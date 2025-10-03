from heuristic.sac.sac import SACDecider
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    args.mode = 'train'
    if args.solver == 'sac':
        solver = SACDecider(args)
    else:
        raise NotImplementedError

    solver.train()
