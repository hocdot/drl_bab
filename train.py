from solver.sac.solver import SACSolver
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    if args.solver == 'sac':
        solver = SACSolver(args)
    else:
        raise NotImplementedError
    solver.train()
