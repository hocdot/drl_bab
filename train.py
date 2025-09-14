from solver.sac.solver import SACSolver
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    solver = SACSolver(args)
    solver.train()
