from solver.baseline import CostBasedSolver, RandomSolver, RoundingErrorSolver
from solver.sac.solver import SACSolver
from arguments import get_args


if __name__ == '__main__':
    args = get_args()
    for solver_class in [CostBasedSolver, RandomSolver, RoundingErrorSolver, SACSolver]:
        solver = solver_class(args)
        solver.test()
        print()
