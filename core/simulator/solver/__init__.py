from . import greedy
from . import propose
from . import heuristic

from .propose import OfflineBuffer

def create_solver(args):
    return eval(args.solver).Solver(args)
