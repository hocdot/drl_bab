from . import greedy, greedy_2, greedy_3
from . import heuristic
from . import propose, propose_2

from .propose import OfflineBuffer

def create_solver(args):
    return eval(args.solver).Solver(args)
