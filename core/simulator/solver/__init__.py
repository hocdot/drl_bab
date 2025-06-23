from . import random, pointer_net

def create(args):
    return eval(args.solver).Solver(args)
