from . import random, pointer_net, fsb

def create(args):
    return eval(args.solver).Solver(args)
