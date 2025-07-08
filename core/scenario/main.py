import simulator
import torch

def main(args):
    # create solver
    solver = simulator.solver.create(args)
    # run training
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        raise NotImplementedError
