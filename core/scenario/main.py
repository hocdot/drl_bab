import numpy as np
import simulator

def main(args):
    solver = simulator.create_solver(args)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    elif args.mode == 'train_offline':
        solver.train_offline()
    else:
        raise NotImplementedError
