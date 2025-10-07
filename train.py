from heuristic.sac.sac import SACDecider
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    if args.solver == 'sac':
        solver = SACDecider(args)
    else:
        raise NotImplementedError

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'train_offline':
        assert args.dataset == 'dnnv'
        solver.train_offline()
    else:
        raise NotImplementedError
