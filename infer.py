import pandas as pd
import os

from heuristic.milp_baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.dnnv_baseline import FsbDecider, UpbDecider, PolarityDecider, BabsrDecider, GnnDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args

def infer(args):
    
    if args.dataset == 'milp':
        base_label = f'{args.dataset}_{args.n_var}_{args.n_constraint}'
        solver_classes = [CostBasedDecider, RandomDecider, RoundingErrorDecider]
    else:
        base_label = f'{args.dataset}'
        solver_classes = [FsbDecider, UpbDecider, GnnDecider]
        
    solver_classes = [SACDecider]
    solver_classes = [GnnDecider]
    for solver_class in solver_classes:
        base_path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}')
        postfix = ''
        if solver_class in [RandomDecider, PolarityDecider]:
            # path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}.csv')
            pass
        elif solver_class in [UpbDecider, FsbDecider, SACDecider, BabsrDecider, GnnDecider]:
            if solver_class in [SACDecider]:
                postfix += f'_{args.decision_heuristic}'
            postfix += f'_top{args.decision_topk_test}'
            # path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}_top{args.decision_topk_test}.csv')
            if solver_class == SACDecider:
                # path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}_top{args.decision_topk_test}_checkpoint_{args.checkpoint_postfix}.csv')
                if args.checkpoint_postfix:
                    postfix += f'_checkpoint_{args.checkpoint_postfix}'
                if args.n_test_fallback > 0:
                    postfix += f'_fallback_{args.n_test_fallback}'
                if args.use_zero_embedding:
                    postfix += f'_zeroemb'
        else:
            raise NotImplementedError(f'{solver_class.__name__} is not supported')
        
        path = f'{base_path}{postfix}.csv'
        print(f'Output: {path=}')
        
        if not os.path.exists(path) or solver_class == SACDecider:
            os.system(f'rm -rf {path}')
            solver = solver_class(args)
            solver.test(output_path=path)

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    assert not args.use_early_stop
    infer(args)
    