import pandas as pd
import os

from heuristic.milp_baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.dnnv_baseline import FsbDecider, BabsrDecider, PolarityDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args

def infer(args):
    
    if args.dataset == 'milp':
        base_label = f'{args.dataset}_{args.n_var}_{args.n_constraint}'
        solver_classes = [CostBasedDecider, RandomDecider, RoundingErrorDecider]
    else:
        base_label = f'{args.dataset}'
        solver_classes = [FsbDecider, PolarityDecider, RandomDecider]
        
    solver_classes += [SACDecider]
    for solver_class in solver_classes:
        if solver_class in [RandomDecider, PolarityDecider]:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}.csv')
        else:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}_top{args.decision_topk_test}.csv')
            
        if not os.path.exists(path) or solver_class == SACDecider:
            solver = solver_class(args)
            solver.test(output_path=path)

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    infer(args)
    