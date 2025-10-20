import pandas as pd
import os

from heuristic.milp_baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.dnnv_baseline.fsb import FsbDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args

def infer(args):
    
    if args.dataset == 'milp':
        base_label = f'{args.mode}_{args.dataset}_{args.n_var}_{args.n_constraint}'
        solver_classes = [CostBasedDecider, RandomDecider, RoundingErrorDecider]
    else:
        base_label = f'{args.mode}_{args.dataset}'
        solver_classes = [FsbDecider]
        
    solver_classes += [SACDecider]
    for solver_class in solver_classes:
        if solver_class == SACDecider:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}' + ('.csv' if args.use_scale_action else '_no_scale.csv'))
        else:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}.csv')
            
        if not os.path.exists(path) or solver_class == SACDecider:
            solver = solver_class(args)
            infos = solver.test()
            df = pd.DataFrame(infos)
            df.to_csv(path, index=False)    

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    infer(args)
    