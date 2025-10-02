import pandas as pd
import os

from heuristic.milp_baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.dnnv_baseline.fsb import FsbDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args

def infer_milp(args):
    base_label = f'{args.mode}_{args.dataset}_{args.n_var}_{args.n_constraint}'
    for solver_class in [CostBasedDecider, RandomDecider, RoundingErrorDecider]:
        path = os.path.join(args.csv_dir, 'test', f'{base_label}_{solver_class.__name__}.csv')
        if not os.path.exists(path):
            solver = solver_class(args)
            infos =solver.test()
            df = pd.DataFrame(infos)
            df.to_csv(path, index=False)    

    solver_class = SACDecider
    solver = solver_class(args)
    for n_enc_dec_layer in [1, 2, 3]:
        for enc_dec_type in ['gru', 'lstm']:
            args.enc_dec_type = enc_dec_type
            args.n_enc_dec_layer = n_enc_dec_layer
            path = os.path.join(args.csv_dir, 'test', f'{base_label}_{solver_class.__name__}_{args.enc_dec_type}_{args.n_enc_dec_layer}.csv')
            # if os.path.exists(path):
            #     continue
            solver = solver_class(args)
            if not solver.load():
                continue
            infos = solver.test()
            df = pd.DataFrame(infos)
            df.to_csv(path, index=False)
            
            
def infer_dnnv(args):
    base_label = f'{args.mode}_{args.dataset}'
    for solver_class in [FsbDecider]:
        path = os.path.join(args.csv_dir, 'test', f'{base_label}_{solver_class.__name__}.csv')
        # if os.path.exists(path):
        #     continue
        solver = solver_class(args)
        infos = solver.test()
        df = pd.DataFrame(infos)
        df.to_csv(path, index=False)

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    
    if args.dataset == 'milp':
        infer_milp(args)
    else:
        infer_dnnv(args)
    