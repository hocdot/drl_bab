import pandas as pd
import os

from heuristic.baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    for solver_class in [CostBasedDecider, RandomDecider, RoundingErrorDecider, SACDecider][:-1]:
        solver = solver_class(args)
        infos =solver.test()
        df = pd.DataFrame(infos)
        df.to_csv(os.path.join(args.csv_dir, 'test', f'{args.mode}_{args.dataset}_{solver_class.__name__}.csv'), index=False)
