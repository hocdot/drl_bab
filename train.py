import pandas as pd
import os

from heuristic.sac.sac import SACDecider
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    args.mode = 'train'
    if args.solver == 'sac':
        solver = SACDecider(args)
    else:
        raise NotImplementedError
    train_infos = solver.train()

    # export csv
    df = pd.DataFrame(train_infos)
    df.to_csv(os.path.join(args.csv_dir, f'{args.mode}_{args.dataset}_{args.solver}.csv'), index=False)
    