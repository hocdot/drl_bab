import pandas as pd
import os

def load(args):
    label = f'{args.solver}'
    path = os.path.join(args.csv_dir, args.mode, f'{label}.csv')
    df = pd.read_csv(path)
    return df

def table(args):
    # list of solvers
    solvers = ['greedy', 'oracle', 'sac']
    # initialize data
    dfs = []
    #
    for solver in solvers:
        # assign args
        args.solver = solver
        # load
        df = load(args)
        df['solver'] = solver
        # append
        dfs.append(df)
    # concate
    df = pd.concat(dfs)
    print(df)
