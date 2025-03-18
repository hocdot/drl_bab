import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import numpy as np
import os

def smooth_line(x, y, args):
    x = x.reshape(-1, args.n_smooth)
    x = x[:, 0]
    y = y.reshape(-1, args.n_smooth)
    y = np.mean(y, axis=1)
    return x, y

def get_label(args):
    if args.solver == 'propose':
        label = f'{args.solver}_{args.score}_{args.gnn_class}'
    else:
        label = f'{args.solver}'
    return label

def load(args):
    label = get_label(args)
    path = os.path.join(args.csv_dir, args.mode, f'{label}.csv')
    df = pd.read_csv(path)
    if 'episode' in df:
        x = df['episode'].to_numpy()
    else:
        x = np.arange(len(df))
    y = df[args.metric].to_numpy()
    x, y = smooth_line(x, y, args)
    return x, y

def plot_line(args):
    # scenarios
    if args.mode == 'test':
#         solvers     = ['heuristic', 'greedy', 'propose', 'propose']
#         scores      = ['', '', 'actor', 'critic']
#         gnn_classes = ['', '', 'Linear', 'Linear']
        solvers     = ['heuristic', 'greedy', 'greedy_2', 'greedy_3', 'propose_2']
        scores      = ['', '', '', '', 'critic']
        gnn_classes = ['', '', '', '', 'Linear']
    elif 'train' in args.mode:
        solvers     = ['propose']
        scores      = ['actor']
        gnn_classes = ['TAGConv']
    else:
        raise NotImplementedError
    indices = None
    for i in range(len(solvers)):
        # assign args
        args.solver    = solvers[i]
        args.score     = scores[i]
        args.gnn_class = gnn_classes[i]
        label          = get_label(args)
        # load csv
        try:
            x, y = load(args)
            if args.mode == 'test':
                if args.solver == 'heuristic':
                    indices = np.argsort(y)
                y = y[indices]
            # plot
            if args.mode == 'test':
                plt.plot(x, y, '-o', label=label)
            elif 'train' in args.mode:
                plt.plot(x, y, '-', label=label)
            else:
                raise NotImplementedError
        except:
            pass
    plt.legend()
    plt.xlabel('objective id')
    plt.ylabel(f'{args.metric}')
    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.mode}_{args.metric}.pdf')
    plt.savefig(path)
