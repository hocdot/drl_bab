import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import numpy as np
import os

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
    return x, y

def plot_boxplot(args):
    # scenarios
    solvers       = ['heuristic', 'greedy', 'propose', 'propose']
    scores        = ['', '', 'actor', 'critic']
    gnn_classes   = ['', '', 'Linear', 'Linear']
    args.n_smooth = 1

    labels = []
    Y = []
    for i in range(len(solvers)):
        # assign args
        args.solver    = solvers[i]
        args.score     = scores[i]
        args.gnn_class = gnn_classes[i]
        label          = get_label(args)
        # load csv
        try:
            x, y = load(args)
            Y.append(y)
            labels.append(label)
        except:
            raise
    # plot
    plt.boxplot(Y, tick_labels=labels)
    # decorate
    plt.legend()
    plt.xlabel('method')
    plt.ylabel(f'{args.metric}')
    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.mode}_{args.metric}.pdf')
    plt.savefig(path)

    plt.yscale('log')
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.mode}_{args.metric}_log.pdf')
    plt.savefig(path)
