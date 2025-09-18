from heapq import merge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from arguments import get_args

def load(args, path, metric):
    df = pd.read_csv(path)
    y = df[metric].to_numpy()
    return y

def plot_cdf(args):
    plt.cla()
    plt.clf()
    # scenarios
    args.dataset = 'milp'
    args.mode = 'test'
    solvers = ['RandomDecider', 'CostBasedDecider', 'RoundingErrorDecider',  'SACDecider']
    for solver in solvers:
        # assign args
        args.solver = solver
        label = f'{args.solver}'
        # load csv
        path = os.path.join(args.csv_dir, 'test', f'{args.mode}_{args.dataset}_{args.solver}.csv')
        y = load(args, path, 'value')
        y = np.sort(y)
        plt.plot(y, label=label)
    # decorate
    plt.ylabel(f'value')
    plt.legend()
    plt.tight_layout()
    # save figure
    path = os.path.join(args.figure_dir, f'cdf.jpg')
    plt.savefig(path)
    plt.cla()
    plt.clf()
    
def plot_line(args):
    plt.cla()
    plt.clf()

    args.solver = 'sac'
    args.mode = 'train'
    label = f'{args.solver}'
    fig, ax = plt.subplots(1, 2)
    for i, metric in enumerate(['actor_loss', 'critic_loss']):
        path = os.path.join(args.csv_dir, 'train', f'{args.mode}_{args.dataset}_{args.solver}.csv')
        y = load(args, path, metric)
        ax[i].plot(y, '-', label=metric)
        ax[i].set_xlabel('epoch')
        ax[i].set_ylabel('loss')
        ax[i].legend()
    plt.tight_layout()
    # save figure
    path = os.path.join(args.figure_dir, f'train_{args.solver}_loss.jpg')
    plt.savefig(path)

if __name__ == '__main__':
    args = get_args()
    plot_cdf(args)
    plot_line(args)