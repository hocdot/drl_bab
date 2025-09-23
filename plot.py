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
    plt.figure(figsize=(8, 10))
    
    args.dataset = 'milp'
    args.mode = 'test'
    solvers = ['RandomDecider', 'CostBasedDecider', 'RoundingErrorDecider']
    base_path = os.path.join(args.csv_dir, 'test', f'{args.mode}_{args.dataset}_{args.n_var}_{args.n_constraint}')
    for solver in solvers:
        # assign args
        args.solver = solver
        label = f'{args.solver}'
        # load csv
        path = f'{base_path}_{args.solver}.csv'
        y = load(args, path, 'value')
        y = np.sort(y)
        plt.plot(y, label=label)
    
    args.solver = 'SACDecider'
    for n_enc_dec_layer in [1, 2, 3]:
        for enc_dec_type in ['gru', 'lstm']:
            label = f'{args.solver}_{enc_dec_type}_{n_enc_dec_layer}'
            path = f'{base_path}_{args.solver}_{enc_dec_type}_{n_enc_dec_layer}.csv'
            if not os.path.exists(path):
                continue
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
    print(f'[+] saved {path=}')
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
        for n_enc_dec_layer in [1, 2, 3]:
            for enc_dec_type in ['gru', 'lstm']:
                label = f'{args.solver}_{enc_dec_type}_{n_enc_dec_layer}'
                path = os.path.join(args.csv_dir, 'train', f'{args.mode}_{args.dataset}_{args.n_var}_{args.n_constraint}_{args.solver}_{enc_dec_type}_{n_enc_dec_layer}.csv')
                if not os.path.exists(path):
                    continue
                y = load(args, path, metric)
                ax[i].plot(y, '-', label=label)
        ax[i].set_xlabel('epoch')
        ax[i].set_ylabel('loss')
        ax[i].legend()
    plt.tight_layout()
    # save figure
    path = os.path.join(args.figure_dir, f'train_{args.solver}_loss.jpg')
    plt.savefig(path)
    print(f'[+] saved {path=}')
    plt.cla()
    plt.clf()

if __name__ == '__main__':
    args = get_args()
    plot_cdf(args)
    # plot_line(args)