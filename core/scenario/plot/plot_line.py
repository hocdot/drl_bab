import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def smooth_line(x, y, args):
    T = int(np.floor(len(x) // args.n_smooth) * args.n_smooth)
    x = x[:T].reshape(-1, args.n_smooth)
    x = x[:, 0]
    y = y[:T].reshape(-1, args.n_smooth)
    y = np.mean(y, axis=1)
    return x, y

def plot_line(args):
    df = pd.read_csv('../data/csv/train/pointer_net.csv')
    metrics = ['value', 'actor_loss', 'critic_loss', 'n_subproblem', 'total_reward']
    for metric in metrics:
        x = df['global_step'].to_numpy()
        y = df[metric].to_numpy()
        x, y = smooth_line(x, y, args)
        plt.plot(x, y, '-o')
        path = os.path.join(args.figure_dir, f'{args.scenario}_{metric}.pdf')
        plt.ylabel(metric)
        plt.savefig(path)
        plt.cla()
        plt.clf()
