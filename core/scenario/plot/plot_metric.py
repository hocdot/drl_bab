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

def load(args):
    path = os.path.join(args.csv_dir, args.mode, f'{args.solver}_{args.capture_effect}.csv')
    df = pd.read_csv(path)
    x = df['step'].to_numpy()
    y = df[args.metric].to_numpy()
    x, y = smooth_line(x, y, args)
    return x, y

def plot_line(args):
    # scenarios
    capture_effects = [True]
    solvers = ['random', 'qmix_boltzman', 'optimal']

    for capture_effect, solver in it.product(capture_effects, solvers):
        # assign args
        args.capture_effect = capture_effect
        args.solver = solver
        # load csv
        try:
            x, y = load(args)
            # plot
            plt.plot(x, y, '-o', label=f'{solver}{"_capture" if capture_effect else ""}')
        except:
            # raise
            pass
    # decorate
    plt.legend()
    plt.xlabel('scheduling step')
    plt.ylabel(f'{args.metric}')
    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.mode}_{args.metric}.jpg')
    plt.savefig(path)
