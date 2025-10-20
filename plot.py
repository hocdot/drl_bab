import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from heuristic.milp_baseline import CostBasedDecider, RandomDecider, RoundingErrorDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args

def load(args, path, metric):
    df = pd.read_csv(path)
    y = df[metric].to_numpy()
    return y

def plot_cdf_milp(args):
    plt.cla()
    plt.clf()
    # scenarios
    plt.figure(figsize=(8, 10))
    
    args.dataset = 'milp'
    args.mode = 'test'
    solver_classes = [CostBasedDecider, RandomDecider, RoundingErrorDecider, SACDecider]
    base_path = os.path.join(args.csv_dir, args.mode, f'{args.mode}_{args.dataset}_{args.n_var}_{args.n_constraint}')
    for solver_class in solver_classes:
        path = f'{base_path}_{solver_class.__name__}.csv'
        y = load(args, path, 'value')
        y = np.sort(y)
        plt.plot(y, label=solver_class.__name__)
    
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
    
# def plot_line(args):
#     plt.cla()
#     plt.clf()

#     args.solver = 'sac'
#     label = f'{args.solver}'
#     fig, ax = plt.subplots(1, 2)
#     for i, metric in enumerate(['actor_loss', 'critic_loss']):
#         path = os.path.join(args.csv_dir, args.mode, f'{args.mode}_{args.dataset}_{args.n_var}_{args.n_constraint}_{SACDecider.__name__}.csv')
#         if not os.path.exists(path):
#             continue
#         print(f'[+] loading {path}')
#         y = load(args, path, metric)
#         ax[i].plot(y, '-', label=SACDecider.__name__)
#         ax[i].set_xlabel('epoch')
#         ax[i].set_ylabel('loss')
#         ax[i].legend()
#     plt.tight_layout()
#     # save figure
#     path = os.path.join(args.figure_dir, f'{args.mode}_{args.solver}_loss.jpg')
#     plt.savefig(path)
#     print(f'[+] saved {path=}')
#     plt.cla()
#     plt.clf()

if __name__ == '__main__':
    args = get_args()
    plot_cdf_milp(args)
    # plot_line(args)