import argparse
import torch
import os

torch.set_printoptions(4, sci_mode=False)

def create_folders(args):
    ls = [args.csv_dir, args.figure_dir, args.benchmark_dir,
          args.model_dir, args.data_dir]
    for folder in ls:
        if not os.path.exists(folder):
            os.makedirs(folder)

def set_default_device(args):
    torch.set_default_device(args.device)

base_folder = os.path.dirname(os.path.dirname(__file__))

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='debug')
    parser.add_argument('--mode', type=str, default='test')
    # environment

    # data directory
    parser.add_argument('--benchmark_dir', type=str, default='../data/benchmark')
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    parser.add_argument('--model_dir', type=str, default='../data/model')
    parser.add_argument('--data_dir', type=str, default='../data/data')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    parser.add_argument('--benchmark', type=str, default='cnn')
    # plot
    parser.add_argument('--metric', type=str, default='total_reward')
    parser.add_argument('--n_smooth', type=int, default=1)
    # other
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    # additional args
    return args
