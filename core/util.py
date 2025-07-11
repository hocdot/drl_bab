import argparse
import torch
import sys
import os

sys.path.append('neuralsat/src/')
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
    parser.add_argument('--scenario', type=str, default='main')
    parser.add_argument('--mode', type=str, default='test')
    # neuralsat
    parser.add_argument('--ns_batch_size', type=int, default=128)
    parser.add_argument('--ns_n_step', type=int, default=4)
    # heuristic solver
    parser.add_argument('--solver', type=str, default='random')
    # pointer net
    parser.add_argument('--softmax_temperature', type=float, default=1.0)
    parser.add_argument('--clip_logit', type=float, default=10.0)
    parser.add_argument('--n_process', type=int, default=3)
    parser.add_argument('--n_glimpse', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_input', type=int, default=4)
    # training
    parser.add_argument('--lrs_step_size', type=int, default=5e3)
    parser.add_argument('--n_train_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lrs_gamma', type=float, default=0.96)
    parser.add_argument('--lr', type=float, default=1e-3)
    # data directory
    parser.add_argument('--benchmark_dir', type=str, default='../data/benchmark')
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    parser.add_argument('--model_dir', type=str, default='../data/model')
    parser.add_argument('--data_dir', type=str, default='../data/data')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    parser.add_argument('--benchmark', type=str, default='mnist')
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
    print(f'Using device: {args.device}')
    return args
