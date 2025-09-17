import argparse
import torch
import os

torch.set_printoptions(4, sci_mode=False)

def create_folders(args):
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.csv_dir, exist_ok=True)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='milp')
    parser.add_argument('--solver', type=str, default='sac')
    
    # train
    parser.add_argument('--n_train_epoch', type=int, default=500)
    parser.add_argument('--n_train_episode', type=int, default=1000)
    parser.add_argument('--n_test_episode', type=int, default=100)
    
    # dataset
    parser.add_argument('--n_var', type=int, default=10)
    parser.add_argument('--n_constraint', type=int, default=10)
    
    # sac parameter
    parser.add_argument('--n_buffer', type=int, default=500000) # 1e6
    parser.add_argument('--n_start_learning', type=int, default=2000) # 20000
    parser.add_argument('--batch_size', type=int, default=100) # 64 by CleanRL, 12000 to utilize all RTX 4080
    parser.add_argument('--actor_lr', type=float, default=1e-3) # 3e-4
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--q_lr', type=float, default=1e-3) # 3e-4
    parser.add_argument('--n_optimize', type=int, default=5)
    parser.add_argument('--n_target_update', type=int, default=1) # 8000
    parser.add_argument('--tau', type=float, default=0.001) # 1.0
    parser.add_argument('--autotune_entropy', action='store_true')
    parser.add_argument('--target_entropy_scale', type=float, default=0.89)
    parser.add_argument('--alpha', type=float, default=0.2) # fixed entropy
    
    # pointernet model parameter
    parser.add_argument('--softmax_temperature', type=float, default=1.0)
    parser.add_argument('--clip_logit', type=float, default=10.0)
    parser.add_argument('--n_glimpse', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_input', type=int, default=3)
    parser.add_argument('--n_embed', type=int, default=128)
    
    # data directory
    parser.add_argument('--checkpoint_dir', type=str, default='./data/checkpoint')
    parser.add_argument('--dataset_dir', type=str, default='./data/dataset')
    # parser.add_argument('--figure_dir', type=str, default='./data/figure')
    parser.add_argument('--csv_dir', type=str, default='./data/csv')
    
    # metric
    # parser.add_argument('--metric', type=str, default='value')
    # parser.add_argument('--n_smooth', type=int, default=1)
    
    # device
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    # other flags
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=36)
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    return args