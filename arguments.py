import argparse
import torch
import os

torch.set_printoptions(4, sci_mode=False)

def create_folders(args):
    args.checkpoint_dir = os.path.join(args.data_dir, args.checkpoint_dirname)
    args.dataset_dir = os.path.join(args.data_dir, args.dataset_dirname)
    args.figure_dir = os.path.join(args.data_dir, args.figure_dirname)
    args.csv_dir = os.path.join(args.data_dir, args.csv_dirname)
    
    os.makedirs(os.path.join(args.checkpoint_dir, args.dataset), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, args.dataset), exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    os.makedirs(os.path.join(args.csv_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.csv_dir, 'test'), exist_ok=True)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='milp')
    parser.add_argument('--solver', type=str, default='sac')
    
    # train
    parser.add_argument('--n_train_epoch', type=int, default=100)
    parser.add_argument('--n_train_episode', type=int, default=100)
    parser.add_argument('--n_train_offline_episode', type=int, default=100)
    parser.add_argument('--n_test_episode', type=int, default=1000)
    parser.add_argument('--save_csv', action='store_true')
    
    # milp parameter
    parser.add_argument('--n_var', type=int, default=10)
    parser.add_argument('--n_constraint', type=int, default=10)
    parser.add_argument('--n_max_remaining_subproblem', type=int, default=1000)
    parser.add_argument('--n_max_step', type=int, default=50)
    
    # dnnv parameter
    parser.add_argument('--use_attack', action='store_true')
    parser.add_argument('--attack_timeout', type=float, default=1.0)
    parser.add_argument('--n_vnnlib_per_onnx', type=int, default=1)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--dnnv_solver_batch', type=int, default=1000)
    parser.add_argument('--dnnv_skip_lower_bound', type=float, default=-10)
    parser.add_argument('--save_offline_data', action='store_true')
    parser.add_argument('--n_save_offline_data', type=int, default=10)
    
    # sac parameter
    parser.add_argument('--enc_dec_type', type=str, default='lstm') # lstm
    parser.add_argument('--n_enc_dec_layer', type=int, default=1) # 3
    parser.add_argument('--n_buffer', type=int, default=150000) # 1e6
    parser.add_argument('--n_start_learning', type=int, default=500) # 20000
    parser.add_argument('--batch_size', type=int, default=128) # 64 by CleanRL, 12000 to utilize all RTX 4080
    parser.add_argument('--actor_lr', type=float, default=1e-4) # 3e-4
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--q_lr', type=float, default=1e-4) # 3e-4
    parser.add_argument('--n_optimize', type=int, default=5)
    parser.add_argument('--n_target_update', type=int, default=1) # 8000
    parser.add_argument('--tau', type=float, default=0.001) # 1.0
    parser.add_argument('--autotune_entropy', action='store_true')
    parser.add_argument('--target_entropy_scale', type=float, default=0.89)
    parser.add_argument('--alpha', type=float, default=0.2) # fixed entropy
    
    # pointernet model parameter
    parser.add_argument('--n_input', type=int, default=3)
    parser.add_argument('--softmax_temperature', type=float, default=1.0)
    parser.add_argument('--clip_logit', type=float, default=10.0)
    parser.add_argument('--n_glimpse', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_embed', type=int, default=128)
    
    # gcn embedding parameter
    parser.add_argument('--gcn_embed_n_hidden', type=int, default=128)
    parser.add_argument('--gcn_embed_n_epoch', type=int, default=50)
    parser.add_argument('--gcn_embed_batch_size', type=int, default=128)
    parser.add_argument('--gcn_embed_lr', type=float, default=3e-4)
    
    # data directory
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dirname', type=str, default='checkpoint')
    parser.add_argument('--dataset_dirname', type=str, default='dataset')
    parser.add_argument('--figure_dirname', type=str, default='figure')
    parser.add_argument('--csv_dirname', type=str, default='csv')
    
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