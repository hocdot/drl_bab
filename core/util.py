import argparse
import torch
import os

def create_folders(args):
    ls = [args.csv_dir, args.figure_dir, args.onnx_dir,
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
    # environment
    parser.add_argument('--iwb_threshold', type=int, default=-5.1)
    parser.add_argument('--ns_batch_size', type=int, default=64)

    parser.add_argument('--n_train_episode', type=int, default=113)
    parser.add_argument('--n_test_episode', type=int, default=1)
    parser.add_argument('--n_prepare_episode', type=int, default=1)
    parser.add_argument('--n_prepare_iter', type=int, default=1)
    parser.add_argument('--n_train_offline_step', type=int, default=10000)
    parser.add_argument('--n_step', type=int, default=100) # maximum number of step per episode
    parser.add_argument('--attacker_timeout', type=float, default=2.0) # maximum number of step per episode
    # solver
    parser.add_argument('--solver', type=str, default='greedy')
    # test
    parser.add_argument('--onnx', type=str, default='mnist-net_256x2.onnx')
    parser.add_argument('--vnnlib', type=str, default='prop_1_0.03.vnnlib')
    parser.add_argument('--objective_id', type=int, default=0)
    # top-k for greedy solver
    parser.add_argument('--top_k', type=int, default=3)
    # top-k for propose solver
    parser.add_argument('--top_k_actor', type=int, default=10000)
    # sac solver
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--actor_n_hidden', type=int, default=64)
    parser.add_argument('--critic_n_hidden', type=int, default=64)
    parser.add_argument('--embed_n_hidden', type=int, default=64)
    parser.add_argument('--n_embed', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.005) # target smoothing coeff
    parser.add_argument('--patience', type=int, default=100) # early stop after 100 episodes no improvement
    # parser.add_argument('--entropy_lr', type=float, default=3e-4)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=5e-4)
    parser.add_argument('--actor_iter', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=100000000)
    parser.add_argument('--start_learning', type=int, default=30) # frames
    parser.add_argument('--end_guiding', type=int, default=3000) # frames
    parser.add_argument('--actor_update_interval', type=int, default=2)
    parser.add_argument('--target_update_interval', type=int, default=1)
    # gnn solver
    parser.add_argument('--embed_class', type=str, default='Linear')
    parser.add_argument('--gnn_class', type=str, default='GCNConv')
    # data directory
    parser.add_argument('--instance_dir', type=str, default='../data/instance')
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    parser.add_argument('--model_dir', type=str, default='../data/model')
    parser.add_argument('--data_dir', type=str, default='../data/data')
    parser.add_argument('--onnx_dir', type=str, default='../data/onnx')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    # plot
    parser.add_argument('--metric', type=str, default='total_reward')
    parser.add_argument('--n_smooth', type=int, default=1)
    # other
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--score', type=str, default='actor')
    parser.add_argument('--single_objective', action='store_true')
    parser.add_argument('--load_state_dict', action='store_true')
    parser.add_argument('--clear_buffer', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    # set default device cuda
    set_default_device(args)
    # additional args
    args.n_observation = 7
    return args
