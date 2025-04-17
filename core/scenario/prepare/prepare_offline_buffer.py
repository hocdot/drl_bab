import pandas as pd
import numpy as np
import simulator
import pickle
import torch
import tqdm
import os

def get_onnx_to_id(args):
    path = os.path.join(args.benchmark_dir, args.benchmark, 'train.csv')
    onnx_to_id = {}
    i = 0
    for _, (onnx, _) in pd.read_csv(path).iterrows():
        if onnx not in onnx_to_id:
            onnx_to_id[onnx] = i
        i += 1
    return onnx_to_id

# def select_action(observation, env, args):
#     # observation: list of Layer_i: [bs, n_active_neuron_i, n_feature]
#     # extract data
#     feature, heuristic_score, active_neuron_index, n_active_neuron = observation
#     # extract hyperparam
#     n_layer = len(feature)
#     bs      = feature[0].shape[0]
#     top_k   = args.top_k_prepare
#     # initialize action, action.shape = (bs, 2)
#     actions = [] # layer of (bs, k)
#     for layer in range(n_layer):
#         score = heuristic_score[layer].cpu() # (bs, n_active_neuron_i)
#         k = min(score.shape[1], top_k)
#         _, best_index = torch.topk(score, k, dim=1) # (bs, k)
#         layer_action = [] # (bs, k)
#         for b in range(bs):
#             la = torch.index_select(active_neuron_index[layer][b], dim=0, index=best_index[b]) # k
#             layer_action.append(la)
#         layer_action = torch.stack(layer_action)
#         actions.append(layer_action)
#     rewards = env.top_k_reward(actions) # layer of (bs, k)
#     # compute selected action
#     best_scores, best_neuron_indices = [], []
#     heuristic_scores = []
#     for layer in range(n_layer):
#         score = torch.zeros_like(rewards[layer])
#         for b in range(bs):
#             mask     = torch.isin(active_neuron_index[layer][b], actions[layer][b])
#             indices  = torch.nonzero(mask, as_tuple=True)[0]
#             score[b] = heuristic_score[layer][b, indices] # bs, k
#         heuristic_scores.append(score)
#         best_score, best_index = torch.max(score, dim=1)
#         best_neuron_index      = torch.tensor([actions[layer][b, best_index[b]] for b in range(bs)])
#         best_scores.append(best_score)
#         best_neuron_indices.append(best_neuron_index)
#     best_scores         = torch.stack(best_scores).T         # (bs, 1)
#     best_neuron_indices = torch.stack(best_neuron_indices).T # (bs, 1)
#     best_layer          = torch.argmax(best_scores, dim=1)   # (bs, )
#     best_neuron         = torch.gather(best_neuron_indices, 1, best_layer.unsqueeze(-1)).squeeze(-1)
#     selected_action = np.array([[best_layer[i].item(), best_neuron[i].item()] for i in range(bs)])
#     # return selected_action, actions, rewards
#     return selected_action, actions, heuristic_scores

def select_action(observation, env, args):
    # observation: list of Layer_i: [bs, n_active_neuron_i, n_feature]
    # extract data
    feature, heuristic_score, active_neuron_index, n_active_neuron = observation
    # extract hyperparam
    n_layer = len(feature)
    bs      = feature[0].shape[0]
    top_k   = args.top_k_prepare
    # initialize action, action.shape = (bs, 2)
    actions = [] # layer of (bs, k)
    for layer in range(n_layer):
        score = heuristic_score[layer].cpu() # (bs, n_active_neuron_i)
        k = min(score.shape[1], top_k)
        _, best_index = torch.topk(score, k, dim=1) # (bs, k)
        layer_action = [] # (bs, k)
        for b in range(bs):
            la = torch.index_select(active_neuron_index[layer][b], dim=0, index=best_index[b]) # k
            layer_action.append(la)
        layer_action = torch.stack(layer_action)
        actions.append(layer_action)
    # heuristic_score = env.top_k_reward(actions) # layer of (bs, k)
    # compute selected action
    # initialize action, action.shape = (bs, 2)
    best_scores, best_neuron_indices = [], []
    for layer in range(n_layer):
        score                  = heuristic_score[layer] # bs, n_active_neuron_i
        best_score, best_index = torch.max(score, dim=1)
        best_neuron_index      = torch.tensor([active_neuron_index[layer][b, best_index[b]] for b in range(bs)])
        best_scores.append(best_score)
        best_neuron_indices.append(best_neuron_index)
    best_scores         = torch.stack(best_scores).T         # (bs, n_layer)
    best_neuron_indices = torch.stack(best_neuron_indices).T # (bs, n_layer)
    best_layer          = torch.argmax(best_scores, dim=1)   # (bs, )
    best_neuron         = torch.gather(best_neuron_indices, 1, best_layer.unsqueeze(-1)).squeeze(-1)
    selected_action = np.array([[best_layer[i].item(), best_neuron[i].item()] for i in range(bs)])
    # return selected_action, actions, rewards
    return selected_action, actions, heuristic_score

def prepare_episode(env, buffer, iteration, args):
    # reset environment
    observation, done = env.reset()
    assert not done, 'cho hai generate ban the nhi'
    # extract onnx_id
    onnx_id = torch.tensor(env.onnx_to_id[env.onnx])
    # iteratively run environment until done
    for step in range(args.n_step):
        try:
            # extract top k reward
            action, actions, rewards = select_action(observation, env, args)
            # send to env
            next_observation, reward_to_print, done, info = env.step(action)
        except:
            raise
        unf_1 = info['unf_1']
        unf_2 = info['unf_2']
        assert len(unf_1[1].shape) == 3 == len(unf_2[1].shape)
        buffer.push(observation[0], # feature
                    observation[2], # active neuron index
                    unf_1,
                    unf_2,
                    actions,
                    rewards,
                    onnx_id,
                    done)
        # update observation
        observation = next_observation
        # display
        if args.verbose:
            print(f'{step=} {reward_to_print=}')
        # stopping condition
        if done:
            break

def prepare_offline_buffer(args):
    # initialize env
    env = simulator.Env(args)
    buffer = simulator.OfflineBuffer(args.buffer_size)
    env.onnx_to_id = get_onnx_to_id(args)
    # clean data dir
    if args.clear_buffer:
        os.system(f'rm -r {args.data_dir}/{args.mode}/*.pkl')
    # extract args
    for episode in tqdm.tqdm(range(args.n_prepare_episode), desc='preparing buffer'):
        # generate and save full feedback buffer if not exists
        d = os.path.join(args.data_dir, args.mode)
        if not os.path.exists(d):
            os.makedirs(d)
        for iter in range(args.n_prepare_iter):
            path = os.path.join(d, f'buffer_{episode}_{iter}.pkl')
            if os.path.exists(path):
                env.reset()
                print(f'[-] {os.path.basename(path)} existed, skipping')
            else:
                prepare_episode(env, buffer, iter, args)
                if len(buffer):
                    with open(path, 'wb') as fp:
                        pickle.dump(buffer, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    # reset buffer
                    buffer.reset()
                    print(f'[+] created {os.path.basename(path)}')
