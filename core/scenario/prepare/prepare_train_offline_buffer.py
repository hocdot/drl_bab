import pandas as pd
import numpy as np
import simulator
import pickle
import torch
import tqdm
import os

def get_onnx_to_id(args):
    path = os.path.join(args.onnx_dir, 'train_instance.csv')
    onnx_to_id = {}
    i = 0
    for _, (onnx, _) in pd.read_csv(path).iterrows():
        if onnx not in onnx_to_id:
            onnx_to_id[onnx] = i
        i += 1
    return onnx_to_id

def select_action(observation, env, args):
    # compute optimal randomized top-k action
    min_rewards, max_rewards, actions = env.full_feedback()
    rewards = 0.5 * (min_rewards + max_rewards)
    bs = rewards.shape[0]
    _actions = []
    for i in range(bs):
        indices = torch.argsort(rewards[i], descending=True)
        indices = indices.detach().cpu().numpy()
        indices = indices[:args.top_k]
        idx     = np.random.choice(indices)
        _actions.append(actions[i, idx])
    action = np.array(_actions)
    # convert reward to desired format
    # actions.shape=(bs, n_active_neuron, 2) rewards.shape=(bs, n_active_neuron)
    # want rewards.shape=(bs,n_layer,n_hidden), inactive neuron's reward = 0
    min_rewards_    = torch.zeros(list(observation.shape[:-1]))
    max_rewards_    = torch.zeros(list(observation.shape[:-1]))
    n_active_neuron = rewards.shape[-1]
    for i in range(bs):
        for j in range(n_active_neuron):
            layer  = actions[i, j, 0]
            neuron = actions[i, j, 1]
            min_rewards_[i, layer, neuron] = min_rewards[i, j]
            max_rewards_[i, layer, neuron] = max_rewards[i, j]
    # extract mask
    mask = observation[..., -1]
    return action, min_rewards_, max_rewards_, mask

def select_action_2(observation, env, args):
    # compute optimal myopic action
    min_rewards, max_rewards, actions = env.full_feedback()
    rewards = 0.5 * (min_rewards + max_rewards)
    bs, _ = rewards.shape
    _actions = []
    for i in range(bs):
        # filter node with both min_reward and max_reward positive
        mask = torch.logical_and(min_rewards[i] >= 0, max_rewards[i] >= 0).int()
        if mask.sum() > 0:
            score = min_rewards[i] * mask + args.iwb_threshold * (1 - mask)
        else:
            # filter node with one among min_reward and max_reward positive
            mask = torch.logical_or(min_rewards[i] >= 0, max_rewards[i] >= 0).int()
            if mask.sum() > 0:
                score = min_rewards[i] * mask + args.iwb_threshold * (1 - mask)
            # filter node with both min_reward and max_reward positive
            else:
                score = max_rewards[i]
        idx = torch.argmax(score).detach().cpu().numpy()
        _action = actions[i, idx]
        _reward = rewards[i, idx]
        _actions.append(_action)
    action = np.array(_actions)
    # convert reward to desired format
    # actions.shape=(bs, n_active_neuron, 2) rewards.shape=(bs, n_active_neuron)
    # want rewards.shape=(bs,n_layer,n_hidden), inactive neuron's reward = 0
    min_rewards_    = torch.zeros(list(observation.shape[:-1]))
    max_rewards_    = torch.zeros(list(observation.shape[:-1]))
    n_active_neuron = rewards.shape[-1]
    for i in range(bs):
        for j in range(n_active_neuron):
            layer  = actions[i, j, 0]
            neuron = actions[i, j, 1]
            min_rewards_[i, layer, neuron] = min_rewards[i, j]
            max_rewards_[i, layer, neuron] = max_rewards[i, j]
    # extract mask
    mask = observation[..., -1]
    return action, min_rewards_, max_rewards_, mask

def prepare_episode(env, train_offline_buffer, iteration, args):
    # reset environment
    observation, done = env.reset()
    assert not done, 'cho hai generate ban the nhi'
    # extract onnx_id
    iwb     = torch.tensor(env.initial_worst_bound)
    onnx_id = torch.tensor(env.onnx_to_id[env.onnx])
    # iteratively run environment until done
    for step in range(args.n_step):
        try:
            # extract all rewards
            # action, min_rewards, max_rewards, mask = select_action(observation, env, args)
            if iteration == 0:
                action, min_rewards, max_rewards, mask = select_action_2(observation, env, args)
            else:
                action, min_rewards, max_rewards, mask = select_action(observation, env, args)
            # send to env
            next_observation, reward, done, info = env.step(action)
        except:
            print(f'[+] error: {env.onnx},{env.vnnlib}')
            path = os.path.join(args.onnx_dir, 'error.csv')
            with open(path, 'a') as fp:
                fp.write(f'{env.onnx},{env.vnnlib}\n')
            break
        next_observation_train = info['next_observation_train']
        for i in range(len(observation)):
            train_offline_buffer.push(observation[i], min_rewards[i],
                                 max_rewards[i], mask[i],
                                 next_observation_train[i],
                                 iwb, onnx_id)
        # update observation
        observation = next_observation
        assert min_rewards.shape == mask.shape
        # display
        if args.verbose:
            print(f'{step=} {reward=}')
        # stopping condition
        if done:
            break

def prepare_train_offline_buffer(args):
    # initialize env
    env = simulator.Env(args)
    train_offline_buffer = simulator.OfflineBuffer(args.buffer_size)
    #
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
        
        for iter in range(10):
            path = os.path.join(d, f'offline_buffer_{episode}_{iter}.pkl')
            if os.path.exists(path):
                env.reset()
                print(f'[-] {os.path.basename(path)} existed, skipping')
            else:
                prepare_episode(env, train_offline_buffer, iter, args)
                if len(train_offline_buffer):
                    with open(path, 'wb') as fp:
                        pickle.dump(train_offline_buffer, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    # reset buffer
                    train_offline_buffer.reset()
                    print(f'[+] created {os.path.basename(path)}')
