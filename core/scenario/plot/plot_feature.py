import matplotlib.pyplot as plt
import numpy as np
import simulator
import torch
import os

def plot_feature(args):
    # initialize env
    env = simulator.Env(args)
    # reset
    observation = env.reset()
    # n_subproblem, n_layer, n_neuron, n_feature
    # obtain full feedback
    rewards, actions = env.full_feedback()
    n_subproblem, _, _ = actions.shape
    n_layer, n_neuron = env.action_shape
    for i in range(n_subproblem):
        # sort indices by reward
        indices = torch.argsort(rewards[i], descending=True).detach().cpu().numpy()
        # sort rewards and actions
        sub_rewards = rewards[i][indices].detach().cpu().numpy()
        sub_actions = actions[i][indices]
        # normalize reward
        sub_rewards = (sub_rewards - sub_rewards.min()) / (sub_rewards.max() - sub_rewards.min())
        for j in range(args.n_observation):
            # plot reward
            plt.plot(sub_rewards, label='reward')
            # plot_feature
            feature = torch.stack([observation[i, k, l, j] for k, l in sub_actions]).detach().cpu().numpy()
            # feature = (feature - feature.min()) / (feature.max() - feature.min())
            plt.plot(feature, label=f'feature_{j}')
            # decorate
            plt.legend()
            # save figure
            path = os.path.join(args.figure_dir, f'{args.scenario}_{i}_{j}.pdf')
            print(f'[+] saving figure at "{path}"')
            plt.savefig(path)
            # clear figure
            plt.cla()
            plt.clf()
