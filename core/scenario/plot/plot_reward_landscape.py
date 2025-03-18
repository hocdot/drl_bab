import matplotlib.pyplot as plt
import numpy as np
import simulator
import os

def plot_reward_landscape(args):
    # initialize env
    env = simulator.Env(args)
    # reset
    observation = env.reset()
    # obtain full feedback
    rewards, actions = env.full_feedback()
    n_subproblem, _, _ = actions.shape
    n_layer, n_neuron = env.action_shape
    for i in range(n_subproblem):
        # initialize y
        y = np.full(env.action_shape, rewards[i].min())
        # assign value
        for l, (j, k) in enumerate(actions[i]):
            y[j, k] = rewards[i][l]
        # plot
        for j in range(n_layer):
            plt.scatter(np.arange(n_neuron), y[j], label=f'layer_{j}')
        plt.legend()
        # plt.imshow(y)
        # save figure
        path = os.path.join(args.figure_dir, f'{args.scenario}_{i}.pdf')
        print(f'[+] saving figure at "{path}"')
        plt.savefig(path)
        # clear figure
        plt.cla()
        plt.clf()
