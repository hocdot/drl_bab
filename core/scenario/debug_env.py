import numpy as np
import simulator
import torch

def select_action(observation):
    print(observation.shape)
    n_subproblem = len(observation)
    action = []
    for i in range(n_subproblem):
        mask = observation[i, :, -1]
        active_neurons = torch.where(mask == 1)[0].detach().cpu().numpy()
        next_node = int(np.random.choice(active_neurons))
        observation[i, next_node, -1] = 0
        action.append(next_node)
    return action

def debug_env(args):
    # create environment
    env = simulator.Env(args)
    # reset environment
    observation, edge_index, edge_weight = env.reset()
    #
    done = False
    step = 0
    while not done:
        action = select_action(observation)
        next_observation, reward, done, info = env.step(action)
        print(f'{step=}', reward.shape)
        step += 1
        observation = next_observation
        if step >= 10:
            break
