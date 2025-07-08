import numpy as np
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        n_subproblem = len(observation)
        action = []
        for i in range(n_subproblem):
            mask = observation[i, :, -1]
            active_neurons = torch.where(mask == 1)[0].detach().cpu().numpy()
            next_node = int(np.random.choice(active_neurons))
            observation[i, next_node, -1] = 0
            action.append(next_node)
        return action

    def test(self):
        # extract args
        monitor = self.monitor
        args = self.args
        env  = self.env
        # TODO: env.reset(), change to another model
        # reset environment
        observation, self.edge_index, self.edge_weight = env.reset()
        self.edge_weight = self.edge_weight.to(args.device)
        self.edge_index  = self.edge_index.to(args.device)
        # embed
        embed = actor.embedding(observation.to(args.device), self.edge_weight, self.edge_index)

        # procedure before test
        self.prepare_test(observation)
        print(f'[+] testing {monitor.label}')
        #
        done = False
        step = 0
        total_reward = 0
        while not done:
            action = self.select_action(embed, observation)
            next_observation, reward, done, info = env.step(action)
            n_subproblem = len(next_observation)
            total_reward += reward.sum().item()
            info.update({
                'step': step,
                'total_reward': total_reward,
                'avg_reward': reward.mean().item(),
                'n_subproblem': n_subproblem,
                'done': done,
            })
            monitor.step(info)
            print()
            step += 1
            observation = next_observation
            if step >= args.ns_n_step:
                break
        monitor.export_csv()
