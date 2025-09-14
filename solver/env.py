import numpy as np
import torch
import time

from dataset.milp import generate_hard_ilp
from dataset import create_dataloader

from . import util

class Env:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = create_dataloader(args)
        if args.mode == 'test':
            self.loader = iter(self.dataloader_dict[args.mode])
        self.episode = 0

    def load_root(self):
        args = self.args
        if args.mode == 'test':
            try:
                instance = next(self.loader)
            except StopIteration:
                self.loader = iter(self.dataloader_dict[args.mode])
                instance = next(self.loader)
            self.c = instance['c'][0]
            self.A = instance['A'][0]
            self.b = instance['b'][0]
            self.bounds = instance['bounds'][0]
        elif args.mode == 'train':
            self.c, self.A, self.b, self.bounds = generate_hard_ilp(args)
        else:
            raise NotImplementedError
        milp = util.create_milp(self.c, self.A, self.b, self.bounds)
        root = util.Node(milp)
        status = root.bound()
        if status == 'optimal':
            self.queue = [root]
            self.expanded_action = [-1]
        else:
            self.queue = []
            self.expanded_action = []

    def get_info(self):
        info = []
        for node in self.queue:
            node_info = {
                'primal_estimate': node.primal_estimate,
                'dual_estimate': node.dual_estimate,
                'active': node.active,
                'c': self.c,
                'fraction_score': node.fraction_score,
                # TODO: may be adding c A b would help
            }
            info.append(node_info)
        return info

    def get_node_observation(self, node):
        observation = np.stack([
            node.fraction_score,
            self.c / 1000 * node.active,
            node.active,
        ]).T
        return observation

    def get_observation(self, info):
        observation = []
        for node_info in info:
            node_observation = np.stack([
                # TODO: add embedding
                node_info['fraction_score'],
                node_info['c'] / 1000 * node_info['active'],
                node_info['active'],
            ]).T
            observation.append(node_observation)
        observation = np.array(observation) # n_subproblem, n_var, n_feature
        return observation

    def reset(self):
        # extract data
        self.episode += 1
        self.n_step = 1
        # load instance
        self.load_root()
        # init
        self.best_x = None
        self.best_y = - np.inf
        # gather observation
        info = self.get_info()
        observation = self.get_observation(info)
        # done
        done = len(self.queue) <= 0
        return observation, done

    def branch_and_bound(self, action):
        # branch and bound
        args = self.args
        queue = []
        expanded_action = [] # list<n_next_subproblem><int> list of action leading to new subproblem, for testing
        train_action = [] # list<n_subproblem><ndarray:2, n_var, n_input>: for training
        train_next_observation = [] # list<n_subproblem><ndarray:2, n_var, n_input>: for training
        train_done = [] # list<n_subproblem><ndarray:2>: for training
        train_reward = [] # list<n_subproblem>: for training
        for node, i in zip(self.queue, action):
            # branch
            assert node.active[i] == 1, f'{i=} {node.active[i]=}'
            if args.verbose:
                print(f'    * branching {i=} {node.x_opt[i]=}')
            # init
            train_i = np.full([2], i, dtype=np.int64)
            next_observation = np.zeros([2, args.n_var, args.n_input], dtype=np.float32)
            done = np.ones([2], dtype=np.int32)
            reward = 0
            for j, child in enumerate(node.branch(i)):
                # bound
                status = child.bound()
                # only process feasible region
                if status == 'optimal':
                    # update best feasible solution
                    if child.active.sum() <= 0:
                        if child.primal_estimate > self.best_y:
                            self.best_x = child.x_opt
                            self.best_y = child.primal_estimate
                            if args.verbose:
                                print(f'    + found {self.best_y=} {len(queue)=}')
                                print(f'      {self.best_x=} {len(queue)=}')
                    # prune branch with dual bound worst than current best solution
                    if child.dual_estimate > self.best_y and child.active.sum() > 0:
                        # add information about new branch
                        queue.append(child)
                        expanded_action.append(i)
                        next_observation[j, ...] = self.get_node_observation(child)
                        done[j] = 0
                        reward -= 1 / 125
                        if args.verbose:
                            print(f'    - appended {len(queue)=}')
                    else:
                        if args.verbose:
                            print(f'    - pruned {len(queue)=}')
                else:
                    if args.verbose:
                        print(f'    ? {status=} {len(queue)=}')
            # store training information
            train_action.append(train_i)
            train_next_observation.append(next_observation)
            train_done.append(done)
            train_reward.append(reward)
        # update self.queue
        # print(f'{len(self.queue)=} {len(queue)=}')
        while len(self.queue):
            node = self.queue.pop(0)
            del node
        self.queue = queue
        # expanded information for inference
        self.expanded_action = expanded_action
        # expanded information for training
        self.train_action           = train_action
        self.train_next_observation = train_next_observation
        self.train_done             = train_done
        self.train_reward           = train_reward

    def step(self, action):
        # update
        self.branch_and_bound(action)
        # get reward
        reward = - len(self.queue)
        # get info
        info = self.get_info()
        # gather next observation
        next_observation = self.get_observation(info)
        #
        done = len(self.queue) <= 0
        # if not done:
            # print(next_observation[0, :])
            # print()
        return next_observation, reward, done, info
