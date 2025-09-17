import numpy as np

from .milp_util import create_model, Node

class MilpBabSolver:
    
    def __init__(self, c, A, b, input_bounds, args):
        self.args = args
        self.c = c
        self.A = A
        self.b = b
        self.input_bounds = input_bounds
        self.n_var = c.shape[0]
        
        # init best solution
        self.best_x = None
        self.best_y = - np.inf
        
        # init solver
        lp_model = create_model(c, A, b, input_bounds)
        root = Node(lp_model)
        status = root.bound()
        self.queue = [root] if status == 'optimal' else []
    
    def branch_and_bound(self, action):
        # branch and bound
        queue = []
        prev_actions = [] # list<n_next_subproblem><int> list of action leading to new subproblem, for testing
        double_actions = [] # list<n_subproblem><ndarray:2, n_var, n_input>: for training
        double_next_observations = [] # list<n_subproblem><ndarray:2, n_var, n_input>: for training
        double_dones = [] # list<n_subproblem><ndarray:2>: for training
        double_rewards = [] # list<n_subproblem>: for training
        for node, i in zip(self.queue, action):
            # branch
            assert node.active[i] == 1, f'{i=} {node.active[i]=}'
            if self.args.verbose:
                print(f'    * branching {i=} {node.x_opt[i]=}')
            # init
            train_i = np.full([2], i, dtype=np.int64)
            next_observation = np.zeros([2, self.n_var, self.args.n_input], dtype=np.float32)
            done = np.ones([2], dtype=np.int32)
            reward = 0
            for j, child in enumerate(node.branch(i)):
                # bound
                status = child.bound()
                # only process feasible region
                if status == 'optimal':
                    # update best feasible solution
                    if child.active.sum() <= 0 and child.primal_estimate > self.best_y: 
                        self.best_x = child.x_opt # best integer solution so far
                        self.best_y = child.primal_estimate
                        if self.args.verbose:
                            print(f'    + found {self.best_y=} {len(queue)=}')
                            print(f'      {self.best_x=} {len(queue)=}')
                    # prune branch with dual bound worst than current best solution
                    if child.dual_estimate > self.best_y and child.active.sum() > 0:
                        # add information about new branch
                        queue.append(child)
                        prev_actions.append(i)
                        next_observation[j] = self.get_node_observation(child)
                        done[j] = 0
                        reward -= 1 / 125
                        if self.args.verbose:
                            print(f'    - appended {len(queue)=}')
                    else:
                        # next_observation[j] = self.get_node_observation(child)
                        if self.args.verbose:
                            print(f'    - pruned {len(queue)=}')
                else:
                    assert status == 'infeasible', f'{status=} {len(queue)=}'
                    if self.args.verbose:
                        print(f'    ? {status=} {len(queue)=}')
            # store training information
            double_actions.append(train_i)
            double_next_observations.append(next_observation)
            double_dones.append(done)
            double_rewards.append(reward)
        
        # update self.queue
        while len(self.queue):
            node = self.queue.pop(0)
            del node
        self.queue = queue
        
        # expanded information for training/inference
        return double_actions, double_next_observations, prev_actions, double_dones, double_rewards


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


    def get_observation(self):
        info = self.get_info()
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
