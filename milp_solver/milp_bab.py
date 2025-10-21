from beartype import beartype
import numpy as np
import torch

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
        root.prev_action = -1
        status = root.bound()
        self.queue = [root] if status == 'optimal' else []
        
    @property
    def remaining_subproblem(self):
        return len(self.queue)
    
    @beartype
    def branch_and_bound(self, action: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # branch and bound
        queue = []
        prev_actions = [] # list<n_next_subproblem><int> list of action leading to new subproblem, for testing
        double_actions = [] # list<n_subproblem><ndarray:2, n_var, n_input>: for training
        double_next_observations = [] # list<n_subproblem><ndarray:2, n_var, n_input>: for training
        double_dones = [] # list<n_subproblem><ndarray:2>: for training
        double_rewards = [] # list<n_subproblem>: for training
        assert self.remaining_subproblem == len(action), f'{self.remaining_subproblem=} {len(action)=}'
        for node, ai in zip(self.queue, action):
            # branch
            assert node.active[ai] == 1, f'{ai=} {node.active[ai]=}'
            if self.args.verbose:
                print(f'    * branching {ai=} {node.x_opt[ai]=}')
            # init
            train_i = np.full([2], ai, dtype=np.int64)
            next_observation = np.zeros([2, self.n_var, self.args.n_input], dtype=np.float32)
            done = np.ones([2], dtype=np.int32)
            reward = 0
            for j, child in enumerate(node.branch(ai)):
                # bound
                status = child.bound()
                # only process feasible region
                if status == 'optimal':
                    # update best feasible solution
                    if child.active.sum() <= 0:
                        if child.primal_estimate > self.best_y: 
                            self.best_x = child.x_opt # best integer solution so far
                            self.best_y = child.primal_estimate
                            if self.args.verbose:
                                print(f'    + found {self.best_y=} {len(queue)=}')
                                print(f'      {self.best_x=} {len(queue)=}')
                    # prune branch with dual bound worst than current best solution
                    if child.dual_estimate > self.best_y and child.active.sum() > 0:
                        # add information about new branch
                        queue.append(child)
                        child.prev_action = ai
                        prev_actions.append(ai)
                        next_observation[j] = self.get_node_observation(child)
                        done[j] = 0
                        reward -= 1 / 125.0
                        if self.args.verbose:
                            print(f'    - appended {len(queue)=}')
                    else:
                        next_observation[j] = self.get_node_observation(child)
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
        while self.remaining_subproblem:
            node = self.queue.pop(0)
            del node
        self.queue = queue
        
        # expanded information for training/inference
        return (torch.tensor(double_actions, dtype=torch.long), 
                torch.tensor(double_next_observations, dtype=torch.float32), 
                torch.tensor(prev_actions, dtype=torch.long), 
                torch.tensor(double_dones, dtype=torch.int8), 
                torch.tensor(double_rewards, dtype=torch.float32))

    @beartype
    def get_node_observation(self, node: Node) -> np.ndarray:
        observation = np.stack([
            node.fraction_score,
            self.c / 1000 * node.active,
            node.active,
        ]).T
        return observation


    @beartype
    def get_observation(self, batch=None) -> tuple[torch.Tensor, torch.Tensor]:
        observation = []
        prev_actions = []
        for node in self.queue:
            node_observation = self.get_node_observation(node)
            observation.append(node_observation)
            prev_actions.append(node.prev_action)
        observation = torch.from_numpy(np.array(observation)).to(torch.float32) # n_subproblem, n_var, n_feature
        prev_actions = torch.tensor(prev_actions, dtype=torch.long) # n_subproblem,
        return observation, prev_actions
