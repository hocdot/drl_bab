from milp_solver.milp_bab import MilpBabSolver
from dataset.milp import generate_hard_milp
from dataset import create_dataloader

class Env:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = create_dataloader(args)
        self.loader = iter(self.dataloader_dict[args.mode])
        self.episode = 0

    def load_milp_instance(self):
        args = self.args
        if args.mode == 'test':
            try:
                instance = next(self.loader)
            except StopIteration:
                self.loader = iter(self.dataloader_dict[args.mode])
                instance = next(self.loader)
            c = instance['c'][0]
            A = instance['A'][0]
            b = instance['b'][0]
            input_bounds = instance['bounds'][0]
        elif args.mode == 'train':
            c, A, b, input_bounds = generate_hard_milp(args)
        else:
            raise NotImplementedError
        return c, A, b, input_bounds
    
    def reset_solver(self):
        if self.args.dataset == 'milp':
            c, A, b, input_bounds = self.load_milp_instance()
            solver = MilpBabSolver(c, A, b, input_bounds, self.args)
        else:
            raise NotImplementedError
        return solver
    
    def reset(self):
        # load new instance
        self.episode += 1
        self.n_step = 1
        self.solver = self.reset_solver()
        self.prev_actions = [-1] if self.solver.remaining_subproblem else []
        observation = self.solver.get_observation()
        done = self.solver.remaining_subproblem <= 0
        return observation, done

    def step(self, action):
        double_actions, double_next_observations, prev_actions, double_dones, double_rewards = self.solver.branch_and_bound(action) # 2 x subproblems
        # assert len(double_actions) == len(double_next_observations), f'{len(double_actions)=} {len(double_next_observations)=}'
        # assert len(double_actions) == len(double_dones), f'{len(double_actions)=} {len(double_dones)=}'
        # assert len(double_actions) == len(double_rewards), f'{len(double_actions)=} {len(double_rewards)=}'
        # assert len(double_actions) == len(prev_actions), f'{double_actions=} {prev_actions=}'
        
        # save for training
        self.train_action = double_actions # (n_subproblem, 2)
        self.train_next_observation = double_next_observations # (n_subproblem, 2, n_var, n_feature)
        self.train_done = double_dones # (n_subproblem, 2)
        self.train_reward = double_rewards # (n_subproblem,)
        # last actions
        self.prev_actions = prev_actions # (n_subproblem,)
        # get reward
        reward = - self.solver.remaining_subproblem
        # get info
        remaining_next_observation = self.solver.get_observation() # remaining subproblems
        info = self.solver.get_info()
        done = self.solver.remaining_subproblem <= 0
        early_stop = self.solver.remaining_subproblem >= self.args.n_max_subproblem
        return remaining_next_observation, reward, done, info, early_stop
