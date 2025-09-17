from verifier.milp_bab import MilpBabSolver
from dataset.milp import generate_hard_milp
from dataset import create_dataloader

class Env:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = create_dataloader(args)
        # if args.mode == 'test':
        self.loader = iter(self.dataloader_dict[args.mode])
        self.episode = 0

    def load_instance(self):
        args = self.args
        if args.mode == 'test' or 1:
            try:
                instance = next(self.loader)
            except StopIteration:
                self.loader = iter(self.dataloader_dict[args.mode])
                instance = next(self.loader)
            c = instance['c'][0]
            A = instance['A'][0]
            b = instance['b'][0]
            input_bounds = instance['bounds'][0]
            # print(f'[debug] loaded instance {args.mode}')
        elif args.mode == 'train':
            c, A, b, input_bounds = generate_hard_milp(args)
        else:
            raise NotImplementedError
        return c, A, b, input_bounds
    
    def reset(self):
        # extract data
        self.episode += 1
        self.n_step = 1
        # load instance
        c, A, b, input_bounds = self.load_instance()
        self.solver = MilpBabSolver(c, A, b, input_bounds, self.args)
        self.prev_actions = [-1] if len(self.solver.queue) else []
        # gather observation
        observation = self.solver.get_observation()
        # done
        done = len(self.solver.queue) <= 0
        return observation, done

    def step(self, action):
        # update
        double_actions, double_next_observations, prev_actions, double_dones, double_rewards = self.solver.branch_and_bound(action) # 2 x subproblems
        # save for training
        self.train_action = double_actions
        self.train_next_observation = double_next_observations
        self.train_done = double_dones
        self.train_reward = double_rewards
        # last actions
        self.prev_actions = prev_actions
        # get reward
        reward = - len(self.solver.queue)
        # get info
        remaining_next_observation = self.solver.get_observation() # remaining subproblems
        info = self.solver.get_info()
        done = len(self.solver.queue) <= 0
        return remaining_next_observation, reward, done, info
