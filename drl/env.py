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
    
    def reset(self):
        # extract data
        self.episode += 1
        self.n_step = 1
        # load instance
        c, A, b, input_bounds = self.load_instance()
        self.solver = MilpBabSolver(c, A, b, input_bounds, self.args)
        self.expanded_action = [-1] if len(self.solver.queue) else []
        # gather observation
        observation = self.solver.get_observation()
        # done
        done = len(self.solver.queue) <= 0
        return observation, done

    def step(self, action):
        # update
        actions, next_observations, expanded_actions, done, rewards = self.solver.branch_and_bound(action)
        self.train_action = actions
        self.train_next_observation = next_observations
        self.expanded_action = expanded_actions
        self.train_done = done
        self.train_reward = rewards
        # get reward
        reward = - len(self.solver.queue)
        # get info
        next_observation = self.solver.get_observation()
        info = self.solver.get_info()
        done = len(self.solver.queue) <= 0
        return next_observation, reward, done, info
