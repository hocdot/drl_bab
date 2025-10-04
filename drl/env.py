from beartype import beartype
import torch
import os

from dataset.milp_dataset import generate_hard_milp
from milp_solver.milp_bab import MilpBabSolver
from dnnv_solver.dnnv_bab import DnnvBabSolver
from prepare.prepare_io import extract_instance
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
        if self.args.mode == 'test':
            try:
                instance = next(self.loader)
            except StopIteration:
                self.loader = iter(self.dataloader_dict[self.args.mode])
                instance = next(self.loader)
            c = instance['c'][0]
            A = instance['A'][0]
            b = instance['b'][0]
            input_bounds = instance['bounds'][0]
        elif self.args.mode == 'train':
            c, A, b, input_bounds = generate_hard_milp(self.args)
        else:
            raise NotImplementedError
        return c, A, b, input_bounds
    
    def load_dnnv_instance(self):
        try:
            instance = next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader_dict[self.args.mode])
            instance = next(self.loader)
        return instance['onnx'][0], instance['vnnlib'][0]
        
    def reset_solver(self):
        if self.args.dataset == 'milp':
            c, A, b, input_bounds = self.load_milp_instance()
            solver = MilpBabSolver(c, A, b, input_bounds, self.args)
            self.observation_embedding = None
        else:
            onnx, vnnlib = self.load_dnnv_instance()
            net, input_shape, objs = extract_instance(onnx, vnnlib)
            assert len(objs) == 1, f'{len(objs)=}'
            solver = DnnvBabSolver(net, input_shape, objs, self.args)
            # load observation embedding
            embed_dir = os.path.join(self.args.dataset_dir, self.args.dataset, 'embed')
            embed_path = os.path.join(embed_dir, f'{os.path.basename(onnx)}____{os.path.basename(vnnlib)}.pt')
            self.observation_embedding = torch.load(embed_path) # (n_active_neuron, n_embed)
        return solver
    
    @beartype
    def reset(self) -> tuple[torch.Tensor, bool]:
        # load new instance
        self.episode += 1
        self.n_step = 1
        self.solver = self.reset_solver()
        self.prev_actions = [-1] if self.solver.remaining_subproblem else []
        done = self.solver.remaining_subproblem <= 0
        observation = self.solver.get_observation() if not done else None
        assert len(observation) == 1, f'{len(observation)=}'
        if self.args.dataset == 'milp':
            return observation, done
        # dnnv
        init_mask = torch.tensor(observation[..., -1])
        self.active_neuron_index = torch.where(init_mask == 1)[1] # (n_active_neuron, )
        self.reverse_active_neuron_index = {int(v): idx for idx, v in enumerate(self.active_neuron_index)}
        # print(f'{self.active_neuron_index=}')
        observation = self._concat_observation_embedding_2d(observation) # (B, n_active_neuron, n_raw_feature + gnn_embedding)
        assert (observation[..., -1] == 1).all(), f'{observation[..., -1]=}'
        return observation, done


    @beartype
    def _concat_observation_embedding_2d(self, observation: torch.Tensor) -> torch.Tensor:
        if self.args.dataset == 'milp':
            return observation
        new_observation = observation[:, self.active_neuron_index] # (B, n_active_neuron, n_raw_feature)
        obs_emb = self.observation_embedding.unsqueeze(0).expand(new_observation.shape[0], -1, -1)  # (B, n_active_neuron, gnn_embedding)
        new_observation = torch.cat([obs_emb, new_observation], dim=-1)
        assert ((new_observation[..., -1] == 0) | (new_observation[..., -1] == 1)).all(), f'{new_observation[..., -1]=}'
        return new_observation

    @beartype
    def _concat_observation_embedding_3d(self, observation: torch.Tensor) -> torch.Tensor:
        if self.args.dataset == 'milp':
            return observation
        new_observation = observation[:, :, self.active_neuron_index] # (B, 2, n_active_neuron, n_raw_feature)
        assert new_observation.shape[1] == 2, f'{new_observation.shape=}'
        obs_emb = self.observation_embedding.unsqueeze(0).unsqueeze(0).expand(new_observation.shape[0], 2, -1, -1)  # (B, 2, n_active_neuron, gnn_embedding)
        new_observation = torch.cat([obs_emb, new_observation], dim=-1)
        assert ((new_observation[..., -1] == 0) | (new_observation[..., -1] == 1)).all(), f'{new_observation[..., -1]=}'
        return new_observation
    
    @beartype
    def _convert_action_to_raw_action(self, action: list[int]) -> list[int]:
        if self.args.dataset == 'milp':
            return action
        return [int(self.active_neuron_index[a]) for a in action]
    
    @beartype
    def _convert_raw_action_to_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        if self.args.dataset == 'milp':
            return raw_action
        orig_shape = raw_action.shape
        action = [self.reverse_active_neuron_index[int(a)] for a in raw_action.flatten()]
        action = torch.tensor(action).view(orig_shape)
        return action
    
    @beartype
    def _get_fsb_action(self) -> list[int]:
        assert self.args.dataset == 'dnnv', f'{self.args.dataset=}'
        # raw action
        action = self.solver.get_fsb_action()
        # convert to active neuron index
        action = [self.reverse_active_neuron_index[int(a)] for a in action]
        return action
        
    
    @beartype
    def step(self, action: list[int]) -> tuple[torch.Tensor, int, bool, bool]:
        self.n_step += 1
        # convert to active neuron format for dnnv
        raw_action = self._convert_action_to_raw_action(action)
        
        # bound
        double_raw_actions, double_next_observations, prev_raw_actions, double_dones, double_rewards = self.solver.branch_and_bound(raw_action) # 2 x subproblems
        
        # convert to active neuron format for dnnv
        double_next_observations = self._concat_observation_embedding_3d(double_next_observations)
        double_actions = self._convert_raw_action_to_action(double_raw_actions)
        prev_actions = self._convert_raw_action_to_action(prev_raw_actions)
        
        # print(f'{np.array(double_actions).shape=} {np.array(double_next_observations).shape=} {np.array(prev_actions).shape=} {np.array(double_dones).shape=} {np.array(double_rewards).shape=}')
        assert len(double_actions) == len(double_next_observations), f'{len(double_actions)=} {len(double_next_observations)=}'
        assert len(double_actions) == len(double_dones), f'{len(double_actions)=} {len(double_dones)=}'
        assert len(double_actions) == len(double_rewards), f'{len(double_actions)=} {len(double_rewards)=}'
        
        # save for training
        self.train_action = double_actions # (n_subproblem, 2)
        self.train_next_observation = double_next_observations # (n_subproblem, 2, n_var, n_feature)
        self.train_done = double_dones # (n_subproblem, 2)
        self.train_reward = double_rewards # (n_subproblem,)
        # last actions
        self.prev_actions = prev_actions # (n_subproblem,)
        # get reward
        reward = - self.solver.remaining_subproblem
        
        # convert to active neuron format for dnnv
        remaining_next_observation = self.solver.get_observation(batch=self.args.dnnv_solver_batch) # remaining subproblems
        remaining_next_observation = self._concat_observation_embedding_2d(remaining_next_observation)
        
        # get info
        done = self.solver.remaining_subproblem <= 0
        early_stop = (
            (self.solver.remaining_subproblem >= self.args.n_max_subproblem) or 
            (self.n_step >= self.args.n_max_step)
        )
        return remaining_next_observation, reward, done, early_stop
