from beartype import beartype
import torch
import copy

from .abstractor.auto_LiRPA.utils import stop_criterion_batch_any
from .abstractor.utils import new_slopes

from .helper.misc.result import AbstractResults

from .heuristic.domains_list import DomainsList
from .heuristic.util import compute_masks

from .setting import Settings


class DnnvBabSolver:
    
    def __init__(self, net, input_shape, objective, args):
        self.args = args
        self.net = net
        self.input_shape = input_shape
        self.objective = objective
    
        # hyper parameters
        self.input_split = False
        self.batch = 100
        
        assert len(objective.lower_bounds) == 1, f'{len(objective.lower_bounds)=}'
        self.queue = self.initialize(objective)
    
    
    def remaining_subproblem(self):
        return len(self.queue)
    
    
    def initialize(self, objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        # init abstractor
        self._setup_restart(0, objective)
        
        # init bounds
        ret = self.abstractor.initialize(objective, reference_bounds=reference_bounds)
        
        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            return []
        
        # compute neuron index mapping
        self._compute_neuron_index_mapping(ret)
        
        # full slopes uses too much memory
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)

        # remaining domains
        return DomainsList(
            net=self.abstractor.net,
            objective_ids=ret.objective_ids,
            output_lbs=ret.output_lbs,
            input_lowers=ret.input_lowers,
            input_uppers=ret.input_uppers,
            lower_bounds=ret.lower_bounds,
            upper_bounds=ret.upper_bounds,
            lAs=ret.lAs,
            slopes=ret.slopes if Settings.update_interm_bounds else slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories),
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )

    def branch_and_bound(self, action):
        # pick all subproblems
        assert self.remaining_subproblem == len(action), f'{self.remaining_subproblem=} {len(action)=}'
        pick_ret = self.queue.pick_out(len(self.queue))
        
        # bounding
        abstraction_ret = self.abstractor.forward(action, pick_ret)
        
        # update queue
        self.queue.add(abstraction_ret, action)
        
        # training information
        double_actions = action * 2
        double_next_observations = self.get_node_data(abstraction_ret)
        
        # unverified indices
        remaining_index = torch.where((abstraction_ret.output_lbs.detach().cpu() <= abstraction_ret.rhs.detach().cpu()).all(1))[0]
        double_dones = torch.ones([2 * len(double_actions)], dtype=torch.int32)
        double_dones[remaining_index] = 0
        double_dones = double_dones.tolist()
        
        double_rewards = sum(torch.chunk(1 - double_dones, 2)) * (-1 / 125) # TODO: magic number
        double_rewards = double_rewards.tolist()
        
        # previous actions
        prev_actions = torch.tensor(double_actions)[remaining_index].tolist()
        
        return double_actions, double_next_observations, prev_actions, double_dones, double_rewards
        
        
    @beartype
    def get_node_data(self, domain_params: AbstractResults) -> list[torch.Tensor]:
        assert len(domain_params.input_lowers) == len(domain_params.input_uppers)
        # assert all([len(v) == 1 for v in domain_params.lower_bounds.values()])
        # assert all([len(v) == 1 for v in domain_params.upper_bounds.values()])
        
        ret = []
        # 1. input features
        lb = domain_params.input_lowers.flatten(1).cpu()
        ub = domain_params.input_uppers.flatten(1).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.zeros_like(lb).cpu() # convert to heursitic form: 1 - not selected, 0 - selected
        x = torch.stack([lb, ub, bias, mask], dim=-1)
        # print(f'input: {x.shape=}')
        ret.append(x)
        
        # 2. hidden features
        ordered_names = [_.name for _ in self.abstractor.net.split_nodes]
        # extract mask
        masks = compute_masks(domain_params.lower_bounds, domain_params.upper_bounds, device='cpu')
        # extract hidden
        for name in ordered_names:
            lb = domain_params.lower_bounds[name].cpu()
            ub = domain_params.upper_bounds[name].cpu()
            bias = self.abstractor.net[name].inputs[-1].param.detach().cpu()
            if lb.ndim == 4: # conv layer
                bs, _, h, w = lb.shape
                bias = bias[None, : , None, None]
                bias = bias.repeat(bs, 1, h, w)
            elif lb.ndim == 2: # fc layer
                bs, _ = lb.shape
                bias = bias[None].repeat(bs, 1)
            else:
                raise NotImplementedError
            mask = masks[name].view(lb.shape) 
            assert (lb[torch.where(mask == 0)] < 0).all()
            assert (ub[torch.where(mask == 0)] > 0).all()
            x = torch.stack([
                lb.flatten(1),
                ub.flatten(1),
                bias.flatten(1),
                mask.flatten(1), # convert to heursitic form: 1 - not selected, 0 - selected
            ], dim=-1)
            ret.append(x)
            # print(f'hidden: {name=} {x.shape=}')
        
        # 3. output features
        lb = domain_params.output_lbs.flatten(1).cpu()
        ub = torch.zeros_like(lb).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.zeros_like(lb).cpu() # convert to heursitic form: 1 - not selected, 0 - selected
        x = torch.stack([lb, ub, bias, mask], dim=-1)
        # print(f'output: {name=} {x.shape=}')
        ret.append(x)
        return ret
    
    def get_observation(self):
        """remaining subproblems"""
        subproblems = self.queue.pick_out_worst_domains(len(self.queue))
        return self.get_node_data(subproblems)
        
        
    def get_info(self):
        return None
    
    from .verifier.utils import _setup_restart, _init_abstractor