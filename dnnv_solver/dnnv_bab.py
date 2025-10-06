from beartype import beartype
import numpy as np
import torch
import copy

from .abstractor.auto_LiRPA.utils import stop_criterion_batch_any
from .abstractor.utils import new_slopes

from .helper.misc.result import AbstractResults

from .heuristic.util import compute_masks, _compute_babsr_scores
from .heuristic.decision_heuristics import DecisionHeuristic
from .heuristic.domains_list import DomainsList


class DnnvBabSolver:
    
    def __init__(self, net, input_shape, objective, args):
        self.args = args
        self.net = net.to(args.device)
        self.input_shape = input_shape
        self.objective = objective
        self.device = args.device
    
        # hyper parameters
        self.input_split = False
        self.batch = 100
        
        assert len(objective.lower_bounds) == 1, f'{len(objective.lower_bounds)=}'
        self.queue = self.initialize(objective)
        
        self.fsb_heuristic = DecisionHeuristic(
            input_split=self.input_split, 
            decision_method='smart',
            decision_topk=30 if args.mode=='train' else 10, 
            use_random_topk=args.mode=='train',
        )        
    
    @property
    def remaining_subproblem(self):
        return len(self.queue)
    
    
    def initialize(self, objective) -> DomainsList | list:
        # check attacked
        if self.args.use_attack:
            is_attacked, _ = self._pre_attack(copy.deepcopy(objective), timeout=self.args.attack_timeout)
            if is_attacked:
                print(f'    - filtered: {is_attacked=}')
                return []
        
        # init abstractor
        self._setup_restart(0, objective)
        
        # init bounds
        ret = self.abstractor.initialize(objective, reference_bounds=None)
        assert len(ret.output_lbs) == len(objective.cs)
        
        # too hard problems
        if ret.output_lbs.min() < self.args.dnnv_skip_lower_bound:
            print(f'    - filtered: {ret.output_lbs=}')
            return []
        
        # check verified
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            print(f'    - filtered: {ret.output_lbs=}')
            return []
        # print(f'    + saved: lower_bound={ret.output_lbs.item()}')
        
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
            slopes=slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories),
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions={},
        )

    @beartype
    def branch_and_bound(self, action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # convert DRL action format to DNNV decision format
        decisions = [list(self.neuron_index_mapping[a]) for a in action]
        # pick all subproblems
        pick_ret = self.queue.pick_out(len(action), device=self.device)
        # bounding
        abstraction_ret = self.abstractor.forward(decisions, pick_ret)
        # update queue
        self.queue.add(abstraction_ret, decisions)
        
        # NOTE: this is NOT in a correct order used in training
        # training information
        double_actions = torch.tensor(action * 2)
        double_next_observations = self.get_node_data(abstraction_ret)
        
        # unverified indices
        remaining_index = torch.where((abstraction_ret.output_lbs.detach().cpu() <= abstraction_ret.rhs.detach().cpu()).all(1))[0]
        double_dones = torch.ones_like(double_actions, dtype=torch.int32)
        double_dones[remaining_index] = 0
        double_rewards = -1 * sum(torch.chunk(1 - double_dones, 2))
        # previous actions
        prev_actions = double_actions[remaining_index]
        
        # NOTE: correct order used in training
        return self._reorder_tensors(double_actions, double_next_observations, prev_actions, double_dones, double_rewards)
    
    @beartype
    def _reorder_tensors(self, actions: torch.Tensor, next_observations: torch.Tensor, prev_actions: torch.Tensor, dones: torch.Tensor, rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Convert [a1, b1, c1, a2, b2, c2] to [[a1, a2], [b1, b2], [c1, c2]]
        '''
        new_actions = torch.stack(torch.chunk(actions, 2), dim=1).cpu()
        new_next_observations = torch.stack(torch.chunk(next_observations, 2), dim=1).cpu()
        new_dones = torch.stack(torch.chunk(dones, 2), dim=1).cpu()
        # aligned with remaining subproblems, no need to reorder
        new_rewards = rewards.cpu()
        new_prev_actions = prev_actions.cpu()
        return new_actions, new_next_observations, new_prev_actions, new_dones, new_rewards
        
        
    @beartype
    def get_node_data(self, domain_params: AbstractResults) -> torch.Tensor:
        assert len(domain_params.input_lowers) == len(domain_params.input_uppers)
        # assert all([len(v) == 1 for v in domain_params.lower_bounds.values()])
        # assert all([len(v) == 1 for v in domain_params.upper_bounds.values()])
        
        ret = []
        # 1. input features
        lb = domain_params.input_lowers.flatten(1).cpu()
        ub = domain_params.input_uppers.flatten(1).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.zeros_like(lb).cpu() # 1 - not selected, 0 - selected
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
            assert (lb[torch.where(mask == 1)] < 0).all()
            assert (ub[torch.where(mask == 1)] > 0).all()
            x = torch.stack([
                lb.flatten(1),
                ub.flatten(1),
                bias.flatten(1),
                mask.flatten(1), # 1 - not selected, 0 - selected
            ], dim=-1)
            ret.append(x)
            # print(f'hidden: {name=} {x.shape=}')
        
        # 3. output features
        lb = domain_params.output_lbs.flatten(1).cpu()
        ub = torch.zeros_like(lb).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.zeros_like(lb).cpu() # 1 - not selected, 0 - selected
        x = torch.stack([lb, ub, bias, mask], dim=-1)
        # print(f'output: {name=} {x.shape=}')
        ret.append(x)
        return torch.cat(ret, dim=1)
    
    
    def get_observation(self, return_fsb_score=False, batch=None):
        """remaining subproblems"""
        subproblems = self.queue.get_all_domains(device=self.device, batch=batch)
        if return_fsb_score:
            return self.get_node_data(subproblems), self.get_fsb_score(subproblems)
        return self.get_node_data(subproblems)
        
        
    def get_info(self):
        return None
    
    def get_fsb_score(self, domain_params: AbstractResults) -> torch.Tensor:
    
        masks = compute_masks(
            lower_bounds=domain_params.lower_bounds,
            upper_bounds=domain_params.upper_bounds,
            device=self.device,
            non_blocking=False,
        )

        # features
        batch = len(domain_params.input_lowers)
        scores_1, scores_2 = _compute_babsr_scores(
            abstractor=self.abstractor,
            lower_bounds=domain_params.lower_bounds,
            upper_bounds=domain_params.upper_bounds,
            lAs=domain_params.lAs,
            batch=batch,
            masks=masks,
            reduce_op=self.fsb_heuristic.decision_reduceop,
            number_bounds=domain_params.cs.shape[1]
        )

        dummy_input = torch.zeros_like(domain_params.input_lowers, device=self.device).flatten(1)
        dummy_output = torch.zeros_like(domain_params.output_lbs, device=self.device).flatten(1)
        score_1 = torch.cat([dummy_input] + scores_1 + [dummy_output], dim=-1)
        score_2 = torch.cat([dummy_input] + scores_2 + [dummy_output], dim=-1)
        score = torch.stack([score_1, score_2], dim=-1)
        return score
    
    def get_fsb_action(self) -> torch.Tensor:
        subproblems = self.queue.get_all_domains(device=self.device, batch=self.args.dnnv_solver_batch)
        decisions = self.fsb_heuristic(self.abstractor, subproblems)
        # convert DNNV decision format to DRL action format
        action = [self.reverse_neuron_index_mapping[(d[0], d[1], d[2])] for d in decisions]
        return action
    
    def _compute_neuron_index_mapping(self, domain_params: AbstractResults) -> None:
        self.neuron_index_mapping = {}
        ordered_names = [_.name for _ in self.abstractor.net.split_nodes]
        count = np.prod(self.input_shape)
        for name in ordered_names:
            assert domain_params.lower_bounds[name].shape[0] == 1
            n_hiddens = domain_params.lower_bounds[name].numel()
            self.neuron_index_mapping.update(
                {int(count + i): (name, i, 0.0) for i in range(n_hiddens)}
            )
            count += n_hiddens
        
        self.reverse_neuron_index_mapping = {v: k for k, v in self.neuron_index_mapping.items()}

    from .verifier.utils import _setup_restart, _init_abstractor, _pre_attack
    from .graph.onnx2networkx import get_edge_data