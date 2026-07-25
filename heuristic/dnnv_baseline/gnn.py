#! /usr/bin/env python
"""Self-contained GNN branching baseline (faithful reference GraphNet + DRL wrapper)."""
from __future__ import annotations

import copy
import os

import torch
import torch as th
from torch import nn
from torch.nn import functional as F

from ..base import Base

try:
    from dnnv_solver.heuristic.util import compute_masks
except ImportError:  # pragma: no cover
    compute_masks = None


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DEFAULT_CKPT = os.path.join(
    _REPO_ROOT,
    'dnnv_solver',
    'heuristic',
    'gnn_heuristic',
    'models',
    'cifar_trained_gnn',
    'best_snapshot_None_0_val_acc_0.826_loss_val_0.1036_epoch_57.pt',
)


def _is_flatten(layer) -> bool:
    return isinstance(layer, nn.Flatten) or type(layer).__name__ == 'Flatten'


def _is_relu(layer) -> bool:
    return isinstance(layer, nn.ReLU) or type(layer).__name__ == 'ReLU'


def _is_linear(layer) -> bool:
    return isinstance(layer, nn.Linear)


def _is_conv2d(layer) -> bool:
    return isinstance(layer, nn.Conv2d)


'''
Training test file for the deep model transferrability studies
1. added norm weight in the backward pass
2. has been tested on wide and deep networks --- satisfactory performances

Currently treated as the correct model
'''


class EmbedLayerUpdate(nn.Module):
    '''
    this class updates embeded layer one time
    '''
    def __init__(self, p, T):
        super(EmbedLayerUpdate, self).__init__()
        '''
        p: embedding size
        T: number of passes
        '''
        self.p = p
        self.T = T

        # inputs
        self.inp_f = nn.Linear(3, p)
        self.inp_f_1 = nn.Linear(p, p)

        self.inp_b = nn.Linear(2, p)
        self.inp_b_1 = nn.Linear(p, p)
        self.inp_b2 = nn.Linear(2 * p, p)
        self.inp_b2_2 = nn.Linear(p, p)

        # for bound features of the node
        self.fc1 = nn.Linear(7, p)
        self.fc1_1 = nn.Linear(p, p)

        # for sum of neighbour embeddings vectors
        self.fc3 = nn.Linear(2 * p, p)
        self.fc3_2 = nn.Linear(p, p)
        # for weights of edges connecting to the node.
        # edge updates
        self.fc4 = nn.Linear(2 * p, p)
        self.fc4_2 = nn.Linear(p, p)

        # outputs
        self.out1 = nn.Linear(4, p)
        self.out2 = nn.Linear(2 * p, p)
        self.out3 = nn.Linear(p, p)

        # backwards
        self.bc1 = nn.Linear(7, p)
        self.bc1_1 = nn.Linear(p, p)
        self.bc1_2 = nn.Linear(p, p)
        self.bc2 = nn.Linear(3 * p, p)
        self.bc2_1 = nn.Linear(p, p)
        self.bc3 = nn.Linear(2 * p, p)
        self.bc3_1 = nn.Linear(p, p)
        self.bc4 = nn.Linear(2 * p, p)
        self.bc4_1 = nn.Linear(p, p)

    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, mu):
        # NOTE: All bounds should be at the same size as the layer outputs
        #       We have assumed that the last property layer is linear

        batch_size = len(lower_bounds_all[0])
        p = self.p
        for i in range(self.T):
            ## FORWARD PASS
            # for the forward pass the first layer is not updated

            # first, deal with the input layer
            if i == 0:
                inp = th.cat([
                    lower_bounds_all[0].view(-1).unsqueeze(-1),
                    primal_inputs.view(-1).unsqueeze(-1),
                    upper_bounds_all[0].view(-1).unsqueeze(-1),
                ], 1)
                temp = self.inp_f_1(F.relu(self.inp_f(inp)))
                mu[0] = temp.reshape(mu[0].size())

            relu_count_idx = 0
            out_features = [-1] + th.tensor(lower_bounds_all[0][0].size()).tolist()

            idx = 0
            for layer_idx, layer in enumerate(layers['fixed_layers']):
                if _is_conv2d(layer):
                    # reshape
                    mu_inp = th.cat([j for j in mu[relu_count_idx]], 1)
                    mu_inp = th.t(mu_inp).reshape(out_features)
                    nb_embeddings_pre = F.conv2d(
                        mu_inp, layer.weight, bias=None,
                        stride=layer.stride, padding=layer.padding,
                        dilation=layer.dilation, groups=layer.groups,
                    )
                    # record and transfer back
                    out_features = th.tensor(nb_embeddings_pre.size()).tolist()
                    nb_embeddings_temp = nb_embeddings_pre.reshape(out_features[0], -1)
                    nb_embeddings_temp = th.cat(
                        [nb_embeddings_temp[j * p:(1 + j) * p] for j in range(batch_size)], 1
                    )
                    nb_embeddings_temp = th.t(nb_embeddings_temp)
                    pre_layer_bias = layer.bias.unsqueeze(1).expand(
                        out_features[1], out_features[2] * out_features[3]
                    )
                    pre_layer_bias = pre_layer_bias.reshape(-1)
                    pre_layer_bias = pre_layer_bias.repeat(batch_size)
                    layer_lower_pre = lower_bounds_all[idx + 1].view(-1)
                    layer_upper_pre = upper_bounds_all[idx + 1].view(-1)
                    idx += 1

                elif _is_linear(layer):
                    nb_embeddings_temp = layer.weight @ mu[relu_count_idx]
                    nb_embeddings_temp = th.cat([j for j in nb_embeddings_temp], 0)
                    pre_layer_bias = layer.bias.repeat(batch_size)
                    out_features = [-1, nb_embeddings_temp.size()[0]]
                    layer_lower_pre = lower_bounds_all[idx + 1].view(-1)
                    layer_upper_pre = upper_bounds_all[idx + 1].view(-1)
                    idx += 1

                elif _is_relu(layer):
                    # node features
                    ratio_0, ratio_1, beta, ambi_mask = compute_ratio(layer_lower_pre, layer_upper_pre)

                    # measure relaxation
                    layer_n_bounds = th.cat([
                        beta.unsqueeze(-1),
                        layer_lower_pre.unsqueeze(-1),
                        layer_upper_pre.unsqueeze(-1),
                        dual_vars[relu_count_idx][:, 1].unsqueeze(-1) - dual_vars[relu_count_idx][:, 2].unsqueeze(-1),
                        primals[layer_idx - 1].unsqueeze(-1),
                        primals[layer_idx].unsqueeze(-1),
                        pre_layer_bias.unsqueeze(-1),
                    ], 1)
                    layer_relax_s1 = self.fc1_1(F.relu(self.fc1(layer_n_bounds)))
                    layer_relax = layer_relax_s1 * ambi_mask.unsqueeze(-1)

                    # embedding vector updates
                    nb_embeddings_input = th.cat([
                        nb_embeddings_temp * ratio_0.unsqueeze(-1),
                        nb_embeddings_temp * ratio_1.unsqueeze(-1),
                    ], 1)
                    layer_nb_embeddings = self.fc3_2(F.relu(self.fc3(nb_embeddings_input)))

                    # update all nodes in a layer
                    layer_input = th.cat([layer_relax, layer_nb_embeddings], dim=1)
                    layer_mu_new = self.fc4_2(F.relu(self.fc4(layer_input)))
                    layer_mu_new = layer_mu_new * (ratio_0 != 0).float().unsqueeze(-1)
                    relu_count_idx += 1
                    mu[relu_count_idx] = layer_mu_new.reshape(mu[relu_count_idx].size())

                elif _is_flatten(layer):
                    out_features = [-1] + th.tensor(lower_bounds_all[idx + 1].size()).tolist()
                else:
                    raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

            # property layer — forward pass
            nb_embeddings_temp = [
                layers['prop_layers'][j].weight @ mu[relu_count_idx][j] for j in range(batch_size)
            ]
            pre_layer_bias = [layers['prop_layers'][j].bias for j in range(batch_size)]
            nb_embeddings_temp = th.cat(nb_embeddings_temp, 0)
            pre_layer_bias = th.cat(pre_layer_bias, 0)
            layer_lower_pre = lower_bounds_all[idx + 1].view(-1)
            layer_upper_pre = upper_bounds_all[idx + 1].view(-1)
            layer_n_bounds = th.cat([
                layer_lower_pre.unsqueeze(-1),
                layer_upper_pre.unsqueeze(-1),
                primals[-1].unsqueeze(-1),
                pre_layer_bias.unsqueeze(-1),
            ], 1)
            layer_relax_output = F.relu(self.out1(layer_n_bounds))
            layer_input = th.cat([layer_relax_output, nb_embeddings_temp], dim=1)
            layer_mu_new = self.out3(F.relu(self.out2(layer_input)))
            relu_count_idx += 1
            mu[relu_count_idx] = layer_mu_new.reshape(mu[relu_count_idx].size())

            # backward pass
            ratio = lower_bounds_all[0].new_full(lower_bounds_all[-1].size(), fill_value=1.)
            ratio = [th.t(layers['prop_layers'][j].weight) @ ratio[j] for j in range(batch_size)]
            ratio = th.stack(ratio, 0)
            nor_weight = [layers['prop_layers'][j].weight for j in range(batch_size)]
            next_layer = 'prop'

            for layer_idx, layer in reversed(list(enumerate(layers['fixed_layers']))):
                if _is_conv2d(layer):
                    ratio = F.conv_transpose2d(
                        ratio, layer.weight, stride=layer.stride, padding=layer.padding
                    )
                    nor_weight = layer.weight
                    next_layer = layer
                    idx -= 1

                elif _is_linear(layer):
                    w_temp = layer.weight
                    ratio = F.linear(ratio, th.t(w_temp))
                    nor_weight = layer.weight
                    next_layer = 'Linear'
                    idx -= 1

                elif _is_relu(layer):
                    layer_lower_pre = lower_bounds_all[idx]
                    layer_upper_pre = upper_bounds_all[idx]
                    layer_lower_pre = layer_lower_pre.view(-1)
                    layer_upper_pre = layer_upper_pre.view(-1)
                    ratio_0, ratio_1, beta, ambi_mask = compute_ratio(layer_lower_pre, layer_upper_pre)

                    # Bias
                    pre_layer_bias = layers['fixed_layers'][layer_idx - 1].bias
                    if _is_conv2d(layers['fixed_layers'][layer_idx - 1]):
                        out_features = lower_bounds_all[idx].size()
                        pre_layer_bias = pre_layer_bias.unsqueeze(1).expand(
                            out_features[1], out_features[2] * out_features[3]
                        )
                        pre_layer_bias = pre_layer_bias.reshape(-1)

                    pre_layer_bias = pre_layer_bias.repeat(batch_size)

                    # measure relaxation
                    layer_n_bounds = th.cat([
                        layer_lower_pre.unsqueeze(-1),
                        layer_upper_pre.unsqueeze(-1),
                        beta.view(-1).unsqueeze(-1),
                        -dual_vars[relu_count_idx - 2][:, 2].unsqueeze(-1)
                        + dual_vars[relu_count_idx - 2][:, 1].unsqueeze(-1),
                        primals[layer_idx].unsqueeze(-1),
                        primals[layer_idx - 1].unsqueeze(-1),
                        pre_layer_bias.unsqueeze(-1),
                    ], dim=1)

                    layer_relax_s1 = self.bc1_2(F.relu(self.bc1_1(F.relu(self.bc1(layer_n_bounds)))))
                    layer_relax_s2_input = th.cat([
                        layer_relax_s1,
                        layer_relax_s1 * (-dual_vars[relu_count_idx - 2][:, 2].unsqueeze(-1)),
                        layer_relax_s1 * (dual_vars[relu_count_idx - 2][:, 1].unsqueeze(-1)),
                    ], dim=1)
                    layer_relax = self.bc2_1(F.relu(self.bc2(layer_relax_s2_input)))
                    layer_relax = layer_relax * (ambi_mask).unsqueeze(-1)

                    # embedding vector updates
                    if _is_conv2d(next_layer):
                        out_features = [-1] + th.tensor(lower_bounds_all[idx + 1].size()).tolist()[1:]
                        mu_inp = th.cat([j for j in mu[relu_count_idx]], 1)
                        mu_inp = th.t(mu_inp).reshape(out_features)
                        nb_embeddings_pre = F.conv_transpose2d(
                            mu_inp, nor_weight, bias=None,
                            stride=next_layer.stride, padding=next_layer.padding,
                            dilation=next_layer.dilation, groups=next_layer.groups,
                        )
                        _, _, w_inp, h_inp = mu_inp.size()
                        _, _, w_wgt, h_wgt = nor_weight.size()
                        device = mu_inp.device
                        dtype = mu_inp.dtype
                        freq_inp = th.ones([1, 1, w_inp, h_inp], device=device, dtype=dtype)
                        freq_weight = th.ones([1, 1, w_wgt, h_wgt], device=device, dtype=dtype)
                        freq = F.conv_transpose2d(
                            freq_inp, freq_weight, bias=None,
                            stride=next_layer.stride, padding=next_layer.padding,
                            dilation=next_layer.dilation, groups=next_layer.groups,
                        )
                        nb_embeddings_pre = nb_embeddings_pre / freq

                        out_features = nb_embeddings_pre.size()
                        nb_embeddings = nb_embeddings_pre.reshape(out_features[0], -1)
                        nb_embeddings = th.cat(
                            [nb_embeddings[j * p:(1 + j) * p] for j in range(batch_size)], 1
                        )
                        nb_embeddings = th.t(nb_embeddings)

                    elif next_layer == 'Linear':
                        nb_embeddings = th.t(nor_weight).matmul(mu[relu_count_idx])
                        nb_embeddings = th.cat([j for j in nb_embeddings], 0)

                    elif next_layer == 'prop':
                        nb_embeddings = [
                            th.t(nor_weight[j]) @ mu[relu_count_idx][j] for j in range(batch_size)
                        ]
                        nb_embeddings = th.cat(nb_embeddings, 0)
                    else:
                        raise NotImplementedError

                    nb_embeddings_0 = nb_embeddings * ratio_0.unsqueeze(-1)
                    nb_embeddings_1 = nb_embeddings * ratio_1.unsqueeze(-1)

                    layer_embeddings_input = th.cat([nb_embeddings_0, nb_embeddings_1], dim=1)
                    layer_nb_embeddings = self.bc3_1(F.relu(self.bc3(layer_embeddings_input)))

                    # update all nodes in a layer
                    layer_input = th.cat([layer_relax, layer_nb_embeddings], dim=1)
                    layer_mu_new = self.bc4_1(F.relu(self.bc4(layer_input)))
                    layer_mu_new = layer_mu_new * ((ratio_0 != 0).view(-1)).float().unsqueeze(-1)

                    mu[relu_count_idx - 1] = layer_mu_new.reshape(mu[relu_count_idx - 1].size())
                    relu_count_idx -= 1

                elif _is_flatten(layer):
                    ratio = ratio.reshape(lower_bounds_all[idx].size())
                else:
                    raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

                if layer_idx == 0:
                    if _is_conv2d(next_layer):
                        out_features = [-1] + th.tensor(lower_bounds_all[idx + 1].size()).tolist()[1:]
                        mu_inp = th.cat([j for j in mu[relu_count_idx]], 1)
                        mu_inp = th.t(mu_inp).reshape(out_features)
                        nb_embeddings_pre = F.conv_transpose2d(
                            mu_inp, nor_weight, bias=None,
                            stride=next_layer.stride, padding=next_layer.padding,
                            dilation=next_layer.dilation, groups=next_layer.groups,
                        )
                        out_features = nb_embeddings_pre.size()
                        nb_embeddings = nb_embeddings_pre.reshape(out_features[0], -1)
                        nb_embeddings = th.cat(
                            [nb_embeddings[j * p:(1 + j) * p] for j in range(batch_size)], 1
                        )
                        nb_embeddings = th.t(nb_embeddings)

                    elif next_layer == 'Linear':
                        nb_embeddings = th.t(nor_weight).matmul(mu[relu_count_idx])
                        nb_embeddings = th.cat([j for j in nb_embeddings], 0)
                    else:
                        raise NotImplementedError

                    inp = th.cat([
                        lower_bounds_all[0].view(-1).unsqueeze(-1),
                        upper_bounds_all[0].view(-1).unsqueeze(-1),
                    ], 1)
                    inp_relax = self.inp_b_1(F.relu(self.inp_b(inp)))
                    domain_input = th.cat([inp_relax, nb_embeddings], dim=1)
                    domain_mu_new = self.inp_b2_2(F.relu(self.inp_b2(domain_input)))
                    mu[relu_count_idx - 1] = domain_mu_new.reshape(mu[relu_count_idx - 1].size())

        return mu


class EmbedUpdates(nn.Module):
    '''
    this class updates embeding vectors from t=1 and t=T
    '''

    def __init__(self, T, p):
        '''
        p_list contains the input and output dimensions of embedding vectors for all layers
        len(p_list) = T+1
        '''
        super(EmbedUpdates, self).__init__()
        self.T = T
        self.p = p
        self.update = EmbedLayerUpdate(p, T)

    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers):
        mu = init_mu(lower_bounds_all, self.p)
        mu = self.update(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, mu)
        return mu


class ComputeFinalScore(nn.Module):
    '''
    this class computes a final score for each node

    p: the dimension of embedding vectors at the final stage
    '''
    def __init__(self, p):
        super(ComputeFinalScore, self).__init__()
        self.p = p
        self.fnode = nn.Linear(p, p)
        self.fscore = nn.Linear(p, 1)

    def utils(self, mu):
        scores = []
        for layer in mu[1:-1]:
            scores_current_layer = mu[0].new_full((layer.size()[0],), fill_value=0.0)
            scores.append(scores_current_layer)
        return scores

    def forward(self, mu, masks):
        scores = []
        for batch_idx in range(len(mu[0])):
            mu_temp = th.cat([j[batch_idx] for j in mu[1:-1]], dim=0)
            mu_temp = mu_temp[masks[batch_idx].nonzero().view(-1)]
            temp = self.fnode(mu_temp)
            score = self.fscore(F.relu(temp))
            scores.append(score.view(-1))
        return scores


class GraphNet(nn.Module):
    def __init__(self, T, p):
        super(GraphNet, self).__init__()
        self.EmbedUpdates = EmbedUpdates(T, p)
        self.ComputeFinalScore = ComputeFinalScore(p)

    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, masks):
        mu = self.EmbedUpdates(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers)
        scores = self.ComputeFinalScore(mu, masks)
        return scores


def init_mu(lower_bounds_all, p):
    mu = []
    batch_size = len(lower_bounds_all[0])
    for i in lower_bounds_all:
        required_size = i[0].view(-1).size()
        mus_current_layer = lower_bounds_all[0].new_full(
            (batch_size, required_size[0], p), fill_value=0.0
        )
        mu.append(mus_current_layer)
    return mu


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    slope_ratio0 = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio0
    ambi_mask = (intercept > 0).float()
    slope_ratio1 = (1 - 2 * (slope_ratio0 * ambi_mask)) * ambi_mask + slope_ratio0
    return slope_ratio0, slope_ratio1, intercept, ambi_mask


# ---------------------------------------------------------------------------
# Feature construction + DRL baseline wrapper
# ---------------------------------------------------------------------------


def _build_layers(pytorch_model, cs_b: torch.Tensor, device: torch.device):
    """fixed_layers = all modules except last Linear; prop_layers from cs @ last Linear."""
    children = list(pytorch_model.children())
    if not children:
        raise RuntimeError('pytorch_model has no children')
    last = children[-1]
    if not _is_linear(last):
        raise NotImplementedError(f'Expected last module to be Linear, got {type(last)}')

    fixed_layers = [copy.deepcopy(m).to(device) for m in children[:-1]]
    for m in fixed_layers:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    W = last.weight.detach().to(device=device)  # (n_out, n_hid)
    bias = last.bias.detach().to(device=device)  # (n_out,)
    cs_b = cs_b.to(device=device, dtype=W.dtype)
    # cs_b: (S, n_out)
    W_prop = cs_b @ W  # (S, n_hid)
    b_prop = (cs_b @ bias.unsqueeze(-1)).squeeze(-1)  # (S,)
    n_hid = W_prop.shape[1]
    n_spec = W_prop.shape[0]
    prop = nn.Linear(n_hid, n_spec, bias=True).to(device)
    with torch.no_grad():
        prop.weight.copy_(W_prop)
        prop.bias.copy_(b_prop)
    prop.eval()
    for p in prop.parameters():
        p.requires_grad_(False)

    return {'fixed_layers': fixed_layers, 'prop_layers': [prop]}


def _build_bounds(domain_params, split_node_names, b: int, device: torch.device):
    """[input, split_0, ..., split_k, output]; output upper filled with 1e3."""
    lower_bounds_all = [domain_params.input_lowers[b:b + 1].to(device)]
    upper_bounds_all = [domain_params.input_uppers[b:b + 1].to(device)]
    for name in split_node_names:
        lower_bounds_all.append(domain_params.lower_bounds[name][b:b + 1].to(device))
        upper_bounds_all.append(domain_params.upper_bounds[name][b:b + 1].to(device))
    out_lb = domain_params.output_lbs[b:b + 1].to(device)
    out_ub = torch.full_like(out_lb, 1e3)
    lower_bounds_all.append(out_lb)
    upper_bounds_all.append(out_ub)
    return lower_bounds_all, upper_bounds_all


def _build_dual_vars(abstractor, domain_params, split_node_names, b: int, device: torch.device, dtype):
    """CROWN lAs → [N, 3] dual channels per ReLU (batch=1)."""
    dual_vars = []
    for name in split_node_names:
        relu_node = abstractor.net.split_activations[name][0][0]
        la = domain_params.lAs[relu_node.name][b:b + 1]  # (1, S, ...)
        # mean over specs → (N,)
        if la.ndim >= 2:
            la = la.mean(dim=1)
        la = la.reshape(-1).to(device=device, dtype=dtype)
        dual = torch.zeros(la.numel(), 3, device=device, dtype=dtype)
        dual[:, 1] = la.clamp(min=0)
        dual[:, 2] = (-la).clamp(min=0)
        dual_vars.append(dual)
    return dual_vars


def _build_primals(layers, primal_input: torch.Tensor):
    """Concrete forward at midpoint; flattened activations after each fixed + prop module."""
    primals = []
    x = primal_input
    with torch.no_grad():
        for layer in layers['fixed_layers']:
            x = layer(x)
            primals.append(x.reshape(-1).detach())
        x = layers['prop_layers'][0](x)
        primals.append(x.reshape(-1).detach())
    return primals


def _build_masks(domain_params, split_node_names, b: int, device: torch.device, dtype):
    """1-D float masks per ReLU: 1.0 unstable, 0.0 stable."""
    masks = []
    if domain_params.masks is not None:
        for name in split_node_names:
            m = domain_params.masks[name][b].reshape(-1).to(device=device, dtype=dtype)
            masks.append(m)
        return masks

    # fallback: (lb < 0) & (ub > 0)
    for name in split_node_names:
        lb = domain_params.lower_bounds[name][b]
        ub = domain_params.upper_bounds[name][b]
        m = ((lb < 0) & (ub > 0)).reshape(-1).to(device=device, dtype=dtype)
        masks.append(m)
    return masks


class GnnDecider(Base):

    def __init__(self, args):
        super().__init__(args)
        self.checkpoint_path = getattr(args, 'gnn_checkpoint', None) or _DEFAULT_CKPT
        self._device = torch.device(getattr(args, 'device', 'cpu'))
        self.model = None  # lazy-load on first select_action

    def _ensure_model(self, device: torch.device):
        if self.model is None:
            model = GraphNet(2, 64)
            state = torch.load(self.checkpoint_path, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
                if isinstance(state, list):
                    state = state[0]
            model.load_state_dict(state)
            self.model = model
        self.model.to(device)
        self.model.eval()
        self._device = device

    def _decide_one(self, abstractor, domain_params, b: int) -> int:
        device = self._device
        dtype = domain_params.input_lowers.dtype
        split_node_names = [n.name for n in abstractor.net.split_nodes]
        split_node_points = {
            k: abstractor.net.split_activations[k][0][0].get_split_point()
            for k in split_node_names
        }

        layers = _build_layers(abstractor.pytorch_model, domain_params.cs[b], device)
        lower_bounds_all, upper_bounds_all = _build_bounds(
            domain_params, split_node_names, b, device
        )
        dual_vars = _build_dual_vars(
            abstractor, domain_params, split_node_names, b, device, dtype
        )
        primal_input = 0.5 * (lower_bounds_all[0] + upper_bounds_all[0])
        primals = _build_primals(layers, primal_input)
        masks = _build_masks(domain_params, split_node_names, b, device, dtype)
        mask_1d = torch.cat(masks, dim=0).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(
                lower_bounds_all,
                upper_bounds_all,
                dual_vars,
                primals,
                primal_input,
                layers,
                mask_1d,
            )[0]

        # GraphChoice-style mapping: argmax over unstable-only scores → flat index → (layer, idx)
        if scores.numel() == 0:
            for lname in split_node_names:
                candidates = (masks[split_node_names.index(lname)] != 0).nonzero(as_tuple=False)
                if len(candidates):
                    dec_lay = split_node_names.index(lname)
                    dec_idx = int(candidates[0].item())
                    break
            else:
                raise RuntimeError('No unstable neuron available for GNN decision')
        else:
            choice = int(scores.argmax().item())
            flat_idx = int(mask_1d[0].nonzero()[choice].item())
            sizes = [int(m.numel()) for m in masks]
            trans_len = torch.cumsum(torch.tensor(sizes, device=device), dim=0)
            dec_lay = int((trans_len > flat_idx).nonzero()[0].item())
            if dec_lay == 0:
                dec_idx = flat_idx
            else:
                dec_idx = int(flat_idx - int(trans_len[dec_lay - 1].item()))

        lname = split_node_names[dec_lay]
        point = split_node_points[lname]
        decision = [lname, int(dec_idx), point]

        solver = self.env.solver
        raw = solver.reverse_neuron_index_mapping[*decision]
        action_i = self.env.reverse_active_neuron_index[int(raw)]
        return int(action_i)

    def select_action(self, observation):
        # action in active neuron index format
        solver = self.env.solver
        device = torch.device(solver.device)
        self._ensure_model(device)

        batch = getattr(self.args, 'dnnv_solver_batch', None)
        domain_params = solver.queue.get_all_domains(device=solver.device, batch=batch)

        # Prefer compute_masks if queue did not attach masks (should already be present)
        if domain_params.masks is None and compute_masks is not None:
            masks = compute_masks(
                lower_bounds=domain_params.lower_bounds,
                upper_bounds=domain_params.upper_bounds,
                device=str(device),
                split_activations=solver.queue.split_activations,
            )
            domain_params = domain_params._replace(masks=masks)

        n = len(domain_params.input_lowers)
        action = [self._decide_one(solver.abstractor, domain_params, b) for b in range(n)]

        assert len(action) == len(observation), f'{len(action)=} {[o.shape for o in observation]=}'
        assert all(0 <= a <= observation.shape[-2] for a in action), f'{observation.shape=} {action=}'
        return action

    def __repr__(self):
        return f'GnnDecider(mode={self.args.mode}, dataset={self.args.dataset})'
