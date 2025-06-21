from neuralsat.heuristic.util import compute_masks
import itertools as it
import neuralsat as ns
import networkx as nx
import numpy as np
import torch

def debug(args):
    #
    onnx_path = '../data/benchmark/cifar/onnx/0.onnx'
    vnnlib_path = '../data/benchmark/cifar/vnnlib/0.vnnlib'
    net, input_shape, objs = ns.test.extract_instance(onnx_path, vnnlib_path)
    iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
    obj = objs.pop(4)
    obj = objs.pop(1)
    obj_id = 4

    #######################
    # EXTRACT NODE FEATURES
    #######################
    # 1. Input Features
    # -----------------
    X = []
    sample = iv.get_sample_observation(obj)
    lb = sample.input_lowers.cpu()
    ub = sample.input_uppers.cpu()
    bias = torch.zeros_like(lb).cpu()
    mask = torch.zeros_like(lb).cpu()
    x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
    X.append(x)
    print(f'input: {x.shape=}')

    # 2. Hidden Features
    # ------------------
    # ordered name of layers
    ordered_names = [_.name for _ in iv.abstractor.net.split_nodes]
    # extract mask
    masks = compute_masks(sample.lower_bounds, sample.upper_bounds, device='cpu')
    # extract hidden
    for i, name in enumerate(ordered_names):
        lb = sample.lower_bounds[name].cpu()
        ub = sample.upper_bounds[name].cpu()
        bias = iv.abstractor.net[name].inputs[-1].param.detach().cpu()
        if lb.ndim == 4: # conv layer
            bs, c, h, w = lb.shape
            bias = bias[None, : , None, None]
            bias = bias.repeat(bs, 1, h, w)
        elif lb.ndim == 2: # fc layer
            bias = bias[None]
        else:
            raise NotImplementedError
        mask = masks[name].view(lb.shape)
        assert (lb[torch.where(mask == 1)] < 0).all()
        assert (ub[torch.where(mask == 1)] > 0).all()
        x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
        X.append(x)
        print(f'hidden: {name=} {x.shape=}')
        # print(f'{i=} {name=} {lb.shape=} {ub.shape=} {bias.shape=} {mask.shape=}')

    # 3. Output Features
    # ------------------
    lb = sample.output_lbs.cpu()
    ub = torch.zeros_like(lb).cpu()
    bias = iv.abstractor.net.final_node().inputs[-1].param.detach().cpu()[obj_id][None, None]
    mask = torch.zeros_like(lb).cpu()
    x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
    X.append(x)
    print(f'output: {name=} {x.shape=}')

    i = 0
    graphs = []
    pre_activation = torch.rand(input_shape)
    for layer in list(net.modules())[1:]:
        # infer post activation
        post_activation = layer(pre_activation)
        # update input/output shape
        input_shape = pre_activation.shape
        output_shape = post_activation.shape
        # create graph map layer i to layer i+1
        G = nx.DiGraph()
        # add nodes
        _, in_h, in_w, _ = X[i].shape
        _, out_h, out_w, _ = X[i].shape
        out_channel, in_channel, filter_h, filter_w = layer.weight.shape
        for c, h, w in it.product(range(in_channel), range(in_h), range(in_w)):
            G.add_node('in_{c}_{h}_{w}')
        for c, h, w in it.product(range(out_channel), range(out_h), range(out_w)):
            G.add_node('out_{c}_{h}_{w}')
        # add edge
        weight = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        # compute row mapping: from current row to input rows
        in_row_idx_mins = np.arange(output_shape[2]) * layer.stride[0] - layer.padding[0]
        in_row_idx_maxs = in_row_idx_mins + weight.shape[2] - 1
        #
        ker_row_mins = np.zeros(output_shape[2], dtype=int)
        ker_row_maxs = np.ones(output_shape[2], dtype=int) * weight.shape[2]
        ker_row_mins[in_row_idx_mins < 0] = -in_row_idx_mins[in_row_idx_mins < 0]
        ker_row_maxs[in_row_idx_maxs >= input_shape[2]] = ker_row_maxs[in_row_idx_maxs >= input_shape[2]] - in_row_idx_maxs[in_row_idx_maxs >= input_shape[2]] + input_shape[2] - 1
        in_row_idx_mins = np.maximum(in_row_idx_mins, 0)
        in_row_idx_maxs = np.minimum(in_row_idx_maxs, input_shape[2] - 1)
        # compute column mapping: from current column to input columns
        in_col_idx_mins = np.arange(output_shape[3]) * layer.stride[1] - layer.padding[1]
        in_col_idx_maxs = in_col_idx_mins + weight.shape[3] - 1
        ker_col_mins = np.zeros(output_shape[3], dtype=int)
        ker_col_maxs = np.ones(output_shape[3], dtype=int) * weight.shape[3]
        ker_col_mins[in_col_idx_mins < 0] = -in_col_idx_mins[in_col_idx_mins < 0]
        ker_col_maxs[in_col_idx_maxs >= input_shape[3]] = ker_col_maxs[in_col_idx_maxs >= input_shape[3]] - in_col_idx_maxs[in_col_idx_maxs >= input_shape[3]] + input_shape[3] - 1
        in_col_idx_mins = np.maximum(in_col_idx_mins, 0)
        in_col_idx_maxs = np.minimum(in_col_idx_maxs, input_shape[3] - 1)
        exit()

        node_labels = []
        for out_chan_idx in range(output_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(output_shape[2]):
                out_row_vars = []
                # get row index range from precomputed arrays
                ker_row_min, ker_row_max = ker_row_mins[out_row_idx], ker_row_maxs[out_row_idx]
                in_row_idx_min, in_row_idx_max = in_row_idx_mins[out_row_idx], in_row_idx_maxs[out_row_idx]
                for out_col_idx in range(output_shape[3]):
                    # get col index range from precomputed arrays
                    ker_col_min, ker_col_max = ker_col_mins[out_col_idx], ker_col_maxs[out_col_idx]
                    in_col_idx_min, in_col_idx_max = in_col_idx_mins[out_col_idx], in_col_idx_maxs[out_col_idx]
                    node_label = f'conv2d_{layer_id}_{node_id}'
                    node_to_id[node_label] = node_id + offset_node_id
                    nx_graph.add_node(node_to_id[node_label], layer=layer_id)
                    # init linear constraint LHS implied by the conv operation
                    for in_chan_idx in range(weight.shape[1]):
                        coeffs = weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)
                        # print(f'{coeffs=}')
                        nodes = nodes_array[0][in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                        # print(f'{nodes=}')
                        assert len(coeffs) == len(nodes)
                        for c_, n_ in zip(coeffs, nodes):
                            nx_graph.add_edge(node_to_id[node_label], node_to_id[n_], label=c_)
                            nx_graph.add_edge(node_to_id[n_], node_to_id[node_label], label=c_)
                    c_ = bias[out_chan_idx]
                    nx_graph.add_edge(node_to_id[node_label], node_to_id[node_label], label=c_)

                    out_row_vars.append(node_label)
                    node_id += 1
                out_chan_vars.append(out_row_vars)
            node_labels.append(out_chan_vars)



        print(X[i].shape, X[i+1].shape, layer.weight.shape)
        # create node from layer i + 1
        # create edge with weight:
        exit()
        if isinstance(layer, torch.nn.Conv2d):
            # map X[i] -(layer.weight)-> X[i+1]
            i += 1
