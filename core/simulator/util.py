import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import torch
import os

def visualize_graph(G, args):
    # Draw the graph with labels
    nx.draw(G, pos=nx.spring_layout(G, seed=42),
               with_labels=True,
               labels=nx.get_node_attributes(G, 'label'))
    # save graph visualization
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(path)

def _prepare_linear(nx_graph, pre_nodes, layer, layer_id, input_shape, output_shape, offset_node_id):
    nodes_array = np.array(pre_nodes)
    assert nodes_array.shape == input_shape

    this_layer_weight = layer.weight.detach().cpu().numpy()
    # TODO: use bias
    # this_layer_bias = layer.bias.detach().cpu().numpy()
    
    node_to_id = {}
    node_labels = []
    for node_id in range(len(this_layer_weight)):
        # bias = this_layer_bias[node_id].item()
        coeffs = this_layer_weight[node_id]
        nodes = nodes_array[0]

        node_label = f'linear_{layer_id}_{node_id}'
        assert len(coeffs) == len(nodes)
        for n_, c_ in zip(coeffs, nodes):
            nx_graph.add_edge(node_label, n_, label=c_)
            nx_graph.add_edge(n_, node_label, label=c_)
            
        node_to_id[node_label] = node_id + offset_node_id
        node_labels.append(node_label)
        
    node_labels = np.array(node_labels)
    node_labels = node_labels.reshape(output_shape)
    return node_labels, node_to_id

def _prepare_conv2d(nx_graph, pre_nodes, layer, layer_id, input_shape, output_shape, offset_node_id):
    assert layer.dilation == (1, 1)
    assert len(layer.padding) == 2
    
    nodes_array = np.array(pre_nodes)
    assert nodes_array.shape == input_shape
    
    this_layer_weight = layer.weight.detach().cpu().numpy()
    # TODO: use bias
    # this_layer_bias = layer.bias.detach().cpu().numpy()

    # compute row mapping: from current row to input rows
    in_row_idx_mins = np.arange(output_shape[2]) * layer.stride[0] - layer.padding[0]
    in_row_idx_maxs = in_row_idx_mins + this_layer_weight.shape[2] - 1
    ker_row_mins = np.zeros(output_shape[2], dtype=int)
    ker_row_maxs = np.ones(output_shape[2], dtype=int) * this_layer_weight.shape[2]
    ker_row_mins[in_row_idx_mins < 0] = -in_row_idx_mins[in_row_idx_mins < 0]
    ker_row_maxs[in_row_idx_maxs >= input_shape[2]] = ker_row_maxs[in_row_idx_maxs >= input_shape[2]] - in_row_idx_maxs[in_row_idx_maxs >= input_shape[2]] + input_shape[2] - 1
    in_row_idx_mins = np.maximum(in_row_idx_mins, 0)
    in_row_idx_maxs = np.minimum(in_row_idx_maxs, input_shape[2] - 1)
        
    # compute column mapping: from current column to input columns
    in_col_idx_mins = np.arange(output_shape[3]) * layer.stride[1] - layer.padding[1]
    in_col_idx_maxs = in_col_idx_mins + this_layer_weight.shape[3] - 1
    ker_col_mins = np.zeros(output_shape[3], dtype=int)
    ker_col_maxs = np.ones(output_shape[3], dtype=int) * this_layer_weight.shape[3]
    ker_col_mins[in_col_idx_mins < 0] = -in_col_idx_mins[in_col_idx_mins < 0]
    ker_col_maxs[in_col_idx_maxs >= input_shape[3]] = ker_col_maxs[in_col_idx_maxs >= input_shape[3]] - in_col_idx_maxs[in_col_idx_maxs >= input_shape[3]] + input_shape[3] - 1
    in_col_idx_mins = np.maximum(in_col_idx_mins, 0)
    in_col_idx_maxs = np.minimum(in_col_idx_maxs, input_shape[3] - 1)
    
    node_to_id = {}
    node_id = 0
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
                
                # TODO: use bias
                # this_neuron_bias = this_layer_bias[out_chan_idx] if this_layer_bias is not None else 0
                node_label = f'conv2d_{layer_id}_{node_id}'
                nx_graph.add_node(node_label, layer=layer_id)
                
                # init linear constraint LHS implied by the conv operation
                for in_chan_idx in range(this_layer_weight.shape[1]):
                    coeffs = this_layer_weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)
                    # print(f'{coeffs=}')
                    nodes = nodes_array[0][in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                    # print(f'{nodes=}')
                    assert len(coeffs) == len(nodes)
                    for n_, c_ in zip(coeffs, nodes):
                        nx_graph.add_edge(node_label, n_, label=c_)
                        nx_graph.add_edge(n_, node_label, label=c_)
                        
                node_to_id[node_label] = node_id + offset_node_id
                out_row_vars.append(node_label)
                node_id += 1
            out_chan_vars.append(out_row_vars)
        node_labels.append(out_chan_vars)
        
    node_labels = np.array(node_labels)
    node_labels = node_labels.reshape(output_shape)
    return node_labels, node_to_id

def _prepare_input(nx_graph, input_shape):
    assert input_shape[0] == 1
    node_to_id = {}
    
    node_labels = []
    for node_id in range(np.prod(input_shape)):
        node_label = f'input_{node_id}'
        nx_graph.add_node(node_label, layer=0)
        node_to_id[node_label] = node_id
        node_labels.append(node_label)

    node_labels = np.array(node_labels)
    node_labels = node_labels.reshape(input_shape)
    return node_labels, node_to_id

def prepare_graph(net, input_shape, output_shape, args):

    G = nx.DiGraph()
    node_to_id = {}
    
    # process input layer
    layer_node_labels, layer_node_to_id = _prepare_input(G, input_shape)
    node_to_id.update(layer_node_to_id)
    
    # process hidden layers
    layer_id = 1
    pre = torch.randn(input_shape)
    for layer in list(net.modules())[1:]: # TODO: update
        post = layer(pre) 
        if isinstance(layer, torch.nn.Linear):
            layer_node_labels, layer_node_to_id = _prepare_linear(
                nx_graph=G, 
                pre_nodes=layer_node_labels,
                layer=layer, 
                layer_id=layer_id, 
                input_shape=pre.shape, 
                output_shape=post.shape,
                offset_node_id=len(node_to_id),
            )
        elif isinstance(layer, torch.nn.Conv2d):
            layer_node_labels, layer_node_to_id = _prepare_conv2d(
                nx_graph=G, 
                pre_nodes=layer_node_labels,
                layer=layer, 
                layer_id=layer_id, 
                input_shape=pre.shape, 
                output_shape=post.shape,
                offset_node_id=len(node_to_id),
            )
        elif isinstance(layer, torch.nn.ReLU):
            continue
        elif isinstance(layer, torch.nn.Flatten):
            layer_node_to_id = {}
            layer_node_labels = layer_node_labels.reshape(1, -1)
        else:
            print('Unsupported:', layer)
            raise
        
        node_to_id.update(layer_node_to_id)
        layer_id += 1
        pre = post
        print(layer, f'{layer_node_labels.shape=}, {pre.shape=}, {post.shape=} {len(node_to_id)=}')
        
    assert len(list(set(node_to_id.keys()))) == len(node_to_id) # no duplicate names
    assert len(list(set(node_to_id.values()))) == len(node_to_id) # no duplicate values
    
    # print(f'{len(list(set(node_to_id.keys())))=}')
    # print(f'{len(list(set(node_to_id.values())))=}')
    exit()
    #



#     G = nx.DiGraph()
#     node_to_id = {}
#     v = 0
#     i = 0
#     # prepare 1st layer
#     for name, param in list(net.named_parameters())[:-2]:
#         print(name, param)
#         if 'weight' in name and len(param.shape) == 4:  # Convolutional filters
#             print(f'[+] loading cnn 1st layer {name=} {param.shape=}')
#             n_out, n_in, h, w = param.shape
#             current_layer_type = 'cnn'
#             for j in range(n_out):
#                 for h_idx in range(h):
#                     for w_idx in range(w):
#                         node_id = f'{i}_{j}_{h_idx}_{w_idx}'
#                         # print(node_id)
#                         G.add_node(node_id)
#                         node_to_id[node_id] = v
#                         v += 1
#             i += 1
#             prev_name, prev_param = name, param
#             previous_layer_type = current_layer_type
#             break
#         elif 'weight' in name and len(param.shape) == 2:  # Fully connected layer
#             print(f'[+] loading fc 1st layer {name=} {param.shape=}')
#             n_out, n_in = param.shape
#             current_layer_type = 'fc'
#             for j in range(n_out):
#                 node_id = f'{i}_{j}'
#                 # print(node_id)
#                 G.add_node(node_id)
#                 node_to_id[node_id] = v
#                 v += 1
#             i += 1
#             prev_name, prev_param = name, param
#             previous_layer_type = current_layer_type
#             break
#
#     # prepare next layers
#     for name, param in list(net.named_parameters())[i:-2]:
#         if 'weight' in name and len(param.shape) == 4:  # Convolutional filters
#             print(f'[+] loading cnn layer {name=} {param.shape=}')
#             current_layer_type = 'cnn'
#             if previous_layer_type == 'cnn':
#                 # list all in nodes
#                 # list all out nodes
#                 for j1, j2 in itertools.product(range(n_in), range(n_out)):
#                     for h_idx in range(h):
#                         for w_idx in range(w):
#                             in_node = f'{i-1}_{j1}_{h_idx}_{w_idx}'
#                             out_node = f'{i}_{j2}_{h_idx}_{w_idx}'
#                             print(f'{in_node}<->{out_node}')
#                             G.add_edge(in_node, out_node, label=param[j2, j1, h_idx, w_idx])
#                             G.add_edge(out_node, in_node, label=param[j2, j1, h_idx, w_idx])
#             elif previous_layer_type == 'fc':  # Handling transition from fc to cnn (unlikely direct, but consider edge cases)
#                 raise NotImplementedError('Direct transition from fc to cnn is unexpected.')
#             previous_layer_type = current_layer_type
#             flat_size = n_out * h * w
#             i += 1
#         elif 'weight' in name and len(param.shape) == 2:  # Fully connected layer
#             print(f'[+] loading fc layer {name=} {param.shape=}')
#             current_layer_type = 'fc'
#             if previous_layer_type == 'cnn':
#                 for j2 in range(n_out):
#                     for flat_idx in range(flat_size):
#                         G.add_edge(f'{i-1}_{flat_idx}', f'{i}_{j2}', label=param[j2, flat_idx])
#                         G.add_edge(f'{i}_{j2}', f'{i-1}_{flat_idx}', label=param[j2, flat_idx])
#             elif previous_layer_type == 'fc':
#                 for j1, j2 in itertools.product(range(n_in), range(n_out)):
#                     G.add_edge(f'{i-1}_{j1}', f'{i}_{j2}', label=param[j2, j1])
#                     G.add_edge(f'{i}_{j2}', f'{i-1}_{j1}', label=param[j2, j1])
#             previous_layer_type = current_layer_type
#             i += 1
#
#     edge_index = []
#     edge_weight = []
#     # from pprint import pprint
#     # pprint(node_to_id)
#     for u, v, data in G.edges(data=True):
#         edge_index.append([node_to_id[u], node_to_id[v]])
#         edge_weight.append(data.get('label'))
#     edge_index = torch.tensor(edge_index, dtype=torch.long).T
#     edge_weight = torch.tensor(edge_weight)
#     edge_weight = torch.abs(edge_weight)
#     return edge_index, edge_weight

# def prepare_fc_graph(net, args):
#     # create networkx graph from net.parameters()
#     G = nx.DiGraph()
#     node_to_id = {}
#     v = 0
#     i = 0
#     for name, param in list(net.named_parameters())[:-2]:
#         # for 1st just create the node with label f'{layer_id}_{neuron_id}'
#         if 'weight' in name:
#             # print(name, param.shape)
#             n_out, n_in = param.shape
#             for j in range(n_out):
#                 G.add_node(f'{i}_{j}')
#                 node_to_id[f'{i}_{j}'] = v
#                 v += 1
#             if i > 0:
#                 for j1, j2 in itertools.product(range(n_in), range(n_out)):
#                     G.add_edge(f'{i-1}_{j1}', f'{i}_{j2}', label=param[j2, j1])
#                     G.add_edge(f'{i}_{j2}', f'{i-1}_{j1}', label=param[j2, j1])
#             i += 1
#     # convert G to edge_index and edge_label
#     edge_index = []
#     edge_weight = []
#     for u, v, data in G.edges(data=True):
#         edge_index.append([node_to_id[u], node_to_id[v]])
#         edge_weight.append(data.get('label'))
#     edge_index = torch.tensor(edge_index, dtype=torch.long).T
#     edge_weight = torch.tensor(edge_weight)
#     # avoid negative edge_weight
#     edge_weight = torch.abs(edge_weight)
#     return edge_index, edge_weight

import torch
import gc

class CudaDefault:
    def __enter__(self):
        # Move all existing tensors to CUDA
        self.tensors = [t for t in gc.get_objects() if isinstance(t, torch.Tensor)]
        for tensor in self.tensors:
            if tensor.is_cuda is False:
                tensor.data = tensor.data.cuda()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Optionally, you could move tensors back to CPU or do other cleanup
        for tensor in self.tensors:
            if tensor.is_cuda:
                tensor.data = tensor.data.cpu()
