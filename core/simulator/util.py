import matplotlib.pyplot as plt
import networkx as nx
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

def prepare_graph(net, input_shape, output_shape, args):
    x = torch.randn(input_shape)
    hidden_shapes = [_.shape for _ in net(x)[1]]
    print(input_shape)
    for _ in hidden_shapes:
        print(_)
    print(output_shape)
    exit()
    #
    G = nx.DiGraph()



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
