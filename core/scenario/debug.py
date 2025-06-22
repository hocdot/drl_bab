from neuralsat.heuristic.util import compute_masks
import itertools as it
import neuralsat as ns
import networkx as nx
import numpy as np
import simulator
import torch

def debug(args):
    # extract problem
    onnx_path = f'../data/benchmark/{args.benchmark}/onnx/0.onnx'
    vnnlib_path = f'../data/benchmark/{args.benchmark}/vnnlib/0.vnnlib'
    net, input_shape, objs = ns.test.extract_instance(onnx_path, vnnlib_path)
    iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device='cpu')
    obj = objs.pop(4)
    obj = objs.pop(1)

    # extract node and edge
    x = torch.cat(iv.get_node_data(obj))[None]
    edge_weight, edge_index = iv.get_edge_data(obj)

    # convert to device
    x = x.to(args.device)
    edge_index = edge_index.to(args.device)
    edge_weight = edge_weight.to(args.device)

    # try infer
#     actor = simulator.pointer_net.Actor(args).to(args.device)
#     critic = simulator.pointer_net.Critic(args).to(args.device)
#     ll, y = actor(x=x, edge_index=edge_index, edge_weight=edge_weight)
#     print(ll, y)
#     f = critic(x=x, edge_index=edge_index, edge_weight=edge_weight)
#     print(f)

    # randomly choose action
    n_subproblem, n_node, n_input = x.shape
    n_next_node = 3
    action = []
    for i in range(n_subproblem):
        mask = x[i, :, -1]
        next_nodes = []
        for j in range(n_next_node):
            active_neurons = torch.where(mask == 1)[0].detach().cpu().numpy()
            next_node = np.random.choice(active_neurons)
            x[i, next_node, -1] = 0
            next_nodes.append(next_node)
        action.append(next_nodes)
    print(action)
