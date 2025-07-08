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
    node_data = iv.get_initial_node_data(obj)
    # print([_.shape for _ in node_data])
    x = torch.cat(node_data, dim=1)
    # print(x.shape)
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

    # initial setup (ngoai vong for)
    subproblems = iv.setup(obj)
    torch.manual_seed(0)
    np.random.seed(0)

    for s in range(5):
        # randomly choose action
        action = []
        n_subproblem = len(subproblems.input_lowers)
        for i in range(n_subproblem):
            mask = x[i, :, -1]
            active_neurons = torch.where(mask == 1)[0].detach().cpu().numpy()
            next_node = int(np.random.choice(active_neurons))
            x[i, next_node, -1] = 0
            action.append(next_node)

        # step: n_subproblems ---->     < 2 * n_subproblems (trong vong for)
        subproblems, rewards, next_observations = iv.step(subproblems, action)
        x = torch.cat(next_observations, dim=1)
        print(f'{s=} {subproblems.output_lbs.shape=} {subproblems.output_lbs.shape=}')
        print(f'{action=}\n{iv.get_last_actions(subproblems)=}')
        print()
