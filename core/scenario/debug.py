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
    net = simulator.pointer_net.Actor(args).to(args.device)
    print(f'{x.device=} {edge_index.device=} {edge_weight.device=} {args.device=}')
    ll, y = net(x=x, edge_index=edge_index, edge_weight=edge_weight)
    print(ll, y)
