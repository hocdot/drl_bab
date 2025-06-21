from neuralsat.heuristic.util import compute_masks
import itertools as it
import neuralsat as ns
import networkx as nx
import numpy as np
import torch

def debug(args):
    #
    onnx_path = f'../data/benchmark/{args.benchmark}/onnx/0.onnx'
    vnnlib_path = f'../data/benchmark/{args.benchmark}/vnnlib/0.vnnlib'
    net, input_shape, objs = ns.test.extract_instance(onnx_path, vnnlib_path)
    iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
    obj = objs.pop(4)
    obj = objs.pop(1)

    #######################
    # EXTRACT NODE FEATURES
    #######################
    node_data = iv.get_node_data(obj)
    print([_.shape for _ in node_data])
    #######################
    # EXTRACT GRAPH DATA
    #######################
    edge_weight, edge_index = iv.get_edge_data()
    print(f'{edge_weight.shape=} {edge_index.shape=}')