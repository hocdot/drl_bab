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
    iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
    obj = objs.pop(4)
    obj = objs.pop(1)

    # extract node and edge
    node_data = torch.cat(iv.get_node_data(obj))
    # print([_.shape for _ in node_data])
    # edge_weight, edge_index = iv.get_edge_data()
    # print(f'{edge_weight.shape=} {edge_index.shape=}')
    # print(edge_index.max())

    # try infer
    net = simulator.pointer_net.Embedding(args).to(device='cuda:0')
    node_data = node_data.to(device='cuda:0')
    x = torch.nn.utils.rnn.pack_sequence([node_data], enforce_sorted=False)
    z = net(x)
    print(z.data.shape)

    # TODO:
    # step 1: embedding all node in node data + edge_weight edge_index
    # step 2: from embedding, pick only mask
