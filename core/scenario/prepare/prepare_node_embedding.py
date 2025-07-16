import neuralsat as ns
import pandas as pd
import torch
import os

from torch_geometric.utils import k_hop_subgraph

def prepare_node_embedding(args):
    # load hard instances list
    path = os.path.join(args.benchmark_dir, args.benchmark, 'instances.csv')
    df = pd.read_csv(path, names=['onnx', 'vnnlib', 'timeout'])
    #
    for i, (onnx, vnnlib, timeout) in df.iterrows():
        # load full graph from onnx and vnnlib
        net, input_shape, objs = ns.test.extract_instance(onnx, vnnlib)
        obj = objs.pop(1)
        iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
        edge_weight, edge_index = iv.get_edge_data(obj)
        # load first observation
        x = iv.get_initial_node_data(obj)
        x = torch.cat(x, dim=1)
        print(f'loaded full graph: {edge_weight.shape=} {edge_index.shape=} {edge_index.max().item()=} {x=}')
        print(f'pruned: {sub_edge_index.shape=}')
        break
