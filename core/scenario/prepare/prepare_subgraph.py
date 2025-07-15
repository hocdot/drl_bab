import neuralsat as ns
import pandas as pd
import torch
import os

from torch_geometric.utils import k_hop_subgraph

def prepare_subgraph(args):
    # load hard instances list
    path = os.path.join(args.benchmark_dir, args.benchmark, 'instances.csv')
    df = pd.read_csv(path, names=['onnx', 'vnnlib', 'timeout'])
    #
    for i, (onnx, vnnlib, timeout) in df.iterrows():
        # load full graph from onnx and vnnlib
        net, input_shape, objs = ns.test.extract_instance(onnx, vnnlib)
        obj = objs.pop(1)
        iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device='cpu')
        edge_weight, edge_index = iv.get_edge_data(obj)
        print(f'full graph: {edge_weight.shape=} {edge_index.shape=} {edge_index.max().item()=}')
        # load first observation
        observation  = iv.get_initial_node_data(obj)
        observation  = torch.cat(observation, dim=1)
        mask         = observation[0, :, -1]
        node_indices = torch.where(mask == 0)[0]
        print(f'first observation: {observation.shape=} {mask.shape=} {mask.mean().item()=:0.3f}')
        related_node_indices, sub_edge_index, node_idx_map, edge_mask = \
            k_hop_subgraph(node_indices, num_hops=1, edge_index=edge_index)
        print(f'pruned: {sub_edge_index.shape=}')
        break
