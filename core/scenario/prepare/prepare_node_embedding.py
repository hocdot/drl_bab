import neuralsat as ns
import pandas as pd
import torch
import os

from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data

def prepare_node_embedding(args):
    # load hard instances list
    path = os.path.join(args.benchmark_dir, args.benchmark, 'instances.csv')
    df = pd.read_csv(path, names=['onnx', 'vnnlib', 'timeout'])
    #
    for i, (onnx, vnnlib, timeout) in df.iterrows():
        # load full graph from onnx and vnnlib
        print(f'[+] procesing {os.path.basename(onnx)} {os.path.basename(vnnlib)}')
        print(f'    - extracting instance')
        net, input_shape, objs = ns.test.extract_instance(onnx, vnnlib)
        obj = objs.pop(1)
        print(f'    - initializing InteractiveVerifier')
        iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
        # save edge_weight + edge_index
        graph_path = os.path.basename(onnx).replace('.onnx', '.pt')
        graph_dir = os.path.join(args.benchmark_dir, args.benchmark, 'graph')
        os.makedirs(graph_dir, exist_ok=True)
        path = os.path.join(graph_dir, graph_path)
        if os.path.exists(path):
            data = torch.load(path)
            edge_weight = data['edge_weight']
            edge_index = data['edge_index']
        else:
            print(f'    - getting edge data from InteractiveVerifier')
            edge_weight, edge_index = iv.get_edge_data(obj)
            data = {'edge_weight': edge_weight, 'edge_index': edge_index}
            torch.save(data, path)
        #
        embed_path = os.path.basename(vnnlib).replace('.vnnlib', '.pt')
        embed_dir = os.path.join(args.benchmark_dir, args.benchmark, 'embed')
        os.makedirs(embed_dir, exist_ok=True)
        path = os.path.join(embed_dir, embed_path)
        if os.path.exists(path):
            e = torch.load(path)
            print(f'    - sub graph embedding {e.shape=}')
        else:
            # load first observation
            x = iv.get_initial_node_data(obj)
            x = torch.cat(x, dim=1)
            print(f'loaded full graph: {edge_weight.shape=} {edge_index.shape=} {edge_index.max().item()=} {x.shape=}')
            # train embedding
            print(f'[+] train node embedding using Node2Vec')
            data = Data(edge_index=edge_index)
            model = Node2Vec(edge_index=data.edge_index,
                             embedding_dim=args.n_embed,
                             walk_length=args.n2v_walk_length,
                             p=args.n2v_p,
                             q=args.n2v_q,
                             context_size=args.n2v_context_size,
                             walks_per_node=args.n2v_n_walk_per_node)
            model = model.to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.n2v_lr)
            for epoch in range(args.n2v_n_epoch):
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = node2vec.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f'-{epoch=} {total_loss=:.4f}')
            e = model.embedding.weight.data
            print(f'    - full graph embedding {e.shape=}')
            # extract embedding
            mask = x[0, :, -1]
            node_indices = torch.where(mask == 0)[0]
            e = e[node_indices]
            e = e.detach().cpu()
            print(f'    - sub graph embedding {e.shape=}')
            # save embedding of active neurons
            torch.save(e, path)
        break
