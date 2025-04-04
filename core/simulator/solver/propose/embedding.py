import torch.nn.functional as F
import torch.nn as nn
import torch

import torch_geometric.nn as gnn

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.embed_class in ['Linear']:
            embed_class = nn.Linear
        else:
            embed_class = getattr(gnn, args.embed_class)
        K = 3
        alpha = 0.8
        if args.embed_class in ['ChebConv', 'TAGConv']:
            self.conv1  = embed_class(args.n_observation, args.embed_n_hidden, K=K)
            self.conv2  = embed_class(args.embed_n_hidden, args.embed_n_hidden, K=K)
        elif args.embed_class in ['SSGConv']:
            self.conv1  = embed_class(args.n_observation, args.embed_n_hidden, alpha=alpha, K=K)
            self.conv2  = embed_class(args.embed_n_hidden, args.embed_n_hidden, alpha=alpha, K=K)
        elif args.embed_class in ['LGConv']:
            self.conv1  = embed_class()
            self.conv2  = embed_class()
        else:
            self.conv1  = embed_class(args.n_observation, args.embed_n_hidden)
            self.conv2  = embed_class(args.embed_n_hidden, args.embed_n_hidden)
        self.fc  = nn.Linear(args.embed_n_hidden, args.n_embed)
        self.args = args

    def forward(self, x, edge_index, edge_weight):
        # print(x.amin(dim=[0, 1, 2]).detach().cpu().numpy(), x.amax(dim=[0, 1, 2]).detach().cpu().numpy())
        # extract args
        args = self.args
        # extract shape
        bs, n_layer, n_neuron, n_observation = x.shape
        x = x.view(bs, -1, n_observation)
        # inference
        if args.embed_class in ['Linear']:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
        elif args.embed_class in ['GCN2Conv', 'FAConv']:
            x = self.conv1(x, x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv2(x, x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        else:
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = self.fc(x)
        x = x.view(bs, n_layer, n_neuron, args.n_embed)
        return x
