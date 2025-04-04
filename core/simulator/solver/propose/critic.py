import torch.nn.functional as F
import torch.nn as nn
import torch

import torch_geometric.nn as gnn


class Critic(nn.Module):
#
    def __init__(self, args):
        super().__init__()
        if args.gnn_class in ['Linear']:
            gnn_class = nn.Linear
        else:
            gnn_class = getattr(gnn, args.gnn_class)
        K = 3
        alpha = 0.8
        if args.gnn_class in ['ChebConv', 'TAGConv']:
            self.conv1  = gnn_class(args.n_embed, args.critic_n_hidden, K=K)
            self.conv2  = gnn_class(args.critic_n_hidden, args.critic_n_hidden, K=K)
        elif args.gnn_class in ['SSGConv']:
            self.conv1  = gnn_class(args.n_embed, args.critic_n_hidden, alpha=alpha, K=K)
            self.conv2  = gnn_class(args.critic_n_hidden, args.critic_n_hidden, alpha=alpha, K=K)
        elif args.gnn_class in ['LGConv']:
            self.conv1  = gnn_class()
            self.conv2  = gnn_class()
        else:
            self.conv1  = gnn_class(args.n_embed, args.critic_n_hidden)
            self.conv2  = gnn_class(args.critic_n_hidden, args.critic_n_hidden)
        self.fc  = nn.Linear(args.critic_n_hidden, 1)
        self.args = args
#
    def forward(self, x, edge_index, edge_weight):
        # print(x.amin(dim=[0, 1, 2]).detach().cpu().numpy(), x.amax(dim=[0, 1, 2]).detach().cpu().numpy())
        # extract args
        args = self.args
        # extract shape
        bs, n_layer, n_neuron, n_embed = x.shape
        x = x.view(bs, -1, n_embed)
        # inference
        if args.gnn_class in ['Linear']:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
        elif args.gnn_class in ['GCN2Conv', 'FAConv']:
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
        x = x.view(bs, n_layer, n_neuron)
        return x

# class Critic(nn.Module):
#
#     def __init__(self, args):
#         super().__init__()
#         self.fc1 = nn.Linear(args.n_embed, args.critic_n_hidden)
#         self.fc2 = nn.Linear(args.critic_n_hidden, args.critic_n_hidden)
#         self.fc  = nn.Linear(args.critic_n_hidden, 1)
#
#     def forward(self, x, edge_index, edge_weight):
#         # extract shape
#         bs, n_layer, n_neuron, n_embed = x.shape
#         x = x.view(bs, -1, n_embed)
#         # inference
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc(x)
#         x = x.view(bs, n_layer, n_neuron)
#         return x
