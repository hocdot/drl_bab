import torch.nn.functional as F
import torch.nn as nn
import torch

import torch_geometric.nn as gnn

# gnnclass = gnn.TAGConv
#
# class Actor(nn.Module):
#
#     def __init__(self, args):
#         super().__init__()
#         self.conv1  = gnnclass(args.n_observation, args.actor_n_hidden)
#         self.relu1  = nn.ReLU()
#         self.conv2  = gnnclass(args.actor_n_hidden, args.actor_n_hidden)
#         self.relu2  = nn.ReLU()
#         self.fc     = nn.Linear(args.actor_n_hidden, 1)
#
#     def forward(self, x, edge_index, edge_weight):
#         # extract shape
#         bs, n_layer, n_neuron, n_observation = x.shape
#         # forward
#         x = x.view(bs, -1, n_observation)
#         x = self.relu1(self.conv1(x, edge_index, edge_weight=edge_weight))
#         x = self.relu2(self.conv2(x, edge_index, edge_weight=edge_weight))
#         x = self.fc(x)
#         x = x.view(bs, n_layer, n_neuron)
#         return x.squeeze(-1)

class Actor(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fc1    = nn.Linear(args.n_observation, args.actor_n_hidden)
        self.relu1  = nn.ReLU()
        self.fc2    = nn.Linear(args.actor_n_hidden, args.actor_n_hidden)
        self.relu2  = nn.ReLU()
        self.fc_out = nn.Linear(args.actor_n_hidden, 1)

    def forward(self, x, edge_index=None, edge_weight=None):
        # extract shape
        bs, n_layer, n_neuron, n_observation = x.shape
        # forward
        # x = x.view(bs, -1, n_observation)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc_out(x)
        return x.squeeze(-1)
