import torch.nn.functional as F
import torch.nn as nn
import torch

import torch_geometric.nn as gnn

class Critic(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.GCNConv(args.n_observation, args.critic_n_hidden)
        self.conv2 = nn.GCNConv(args.critic_n_hidden, args.critic_n_hidden)
        self.fc  = nn.Linear(args.critic_n_hidden, 1)

    def forward(self, x, edge_index, edge_weight):
        # extract shape
        bs, n_layer, n_neuron, n_observation = x.shape
        x = x.view(bs, -1, n_observation)
        # inference
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.fc(x)
        x = x.view(bs, n_layer, n_neuron)
        return x
