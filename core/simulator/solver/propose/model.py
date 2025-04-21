import torch.nn.functional as F
import torch.nn as nn
import torch
import math

import torch_geometric.nn as gnn

# class DQN(nn.Module):
#
#     def __init__(self, args):
#         super().__init__()
#         self.conv1 = gnn.SAGEConv(args.n_feature, args.n_hidden)
#         nn.ReLU(),
#         self.conv2 = gnn.SAGEConv(args.n_hidden, args.n_hidden)
#         nn.ReLU(),
#         self.conv3 = gnn.SAGEConv(args.n_hidden, args.n_hidden)
#         nn.ReLU(),
#         self.conv4 = gnn.SAGEConv(args.n_hidden, 1)
#         self.args = args
#
#     def forward(self, observation, env):
#         # extract args
#         args = self.args
#         edge_weight, edge_index, G, hidden_shapes = env.edge_weight, env.edge_index, env.G, env.hidden_shapes
#         # place features to [n_subproblem, n_active_neuron at env.reset(), n_feature]
#         feature, _, active_neurons, n_active_neuron = observation
#         # extract dimension
#         n_layer = len(feature)
#         n_subproblem, _, n_feature = feature[0].shape
#         n_init_active_neuron = G.number_of_nodes()
#         init_active_neurons = torch.tensor([_ for _ in G.nodes()], device=args.device)
#         # flatten all layers
#         x = torch.zeros(n_subproblem, n_init_active_neuron, n_feature, device=args.device)
#         offset = 0
#         for layer in range(n_layer):
#             for p in range(n_subproblem):
#                 nan           = n_active_neuron[layer][p]
#                 an            = active_neurons[layer][p, :nan] + offset
#                 mask          = torch.isin(an, init_active_neurons)
#                 indices       = torch.nonzero(mask, as_tuple=True)[0]
#                 x[p, indices, :] = feature[layer][p, :nan, :]
#             offset += math.prod(hidden_shapes[layer])
#         x = self.conv1(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = self.conv4(x, edge_index, edge_weight=edge_weight).squeeze(-1) # n_subproblem, n_init_active_neuron
#         # scores layer_i of [n_subproblem, n_active_neuron_i]
#         scores = []
#         offset = 0
#         for layer in range(n_layer):
#             score = torch.ones([n_subproblem, feature[layer].shape[1]]) * -5
#             for p in range(n_subproblem):
#                 nan            = n_active_neuron[layer][p]
#                 an             = active_neurons[layer][p, :nan] + offset
#                 mask           = torch.isin(an, init_active_neurons)
#                 indices        = torch.nonzero(mask, as_tuple=True)[0]
#                 score[p, :nan] = x[p, indices]
#             offset += math.prod(hidden_shapes[layer])
#             scores.append(score)
#         return scores

class DQN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.conv1 = gnn.SAGEConv(args.n_feature, args.n_hidden)
        nn.ReLU(),
        self.conv2 = gnn.SAGEConv(args.n_hidden, args.n_hidden)
        nn.ReLU(),
        self.conv3 = gnn.SAGEConv(args.n_hidden, args.n_hidden)
        nn.ReLU(),
        self.conv4 = gnn.SAGEConv(args.n_hidden, 1)
        self.args = args

    def forward(self, observation, env):
        # extract args
        args = self.args
        edge_weight, edge_index, G, hidden_shapes = env.edge_weight, env.edge_index, env.G, env.hidden_shapes
        # place features to [n_subproblem, n_active_neuron at env.reset(), n_feature]
        feature, _, active_neurons, n_active_neuron = observation
        # extract dimension
        n_layer = len(feature)
        n_subproblem, _, n_feature = feature[0].shape
        n_init_active_neuron = G.number_of_nodes()
        init_active_neurons = torch.tensor([_ for _ in G.nodes()], device=args.device)
        # flatten all layers
        x = torch.zeros(n_subproblem, n_init_active_neuron, n_feature, device=args.device)
        offset = 0
        for layer in range(n_layer):
            for p in range(n_subproblem):
                nan           = n_active_neuron[layer][p]
                an            = active_neurons[layer][p, :nan] + offset
                mask          = torch.isin(an, init_active_neurons)
                indices       = torch.nonzero(mask, as_tuple=True)[0]
                x[p, indices, :] = feature[layer][p, :nan, :]
            offset += math.prod(hidden_shapes[layer])
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight=edge_weight).squeeze(-1) # n_subproblem, n_init_active_neuron
        # scores layer_i of [n_subproblem, n_active_neuron_i]
        scores = []
        offset = 0
        for layer in range(n_layer):
            score = torch.ones([n_subproblem, feature[layer].shape[1]]) * -5
            for p in range(n_subproblem):
                nan            = n_active_neuron[layer][p]
                an             = active_neurons[layer][p, :nan] + offset
                mask           = torch.isin(an, init_active_neurons)
                indices        = torch.nonzero(mask, as_tuple=True)[0]
                score[p, :nan] = x[p, indices]
            offset += math.prod(hidden_shapes[layer])
            scores.append(score)
        return scores
