import torch.nn.functional as F
import torch.nn as nn
import torch

import torch_geometric.nn as gnn

import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.conv1     = gnn.GCNConv(args.n_observation, args.actor_n_hidden)
        self.relu1     = nn.ReLU()
        self.conv2     = gnn.GCNConv(args.actor_n_hidden, args.actor_n_hidden)
        self.relu2     = nn.ReLU()
        self.fc_mean   = nn.Linear(args.actor_n_hidden, 1)
        self.fc_logstd = nn.Linear(args.actor_n_hidden, 1)
        # action rescaling
        high = 1
        low  = 0
        self.register_buffer('action_scale', torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, x, edge_index, edge_weight):
        bs, n_layer, n_neuron, n_observation = x.shape
        # forward
        x    = x.view(bs, -1, n_observation)
        x    = self.relu1(self.conv1(x, edge_index, edge_weight=edge_weight))
        x    = self.relu2(self.conv2(x, edge_index, edge_weight=edge_weight))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        # reshape
        mean    = mean.view(bs, n_layer, n_neuron)
        log_std = log_std.view(bs, n_layer, n_neuron)
        return mean, log_std

    def select_action(self, x, edge_index, edge_weight):
        mean, log_std = self(x, edge_index, edge_weight)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # reshape
        bs, n_layer, n_neuron = log_prob.shape
        log_prob = log_prob.view(bs, -1)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.reshape(bs, -1).pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
