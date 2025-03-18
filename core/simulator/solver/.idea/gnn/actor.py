import torch.nn.functional as F
import torch.nn as nn
import torch

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fc1    = nn.Linear(args.n_observation, args.actor_n_hidden)
        self.relu1  = nn.ReLU()
        self.fc2    = nn.Linear(args.actor_n_hidden, args.actor_n_hidden)
        self.relu2  = nn.ReLU()
        self.fc_out = nn.Linear(args.actor_n_hidden, 1)

    def forward(self, x):
        # extract shape
        bs, n_layer, n_neuron, n_observation = x.shape
        # forward
        x = x.view(bs, -1, n_observation)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc_out(x)
        return x
