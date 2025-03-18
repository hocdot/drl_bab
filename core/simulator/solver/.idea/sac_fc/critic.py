import torch.nn.functional as F
import torch.nn as nn
import torch

class Critic(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.n_observation, args.critic_n_hidden)
        self.fc2 = nn.Linear(args.critic_n_hidden, args.critic_n_hidden)
        self.fc  = nn.Linear(args.critic_n_hidden, 1)

    def forward(self, x, a):
        # extract shape
        bs, n_layer, n_neuron, n_observation = x.shape
        x = x.view(bs, -1, n_observation)
        # inference
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc(x)
        x = x.view(bs, n_layer, n_neuron)
        x_ = []
        for i in range(bs):
            x_.append(x[i, a[i, 0], a[i, 1]])
        x = torch.stack(x_)
        return x
