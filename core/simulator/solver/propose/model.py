import torch.nn.functional as F
import torch.nn as nn
import torch

class DQN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(args.n_feature + args.n_embed, args.n_hidden),
            nn.ReLU(),
            nn.Linear(args.n_hidden, args.n_hidden),
            nn.ReLU(),
            nn.Linear(args.n_hidden, args.n_embed),
        )
        self.score = nn.Sequential(
            nn.Linear(args.n_embed, args.n_hidden),
            nn.ReLU(),
            nn.Linear(args.n_hidden, args.n_hidden),
            nn.ReLU(),
            nn.Linear(args.n_hidden, args.n_hidden),
            nn.ReLU(),
            nn.Linear(args.n_hidden, 1),
        )
        self.args  = args

    # def forward(self, xs):
    #     args = self.args
    #     bs = xs[0].shape[0]
    #     n_layer = len(xs)
    #     # extract shape
    #     ss = []
    #     for x in xs:
    #         # print(f'{x.shape=}')
    #         # bs, n_active_neuron, _ = x.shape
    #         e = self.embed(x)                              # (bs, n_active_neuron, n_embed)
    #         s = self.score(e).squeeze(-1)                  # (bs, n_active_neuron)
    #         ss.append(s)
    #     return ss

    def forward(self, xs):
        args = self.args
        bs = xs[0].shape[0]
        n_layer = len(xs)
        e = torch.zeros(bs, args.n_embed).to(xs[0])
        # extract shape
        ss = []
        # forward
        for x in xs:
            bs, n_active_neuron, _ = x.shape
            e = e[:, None].expand(-1, n_active_neuron, -1) # (bs, n_active_neuron, n_embed)
            x = torch.cat([x, e], dim=2)                   # (bs, n_active_neuron, n_feature + n_embed)
            e = self.embed(x)                              # (bs, n_active_neuron, n_embed)
            s = self.score(e).squeeze(-1)                  # (bs, n_active_neuron)
            e = e.mean(1)                                   # (bs, n_embed)
            ss.append(s)
        # print([_.shape for _ in ss])
        
        # backward
        for x in reversed(xs[:-1]):
            bs, n_active_neuron, _ = x.shape
            e = e[:, None].expand(-1, n_active_neuron, -1) # (bs, n_active_neuron, n_embed)
            x = torch.cat([x, e], dim=2)                   # (bs, n_active_neuron, n_feature + n_embed)
            e = self.embed(x)                              # (bs, n_active_neuron, n_embed)
            s = self.score(e).squeeze(-1)                  # (bs, n_active_neuron)
            e = e.mean(1)                                   # (bs, n_embed)
            ss.append(s)
        ss = list(reversed(ss[-n_layer:]))
        # print([_.shape for _ in ss])
        return ss