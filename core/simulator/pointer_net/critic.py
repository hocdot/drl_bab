import torch.nn as nn
import torch

from . import module

class Critic(nn.Module):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # init model
        self.embedding  = module.Embedding(args)
        self.glimpse    = module.Glimpse(args)
        self.encoder    = module.Encoder(args)
        self.regression = module.Regression(args)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            nn.init.uniform_(p.data, -0.08, 0.08)

    def forward(self, x, edge_index, edge_weight):
        # extract parameters
        args = self.args
        bs   = x.shape[0]
        # init
        x = x.to(args.device)
        # embed
        mask = x[:, :, -1]
        e = self.embedding(x, edge_index, edge_weight)
        # encode
        ref, (h, c) = self.encoder(e)
        q = h[-1]
        for _ in range(args.n_process):
            for _ in range(args.n_glimpse):
                q = self.glimpse(q, ref)
        # predict
        value = self.regression(q)
        return value
