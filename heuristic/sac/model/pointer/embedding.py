import torch.nn as nn
import torch

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.use_embedding:
            self.embedding = nn.Linear(args.n_input + args.gcn_embed_n_hidden, args.n_embed, bias=False, device=args.device)
        else:
            self.embedding = nn.Linear(args.n_input, args.n_embed, bias=False, device=args.device)

    def forward(self, x):
        z = self.embedding(x)
        return z
