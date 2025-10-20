import torch.nn as nn
import torch

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.use_embedding:
            self.embedding = nn.Linear(2 * args.gcn_embed_n_hidden, args.n_embed, bias=False, device=args.device)
            self.projection = nn.Linear(args.n_input, args.gcn_embed_n_hidden, bias=False, device=args.device)
        else:
            self.embedding = nn.Linear(args.n_input, args.n_embed, bias=False, device=args.device)
            self.projection = None

    def forward(self, x):
        if self.args.use_embedding:
            x_emb  = x[..., :-self.args.n_input]
            x_proj = x[..., -self.args.n_input:]
            x_proj = self.projection(x_proj)
            x = torch.cat([x_emb, x_proj], dim=-1)
        z = self.embedding(x)
        return z
