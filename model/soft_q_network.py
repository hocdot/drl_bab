import torch.nn as nn
import torch

from .pointer import Embedding, Glimpse, Encoder, Decoder, Pointer2

class SoftQNetwork(nn.Module):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # init model
        self.embedding  = Embedding(args)
        self.glimpse    = Glimpse(args)
        self.encoder    = Encoder(args)
        self.decoder    = Decoder(args)
        self.pointer    = Pointer2(args)
        self.device     = args.device
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            nn.init.uniform_(p.data, -0.01, 0.01)
            # nn.init.uniform_(p.data, -0.08, 0.08)

    def forward(self, x, z):
        x = x.to(self.device)
        mask = x[:, :, -1]
        # embed
        e = self.embedding(x)
        # encode
        ref, (h0, c0) = self.encoder(e)
        # decode
        _, (h, _) = self.decoder(z, h0, c0)
        # q = h.squeeze(0)    
        q = h[-1] # or mean?
        # glimpse
        for i in range(self.args.n_glimpse * 3):
            q = self.glimpse(q, ref, mask)
        # pointer
        logits = self.pointer(q, ref, mask)
        return logits

    def decode_and_forward(self, x, a):
        n_subproblem = x.shape[0]
        # compute embedding
        e = self.embedding(x)
        # initialize z
        z = torch.zeros(n_subproblem, 1, self.args.n_embed, device=self.device)
        # gather z after reset (no action has been selected yet)
        indices = torch.where(a == -1)[0]
        if len(indices) > 0:
            z[indices] = self.decoder.get_z0(x[indices])
        # gather z when action has been selected
        indices = torch.where(a != -1)[0]
        if len(indices) > 0:
            z[indices] = self.decoder.gather_z(e, a[indices])
        return self(x, z)
