import torch.nn.functional as F
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
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            nn.init.uniform_(p.data, -0.08, 0.08)

    def forward(self, x, z):
        # extract parameters
        args = self.args
        bs, n_node, _ = x.size()
        x = x.to(args.device)
        mask = x[:, :, -1]
        embedding, glimpse, encoder, decoder, pointer = \
                self.embedding, self.glimpse, self.encoder, self.decoder, self.pointer
        # embed
        e = embedding(x)
        # encode
        ref, (h, c) = encoder(e)
        # decode
        _, (h, c) = decoder(z, h, c)
        q = h.squeeze(0)
        # glimpse
        for i in range(args.n_glimpse):
            q = glimpse(q, ref, mask)
        # pointer
        logits = pointer(q, ref, mask)
        return logits

    def decode_and_forward(self, x, a):
        # extract args
        args = self.args
        n_subproblem = x.shape[0]
        # compute embedding
        e = self.embedding(x)
        # initialize z
        z = torch.zeros(n_subproblem, 1, args.n_embed, device=args.device)
        # gather z after reset (no action has been selected yet)
        indices = torch.where(a == -1)[0]
        if len(indices) > 0:
            z[indices] = self.decoder.get_z0(x[indices])
        # gather z when action has been selected
        indices = torch.where(a != -1)[0]
        if len(indices) > 0:
            z[indices] = self.decoder.gather_z(e, a[indices])
        return self(x, z)
