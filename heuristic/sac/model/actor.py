from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch

from .pointer import Embedding, Glimpse, Encoder, Decoder, Pointer

class Actor(nn.Module):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # init model
        self.embedding = Embedding(args)
        self.glimpse   = Glimpse(args)   
        self.encoder   = Encoder(args)
        self.decoder   = Decoder(args)
        self.pointer   = Pointer(args)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            nn.init.uniform_(p.data, -0.08, 0.08)

    def forward(self, x, z):
        # extract parameters
        args = self.args
        bs, n_node, _ = x.shape
        # init
        # mask = torch.zeros([bs, n_node], device=args.device)
        mask     = x[:, :, -1]
        sum_mask = (mask.sum(dim=1, keepdims=True) > 0).to(torch.float)
        ones     = torch.ones_like(mask)
        mask     = sum_mask * (mask) + (1 - sum_mask) * ones
        x = x.to(args.device)
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

    def get_action(self, x, z):
        logits = self(x, z)
        probs  = F.softmax(logits, dim=1)
        m      = x[:, :, -1]
        sum_m  = (m.sum(dim=1, keepdims=True) > 0).to(torch.float)
        ones   = torch.ones_like(m)
        m      = sum_m * (m) + (1 - sum_m) * ones
        probs  = probs * m
        probs /= probs.sum(dim=1, keepdims=True)
        policy_dist = Categorical(probs=probs)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

    def decode_and_get_action(self, x, a):
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
        return self.get_action(x, z)
