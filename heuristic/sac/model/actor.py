from torch.distributions.categorical import Categorical
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
        self.device    = args.device
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            nn.init.uniform_(p.data, -0.08, 0.08)

    def forward(self, x, z):
        mask     = x[:, :, -1]
        x = x.to(self.device)
        # embed
        e = self.embedding(x)
        # encode
        ref, (h0, c0) = self.encoder(e)
        # decode
        _, (h, _) = self.decoder(z, h0, c0)
        # q = h.squeeze(0)
        q = h[-1] # or mean?
        # glimpse
        for _ in range(self.args.n_glimpse):
            q = self.glimpse(q, ref, mask)
        # pointer
        logits = self.pointer(q, ref, mask)
        return logits

    def get_action(self, x, z):
        logits = self(x, z)
        probs  = logits.softmax(dim=1)
        m      = x[:, :, -1]
        probs  = probs * m
        probs /= probs.sum(dim=1, keepdims=True)
        policy_dist = Categorical(probs=probs)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = logits.log_softmax(dim=1)
        # check if action is not repeated
        assert all(m[i, a] == 1 for i, a in enumerate(action))
        return action, log_prob, action_probs

    def decode_and_get_action(self, x, a):
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
        return self.get_action(x, z)
