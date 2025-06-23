from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .base import BaseSolver

import torch.optim as optim
import torch.nn as nn
import numpy as np
import simulator
import torch


class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        # create actor/critic model
        self.create_model()

    def create_model(self):
        args = self.args
        self.actor  = simulator.pointer_net.Actor(args).to(device=args.device)
        self.critic = simulator.pointer_net.Critic(args).to(device=args.device)
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = \
            optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()),
                       lr=args.lr)
        self.actor_scheduler = \
            optim.lr_scheduler.StepLR(self.actor_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)
        self.critic_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                          lr=args.lr)
        self.critic_scheduler = \
            optim.lr_scheduler.StepLR(self.critic_optimizer,
                                      step_size=args.lrs_step_size,
                                      gamma=args.lrs_gamma)

    def prepare_test(self, observation):
        self.z = self.actor.decoder.get_z0(observation)

    def select_action(self, observation):
        # extract parameters
        actor = self.actor
        args  = self.args
        bs    = observation.shape[0]
        x     = observation.to(args.device)
        ew    = self.edge_weight
        ei    = self.edge_index
        # embed
        e = actor.embedding(x, ei, ew)
        # gather
        # TODO: env splited action
        # z = actor.decoder.gather_z(e, env.split_action)
        mask = x[:, :, -1]
        # pick embedding of only active neurons
        indices  = [] # indices of active neuron
        e_ = [] # packed
        for b in range(bs):
            idx = torch.where(mask[b] == 1)[0]
            e_.append(e[b, idx])
            indices.append(idx)
        e = nn.utils.rnn.pack_sequence(e_, enforce_sorted=False)
        # encode
        ref, (h, c) = actor.encoder(e)
        # decode loop
        # get actual number of nodes
        _, n_nodes = pad_packed_sequence(e)
        max_node   = len(e.batch_sizes)
        # init
        nodes, log_probs = [], []
        mask = torch.zeros([bs, max_node], device=args.device)
        # fill mask with 1 for padding
        for b in range(bs):
            mask[b, n_nodes[b]:] = 1.0
        # one step decoding
        print(f'{z.shape=} {h.shape=} {c.shape=}')
        _, (h, c) = actor.decoder(z, h, c)
        q = h.squeeze(0)
        # glimpse
        for _ in range(args.n_glimpse):
            q = actor.glimpse(q, ref, mask)
        # pointer
        logits = actor.pointer(q, ref, mask)
        # logits: (bs, max_node)
        log_prob = torch.log_softmax(logits, dim=-1)
        # select next node
        action = torch.argmax(log_prob, dim=1, keepdim=True).long() # (bs, )
        # convert active neuron indices to original indices
        action = action.tolist()
        action = [indices[i][a[:n_nodes[i]]].item() for i, a in enumerate(action)]
        return action
