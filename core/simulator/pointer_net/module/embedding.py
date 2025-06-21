from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from beartype import beartype

import torch_geometric.nn as gnn
import torch.nn as nn
import torch

# class Embedding(nn.Module):
#
#     def __init__(self, args):
#         super().__init__()
#         self.embedding = nn.Linear(args.n_input,
#                                    args.n_embed,
#                                    bias=False,
#                                    device=args.device)
#
#     @beartype
#     def forward(self, x: PackedSequence) -> PackedSequence:
#         padded_x, lengths = pad_packed_sequence(x, batch_first=True)
#         z = self.embedding(padded_x)
#         return pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embedding = gnn.GCN2Conv(args.n_input, args.n_embed)

    @beartype
    def forward(self, x: torch.Tensor,
                      x0: torch.Tensor,
                      edge_index: torch.Tensor,
                      edge_weight: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x, x0, edge_index, edge_weight)
        return z
