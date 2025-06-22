from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from beartype import beartype

import torch_geometric.nn as gnn
import torch.nn as nn
import torch

class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embedding = gnn.GCNConv(args.n_input, args.n_embed)

    @beartype
    def forward(self, x: torch.Tensor,
                      edge_index: torch.Tensor,
                      edge_weight: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x, edge_index, edge_weight)
        return z
