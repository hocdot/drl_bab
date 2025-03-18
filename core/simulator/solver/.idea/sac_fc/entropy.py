import torch.nn.functional as F
import torch.nn as nn
import torch

class Entropy(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.target_entropy = - torch.tensor(1).item()
        self.log_alpha      = nn.Parameter(torch.zeros(1))
        self.alpha          = self.log_alpha.exp().item()
