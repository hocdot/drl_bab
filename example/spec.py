import torch

def create():
    lb = torch.tensor([-2.0, -1.0], dtype=torch.float)
    ub = torch.tensor([2.0, 1.0], dtype=torch.float)
    lb = lb[None]
    ub = ub[None]
    x0 = torch.tensor([0.5, 0.5], dtype=torch.float)
    epsilon = 0.5
    return lb, ub, x0, epsilon
