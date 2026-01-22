import torch.nn as nn
import torch

class DNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)
        self.relu = nn.ReLU()

        self.layers = [self.linear1, self.relu, self.linear2, self.relu, self.linear3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def create():
    net = DNN()
    weights = [
        [[-0.5, 0.2], [0.3, 0.2]],
        [[-0.9, 0.6], [0.6, 0.9]],
        [[-0.1, 0.2], [-0.3, 0.5]],
        [[1, -1], ],
    ]
    biases = [
        [-0.5, 0.3],
        [0.2, -0.1],
        [-0.4, -0.5],
        [0],
    ]
    linear_layers = [layer for layer in net.layers if isinstance(layer, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        layer.weight = nn.Parameter(torch.tensor(weights[i], dtype=torch.float))
        layer.bias = nn.Parameter(torch.tensor(biases[i], dtype=torch.float))
    return net
