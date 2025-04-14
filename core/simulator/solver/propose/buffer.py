from collections import namedtuple, deque
import numpy as np
import random

OfflineTransition = namedtuple('OfflineTransition',
                              ('feature',             # bs, n_active_neuron, n_feature
                               'active_neuron_index', # bs, n_active_neuron
                               'unf_1',               # bs, n_active_neuron, n_feature
                               'unf_2',               # bs, n_active_neuron, n_feature
                               'actions',             #
                               'rewards',             #
                               'onnx_id',             #
                               'done'))               #

class OfflineBuffer:

    def __init__(self, capacity):
        self.memory   = deque([], maxlen=capacity)
        self.capacity = capacity

    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        self.memory.append(OfflineTransition(*args))

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self)])
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
