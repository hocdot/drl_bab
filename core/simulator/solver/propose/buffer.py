from collections import namedtuple, deque
import numpy as np
import random

OfflineTransition = namedtuple('OfflineTransition',
                              ('observation', 
                               'unpruned_next_observation_1', 
                               'unpruned_next_observation_2', 
                               'mask', 
                               'reward', 
                               'onnx_id', 
                               'done'))

class OfflineBuffer:

    def __init__(self, capacity):
        self.memory   = deque([], maxlen=capacity)
        self.capacity = capacity

    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        # convert all items to cpu
        args_cpu = []
        for _ in args:
            if type(_) is not bool:
                args_cpu.append(_.detach().cpu())
            else:
                args_cpu.append(_)
        self.memory.append(OfflineTransition(*args_cpu))

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self)])
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
