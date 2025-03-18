from collections import namedtuple, deque
import numpy as np
import random

Transition = namedtuple('Transition',
                        ('observation', 'action', 'next_observation', 'reward'))
PretrainTransition = namedtuple('PretrainTransition',
                                    ('observation', 'mask', 'min_reward', 'max_reward', 'next_observation', 'iwb', 'onnx_id'))

class ReplayBuffer:

    def __init__(self, capacity):
        self.memory   = deque([], maxlen=capacity)
        self.capacity = capacity

    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

#     def push(self, *args):
#         self.memory.append(Transition(*args))

    def push(self, *args):
        # convert all items to cpu
        args_cpu = [_.detach().cpu() for _ in args]
        self.memory.append(FullFeedbackTransition(*args_cpu))

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self)])
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PretrainBuffer:

    def __init__(self, capacity):
        self.memory   = deque([], maxlen=capacity)
        self.capacity = capacity

    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        # convert all items to cpu
        args_cpu = [_.detach().cpu() for _ in args]
        self.memory.append(PretrainTransition(*args_cpu))

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self)])
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
