from collections import namedtuple, deque
import random
import torch

Transition = namedtuple('Transition',
                        ('prev_action', 'observation', 'action', 'next_observation', 'done', 'reward'))

class ReplayBuffer:

    def __init__(self, capacity):
        self.memory   = deque([], maxlen=capacity)
        self.capacity = capacity

    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TrainingData:

    def __init__(self, transitions, args):
        batch                 = Transition(*zip(*transitions))

        self.prev_action      = torch.tensor(batch.prev_action).to(device=args.device, dtype=torch.long) # (b, )
        self.action           = torch.stack(batch.action).to(device=args.device, dtype=torch.long) # (b, 2)
        self.done             = torch.stack(batch.done).to(device=args.device, dtype=torch.int8) # (b, 2)
        self.reward           = torch.tensor(batch.reward).to(device=args.device, dtype=torch.float32) # (b, )

        # padding due to active neuron index
        self.observation = torch.nn.utils.rnn.pad_sequence(
                batch.observation, batch_first=True, padding_value=0.0
            ).to(device=args.device, dtype=torch.float32)
        self.next_observation = torch.nn.utils.rnn.pad_sequence(
                [b.transpose(0, 1) for b in batch.next_observation], batch_first=True, padding_value=0.0
            ).to(device=args.device, dtype=torch.float32).transpose(1, 2)
        