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

    def __init__(self, transitions, reward_scaler, args):
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
        
        assert self.prev_action.shape[0] == self.action.shape[0] == self.done.shape[0] == self.reward.shape[0] == self.observation.shape[0] == self.next_observation.shape[0]
        
        # # filter out batch where all neurons are split on all branch
        # m1 = self.next_observation[:, 0][:, :, -1]
        # m2 = self.next_observation[:, 1][:, :, -1]
        # active_m = (m1.sum(dim=1) > 0) & (m2.sum(dim=1) > 0) # not all neurons are split on all branch
        # print(f'{m1.shape=} {m2.shape=} {active_m.shape=} {self.next_observation.shape=}')
        # self.prev_action = self.prev_action[active_m]
        # self.action = self.action[active_m]
        # self.done = self.done[active_m]
        # self.reward = self.reward[active_m]
        # self.observation = self.observation[active_m]
        # self.next_observation = self.next_observation[active_m]
        
        # if args.verbose:
        #     if (active_m == 0).any():
        #         print(f'    ! filtered out all neurons are split: {m1.shape[0]} => {self.next_observation.shape[0]}')
        
        # if not args.save_offline_data:
        #     if not reward_scaler.fitted:
        #         self.reward = reward_scaler.fit_transform(self.reward) / args.n_max_step
        #     else:
        #         self.reward = reward_scaler.transform(self.reward) / args.n_max_step

        assert self.prev_action.shape[0] == self.action.shape[0] == self.done.shape[0] == self.reward.shape[0] == self.observation.shape[0] == self.next_observation.shape[0]

    def export(self, path):
        data = {
            'prev_action': self.prev_action.detach().cpu(),
            'action': self.action.detach().cpu(),
            'done': self.done.detach().cpu(),
            'reward': self.reward.detach().cpu(),
            'observation': self.observation.detach().cpu(),
            'next_observation': self.next_observation.detach().cpu(),
        }
        torch.save(data, path)