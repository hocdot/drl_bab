from torch.utils.data import DataLoader

from dataset.offline_dataset import OfflineDataset
from .replay_buffer import Transition

def train_offline(self):
    dataset = OfflineDataset(self.args, files_per_batch=20)
    for i in range(self.args.n_train_offline_episode):
        dataset.reload_next_batch()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        for batch in dataloader:
            unbatched = {k: v[0].to(self.args.device) for k, v in batch.items()}
            data = Transition(**unbatched)
            actor_loss, critic_loss, alpha_loss = self.optimize(data)
            print(f"    + {actor_loss=} {critic_loss=} {alpha_loss=}")