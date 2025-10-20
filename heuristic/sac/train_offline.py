from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import os

from dataset.offline_dataset import OfflineDataset
from .replay_buffer import Transition

def train_offline(self):
    # self.load()
    dataset = OfflineDataset(self.args, files_per_batch=50, use_tqdm=True)
    for i in range(self.args.n_train_offline_epoch):
        dataset.reload_next_batch()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        info = {
            'epoch': i,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'alpha_loss': 0.0,
        }
        for batch in tqdm.tqdm(dataloader, desc=f'[train_offline] {i=}/{self.args.n_train_offline_epoch}'):
            unbatched = {k: v[0].to(self.args.device) for k, v in batch.items()}
            # unbatched['reward'] = unbatched['reward'] / self.args.n_max_visited_subproblem
            if not self.reward_scaler.fitted:
                unbatched['reward'] = self.reward_scaler.fit_transform(unbatched['reward']) / self.args.n_max_visited_subproblem
            else:
                unbatched['reward'] = self.reward_scaler.transform(unbatched['reward']) / self.args.n_max_visited_subproblem
            data = Transition(**unbatched)
            actor_loss, critic_loss, alpha_loss = self.optimize(data, n_qf_optimize=5)
            info['actor_loss'] += actor_loss
            info['critic_loss'] += critic_loss
            info['alpha_loss'] += alpha_loss
            self.global_step += 1
        info['actor_loss'] /= len(dataloader)
        info['critic_loss'] /= len(dataloader)
        info['alpha_loss'] /= len(dataloader)
        print(info, self.reward_scaler)

        # save checkpoint
        self.save()
    
        if self.args.save_csv:
            path = os.path.join(
                self.args.csv_dir, 
                self.args.mode, 
                f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.csv'
            )
        
            df = pd.DataFrame([info])
            df.to_csv(
                path, 
                index=False, 
                mode='w' if i == 0 else 'a',
                header=i == 0
            )