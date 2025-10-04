import pandas as pd
import numpy as np
import torch
import tqdm
import os

from drl.env import Env

class Base:

    def __init__(self, args):
        # save args
        self.args = args
        # create env
        self.env = Env(args)

    def select_action(self, observation):
        raise NotImplementedError

    def test(self, use_tqdm=True):
        self.args.mode = 'test'
        self.load()
        infos = []
        pbar = tqdm.tqdm(range(self.args.n_test_episode), desc=f'[test]') if use_tqdm else range(self.args.n_test_episode)
        for _ in pbar:
            # test each property
            info = self.test_episode()
            # print(info)
            infos.append(info)
            if use_tqdm:
                pbar.set_postfix(name=self, total=sum(info['value'] for info in infos), done=sum(info['done'] for info in infos), mode=self.args.mode)
        return infos

    def test_episode(self):
        # reset environment
        observation, done = self.env.reset()
        info = {'episode': self.env.episode, 'value': 0, 'done': done}
        if done:
            return info
        
        self.init_hook(observation)
        # step loop
        while not done:
            with torch.no_grad():
                action = self.select_action(observation)
            remaining_next_observation, reward, done, early_stop = self.env.step(action)
            if early_stop:
                break
            self.step_hook(remaining_next_observation)
            observation = remaining_next_observation
            # update info
            info['value'] += reward
        info['done'] = done
        info['early_stop'] = early_stop
        return info

    def train(self):
        self.args.mode = 'train'
        self.load()
        # test each property
        for epoch in range(self.args.n_train_epoch):
            # train
            episode_infos = []
            pbar = tqdm.tqdm(range(self.args.n_train_episode), desc=f'[train] {epoch=}/{self.args.n_train_epoch}')
            for episode in pbar:
                info = self.train_episode()
                pbar.set_postfix(info)
                episode_infos.append(info)
            
            # save checkpoint
            self.save()
            
            # update info
            epoch_info = {
                'epoch': epoch,
                'actor_loss': float(np.mean([info['actor_loss'] for info in episode_infos])),
                'critic_loss': float(np.mean([info['critic_loss'] for info in episode_infos])),
                'alpha_loss': float(np.mean([info['alpha_loss'] for info in episode_infos])),
                'early_stop': sum([info['early_stop'] for info in episode_infos]),
                'done': sum([info['done'] for info in episode_infos]),
                'total': len(episode_infos),
                'train_value': float(np.mean([info['value'] for info in episode_infos])),
            }
            print(epoch_info)
            if self.args.save_csv:
                df = pd.DataFrame([epoch_info])
                df.to_csv(
                    os.path.join(
                        self.args.csv_dir, 
                        self.args.mode, 
                        f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.csv'), 
                    index=False, 
                    mode='a',
                    header=False if os.path.exists(os.path.join(
                        self.args.csv_dir, 
                        self.args.mode, 
                        f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.csv')) else True
                )
                            

    def init_hook(self, observation):
        pass

    def step_hook(self, next_observation):
        pass

    def load(self):
        pass

    def save(self):
        raise NotImplementedError
