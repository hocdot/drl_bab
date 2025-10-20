import pandas as pd
import numpy as np
import torch
import tqdm
import os

from drl.env import Env
from .meter import AvgMeter

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
        pbar = tqdm.tqdm(range(self.args.n_test_instance), desc=f'[test]') if use_tqdm else range(self.args.n_test_instance)
        for _ in pbar:
            # test each property
            info = self.test_instance()
            # print(info)
            infos.append(info)
            if use_tqdm:
                pbar.set_postfix(name=self, total=sum(info['value'] for info in infos), done=sum(info['done'] for info in infos), mode=self.args.mode)
        return infos

    def test_instance(self):
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
            remaining_next_observation, reward, done, early_stop, _ = self.env.step(action)
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
        if self.args.load_checkpoint:
            self.load()
            
        for epoch in range(self.args.n_train_epoch):
            actor_loss_meter = AvgMeter('actor_loss')
            critic_loss_meter = AvgMeter('critic_loss')
            alpha_loss_meter = AvgMeter('alpha_loss')
            value_meter = AvgMeter('value')
            episode_infos = []
            pbar = tqdm.tqdm(range(self.args.n_train_instance), desc=f'[train] {epoch=}/{self.args.n_train_epoch}')
            for episode in pbar:
                info = self.train_instance()
                episode_infos.append(info)
                
                actor_loss_meter.update(info['actor_loss'])
                critic_loss_meter.update(info['critic_loss'])
                alpha_loss_meter.update(info['alpha_loss'])
                value_meter.update(info['value'])
                pbar.set_postfix({
                    'actor_loss': actor_loss_meter.avg,
                    'critic_loss': critic_loss_meter.avg,
                    'alpha_loss': alpha_loss_meter.avg,
                    'value': value_meter.avg,
                    'global_step': self.global_step,
                    'buffer': len(self.replay_buffer),
                })
            
            # save checkpoint
            self.save()
            
            # update info
            epoch_info = {
                'epoch': epoch,
                'actor_loss': actor_loss_meter.avg,
                'critic_loss': critic_loss_meter.avg,
                'alpha_loss': alpha_loss_meter.avg,
                'early_stop': sum([info['early_stop'] for info in episode_infos]),
                'done': sum([info['done'] for info in episode_infos]),
                'total': len(pbar),
                'train_value': value_meter.avg,
            }
            print(epoch_info)
            if self.args.save_csv:
                path = os.path.join(
                    self.args.csv_dir, 
                    self.args.mode, 
                    f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.csv'
                ) 
                df = pd.DataFrame([epoch_info])
                df.to_csv(
                    path, 
                    index=False, 
                    mode='w' if epoch == 0 else 'a',
                    header=epoch == 0
                )
                            

    def init_hook(self, observation):
        pass

    def step_hook(self, next_observation):
        pass

    def load(self):
        pass

    def save(self):
        raise NotImplementedError
