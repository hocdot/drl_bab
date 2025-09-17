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
            infos.append(info)
            if use_tqdm:
                pbar.set_postfix(name=self, total=sum(info['value'] for info in infos), done=sum(info['done'] for info in infos))
        return infos

    def test_episode(self):
        self.args.mode = 'test'
        # reset environment
        observation, done = self.env.reset()
        info = {'episode': self.env.episode, 'value': 0, 'done': done}
        self.init_hook(observation)
        # step loop
        while not done:
            with torch.no_grad():
                action = self.select_action(observation)
            next_observation, reward, done, step_info = self.env.step(action)
            self.step_hook(next_observation)
            observation = next_observation
            # update info
            info['value'] += reward
        info['done'] = done
        return info

    def train(self):
        self.args.mode = 'train'
        self.load()
        # test each property
        with open(os.path.join(self.args.csv_dir, f'{self.args.mode}_{self.args.dataset}_{self.args.solver}.csv'), 'w') as f:
            f.write('episode,actor_loss,critic_loss,alpha_loss,train_value,test_value\n')
            for epoch in range(self.args.n_train_epoch):
                train_infos = []
                # train
                pbar = tqdm.tqdm(range(self.args.n_train_episode), desc=f'[train] {epoch=}')
                for episode in pbar:
                    train_info = self.train_episode()
                    train_info['global_step'] = self.global_step
                    pbar.set_postfix(train_info)
                    
                    train_infos.append(train_info)
                
                # save checkpoint
                self.save()
                
                # test
                test_infos = self.test(use_tqdm=True)
                
                # update info
                info = {
                    'actor_loss': sum(info['actor_loss'] for info in train_infos) / self.args.n_train_episode,
                    'critic_loss': sum(info['critic_loss'] for info in train_infos) / self.args.n_train_episode,
                    'alpha_loss': sum(info['alpha_loss'] for info in train_infos) / self.args.n_train_episode,
                    'train_value': sum(info['value'] for info in train_infos),
                    'test_value': sum(info['value'] for info in test_infos)
                }
                print(info)
                
                # export csv
                print(f'{epoch},{info["actor_loss"]},{info["critic_loss"]},{info["alpha_loss"]},{info["train_value"]},{info["test_value"]}', file=f)
                
                            
        self.save()


    def init_hook(self, observation):
        pass

    def step_hook(self, next_observation):
        pass

    def load(self):
        pass

    def save(self):
        raise NotImplementedError
