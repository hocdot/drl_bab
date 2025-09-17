import torch
import tqdm

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
                pbar.set_postfix(name=self, total=sum(info['value'] for info in infos), done=sum(info['done'] for info in infos), mode=self.args.mode)
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
        train_infos = []
        for epoch in range(self.args.n_train_epoch):
            # train
            episode_infos = []
            pbar = tqdm.tqdm(range(self.args.n_train_episode), desc=f'[train] {epoch=}')
            for episode in pbar:
                info = self.train_episode()
                pbar.set_postfix(info)
                episode_infos.append(info)
            
            # save checkpoint
            self.save()
            
            # test
            test_infos = self.test(use_tqdm=True)
            
            # update info
            epoch_info = {
                'epoch': epoch,
                'actor_loss': sum(info['actor_loss'] for info in episode_infos),
                'critic_loss': sum(info['critic_loss'] for info in episode_infos),
                'alpha_loss': sum(info['alpha_loss'] for info in episode_infos),
                'train_value': sum(info['value'] for info in episode_infos),
                'test_value': sum(info['value'] for info in test_infos),
            }
            print(epoch_info)
            train_infos.append(epoch_info)
        return train_infos
                            

    def init_hook(self, observation):
        pass

    def step_hook(self, next_observation):
        pass

    def load(self):
        pass

    def save(self):
        raise NotImplementedError
