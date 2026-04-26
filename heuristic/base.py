import pandas as pd
import torch
import time
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

    def test(self, use_tqdm=True, output_path=None):
        self.args.mode = 'test'
        self.load()
        infos = []
        pbar = tqdm.tqdm(range(self.args.n_test_instance), desc=f'[test]') if use_tqdm else range(self.args.n_test_instance)
        print(f'{output_path=}')
        for _ in pbar:
            # test each property
            info = self.test_instance()
            infos.append(info)
            if use_tqdm:
                pbar.set_postfix(name=self, total=sum(info['value'] for info in infos), done=sum(info['done'] for info in infos), mode=self.args.mode)
            self.export_csv(info, path=output_path)
        return infos

    def test_instance(self):
        # reset environment
        start_time = time.time()
        
        observation, done = self.env.reset()
        
        info = {
            'episode': self.env.episode, 
            'value': 0, 
            'done': done, 
            'decision_time': 0.0, 
            'iteration': 0,
            'total_time': 0.0, 
            'active_neuron': observation.shape[1] if observation is not None else 0,
        }

        if done:
            return info
        
        # initialize
        self.init_hook(observation)
        
        # step loop
        while not done:
            if time.time() - start_time > self.args.max_runtime:
                break
            
            try:
                with torch.no_grad():
                    decision_start_time = time.time()
                    action = self.select_action(observation)
                    decision_time = time.time() - decision_start_time
                    info['decision_time'] += decision_time
                    info['iteration'] += 1
                remaining_next_observation, reward, done, early_stop, _ = self.env.step(action)
            except:
                import traceback; traceback.print_exc()
                early_stop = True
                
            if early_stop:
                break
            self.step_hook(remaining_next_observation)
            observation = remaining_next_observation
            # update info
            info['value'] += reward
        info['done'] = done
        # info['early_stop'] = early_stop
        info['total_time'] = time.time() - start_time

        return info

    def train(self):
        self.args.mode = 'train'
        if self.args.load_checkpoint:
            self.load()
            
        meter_metrics = ['done', 'early_stop', 'alpha_loss', 'actor_loss', 'critic_loss', 'value']
        for epoch in range(self.args.n_train_epoch):
            # initialize meters
            meters = {k: AvgMeter(k) for k in meter_metrics}
            pbar = tqdm.tqdm(range(self.args.n_train_instance), desc=f'[train] {epoch=}/{self.args.n_train_epoch}')
            for episode in pbar:
                info = self.train_instance()
                # update meters
                [meters[k].update(info[k], count=1) for k in meter_metrics]
                pbar.set_postfix({
                    'global_step': self.global_step,
                    'current_value': info['value'],
                    'buffer': len(self.replay_buffer),
                    **{k: v.avg for k, v in meters.items()},
                })
                
                if episode > 0 and episode % self.args.n_export_csv == 0:
                    epoch_info = {
                        'epoch': epoch,
                        'total': len(pbar),
                        'step': self.global_step,
                        **{k: v.avg for k, v in meters.items()},
                    }
                    self.export_csv(epoch_info)
            
            # save checkpoint
            self.save()
            
            # update info
            epoch_info = {
                'epoch': epoch,
                'total': len(pbar),
                'step': self.global_step,
                **{k: v.avg for k, v in meters.items()},
            }
            print(epoch_info)
            self.export_csv(epoch_info)
            
    def export_csv(self, epoch_info, path=None):
        if self.args.save_csv:
            if path is None:
                path = os.path.join(
                    self.args.csv_dir, 
                    self.args.mode, 
                    f'{self.args.dataset}_{self.args.solver}_{self.args.enc_dec_type}_{self.args.n_enc_dec_layer}.csv'
                ) 
            df = pd.DataFrame([epoch_info])
            df.to_csv(
                path, 
                index=False, 
                mode='w' if not os.path.exists(path) else 'a',
                header=not os.path.exists(path)
            )
                            

    def init_hook(self, observation):
        pass

    def step_hook(self, next_observation):
        pass

    def load(self):
        pass

    def save(self):
        raise NotImplementedError

    def solve_one_instance(self, onnx, vnnlib):
        assert self.args.mode == 'test', f'{self.args.mode=}'
        self.load()
        self.env.loader = iter([{'onnx': [onnx], 'vnnlib': [vnnlib]}])
        info = self.test_instance()
        return info