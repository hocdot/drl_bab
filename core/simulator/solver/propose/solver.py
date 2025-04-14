import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import torch
import tqdm
import os

from .buffer import OfflineBuffer, OfflineTransition
from .normalizer import RunningMeanStd
from .model import DQN

from ...util import CudaDefault
from ..base import BaseSolver

# np.set_printoptions(precision=2)

def get_onnx_to_id(args):
    path = os.path.join(args.benchmark_dir, args.benchmark, 'train.csv')
    onnx_to_id = {}
    i = 0
    for _, (onnx, _) in pd.read_csv(path).iterrows():
        if onnx not in onnx_to_id:
            onnx_to_id[onnx] = i
        i += 1
    return onnx_to_id

class Solver(BaseSolver):

    def __init__(self, args): # share
        super().__init__(args)
        # initialize model
        self.initialize_model()

    def initialize_model(self):
        # extract args
        args = self.args
        env  = self.env
        # initialize critic
        self.policy_net = DQN(args).to(args.device)
        self.target_net = DQN(args).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=args.lr,
                                     amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        #
        self.onnx_to_id = get_onnx_to_id(args)

    # =========================================================
    # PRETRAIN
    # =========================================================
    def train_offline_add_info(self, loss):
        if loss < self.min_loss:
            self.save()
        self.min_loss = min(loss, self.min_loss)
        self.max_loss = max(loss, self.max_loss)
        info = {
            'step'     : self.step,
            'loss'     : loss,
            'min_loss' : self.min_loss,
            'max_loss' : self.max_loss,
        }
        return info

    def train_offline_load_buffer(self):
        # extract args
        args = self.args
        #
        self.buffer = OfflineBuffer(args.buffer_size)
        n_visited = 0
        d = os.path.join(args.data_dir, 'train')
        for name in tqdm.tqdm(list(sorted(set(os.listdir(d)))), desc='train_offline load buffer'):
            path = os.path.join(d, name)
            if args.verbose:
                print(f'[+] Loading {path=}')
            with open(path, 'rb') as fp:
                buffer = pickle.load(fp)
                n_visited += len(buffer)
                for i, transition in enumerate(buffer.memory):
                    self.buffer.push(
                        transition.feature,
                        transition.active_neuron_index,
                        transition.unf_1,
                        transition.unf_2,
                        transition.actions,
                        transition.rewards,
                        transition.onnx_id,
                        transition.done,
                    )
                print(f'[+] loaded {name} +{len(buffer)} new transitions')
        print(f'[+] loaded {n_visited=}')

    def soft_target_update(self):
        # extract args
        args       = self.args
        policy_net = self.policy_net
        target_net = self.target_net
        # update
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + \
                                         target_net_state_dict[key] * (1 - args.tau)
        target_net.load_state_dict(target_net_state_dict)

    def train_offline_step(self):
        # extract args
        args       = self.args
        env        = self.env
        buffer     = self.buffer
        criterion  = self.criterion
        optimizer  = self.optimizer
        policy_net = self.policy_net
        target_net = self.target_net
        policy_net.train()

        with CudaDefault():
            loss = 0
            for _ in range(args.batch_size):
                # extract data
                transition = buffer.sample(1)[0]
                feature             = transition.feature             # [layer (bs, n_active_neuron), ...]
                active_neuron_index = transition.active_neuron_index
                unf_1               = transition.unf_1
                unf_2               = transition.unf_2
                actions             = transition.actions
                rewards             = transition.rewards
                done                = transition.done
                n_layer             = len(feature)
                # =========
                # train DQN
                # =========
                # compute estimate
                q_value      = policy_net(feature) # [layer (bs, n_active_neuron), ...]
                # compute target
                next_q_value_1 = target_net(unf_1)
                next_q_value_2 = target_net(unf_2)
                next_q_value = [0.5 * (next_q_value_1[layer] + next_q_value_2[layer]) for layer in range(n_layer)]
                # compute expected_q_value and loss per layer
                expected_q_value = [torch.zeros_like(q_value[layer]) for layer in range(n_layer)]
                for layer in range(n_layer):
                    bs, nan = q_value[layer].shape
                    for b in range(bs):
                        mask    = torch.isin(active_neuron_index[layer][b], actions[layer][b])
                        indices = torch.nonzero(mask, as_tuple=True)[0]
                        expected_q_value[layer][b][indices] = rewards[layer][b]
                        if not done:
                            # print(next_q_value[layer].max(-1, keepdim=True).values.shape)
                            # print(expected_q_value[layer].shape)
                            expected_q_value[layer] += args.gamma * next_q_value[layer].max(-1, keepdim=True).values
                    print(layer, feature[layer].mean(), q_value[layer].mean().item(), expected_q_value[layer].mean().item())
#                     loss += criterion(q_value[layer], expected_q_value[layer]) / (bs * nan)
#             loss /= n_layer * args.batch_size
#             # optimize the critic
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        # report data
        return loss.item()

    def train_offline(self):
        # extract args
        args       = self.args
        monitor    = self.monitor
        policy_net = self.policy_net

        # load buffer
        self.train_offline_load_buffer()

        # init
        self.min_loss = torch.inf
        self.max_loss = -torch.inf

        # load dqn model
        if args.load_state_dict:
            path = os.path.join(args.model_dir, f'{monitor.label}.pkl')
            print(f'[+] loaded {path=}')
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        # optimization loop
        try:
            self.step = 0
            for _ in range(args.n_train_offline_step):
                loss = self.train_offline_step()
                self.soft_target_update()
                # add more info
                info = self.train_offline_add_info(loss)
                # logging
                monitor.step(info)
                #
                self.step += 1
                # export csv
                monitor.export_csv()
        except KeyboardInterrupt:
            pass
    # =========================================================

    # =========================================================
    # TEST
    # =========================================================
    def select_action(self, observation):
        # extract args
        args       = self.args
        policy_net = self.policy_net
        env        = self.env
        policy_net.eval()
        # ===================
        # extract observation
        # ===================
        # extract data
        # feature: list of Layer_i: [bs, n_active_neuron_i, n_feature]
        feature, active_neuron_index, n_active_neuron = observation
        n_layer = len(feature)
        bs      = feature[0].shape[0]
        # extract
        scores = policy_net(feature)
        # initialize action, action.shape = (bs, 2)
        best_scores, best_neuron_indices = [], []
        for layer in range(n_layer):
            score                  = scores[layer].cpu() # bs, n_active_neuron_i
            print(f'{score.shape=}')
            best_score, best_index = torch.max(score, dim=1)
            best_neuron_index      = torch.tensor([active_neuron_index[layer][b, best_index[b]] for b in range(bs)])
            best_scores.append(best_score)
            best_neuron_indices.append(best_neuron_index)
        best_scores         = torch.stack(best_scores).T         # (bs, n_layer)
        best_neuron_indices = torch.stack(best_neuron_indices).T # (bs, n_layer)
        best_layer          = torch.argmax(best_scores, dim=1)   # (bs, )
        best_neuron         = torch.gather(best_neuron_indices, 1, best_layer.unsqueeze(-1)).squeeze(-1)
        action = np.array([[best_layer[i].item(), best_neuron[i].item()] for i in range(bs)])
        return np.array(action)

    def test(self):
        self.load()
        super().test()

    def save(self):
        args = self.args
        state_dict = self.policy_net.state_dict()
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        with open(path, 'wb') as fp:
            pickle.dump(state_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as fp:
                state_dict = pickle.load(fp)
                self.policy_net.load_state_dict(state_dict)
                self.policy_net.eval()
