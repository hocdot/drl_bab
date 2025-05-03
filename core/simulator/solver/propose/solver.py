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
        # TODO: remove this hard code
        if args.mode == 'train_offline':
            env.reset()
#         # TODO: load all graph, remove hardcode
#         if args.mode == 'train_offline':
#             env.onnx = 'cifar10_2_255_simplified.onnx'
#             env.vnnlib = 'cifar10_spec_idx_4_eps_0.00784_n1.vnnlib'
#             env.load_graph()
        # initialize critic
        self.policy_net = DQN(args).to(args.device)
        self.target_net = DQN(args).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=args.lr,
                                     amsgrad=True)
        self.criterion = nn.SmoothL1Loss(reduction='sum')
        # feature normalizer
        self.feature_normalizer = RunningMeanStd(shape=args.n_feature)
        self.reward_normalizer      = RunningMeanStd(shape=1)
        #
        self.onnx_to_id = get_onnx_to_id(args)

    # =========================================================
    # PRETRAIN
    # =========================================================
    def train_offline_add_info(self, loss):
        if loss < self.min_loss or (self.step % self.args.n_save_interval == 0):
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
                        transition.n_active_neuron,
                        transition.unf_1,
                        transition.unf_2,
                        transition.actions,
                        transition.rewards,
                        transition.onnx_id,
                        transition.done,
                    )
                    # TODO: remove this
                    break
                print(f'[+] loaded {name} +{i+1} new transitions')
                n_layer = len(transition.feature)
                for layer in range(n_layer):
                    self.feature_normalizer.update(transition.feature[layer].detach().cpu().numpy())
                    self.reward_normalizer.update(transition.rewards[layer].unsqueeze(-1).detach().cpu().numpy())
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
        f_norm     = self.feature_normalizer
        r_norm     = self.reward_normalizer
        policy_net.train()

        with CudaDefault():
            loss = 0.
            total_n_active_neuron = 0
            for _ in range(args.batch_size):
                # extract data
                transition          = buffer.sample(1)[0] # TODO: update this
                feature             = transition.feature # [layer (n_subproblem, n_active_neuron, n_feature), ...]
                unf_1               = transition.unf_1
                unf_2               = transition.unf_2
                # actions             = transition.actions
                active_neuron_index = transition.active_neuron_index
                n_active_neuron     = transition.n_active_neuron
                rewards             = transition.rewards
                done                = transition.done
                n_layer             = len(feature)
                # extract a random subproblem
                norm_feature = []
                norm_rewards = []
                for layer in range(n_layer):
                    n_subproblem, _, n_feature = feature[layer].shape
                    p = np.random.randint(low=0, high=n_subproblem)
                    norm_feature.append(f_norm.normalize_torch(feature[layer].to(args.device))[p].unsqueeze(0))
                    norm_rewards.append(r_norm.normalize_torch(rewards[layer].to(args.device))[p].unsqueeze(0))
                feature = norm_feature
                rewards = norm_rewards
                # =========
                # train DQN
                # =========
                # compute target
                with torch.no_grad():
                    # TODO next active neuron index
                    # uno_1 = (unf_1, None,  active_neuron_index, n_active_neuron)
                    # uno_2 = (unf_2, None,  active_neuron_index, n_active_neuron)
                    # next_q_value_1 = target_net(uno_1)
                    # next_q_value_2 = target_net(uno_2)
                    # next_q_value = [0.5 * (next_q_value_1[layer] + next_q_value_2[layer]) for layer in range(n_layer)]
                    expected_q_value = rewards
                # compute expected_q_value and loss per layer
                observation = (feature, None,  active_neuron_index, n_active_neuron)
                q_value = policy_net(observation, env) # [layer (n_subproblem, n_active_neuron), ...]
                for layer in range(n_layer):
                    # if not done:
                    #     expected_q_value[layer] += args.gamma * next_q_value[layer].max(-1, keepdim=True).values
                    _, n_neuron, _ = feature[layer].shape
                    loss += criterion(q_value[layer].to(args.device), expected_q_value[layer].to(args.device))
                    total_n_active_neuron += n_neuron
            loss /= total_n_active_neuron
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        args        = self.args
        policy_net  = self.policy_net
        env         = self.env
        f_norm      = self.feature_normalizer

        policy_net.eval().to(args.device)
        # ===================
        # extract observation
        # ===================
        # extract data
        # feature: list of Layer_i: [bs, n_active_neuron_i, n_feature]
        feature, heuristic_score, active_neuron_index, n_active_neuron = observation
        feature = [f_norm.normalize_torch(f.to(args.device)) for f in feature]
        active_neuron_index = [ani.to(args.device) for ani in active_neuron_index]
        observation = feature, heuristic_score, active_neuron_index, n_active_neuron
        # feature = [f.to(args.device) for f in feature]
        n_layer = len(feature)
        bs      = feature[0].shape[0]
        # extract
        scores = policy_net(observation, env)
        # initialize action, action.shape = (bs, 2)
        best_scores, best_neuron_indices = [], []
        for layer in range(n_layer):
            score                  = scores[layer].cpu() # bs, n_active_neuron_i
            # print(layer, score.mean().item(), score.max().item(), score.min().item())
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
        # save model
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        data = {'state_dict': state_dict,
                'f_norm': self.feature_normalizer,
                'r_norm': self.reward_normalizer,
        }
        with open(path, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'[+] saved at {self.step=} {path=}')

    def load(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        if os.path.exists(path):
            print(f'[+] loading {path}')
            with open(path, 'rb') as fp:
                data = pickle.load(fp)
            self.policy_net.load_state_dict(data['state_dict'])
            self.policy_net.eval()
            self.feature_normalizer = data['f_norm']
            self.reward_normalizer  = data['r_norm']
