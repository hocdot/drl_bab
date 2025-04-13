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
                                     lr=args.dqn_lr,
                                     amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        #
        self.onnx_to_id = get_onnx_to_id(args)

    # =========================================================
    # PRETRAIN
    # =========================================================
    def train_offline_add_info(self, actor_loss, critic_loss):
        if critic_loss < self.min_critic_loss:
            self.save()
        self.min_critic_loss = min(critic_loss, self.min_critic_loss)
        self.max_critic_loss = max(critic_loss, self.max_critic_loss)
        info = {
            'step': self.step,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'min_critic_loss': self.min_critic_loss,
            'max_critic_loss': self.max_critic_loss,
        }
        return info

    def train_offline_load_buffer(self):
        # extract args
        args = self.args
        on   = self.observation_normalizer
        # rn   = self.reward_normalizer
        #
        self.train_offline_buffer = OfflineBuffer(args.buffer_size)
        n_visited = 0
        d = os.path.join(args.data_dir, 'train')
        for name in tqdm.tqdm(list(sorted(set(os.listdir(d)))), desc='train_offline load buffer'):
            # TODO: remove
            # instance_id, iter_id = name.split('.')[0].split('_')[-2:]
            # if int(instance_id) >= 1:
            #     continue

            path = os.path.join(d, name)
            if args.verbose:
                print(f'[+] Loading {path=}')
            with open(path, 'rb') as fp:
                buffer = pickle.load(fp)
                n_visited += len(buffer)
                for i, transition in enumerate(buffer.memory):
                    self.train_offline_buffer.push(
                        transition.observation,
                        transition.unpruned_next_observation_1,
                        transition.unpruned_next_observation_2,
                        transition.mask,
                        transition.reward,
                        transition.onnx_id,
                        transition.done,
                    )
                    on.update(transition.observation.detach().cpu().numpy())
                    # print(transition.observation[..., -1])
                    # indices = torch.where(transition.mask)
                    # reward = reward[indices].detach().cpu().numpy()[:, None]
                    # rn.update(reward)
                print(f'[+] loaded {name} +{len(buffer)} new transitions')
        print(f'[+] loaded {n_visited=}')

    def train_offline_step(self):
        # extract args
        args             = self.args
        env              = self.env
        buffer           = self.train_offline_buffer
        actor            = self.actor
        actor_optimizer  = self.actor_optimizer
        critic1          = self.critic1
        critic2          = self.critic2
        embedding        = self.embedding
        critic1_target   = self.critic1_target
        critic2_target   = self.critic2_target
        critic_optimizer = self.critic_optimizer
        on               = self.observation_normalizer
        # rn               = self.reward_normalizer

        actor.train()
        critic1.train()
        critic2.train()
        critic1_target.train()
        critic2_target.train()
        embedding.train()

        critic_loss = 0
        for _ in range(args.n_delay_gradient):
            # =====================
            # extract batch of data
            # =====================
            transitions             = buffer.sample(args.batch_size)
            batch                   = OfflineTransition(*zip(*transitions))
            observation             = torch.stack(batch.observation).to(args.device)
            observation             = on.normalize_torch(observation)
            # augment observation
            mask                    = torch.stack(batch.mask).to(args.device)
            reward                  = torch.stack(batch.reward).to(args.device)
            onnx_id                 = batch.onnx_id[0].to(args.device)
            done                    = torch.tensor(batch.done).to(args.device)[:, None, None]
            # reward                  = rn.normalize_torch(reward)

            uno_1                   = on.normalize_torch(torch.stack(batch.unpruned_next_observation_1).to(args.device))
            uno_2                   = on.normalize_torch(torch.stack(batch.unpruned_next_observation_2).to(args.device))
            edge_index, edge_weight = self.edge_indices[onnx_id], self.edge_weights[onnx_id]

            # ============
            # train critic
            # ============
            # compute target
            bs = reward.shape[0]
            nes_1 = embedding(uno_1, edge_index, edge_weight) # next embeddings 1
            nes_2 = embedding(uno_2, edge_index, edge_weight) # next embeddings 2
            with torch.no_grad():
                target_q_value1        = 0.5 * (critic1_target(nes_1, edge_index, edge_weight).amax(dim=(1, 2), keepdims=True) + \
                                                critic1_target(nes_2, edge_index, edge_weight).amax(dim=(1, 2), keepdims=True))
                target_q_value2        = 0.5 * (critic2_target(nes_1, edge_index, edge_weight).amax(dim=(1, 2), keepdims=True) + \
                                                critic2_target(nes_2, edge_index, edge_weight).amax(dim=(1, 2), keepdims=True))
                min_max_target_q_value = torch.min(target_q_value1, target_q_value2)
                next_q_value = reward + args.gamma * min_max_target_q_value # * (1 - done) # Trick
            # compute estimate
            es            = embedding(observation, edge_index, edge_weight)
            q_value1      = critic1(es, edge_index, edge_weight)
            q_value2      = critic2(es, edge_index, edge_weight)
            critic1_loss  = F.mse_loss(q_value1 * mask, next_q_value * mask)
            critic2_loss  = F.mse_loss(q_value2 * mask, next_q_value * mask)
            critic_loss  += (critic1_loss + critic2_loss)
        critic_loss /= args.n_delay_gradient
        # optimize the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        print()

        total_actor_loss = 0
        # if self.step > 2000:
#             # ===========
#             # train actor
#             # ===========
#             for _ in range(10):
#                 # compute actor score
#                 score = actor(observation, edge_index, edge_weight)
#                 # print(f'{score.max().item()=} {score.min().item()=}')
#                 old_shape = score.shape
#                 # bs, n_layer, n_neuron = score.shape
#                 score = score.flatten(1)
#                 prob = F.softmax(score).view(old_shape)
#                 # compute full feedback from critic
#                 with torch.no_grad():
#                     q_value1 = critic1(observation, edge_index, edge_weight)
#                     q_value2 = critic2(observation, edge_index, edge_weight)
#                     min_q_value = torch.min(q_value1, q_value2)
#                 actor_loss = - (min_q_value * prob).sum(dim=(1, 2)).mean()
#                 # optimize actor
#                 actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 actor_optimizer.step()
#                 total_actor_loss += actor_loss.item()
#             total_actor_loss = total_actor_loss / 10.0

        # ===============================
        # update target_network of critic
        # ===============================
        if self.step % args.target_update_interval == 0:
            for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        # report data
        return total_actor_loss, critic_loss.item()

    def train_offline(self):
        # extract args
        args      = self.args
        monitor   = self.monitor
        actor     = self.actor
        critic    = self.critic1
        embedding = self.embedding
        on        = self.observation_normalizer
        # rn      = self.reward_normalizer
        # load full feedback buffer
        self.train_offline_load_buffer()
        # load pretrained paths
        paths = self.get_pretrained_paths()
        # init
        self.min_critic_loss = torch.inf
        self.max_critic_loss = -torch.inf

        # load train_offline actor/critic model
        if args.load_state_dict:
            path = paths['critic']
            print(f'[+] loaded {path=}')
            self.critic1.load_state_dict(torch.load(path))
            self.critic2.load_state_dict(torch.load(path))
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())

            path = paths['actor']
            self.actor.load_state_dict(torch.load(path))
            print(f'[+] loaded {path=}')

            path = paths['embedding']
            self.embedding.load_state_dict(torch.load(path))
            print(f'[+] loaded {path=}')
        # optimization loop
        try:
            self.step = 0
            for _ in range(args.n_train_offline_step):
                actor_loss, critic_loss = self.train_offline_step()
                # add more info
                info = self.train_offline_add_info(actor_loss, critic_loss)
                # logging
                monitor.step(info)
                #
                self.step += 1
                # export csv
                monitor.export_csv()
        except KeyboardInterrupt:
            pass
        finally:
            # self.save()
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
        if args.verbose:
            # print(f'{action=}')
            pass
        return np.array(action)


    def test(self):
        super().test()
