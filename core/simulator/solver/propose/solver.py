import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import torch
import tqdm
import os

from .buffer import OfflineBuffer, OfflineTransition
from .normalizer import RunningMeanStd
from .embedding import Embedding
from .critic import Critic
from .actor import Actor

from ..base import BaseSolver

np.set_printoptions(precision=2)

def get_onnx_to_id(args):
    path = os.path.join(args.onnx_dir, 'train_instance.csv')
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
        # initialize actor
        self.actor = Actor(args).to(args.device)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()),
                                          lr=args.actor_lr)
        # initialize critic
        self.critic1   = Critic(args).to(args.device)
        self.critic2   = Critic(args).to(args.device)
        # initialize embedding
        self.embedding = Embedding(args).to(args.device)
        if 'train' in args.mode:
            self.critic1_target = Critic(args).to(args.device)
            self.critic2_target = Critic(args).to(args.device)
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())
        # initialize optimizer
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + \
                                           list(self.critic2.parameters()),
                                           lr=args.critic_lr)
        #
        self.onnx_to_id = get_onnx_to_id(args)
        self.edge_indices, self.edge_weights = [], []
        for onnx in self.onnx_to_id:
            self.edge_indices.append(torch.load(os.path.join(args.onnx_dir, f'{onnx}.index.pth')))
            self.edge_weights.append(torch.load(os.path.join(args.onnx_dir, f'{onnx}.weight.pth')))
        # normalizer
        self.observation_normalizer = RunningMeanStd(shape=[args.n_observation])
        # self.reward_normalizer      = RunningMeanStd(shape=[1])

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
            # instance_id, iter_id = name.split('.')[0].split('_')[-2:]
            # if int(instance_id) < 1 and int(iter_id) <= 3: # TODO: remove this
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
        assert not torch.isnan(observation).any(), f'{torch.where(observation.isnan())=}'
        # observation: [bs, n_layer, n_neuron, n_feature]
        # extract data
        # extract args
        args      = self.args
        critic    = self.critic1
        embedding = self.embedding
        actor     = self.actor
        env       = self.env
        on        = self.observation_normalizer
        actor.eval()
        critic.eval()
        embedding.eval()
        # rn     = self.reward_normalizer
        edge_index, edge_weight = env.edge_index, env.edge_weight
        # extract bs
        bs          = observation.shape[0]
        mask        = observation[..., -1].clone()
        observation = on.normalize_torch(observation)
        assert not observation.isnan().any()

        es          = embedding(observation, edge_index, edge_weight)
        assert not es.isnan().any()
        # combined score
        def combined(observation, edge_index, edge_weight):
            # compute score via actor
            actor_mask = torch.zeros_like(mask)
            actor_score = actor(es) # bs, n_layer, n_neuron -> mask: bs, n_layer, n_neuron
            # compute mask by actor
            for i in range(bs):
                threshold = torch.sort(actor_score[i].flatten(), descending=True).values[args.top_k_actor]
                actor_mask[i, ...] = (actor_score[i] > threshold).to(actor_mask)
            # compute critic score for only mask*actor_mask==1
            critic_score = critic(es, edge_index, edge_weight)
            critic_score = (actor_mask * mask) * critic_score + (1 - actor_mask * mask) * args.iwb_threshold
            return critic_score
        # compute score
        score = eval(args.score)(es, edge_index, edge_weight)
        assert not score.isnan().any()
        score    = mask * score + (1 - mask) * (-999)
#         # compute full feedback
#         min_rewards, max_rewards, actions = env.full_feedback()
#         # convert score to env.full_feedback format
#         scores = []
#         for i, action in enumerate(actions): # subproblem
#             s = []
#             for j, a in enumerate(action):   # active neuron
#                 # print(i, j, a, score[i, a[0], a[1]])
#                 s.append(score[i, a[0], a[1]].item())
#             scores.append(np.array(s))
#         # for each subproblem, filter top_k_actor neuron by score
#         # then select best neuron by full feedback
#         ret_action = []
#         for i in range(bs):
#             indices = np.argsort(scores[i])[::-1][:args.top_k_actor]
#             indices = torch.tensor(indices.copy())
#             filtered_rewards = rewards[i][indices]
#             filtered_action  = actions[i][indices.detach().cpu().numpy()]
#             idx = torch.argmax(filtered_rewards).detach().cpu().numpy()
#             a = filtered_action[idx]
#             ret_action.append(a)
        # initialize action
        ret_action = []
        for i in range(bs):
            candidates = torch.where(score[i] == score[i].max()) # .tolist()
            # print(f'{bs=} {candidates=} {score[i].max()=}')
            index      = np.random.choice(np.arange(len(candidates[0])))
            a          = [_[index].item() for _ in candidates]
            ret_action.append(a)
        if args.verbose:
            # print(f'{action=}')
            pass
        # for here to eval
        return np.array(ret_action)

    def get_pretrained_paths(self):
        # extract args
        args = self.args

        paths = {
            'critic'   : os.path.join(args.model_dir, f'critic_{args.onnx}_{args.solver}_{args.gnn_class}.pth'),
            'actor'    : os.path.join(args.model_dir, f'actor_{args.onnx}_{args.solver}_{args.gnn_class}.pth'),
            'embedding': os.path.join(args.model_dir, f'embed_{args.onnx}_{args.solver}_{args.embed_class}.pth'),
            'on'       : os.path.join(args.model_dir, f'on_{args.onnx}_{args.solver}_{args.gnn_class}.pkl'),
            'rn'       : os.path.join(args.model_dir, f'rn_{args.onnx}_{args.solver}_{args.gnn_class}.pkl')
        }
        return paths

    def test(self):
        # load pretrained paths
        paths = self.get_pretrained_paths()

        # critic
        path = paths['critic']
        self.critic1.load_state_dict(torch.load(path, weights_only=True))
        self.critic1.eval()
        print(f'[+] loaded {path=}')

        # actor
        path = paths['actor']
        self.actor.load_state_dict(torch.load(path, weights_only=True))
        self.actor.eval()
        print(f'[+] loaded {path=}')

        # embedding
        path = paths['embedding']
        self.embedding.load_state_dict(torch.load(path, weights_only=True))
        self.embedding.eval()
        print(f'[+] loaded {path=}')

        # normalizer
        path = paths['on']
        with open(path, 'rb') as fp:
            self.observation_normalizer = pickle.load(fp)
        print(f'[+] loaded {path=}')

#         path = paths['rn']
#         with open(path, 'rb') as fp:
#             self.reward_normalizer = pickle.load(fp)
#             print(f'[+] loaded {path=}')

        super().test()

    def save(self):
        # load pretrained paths
        paths = self.get_pretrained_paths()
        # save train_offline actor/critic model
        path = paths['critic']
        torch.save(self.critic1.state_dict(), path)
        print(f'[+] saved {path=}')

        path = paths['actor']
        torch.save(self.actor.state_dict(), path)
        # print(f'[+] saved {path=}')

        path = paths['embedding']
        torch.save(self.embedding.state_dict(), path)
        # print(f'[+] saved {path=}')

        # save normalizer
        path = paths['on']
        with open(path, 'wb') as fp:
            pickle.dump(self.observation_normalizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f'[+] saved {path=}')

#             path = paths['rn']
#             with open(path, 'wb') as fp:
#                 pickle.dump(rn, fp, protocol=pickle.HIGHEST_PROTOCOL)
#                 print(f'[+] saved {path=}')
