import gymnasium as gym
import neuralsat as ns
import pandas as pd
import numpy as np
import torch
import os

class Env(gym.Env):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # load instance
        self.load_instances()

    def load_instances(self):
        # extract args
        args = self.args
        # load list of instances
        if 'train' in args.mode:
            path = os.path.join(args.onnx_dir, 'train_instance.csv')
        elif args.mode == 'test':
            path = os.path.join(args.onnx_dir, 'test_instance.csv')
            # path = os.path.join(args.onnx_dir, 'train_instance.csv')
        else:
            raise NotImplementedError
        # loading
        self.instance_df = pd.read_csv(path)
        self.instance_id = 0
        self.loaded_onnx = None

    def load_next_instance(self):
        # extract args
        args = self.args
        # reset neuralsat setting
        ns.test.reset_settings()
        #
        n_instance  = len(self.instance_df)
        self.onnx   = self.instance_df.loc[self.instance_id % n_instance, 'onnx']
        self.vnnlib = self.instance_df.loc[self.instance_id % n_instance, 'vnnlib']
        #
        if self.onnx != self.loaded_onnx:
            d = os.path.join(args.onnx_dir, self.onnx.replace('.onnx', ''))
            path = os.path.join(args.onnx_dir, f'{self.onnx}.index.pth')
            self.edge_index = torch.load(path)
            path = os.path.join(args.onnx_dir, f'{self.onnx}.weight.pth')
            self.edge_weight = torch.load(path)
            self.loaded_onnx = self.onnx
        if args.verbose:
            print(f'[+] loading new instance: {self.onnx=} {self.vnnlib=}')
        # load environment
        # TODO: extract moi objectives thoi ma khong phai extract lai net
        d           = os.path.join(args.onnx_dir, self.onnx.replace('.onnx', ''))
        onnx_path   = os.path.join(d, 'onnx', self.onnx)
        vnnlib_path = os.path.join(d, 'vnnlib', self.vnnlib)
        net, input_shape, objectives = ns.test.extract_instance(onnx_path, vnnlib_path)
        self.net           = net
        self.input_shape   = input_shape
        self.iv            = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
        objectives, _      = self.iv._preprocess(objectives, force_split=None)
        self.objective     = objectives.pop(1)
        # increase instance id counter
        self.instance_id += 1

    def reset(self):
        # extract args
        args = self.args
        # load next instance if there is no objective left for current instance
        self.load_next_instance()
        done = self.iv.init(objective=self.objective, reference_bounds=None, preconditions={})
        self.initial_worst_bound = self.iv.domains_list.minimum_lowers
        iv_observation, self.subproblems = self.iv.get_observation(args.ns_batch_size)
        # bs in the 1st dimension of feature
        # save interactive verifier
        observation = self.convert_iv_observation(iv_observation)
        # extract action space
        # (batch_size or n_subproblem, layer_id, neuron_id)
        self.action_shape = list(np.array(observation.shape)[1:-1])
        self.action_space = gym.spaces.MultiDiscrete(self.action_shape)
        args.n_action     = np.prod(self.action_shape)
        # extract dict to convert layer id to layer name
        self.name_dict = {i: node.name for i, node in enumerate(self.iv.abstractor.net.split_nodes)}
        self.id_dict = {node.name: i for i, node in enumerate(self.iv.abstractor.net.split_nodes)}
        return observation, done

    def step(self, action):
        # extract args
        args = self.args
        # convert action to interative verifier format
        iv_action = self.convert_action(action)
        # interact with interactive verifier
        reward, done, info = self.iv.step(iv_action)
        # goi ham rieng khac step
        split_observation = self.convert_iv_observation(info['split_observation'])
        assert not split_observation.isnan().any()
        so1, so2 = torch.chunk(split_observation, 2)
        info['next_observation_train'] = torch.stack((so1, so2))[info['reward_indices']][0]
        if not done:
            iv_observation, self.subproblems = self.iv.get_observation(args.ns_batch_size)
            next_observation = self.convert_iv_observation(iv_observation)
            assert not next_observation.isnan().any()
        else:
            next_observation = None
        info['reward'] = reward
        info['iwb']    = self.initial_worst_bound
        # convert to device
        reward = reward.to(args.device)
        return next_observation, reward, done, info

    def convert_iv_observation(self, iv_observation):
        feature, mask = iv_observation
        # merge feature and mask
        feature  = torch.stack(feature, dim=1)
        mask     = torch.stack(mask, dim=1) # node still able to pick
        indices  = torch.where(mask == 0)
        feature0 = feature[..., 0] # score 1
        feature0[indices] = 0
        feature1 = 1 - feature[..., 1] # score 2
        feature1[indices] = 0
        feature5 = feature[..., 5] # abs
        feature5[indices] = 0
        observation = torch.stack([feature0, feature1, feature5, mask], dim=-1)
        return observation

    def convert_action(self, action):
        named_action = []
        for a in action:
            layer_id, neuron_id = a
            named_a = [self.name_dict[layer_id], neuron_id, 0.0]
            named_action.append(named_a)
        iv_action = (named_action, self.subproblems)
        return iv_action

    def convert_iv_action(self, iv_action):
        action = []
        for layer_name, neuron_id, _ in iv_action:
            a = [self.id_dict[layer_name], neuron_id]
            action.append(a)
        return action

    def full_feedback(self):
        min_rewards, iv_actions = self.iv.get_rewards(self.subproblems, reduce_op=torch.min)
        max_rewards, iv_actions = self.iv.get_rewards(self.subproblems, reduce_op=torch.max)
        min_rewards = min_rewards.permute(1, 0)
        max_rewards = max_rewards.permute(1, 0)
        # n_subproblem, n_active_neuron
        actions = np.array([self.convert_iv_action(iv_action) for iv_action in iv_actions])
        actions = actions.transpose(1, 0, 2)
        # n_subproblem, n_active_neuron, 2
        return min_rewards, max_rewards, actions


    def topk_action(self, scores, topk=1):
        actions = self.iv.topk_action(scores, self.subproblems, topk)
        actions = self.convert_iv_action(actions)
        return actions