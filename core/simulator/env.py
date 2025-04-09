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
            path = os.path.join(args.benchmark_dir, args.benchmark, 'train.csv')
        elif args.mode == 'test':
            path = os.path.join(args.benchmark_dir, args.benchmark, 'test.csv')
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
        # if self.onnx != self.loaded_onnx:
        #     d = os.path.join(args.benchmark_dir, args.benchmark, self.onnx.replace('.onnx', ''))
        #     path = os.path.join(args.benchmark_dir, args.benchmark, f'{self.onnx}.index.pth')
        #     self.edge_index = torch.load(path)
        #     path = os.path.join(args.benchmark_dir, args.benchmark, f'{self.onnx}.weight.pth')
        #     self.edge_weight = torch.load(path)
        #     self.loaded_onnx = self.onnx
        if args.verbose:
            print(f'[+] loading new instance: {self.onnx=} {self.vnnlib=}')
        # load environment
        # TODO: extract moi objectives thoi ma khong phai extract lai net
        d           = os.path.join(args.benchmark_dir, args.benchmark)
        onnx_path   = os.path.join(d, 'onnx', self.onnx)
        vnnlib_path = os.path.join(d, 'vnnlib', self.vnnlib)
        net, input_shape, objectives = ns.test.extract_instance(onnx_path, vnnlib_path)
        self.net           = net
        self.input_shape   = input_shape
        self.iv            = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
        objectives, _      = self.iv._preprocess(objectives, force_split=None)
        self.objective     = objectives.pop(3)
        # increase instance id counter
        self.instance_id += 1

    def reset(self):
        # extract args
        args = self.args
        # load next instance if there is no objective left for current instance
        self.load_next_instance()
        done = self.iv.init(objective=self.objective, reference_bounds=None, preconditions={})
        # self.initial_worst_bound = self.iv.domains_list.minimum_lowers
        observation, self.subproblems = self.iv.get_observation(args.ns_batch_size)
        # feature, active_neuron_index, n_active_neuron = observation
        # feature            : list of layer (bs, padded_n_active_neuron, n_feature)
        # active_neuron_index: list of layer (bs, padded_n_active_neuron)
        # n_active_neuron    : list of layer (bs)
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
        unpruned_next_observations = info['unpruned_next_observations']
        print(type(unpruned_next_observations))
#         assert not unpruned_next_observations.isnan().any()

#         uno_1, uno_2 = torch.chunk(unpruned_next_observations, 2)
#         info['unpruned_next_observation_1'] = uno_1
#         info['unpruned_next_observation_2'] = uno_2

        if not done:
            next_observation, self.subproblems = self.iv.get_observation(args.ns_batch_size)
        else:
            next_observation = None
        info['reward'] = reward
        reward = reward.to(args.device)
        return next_observation, reward, done, info

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

    def full_feedback(self, return_min_max_rewards=False):
        active_neuron_rewards, iv_actions = self.iv.full_feedback(self.subproblems,
                                                                  return_min_max_rewards=return_min_max_rewards)
        # n_subproblem, n_active_neuron
        actions = np.array([self.convert_iv_action(iv_action) for iv_action in iv_actions])
        actions = actions.transpose(1, 0, 2) # n_subproblem, n_active_neuron, 2
        if return_min_max_rewards:
            assert isinstance(active_neuron_rewards, tuple)
            return active_neuron_rewards[0], active_neuron_rewards[1], actions
        return active_neuron_rewards, actions

    def topk_action(self, scores, topk=1):
        actions = self.iv.topk_action(scores, self.subproblems, topk)
        actions = self.convert_iv_action(actions)
        return actions
