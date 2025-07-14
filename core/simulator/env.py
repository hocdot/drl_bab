import neuralsat as ns
import pandas as pd
import torch
import os

class Env:

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # load list of instances
        self.load_instances()
        # initialize
        self.n_episode = 0

    def load_instances(self):
        # extract args
        args = self.args
        # load csv
        names = ['instance', 'onnx', 'vnnlib']
        path = os.path.join(args.benchmark_dir, args.benchmark, 'instances.csv')
        df = pd.read_csv(path, names=names)
        # load net
        onnx_path = df['onnx'].iloc[0]
        vnnlib_path = df['vnnlib'].iloc[0]
        self.net, self.input_shape, _ = ns.test.extract_instance(onnx_path, vnnlib_path)
        # load objectives from vnnlibs
        self.objectives = []
        for vnnlib_path in df['vnnlib']:
            net, input_shape, objs = ns.test.extract_instance(onnx_path, vnnlib_path)
            obj = objs.pop(1)
            # initialize interactive verifier
            iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device='cpu')
            observation = iv.get_initial_node_data(obj)
            if observation is not None:
                self.objectives.append(obj)
        print(f'[+] loaded {len(self)} valid properties')

    def __len__(self):
        return len(self.objectives)

    def reset(self):
        # extract args
        args = self.args
        # extract net and obj
        net = self.net
        input_shape = self.input_shape
        obj = self.objectives[self.n_episode % len(self.objectives)]
        # initialize interactive verifier
        self.iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device='cpu')
        # extract observation + topo
        observation = self.iv.get_initial_node_data(obj)
        observation = torch.cat(observation, dim=1)
        edge_weight, edge_index = self.iv.get_edge_data(obj)
        # extract subproblems
        self.subproblem = self.iv.setup(obj)
        # step
        self.n_step = 0
        self.n_episode += 1
        return observation, edge_index, edge_weight

    @property
    def last_action(self):
        return self.iv.get_last_actions(self.subproblem)

    def get_fsb_action(self):
        return self.iv.get_fsb_action(self.subproblem)

    def step(self, action):
        args = self.args
        self.subproblem, reward, next_observation, subproblem_indices = self.iv.step(self.subproblem, action)
        if self.subproblem is not None:
            next_observation = torch.cat(next_observation, dim=1)
        done = self.subproblem is None
        if self.n_step > args.ns_n_step:
            done = True
        info = {'subproblem_indices': subproblem_indices}
        return next_observation, reward, done, info
