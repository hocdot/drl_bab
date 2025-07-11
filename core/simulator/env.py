import neuralsat as ns
import torch

class Env:

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args

    def reset(self):
        # extract args
        args = self.args
        # extract problem (TODO: remove hardcode)
        onnx_path = f'../data/benchmark/{args.benchmark}/onnx/mnistfc-medium-net-151.onnx'
        vnnlib_path = f'../data/benchmark/{args.benchmark}/vnnlib/prop_2_0.03.vnnlib'
        net, input_shape, objs = ns.test.extract_instance(onnx_path, vnnlib_path)
        obj = objs.pop(1)
        # initialize interactive verifier
        self.iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device='cpu')
        # extract observation + topo
        observation = self.iv.get_initial_node_data(obj)
        observation = torch.cat(observation, dim=1)
        edge_weight, edge_index = self.iv.get_edge_data(obj)
        # extract subproblems
        self.subproblem = self.iv.setup(obj)
        return observation, edge_index, edge_weight

    @property
    def last_action(self):
        return self.iv.get_last_actions(self.subproblem)

    def get_fsb_action(self):
        return self.iv.get_fsb_action(self.subproblem)

    def step(self, action):
        self.subproblem, reward, next_observation, subproblem_indices = self.iv.step(self.subproblem, action)
        if self.subproblem is not None:
            next_observation = torch.cat(next_observation, dim=1)
            print(f'{next_observation.shape=} {reward.shape=}', subproblem_indices)
        done = self.subproblem is None
        info = {}
        return next_observation, reward, done, info
