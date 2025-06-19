import neuralsat as ns
import torch

def debug(args):
    #
    net, input_shape, objs = ns.test.extract_instance(
            net_path='../data/benchmark/cifar/onnx/0.onnx',
            vnnlib_path='../data/benchmark/cifar/vnnlib/0.vnnlib',
    )

    # extract input node feature
    obj = objs.pop(1)
    lb = obj.lower_bounds[0]
    ub = obj.upper_bounds[0]
    bias = torch.zeros_like(lb)
    mask = torch.zeros_like(lb)
    x = torch.stack([lb, ub, bias, mask])
    print(x.shape)
