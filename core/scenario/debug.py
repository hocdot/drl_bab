import onnx

def debug(args):
    #
    path = '../data/benchmark/cifar/0.onnx'
    model = onnx.load(path)

    help(model)
