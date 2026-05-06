from dnnv_solver.helper.network.onnx2pytorch import ConvertModel
import torch
import onnx
import os


def get_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
    
def count(path, input_shape):
    print(path)

    onnx_model = onnx.load(path)
    model = ConvertModel(onnx_model, enable_recording=True)
    # print(model)
    
    x = torch.randn(input_shape)
    y, hs = model(x)
    assert all([h.shape[0]==1 for h in hs])
    # print(y.shape)
    print(os.path.basename(path), 'neurons:', sum([h.numel() for h in hs]))
    print(os.path.basename(path), 'parameters:', get_parameters(model))
    
if __name__ == "__main__":
    instances = {
        'fnn': [
            ('data/dnnv_fnn_new/dnnv/onnx/mnist-net_256x2.onnx', (1, 1, 28, 28)),
            ('data/dnnv_fnn_new/dnnv/onnx/mnist-net_256x4.onnx', (1, 1, 28, 28)),
            ('data/dnnv_fnn_new/dnnv/onnx/mnist-net_256x6.onnx', (1, 1, 28, 28)),
        ],
        'cnn': [
            ('data/dnnv_cnn_new/dnnv/onnx/cifar_base_kw.onnx', (1, 3, 32, 32)),
            ('data/dnnv_cnn_new/dnnv/onnx/cifar_deep_kw.onnx', (1, 3, 32, 32)),
            ('data/dnnv_cnn_new/dnnv/onnx/cifar_wide_kw.onnx', (1, 3, 32, 32)),
        ],
        'sigmoid': [
            ('data/dnnv_distshift/dnnv/onnx/mnist_concat.onnx', (1, 792)),
        ],
    }
    for benchmark, instances in instances.items():
        for onnx_path, input_shape in instances:
            count(onnx_path, input_shape)
        print('-' * 100)
 