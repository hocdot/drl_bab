from heuristic.dnnv_baseline import FsbDecider, UpbDecider, GnnDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


def test(args):
    # solvers = [FsbDecider(args)]
    solvers = [SACDecider(args)]
    # solvers = [UpbDecider(args)]
    # solvers = [GnnDecider(args)]
    # onnx    = 'data/dist_shift_2023/dnnv/onnx/mnist_concat.onnx'
    # vnnlib  = 'data/dist_shift_2023/dnnv/vnnlib_processed/index7027_delta0.13_output_9.vnnlib'
    
    onnx = "data/dnnv_fnn_new/dnnv/onnx/mnist-net_256x4.onnx"
    vnnlib = "data/dnnv_fnn_new/dnnv/vnnlib_processed/prop_mnist-net_256x4_522_0.06_seed_7_output_9.vnnlib"
    
    # onnx = "data/dataset/dnnv/onnx/mnist-net_256x6.onnx"
    # vnnlib = "data/dataset/dnnv/vnnlib_processed/prop_mnist-net_256x6_2273_0.06_seed_8_output_6.vnnlib"
    
    for solver in solvers[:1]:
        info = solver.solve_one_instance(onnx, vnnlib)
        print(f'{solver}: {info}')
    
'''
python3 -m prepare.prepare_io --use_attack --config config/dnnv_cnn.yaml
python3 -m prepare.prepare_gnn_embedding --config config/dnnv_cnn.yaml
CUDA_VISIBLE_DEVICES= python3 test.py --config config/dnnv_cnn.yaml --verbose

old:
python3 test.py --config config/dnnv.yaml --checkpoint_postfix 4 --verbose --device cpu --decision_topk_test 5 

python3 test.py --config config/dnnv.yaml --verbose --device cuda --checkpoint_postfix 32
'''

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    # args.device = 'cpu'
    test(args)