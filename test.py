from heuristic.dnnv_baseline import FsbDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


def test(args):
    solver = FsbDecider(args)
    # solver_classes = SACDecider
    onnx = 'data/dataset/dnnv/onnx/simple_cnn_3x21.onnx'
    vnnlib = 'data/dataset/dnnv/vnnlib_cnn_small/spec_simple_cnn_3x21_idx_2_eps_0.03825_0.vnnlib'
    
    info = solver.solve_one_instance(onnx, vnnlib)
    print(info)
    
'''
CUDA_VISIBLE_DEVICES= python3 test.py --config config/dnnv.yaml --verbose
'''

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    args.device = 'cpu'
    test(args)