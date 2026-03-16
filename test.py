from heuristic.dnnv_baseline import FsbDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


def test(args):
    # solver = FsbDecider(args)
    # solver = SACDecider(args)
    solvers = [FsbDecider(args), SACDecider(args)]
    onnx = 'data/dnnv_cnn/dnnv/onnx/simple_cnn_3x21.onnx'
    vnnlib = 'data/dnnv_cnn/dnnv/vnnlib_processed/spec_simple_cnn_3x21_idx_2_eps_0.03825_0_output_8.vnnlib'
    
    for solver in solvers[1:]:
        info = solver.solve_one_instance(onnx, vnnlib)
        print(f'{solver}: {info}')
    
'''
python3 -m prepare.prepare_io --use_attack --config config/dnnv_cnn.yaml
python3 -m prepare.prepare_gnn_embedding --config config/dnnv_cnn.yaml
CUDA_VISIBLE_DEVICES= python3 test.py --config config/dnnv_cnn.yaml --verbose
'''

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    args.device = 'cpu'
    test(args)