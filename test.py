from heuristic.sac.sac import SACDecider
from heuristic.dnnv_baseline import FsbDecider, BabsrDecider, UpbDecider, GnnDecider
from arguments import get_args


def test(args):
    solver = SACDecider(args)
    # solver = FsbDecider(args)
    # solver = GnnDecider(args)
    # solver = BabsrDecider(args)
    # solver = UpbDecider(args)
    onnx = 'data/dnnv_fnn_new/dnnv/onnx/mnist-net_256x6.onnx'
    vnnlib = 'data/dnnv_fnn_new/dnnv/vnnlib_processed/prop_mnist-net_256x6_754_0.08_seed_11_output_8.vnnlib'
    
    print(f'Using {solver.__class__.__name__} with decision_heuristic={args.decision_heuristic}, use_one_shot_inference={args.use_one_shot_inference}')
    info = solver.solve_one_instance(onnx, vnnlib)
    print(info)
    
'''
CUDA_VISIBLE_DEVICES= python3 test.py --config config/dnnv_fnn.yaml --verbose
CUDA_VISIBLE_DEVICES= python3 test.py --config config/dnnv_fnn.yaml --verbose --use_one_shot_inference
# set decision_heuristic in yaml (fsb | babsr | upb), or omit for default fsb
'''

if __name__ == '__main__':
    args = get_args()
    args.mode = 'test'
    args.device = 'cpu'
    test(args)
