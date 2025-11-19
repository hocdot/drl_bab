import json
import os

from heuristic.dnnv_baseline.fsb import FsbDecider
from .prepare_io import instance_generator
from arguments import get_args

if __name__ == '__main__':
    
    args = get_args()
    assert args.dataset == 'dnnv', f'{args.dataset=}'
    log_dir = os.path.join(args.dataset_dir, args.dataset, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    output_path = os.path.join(args.dataset_dir, args.dataset, 'instances_hard.csv')
    os.system(f'rm -rf {output_path}')
    
    for onnx, vnnlib in instance_generator('instances_processed.csv', args):
        path = os.path.join(log_dir, f'{os.path.basename(onnx)}____{os.path.basename(vnnlib)}.json')
        if os.path.exists(path):
            info = json.load(open(path))
        else:
            solver = FsbDecider(args)
            try:
                info = solver.solve_one_instance(onnx, vnnlib)
            except AssertionError:
                info = {'episode': 1, 'value': 0, 'done': False, 'early_stop': True}
            except:
                raise NotImplementedError
            
            with open(path, 'w') as f:
                json.dump(info, f)

        if info['value'] < -500 and info['done']:
            line = f'onnx/{os.path.basename(onnx)},vnnlib_processed/{os.path.basename(vnnlib)}'
            with open(output_path, 'a') as f:
                print(line, file=f)
