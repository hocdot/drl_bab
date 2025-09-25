import pandas as pd
import torch
import tqdm
import os
import gc

from joblib import Parallel, delayed, parallel_config


from dnnv_solver.helper.misc.torch_cuda_memory import gc_cuda
from dnnv_solver.helper.network.read_onnx import parse_onnx
from dnnv_solver.helper.spec.objective import parse_vnnlib
from dnnv_solver.dnnv_bab import DnnvBabSolver



def extract_instance(net_path, vnnlib_path):
    model, input_shape, output_shape = parse_onnx(net_path)
    objectives = parse_vnnlib(vnnlib_path, input_shape)
    return model, input_shape, objectives


def instance_generator(args):
    # load hard instances list
    path = os.path.join(args.benchmark_dir, args.benchmark, 'instances.csv')
    df = pd.read_csv(path)
    #
    for i, (onnx, vnnlib, timeout) in tqdm.tqdm(df.iterrows(), total=len(df)):
        yield onnx, vnnlib

def read_instance(onnx, vnnlib, args):
    path = vnnlib.replace('/vnnlib', '/io').replace('.vnnlib', '.pt')
    if os.path.exists(path):
        return
    # initialize iv
    net, input_shape, objs = extract_instance(onnx, vnnlib)
    obj = objs.pop(1)
    iv = DnnvBabSolver(net=net, input_shape=input_shape, objective=obj, args=args)
    # load first observation
    x, y = iv.get_observation(return_fsb_score=True)
    x = torch.cat(x, dim=1)
    y = y.cpu()
    # print(f'{x[0].max(dim=0)} {x[0].min(dim=0)} {y[0].max(dim=0)} {y[0].min(dim=0)}')
    # save data
    data = {'x': x, 'y': y}
    torch.save(data, path)
    # collect garbage
    del iv
    gc_cuda()
    gc.collect()

def prepare_io(args):
    # create result dirs if not exist
    io_dir = os.path.join(args.benchmark_dir, args.benchmark, 'io')
    if args.clear:
        print(f'[+] clearing caches')
        cmd = f'rm -rf {io_dir}'
        print(f'    $ {cmd}')
        os.system(cmd)
    os.makedirs(io_dir, exist_ok=True)
    # form a torch geometric dataset from hard instances list
    print('[+] reading onnxs and vnnlibs')
    if args.parallel:
        with parallel_config(backend='threading', n_jobs=args.n_job):
            Parallel()(delayed(read_instance)(onnx, vnnlib, args) for onnx, vnnlib in instance_generator(args))
    else:
        for onnx, vnnlib in tqdm.tqdm(instance_generator(args), total=args.n_property):
            read_instance(onnx, vnnlib, args)

if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    prepare_io(args)