import neuralsat as ns
import pandas as pd
import torch
import onnx
import os

from . import util

def prepare_properties(args):
    onnx_dir = os.path.join(args.benchmark_dir, args.benchmark, 'onnx')
    vnnlib_dir = os.path.join(args.benchmark_dir, args.benchmark, 'vnnlib')
    csv_dir = os.path.join(args.benchmark_dir, args.benchmark)
    os.makedirs(csv_dir, exist_ok=True)

    path = os.path.join(csv_dir, 'instances.csv')
    data = {}

    for name in os.listdir(onnx_dir):
        # read onnx
        path = os.path.join(onnx_dir, name)
        model, input_shape = ns.util.network.read_onnx.parse_onnx(onnx_path)[:2]
        model.to(args.device)
        model.eval()
        # get input data
        dataloader = util.get_dataloader(args)
        # loop to create properties
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(args.device)
            y = y.to(args.device)
            y_hat = model(x.view(input_shape))
            y_pred = y_hat.argmax(-1)
            # only write properties for input with correct prediction
            if y_pred == y:
                # write temporary property
                path = os.path.join('/tmp', f'{idx}_{onnx_name[:-5]}.vnnlib')
                # only write hard properties (not verified immediately)
                path = os.path.join(vnnlib_dir, f'{idx}_{onnx_name[:-5]}.vnnlib')
                util.write_vnnlib_classify(
                    spec_path=path,
                    data_lb=x - args.eps,
                    data_ub=x + args.eps,
                    prediction=prediction,
                    negate_spec=True,
                    single_output=True,
                )
                # TODO: check attack
                fp.write(f'{count},{onnx_path},{vnnlib_path}\n')
                count += 1
                if count >= args.n_instance:
                    # TODO: next network
                    return
