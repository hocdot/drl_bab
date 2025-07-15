import neuralsat as ns
import pandas as pd
import torch
import onnx
import os

from . import util

def write_vnnlib(path, x, y_hat, eps):
    util.write_vnnlib_classify(
        spec_path=path,
        data_lb=x-eps,
        data_ub=x+eps,
        prediction=y_hat,
        negate_spec=True,
        single_output=True,
    )

def prepare_properties(args):
    onnx_dir = os.path.join(args.benchmark_dir, args.benchmark, 'onnx')
    vnnlib_dir = os.path.join(args.benchmark_dir, args.benchmark, 'vnnlib')
    os.makedirs(vnnlib_dir, exist_ok=True)

    data = {'onnx': [], 'vnnlib': [], 'timeout': []}

    for onnx_name in os.listdir(onnx_dir):
        # read onnx
        onnx_path = os.path.join(onnx_dir, onnx_name)
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
                write_vnnlib(path, x, y_hat, args.eps)
                # check if hard property
                net, input_shape, objs = ns.test.extract_instance(onnx_path, path)
                obj = objs.pop(1)
                # initialize interactive verifier
                iv = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device='cpu')
                observation = iv.get_initial_node_data(obj)
                if observation is not None: # a.k.a hard property
                    # only write hard property (not verified immediately)
                    path = os.path.join(vnnlib_dir, f'{idx}_{onnx_name[:-5]}.vnnlib')
                    write_vnnlib(path, x, y_hat, args.eps)
                    # add to data
                    data['onnx'].append(onnx_path)
                    data['vnnlib'].append(path)
                    data['timeout'].append(args.timeout)
                    # print progress
                    print(f'+ found {len(data["onnx"])}/{args.n_instance} properties')
                    if len(data['onnx']) > args.n_instance:
                        # save instances.csv
                        df = pd.DataFrame(data)
                        csv_dir = os.path.join(args.benchmark_dir, args.benchmark)
                        os.makedirs(csv_dir, exist_ok=True)
                        path = os.path.join(csv_dir, 'instances.csv')
                        df.to_csv(path, header=None)
                        break
