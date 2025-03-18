import neuralsat as ns
import pandas as pd
import simulator
import torch
import tqdm
import os

def prepare_graph(args):
    # list all onnx
    for instance in ['train_instance.csv', 'test_instance.csv']:
        path = os.path.join(args.onnx_dir, instance)
        df = pd.read_csv(path)
        onnxs = df['onnx'].unique()
        vnnlib = df['vnnlib'][0]
        for onnx in onnxs:
            if f'{onnx}.index.pth' not in os.listdir(args.onnx_dir):
                d = os.path.join(args.onnx_dir, onnx.replace('.onnx', ''))
                onnx_path = os.path.join(d, 'onnx', onnx)
                vnnlib_path = os.path.join(d, 'vnnlib', vnnlib)
                net, _, _ = ns.test.extract_instance(onnx_path, vnnlib_path)
                ei, ew = simulator.prepare_graph(net, args)
                # save ei ew
                path = os.path.join(args.onnx_dir, f'{onnx}.index.pth')
                torch.save(ei, path)
                print(f' saved {path}')
                path = os.path.join(args.onnx_dir, f'{onnx}.weight.pth')
                torch.save(ew, path)
                print(f' saved {path}')
