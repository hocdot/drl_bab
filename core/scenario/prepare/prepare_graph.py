import neuralsat as ns
import networkx as nx
import pandas as pd
import simulator
import itertools
import pickle
import torch
import tqdm
import os


def prepare_graph(args):
    # list all onnx
    for instance in ['train.csv']:
        path = os.path.join(args.benchmark_dir, args.benchmark, instance)
        df = pd.read_csv(path)
        onnxs = df['onnx'].unique()
        vnnlib = df['vnnlib'][0]
        for onnx in onnxs:
            graph_dir = os.path.join(args.benchmark_dir, args.benchmark, 'graph')
            if f'{onnx}.index.pth' not in os.listdir(graph_dir):
                d = os.path.join(args.benchmark_dir, args.benchmark)
                onnx_path = os.path.join(d, 'onnx', onnx)
                vnnlib_path = os.path.join(d, 'vnnlib', vnnlib)
                print(f'[+] loading net {onnx_path=} {vnnlib_path=}')
                # net, _, _ = ns.test.extract_instance(onnx_path, vnnlib_path)
                net, input_shape, output_shape, _ = ns.util.network.read_onnx.parse_onnx(onnx_path)
                print('[+] loading preparing graph')
                G = simulator.prepare_graph(net, input_shape, output_shape, args)
                # save graph
                path = os.path.join(graph_dir, f'{onnx}.pkl')
                with open(path, 'wb') as fp:
                    pickle.dump(G, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f' saved {path}')
