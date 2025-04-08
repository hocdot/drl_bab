import pandas as pd
import numpy as np
import os

def split_train_test(args):
    # load list of instances
    path = os.path.join(args.benchmark_dir, args.benchmark)
    data = {'onnx': [], 'vnnlib': []}
    for name in list(sorted(set(os.listdir(path)))):
        # load instances
        path = os.path.join(args.benchmark_dir, args.benchmark, name, 'instance.csv')
        df = pd.read_csv(path, names=['onnx', 'vnnlib', 'timeout'])
        for i, (onnx, vnnlib, _) in df.iterrows():
            onnx = os.path.basename(onnx)
            vnnlib = os.path.basename(vnnlib)
            data['onnx'].append(onnx)
            data['vnnlib'].append(vnnlib)
            if len(data['onnx']) >= 1000:
                break
        if len(data['onnx']) >= 1000:
            break
    df = pd.DataFrame(data)
    # split train test
    indices  = np.random.permutation(len(df))
    n_train  = int(len(indices) * 0.8)
    df_train = df.iloc[indices[:n_train]]
    df_test  = df.iloc[indices[n_train:]]
    # load list of instances
    path = os.path.join(args.benchmark_dir, args.benchmark, 'train.csv')
    df_train.to_csv(path, index=None)
    path = os.path.join(args.benchmark_dir, args.benchmark, 'test.csv')
    df_test.to_csv(path, index=None)
