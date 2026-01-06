from torch.utils.data import Dataset
import pandas as pd
import os

class DnnvDataset(Dataset):

    def __init__(self, args):
        self.args = args
        path = os.path.join(args.dataset_dir, args.dataset, f'instances_{args.mode}.csv')
        if os.path.exists(path):
            print(f'[+] loading {args.mode} dataset {path=}')
            df = pd.read_csv(path, header=None)
            if args.mode == 'train':
                df = df.sample(frac=1).reset_index(drop=True)
            self.instances = df.values.tolist()
        else:
            self.instances = []

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        onnx, vnnlib = self.instances[idx]
        onnx_path = os.path.join(self.args.dataset_dir, self.args.dataset, onnx)
        vnnlib_path = os.path.join(self.args.dataset_dir, self.args.dataset, vnnlib)
        item = {'onnx': onnx_path, 'vnnlib': vnnlib_path}
        return item