from torch.utils.data import Dataset
import pandas as pd
import os

from prepare.prepare_io import extract_instance

class DnnvDataset(Dataset):

    def __init__(self, args):
        self.args = args
        path = os.path.join(args.dataset_dir, args.dataset, 'instances_processed.csv')
        df = pd.read_csv(path, header=None)
        self.instances = df.values.tolist()

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        onnx, vnnlib = self.instances[idx]
        onnx_path = os.path.join(self.args.dataset_dir, self.args.dataset, onnx)
        vnnlib_path = os.path.join(self.args.dataset_dir, self.args.dataset, vnnlib)
        item = {'onnx': onnx_path, 'vnnlib': vnnlib_path}
        return item