from torch.utils.data import Dataset
import pandas as pd
import os


class DnnvDataset(Dataset):

    def __init__(self, args):
        self.args = args
        path = os.path.join(args.dataset_dir, args.dataset, 'instances.csv')
        df = pd.read_csv(path, header=None)
        self.instances = df.values.tolist()

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]