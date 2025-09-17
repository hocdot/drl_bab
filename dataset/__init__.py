from torch.utils.data import DataLoader
from .milp import MILPDataset
import os

def create_dataloader(args):
    dataloader_dict = {}
    # TODO: remove overfit here
    for mode in ['train', 'test']:
        args.mode = mode
        dataset = MILPDataset(args)
        dataset.prepare()
        dataloader_dict[args.mode] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=args.mode=='train',
            num_workers=os.cpu_count(),
        )
    return dataloader_dict