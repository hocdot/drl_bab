from torch.utils.data import DataLoader
import os

from .milp_dataset import MILPDataset
from .dnnv_dataset import DnnvDataset

def create_dataloader(args):
    dataloader_dict = {}
    if args.mode == 'train_offline':
        dataloader_dict[args.mode] = iter(())
        return dataloader_dict
    
    if args.dataset == 'milp':
        dataset = MILPDataset(args)
        dataset.prepare()
        dataloader_dict[args.mode] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=os.cpu_count(),
        )
    else:
        dataset = DnnvDataset(args)
        # FIXME: split train and test
        dataloader_dict[args.mode] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=os.cpu_count(),
        )
    return dataloader_dict