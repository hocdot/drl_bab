from torch.utils.data import DataLoader
import os

from .milp import MILPDataset

def create_dataloader(args):
    dataloader_dict = {}
    # TODO: remove overfit here
    if args.dataset == 'milp':
        args.mode = 'test' # no need train dataset, generate on-the-fly
        dataset = MILPDataset(args)
        dataset.prepare()
        dataloader_dict[args.mode] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=os.cpu_count(),
        )
    else:
        raise NotImplementedError
    return dataloader_dict