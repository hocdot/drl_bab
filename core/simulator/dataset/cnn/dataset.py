from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, mode, args):
        # save args
        self.args = args
        self.mode = mode

    def prepare(self):
        # extract args
        args = self.args
        # TODO: paste training code here
        # OUTPUT: save at `args.benchmark_dir/args.benchmark/ *.onnx *.vnnlib`
        # convert onnx, vnnlib to node feature, edge index, edge weight of active neurons

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sample = {'x': self.X[i], 'y': self.Y[i]}
        # x:
        # input: lb, ub, bias=0, mask=0 for all
        # hidden: lb, ub, bias, mask=1 for active neuron, mask=0 otherwise
        # output: lb, ub, bias, mask=1 for active neuron, mask=0 otherwise
        # edge_index: [list of [source_node_id, target_node_id]]
        # edge_weight: weight of connections [list of [weight]]
        # y:
        # tour from heuristic (for comparison)
        return sample
