import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import os

from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv

from .prepare_io import instance_generator, extract_instance
from dnnv_solver.dnnv_bab import DnnvBabSolver


class Embedding(nn.Module):

    def __init__(self, n_input, n_hidden, device):
        super().__init__()
        self.conv1 = GCNConv(n_input, n_hidden)
        # self.conv2 = GCNConv(n_hidden, n_hidden)
        self.device = device

    def forward(self, data, edge_index, edge_weight):
        x = data.x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_weight)
        return x

class Regression(nn.Module):

    def __init__(self, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_hidden, n_output)
        # self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x

class FsbRegression(nn.Module):

    def __init__(self, n_input, n_hidden, n_output, device):
        super().__init__()
        self.embedding  = Embedding(n_input, n_hidden, device)
        self.regression = Regression(n_hidden, n_output)

    def forward(self, data, edge_index, edge_weight):
        x = self.embedding(data, edge_index, edge_weight)
        y_hat = self.regression(x)
        return y_hat

def prepare_embed(args):
    # create result dirs if not exist
    embed_dir = os.path.join(args.benchmark_dir, args.benchmark, 'embed')
    if args.clear:
        print(f'[+] clearing caches')
        cmd = f'rm -rf {embed_dir}'
        print(f'    $ {cmd}')
        os.system(cmd)
    os.makedirs(embed_dir, exist_ok=True)
    # load graph
    # load a shared edge_index edge_weight
    for onnx, vnnlib in instance_generator(args):
        break
    graph_dir = os.path.join(args.benchmark_dir, args.benchmark, 'graph')
    graph_path = os.path.basename(onnx).replace('.onnx', '.pt')
    path = os.path.join(graph_dir, graph_path)
    os.makedirs(graph_dir, exist_ok=True)
    if os.path.exists(path):
        data = torch.load(path)
        edge_weight = data['edge_weight']
        edge_index = data['edge_index']
    else:
        net, input_shape, objs = extract_instance(onnx, vnnlib)
        obj = objs.pop(1)
        iv = DnnvBabSolver(net=net, input_shape=input_shape, objective=obj, args=args)
        edge_weight, edge_index = iv.get_edge_data(obj)
        data = {'edge_weight': edge_weight, 'edge_index': edge_index}
        torch.save(data, path)
    # load prepared io
    io_dir = os.path.join(args.benchmark_dir, args.benchmark, 'io')
    data_list = []
    print('[+] loading preprocessed io')
    for name in tqdm.tqdm(os.listdir(io_dir)):
        path = os.path.join(io_dir, name)
        data = torch.load(path)
        data = Data(x=data['x'],
                    y=data['y'],
                    name=os.path.basename(path))
        data_list.append(data)
    # create dataloader from graphs dataset
    loader = DataLoader(data_list, batch_size=args.gcn_batch_size, shuffle=True)
    #
    net = FsbRegression(4, args.gcn_n_hidden, 2, args.device).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.gcn_lr)
    # training loop
    print('[+] training FsbRegression using GCNConv')
    for epoch in range(args.gcn_n_epoch):
        total_loss = 0
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            m = (1 - data.x[:, :, -1]).unsqueeze(2).repeat(1, 1, 2).to(args.device)
            y_hat = net(data, edge_index, edge_weight)
            loss = F.mse_loss(y_hat * m, data.y.to(args.device) * m)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # if i % 100 == 0:
                # print(f'    * {epoch=}/{args.gcn_n_epoch} {i=} {loss.item()=:0.6f}')
            # if i > 1000:
            #     break
        loss = total_loss / len(loader)
        print(f'    * {epoch=}/{args.gcn_n_epoch} {i=} {loss=:0.6f}')
    # exporting embedding
    print('[+] exporting GCN embeddings')
    for i, data in enumerate(tqdm.tqdm(loader)):
        # forward
        indices = torch.where(data.x[0, :, -1] == 0)[0]
        e = net.embedding(data, edge_index, edge_weight)[0, indices].detach().cpu()
        # save gcn embedding
        path = os.path.join(embed_dir, data.name[0])
        torch.save(e, path)
        # print(f'    + saved {e.shape=} at {path}')


if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    prepare_embed(args)