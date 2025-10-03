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
    assert args.dataset == 'dnnv', f'{args.dataset=}'
    # create result dirs if not exist
    embed_dir = os.path.join(args.dataset_dir, args.dataset, 'embed')
    os.makedirs(embed_dir, exist_ok=True)
    # load graph
    # FIXME: handle multiple graphs
    for onnx, _ in instance_generator('instances_train.csv',args):
        break
    graph_path = os.path.join(args.dataset_dir, args.dataset, 'graph', f'{os.path.basename(onnx)}.pt')
    data = torch.load(graph_path)
    edge_weight = data['edge_weight']
    edge_index = data['edge_index']
    # load prepared io
    io_dir = os.path.join(args.dataset_dir, args.dataset, 'io')
    data_list = []
    print('[+] loading preprocessed io')
    for name in tqdm.tqdm(os.listdir(io_dir)):
        if name.endswith('.txt'):
            continue
        if not name.startswith(os.path.basename(onnx)):
            continue
        path = os.path.join(io_dir, name)
        data = torch.load(path)
        data = Data(x=data['observation'], y=data['score'], name=os.path.basename(path))
        data_list.append(data)
    assert len(data_list) > 0, f'{len(data_list)=}'
    # create dataloader from graphs dataset
    loader = DataLoader(data_list, batch_size=args.gcn_embed_batch_size, shuffle=True)
    # initialize model
    net = FsbRegression(4, args.gcn_embed_n_hidden, 2, args.device).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.gcn_embed_lr)
    # training loop
    print('[+] training FsbRegression using GCNConv')
    for epoch in range(args.gcn_embed_n_epoch):
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
        print(f'    * {epoch=}/{args.gcn_embed_n_epoch} {i=} {loss=:0.6f}')
    # exporting embedding
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    print(f'[+] exporting GCN embeddings for {len(loader)=} instances')
    for i, data in enumerate(tqdm.tqdm(loader)):
        # forward
        indices = torch.where(data.x[0, :, -1] == 1)[0] # 1 - not selected, 0 - selected
        e = net.embedding(data, edge_index, edge_weight)[0, indices].detach().cpu()
        # save gcn embedding
        path = os.path.join(embed_dir, data.name[0])
        torch.save(e, path)
        print(f'    + saved {e.shape=} at {path}')


if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    prepare_embed(args)