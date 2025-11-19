import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import os

from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv

from .prepare_io import instance_generator


class Embedding(nn.Module):

    def __init__(self, n_input, n_hidden, device):
        super().__init__()
        self.conv1 = GCNConv(n_input, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.device = device

    def forward(self, data, edge_index, edge_weight):
        x = data.x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
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
    
def get_train_data(csv_name):
    # load prepared io
    io_dir = os.path.join(args.dataset_dir, args.dataset, 'io')
    data_list = []
    for onnx, vnnlib in instance_generator(csv_name, args):
        onnx_name = os.path.basename(onnx)
        vnnlib_name = os.path.basename(vnnlib)
        io_path = os.path.join(io_dir, f'{onnx_name}____{vnnlib_name}.pt')
        assert os.path.exists(io_path), f'{io_path=}'
        data = torch.load(io_path)
        data = Data(x=data['observation'], y=data['score'], name=os.path.basename(io_path), graph_name=onnx_name)
        data_list.append(data)
    print(f'    + loaded {len(data_list)} io data')
    assert len(data_list) > 0, f'{len(data_list)=}'
    return data_list

def get_graph_data(csv_name):
    all_graphs = {}
    for onnx, _ in instance_generator(csv_name, args):
        onnx_name = os.path.basename(onnx)
        if onnx_name not in all_graphs:
            graph_path = os.path.join(args.dataset_dir, args.dataset, 'graph', f'{os.path.basename(onnx)}.pt')
            data = torch.load(graph_path)
            edge_weight = data['edge_weight']
            edge_index = data['edge_index']
            all_graphs[onnx_name] = (edge_weight, edge_index)
    print(f'    + loaded {len(all_graphs)} graphs')
    return all_graphs
    
def prepare_embed(args):
    assert args.dataset == 'dnnv', f'{args.dataset=}'
    # create result dirs if not exist
    embed_dir = os.path.join(args.dataset_dir, args.dataset, 'embed')
    os.makedirs(embed_dir, exist_ok=True)

    # load graphs
    all_graphs = get_graph_data('instances_processed.csv')
    
    # create dataloader with batch_size=1 and use gradient accumulation for effective larger batch size
    # data_list = get_train_data('instances_processed.csv')
    data_list = get_train_data('instances_hard.csv')
    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    accumulation_steps = args.gcn_embed_batch_size

    # initialize model
    net = FsbRegression(4, args.gcn_embed_n_hidden, 2, args.device).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.gcn_embed_lr)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, args.dataset, f'embed.pt')
    if os.path.exists(checkpoint_path):
        net.load_state_dict(torch.load(checkpoint_path))
        print(f'[+] loaded checkpoint from {checkpoint_path}')
    else:
        # training loop with gradient accumulation
        print('[+] training FsbRegression using GCNConv')
        for epoch in range(args.gcn_embed_n_epoch):
            total_loss = 0
            total_batches = 0
            accumulated_loss = 0
            
            pbar = tqdm.tqdm(loader)
            for i, data in enumerate(pbar):
                # get edge_index and edge_weight for this specific graph
                edge_weight, edge_index = all_graphs[data.graph_name[0]]
                # print(f'{data.graph_name[0]=} {edge_weight.shape=} {edge_index.shape=}')
                
                # create mask and forward
                m = data.x[:, :, -1].unsqueeze(2).repeat(1, 1, 2).to(args.device) # 1 - not selected, 0 - selected
                y_hat = net(data, edge_index, edge_weight)
                loss = F.mse_loss(y_hat * m, data.y.to(args.device) * m)
                # TODO: add contrastive loss

                # normalize loss by accumulation steps for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                
                # update weights every accumulation_steps
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += accumulated_loss
                    pbar.set_postfix({'accumulated_loss': accumulated_loss})
                    accumulated_loss = 0
                    total_batches += 1
                
            avg_loss = total_loss / total_batches if total_batches > 0 else 0
            print(f'    * {epoch=}/{args.gcn_embed_n_epoch} {avg_loss=:0.6f}')
            # save
            torch.save(net.state_dict(), checkpoint_path)
            print(f'[+] saved checkpoint to {checkpoint_path}')
            
    # exporting embedding
    data_list = get_train_data('instances_hard.csv')
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    print(f'[+] exporting GCN embeddings for {len(loader)=} instances')
    for i, data in enumerate(tqdm.tqdm(loader)):
        # graph
        edge_weight, edge_index = all_graphs[data.graph_name[0]]
        # forward
        e = net.embedding(data, edge_index, edge_weight)[0].detach().cpu()
        # save gcn embedding
        path = os.path.join(embed_dir, data.name[0])
        torch.save(e, path)
        print(f'    + saved {e.shape=} at {path}')


if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    prepare_embed(args)