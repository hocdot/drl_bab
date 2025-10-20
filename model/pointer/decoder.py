import torch.nn as nn
import torch

class Decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.enc_dec_type == 'gru':
            layer_type = nn.GRU
        elif args.enc_dec_type == 'lstm':
            layer_type = nn.LSTM
        else:
            raise NotImplementedError
        self.layer = layer_type(
            input_size=args.n_embed,
            hidden_size=args.n_hidden,
            batch_first=True,
            device=args.device,
            num_layers=args.n_enc_dec_layer,
        )
        self.g = nn.Parameter(torch.FloatTensor(args.n_embed).to(args.device))
        self.device = args.device

    def get_z0(self, x):
        bs, _, _ = x.shape
        z0 = self.g.unsqueeze(0).repeat(bs, 1).unsqueeze(1).to(self.device)
        return z0

    def gather_z(self, e, a):
        index = a.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.args.n_embed)
        z = torch.gather(e, dim=1, index=index)
        return z

    def forward(self, z, h, c):
        if self.args.enc_dec_type == 'gru':
            z, h = self.layer(z, h)
        elif self.args.enc_dec_type == 'lstm':
            z, (h, c) = self.layer(z, (h, c))
        else:
            raise NotImplementedError
        return z, (h, c)
