import torch.nn as nn

class Encoder(nn.Module):

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

    def forward(self, e):
        if self.args.enc_dec_type == 'gru':
            z, h = self.layer(e, None)
            c = None
        elif self.args.enc_dec_type == 'lstm':
            z, (h, c) = self.layer(e, None)
        else:
            raise NotImplementedError
        return z, (h, c)
