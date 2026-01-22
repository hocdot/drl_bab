import torch.nn as nn
import torch

class Abstraction(nn.Module):

    def __init__(self, net, input_shape, back_sub_steps=1e9, device='cpu'):
        super().__init__()

        self.net = net
        self.input_shape = input_shape
        self.back_sub_steps = back_sub_steps
        self.device = device
        self._build_network_transformer()

    def _build_network_transformer(self):
        last = BatchInputTransformer(self.input_shape).to(self.device)
        layers = [last]
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                last = BatchLinearTransformer(layer, last=last, back_sub_steps=self.back_sub_steps, idx=idx).to(self.device)
            elif isinstance(layer, nn.ReLU):
                last = BatchReLUTransformer(last=last, back_sub_steps=self.back_sub_steps, idx=idx).to(self.device)
                idx += 1
            else:
                print(type(layer))
                raise NotImplementedError
            layers += [last]

        self.layers = nn.Sequential(*layers)

    @torch.no_grad()
    def __call__(self, lower, upper):
        bounds = (lower, upper)
        hidden_params = []
        layer_to_layer_params = []
        for layer in self.layers:
            # print('[+] processing:', layer)
            bounds, params = layer(bounds)
            if params is not None:
                # assert all([_.shape[0] == 2 for _ in params]), f'{layer} has invalid params: {[_.shape for _ in params]}'
                hidden_params.append(params)
            # Get layer-to-layer params if this layer has a preceding layer
            if hasattr(layer, 'get_layer_to_layer_params'):
                l2l_params = layer.get_layer_to_layer_params()
                if l2l_params is not None:
                    layer_to_layer_params.append(l2l_params)
        return (bounds[..., 0].transpose(0, 1), bounds[..., 1].transpose(0, 1)), hidden_params, layer_to_layer_params


class BatchInputTransformer(nn.Module):

    def __init__(self, input_shape, last=None):
        super(BatchInputTransformer, self).__init__()
        self.last = last
        self.input_shape = input_shape

    def forward(self, bounds):
        self.bounds = torch.stack([bounds[0], bounds[1]], dim=2).transpose(0, 1) # H x B x 2
        return self.bounds, None

    def get_layer_to_layer_params(self):
        """Input layer has no preceding layer"""
        return None

    def __str__(self):
        return 'Input'


class BatchLinearTransformer(nn.Module):

    def __init__(self, layer, last=None, back_sub_steps=0, idx=None):
        super(BatchLinearTransformer, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.last = last
        self.back_sub_steps = back_sub_steps
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)
        self.idx = idx
        self.params = None

    def forward(self, bounds):
        upper = self.W_plus @ bounds[..., 1] + self.W_minus @ bounds[..., 0] # H x B
        lower = self.W_plus @ bounds[..., 0] + self.W_minus @ bounds[..., 1] # H x B
        self.bounds  = torch.stack([lower, upper], dim=2) + self.bias[:, None, None] # H x B x 2
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds, self.params

    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps) # H x B x 2
        indl = new_bounds[..., 0] > self.bounds[..., 0]
        indu = new_bounds[..., 1] < self.bounds[..., 1]
        self.bounds[..., 0][indl] = new_bounds[..., 0][indl]
        self.bounds[..., 1][indu] = new_bounds[..., 1][indu]
        self.params = new_params

    def _back_sub(self, max_steps, params=None):
        if params is None:
            params = self.weight.data, self.weight.data, self.bias.data, self.bias.data

        Ml, Mu, bl, bu = params

        if max_steps > 0 and self.last.last is not None:
            if self.last.beta is not None:
                H, B = self.last.beta.shape
                if len(Ml.shape) == 2:
                    Ml = Ml.repeat(B, 1, 1).permute(1, 2, 0) # H, H2, B
                    Mu = Mu.repeat(B, 1, 1).permute(1, 2, 0) # H, H2, B
                    Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0) * self.last.lmbda # H, H2, B
                    Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0) * self.last.beta # H, H2, B
                    blnew = torch.bmm(torch.clamp(Ml.permute(2, 0, 1), max=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl # B x H
                    bunew = torch.bmm(torch.clamp(Mu.permute(2, 0, 1), min=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu # B x H
                else:
                    Ml2 = Ml.permute(1, 2, 0) # H, H2, B
                    Mu2 = Mu.permute(1, 2, 0) # H, H2, B
                    Mlnew = torch.clamp(Ml2, min=0) * self.last.beta + torch.clamp(Ml2, max=0) * self.last.lmbda # H, H2, B
                    Munew = torch.clamp(Mu2, min=0) * self.last.lmbda + torch.clamp(Mu2, max=0) * self.last.beta # H, H2, B
                    blnew = torch.bmm(torch.clamp(Ml, max=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl # B x H
                    bunew = torch.bmm(torch.clamp(Mu, min=0), self.last.mu.transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu # B x H

                Mlnew = Mlnew.permute(2, 0, 1)
                Munew = Munew.permute(2, 0, 1)
                blnew = blnew.permute(1, 0)
                bunew = bunew.permute(1, 0)
                return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
            else:
                return self.last._back_sub(max_steps-1, params=params)
        else:
            if len(Ml.shape) == 2: # first layer
                lower = torch.clamp(Ml, min=0) @ self.last.bounds[..., 0] + torch.clamp(Ml, max=0) @ self.last.bounds[..., 1] + bl[:, None] # H x B
                upper = torch.clamp(Mu, min=0) @ self.last.bounds[..., 1] + torch.clamp(Mu, max=0) @ self.last.bounds[..., 0] + bu[:, None] # H x B
                repeat_params = [p[None].repeat([lower.shape[1]] + [1 for _ in range(p.ndim)]) for p in params]
                return torch.stack([lower, upper], dim=2), repeat_params # H x B x 2
            else:
                lower = torch.bmm(torch.clamp(Ml, min=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Ml, max=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl # B x H
                upper = torch.bmm(torch.clamp(Mu, min=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Mu, max=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu # B x H
                output = torch.stack([lower.transpose(0, 1), upper.transpose(0, 1)], dim=2) # H x B x 2
                return output, params # H x B x 2

    def get_layer_to_layer_params(self):
        """Return linear bounds from preceding layer to this layer"""
        if self.last is None or isinstance(self.last, BatchInputTransformer):
            return None
        # For a linear layer: y = Wx + b
        # Lower bound: y_L = W_plus @ x_L + W_minus @ x_U + b
        # Upper bound: y_U = W_plus @ x_U + W_minus @ x_L + b
        # So the bounds are simply the weight matrix and bias
        W_L = self.weight.data  # H_out x H_in
        W_U = self.weight.data  # H_out x H_in
        b_L = self.bias.data    # H_out
        b_U = self.bias.data    # H_out
        # Repeat for batch dimension to match the format
        B = self.bounds.shape[1]
        W_L_batch = W_L[None].repeat([B] + [1 for _ in range(W_L.ndim)])
        W_U_batch = W_U[None].repeat([B] + [1 for _ in range(W_U.ndim)])
        b_L_batch = b_L[None].repeat([B] + [1 for _ in range(b_L.ndim)])
        b_U_batch = b_U[None].repeat([B] + [1 for _ in range(b_U.ndim)])
        return (W_L_batch, W_U_batch, b_L_batch, b_U_batch)

    def __str__(self):
        return f'Linear {self.idx}'



class BatchReLUTransformer(nn.Module):

    def __init__(self, last=None, back_sub_steps=0, idx=None):
        super(BatchReLUTransformer, self).__init__()
        self.last = last
        self.back_sub_steps = back_sub_steps
        self.idx = idx
        self.params = None

    def forward(self, bounds):
        self.beta = torch.zeros(bounds.shape[0], bounds.shape[1], device=bounds.device)
        device = bounds.device
        ind1 = bounds[..., 1] <= 0 # H x B
        ind2 = bounds[..., 0] > 0 # H x B
        ind3 = (bounds[..., 1] > 0) * (bounds[..., 0] < 0) # H x B
        ind4 = (bounds[..., 1] > -bounds[..., 0]) * ind3

        self.bounds = torch.zeros_like(bounds, device=device) # H x B x 2
        self.bounds[..., 1][ind3] = bounds[..., 1][ind3]
        self.bounds[..., :][ind4] = bounds[..., :][ind4]

        self.lmbda = torch.zeros_like(bounds[..., 1], device=device) # H x B
        self.mu = torch.zeros_like(bounds[..., 1], device=device) # H x B
        self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2], device=device)

        self.beta[ind1] = torch.zeros_like(self.beta[ind1], device=device) # note: inactive indices should be set to 0
        self.beta[ind2] = torch.ones_like(self.beta[ind2], device=device) # note: active indices should be set to 1

        diff = bounds[..., 1][ind3] - bounds[..., 0][ind3]
        self.lmbda[ind3] = torch.div(bounds[..., 1][ind3], diff)
        self.beta[ind4] = torch.ones_like(self.beta[ind4])

        self.mu[ind3] = torch.div(-bounds[..., 0][ind3] * bounds[..., 1][ind3], diff)
        self.bounds[..., :][ind2] = bounds[..., :][ind2]

        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)

        return self.bounds, None

    def get_layer_to_layer_params(self):
        """Return linear bounds from preceding layer to this layer (after ReLU)"""
        if self.last is None or isinstance(self.last, BatchInputTransformer):
            return None
        # For ReLU, the relationship between input x and output y is:
        # Lower bound: y_L = beta * x + 0 (where beta is 0 for inactive, 1 for active)
        # Upper bound: y_U = lambda * x + mu (where lambda is the slope, mu is the offset)
        # Return diagonal matrices for the slopes and the offsets
        device = self.lmbda.device
        B = self.bounds.shape[1]
        # Create diagonal matrices for the linear transformations
        W_L = torch.diag_embed(self.beta.transpose(0, 1))  # B x H x H (diagonal)
        W_U = torch.diag_embed(self.lmbda.transpose(0, 1))  # B x H x H (diagonal)
        b_L = torch.zeros_like(self.mu, device=device)  # H x B
        b_U = self.mu  # H x B
        return (W_L, W_U, b_L, b_U)

    def __str__(self):
        return f'Relu {self.idx}'

    def back_sub(self, max_steps):
        new_bounds, new_params = self._back_sub(max_steps) # H x B x 2
        indl = new_bounds[..., 0] > self.bounds[..., 0]
        indu = new_bounds[..., 1] < self.bounds[..., 1]
        self.bounds[..., 0][indl] = new_bounds[..., 0][indl].clamp(min=0)
        self.bounds[..., 1][indu] = new_bounds[..., 1][indu]
        self.params = new_params

    def _back_sub(self, max_steps, params=None):
        if params is None:
            device = self.lmbda.device
            params = torch.diag_embed(self.beta.transpose(0, 1)), torch.diag_embed(self.lmbda.transpose(0, 1)), torch.zeros_like(self.mu, device=device), self.mu
        Ml, Mu, bl, bu = params
        if max_steps > 0 and self.last.last is not None:
            Mlnew = Ml @ self.last.weight # B x H x H2
            Munew = Mu @ self.last.weight # B x H x H2
            blnew = bl.transpose(0, 1) + (Ml @ self.last.bias) # B x H
            bunew = bu.transpose(0, 1) + (Mu @ self.last.bias) # B x H
            return self.last._back_sub(max_steps-1, params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = torch.bmm(torch.clamp(Ml, min=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Ml, max=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bl.transpose(0, 1)
            upper = torch.bmm(torch.clamp(Mu, min=0), self.last.bounds[..., 1].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + torch.bmm(torch.clamp(Mu, max=0), self.last.bounds[..., 0].transpose(0, 1).unsqueeze(-1)).squeeze(-1) + bu.transpose(0, 1)
            output = torch.stack([lower.transpose(0, 1), upper.transpose(0, 1)], dim=2) # H x B x 2
            return output, params # H x B x 2


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)
        self.relu = nn.ReLU()

        self.layers = [self.linear1, self.relu, self.linear2, self.relu, self.linear3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    net = Net()
    net.eval()
    input_shape = (1, 2)
    # print(net)

    deeppoly = Abstraction(net, input_shape)
    print(deeppoly.layers)

    lb = torch.randn(1, 2)
    ub = lb + 5

    (lb_out, ub_out), hidden_params = deeppoly(lb, ub)

    print(lb_out)
    print(ub_out)

    for i, h in enumerate(hidden_params):
        print(f'Hidden params {i}:', [_.shape for _ in h])
