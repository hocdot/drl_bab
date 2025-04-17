import numpy as np
import torch

from ..base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args): # share
        super().__init__(args)

    def select_action(self, observation):
        # observation: list of Layer_i: [bs, n_active_neuron_i, n_feature]
        # extract data
        args    = self.args
        feature, heuristic_score, active_neuron_index, n_active_neuron = observation
        # extract hyperparam
        n_layer = len(feature)
        bs      = feature[0].shape[0]
        # initialize action, action.shape = (bs, 2)
        best_scores, best_neuron_indices = [], []
        for layer in range(n_layer):
            # score                  = - feature[layer][..., 1].cpu() # bs, n_active_neuron_i
            score                  = heuristic_score[layer].cpu()
            best_score, best_index = torch.max(score, dim=1)
            best_neuron_index      = torch.tensor([active_neuron_index[layer][b, best_index[b]] for b in range(bs)])
            best_scores.append(best_score)
            best_neuron_indices.append(best_neuron_index)
        best_scores         = torch.stack(best_scores).T         # (bs, n_layer)
        best_neuron_indices = torch.stack(best_neuron_indices).T # (bs, n_layer)
        best_layer          = torch.argmax(best_scores, dim=1)   # (bs, )
        best_neuron         = torch.gather(best_neuron_indices, 1, best_layer.unsqueeze(-1)).squeeze(-1)
        action = np.array([[best_layer[i].item(), best_neuron[i].item()] for i in range(bs)])
        if args.verbose:
            # print(f'{action=}')
            pass
        return np.array(action)
