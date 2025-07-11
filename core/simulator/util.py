import torch

def split_subproblem(x, output_indices):
    new_x = []
    n_input_subproblem = x.shape[0]
    for output_idx in output_indices:
        input_idx = output_idx % n_input_subproblem
        new_x.append(x[input_idx])
        # TODO: split with left right condition
    return torch.stack(new_x)
