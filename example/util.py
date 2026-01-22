import torch

def get_intermediate_bounds(x_L, x_U, hidden_params):
    intermediate_bounds = []
    for i, (W_L, W_U, b_L, b_U) in enumerate(hidden_params[:-1]):
        # print(f'{W_L.shape=} {x_L.shape=} {b_L.shape=}')
        z_L = W_L[0] @ x_L[0] + b_L[0]
        z_U = W_U[0] @ x_U[0] + b_U[0]
        # print(f'{i=} {z_L=} {z_U=}')
        intermediate_bounds.append((z_L, z_U))
    return intermediate_bounds

def get_polarity_scores(intermediate_bounds):
    polarity_scores = []
    for i, (z_L, z_U) in enumerate(intermediate_bounds):
        omega = (z_L + z_U) / (z_U - z_L)
        # print(i, omega)
        polarity_scores.append(omega)
    return polarity_scores

def get_active_masks(intermediate_bounds):
    active_masks = []
    for i, (z_L, z_U) in enumerate(intermediate_bounds):
        m = torch.logical_and(z_L < 0, 0 < z_U)
        # print(m.shape)
        active_masks.append(m)
    return active_masks
