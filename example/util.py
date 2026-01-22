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

def compute_layer_to_layer_bounds(prev_layer_L, prev_layer_U, layer_to_layer_params):
    """
    Compute bounds for current layer given bounds from previous layer.
    
    Args:
        prev_layer_L: Lower bounds of previous layer (shape: batch_size x hidden_dim)
        prev_layer_U: Upper bounds of previous layer (shape: batch_size x hidden_dim)
        layer_to_layer_params: Tuple of (W_L, W_U, b_L, b_U) representing the linear bounds
            W_L, W_U: Weight matrices (shape: batch_size x out_dim x in_dim)
            b_L, b_U: Bias vectors (shape: varies depending on layer type)
    
    Returns:
        curr_layer_L: Lower bounds of current layer
        curr_layer_U: Upper bounds of current layer
    """
    W_L, W_U, b_L, b_U = layer_to_layer_params
    
    # For lower bound: y_L = W_L_plus @ x_L + W_L_minus @ x_U + b_L
    # For upper bound: y_U = W_U_plus @ x_U + W_U_minus @ x_L + b_U
    
    # Handle batch dimension
    if len(W_L.shape) == 3:  # B x H_out x H_in
        W_L_plus = torch.clamp(W_L, min=0)
        W_L_minus = torch.clamp(W_L, max=0)
        W_U_plus = torch.clamp(W_U, min=0)
        W_U_minus = torch.clamp(W_U, max=0)
        
        # Reshape prev_layer bounds to match: B x H_in x 1
        prev_L_reshaped = prev_layer_L.unsqueeze(-1)  # B x H_in x 1
        prev_U_reshaped = prev_layer_U.unsqueeze(-1)  # B x H_in x 1
        
        curr_layer_L = torch.bmm(W_L_plus, prev_L_reshaped).squeeze(-1) + \
                       torch.bmm(W_L_minus, prev_U_reshaped).squeeze(-1)
        curr_layer_U = torch.bmm(W_U_plus, prev_U_reshaped).squeeze(-1) + \
                       torch.bmm(W_U_minus, prev_L_reshaped).squeeze(-1)
        
        # Add bias - handle different bias shapes
        if len(b_L.shape) == 2:
            if b_L.shape[0] == curr_layer_L.shape[1]:  # H x B format
                curr_layer_L = curr_layer_L + b_L.transpose(0, 1)
                curr_layer_U = curr_layer_U + b_U.transpose(0, 1)
            else:  # B x H format
                curr_layer_L = curr_layer_L + b_L
                curr_layer_U = curr_layer_U + b_U
    else:  # 2D weights
        W_L_plus = torch.clamp(W_L, min=0)
        W_L_minus = torch.clamp(W_L, max=0)
        W_U_plus = torch.clamp(W_U, min=0)
        W_U_minus = torch.clamp(W_U, max=0)
        
        curr_layer_L = (W_L_plus @ prev_layer_L.T + W_L_minus @ prev_layer_U.T + b_L.unsqueeze(-1)).T
        curr_layer_U = (W_U_plus @ prev_layer_U.T + W_U_minus @ prev_layer_L.T + b_U.unsqueeze(-1)).T
    
    return curr_layer_L, curr_layer_U
