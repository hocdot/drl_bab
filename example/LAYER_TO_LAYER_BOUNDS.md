# Layer-to-Layer Bounds Documentation

## Overview

The DeepPoly abstraction has been extended to return **layer-to-layer bounds** in addition to the existing **input-to-layer bounds**.

## What Changed

### 1. Modified Return Value

The `Abstraction.__call__()` method now returns three values instead of two:

```python
(y_L, y_U), hidden_params, layer_to_layer_params = abstraction(x_L, x_U)
```

- `(y_L, y_U)`: Output bounds (unchanged)
- `hidden_params`: Linear bounds from **input** to each hidden layer (unchanged)
- `layer_to_layer_params`: **NEW** - Linear bounds from each layer to its **preceding** layer

### 2. Layer-to-Layer Parameters Format

Each entry in `layer_to_layer_params` is a tuple `(W_L, W_U, b_L, b_U)` representing:

- **W_L**: Weight matrix for lower bound (shape: `B x H_out x H_in`)
- **W_U**: Weight matrix for upper bound (shape: `B x H_out x H_in`)
- **b_L**: Bias vector for lower bound
- **b_U**: Bias vector for upper bound

Where:
- `B` is the batch size
- `H_in` is the input dimension to the layer
- `H_out` is the output dimension of the layer

### 3. Bound Computation

Given bounds `(x_L, x_U)` from the previous layer, the bounds for the current layer are computed as:

```
y_L = W_L_plus @ x_L + W_L_minus @ x_U + b_L
y_U = W_U_plus @ x_U + W_U_minus @ x_L + b_U
```

Where:
- `W_L_plus = clamp(W_L, min=0)` (positive part)
- `W_L_minus = clamp(W_L, max=0)` (negative part)
- `W_U_plus = clamp(W_U, min=0)` (positive part)
- `W_U_minus = clamp(W_U, max=0)` (negative part)

## Layer Types

### Linear Layers

For a linear layer `y = Wx + b`:
- `W_L = W_U = W` (the weight matrix)
- `b_L = b_U = b` (the bias vector)

### ReLU Layers

For a ReLU layer, the bounds are represented by diagonal matrices:
- `W_L = diag(beta)` where `beta[i]` is 0 (inactive) or 1 (active)
- `W_U = diag(lambda)` where `lambda[i]` is the upper bound slope
- `b_L = 0` (no offset for lower bound)
- `b_U = mu` (offset for upper bound)

## Usage Example

```python
# Get bounds
(y_L, y_U), hidden_params, layer_to_layer_params = abstraction(x_L, x_U)

# Propagate bounds layer by layer
prev_L, prev_U = x_L, x_U
for i, l2l_params in enumerate(layer_to_layer_params):
    curr_L, curr_U = util.compute_layer_to_layer_bounds(prev_L, prev_U, l2l_params)
    print(f'Layer {i}: bounds = ({curr_L}, {curr_U})')
    prev_L, prev_U = curr_L, curr_U
```

## Key Differences

| Aspect | Input-to-Layer Bounds | Layer-to-Layer Bounds |
|--------|----------------------|----------------------|
| **What it represents** | Linear relation from **input** to each layer | Linear relation from **previous layer** to current layer |
| **Use case** | Global analysis from input | Local layer-wise analysis |
| **Stored in** | `hidden_params` | `layer_to_layer_params` |
| **Backsubstitution** | Yes (through all layers) | No (only one layer back) |

## Implementation Details

### New Methods

Three new methods were added:

1. `BatchInputTransformer.get_layer_to_layer_params()` - Returns `None` (no preceding layer)
2. `BatchLinearTransformer.get_layer_to_layer_params()` - Returns the weight and bias matrices
3. `BatchReLUTransformer.get_layer_to_layer_params()` - Returns diagonal matrices representing ReLU relaxation

### Utility Function

A helper function `compute_layer_to_layer_bounds()` was added to `util.py` to compute bounds using layer-to-layer parameters.

## Network Structure Example

For a network: `Input -> Linear1 -> ReLU1 -> Linear2 -> ReLU2 -> Linear3`

The `layer_to_layer_params` contains 4 entries:
- Entry 0: Linear1 (Input → Hidden1)
- Entry 1: ReLU1 (Hidden1_pre → Hidden1_post)
- Entry 2: Linear2 (Hidden1 → Hidden2)
- Entry 3: ReLU2 (Hidden2_pre → Hidden2_post)

Note: Linear3 (output layer) is not included as it has no following layer.
