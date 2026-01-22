import net, spec, deeppoly, util

# create net
net = net.create()
net.eval()

# create spec
x_L, x_U, x0, epsilon = spec.create()

# create abstraction
abstraction = deeppoly.Abstraction(net=net, input_shape=x_L.shape)

# linear bound propagation
(y_L, y_U), hidden_params, l2l_hidden_params = abstraction(x_L, x_U)

# print intermediate bounds
print('[+] intermediate bounds')
intermediate_bounds = util.get_intermediate_bounds(x_L, x_U, hidden_params)
for i, (z_L, z_U) in enumerate(intermediate_bounds):
    print(f'{i=} {z_L=} {z_U=}')
print()

# print polarity score
print('[+] polarity scores')
polarity_scores = util.get_polarity_scores(intermediate_bounds)
for i, omega in enumerate(polarity_scores):
    print(i, omega)
print()

# print active mask
print('[+] active masks')
active_masks = util.get_active_masks(intermediate_bounds)
for i, m in enumerate(active_masks):
    print(i, m)
print()

# demonstrate layer-to-layer bound propagation
print('[+] intermediate bounds from l2l linear bounds')
prev_z_L, prev_z_U = x_L, x_U
for i, l2l_params in enumerate(l2l_hidden_params):
    z_L, z_U = util.compute_layer_to_layer_bounds(prev_z_L, prev_z_U, l2l_params)
    print(f'{i=} {z_L=} {z_U=}')
    prev_z_L, prev_z_U = z_L, z_U
print()
