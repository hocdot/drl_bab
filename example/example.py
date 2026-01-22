import net, spec, deeppoly

# create net
net = net.create()
net.eval()

# create spec
x_L, x_U, x0, epsilon = spec.create()

# create abstraction
abstraction = deeppoly.Abstraction(net=net, input_shape=x_L.shape)

# linear bound propagation
(y_L, y_U), hidden_params = abstraction(x_L, x_U)

# print intermediate bounds
print('[+] intermediate bounds')
intermediate_bounds = []
for i, (W_L, W_U, b_L, b_U) in enumerate(hidden_params[:-1]):
    # print(f'{W_L.shape=} {x_L.shape=} {b_L.shape=}')
    z_L = W_L[0] @ x_L[0] + b_L[0]
    z_U = W_U[0] @ x_U[0] + b_U[0]
    print(f'{i=} {z_L=} {z_U=}')
    intermediate_bounds.append((z_L, z_U))

# print polarity score
polarity_scores = []
for i, (z_L, z_U) in enumerate(intermediate_bounds):
    omega = (z_L + z_U) / (z_U - z_L)
    print(i, omega)
    polarity_scores.append(omega)
