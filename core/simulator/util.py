import matplotlib.pyplot as plt
import networkx as nx
import itertools
import simulator
import torch
import os

def visualize_graph(G, args):
    # Draw the graph with labels
    nx.draw(G, pos=nx.spring_layout(G, seed=42),
               with_labels=True,
               labels=nx.get_node_attributes(G, 'label'))
    # save graph visualization
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(path)

def prepare_graph(net, args):
    # create networkx graph from net.parameters()
    G = nx.DiGraph()
    node_to_id = {}
    v = 0
    i = 0
    for name, param in list(net.named_parameters())[:-2]:
        # for 1st just create the node with label f'{layer_id}_{neuron_id}'
        if 'weight' in name:
            # print(name, param.shape)
            n_out, n_in = param.shape
            for j in range(n_out):
                G.add_node(f'{i}_{j}')
                node_to_id[f'{i}_{j}'] = v
                v += 1
            if i > 0:
                for j1, j2 in itertools.product(range(n_in), range(n_out)):
                    G.add_edge(f'{i-1}_{j1}', f'{i}_{j2}', label=param[j2, j1])
                    G.add_edge(f'{i}_{j2}', f'{i-1}_{j1}', label=param[j2, j1])
            i += 1
    # convert G to edge_index and edge_label
    edge_index = []
    edge_weight = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_id[u], node_to_id[v]])
        edge_weight.append(data.get('label'))
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_weight = torch.tensor(edge_weight)
    # avoid negative edge_weight
    edge_weight = torch.abs(edge_weight)
    return edge_index, edge_weight
