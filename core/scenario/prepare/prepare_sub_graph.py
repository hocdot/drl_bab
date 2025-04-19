import neuralsat as ns
import networkx as nx
import pandas as pd
import simulator
import pickle
import torch
import math
import os

def load_next_instance(self):
    # extract args
    args = self.args
    # reset neuralsat setting
    ns.test.reset_settings()
    #
    if args.verbose:
        print(f'[+] loading new instance: {self.onnx=} {self.vnnlib=}')
    # load environment
    d           = os.path.join(args.benchmark_dir, args.benchmark)
    onnx_path   = os.path.join(d, 'onnx', self.onnx)
    vnnlib_path = os.path.join(d, 'vnnlib', self.vnnlib)
    net, input_shape, objectives = ns.test.extract_instance(onnx_path, vnnlib_path)
    self.net           = net
    self.input_shape   = input_shape
    self.iv            = ns.InteractiveVerifier(net=net, input_shape=input_shape, batch=args.ns_batch_size, device=args.device)
    objectives, _      = self.iv._preprocess(objectives, force_split=None)
    # TODO: Paul Pogba
    self.objective     = objectives.pop(1)
    # increase instance id counter
    self.instance_id += 1

def extract_sub_graph(G, nodes):
    # Create a new empty graph
    sub_G = nx.DiGraph()
    # Add nodes with data
    for n in nodes:
        if n in G.nodes:
            sub_G.add_node(n, **G.nodes[n])
    # Add edges with data, only between nodes in 'nodes'
    for u in nodes:
        for v in nodes:
            if G.has_edge(u, v):
                sub_G.add_edge(u, v, **G[u][v])
                sub_G.add_edge(v, u, **G[v][u])
    return sub_G

def prepare_sub_graph(args):
    # create env
    env = simulator.Env(args)
    env.load_next_instance = load_next_instance.__get__(env)
    # list all onnx
    for instance in ['train.csv']:
        path    = os.path.join(args.benchmark_dir, args.benchmark, instance)
        df      = pd.read_csv(path)
        onnxs   = df['onnx'].to_list()
        vnnlibs = df['vnnlib'].to_list()
        for onnx, vnnlib in zip(onnxs, vnnlibs):
            print(f'[+] preparing sub-graph for {onnx=} {vnnlib=}')
            env.onnx = onnx
            env.vnnlib = vnnlib
            env.load_next_instance()
            observation, done = env.reset()
            feature, heuristic_score, active_neuron_index, n_active_neuron = observation

            graph_dir = os.path.join(args.benchmark_dir, args.benchmark, 'graph')
            path = os.path.join(graph_dir, f'{onnx}.pkl')
            with open(path, 'rb') as fp:
                G = pickle.load(fp)
            nodes = []
            offset = 0
            assert len(env.hidden_shapes) == len(active_neuron_index)
            for hidden_shape, active_neuron in zip(env.hidden_shapes, active_neuron_index):
                assert active_neuron.shape[0] == 1
                nodes  += (active_neuron[0].detach().cpu().numpy() + offset).tolist()
                offset += math.prod(hidden_shape)
            sub_G = extract_sub_graph(G, nodes)
            # save subgraph
            path = os.path.join(graph_dir, f'{onnx}_{vnnlib}.pkl')
            with open(path, 'wb') as fp:
                pickle.dump(sub_G, fp, protocol=pickle.HIGHEST_PROTOCOL)
                print(f' saved {path}')
