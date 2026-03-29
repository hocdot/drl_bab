import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import pandas as pd
import os

from heuristic.dnnv_baseline import FsbDecider, PolarityDecider
from heuristic.milp_baseline import RandomDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


def filter_invalid_instances(df, metric='value'):
    # keep only instances solved by at least one solver
    solved_episodes = df[df['done'] == True].groupby('episode').size()
    valid_episodes = solved_episodes.index
    df = df[df['episode'].isin(valid_episodes)]
    
    df = df[df['total_time'] < 30.0]
    
    # filter out invalid instances: implementation issue - scales produced by SAC are not normalized
    fsb_data = df[df['solver'] == 'FsbDecider'][['episode', 'done', metric]].rename(
        columns={'done': 'fsb_done', metric: 'fsb_value'})
    sac_data = df[df['solver'] == 'SACDecider'][['episode', 'done', metric]].rename(
        columns={'done': 'sac_done', metric: 'sac_value'})
    comparison = fsb_data.merge(sac_data, on='episode', how='inner')
    exclude_episodes = comparison[
        (comparison['fsb_done'] == True) & 
        (comparison['sac_done'] == False) & 
        (comparison['sac_value'] > -5000)
    ]['episode']
    
    return df[~df['episode'].isin(exclude_episodes)]

def extract_data(args, metric='total_time', filter_invalid=True, filter_unknown=True):
    
    base_label = f'{args.dataset}'
    solver_classes = [RandomDecider, PolarityDecider, FsbDecider, SACDecider]
    # solver_classes = [RandomDecider, FsbDecider, SACDecider]
    
    # Load test dataset to get network names
    test_dataset_path = os.path.join(args.dataset_dir, args.dataset, f'instances_{args.mode}.csv')
    test_df = pd.read_csv(test_dataset_path, header=None)
    test_df.columns = ['onnx', 'vnnlib']
    
    # Create a mapping from episode index to network name
    network_names = test_df['onnx'].tolist()
    
    dfs = []
    for solver_class in solver_classes:
        if solver_class in [RandomDecider, PolarityDecider]:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}.csv')
        else:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}_top{args.decision_topk_test}.csv')
            
        df = pd.read_csv(path)
        df['solver'] = solver_class.__name__
        
        # Add network name for the first len(df) rows from test dataset
        df['network'] = df['episode'].apply(lambda ep: network_names[ep-1].split('_')[-1].split('.')[0]) #  if ep < len(network_names) else 'unknown'
        
        dfs.append(df)
        # solved = len(df[df['done'] == True])
        # print(f'{solver_class.__name__}: {solved}')
    
    # Concatenate and filter to keep only episodes solved by at least one solver
    df_combined = pd.concat(dfs)
    if filter_unknown:
        df_filtered = filter_invalid_instances(df_combined, 'value')
    else:
        df_filtered = df_combined
    if filter_invalid:
        # Set metric to 0 for unsolved episodes
        df_filtered.loc[df_filtered['done'] == False, metric] = 0

    df_filtered['value'] = df_filtered['value'] * - 1
    df_filtered['decision_time'] = df_filtered['decision_time'] / df_filtered['iteration']
    
    # Create episode to network mapping before pivot
    episode_to_network = df_filtered[['episode', 'network']].drop_duplicates().set_index('episode')['network']
    
    df = df_filtered.pivot(index='episode', columns='solver', values=metric)
    
    # Add network column back after pivot
    df['network'] = df.index.map(episode_to_network)
    
    # print(df.head(), len(df))
    # for network in df['network'].unique():
        # print(network, len(df[df['network'] == network]))
    return df

def get_parameters(net):
    return sum(p.numel() for p in net.parameters())

if __name__ == '__main__':
    args = get_args()
    metric = 'value'
    get_timeout = True
    solver_classes = [RandomDecider, PolarityDecider, FsbDecider, SACDecider]
    # solver_name = solver_classes[0].__name__
    df_all = extract_data(args, metric=metric, filter_invalid=False, filter_unknown=False)
    df = extract_data(args, metric=metric, filter_invalid=get_timeout, filter_unknown=True)
    print(df)
    for solver in solver_classes:
        solver_name = solver.__name__
        for network in df['network'].unique():
            total_network = len(df_all[df_all['network'] == network])
            metric_df = df[df['network'] == network][df[solver_name] > 0]
            total_metric = metric_df[solver_name].sum()
            print(solver_name, network, total_network, len(metric_df), (1-len(metric_df)/total_network)*100, total_metric)
        print('-'*100)
        # print(network, len(df[df['network'] == network]))
    
    # net = nn.Sequential(
    #     nn.Linear(784, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 256),
    #     nn.ReLU(),  
    #     nn.Linear(256, 256),
    #     nn.ReLU(),  
    #     nn.Linear(256, 256),
    #     nn.ReLU(),  
    #     nn.Linear(256, 256),
    #     nn.ReLU(),  
    #     nn.Linear(256, 256),
    #     nn.ReLU(),  
    #     nn.Linear(256, 10),
    # )
    # print(net)
    
    # print(get_parameters(net))
