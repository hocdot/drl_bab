import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from heuristic.dnnv_baseline import FsbDecider, PolarityDecider
from heuristic.milp_baseline import RandomDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args


def filter_invalid_instances(df, metric='value'):
    # keep only instances solved by at least one solver
    solved_episodes = df[df['done'] == True].groupby('episode').size()
    valid_episodes = solved_episodes.index
    df = df[df['episode'].isin(valid_episodes)]
    
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

def extract_data(args, metric='total_time', filter_invalid=True):
    
    base_label = f'{args.dataset}'
    solver_classes = [RandomDecider, PolarityDecider, FsbDecider, SACDecider]
    # solver_classes = [RandomDecider, FsbDecider, SACDecider]
    dfs = []
    for solver_class in solver_classes:
        if solver_class in [RandomDecider, PolarityDecider]:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}.csv')
        else:
            path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}_top{args.decision_topk_test}.csv')
            
        df = pd.read_csv(path)
        df['solver'] = solver_class.__name__
        dfs.append(df)
        # solved = len(df[df['done'] == True])
        # print(f'{solver_class.__name__}: {solved}')
    
    # Concatenate and filter to keep only episodes solved by at least one solver
    df_combined = pd.concat(dfs)
    df_filtered = filter_invalid_instances(df_combined, 'value')
    if filter_invalid:
        # Set metric to 0 for unsolved episodes
        df_filtered.loc[df_filtered['done'] == False, metric] = 0

    df_filtered['value'] = df_filtered['value'] * - 1
    
    df = df_filtered.pivot(index='episode', columns='solver', values=metric)
    print(df.head(), len(df))
    for solver in solver_classes:
        solved = len(df[df[solver.__name__] > 0])
        print(f'{solver.__name__}: {solved}')
        
    return df


def plot_cactus(args, metric='total_time'):
    df = extract_data(args, metric)
    
    plt.figure(figsize=(5, 3))
    
    # Set color palette
    colors = sns.color_palette("muted", n_colors=len(df.columns))
    
    for idx, solver in enumerate(df.columns):
        values = df[solver][df[solver] > 0].sort_values() # excluding unsolved instances (0)
        x = list(range(1, len(values) + 1))
        
        # if len(values) == 0:
        #     # Add empty plot for legend entry only
        #     plt.plot([], [], label=solver, linewidth=2, color=colors[idx])
        # el
        if len(values) == 1:
            # Use scatter plot for single point
            sns.scatterplot(x=x, y=values.values, label=solver, s=50, zorder=3, color=colors[idx])
        else:
            # Use line plot for multiple points
            plt.plot(x, values.values, label=solver, linewidth=2, color=colors[idx])
    
    plt.xlabel('Solved Instances')
    plt.ylabel(f'{metric.replace("_", " ").title()}')
    plt.legend(fancybox=True, loc='upper center')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(args.figure_dir, f'cactus_{args.dataset}_{metric}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'[+] saved {output_path=}')
    
def plot_bar_improvement(args, metric='value'):
    df = extract_data(args, metric)
    
    # Filter to keep only instances solved by both SAC and FSB
    # common_solved = df[(df['SACDecider'] > 0) & (df['FsbDecider'] > 0)].copy()
    common_solved = df.copy()
    
    # Calculate improvement: SAC - FSB (positive = SAC is better/higher)
    common_solved['improvement'] = common_solved['SACDecider'] - common_solved['FsbDecider']
    
    # Sort by improvement for better visualization
    common_solved = common_solved.sort_values('improvement')
    
    # Create colors: muted red for negative (degradation), muted green for positive (improvement)
    muted_palette = sns.color_palette('muted')
    muted_red = muted_palette[3]  # red
    muted_green = muted_palette[2]  # green
    colors = [muted_red if x < 0 else muted_green for x in common_solved['improvement']]
    
    plt.figure(figsize=(6, 3))
    bars = plt.bar(range(len(common_solved)), common_solved['improvement'], color=colors, width=1.0)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.xlabel('Instance')
    plt.ylabel(f'{metric.replace("_", " ").title()} Improvement')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(args.figure_dir, f'bar_improvement_{args.dataset}_{metric}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'[+] saved {output_path=}')
    
    # Print statistics
    positive = len(common_solved[common_solved['improvement'] > 0])
    negative = len(common_solved[common_solved['improvement'] < 0])
    print(f'Improvement: {positive}, Degradation: {negative}')
    
def plot_box(args, metric='value', filter_invalid=False, showfliers=False):
    df = extract_data(args, metric, filter_invalid=filter_invalid)
    
    # Reshape data for seaborn: convert from wide to long format
    df_long = df.reset_index().melt(id_vars='episode', var_name='solver', value_name=metric)
    
    # Filter out unsolved instances (value = 0)
    df_long = df_long[df_long[metric] > 0]
    
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df_long, x='solver', y=metric, palette='muted', showfliers=showfliers, width=0.5)
    
    plt.xlabel('')
    plt.ylabel(f'{metric.replace("_", " ").title()}')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(args.figure_dir, f'box_{args.dataset}_{metric}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'[+] saved {output_path=}')
    
if __name__ == '__main__':
    args = get_args()
    plot_cactus(args, metric='total_time')
    plot_cactus(args, metric='value')
    plot_bar_improvement(args, metric='iteration')
    # plot_bar_improvement(args, metric='value')
    plot_box(args, metric='value', showfliers=False)
    plot_box(args, metric='active_neuron', filter_invalid=True, showfliers=True)
    # plot_box(args, metric='total_time')
    # plot_box(args, metric='iteration')