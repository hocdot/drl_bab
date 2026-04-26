import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import math
import os
import re



from heuristic.dnnv_baseline import FsbDecider, UpbDecider, PolarityDecider
from heuristic.milp_baseline import RandomDecider
from heuristic.sac.sac import SACDecider
from arguments import get_args, load_yaml_config, update_args_from_yaml, create_folders

BENCHMARK_TO_CONFIG = {
    'fnn': 'config/dnnv_fnn.yaml',
    'cnn': 'config/dnnv_cnn.yaml',
    'sigmoid': 'config/dnnv_distshift.yaml',
}

METRIC_COLS = ['value', 'done', 'decision_time', 'iteration', 'total_time', 'active_neuron']

EASY_THRESHOLD = -1000
TIMEOUT_THRESHOLD = -float('inf')
INSTANCES_PER_BENCHMARK = 50

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_EXPORT_DIR = os.path.join(CURRENT_DIR, 'csv')
os.makedirs(CSV_EXPORT_DIR, exist_ok=True)

def eps_from_vnnlib(vnnlib_path) -> float:
    name = os.path.basename(str(vnnlib_path))
    m = re.search(r"delta(\d+\.\d+)", name)
    if m:
        return float(m.group(1))
    if "_seed_" in name:
        prefix = name.split("_seed_", 1)[0]
        return float(prefix.rsplit("_", 1)[-1])
    return math.nan

def _solver_has_placeholder_metrics(df: pd.DataFrame, solver_name: str) -> pd.Series:
    cols = [f'{m}_{solver_name}' for m in ('value', 'done', 'decision_time', 'iteration', 'total_time')]
    if not all(c in df.columns for c in cols):
        return pd.Series(False, index=df.index)
    v, d, dt, it, tt = cols
    done_true = df[d].eq(True) | (df[d].astype(str).str.lower() == 'true')
    return (
        pd.to_numeric(df[v], errors='coerce').eq(0)
        & done_true
        & pd.to_numeric(df[dt], errors='coerce').eq(0)
        & pd.to_numeric(df[it], errors='coerce').eq(0)
        & pd.to_numeric(df[tt], errors='coerce').eq(0)
    )


def filter_invalid_instances(df: pd.DataFrame, solver_classes) -> pd.DataFrame:
    bad = pd.Series(False, index=df.index)
    for sc in solver_classes:
        bad |= _solver_has_placeholder_metrics(df, sc.__name__)
    return df.loc[~bad].reset_index(drop=True)


def filter_easy_fsb_instances(df: pd.DataFrame, threshold: float = EASY_THRESHOLD) -> pd.DataFrame:
    col = f'value_{FsbDecider.__name__}'
    if col not in df.columns:
        return df
    v = pd.to_numeric(df[col], errors='coerce')
    easy = v > threshold
    return df.loc[~easy].reset_index(drop=True)


def _solver_early_abort(df: pd.DataFrame, solver_name: str, max_total_time: float) -> pd.Series:
    d_col = f'done_{solver_name}'
    tt_col = f'total_time_{solver_name}'
    done_true = df[d_col].eq(True) | (df[d_col].astype(str).str.lower() == 'true')
    done_false = ~done_true
    tt = pd.to_numeric(df[tt_col], errors='coerce')
    return done_false & (tt < max_total_time)


def filter_early_abort_instances(df: pd.DataFrame, solver_classes, max_total_time: float = 100.0) -> pd.DataFrame:
    bad = pd.Series(False, index=df.index)
    for sc in solver_classes:
        bad |= _solver_early_abort(df, sc.__name__, max_total_time)
    return df.loc[~bad].reset_index(drop=True)


def get_output_csv_path(args, solver_class):
    base_label = f'{args.dataset}'
    base_path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}')
    prefix = ''
    if solver_class in [RandomDecider, PolarityDecider]:
        # path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}.csv')
        pass
    elif solver_class in [UpbDecider, FsbDecider, SACDecider]:
        prefix += f'_top{args.decision_topk_test}'
        # path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}_top{args.decision_topk_test}.csv')
        if solver_class == SACDecider:
            # path = os.path.join(args.csv_dir, args.mode, f'{base_label}_{solver_class.__name__}_top{args.decision_topk_test}_checkpoint_{args.checkpoint_postfix}.csv')
            if args.checkpoint_postfix:
                prefix += f'_checkpoint_{args.checkpoint_postfix}'
            if args.n_test_fallback > 0:
                prefix += f'_fallback_{args.n_test_fallback}'
    else:
        raise NotImplementedError(f'{solver_class.__name__} is not supported')
    path = f'{base_path}{prefix}.csv'
    assert os.path.exists(path), f'{path} does not exist'
    return path
    

def extract_csv(args):
    solver_classes = [FsbDecider, PolarityDecider, RandomDecider, UpbDecider, SACDecider]
    
    for benchmark, config_file in BENCHMARK_TO_CONFIG.items():
        config = load_yaml_config(config_file)
        custom_args = copy.deepcopy(args)
        custom_args = update_args_from_yaml(custom_args, config)
        create_folders(custom_args)
        
            
        # Load test dataset
        test_instances = pd.read_csv(os.path.join(custom_args.dataset_dir, custom_args.dataset, f'instances_{custom_args.mode}.csv'), names=['onnx', 'vnnlib'])
        instance_keys = test_instances.reset_index().rename(columns={'index': 'instance_idx'})
        instance_keys['episode'] = instance_keys['instance_idx'] + 1
        instance_keys['vnnlib'] = instance_keys['vnnlib'].map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
        instance_keys['network'] = instance_keys['onnx'].map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
        instance_keys['eps'] = instance_keys['vnnlib'].map(eps_from_vnnlib)
        instance_keys = instance_keys[['episode', 'network', 'eps', 'vnnlib']]
        # print(f'{benchmark}\n{instance_keys}')
        
        df_combined = instance_keys
        for solver_class in solver_classes:
            path = get_output_csv_path(custom_args, solver_class)
            # print(f'{solver_class.__name__}: {path}')
            solver_df = pd.read_csv(path)
            solver_df.drop(columns=['active_neuron'], inplace=True)
            # set instances with value < TIMEOUT_THRESHOLD to timeout
            solver_df.loc[solver_df['value'] < TIMEOUT_THRESHOLD, 'done'] = False
            
            # print(solver_df)
            
            present = [c for c in METRIC_COLS if c in solver_df.columns]
            rename = {c: f'{c}_{solver_class.__name__}' for c in present}
            solver_df = solver_df[['episode', *present]].rename(columns=rename)
            # print(solver_df)
            df_combined = df_combined.merge(solver_df, on='episode', how='left')
            
            # print(rename)
            # exit()
            
        ordered_metric_cols = [
            f'{m}_{s.__name__}' for s in solver_classes for m in METRIC_COLS if f'{m}_{s.__name__}' in df_combined.columns
        ]
        df_combined = df_combined[['episode', 'network', 'eps', 'vnnlib', *ordered_metric_cols]]
        df_combined = df_combined.dropna(how='any', subset=ordered_metric_cols)
        
        # filter invalid instances
        df_combined = filter_invalid_instances(df_combined, solver_classes)
        df_combined = filter_early_abort_instances(df_combined, solver_classes)
        df_combined = filter_easy_fsb_instances(df_combined)

        # save to csv
        df_combined.to_csv(os.path.join(CSV_EXPORT_DIR, f'{benchmark}.csv'), index=False)
        print(f'[+] {benchmark}: {len(df_combined)} rows')
        print(f'Exported: {os.path.join(CSV_EXPORT_DIR, f"{benchmark}.csv")}')
        # print(df_combined)
        # print(CURRENT_DIR)
        # exit()
        
            # df = pd.read_csv(path)
            # print(f'{solver_class.__name__}\n{df}')


def sample_csv(args, n_sample: int = INSTANCES_PER_BENCHMARK):
    rng = np.random.default_rng(args.seed)
    for benchmark in BENCHMARK_TO_CONFIG:
        src = os.path.join(CSV_EXPORT_DIR, f'{benchmark}.csv')
        out_path = os.path.join(CSV_EXPORT_DIR, f'{benchmark}_sample.csv')
        assert os.path.exists(src), f'{src} does not exist'
        df = pd.read_csv(src)
        chunks = []
        for (network, eps), g in df.groupby(['network', 'eps'], sort=True):
            g = g.reset_index(drop=True)
            n = min(n_sample, len(g))
            pick = rng.choice(len(g), size=n, replace=False)
            # print(network, eps, n)
            # print(f'\t[+] ({network}, {eps}) picked {n} instances')
            chunks.append(g.iloc[np.sort(pick)])
        out = pd.concat(chunks, ignore_index=True)
        out.to_csv(out_path, index=False)
        # print(f'[+] {benchmark}: {len(df)} rows -> {len(out)} rows -> {out_path}')


def sample_csv_v2(args, n_sample: int = INSTANCES_PER_BENCHMARK, verbose: bool = False):
    """Sample up to ``n_sample`` rows per ``network`` (all eps pooled per network)."""
    rng = np.random.default_rng(args.seed)
    for benchmark in BENCHMARK_TO_CONFIG:
        src = os.path.join(CSV_EXPORT_DIR, f'{benchmark}.csv')
        out_path = os.path.join(CSV_EXPORT_DIR, f'{benchmark}_sample.csv')
        assert os.path.exists(src), f'{src} does not exist'
        df = pd.read_csv(src)
        chunks = []
        for network, g in df.groupby('network', sort=True):
            g = g.reset_index(drop=True)
            n = min(n_sample, len(g))
            pick = rng.choice(len(g), size=n if benchmark != 'sigmoid' else n * 2, replace=False)
            if verbose:
                print(f'\t[+] {network} picked {len(pick)} instances')
            chunks.append(g.iloc[np.sort(pick)])
        out = pd.concat(chunks, ignore_index=True)
        out.to_csv(out_path, index=False)
        # if verbose:
            # print(f'[+] {benchmark}: {len(df)} rows -> {len(out)} rows -> {out_path}')


if __name__ == '__main__':
    args = get_args()
    extract_csv(args)
    sample_csv(args)