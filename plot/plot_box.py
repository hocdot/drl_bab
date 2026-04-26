from typing import List, Optional

import copy
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .extract_csv import INSTANCES_PER_BENCHMARK, sample_csv_v2
from .plot_cactus import (
    FIGURE_DIR,
    SOLVER_DISPLAY,
    _done_mask,
    _ylabel_for_metric,
    load_merged_sampled,
)

from arguments import get_args

N_SEEDS = 10


def new_seed(base_seed: int, i: int) -> int:
    return int(base_seed) + i * 4


def merge_df_at_least_one_solved(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where at least one solver has ``done`` true (instance in the pooled evaluation set)."""
    m = pd.Series(False, index=df.index)
    for cls, _ in SOLVER_DISPLAY:
        dcol = f'done_{cls}'
        if dcol in df.columns:
            m = m | _done_mask(df[dcol])
    return df.loc[m].reset_index(drop=True)


def collect_solved_counts_long(
    args,
    n_seeds: int = N_SEEDS,
    n_sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each seed, subsample, merge to instances solved by ≥1 tool, then record each solver's
    count of solved instances in that subset (one row per solver per seed).
    """
    if n_sample is None:
        n_sample = INSTANCES_PER_BENCHMARK
    base_seed = args.seed
    records: List[dict] = []
    try:
        for i in range(n_seeds):
            args.seed = new_seed(base_seed, i)
            sample_csv_v2(args, n_sample)
            df = load_merged_sampled()
            df_f = merge_df_at_least_one_solved(df)
            for cls, label in SOLVER_DISPLAY:
                dcol = f'done_{cls}'
                if dcol not in df_f.columns:
                    continue
                n_solved = int(_done_mask(df_f[dcol]).sum())
                records.append({'solver': label, 'n_solved': n_solved, 'seed': args.seed})
    finally:
        args.seed = base_seed
    return pd.DataFrame(records)


def collect_metric_long(
    args,
    metric: str,
    n_seeds: int = N_SEEDS,
    n_sample: Optional[int] = INSTANCES_PER_BENCHMARK,
) -> pd.DataFrame:
    base_seed = args.seed
    records: List[dict] = []
    try:
        for i in range(n_seeds):
            args.seed = new_seed(base_seed, i)
            sample_csv_v2(args, n_sample)
            df = merge_df_at_least_one_solved(load_merged_sampled())
            for cls, label in SOLVER_DISPLAY:
                mcol = f'{metric}_{cls}'
                if metric == 'value':
                    vals = pd.to_numeric(df[mcol], errors='coerce').dropna()
                    vals = -vals
                else:
                    dcol = f'done_{cls}'
                    m = _done_mask(df[dcol])
                    vals = pd.to_numeric(df.loc[m, mcol], errors='coerce').dropna()
                for v in vals:
                    records.append({'solver': label, metric: float(v), 'seed': args.seed})
    finally:
        args.seed = base_seed
    return pd.DataFrame(records)


def plot_box_solved_counts(
    args,
    n_seeds: int = N_SEEDS,
    n_sample: Optional[int] = None,
    fmt: str = 'pdf',
    showfliers: bool = True,
) -> str:
    df_long = collect_solved_counts_long(args, n_seeds=n_seeds, n_sample=n_sample)
    order = [lbl for _, lbl in SOLVER_DISPLAY if lbl in set(df_long['solver'])]
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df_long,
        x='solver',
        y='n_solved',
        order=order,
        palette='muted',
        width=0.55,
        showfliers=showfliers,
    )
    plt.xlabel('')
    plt.ylabel('Solved instances')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f'box_solved_counts.{fmt}')
    plt.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
    plt.close()
    print(f'[+] saved {path}')
    return path


def plot_box_metric(
    args,
    metric: str,
    n_seeds: int = N_SEEDS,
    n_sample: Optional[int] = None,
    fmt: str = 'pdf',
    showfliers: bool = True,
) -> str:
    df_long = collect_metric_long(args, metric, n_seeds=n_seeds, n_sample=n_sample)
    order = [lbl for _, lbl in SOLVER_DISPLAY if lbl in set(df_long['solver'])]
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df_long,
        x='solver',
        y=metric,
        order=order,
        palette='muted',
        width=0.55,
        showfliers=showfliers,
    )
    plt.xlabel('')
    plt.ylabel(_ylabel_for_metric(metric))
    plt.grid(True, alpha=0.3, axis='y')
    if metric == 'value':
        plt.yscale('log')
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f'box_{metric}.{fmt}')
    plt.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
    plt.close()
    print(f'[+] saved {path}')
    return path


def plot_box_all(
    args,
    n_seeds: int = N_SEEDS,
    n_sample: Optional[int] = None,
    fmt: str = 'pdf',
    showfliers: bool = True,
) -> List[str]:
    paths: List[str] = []
    a = copy.deepcopy(args)
    paths.append(plot_box_solved_counts(a, n_seeds=n_seeds, n_sample=n_sample, fmt=fmt, showfliers=showfliers))
    paths.append(plot_box_metric(a, 'value', n_seeds=n_seeds, n_sample=n_sample, fmt=fmt, showfliers=showfliers))
    return paths


if __name__ == '__main__':
    args = get_args()
    args.seed = 9750
    n_sample = 100
    plot_box_all(args, n_sample=n_sample, fmt='png', showfliers=False)
