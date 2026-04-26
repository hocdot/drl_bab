from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import tqdm
import os

# BENCHMARK_TYPES = ('fnn', 'cnn', 'sigmoid')
from .extract_csv import CURRENT_DIR, CSV_EXPORT_DIR, BENCHMARK_TO_CONFIG, sample_csv, sample_csv_v2, extract_csv
from arguments import get_args

SOLVER_DISPLAY = (
    # ('RandomDecider', 'Random'),
    ('PolarityDecider', 'Polarity'),
    ('UpbDecider', 'UPB'),
    ('FsbDecider', 'FSB'),
    ('SACDecider', 'RSB'),
)

FIGURE_DIR = os.path.join(CURRENT_DIR, 'figure')
os.makedirs(FIGURE_DIR, exist_ok=True)

BENCHMARK_PANELS = (
    ('all', None),
    *[(k, k) for k in BENCHMARK_TO_CONFIG],
)

PANEL_TITLE = {
    'all': 'All',
    'fnn': 'FNN',
    'cnn': 'CNN',
    'sigmoid': 'Sigmoid',
}


def _ylabel_for_metric(metric: str) -> str:
    if metric == 'value':
        return 'Explored branches'
    if metric == 'total_time':
        return 'Total time (s)'
    if metric == 'decision_time':
        return 'Decision time (s)'
    if metric == 'iteration':
        return 'Iterations'
    return metric.replace('_', ' ').title()


def load_merged_sampled() -> pd.DataFrame:
    parts = []
    for name in BENCHMARK_TO_CONFIG:
        path = os.path.join(CSV_EXPORT_DIR, f'{name}_sample.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing sampled CSV: {path}')
        df = pd.read_csv(path)
        df['suite'] = name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _done_mask(series: pd.Series) -> pd.Series:
    return series.eq(True) | (series.astype(str).str.lower() == 'true')


def count_solved_by_solver(df: pd.DataFrame) -> dict:
    counts = {}
    for cls, _ in SOLVER_DISPLAY:
        col = f'done_{cls}'
        if col not in df.columns:
            continue
        counts[cls] = int(_done_mask(df[col]).sum())
    return counts


def rsb_strictly_best_on_all_benchmark_suites(df_all: pd.DataFrame) -> bool:
    for suite in BENCHMARK_TO_CONFIG:
        sub = df_all[df_all['suite'] == suite]
        if len(sub) == 0:
            return False
        counts = count_solved_by_solver(sub)
        n_rsb = counts['SACDecider']
        for cls, n in counts.items():
            if cls == 'SACDecider':
                continue
            if n_rsb <= n:
                return False
    return True


def _solvers_for_df(df: pd.DataFrame, metric: str):
    return [(cls, label) for cls, label in SOLVER_DISPLAY if f'{metric}_{cls}' in df.columns]


def _series_for_cactus(df: pd.DataFrame, done_mask: pd.Series, metric_col: str, metric: str) -> pd.Series:
    """Values for solved rows only; ``value`` is stored negative in CSV — use positive branch counts for plotting."""
    raw = pd.to_numeric(df.loc[done_mask, metric_col], errors='coerce').dropna()
    if metric == 'value':
        raw = -raw
    return raw.sort_values()


def _plot_cactus_axes(ax, df: pd.DataFrame, solvers, colors, metric: str, title: Optional[str] = None):
    for idx, (solver_cls_name, plot_name) in enumerate(solvers):
        done_col = f'done_{solver_cls_name}'
        metric_col = f'{metric}_{solver_cls_name}'
        m = _done_mask(df[done_col])
        metrics = _series_for_cactus(df, m, metric_col, metric)
        if len(metrics) == 0:
            continue
        x = np.arange(1, len(metrics) + 1, dtype=float)
        y = metrics.to_numpy()
        if len(metrics) == 1:
            sns.scatterplot(
                x=x, y=y, label=plot_name, s=55, zorder=3, color=colors[idx], ax=ax
            )
        else:
            ax.plot(x, y, label=plot_name, linewidth=2, color=colors[idx])
    ax.set_xlabel('Solved instances')
    ax.set_ylabel(_ylabel_for_metric(metric))
    ax.legend(fancybox=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)


def plot_cactus(
    output_stem: str = 'cactus_time',
    metric: str = 'total_time',
    fmt: str = 'pdf',
    combined_figure: bool = False,
) -> List[str]:
    """
    Cactus plots for all suites merged and per benchmark (fnn / cnn / sigmoid).

    Default: four files ``{output_stem}_{all|fnn|cnn|sigmoid}.{fmt}``.
    If ``combined_figure`` is True: one 2×2 figure ``{output_stem}_panels.{fmt}``
    (separate files use ``{output_stem}_{all|fnn|cnn|sigmoid}.{fmt}``).

    Returns list of written paths.
    """
    df_all = load_merged_sampled()
    solvers = _solvers_for_df(df_all, metric)
    if not solvers:
        raise ValueError(f'No {metric}_* solver columns in merged sample data')
    colors = sns.color_palette('muted', n_colors=len(solvers))

    def subset(suite: Optional[str]) -> pd.DataFrame:
        if suite is None:
            return df_all
        return df_all[df_all['suite'] == suite]

    written: List[str] = []

    if combined_figure:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (key, suite) in zip(np.ravel(axes), BENCHMARK_PANELS):
            _plot_cactus_axes(ax, subset(suite), solvers, colors, metric=metric, title=PANEL_TITLE[key])
        fig.tight_layout()
        path = os.path.join(FIGURE_DIR, f'{output_stem}_all.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
        plt.close(fig)
        written.append(path)
    else:
        for key, suite in BENCHMARK_PANELS:
            fig, ax = plt.subplots(figsize=(6, 4))
            _plot_cactus_axes(ax, subset(suite), solvers, colors, metric=metric, title=PANEL_TITLE[key])
            fig.tight_layout()
            path = os.path.join(FIGURE_DIR, f'{output_stem}_{key}.{fmt}')
            fig.savefig(path, dpi=300, bbox_inches='tight', format=fmt)
            plt.close(fig)
            written.append(path)

    # for p in written:
    #     print(f'[+] saved {p}')
    return written


if __name__ == '__main__':
    args = get_args()
    combined = True
    n_sample = 100
    found = False

    args.seed = 9750
    extract_csv(args)
    sample_csv_v2(args, n_sample, verbose=True)
    
    for metric in ['value', 'total_time']:
        stem = f'cactus_{metric}_sample_{n_sample}'
        plot_cactus(output_stem=stem, metric=metric, fmt='png', combined_figure=combined)
