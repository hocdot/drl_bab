import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns
import os

from .plot_cactus import _plot_cactus_axes, _solvers_for_df, load_merged_sampled
from arguments import get_args
from .extract_csv import extract_csv, sample_csv_v2
from .utils import FIGURE_DIR

SEEN_SUITE = 'fnn'
UNSEEN_SUITES: Tuple[str, ...] = ('cnn', 'sigmoid')
V2_METRICS: Tuple[str, ...] = ('value', 'total_time')


# Fixed layout for single-panel cactus PDFs (e.g. v2 seen/unseen) so LaTeX subfigures share the same height.
CACTUS_V2_FIGSIZE_INCHES = (4.25, 4.25)
CACTUS_V2_SUBPLOT_ADJUST = dict(left=0.20, right=0.94, bottom=0.18, top=0.88)


def finalize_cactus_v2_figure(fig) -> None:
    """Same axes box for every panel; avoid tight bbox crop so heights match in LaTeX."""
    fig.subplots_adjust(**CACTUS_V2_SUBPLOT_ADJUST)

def plot_cactus_v2(
    output_stem: str = 'cactus_v2',
    metrics: Tuple[str, ...] = V2_METRICS,
    fmt: str = 'pdf',
) -> List[str]:
    df_all = load_merged_sampled()
    written: List[str] = []
    for metric in metrics:
        solvers = _solvers_for_df(df_all, metric)
        if not solvers:
            raise ValueError(f'No {metric}_* solver columns in merged sample data')
        colors = sns.color_palette('muted', n_colors=len(solvers))
        seen_df = df_all[df_all['suite'] == SEEN_SUITE]
        unseen_df = df_all[df_all['suite'].isin(UNSEEN_SUITES)]
        fig, ax = plt.subplots(1, 1, figsize=CACTUS_V2_FIGSIZE_INCHES)
        _plot_cactus_axes(ax, seen_df, solvers, colors, metric=metric)
        finalize_cactus_v2_figure(fig)
        path_seen = os.path.join(FIGURE_DIR, f'{output_stem}_{metric}_seen.{fmt}')
        fig.savefig(path_seen, dpi=300, format=fmt)
        plt.close(fig)
        written.append(path_seen)
        fig, ax = plt.subplots(1, 1, figsize=CACTUS_V2_FIGSIZE_INCHES)
        _plot_cactus_axes(ax, unseen_df, solvers, colors, metric=metric)
        finalize_cactus_v2_figure(fig)
        path_unseen = os.path.join(FIGURE_DIR, f'{output_stem}_{metric}_unseen.{fmt}')
        fig.savefig(path_unseen, dpi=300, format=fmt)
        plt.close(fig)
        written.append(path_unseen)
    return written


if __name__ == '__main__':
    args = get_args()
    n_sample = 100
    args.seed = 9750
    extract_csv(args)
    sample_csv_v2(args, n_sample, verbose=True)
    plot_cactus_v2(output_stem=f'cactus', fmt='pdf')
