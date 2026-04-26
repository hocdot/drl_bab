from typing import List, Tuple

from .plot_cactus import *
from .plot_cactus import _plot_cactus_axes, _solvers_for_df

SEEN_SUITE = 'fnn'
UNSEEN_SUITES: Tuple[str, ...] = ('cnn', 'sigmoid')
V2_METRICS: Tuple[str, ...] = ('value', 'total_time')


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
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        _plot_cactus_axes(ax, seen_df, solvers, colors, metric=metric, title='Seen networks')
        fig.tight_layout()
        path_seen = os.path.join(FIGURE_DIR, f'{output_stem}_{metric}_seen.{fmt}')
        fig.savefig(path_seen, dpi=300, bbox_inches='tight', format=fmt)
        plt.close(fig)
        written.append(path_seen)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        _plot_cactus_axes(ax, unseen_df, solvers, colors, metric=metric, title='Unseen networks')
        fig.tight_layout()
        path_unseen = os.path.join(FIGURE_DIR, f'{output_stem}_{metric}_unseen.{fmt}')
        fig.savefig(path_unseen, dpi=300, bbox_inches='tight', format=fmt)
        plt.close(fig)
        written.append(path_unseen)
    return written


if __name__ == '__main__':
    args = get_args()
    n_sample = 100
    args.seed = 9750
    extract_csv(args)
    sample_csv_v2(args, n_sample, verbose=True)
    plot_cactus_v2(output_stem=f'cactus', fmt='png')
