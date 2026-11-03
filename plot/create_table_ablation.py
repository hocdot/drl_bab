"""
Ablation table in LaTeX: GNN heuristic / RSB no-embedding / RSB one-shot / RSB.

Same data and aggregation as the "Embedding ablation" and "One-shot ablation"
tables of ``create_table.py`` (seed 9750, ``n_sample=100``, at-least-one-solved
pool), merged into one table evaluated on a shared episode set.

Run from the repo root:  python -m plot.create_table_ablation

Rewrites ``plot/csv/{fnn,cnn,sigmoid}.csv`` and ``plot/csv/*_sample.csv``.
"""

from __future__ import annotations

import pandas as pd

from .create_table import (
    _agg_solver,
    _coverage_episodes,
    _format_branches,
    _gnn_csv_path,
    _is_sufficient_coverage,
    _oneshot_csv_path,
    _zeroemb_csv_path,
    attach_solver_by_episode,
)
from .plot_box import merge_df_at_least_one_solved
from .utils import PANEL_TITLE

# Fixed: the one-shot / sample-filtered runs were made on this seed's sample.
SEED = 9750
N_SAMPLE = 100
_BENCH_KEYS = ("fnn", "cnn", "sigmoid")

METHODS = (
    ("GnnDecider", "GNN heuristic"),
    ("SACDeciderZeroEmb", "RSB no-embedding"),
    ("SACDeciderOneshot", "RSB one-shot"),
    ("SACDecider", "RSB"),
)


def _suite_eval(df: pd.DataFrame, suite: str) -> tuple[pd.DataFrame | None, dict[str, bool]]:
    """Pool rows covered by every method with a usable run, and which methods are usable."""
    pool = merge_df_at_least_one_solved(df[df["suite"] == suite])
    covs = {cls: _coverage_episodes(pool, cls) for cls, _ in METHODS}
    ok = {cls: _is_sufficient_coverage(cov, len(pool)) for cls, cov in covs.items()}
    present = [covs[cls] for cls, _ in METHODS if ok[cls]]
    if not present:
        return None, ok
    eps = set.intersection(*present)
    return pool[pool["episode"].astype(int).isin(eps)].reset_index(drop=True), ok


def make_latex(df: pd.DataFrame) -> str:
    evals = {suite: _suite_eval(df, suite) for suite in _BENCH_KEYS}
    groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{PANEL_TITLE[k]}}}" for k in _BENCH_KEYS)
    lines = [
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        f" & {groups} \\\\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        "Method" + " & Time(s) & Branches & Solved" * len(_BENCH_KEYS) + r" \\",
        r"\midrule",
    ]
    for cls, label in METHODS:
        cells = [label]
        for suite in _BENCH_KEYS:
            frame, ok = evals[suite]
            agg = _agg_solver(frame, cls) if frame is not None and ok[cls] else None
            if agg is None:
                cells.extend(["--", "--", "--"])
            else:
                tt, br, n_sol = agg
                cells.extend([f"{tt:.2f}", _format_branches(br), str(n_sol)])
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    from arguments import get_args
    from .extract_csv import extract_csv, sample_csv_v2
    from .plot_cactus import load_merged_sampled

    args = get_args()
    args.seed = SEED
    extract_csv(args)
    sample_csv_v2(args, N_SAMPLE)
    df = load_merged_sampled()
    df = attach_solver_by_episode(df, args, "GnnDecider", _gnn_csv_path)
    df = attach_solver_by_episode(df, args, "SACDeciderZeroEmb", _zeroemb_csv_path)
    df = attach_solver_by_episode(df, args, "SACDeciderOneshot", _oneshot_csv_path)
    print()
    print(make_latex(df))
