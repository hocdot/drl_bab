"""
FNN ablation table in LaTeX: UPB / RSB on UPB / GNN heuristic / RSB without
embedding / RSB one-shot / RSB, all evaluated on one shared FNN instance set
(seed 9750, ``n_sample=100``, at-least-one-solved pool) and aggregated as in
``create_table.py``.

Run from the repo root:  python -m plot.create_table_fnn_ablation

Rewrites ``plot/csv/{fnn,cnn,sigmoid}.csv`` and ``plot/csv/*_sample.csv``.
"""

from __future__ import annotations

import pandas as pd

from .create_table import (
    _agg_solver,
    _coverage_episodes,
    _format_branches,
    _is_sufficient_coverage,
    attach_optional_solvers,
)
from .plot_box import merge_df_at_least_one_solved

# Fixed: the one-shot run was made on this seed's FNN pool.
SEED = 9750
N_SAMPLE = 100
SUITE = "fnn"

METHODS = (
    ("UpbDecider", "UPB"),
    ("SACDeciderUpb", "RSB on UPB"),
    ("GnnDecider", "GNN heuristic"),
    ("SACDeciderZeroEmb", "RSB without embedding"),
    ("SACDeciderOneshot", "RSB one-shot"),
    ("SACDecider", "RSB"),
)
COLUMNS = ("Time (s)", "Branches", "Solved")


def eval_set(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pool rows covered by every method with a usable run, and the usable methods."""
    pool = merge_df_at_least_one_solved(df[df["suite"] == SUITE])
    covs = {cls: _coverage_episodes(pool, cls) for cls, _ in METHODS}
    usable = [cls for cls, cov in covs.items() if _is_sufficient_coverage(cov, len(pool))]
    eps = set.intersection(*(covs[cls] for cls in usable)) if usable else set()
    return pool[pool["episode"].astype(int).isin(eps)].reset_index(drop=True), usable


def _cells(frame: pd.DataFrame, cls: str, usable: list[str]) -> list[str]:
    agg = _agg_solver(frame, cls) if cls in usable else None
    if agg is None:
        return ["--"] * len(COLUMNS)
    tt, br, n_sol = agg
    return [f"{tt:.2f}", _format_branches(br), str(n_sol)]


def make_latex(frame: pd.DataFrame, usable: list[str]) -> str:
    lines = [
        rf"\begin{{tabular}}{{l{'c' * len(COLUMNS)}}}",
        r"\toprule",
        " & ".join(("Method", *COLUMNS)) + r" \\",
        r"\midrule",
    ]
    for cls, label in METHODS:
        lines.append(" & ".join([label, *_cells(frame, cls, usable)]) + r" \\")
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
    df = attach_optional_solvers(load_merged_sampled(), args)
    frame, usable = eval_set(df)
    print(f"\n[+] {SUITE}: {len(frame)} instances\n")
    print(make_latex(frame, usable))
