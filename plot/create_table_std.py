from __future__ import annotations

import pandas as pd

from .create_table import _agg_solver
from .plot_box import N_SEEDS, load_merged_subsampled, new_seed
from .utils import PANEL_TITLE, POOL_EXCLUDE_SOLVERS, SOLVER_DISPLAY

BASE_SEED = 9750
N_SAMPLE = 100
_BENCH_KEYS = ("fnn", "cnn", "sigmoid")
_METRICS = ["time", "branches", "solved"]

# Polarity / Upb / Fsb / Rsb (post-hoc ablation solvers excluded).
MAIN_SOLVERS = tuple((cls, label) for cls, label in SOLVER_DISPLAY if cls not in POOL_EXCLUDE_SOLVERS)


def collect_per_seed(args, n_seeds: int = N_SEEDS, n_sample: int = N_SAMPLE) -> pd.DataFrame:
    """One row per (seed, suite, solver): summed time, summed branches, #solved on the pool."""
    base_seed = args.seed
    rows = []
    try:
        for i in range(n_seeds):
            args.seed = new_seed(base_seed, i)
            pool = load_merged_subsampled(args, n_sample)
            for suite in _BENCH_KEYS:
                sub = pool[pool["suite"] == suite]
                for cls, _ in MAIN_SOLVERS:
                    agg = _agg_solver(sub, cls)
                    if agg is None:
                        continue
                    tt, br, n_sol = agg
                    rows.append(
                        {"seed": args.seed, "suite": suite, "solver": cls,
                         "time": tt, "branches": float(br), "solved": float(n_sol)}
                    )
    finally:
        args.seed = base_seed
    return pd.DataFrame(rows)


def _num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _pm(mean: str, std: str) -> str:
    return rf"{mean} {{\scriptsize$\pm$ {std}}}"


def _branches(mean: float, std: float) -> str:
    """Mean and std share one unit (M / K / raw), chosen by the mean."""
    for scale, unit in ((1_000_000, "M"), (1_000, "K")):
        if abs(mean) >= scale:
            return _pm(_num(mean / scale) + unit, _num(std / scale) + unit)
    return _pm(str(int(round(mean))), str(int(round(std))))


def _cells(stats: pd.DataFrame, cls: str, suite: str) -> list[str]:
    sub = stats[(stats["solver"] == cls) & (stats["suite"] == suite)]
    if sub.empty:
        return ["--", "--", "--"]
    m = sub[_METRICS].mean()
    s = sub[_METRICS].std(ddof=1).fillna(0.0)
    return [
        _pm(f"{m['time']:.2f}", f"{s['time']:.2f}"),
        _branches(m["branches"], s["branches"]),
        _pm(f"{m['solved']:.1f}", f"{s['solved']:.1f}"),
    ]


def make_latex(stats: pd.DataFrame) -> str:
    groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{PANEL_TITLE[k]}}}" for k in _BENCH_KEYS)
    lines = [
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        f" & {groups} \\\\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        "Method" + " & Time(s) & Branches & Solved" * len(_BENCH_KEYS) + r" \\",
        r"\midrule",
    ]
    for cls, label in MAIN_SOLVERS:
        cells = [label] + [c for suite in _BENCH_KEYS for c in _cells(stats, cls, suite)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    from arguments import get_args
    from .extract_csv import extract_csv

    args = get_args()
    args.seed = BASE_SEED
    extract_csv(args)
    stats = collect_per_seed(args)
    print(f"[+] seeds {BASE_SEED}..{BASE_SEED + N_SEEDS - 1}, n_sample={N_SAMPLE}, {len(stats)} records\n")
    print(make_latex(stats))
