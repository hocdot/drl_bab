"""
Build LaTeX tabulars from merged sample CSVs (``suite`` in ``fnn`` / ``cnn`` / ``sigmoid``).

Example (paper-style):

# \\begin{tabular}{c ccc ccc ccc}
# \\toprule
# & \\multicolumn{3}{c}{FNN} & \\multicolumn{3}{c}{CNN} & \\multicolumn{3}{c}{Sigmoid} \\
# \\midrule
# Method & Time(s) & Branches & Solved & Time(s) & Branches & Solved & Time(s) & Branches & Solved \\
# ...
"""

import pandas as pd

from .plot_box import merge_df_at_least_one_solved
from .plot_cactus import _done_mask
from .utils import PANEL_TITLE, SOLVER_DISPLAY

# Column groups left-to-right: FNN, CNN, Sigmoid
_BENCH_KEYS = ("fnn", "cnn", "sigmoid")


def _format_branches(n: int) -> str:
    """Compact branch count: ``18M`` for millions, ``1.2K`` for thousands, else the integer."""
    n = int(round(n))
    if n >= 1_000_000:
        v = n / 1_000_000.0
        t = f"{v:.1f}".rstrip("0").rstrip(".")
        return t + "M"
    if n >= 1_000:
        v = n / 1_000.0
        t = f"{v:.1f}".rstrip("0").rstrip(".")
        return t + "K"
    return str(n)


def _agg_solver(pool: pd.DataFrame, cls: str) -> tuple[float, int, int]:
    """
    ``total_time`` and ``value`` (branches) are summed over *every* row in the pool for this
    solver, including ``done=False`` (e.g. timeout runs). ``Solved`` is still the number of
    rows with ``done=True`` for this solver.
    """
    dcol, tcol, vcol = f"done_{cls}", f"total_time_{cls}", f"value_{cls}"
    n = int(_done_mask(pool[dcol]).sum())
    tsum = float(pd.to_numeric(pool[tcol], errors="coerce").sum())
    bsum = float((-pd.to_numeric(pool[vcol], errors="coerce")).sum())
    b_int = int(round(bsum))
    return tsum, b_int, n


def create_table(df: pd.DataFrame) -> str:
    lines: list[str] = [
        r"\begin{tabular}{c ccc ccc ccc}",
        r"\toprule",
    ]
    mc = " & ".join("\\multicolumn{3}{c}{" + PANEL_TITLE[k] + "}" for k in _BENCH_KEYS)
    lines.append(" & " + mc + " \\\\")
    lines.append(r"\midrule")
    lines.append(
        "Method & Time(s) & Branches & Solved & Time(s) & Branches & Solved & Time(s) & Branches & Solved \\\\"
    )
    lines.append(
        r"\cmidrule(lr){1-1} \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}"
    )

    for cls, method_tex in SOLVER_DISPLAY:
        cells: list[str] = [method_tex]
        for key in _BENCH_KEYS:
            sub = df[df["suite"] == key]
            pool = merge_df_at_least_one_solved(sub)
            tt, br, n_sol = _agg_solver(pool, cls)
            cells.append(f"{tt:.2f}")
            cells.append(_format_branches(br))
            cells.append(str(int(n_sol)))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    from .plot_box import load_merged_subsampled
    from arguments import get_args

    args = get_args()
    args.seed = 9750
    n_sample = 100
    df = load_merged_subsampled(args, n_sample)
    print(create_table(df))
