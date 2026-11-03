"""
Build LaTeX / Markdown tables from merged sample CSVs (``fnn`` / ``cnn`` / ``sigmoid``).

Optional solvers (GNN, RSB-upb, zero-emb, oneshot) are joined post-hoc by episode.
Filtered infer CSVs (episodes ``1..N``) are remapped to original sample episodes.
"""

from __future__ import annotations

import copy
import os
from typing import Callable

import pandas as pd

from .plot_box import merge_df_at_least_one_solved
from .plot_cactus import _done_mask
from .utils import (
    BENCHMARK_TO_CONFIG,
    METRIC_COLS,
    PANEL_TITLE,
    TIMEOUT_THRESHOLD,
)
from arguments import load_yaml_config, update_args_from_yaml, create_folders

_BENCH_KEYS = ("fnn", "cnn", "sigmoid")

# Ablation tables: (title, methods, use_solved_pool).
ABLATION_TABLES = (
    (
        "Embedding ablation",
        (
            ("SACDeciderZeroEmb", "RSB without embedding"),
            ("SACDecider", "RSB"),
        ),
        True,
    ),
    (
        "UPB backend ablation",
        (
            ("UpbDecider", "UPB"),
            ("SACDeciderUpb", "RSB on UPB"),
        ),
        False,
    ),
    (
        "GNN vs RSB",
        (
            ("GnnDecider", "GNN"),
            ("SACDecider", "RSB"),
        ),
        True,
    ),
    (
        "One-shot ablation",
        (
            ("GnnDecider", "GNN heuristic"),
            ("SACDeciderOneshot", "RSB one-shot"),
            ("SACDecider", "RSB"),
        ),
        True,
    ),
)


def _format_branches(n: int) -> str:
    n = int(round(n))
    if n >= 1_000_000:
        v = n / 1_000_000.0
        t = f"{v:.2f}".rstrip("0").rstrip(".")
        return t + "M"
    if n >= 1_000:
        v = n / 1_000.0
        t = f"{v:.2f}".rstrip("0").rstrip(".")
        return t + "K"
    return str(n)


def _gnn_csv_path(args) -> str | None:
    path = os.path.join(
        args.csv_dir, args.mode, f"{args.dataset}_GnnDecider_top{args.decision_topk_test}.csv"
    )
    return path if os.path.exists(path) else None


def _zeroemb_csv_path(args) -> str | None:
    heuristic = getattr(args, "decision_heuristic", "fsb")
    path = os.path.join(
        args.csv_dir,
        args.mode,
        f"{args.dataset}_SACDecider_{heuristic}_top{args.decision_topk_test}"
        f"_checkpoint_{args.checkpoint_postfix}_fallback_{args.n_test_fallback}_zeroemb.csv",
    )
    return path if os.path.exists(path) else None


def _oneshot_csv_path(args) -> str | None:
    heuristic = getattr(args, "decision_heuristic", "fsb")
    path = os.path.join(
        args.csv_dir,
        args.mode,
        f"{args.dataset}_SACDecider_{heuristic}_top{args.decision_topk_test}"
        f"_checkpoint_{args.checkpoint_postfix}_fallback_{args.n_test_fallback}_oneshot.csv",
    )
    return path if os.path.exists(path) else None


def _sac_upb_csv_path(args) -> str | None:
    path = os.path.join(
        args.csv_dir,
        args.mode,
        f"{args.dataset}_SACDecider_upb_top{args.decision_topk_test}"
        f"_checkpoint_{args.checkpoint_postfix}_fallback_{args.n_test_fallback}.csv",
    )
    return path if os.path.exists(path) else None


def _align_extra_episodes(extra: pd.DataFrame, suite_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Remap filtered-run episodes ``1..N`` to original sample episodes.

    - ``N == pool size`` and dense ``1..N`` → pool-filtered (e.g. FNN oneshot).
    - ``N == sample size`` and dense ``1..N`` → sample-filtered (e.g. CNN/sigmoid cactus).
    - otherwise → full run; keep episodes as-is.
    """
    extra = extra.copy()
    sample_sorted = suite_sample.sort_values("episode").reset_index(drop=True)
    pool_sorted = merge_df_at_least_one_solved(suite_sample).sort_values("episode").reset_index(drop=True)
    eps = extra["episode"].astype(int)
    n = len(extra)
    dense = set(eps.tolist()) == set(range(1, n + 1))
    if not dense:
        return extra

    if n == len(pool_sorted):
        mapping = {i + 1: int(pool_sorted.loc[i, "episode"]) for i in range(n)}
        extra["episode"] = eps.map(mapping)
        print(f"    [+] remapped {n} pool-filtered episodes")
    elif n == len(sample_sorted):
        mapping = {i + 1: int(sample_sorted.loc[i, "episode"]) for i in range(n)}
        extra["episode"] = eps.map(mapping)
        print(f"    [+] remapped {n} sample-filtered episodes")
    return extra


def attach_solver_by_episode(
    df: pd.DataFrame,
    args,
    column_suffix: str,
    path_fn: Callable,
) -> pd.DataFrame:
    """Join solver metrics onto sampled rows by original ``episode`` per suite."""
    parts = []
    for suite, config_file in BENCHMARK_TO_CONFIG.items():
        sub = df[df["suite"] == suite].copy()
        drop_cols = [c for c in sub.columns if c.endswith(f"_{column_suffix}")]
        if drop_cols:
            sub = sub.drop(columns=drop_cols)

        custom_args = update_args_from_yaml(copy.deepcopy(args), load_yaml_config(config_file))
        create_folders(custom_args)
        path = path_fn(custom_args)
        if path is None:
            print(f"    [!] missing {column_suffix} for {suite}")
            parts.append(sub)
            continue

        print(f"    [+] {suite} {column_suffix}: {path}")
        extra = pd.read_csv(path)
        if "active_neuron" in extra.columns:
            extra = extra.drop(columns=["active_neuron"])
        extra.loc[extra["value"] < TIMEOUT_THRESHOLD, "done"] = False
        extra = _align_extra_episodes(extra, sub)
        present = [c for c in METRIC_COLS if c in extra.columns]
        rename = {c: f"{c}_{column_suffix}" for c in present}
        extra = extra[["episode", *present]].rename(columns=rename)
        parts.append(sub.merge(extra, on="episode", how="left"))

    return pd.concat(parts, ignore_index=True)


def attach_optional_solvers(df: pd.DataFrame, args) -> pd.DataFrame:
    """Post-hoc joins that must not affect sampling / instance-pool selection."""
    print("[attach optional solvers]")
    df = attach_solver_by_episode(df, args, "GnnDecider", _gnn_csv_path)
    df = attach_solver_by_episode(df, args, "SACDeciderUpb", _sac_upb_csv_path)
    df = attach_solver_by_episode(df, args, "SACDeciderZeroEmb", _zeroemb_csv_path)
    df = attach_solver_by_episode(df, args, "SACDeciderOneshot", _oneshot_csv_path)
    return df


def _agg_solver(pool: pd.DataFrame, cls: str) -> tuple[float, int, int] | None:
    """Sum time/branches and count solved over rows where ``cls`` has non-null metrics."""
    dcol, tcol, vcol = f"done_{cls}", f"total_time_{cls}", f"value_{cls}"
    if dcol not in pool.columns or tcol not in pool.columns or vcol not in pool.columns:
        return None
    valid = pool[dcol].notna() & pool[tcol].notna() & pool[vcol].notna()
    if not valid.any():
        return None
    sub = pool.loc[valid]
    n = int(_done_mask(sub[dcol]).sum())
    tsum = float(pd.to_numeric(sub[tcol], errors="coerce").sum())
    bsum = float((-pd.to_numeric(sub[vcol], errors="coerce")).sum())
    return tsum, int(round(bsum)), n


def _md_cell(agg: tuple[float, int, int] | None, n_eval: int | None = None) -> list[str]:
    if agg is None:
        return ["--", "--", "--"]
    tt, br, n_sol = agg
    solved = str(int(n_sol)) if n_eval is None else f"{int(n_sol)}/{int(n_eval)}"
    return [f"{tt:.2f}", _format_branches(br), solved]


def _coverage_episodes(suite_df: pd.DataFrame, cls: str) -> set[int]:
    dcol = f"done_{cls}"
    if dcol not in suite_df.columns:
        return set()
    return set(suite_df.loc[suite_df[dcol].notna(), "episode"].astype(int))


def _is_sufficient_coverage(cov: set[int], n_sample: int) -> bool:
    """Drop stub/incomplete runs (e.g. 5-row GNN) but keep pool-filtered oneshots (~51)."""
    if not cov:
        return False
    if len(cov) >= n_sample:
        return True
    return len(cov) >= 50


def create_ablation_tables_markdown(df: pd.DataFrame) -> str:
    """
    Four ablation Markdown tables with FNN / CNN / Sigmoid columns.
    """
    blocks: list[str] = []
    for title, methods, use_solved_pool in ABLATION_TABLES:
        headers = ["Method"]
        for k in _BENCH_KEYS:
            t = PANEL_TITLE[k]
            headers.extend([f"{t} Time(s)", f"{t} Branches", f"{t} Solved"])
        sep = ["---"] + ["---:"] * (len(headers) - 1)
        lines = [
            f"### {title}",
            "",
            "|" + "|".join(headers) + "|",
            "|" + "|".join(sep) + "|",
        ]

        eval_frames: dict[str, pd.DataFrame | None] = {}
        available: dict[str, dict[str, bool]] = {}
        for suite in _BENCH_KEYS:
            suite_df = df[df["suite"] == suite]
            base = merge_df_at_least_one_solved(suite_df) if use_solved_pool else suite_df
            n_ref = len(suite_df)  # coverage threshold vs full sample size
            covs = {cls: _coverage_episodes(base, cls) for cls, _ in methods}
            # For pool tables, require coverage on the pool itself (not full 200).
            cov_ref = len(base) if use_solved_pool else n_ref
            ok = {cls: _is_sufficient_coverage(covs[cls], cov_ref) for cls, _ in methods}
            available[suite] = ok
            present = [covs[cls] for cls, _ in methods if ok[cls]]
            if not present:
                eval_frames[suite] = None
            else:
                eps = set.intersection(*present) if len(present) > 1 else present[0]
                eval_frames[suite] = base[base["episode"].astype(int).isin(eps)].reset_index(drop=True)

        for cls, name in methods:
            cells = [name]
            for suite in _BENCH_KEYS:
                frame = eval_frames[suite]
                if frame is None or not available[suite].get(cls, False):
                    cells.extend(["--", "--", "--"])
                else:
                    cells.extend(_md_cell(_agg_solver(frame, cls), n_eval=len(frame)))
            lines.append("|" + "|".join(cells) + "|")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


if __name__ == "__main__":
    from .extract_csv import extract_csv, sample_csv_v2
    from .plot_cactus import load_merged_sampled
    from arguments import get_args

    args = get_args()
    args.seed = 9750
    n_sample = 100
    extract_csv(args)
    sample_csv_v2(args, n_sample)
    # attach + aggregate on the cactus sample (200 / suite), not the solved-pool subset
    df = load_merged_sampled()
    df = attach_optional_solvers(df, args)
    print()
    print(create_ablation_tables_markdown(df))
