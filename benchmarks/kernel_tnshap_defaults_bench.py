#!/usr/bin/env python3
"""Focused comparison of exact QuadraSHAP with TN-SHAP repository defaults."""

from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import kernel_tnshap_bench as common


OUTPUT_DIR = ROOT / "benchmarks" / "results" / "kernel_tnshap_defaults"


def measure(fn, repeats: int) -> tuple[list[np.ndarray], float]:
    elapsed: list[float] = []
    values: list[np.ndarray] = []
    for _ in range(repeats):
        started = time.perf_counter()
        values.append(np.asarray(fn(), dtype=np.float64))
        elapsed.append(time.perf_counter() - started)
    return values, statistics.median(elapsed)


def run_case(
    *,
    regime: str,
    n_features: int,
    n_terms: int,
    gamma: float,
    seed: int,
    repeats: int,
) -> list[dict[str, object]]:
    K, alpha = common.make_kernel_game(
        n_features=n_features,
        n_terms=n_terms,
        gamma=gamma,
        seed=seed,
    )
    reference = common.high_precision_reference(K, alpha, decimal_digits=100)
    reference_norm = float(np.linalg.norm(reference))
    target = common.efficiency_target(K, alpha)
    exact_nodes = (n_features + 1) // 2

    methods = [
        (
            "quadrashap_exact",
            exact_nodes,
            lambda: common.quadrashap(K, alpha, exact_nodes),
        ),
        (
            "tnshap_upstream_calculator",
            n_features,
            lambda: common.tnshap_upstream_calculator(K, alpha),
        ),
    ]

    rows: list[dict[str, object]] = []
    for method, nodes, fn in methods:
        values, seconds = measure(fn, repeats)
        deltas = [value - reference for value in values]
        relative_errors = [
            float(np.linalg.norm(delta) / max(reference_norm, 1e-300))
            for delta in deltas
        ]
        max_abs_errors = [
            float(np.max(np.abs(delta))) for delta in deltas
        ]
        efficiency_errors = [
            float(abs(np.sum(value) - target)) for value in values
        ]
        row = {
            "regime": regime,
            "n_terms": n_terms,
            "n_features": n_features,
            "gamma": gamma,
            "method": method,
            "nodes": nodes,
            "median_seconds": seconds,
            "rel_l2_error": statistics.median(relative_errors),
            "rel_l2_error_min": min(relative_errors),
            "rel_l2_error_max": max(relative_errors),
            "max_abs_error": statistics.median(max_abs_errors),
            "efficiency_error": statistics.median(efficiency_errors),
        }
        rows.append(row)
        print(
            f"{regime:12s} terms={n_terms:2d} d={n_features:3d} "
            f"{method:25s} m={nodes:3d} time={1000*seconds:9.3f} ms "
            f"efficiency={row['efficiency_error']:.3e}",
            flush=True,
        )
    return rows


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Exact QuadraSHAP vs TN-SHAP defaults",
        "",
        "Single-threaded CPU medians over three end-to-end runs. "
        "TN-SHAP is the upstream `TNShapCalculator` from the synthetic "
        "experiments, using `d` Chebyshev nodes and a square Vandermonde "
        "solve per feature.",
        "",
        "| Regime | Terms | d | Method | Nodes | Time (ms) | Efficiency error |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime']} | {row['n_terms']} | {row['n_features']} | "
            f"{row['method']} | {row['nodes']} | "
            f"{1000*float(row['median_seconds']):.3f} | "
            f"{float(row['efficiency_error']):.3e} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows: list[dict[str, object]] = []

    with threadpool_limits(limits=1):
        for d in [10, 20, 50, 100, 200, 500]:
            rows.extend(
                run_case(
                    regime="gamma_0.5",
                    n_features=d,
                    n_terms=1,
                    gamma=0.5,
                    seed=42 + d,
                    repeats=3,
                )
            )

        for n_terms in [16, 64]:
            for d in [50, 100]:
                rows.extend(
                    run_case(
                        regime="krr_sum",
                        n_features=d,
                        n_terms=n_terms,
                        gamma=0.5,
                        seed=42 + 10_000 + 100 * n_terms + d,
                        repeats=3,
                    )
                )

    with (OUTPUT_DIR / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "README.md").write_text(render_markdown(rows))
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "threads": 1,
                "repeats": 3,
                "quadrashap_revision": common.git_revision(ROOT),
                "tnshap_revision": common.git_revision(common.TN_SHAP_DIR),
                "tnshap_entrypoint": (
                    "experiments/03_synthetic_experiments/scripts/"
                    "synthetic_rank_sweep_basic.py::"
                    "TNShapCalculator.shapley_values_tnshap"
                ),
                "tnshap_local_component": (
                    "model-interface adapter only"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
