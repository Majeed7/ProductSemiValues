#!/usr/bin/env python3
"""Benchmark QuadraSHAP and TN-SHAP defaults on TF-IDF RBF product games.

The inputs and fitted SVC bundles come from the repository's text benchmark.
For each dataset, the densest of the ten cached held-out TF-IDF rows is used
as the explained point.  The first R fitted support vectors and their actual
dual coefficients define a truncated RBF decision-function expansion.  The
RBF gamma is fixed to 0.5 for every case.
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import kernel_tnshap_bench as common


OUTPUT_DIR = (
    ROOT / "benchmarks" / "results" / "kernel_tnshap_tfidf_gamma_0p5"
)
DATASETS = ("imdb", "rotten_tomatoes", "sms_spam", "sst2")
GAMMA = 0.5
REPEATS = 3


def load_case(dataset: str, n_terms: int) -> tuple[np.ndarray, np.ndarray, dict]:
    bundle_path = ROOT / "model" / dataset / "svc_rbf.joblib"
    input_path = (
        ROOT / "benchmarks" / "results" / "gpu" / "text_inputs"
        / f"{dataset}.npz"
    )
    bundle = joblib.load(bundle_path)
    estimator = bundle["estimator"]
    inputs = np.load(input_path)["X"].astype(np.float64, copy=False)
    sample_index = int(np.argmax(np.count_nonzero(inputs, axis=1)))
    x = inputs[sample_index]

    support_vectors = np.asarray(
        estimator.support_vectors_[:n_terms], dtype=np.float64
    )
    alpha = np.asarray(
        estimator.dual_coef_[0, :n_terms], dtype=np.float64
    )
    if len(support_vectors) != n_terms:
        raise ValueError(
            f"{dataset} has only {len(support_vectors)} support vectors"
        )

    factors = np.exp(-GAMMA * (support_vectors - x[None, :]) ** 2)
    K = factors - 1.0
    active = np.count_nonzero(K, axis=1)
    details = {
        "sample_index": sample_index,
        "query_nnz": int(np.count_nonzero(x)),
        "active_min": int(active.min()),
        "active_median": float(np.median(active)),
        "active_max": int(active.max()),
        "model_support_vectors": int(estimator.support_vectors_.shape[0]),
        "model_fitted_gamma": float(estimator._gamma),
    }
    return K, alpha, details


def run_case(
    *, regime: str, dataset: str, n_terms: int
) -> list[dict[str, object]]:
    K, alpha, details = load_case(dataset, n_terms)
    d = K.shape[1]
    target = common.efficiency_target(K, alpha)
    exact_nodes = (d + 1) // 2
    methods = [
        (
            "quadrashap_exact",
            exact_nodes,
            lambda: common.quadrashap(K, alpha, exact_nodes),
        ),
        (
            "tnshap_upstream_calculator",
            d,
            lambda: common.tnshap_upstream_calculator(K, alpha),
        ),
    ]

    rows: list[dict[str, object]] = []
    for method, nodes, fn in methods:
        elapsed: list[float] = []
        efficiency_errors: list[float] = []
        finite = True
        for _ in range(REPEATS):
            started = time.perf_counter()
            value = np.asarray(fn(), dtype=np.float64)
            elapsed.append(time.perf_counter() - started)
            finite = finite and bool(np.isfinite(value).all())
            efficiency_errors.append(float(abs(np.sum(value) - target)))

        row = {
            "regime": regime,
            "dataset": dataset,
            "sample_index": details["sample_index"],
            "n_features": d,
            "query_nnz": details["query_nnz"],
            "n_terms": n_terms,
            "active_min": details["active_min"],
            "active_median": details["active_median"],
            "active_max": details["active_max"],
            "gamma": GAMMA,
            "model_fitted_gamma": details["model_fitted_gamma"],
            "model_support_vectors": details["model_support_vectors"],
            "method": method,
            "nodes": nodes,
            "median_seconds": statistics.median(elapsed),
            "min_seconds": min(elapsed),
            "max_seconds": max(elapsed),
            "efficiency_error": statistics.median(efficiency_errors),
            "efficiency_error_min": min(efficiency_errors),
            "efficiency_error_max": max(efficiency_errors),
            "finite": finite,
        }
        rows.append(row)
        print(
            f"{dataset:16s} R={n_terms:2d} {method:25s} "
            f"time={1000 * float(row['median_seconds']):9.3f} ms "
            f"eff={float(row['efficiency_error']):.3e}",
            flush=True,
        )
    return rows


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# TF-IDF + RBF: exact QuadraSHAP vs TN-SHAP defaults",
        "",
        "Gamma is fixed to 0.5. Times and efficiency errors are medians over "
        "three complete, single-threaded CPU calls.",
        "",
        "| Regime | Dataset | Terms | Method | Nodes | Time (ms) | "
        "Efficiency error |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime']} | {row['dataset']} | {row['n_terms']} | "
            f"{row['method']} | {row['nodes']} | "
            f"{1000 * float(row['median_seconds']):.3f} | "
            f"{float(row['efficiency_error']):.3e} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows: list[dict[str, object]] = []

    with threadpool_limits(limits=1):
        for dataset in DATASETS:
            rows.extend(
                run_case(
                    regime="dataset_sweep",
                    dataset=dataset,
                    n_terms=16,
                )
            )
        for n_terms in (1, 64):
            rows.extend(
                run_case(
                    regime="imdb_term_sweep",
                    dataset="imdb",
                    n_terms=n_terms,
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
                "repeats": REPEATS,
                "gamma": GAMMA,
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
