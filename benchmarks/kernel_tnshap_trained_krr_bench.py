#!/usr/bin/env python3
"""Paper-style trained-KRR benchmark for QuadraSHAP and upstream TN-SHAP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import kernel_tnshap_bench as common


OUTPUT_DIR = (
    ROOT / "benchmarks" / "results" / "kernel_tnshap_trained_krr_gamma_0p5"
)
N_TOTAL = 500
TEST_SIZE = 0.2
GAMMA = 0.5
SEED = 42


def make_trained_case(
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=N_TOTAL,
        n_features=n_features,
        n_informative=math.ceil(n_features / 4),
        noise=0.5,
        random_state=SEED + n_features,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train = x_scaler.transform(X_train).astype(np.float64)
    X_test = x_scaler.transform(X_test).astype(np.float64)
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

    model = KernelRidge(kernel="rbf", alpha=1.0, gamma=GAMMA)
    model.fit(X_train, y_train)
    alpha = np.asarray(model.dual_coef_, dtype=np.float64).ravel()
    return X_train, X_test, alpha


def make_product_game(
    X_train: np.ndarray, x: np.ndarray
) -> np.ndarray:
    return (
        np.exp(-GAMMA * (X_train - x[None, :]) ** 2) - 1.0
    ).astype(np.float64)


def _method_child(
    method: str,
    K: np.ndarray,
    alpha: np.ndarray,
    target: float,
    conn,
) -> None:
    try:
        torch.set_num_threads(1)
        started = time.perf_counter()
        if method == "quadrashap_exact":
            value = common.quadrashap(K, alpha, (K.shape[1] + 1) // 2)
        elif method == "tnshap_upstream_calculator":
            value = common.tnshap_upstream_calculator(K, alpha)
        else:
            raise ValueError(method)
        seconds = time.perf_counter() - started
        conn.send(
            {
                "status": "ok",
                "seconds": float(seconds),
                "efficiency_error": float(abs(np.sum(value) - target)),
                "finite": bool(np.isfinite(value).all()),
            }
        )
    except BaseException as exc:
        conn.send(
            {
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}"[:500],
            }
        )
    finally:
        conn.close()


def run_with_timeout(
    method: str,
    K: np.ndarray,
    alpha: np.ndarray,
    target: float,
    timeout: float,
) -> dict[str, object]:
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_method_child,
        args=(method, K, alpha, target, child),
    )
    process.start()
    child.close()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        return {"status": "timeout", "message": f"exceeded {timeout:g}s"}
    try:
        if parent.poll():
            return parent.recv()
    except EOFError:
        pass
    finally:
        parent.close()
    return {
        "status": "failed",
        "message": f"child exited {process.exitcode} without a result",
    }


def run(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows: list[dict[str, object]] = []

    with threadpool_limits(limits=1):
        for d in args.feature_counts:
            print(f"\ntraining KRR: d={d}", flush=True)
            X_train, X_test, alpha = make_trained_case(d)
            n_explain = min(args.explain_samples, len(X_test))
            methods = (
                "quadrashap_exact",
                "tnshap_upstream_calculator",
            )
            for method in methods:
                for sample_index in range(n_explain):
                    K = make_product_game(X_train, X_test[sample_index])
                    target = common.efficiency_target(K, alpha)
                    result = run_with_timeout(
                        method, K, alpha, target, args.timeout
                    )
                    row = {
                        "n_features": d,
                        "n_total_samples": N_TOTAL,
                        "n_train_samples": len(X_train),
                        "n_test_samples": len(X_test),
                        "requested_explanations": n_explain,
                        "sample_index": sample_index,
                        "n_terms": len(alpha),
                        "gamma": GAMMA,
                        "method": method,
                        **result,
                    }
                    rows.append(row)
                    if result["status"] == "ok":
                        print(
                            f"  {method:28s} sample={sample_index:2d} "
                            f"time={float(result['seconds']):8.3f}s "
                            f"eff={float(result['efficiency_error']):.3e}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  {method:28s} sample={sample_index:2d} "
                            f"{result['status']}: {result.get('message', '')}",
                            flush=True,
                        )
                        # A paper-style per-instance timeout means the method
                        # is unavailable at this dimension; do not spend the
                        # same timeout on the remaining 49 samples.
                        break

            with (OUTPUT_DIR / "raw_results.csv").open(
                "w", newline=""
            ) as handle:
                fields = sorted({key for row in rows for key in row})
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    summary: list[dict[str, object]] = []
    for d in args.feature_counts:
        for method in ("quadrashap_exact", "tnshap_upstream_calculator"):
            selected = [
                row
                for row in rows
                if row["n_features"] == d and row["method"] == method
            ]
            ok = [row for row in selected if row["status"] == "ok"]
            status = "ok" if len(ok) == args.explain_samples else (
                selected[-1]["status"] if selected else "missing"
            )
            summary.append(
                {
                    "n_features": d,
                    "method": method,
                    "status": status,
                    "completed_explanations": len(ok),
                    "requested_explanations": args.explain_samples,
                    "mean_seconds": (
                        statistics.mean(float(row["seconds"]) for row in ok)
                        if ok else math.nan
                    ),
                    "std_seconds": (
                        statistics.stdev(float(row["seconds"]) for row in ok)
                        if len(ok) > 1 else 0.0 if ok else math.nan
                    ),
                    "max_efficiency_error": (
                        max(float(row["efficiency_error"]) for row in ok)
                        if ok else math.nan
                    ),
                    "mean_efficiency_error": (
                        statistics.mean(
                            float(row["efficiency_error"]) for row in ok
                        )
                        if ok else math.nan
                    ),
                }
            )

    with (OUTPUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "threads": 1,
                "n_total_samples": N_TOTAL,
                "n_train_samples": int(N_TOTAL * (1 - TEST_SIZE)),
                "n_test_samples": int(N_TOTAL * TEST_SIZE),
                "explain_samples": args.explain_samples,
                "n_informative": "ceil(n_features / 4)",
                "regression_noise": 0.5,
                "data_seed": "42 + n_features",
                "split_seed": SEED,
                "standardization": (
                    "StandardScaler fit on training X and training y"
                ),
                "gamma": GAMMA,
                "krr_alpha": 1.0,
                "seed": SEED,
                "timeout_seconds_per_instance": args.timeout,
                "timed_region": (
                    "Shapley method body only; data generation, scaling, "
                    "KRR fitting, product-game construction, and child-process "
                    "startup excluded"
                ),
                "quadrashap_revision": common.git_revision(ROOT),
                "tnshap_revision": common.git_revision(common.TN_SHAP_DIR),
                "tnshap_entrypoint": (
                    "experiments/03_synthetic_experiments/scripts/"
                    "synthetic_rank_sweep_basic.py::"
                    "TNShapCalculator.shapley_values_tnshap"
                ),
                "tnshap_local_component": "model-interface adapter only",
            },
            indent=2,
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-counts",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100, 200, 500],
    )
    parser.add_argument("--explain-samples", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
