"""Benchmark XGBoost's integrated GPU TreeSHAP on the paper forests.

The cached sklearn forests are converted to XGBoost's model format without
changing topology or path covers. XGBoost stores thresholds and leaf values
as float32, so the bridge records the resulting prediction delta.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gpu_treeshap_bench import (
    FEATURES,
    LEAVES,
    MODEL_CACHE,
    N_REPEATS,
    N_SAMPLES,
    RESULTS,
    SEEDS,
    TEXT_DATASETS,
    TEXT_INPUTS,
    TEXT_MODELS,
    TEXT_TARGETS,
    _max_depth,
    _normalise_values,
)
from xgboost_sklearn_bridge import sklearn_forest_to_xgboost


MODES = ("cached_dmatrix", "full_call")
DEFAULT_BATCHES = (
    1,
    10,
    100,
    500,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    128_000,
)
OUTPUT = RESULTS / "gpu_xgboost_treeshap_results.json"


def _json_line(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def worker(args: argparse.Namespace) -> int:
    import cupy as cp
    import joblib
    import xgboost as xgb

    loaded = joblib.load(args.model)
    if isinstance(loaded, tuple):
        model, cached_X = loaded
    elif isinstance(loaded, dict):
        model = loaded["estimator"]
        cached_X = None
    else:
        model, cached_X = loaded, None

    X = cached_X if args.inputs is None else np.load(args.inputs)["X"]
    X = np.asarray(X)
    if len(X) < args.n_samples:
        X = X[np.arange(args.n_samples) % len(X)]
    else:
        X = X[: args.n_samples]
    X = np.ascontiguousarray(X, dtype=np.float64)

    target = int(args.target)
    if hasattr(model, "predict_proba"):
        reference_prediction = np.asarray(
            model.predict_proba(X),
            dtype=np.float64,
        )[:, target]
        bridge_target = target
    else:
        reference_prediction = np.asarray(
            model.predict(X),
            dtype=np.float64,
        )
        bridge_target = None

    booster = sklearn_forest_to_xgboost(
        model,
        target=bridge_target,
        device="cuda:0",
    )
    cached_dmatrix = xgb.DMatrix(X)

    if args.mode == "cached_dmatrix":

        def explain():
            return booster.predict(cached_dmatrix, pred_contribs=True)

    elif args.mode == "full_call":

        def explain():
            return booster.predict(
                xgb.DMatrix(X),
                pred_contribs=True,
            )

    else:
        _json_line({"status": "error", "message": f"unknown mode {args.mode}"})
        return 2

    try:
        values = explain()
        cp.cuda.runtime.deviceSynchronize()
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            values = explain()
            cp.cuda.runtime.deviceSynchronize()
            times.append(time.perf_counter() - t0)
    except Exception as exc:
        _json_line(
            {
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return 0

    xgb_prediction = np.asarray(
        booster.predict(cached_dmatrix, output_margin=True),
        dtype=np.float64,
    )
    values = np.asarray(values, dtype=np.float64)
    phi = values[:, :-1]
    bias = values[:, -1]
    residual = np.abs(values.sum(axis=1) - xgb_prediction)
    result = {
        "status": "ok",
        "mode": args.mode,
        "elapsed_s": float(np.median(times)),
        "times_s": [float(value) for value in times],
        "ms_per_instance": float(1e3 * np.median(times) / len(X)),
        "max_additivity_error": float(np.max(residual)),
        "mean_additivity_error": float(np.mean(residual)),
        "bridge_prediction_max_abs_error": float(
            np.max(np.abs(xgb_prediction - reference_prediction))
        ),
        "n_samples": int(len(X)),
        "n_trees": int(len(np.ravel(model.estimators_))),
        "n_leaves": int(
            sum(est.tree_.n_leaves for est in np.ravel(model.estimators_))
        ),
        "max_depth": _max_depth(model),
    }

    if args.validate_reference:
        from quadrashap import TreeExplainer

        explainer = TreeExplainer(
            model,
            tree_solver="quadrature_tree",
            device="cuda",
        )
        reference_phi = _normalise_values(
            explainer.shap_values(X, check_additivity=False)
        )
        if reference_phi.ndim == 3:
            reference_phi = reference_phi[:, :, target]
            reference_bias = float(
                np.asarray(explainer.expected_value).ravel()[target]
            )
        else:
            reference_bias = float(
                np.asarray(explainer.expected_value).ravel()[0]
            )
        delta = np.abs(reference_phi - phi)
        result.update(
            {
                "quadrashap_phi_max_abs_difference": float(np.max(delta)),
                "quadrashap_phi_mean_abs_difference": float(np.mean(delta)),
                "quadrashap_bias_abs_difference": float(
                    abs(reference_bias - bias[0])
                ),
            }
        )

    _json_line(result)
    return 0


def _last_payload(proc: subprocess.CompletedProcess) -> dict:
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {
        "status": "error",
        "message": (proc.stderr or proc.stdout)[-2000:],
        "returncode": proc.returncode,
    }


def run_worker(
    *,
    model: Path,
    mode: str,
    target: int = 0,
    inputs: Path | None = None,
    n_samples: int = N_SAMPLES,
    repeats: int = N_REPEATS,
    validate_reference: bool = False,
) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        str(model),
        "--mode",
        mode,
        "--target",
        str(target),
        "--repeats",
        str(repeats),
        "--n-samples",
        str(n_samples),
    ]
    if inputs is not None:
        cmd += ["--inputs", str(inputs)]
    if validate_reference:
        cmd.append("--validate-reference")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    result = _last_payload(proc)
    if proc.returncode and result.get("status") == "ok":
        return {
            "status": "error",
            "message": proc.stderr[-2000:],
            "returncode": proc.returncode,
        }
    return result


def metadata() -> dict:
    import cupy
    import sklearn
    import xgboost

    props = cupy.cuda.runtime.getDeviceProperties(0)
    return {
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": props["name"].decode(),
        "gpu_compute_capability": f"{props['major']}.{props['minor']}",
        "gpu_memory_bytes": int(props["totalGlobalMem"]),
        "versions": {
            "numpy": np.__version__,
            "cupy": cupy.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "implementation": {
            "api": "xgboost.Booster.predict(pred_contribs=True)",
            "device": "cuda:0",
            "backend_verification": (
                "XGBoost 3.1.3 libxgboost contains gpu_treeshap::"
                "GPUTreeShap kernels"
            ),
        },
    }


def aggregate(per_seed: list[dict]) -> dict:
    ok = [result for result in per_seed if result.get("status") == "ok"]
    if not ok:
        return {**per_seed[0], "per_seed": per_seed}
    keys_max = (
        "max_additivity_error",
        "bridge_prediction_max_abs_error",
        "quadrashap_phi_max_abs_difference",
        "quadrashap_phi_mean_abs_difference",
        "quadrashap_bias_abs_difference",
    )
    result = {
        "status": "ok",
        "elapsed_s": float(np.median([item["elapsed_s"] for item in ok])),
        "ms_per_instance": float(
            np.median([item["ms_per_instance"] for item in ok])
        ),
        "per_seed": per_seed,
    }
    for key in keys_max:
        present = [item[key] for item in ok if key in item]
        if present:
            result[key] = float(np.max(present))
    return result


def benchmark(args: argparse.Namespace) -> int:
    output = {
        "metadata": metadata(),
        "protocol": {
            "modes": args.modes,
            "repeats": args.repeats,
            "paper_samples": N_SAMPLES,
            "paper_seeds": list(SEEDS),
            "batch_sizes": args.batches,
            "timed_region": {
                "cached_dmatrix": (
                    "warmed Booster.predict call with a reused DMatrix"
                ),
                "full_call": (
                    "warmed DMatrix construction plus Booster.predict call"
                ),
            },
            "construction_excluded": (
                "sklearn-to-XGBoost model conversion and booster construction"
            ),
        },
        "synthetic": {},
        "text": {},
        "batch_scaling": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(output, indent=2) + "\n")

    if args.suite in ("paper", "synthetic", "all"):
        for n_features in FEATURES:
            by_feature = output["synthetic"].setdefault(
                str(n_features),
                {},
            )
            for leaves in LEAVES:
                print(
                    f"[synthetic] d={n_features} leaves={leaves}",
                    flush=True,
                )
                by_leaves = by_feature.setdefault(str(leaves), {})
                for mode in args.modes:
                    per_seed = []
                    for seed in SEEDS:
                        model = MODEL_CACHE / (
                            f"sklearn_rf_f{n_features}_l{leaves}_s{seed}.pkl"
                        )
                        per_seed.append(
                            run_worker(
                                model=model,
                                mode=mode,
                                repeats=args.repeats,
                                validate_reference=(
                                    mode == "cached_dmatrix"
                                ),
                            )
                        )
                    result = aggregate(per_seed)
                    by_leaves[mode] = result
                    print(
                        f"  {mode:16s} "
                        f"{result.get('ms_per_instance', float('nan')):.3f} "
                        "ms/instance "
                        f"status={result.get('status')}",
                        flush=True,
                    )
                    save()

    if args.suite in ("paper", "text", "all"):
        for dataset in TEXT_DATASETS:
            print(f"[text] {dataset}", flush=True)
            by_dataset = output["text"].setdefault(dataset, {})
            model = TEXT_MODELS / dataset / "random_forest.joblib"
            inputs = TEXT_INPUTS / f"{dataset}.npz"
            for mode in args.modes:
                result = run_worker(
                    model=model,
                    mode=mode,
                    target=TEXT_TARGETS[dataset],
                    inputs=inputs,
                    repeats=args.repeats,
                    validate_reference=(mode == "cached_dmatrix"),
                )
                by_dataset[mode] = result
                print(
                    f"  {mode:16s} "
                    f"{result.get('ms_per_instance', float('nan')):.3f} "
                    f"ms/instance status={result.get('status')}",
                    flush=True,
                )
                save()

    if args.suite in ("batch", "all"):
        for n_features in FEATURES:
            by_feature = output["batch_scaling"].setdefault(
                str(n_features),
                {},
            )
            model = MODEL_CACHE / (
                f"sklearn_rf_f{n_features}_l100000_s42.pkl"
            )
            for batch in args.batches:
                print(
                    f"[batch] d={n_features} rows={batch}",
                    flush=True,
                )
                by_batch = by_feature.setdefault(str(batch), {})
                for mode in args.modes:
                    result = run_worker(
                        model=model,
                        mode=mode,
                        n_samples=batch,
                        repeats=args.repeats,
                    )
                    by_batch[mode] = result
                    print(
                        f"  {mode:16s} "
                        f"{1e3 * result.get('elapsed_s', float('nan')):.3f} "
                        f"ms status={result.get('status')}",
                        flush=True,
                    )
                    save()

    save()
    print(f"Wrote {args.output}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--validate-reference", action="store_true")
    parser.add_argument(
        "--suite",
        choices=("synthetic", "text", "paper", "batch", "all"),
        default="all",
    )
    parser.add_argument(
        "--modes",
        choices=MODES,
        nargs="+",
        default=MODES,
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=DEFAULT_BATCHES,
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker(parsed) if parsed.worker else benchmark(parsed))
