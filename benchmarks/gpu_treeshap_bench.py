"""Reproduce Tables 1 and 2 with CUDA tree explainers.

The benchmark uses the exact cached scikit-learn forests and ten-instance
batches from the CPU experiments.  QuadraSHAP and SHAP's GPUTreeExplainer
therefore see identical trees and inputs.

Requirements
------------
* ``cupy-cuda12x`` (or the CuPy build matching the host CUDA runtime)
* SHAP built from source with its ``_cext_gpu`` extension

Run from the repository root:

    .venv/bin/python benchmarks/gpu_treeshap_bench.py

Results are written under ``benchmarks/results/gpu``.  Explainer construction
and JIT compilation are warmed up and excluded.  The full explanation call is
timed, including input/output transfers and any other transfers a backend
performs internally.
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


REPO = Path(__file__).resolve().parent.parent
MODEL_CACHE = REPO / "benchmarks" / "_treeshap_bench_models"
TEXT_MODELS = REPO / "model"
RESULTS = REPO / "benchmarks" / "results" / "gpu"
TEXT_INPUTS = RESULTS / "text_inputs"
RAW_JSON = RESULTS / "gpu_treeshap_results.json"
CSV_PATH = RESULTS / "gpu_treeshap_results.csv"
TEX_PATH = RESULTS / "gpu_treeshap_tables.tex"
N_SAMPLES = 10
N_REPEATS = 3
SEEDS = (42, 43, 44)
FEATURES = (10, 100)
LEAVES = (10, 100, 1_000, 10_000, 100_000)
TEXT_DATASETS = ("emotion", "imdb", "sms_spam", "sst2", "rotten_tomatoes")
TEXT_TARGETS = {
    "emotion": 0,
    "imdb": 1,
    "sms_spam": 0,
    "sst2": 1,
    "rotten_tomatoes": 1,
}
METHODS = ("quadrashap_gpu", "gputreeshap")


def _json_line(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _normalise_values(values) -> np.ndarray:
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=-1)
    return np.asarray(values)


def _max_depth(model) -> int:
    trees = getattr(model, "estimators_", [model])
    return max(int(est.tree_.max_depth) for est in np.ravel(trees))


def worker(args: argparse.Namespace) -> int:
    import cupy as cp
    import joblib

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
        prediction = np.asarray(model.predict_proba(X), dtype=np.float64)[:, target]
    else:
        prediction = np.asarray(model.predict(X), dtype=np.float64)

    if args.method == "gputreeshap":
        # The upstream CUDA implementation terminates the process from C++ for
        # longer paths instead of returning a Python exception.
        depth = _max_depth(model)
        if depth >= 32:
            _json_line({
                "status": "unsupported",
                "message": "GPUTreeSHAP requires every tree path to have length <= 32",
                "max_depth": depth,
            })
            return 0
        import shap
        try:
            import shap._cext_gpu  # noqa: F401
        except ImportError as exc:
            _json_line({"status": "unavailable", "message": str(exc)})
            return 0
        explainer = shap.GPUTreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        )
    elif args.method == "quadrashap_gpu":
        from quadrashap import TreeExplainer

        explainer = TreeExplainer(
            model, tree_solver="quadrature_tree", device="cuda"
        )
    else:
        _json_line({"status": "error", "message": f"unknown method {args.method}"})
        return 2

    def explain():
        return explainer.shap_values(X, check_additivity=False)

    # Warm-up includes CuPy NVRTC compilation or GPUTreeSHAP's first-use setup.
    values = explain()
    cp.cuda.runtime.deviceSynchronize()

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        values = explain()
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)

    values = _normalise_values(values)
    if values.ndim == 3:
        values = values[:, :, target]
        expected = float(np.asarray(explainer.expected_value).ravel()[target])
    else:
        expected = float(np.asarray(explainer.expected_value).ravel()[0])
    residual = np.abs(expected + values.sum(axis=1) - prediction)
    _json_line({
        "status": "ok",
        "elapsed_s": float(np.median(times)),
        "times_s": [float(x) for x in times],
        "ms_per_instance": float(1e3 * np.median(times) / len(X)),
        "max_additivity_error": float(np.max(residual)),
        "mean_additivity_error": float(np.mean(residual)),
        "n_samples": int(len(X)),
        "max_depth": _max_depth(model),
    })
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
        "message": (proc.stderr or proc.stdout)[-1000:],
        "returncode": proc.returncode,
    }


def run_worker(
    model: Path,
    method: str,
    target: int = 0,
    inputs: Path | None = None,
    n_samples: int = N_SAMPLES,
) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        str(model),
        "--method",
        method,
        "--target",
        str(target),
        "--repeats",
        str(N_REPEATS),
        "--n-samples",
        str(n_samples),
    ]
    if inputs is not None:
        cmd += ["--inputs", str(inputs)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    result = _last_payload(proc)
    if proc.returncode and result.get("status") == "ok":
        result = {
            "status": "error",
            "message": proc.stderr[-1000:],
            "returncode": proc.returncode,
        }
    return result


def prepare_text_inputs() -> dict[str, tuple[Path, Path, int]]:
    import joblib

    TEXT_INPUTS.mkdir(parents=True, exist_ok=True)
    out = {}
    dataset_helpers = None
    for key in TEXT_DATASETS:
        model_path = TEXT_MODELS / key / "random_forest.joblib"
        bundle = joblib.load(model_path)
        input_path = TEXT_INPUTS / f"{key}.npz"
        if not input_path.exists():
            if dataset_helpers is None:
                sys.path.insert(0, str(REPO))
                from benchmarks.text_classification_experiment import (
                    DATASET_SPECS,
                    dataset_to_xy,
                    dense_rows,
                    load_dataset_splits,
                )

                dataset_helpers = (
                    DATASET_SPECS,
                    dataset_to_xy,
                    dense_rows,
                    load_dataset_splits,
                )
            DATASET_SPECS, dataset_to_xy, dense_rows, load_dataset_splits = (
                dataset_helpers
            )
            spec = DATASET_SPECS[key]
            splits = load_dataset_splits(spec, seed=42)
            texts, _ = dataset_to_xy(splits["test"], spec)
            X = dense_rows(bundle["vectorizer"].transform(texts[:N_SAMPLES]))
            np.savez_compressed(input_path, X=np.asarray(X, dtype=np.float64))
        out[key] = (model_path, input_path, TEXT_TARGETS[key])
    return out


def system_metadata() -> dict:
    gpu = "unknown"
    driver = "unknown"
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        gpu, driver, memory = [x.strip() for x in text.split(",")[:3]]
    except Exception:
        memory = "unknown"
    versions = {}
    for name in ("numpy", "cupy", "shap", "sklearn"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu,
        "driver": driver,
        "gpu_memory": memory,
        "versions": versions,
        "gputreeshap": {
            "integration": "shap.GPUTreeExplainer",
            "source": "https://github.com/shap/shap",
            "source_revision": os.environ.get("GPUTREESHAP_REVISION", "unknown"),
        },
        "protocol": {
            "n_samples": N_SAMPLES,
            "n_repeats": N_REPEATS,
            "synthetic_seeds": list(SEEDS),
            "timed_region": (
                "full explanation call including backend transfers; "
                "construction and warm-up excluded"
            ),
        },
    }


def aggregate_synthetic(per_seed: list[dict]) -> dict:
    ok = [x for x in per_seed if x.get("status") == "ok"]
    if not ok:
        return {**per_seed[0], "per_seed": per_seed}
    return {
        "status": "ok",
        "ms_per_instance": float(np.median([x["ms_per_instance"] for x in ok])),
        "ms_per_instance_min": float(np.min([x["ms_per_instance"] for x in ok])),
        "ms_per_instance_max": float(np.max([x["ms_per_instance"] for x in ok])),
        "max_additivity_error": float(
            np.max([x["max_additivity_error"] for x in ok])
        ),
        "per_seed": per_seed,
    }


def _fmt_time(result: dict) -> str:
    if result.get("status") == "ok":
        return f"{result['ms_per_instance']:.3g}"
    if result.get("status") == "unsupported":
        return r"\textsc{n/s}"
    return r"\textsc{err}"


def write_outputs(results: dict) -> None:
    import pandas as pd

    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW_JSON.write_text(json.dumps(results, indent=2) + "\n")
    rows = []
    for d, by_leaves in results["synthetic"].items():
        for leaves, by_method in by_leaves.items():
            for method, result in by_method.items():
                rows.append({
                    "suite": "synthetic",
                    "dataset": None,
                    "n_features": int(d),
                    "leaves": int(leaves),
                    "method": method,
                    **{k: v for k, v in result.items() if k != "per_seed"},
                })
    for dataset, by_method in results["text"].items():
        for method, result in by_method.items():
            rows.append({
                "suite": "text",
                "dataset": dataset,
                "n_features": 300,
                "leaves": None,
                "method": method,
                **result,
            })
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    lines = [
        "% Generated by benchmarks/gpu_treeshap_bench.py",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{5}{c}{$d=10$} & \multicolumn{5}{c}{$d=100$}\\",
        r"\# leaves & 10 & 100 & 1k & 10k & 100k & 10 & 100 & 1k & 10k & 100k\\",
        r"\midrule",
    ]
    for method, label in (
        ("gputreeshap", r"\textsc{GPUTreeSHAP}"),
        ("quadrashap_gpu", r"\textsc{QuadraSHAP-GPU}"),
    ):
        vals = []
        for d in FEATURES:
            for leaves in LEAVES:
                result = (
                    results["synthetic"]
                    .get(str(d), {})
                    .get(str(leaves), {})
                    .get(method, {})
                )
                vals.append(_fmt_time(result))
        lines.append(label + " & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\begin{tabular}{lrrrrr}", r"\toprule"]
    lines += [r"& emotion & imdb & sms spam & sst2 & RT\\", r"\midrule"]
    for method, label in (
        ("gputreeshap", r"\textsc{GPUTreeSHAP}"),
        ("quadrashap_gpu", r"\textsc{QuadraSHAP-GPU}"),
    ):
        vals = [
            _fmt_time(results["text"].get(key, {}).get(method, {}))
            for key in TEXT_DATASETS
        ]
        lines.append(label + " & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    TEX_PATH.write_text("\n".join(lines))


def benchmark_all() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = {"metadata": system_metadata(), "synthetic": {}, "text": {}}
    print(f"GPU: {results['metadata']['gpu']}", flush=True)

    for d in FEATURES:
        results["synthetic"][str(d)] = {}
        for leaves in LEAVES:
            print(f"[synthetic] d={d} leaves={leaves}", flush=True)
            results["synthetic"][str(d)][str(leaves)] = {}
            for method in METHODS:
                per_seed = []
                for seed in SEEDS:
                    path = MODEL_CACHE / f"sklearn_rf_f{d}_l{leaves}_s{seed}.pkl"
                    result = run_worker(path, method)
                    per_seed.append(result)
                aggregate = aggregate_synthetic(per_seed)
                results["synthetic"][str(d)][str(leaves)][method] = aggregate
                print(
                    f"  {method:18s} {_fmt_time(aggregate):>9s} ms/instance "
                    f"err={aggregate.get('max_additivity_error', 'n/a')}",
                    flush=True,
                )
                write_outputs(results)

    text_inputs = prepare_text_inputs()
    for key in TEXT_DATASETS:
        print(f"[text] {key}", flush=True)
        results["text"][key] = {}
        model, inputs, target = text_inputs[key]
        for method in METHODS:
            result = run_worker(model, method, target=target, inputs=inputs)
            results["text"][key][method] = result
            print(
                f"  {method:18s} {_fmt_time(result):>9s} ms/instance "
                f"err={result.get('max_additivity_error', 'n/a')} "
                f"status={result.get('status')}",
                flush=True,
            )
            write_outputs(results)
    write_outputs(results)
    print(f"Wrote {RAW_JSON}, {CSV_PATH}, and {TEX_PATH}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker(parsed) if parsed.worker else benchmark_all())
