"""Benchmark CUDA TreeSHAP implementations across increasing batch sizes.

This reuses the worker and timing protocol from ``gpu_treeshap_bench.py``:
explainer construction and one warm-up call are excluded, while each timed
call includes input/output transfers and ends with a device synchronization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gpu_treeshap_bench import (
    METHODS,
    MODEL_CACHE,
    RESULTS,
    run_worker,
    system_metadata,
)


DEFAULT_BATCHES = (1, 10, 100, 500, 2_000, 4_000, 8_000, 16_000, 32_000)
DEFAULT_FEATURES = (10, 100)
DEFAULT_LEAVES = 100_000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=DEFAULT_BATCHES,
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=DEFAULT_FEATURES,
    )
    parser.add_argument("--leaves", type=int, default=DEFAULT_LEAVES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods",
        choices=METHODS,
        nargs="+",
        default=METHODS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "gpu_batch_scaling_results.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # run_worker reads this module-level setting when it constructs the worker
    # command. Keeping one worker implementation prevents protocol drift.
    import gpu_treeshap_bench as paper_bench

    paper_bench.N_REPEATS = args.repeats
    results = {
        "metadata": system_metadata(),
        "protocol": {
            "features": args.features,
            "leaves": args.leaves,
            "seed": args.seed,
            "batch_sizes": args.batches,
            "repeats": args.repeats,
            "methods": args.methods,
            "batch_inputs": "cached ten inputs repeated cyclically",
            "scheduling": "methods interleaved within each batch size",
            "timed_region": (
                "warm full shap_values call including backend transfers; "
                "construction and one warm-up call excluded"
            ),
        },
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for n_features in args.features:
        feature_results = results["results"].setdefault(str(n_features), {})
        model = MODEL_CACHE / (
            f"sklearn_rf_f{n_features}_l{args.leaves}_s{args.seed}.pkl"
        )
        for batch_size in args.batches:
            print(
                f"[batch] features={n_features} rows={batch_size}",
                flush=True,
            )
            batch_results = feature_results.setdefault(str(batch_size), {})
            for method in args.methods:
                started = time.time()
                result = run_worker(
                    model,
                    method,
                    n_samples=batch_size,
                )
                result["wall_started_utc"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)
                )
                batch_results[method] = result
                print(
                    f"  {method:18s} "
                    f"{1e3 * result.get('elapsed_s', float('nan')):10.3f} ms "
                    f"err={result.get('max_additivity_error', 'n/a')} "
                    f"status={result.get('status')}",
                    flush=True,
                )
                args.output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
