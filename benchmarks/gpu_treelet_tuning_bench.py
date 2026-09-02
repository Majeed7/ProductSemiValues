"""Time A6000-relevant CUDA treelet launch and crossover constants."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from gpu_treeshap_bench import MODEL_CACHE, RESULTS, run_worker, system_metadata


OUTPUT = RESULTS / "gpu_treelet_tuning_results.json"
FEATURES = (10, 100)
TREELET_BATCHES = (8_000, 32_000)
CROSSOVER_BATCHES = (500, 1_000, 1_536, 2_000, 4_000)
TUNING_VARS = (
    "QUADRASHAP_CUDA_TREELET_ROWS_PER_WARP",
    "QUADRASHAP_CUDA_TREELET_SIZE",
    "QUADRASHAP_CUDA_TREELET_MIN_ROWS",
)


@contextmanager
def tuning_environment(overrides: dict[str, int]):
    old = {name: os.environ.get(name) for name in TUNING_VARS}
    try:
        for name in TUNING_VARS:
            os.environ.pop(name, None)
        os.environ.update({name: str(value) for name, value in overrides.items()})
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def save(results: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")


def measure(
    results: dict,
    *,
    group: str,
    config: str,
    overrides: dict[str, int],
    batches: tuple[int, ...],
) -> None:
    config_results = results["results"].setdefault(group, {}).setdefault(
        config,
        {"overrides": overrides, "measurements": {}},
    )
    with tuning_environment(overrides):
        for n_features in FEATURES:
            by_feature = config_results["measurements"].setdefault(
                str(n_features), {}
            )
            model = MODEL_CACHE / (
                f"sklearn_rf_f{n_features}_l100000_s42.pkl"
            )
            for batch in batches:
                result = run_worker(
                    model,
                    "quadrashap_gpu",
                    n_samples=batch,
                )
                by_feature[str(batch)] = result
                print(
                    f"[{group}/{config}] d={n_features} rows={batch}: "
                    f"{1e3 * result.get('elapsed_s', float('nan')):.3f} ms "
                    f"status={result.get('status')}",
                    flush=True,
                )
                save(results)


def main() -> int:
    results = {
        "metadata": system_metadata(),
        "protocol": {
            "model": "100000-leaf synthetic random forest, seed 42",
            "repeats": 3,
            "method": "quadrashap_gpu",
            "timed_region": (
                "warm full shap_values call including backend transfers; "
                "construction and one warm-up call excluded"
            ),
        },
        "results": {},
    }

    for rows_per_warp in (32, 64, 128, 256):
        measure(
            results,
            group="row_bank",
            config=f"rows_{rows_per_warp}",
            overrides={
                "QUADRASHAP_CUDA_TREELET_ROWS_PER_WARP": rows_per_warp
            },
            batches=TREELET_BATCHES,
        )

    for treelet_size in (7, 15):
        for rows_per_warp in (64, 128):
            measure(
                results,
                group="treelet_size",
                config=f"p{treelet_size}_rows_{rows_per_warp}",
                overrides={
                    "QUADRASHAP_CUDA_TREELET_SIZE": treelet_size,
                    "QUADRASHAP_CUDA_TREELET_ROWS_PER_WARP": rows_per_warp,
                },
                batches=TREELET_BATCHES,
            )

    for mode, min_rows in (("treelet", 1), ("compact", 1_000_000_000)):
        measure(
            results,
            group="crossover",
            config=mode,
            overrides={"QUADRASHAP_CUDA_TREELET_MIN_ROWS": min_rows},
            batches=CROSSOVER_BATCHES,
        )

    results["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    save(results)
    print(f"Wrote {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
