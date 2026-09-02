#!/usr/bin/env python3
"""Compare QuadraSHAP with TN-SHAP on RBF-kernel product games.

For an RBF kernel term and an explained point ``x``, define

    u_j = exp(-gamma * (x_j - z_j)^2)
    v(S) = product_{j in S} u_j.

This is a tensor train of rank one.  A KRR-style value function is represented
as a weighted sum of these rank-one terms.

The TN-SHAP implementation follows the feature-space selector experiment in
farzana0/TN-SHAP:

* Chebyshev nodes on [0, 1]
* a monomial Vandermonde pseudoinverse
* coefficient integration with weights 1 / (s + 1)

We report both the upstream-style evaluator and a rank-one specialization that
shares the full product across leave-one-feature-out evaluations.  QuadraSHAP
uses the repository's NumPy log-space implementation.  Accuracy is measured
against an independent high-precision polynomial recurrence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import mpmath as mp
import numpy as np
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quadrashap.product_games.shapley import ProductGamesShapleyNumpy


TN_SHAP_DIR = ROOT / "benchmarks" / "_external" / "TN-SHAP"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "kernel_tnshap"
TN_SHAP_SYNTHETIC_DIR = (
    TN_SHAP_DIR / "experiments" / "03_synthetic_experiments" / "scripts"
)
if str(TN_SHAP_SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(TN_SHAP_SYNTHETIC_DIR))

import synthetic_rank_sweep_basic as tnshap_upstream


@dataclass
class Result:
    regime: str
    n_terms: int
    n_features: int
    gamma: float
    method: str
    nodes: int
    median_seconds: float
    min_seconds: float
    rel_l2_error: float
    max_abs_error: float
    efficiency_error: float
    status: str = "ok"
    message: str = ""


class KernelProductSum(torch.nn.Module):
    """Weighted sum of rank-one product games in multilinear coordinates."""

    def __init__(self, K: np.ndarray, alpha: np.ndarray):
        super().__init__()
        self.register_buffer("K", torch.as_tensor(K, dtype=torch.float64))
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float64))

    def forward(self, selectors: torch.Tensor) -> torch.Tensor:
        # selectors: [batch, d].  Every product term is a rank-one TT.
        factors = 1.0 + selectors[:, None, :] * self.K[None, :, :]
        terms = torch.prod(factors, dim=2)
        return terms @ self.alpha


class UpstreamNumpyProductGame:
    """Expose a product game through TN-SHAP's upstream ``evaluate`` API."""

    def __init__(self, K: np.ndarray, alpha: np.ndarray):
        self.K = np.asarray(K, dtype=np.float64)
        self.alpha = np.asarray(alpha, dtype=np.float64)

    def evaluate(self, selectors: np.ndarray) -> np.ndarray:
        selectors = np.asarray(selectors, dtype=np.float64)
        batch, d = selectors.shape
        # Keep temporary B x R_chunk x d tensors around 64 MiB. This does not
        # change the model; it only prevents the exact KRR expansion from
        # allocating multi-gigabyte intermediates at large d.
        max_elements = 8_000_000
        chunk = max(1, min(len(self.alpha), max_elements // (batch * d)))
        out = np.zeros(batch, dtype=np.float64)
        for start in range(0, len(self.alpha), chunk):
            stop = min(start + chunk, len(self.alpha))
            factors = (
                1.0
                + selectors[:, None, :]
                * self.K[None, start:stop, :]
            )
            out += np.prod(factors, axis=2) @ self.alpha[start:stop]
        return out


def chebyshev_nodes_01(m: int) -> torch.Tensor:
    k = torch.arange(m, dtype=torch.float64)
    # This matches experiments/04_scaling/scaling_train_tensor_network.py.
    return 0.5 * (torch.cos((2 * k + 1) * math.pi / (2 * m)) + 1.0)


def tnshap_plan(m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nodes, Vandermonde pseudoinverse, and coefficient integrals."""
    nodes = chebyshev_nodes_01(m)
    vandermonde = torch.vander(nodes, N=m, increasing=True)
    pinv = torch.linalg.pinv(vandermonde)
    coefficient_integrals = 1.0 / torch.arange(1, m + 1, dtype=torch.float64)
    return nodes, pinv, coefficient_integrals


@torch.no_grad()
def tnshap_repo_path(K: np.ndarray, alpha: np.ndarray, m: int) -> np.ndarray:
    """Upstream-style selector computation.

    This follows ``src/shapiq_benchmark/tnshap_featuremapped_selector_eval.py``:
    other features receive selector ``t``, while the explained feature is
    clamped to one on the on-path and zero on the off-path.  The shared
    Vandermonde pseudoinverse is applied to every resulting G_i(t).
    """
    _, d = K.shape
    model = KernelProductSum(K, alpha)
    nodes, pinv, coefficient_integrals = tnshap_plan(m)

    path_base = nodes[:, None].repeat(1, d)
    phi = torch.empty(d, dtype=torch.float64)
    for i in range(d):
        path_on = path_base.clone()
        path_off = path_base.clone()
        path_on[:, i] = 1.0
        path_off[:, i] = 0.0
        gi = model(path_on) - model(path_off)
        coefficients = pinv @ gi
        phi[i] = coefficient_integrals @ coefficients
    return phi.numpy()


@torch.no_grad()
def tnshap_selector_default(K: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """TN-SHAP selector CLI defaults: full degree and one lstsq per feature.

    ``tnshap_featuremapped_selector_eval.py`` defaults ``max_degree`` to
    ``D - 1`` and the number of Chebyshev nodes to ``D``.  Its implementation
    calls ``torch.linalg.lstsq`` separately for every feature.
    """
    _, d = K.shape
    model = KernelProductSum(K, alpha)
    nodes = chebyshev_nodes_01(d)
    vandermonde = torch.vander(nodes, N=d, increasing=True)
    coefficient_integrals = 1.0 / torch.arange(1, d + 1, dtype=torch.float64)
    path_base = nodes[:, None].repeat(1, d)
    phi = torch.empty(d, dtype=torch.float64)

    for i in range(d):
        path_on = path_base.clone()
        path_off = path_base.clone()
        path_on[:, i] = 1.0
        path_off[:, i] = 0.0
        gi = model(path_on) - model(path_off)
        coefficients = torch.linalg.lstsq(
            vandermonde, gi.unsqueeze(-1)
        ).solution.squeeze(-1)
        phi[i] = coefficient_integrals @ coefficients
    return phi.numpy()


@torch.no_grad()
def tnshap_upstream_calculator(
    K: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Call TN-SHAP's upstream synthetic-experiment calculator directly."""
    d = K.shape[1]
    nodes = tnshap_upstream.TensorTreeTeacher._generate_chebyshev_nodes(
        None, d
    )
    calculator = tnshap_upstream.TNShapCalculator(
        UpstreamNumpyProductGame(K, alpha),
        nodes,
        d,
    )
    return np.asarray(
        calculator.shapley_values_tnshap(np.ones(d, dtype=np.float64)),
        dtype=np.float64,
    )


@torch.no_grad()
def tnshap_rank1_shared(K: np.ndarray, alpha: np.ndarray, m: int) -> np.ndarray:
    """TN-SHAP with the rank-one leave-one-out products shared explicitly."""
    Kt = torch.as_tensor(K, dtype=torch.float64)
    at = torch.as_tensor(alpha, dtype=torch.float64)
    nodes, pinv, coefficient_integrals = tnshap_plan(m)
    interpolation_integrals = coefficient_integrals @ pinv

    gi = torch.empty((m, K.shape[1]), dtype=torch.float64)
    for q, node in enumerate(nodes):
        factors = 1.0 + node * Kt
        full_products = torch.prod(factors, dim=1)
        term_weights = at * full_products
        gi[q] = torch.sum(
            term_weights[:, None] * Kt / factors,
            dim=0,
        )
    return (interpolation_integrals @ gi).numpy()


def quadrashap(K: np.ndarray, alpha: np.ndarray, m: int) -> np.ndarray:
    phi_terms = ProductGamesShapleyNumpy().phi_matrix_logspace(K, m_q=m)
    return np.sum(phi_terms * alpha[:, None], axis=0)


def quadrashap_rbf_shared(K: np.ndarray, alpha: np.ndarray, m: int) -> np.ndarray:
    """RBF-specific shared-product path.

    RBF factors satisfy ``0 < 1 + K <= 1``.  Consequently the direct product
    has no sign/overflow problem, and leave-one-feature-out products can be
    obtained by division just as in the rank-one TN specialization.
    """
    nodes, weights = np.polynomial.legendre.leggauss(m)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    out = np.zeros(K.shape[1], dtype=np.float64)
    for node, weight in zip(nodes, weights, strict=True):
        factors = 1.0 + node * K
        full_products = np.prod(factors, axis=1)
        out += weight * np.sum(
            (alpha * full_products)[:, None] * K / factors,
            axis=0,
        )
    return out


def high_precision_reference(
    K: np.ndarray,
    alpha: np.ndarray,
    *,
    decimal_digits: int,
) -> np.ndarray:
    """Exact coefficient integration using high-precision synthetic division."""
    mp.mp.dps = decimal_digits
    n_terms, d = K.shape
    out = [mp.mpf("0") for _ in range(d)]

    for r in range(n_terms):
        kr = [mp.mpf(float(v)) for v in K[r]]
        # Coefficients of P(t) = product_j (1 + K_j t), ascending in degree.
        coeff = [mp.mpf("1")]
        for k in kr:
            nxt = [mp.mpf("0")] * (len(coeff) + 1)
            for s, c in enumerate(coeff):
                nxt[s] += c
                nxt[s + 1] += k * c
            coeff = nxt

        ar = mp.mpf(float(alpha[r]))
        for i, k in enumerate(kr):
            # Divide P(t) by (1 + K_i t), then integrate the quotient.
            quotient = [mp.mpf("0")] * d
            quotient[0] = coeff[0]
            for s in range(1, d):
                quotient[s] = coeff[s] - k * quotient[s - 1]
            integral = mp.fsum(
                quotient[s] / mp.mpf(s + 1) for s in range(d)
            )
            out[i] += ar * k * integral

    return np.asarray([float(v) for v in out], dtype=np.float64)


def make_kernel_game(
    *,
    n_features: int,
    n_terms: int,
    gamma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_features)
    landmarks = rng.normal(size=(n_terms, n_features))
    factors = np.exp(-gamma * (landmarks - x[None, :]) ** 2)
    K = factors - 1.0

    if n_terms == 1:
        alpha = np.ones(1, dtype=np.float64)
    else:
        # Signed KRR-like dual weights, normalized to a stable overall scale.
        alpha = rng.normal(size=n_terms)
        alpha /= np.sum(np.abs(alpha))
    return K.astype(np.float64), alpha.astype(np.float64)


def efficiency_target(K: np.ndarray, alpha: np.ndarray) -> float:
    value_all = np.sum(alpha * np.prod(1.0 + K, axis=1))
    value_empty = np.sum(alpha)
    return float(value_all - value_empty)


def timed(
    fn: Callable[[], np.ndarray],
    *,
    repeats: int,
) -> tuple[np.ndarray, float, float]:
    values: np.ndarray | None = None
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        values = np.asarray(fn(), dtype=np.float64)
        times.append(time.perf_counter() - t0)
    assert values is not None
    return values, statistics.median(times), min(times)


def method_result(
    *,
    regime: str,
    K: np.ndarray,
    alpha: np.ndarray,
    gamma: float,
    reference: np.ndarray,
    target: float,
    method: str,
    nodes: int,
    fn: Callable[[], np.ndarray],
    repeats: int,
) -> Result:
    try:
        values, median_s, min_s = timed(fn, repeats=repeats)
        delta = values - reference
        reference_norm = float(np.linalg.norm(reference))
        return Result(
            regime=regime,
            n_terms=K.shape[0],
            n_features=K.shape[1],
            gamma=gamma,
            method=method,
            nodes=nodes,
            median_seconds=median_s,
            min_seconds=min_s,
            rel_l2_error=float(np.linalg.norm(delta) / max(reference_norm, 1e-300)),
            max_abs_error=float(np.max(np.abs(delta))),
            efficiency_error=float(abs(np.sum(values) - target)),
        )
    except BaseException as exc:
        return Result(
            regime=regime,
            n_terms=K.shape[0],
            n_features=K.shape[1],
            gamma=gamma,
            method=method,
            nodes=nodes,
            median_seconds=float("nan"),
            min_seconds=float("nan"),
            rel_l2_error=float("nan"),
            max_abs_error=float("nan"),
            efficiency_error=float("nan"),
            status="failed",
            message=f"{type(exc).__name__}: {exc}"[:300],
        )


def benchmark_case(
    *,
    regime: str,
    n_features: int,
    n_terms: int,
    gamma: float,
    seed: int,
    budget: int,
    full_tn_max_features: int,
    repeats: int,
    reference_digits: int,
) -> list[Result]:
    K, alpha = make_kernel_game(
        n_features=n_features,
        n_terms=n_terms,
        gamma=gamma,
        seed=seed,
    )
    reference = high_precision_reference(
        K, alpha, decimal_digits=reference_digits
    )
    target = efficiency_target(K, alpha)
    exact_nodes = (n_features + 1) // 2
    budget_nodes = min(budget, exact_nodes)
    tn_budget_nodes = min(budget, n_features)

    methods: list[tuple[str, int, Callable[[], np.ndarray]]] = [
        (
            "quadrashap_exact",
            exact_nodes,
            lambda: quadrashap(K, alpha, exact_nodes),
        ),
        (
            "quadrashap_rbf_shared_exact",
            exact_nodes,
            lambda: quadrashap_rbf_shared(K, alpha, exact_nodes),
        ),
        (
            f"quadrashap_m{budget}",
            budget_nodes,
            lambda: quadrashap(K, alpha, budget_nodes),
        ),
        (
            f"quadrashap_rbf_shared_m{budget}",
            budget_nodes,
            lambda: quadrashap_rbf_shared(K, alpha, budget_nodes),
        ),
        (
            f"tnshap_repo_m{budget}",
            tn_budget_nodes,
            lambda: tnshap_repo_path(K, alpha, tn_budget_nodes),
        ),
        (
            f"tnshap_tt1_shared_m{budget}",
            tn_budget_nodes,
            lambda: tnshap_rank1_shared(K, alpha, tn_budget_nodes),
        ),
    ]
    if n_features <= full_tn_max_features:
        methods.append(
            (
                "tnshap_tt1_full",
                n_features,
                lambda: tnshap_rank1_shared(K, alpha, n_features),
            )
        )

    rows: list[Result] = []
    for method, nodes, fn in methods:
        result = method_result(
            regime=regime,
            K=K,
            alpha=alpha,
            gamma=gamma,
            reference=reference,
            target=target,
            method=method,
            nodes=nodes,
            fn=fn,
            repeats=repeats,
        )
        rows.append(result)
        print(
            f"{regime:13s} terms={n_terms:3d} d={n_features:4d} "
            f"{method:25s} m={nodes:4d} "
            f"time={result.median_seconds:9.5f}s "
            f"relL2={result.rel_l2_error:9.2e} "
            f"eff={result.efficiency_error:9.2e} "
            f"[{result.status}]",
            flush=True,
        )
    return rows


def markdown_table(rows: list[Result]) -> str:
    lines = [
        "# QuadraSHAP vs TN-SHAP on RBF product games",
        "",
        "Median end-to-end CPU runtime includes node construction and the "
        "Vandermonde pseudoinverse or Gauss–Legendre rule construction. "
        "Errors are measured against a high-precision coefficient recurrence.",
        "",
        "| Regime | Terms | d | Method | Nodes | Time (ms) | Relative L2 error | Max abs. error | Efficiency error |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        time_ms = r.median_seconds * 1000.0
        lines.append(
            f"| {r.regime} | {r.n_terms} | {r.n_features} | {r.method} | "
            f"{r.nodes} | {time_ms:.3f} | {r.rel_l2_error:.3e} | "
            f"{r.max_abs_error:.3e} | {r.efficiency_error:.3e} |"
        )
    return "\n".join(lines) + "\n"


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-counts",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100, 200, 500],
    )
    parser.add_argument("--krr-feature-counts", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--krr-term-counts", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--full-tn-max-features", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-digits", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    rows: list[Result] = []
    with threadpool_limits(limits=args.threads):
        for d in args.feature_counts:
            for gamma_label, gamma in (("scaled_gamma", 1.0 / d), ("gamma_0.5", 0.5)):
                rows.extend(
                    benchmark_case(
                        regime=gamma_label,
                        n_features=d,
                        n_terms=1,
                        gamma=gamma,
                        seed=args.seed + d,
                        budget=args.budget,
                        full_tn_max_features=args.full_tn_max_features,
                        repeats=args.repeats,
                        reference_digits=args.reference_digits,
                    )
                )

        for n_terms in args.krr_term_counts:
            for d in args.krr_feature_counts:
                rows.extend(
                    benchmark_case(
                        regime="krr_sum",
                        n_features=d,
                        n_terms=n_terms,
                        gamma=1.0 / d,
                        seed=args.seed + 10_000 + 100 * n_terms + d,
                        budget=args.budget,
                        full_tn_max_features=args.full_tn_max_features,
                        repeats=args.repeats,
                        reference_digits=args.reference_digits,
                    )
                )

    csv_path = args.output_dir / "kernel_tnshap_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    metadata = {
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": args.threads,
        "quadrashap_revision": git_revision(ROOT),
        "tnshap_revision": git_revision(TN_SHAP_DIR),
        "tnshap_source": "https://github.com/farzana0/TN-SHAP",
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (args.output_dir / "README.md").write_text(markdown_table(rows))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
