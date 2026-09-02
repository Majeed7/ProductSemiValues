"""
Benchmark: SHAP TreeExplainer vs quadrashap TreeExplainer (various backends).

Trees range from 10 to 10 000 leaves (exponential scale).

Run with:
    pytest benchmarks/test_bench_treeshap_e2e.py -v \
        --benchmark-group-by=param:n_leaves --benchmark-sort=mean
"""

import numpy as np
import pytest

import shap

from quadrashap import TreeExplainer as PGTreeExplainer
from quadrashap.product_games.shapley import JAX_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 10
N_TEST_SAMPLES = 10

# Exponential scale: 10, ~32, 100, ~316, 1000, ~3162, 10000
TREE_SIZES = [int(round(x)) for x in np.geomspace(10, 10_000, num=7)]


def _build_tree(n_leaves: int, seed: int = 42):
    """Train a DecisionTreeRegressor with approximately *n_leaves* leaves."""
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    # Need enough samples so the tree can actually grow to the desired size.
    n_samples = max(500, n_leaves * 5)
    X, y = make_regression(
        n_samples=n_samples, n_features=N_FEATURES, random_state=seed
    )
    model = DecisionTreeRegressor(
        max_leaf_nodes=n_leaves, random_state=seed
    ).fit(X, y)
    X_test = X[:N_TEST_SAMPLES]
    return model, X_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=TREE_SIZES, ids=[f"leaves={s}" for s in TREE_SIZES])
def n_leaves(request):
    return request.param


def _explainer_factories():
    """Yield (id_string, callable(model) -> explainer_with_shap_values)."""

    # --- SHAP baseline ---
    def _shap(model):
        expl = shap.TreeExplainer(model)
        return expl.shap_values

    yield pytest.param(_shap, id="shap")

    # --- quadrashap backends ---
    for method in ("numpy_prefix_scan", "numpy_logspace"):

        def _pg(model, _m=method):
            expl = PGTreeExplainer(model, backend_method=_m)
            return lambda X: expl.shap_values(X, check_additivity=False)

        yield pytest.param(_pg, id=f"pg_{method}")

    # --- quadrashap O(m_q * L) quadrature-tree backend (Python DFS) ---
    def _pg_quadrature_tree_py(model):
        expl = PGTreeExplainer(
            model, tree_solver="quadrature_tree", use_cpp=False
        )
        return lambda X: expl.shap_values(X, check_additivity=False)

    yield pytest.param(_pg_quadrature_tree_py, id="pg_quadrature_tree_py")

    # --- quadrashap O(m_q * L) quadrature-tree backend (C++ kernel) ---
    def _pg_quadrature_tree_cpp(model):
        expl = PGTreeExplainer(
            model, tree_solver="quadrature_tree", use_cpp=True
        )
        return lambda X: expl.shap_values(X, check_additivity=False)

    yield pytest.param(_pg_quadrature_tree_cpp, id="pg_quadrature_tree_cpp")

    if JAX_AVAILABLE:
        for method in ("jax_prefix_scan", "jax_logspace"):

            def _pg_jax(model, _m=method):
                expl = PGTreeExplainer(model, backend_method=_m)
                return lambda X: expl.shap_values(X, check_additivity=False)

            yield pytest.param(_pg_jax, id=f"pg_{method}")


@pytest.fixture(params=list(_explainer_factories()))
def explainer_factory(request):
    return request.param


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def test_bench_treeshap_e2e(benchmark, explainer_factory, n_leaves):
    model, X_test = _build_tree(n_leaves)
    shap_values_fn = explainer_factory(model)

    # Warmup (also triggers JAX JIT compilation if applicable).
    shap_values_fn(X_test)

    benchmark(shap_values_fn, X_test)
