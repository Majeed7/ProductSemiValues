"""
Compare SHAP vs pgshapley-Python vs pgshapley-C++ across tree sizes.
"""

import time
from unittest.mock import patch

import numpy as np

import shap
from pgshapley import TreeExplainer as PGTreeExplainer
from pgshapley._cpp_ext import HAS_CPP_EXT


N_FEATURES = 10
N_TEST_SAMPLES = 10
TREE_SIZES = [int(round(x)) for x in np.geomspace(10, 10_000, num=7)]
N_REPEATS = 5  # median of this many runs


def _build_tree(n_leaves, seed=42):
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    n_samples = max(500, n_leaves * 5)
    X, y = make_regression(n_samples=n_samples, n_features=N_FEATURES, random_state=seed)
    model = DecisionTreeRegressor(max_leaf_nodes=n_leaves, random_state=seed).fit(X, y)
    X_test = X[:N_TEST_SAMPLES]
    return model, X_test


def _time_fn(fn, X, n_repeats):
    # warmup
    fn(X)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def main():
    if not HAS_CPP_EXT:
        print("WARNING: C++ extension not available, skipping C++ column")

    header = f"{'leaves':>8}  {'SHAP (ms)':>12}  {'Python (ms)':>12}"
    if HAS_CPP_EXT:
        header += f"  {'C++ (ms)':>12}  {'py/C++':>8}  {'SHAP/C++':>8}"
    print(header)
    print("-" * len(header))

    for n_leaves in TREE_SIZES:
        model, X_test = _build_tree(n_leaves)

        # SHAP
        expl_shap = shap.TreeExplainer(model)
        t_shap = _time_fn(expl_shap.shap_values, X_test, N_REPEATS)

        # pgshapley Python (force Python path)
        with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", False):
            expl_py = PGTreeExplainer(model)
            fn_py = lambda X: expl_py.shap_values(X, check_additivity=False)
            t_py = _time_fn(fn_py, X_test, N_REPEATS)

        # pgshapley C++
        if HAS_CPP_EXT:
            with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", True):
                expl_cpp = PGTreeExplainer(model)
                fn_cpp = lambda X: expl_cpp.shap_values(X, check_additivity=False)
                t_cpp = _time_fn(fn_cpp, X_test, N_REPEATS)

        row = f"{n_leaves:>8}  {t_shap*1000:>12.3f}  {t_py*1000:>12.3f}"
        if HAS_CPP_EXT:
            speedup_py = t_py / t_cpp
            speedup_shap = t_shap / t_cpp
            row += f"  {t_cpp*1000:>12.3f}  {speedup_py:>8.2f}x  {speedup_shap:>8.2f}x"
        print(row)


if __name__ == "__main__":
    main()
