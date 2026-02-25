"""Sweep m_q for trees with 1000 and 10000 leaves.
Measures C++, Python speed and accuracy vs SHAP ground truth."""

import time
from unittest.mock import patch

import numpy as np
import shap
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from pgshapley import TreeExplainer as PGTreeExplainer
from pgshapley._cpp_ext import HAS_CPP_EXT

N_FEATURES = 20
N_TEST_SAMPLES = 10
N_REPEATS = 7


def build_tree(n_leaves, seed=42):
    n_samples = max(500, n_leaves * 5)
    X, y = make_regression(
        n_samples=n_samples, n_features=N_FEATURES,
        n_informative=N_FEATURES, random_state=seed,
    )
    model = DecisionTreeRegressor(max_leaf_nodes=n_leaves, random_state=seed).fit(X, y)
    return model, X[:N_TEST_SAMPLES]


def time_fn(fn, X, n_repeats):
    fn(X)  # warmup
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def main():
    for n_leaves in [1000, 10000]:
        model, X_test = build_tree(n_leaves)

        # SHAP ground truth
        expl_shap = shap.TreeExplainer(model)
        sv_shap = expl_shap.shap_values(X_test)
        t_shap = time_fn(expl_shap.shap_values, X_test, N_REPEATS)

        # Determine D
        from pgshapley.treeshap.sklearn import sklearn_to_unified
        from pgshapley.treeshap.product_games import _dfs_build_leaf_rules
        unified = sklearn_to_unified(model)
        tree0 = unified.trees[0]
        rules = _dfs_build_leaf_rules(tree0)
        D = max(len(f) for f in rules.feature_ids)
        default_mq = max(1, (D + 1) // 2)

        print(f"\n{'='*90}")
        print(f"n_leaves={n_leaves}, n_features={N_FEATURES}, D={D}, default m_q={default_mq}")
        print(f"SHAP baseline: {t_shap*1000:.3f} ms")
        print(f"{'='*90}")

        header = (
            f"{'m_q':>4}  {'Python (ms)':>12}  {'C++ (ms)':>12}"
            f"  {'max|err|':>12}  {'mean|err|':>12}"
            f"  {'py/shap':>8}  {'c++/shap':>8}"
        )
        print(header)
        print("-" * len(header))

        for mq in range(1, D + 1):
            # Python path
            with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", False):
                expl_py = PGTreeExplainer(model, m_q=mq)
                fn_py = lambda X: expl_py.shap_values(X, check_additivity=False)
                t_py = time_fn(fn_py, X_test, N_REPEATS)

            # C++ path
            if HAS_CPP_EXT:
                with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", True):
                    expl_cpp = PGTreeExplainer(model, m_q=mq)
                    fn_cpp = lambda X: expl_cpp.shap_values(X, check_additivity=False)
                    t_cpp = time_fn(fn_cpp, X_test, N_REPEATS)
                    sv_cpp = expl_cpp.shap_values(X_test, check_additivity=False)
            else:
                t_cpp = float('nan')
                sv_cpp = sv_py

            err = np.abs(sv_cpp - sv_shap)
            max_err = err.max()
            mean_err = err.mean()

            marker = " <-- default" if mq == default_mq else ""
            print(
                f"{mq:>4}  {t_py*1000:>12.3f}  {t_cpp*1000:>12.3f}"
                f"  {max_err:>12.2e}  {mean_err:>12.2e}"
                f"  {t_py/t_shap:>8.2f}x  {t_cpp/t_shap:>8.2f}x"
                f"{marker}"
            )


if __name__ == "__main__":
    main()
