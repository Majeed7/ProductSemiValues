"""Benchmark: Python vs C++ vs SHAP with m_q = ceil(D/3), across tree sizes."""

import time
from unittest.mock import patch

import numpy as np
import shap
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from pgshapley import TreeExplainer as PGTreeExplainer
from pgshapley._cpp_ext import HAS_CPP_EXT
from pgshapley.treeshap.sklearn import sklearn_to_unified
from pgshapley.treeshap.product_games import _dfs_build_leaf_rules

N_FEATURES = 20
N_TEST_SAMPLES = 10
N_REPEATS = 7
TREE_SIZES = [int(round(x)) for x in np.geomspace(10, 10_000, num=7)]


def build_tree(n_leaves, seed=42):
    n_samples = max(500, n_leaves * 5)
    X, y = make_regression(
        n_samples=n_samples, n_features=N_FEATURES,
        n_informative=N_FEATURES, random_state=seed,
    )
    model = DecisionTreeRegressor(max_leaf_nodes=n_leaves, random_state=seed).fit(X, y)
    return model, X[:N_TEST_SAMPLES]


def get_D(model):
    unified = sklearn_to_unified(model)
    rules = _dfs_build_leaf_rules(unified.trees[0])
    return max(len(f) for f in rules.feature_ids)


def time_fn(fn, X, n_repeats):
    fn(X)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def main():
    import math

    header = (
        f"{'leaves':>8}  {'D':>3}  {'m_q':>4}"
        f"  {'SHAP (ms)':>10}  {'Py (ms)':>10}  {'C++ (ms)':>10}"
        f"  {'py/SHAP':>8}  {'C++/SHAP':>8}"
        f"  {'max|err|':>12}  {'mean|err|':>12}"
    )
    print(header)
    print("-" * len(header))

    for n_leaves in TREE_SIZES:
        model, X_test = build_tree(n_leaves)
        D = get_D(model)
        mq = max(1, math.ceil(D / 3))

        # SHAP
        expl_shap = shap.TreeExplainer(model)
        sv_shap = expl_shap.shap_values(X_test)
        t_shap = time_fn(expl_shap.shap_values, X_test, N_REPEATS)

        # Python
        with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", False):
            expl_py = PGTreeExplainer(model, m_q=mq)
            fn_py = lambda X: expl_py.shap_values(X, check_additivity=False)
            sv_py = expl_py.shap_values(X_test, check_additivity=False)
            t_py = time_fn(fn_py, X_test, N_REPEATS)

        # C++
        with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", True):
            expl_cpp = PGTreeExplainer(model, m_q=mq)
            fn_cpp = lambda X: expl_cpp.shap_values(X, check_additivity=False)
            sv_cpp = expl_cpp.shap_values(X_test, check_additivity=False)
            t_cpp = time_fn(fn_cpp, X_test, N_REPEATS)

        err = np.abs(sv_cpp - sv_shap)

        print(
            f"{n_leaves:>8}  {D:>3}  {mq:>4}"
            f"  {t_shap*1000:>10.3f}  {t_py*1000:>10.3f}  {t_cpp*1000:>10.3f}"
            f"  {t_py/t_shap:>8.2f}x  {t_cpp/t_shap:>8.2f}x"
            f"  {err.max():>12.2e}  {err.mean():>12.2e}"
        )


if __name__ == "__main__":
    main()
