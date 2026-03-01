"""Tests that the C++ extension produces identical results to the Python backend."""

import numpy as np
import pytest
from unittest.mock import patch

from pgshapley._cpp_ext import HAS_CPP_EXT

pytestmark = pytest.mark.skipif(not HAS_CPP_EXT, reason="C++ extension not available")


def _explain_python(model, X):
    """Force Python path and return SHAP values."""
    with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", False):
        from pgshapley import TreeExplainer

        expl = TreeExplainer(model)
        return expl.shap_values(X, check_additivity=False), expl.expected_value


def _explain_cpp(model, X):
    """Use C++ path and return SHAP values."""
    with patch("pgshapley.treeshap.product_games.HAS_CPP_EXT", True):
        from pgshapley import TreeExplainer

        expl = TreeExplainer(model)
        return expl.shap_values(X, check_additivity=False), expl.expected_value


def test_cpp_matches_python_decision_tree_regressor():
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression(n_samples=200, n_features=7, random_state=0)
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)

    X_test = X[:25]
    sv_py, ev_py = _explain_python(model, X_test)
    sv_cpp, ev_cpp = _explain_cpp(model, X_test)

    np.testing.assert_allclose(ev_cpp, ev_py, atol=1e-12)
    np.testing.assert_allclose(sv_cpp, sv_py, atol=1e-12, rtol=1e-12)


def test_cpp_matches_python_random_forest():
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression(n_samples=300, n_features=6, noise=0.1, random_state=1)
    model = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=1).fit(X, y)

    X_test = X[:20]
    sv_py, ev_py = _explain_python(model, X_test)
    sv_cpp, ev_cpp = _explain_cpp(model, X_test)

    np.testing.assert_allclose(ev_cpp, ev_py, atol=1e-12)
    np.testing.assert_allclose(sv_cpp, sv_py, atol=1e-12, rtol=1e-12)


def test_cpp_matches_python_classifier():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    X_test = X[:10]
    sv_py, ev_py = _explain_python(model, X_test)
    sv_cpp, ev_cpp = _explain_cpp(model, X_test)

    np.testing.assert_allclose(ev_cpp, ev_py, atol=1e-12)
    np.testing.assert_allclose(sv_cpp, sv_py, atol=1e-12, rtol=1e-12)


def test_cpp_additivity():
    """Verify that C++ SHAP values satisfy the additivity property."""
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X, y)

    from pgshapley import TreeExplainer

    expl = TreeExplainer(model)
    X_test = X[:15]
    sv = expl.shap_values(X_test, check_additivity=True)
    # If we get here, additivity check passed
    assert sv.shape == (15, 5)
