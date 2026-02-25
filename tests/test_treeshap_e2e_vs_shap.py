import numpy as np
import pytest

import shap

from pgshapley import TreeExplainer as PGTreeExplainer


def test_e2e_decision_tree_regressor_matches_shap():
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression(n_samples=200, n_features=7, random_state=0)
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)

    expl_shap = shap.TreeExplainer(model)
    expl_pg = PGTreeExplainer(model)

    np.testing.assert_allclose(expl_pg.expected_value, expl_shap.expected_value, atol=1e-10)

    X_test = X[:25]
    sv_shap = expl_shap.shap_values(X_test)
    sv_pg = expl_pg.shap_values(X_test)

    np.testing.assert_allclose(sv_pg, sv_shap, atol=1e-6, rtol=1e-6)


def test_e2e_random_forest_regressor_matches_shap():
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression(n_samples=300, n_features=6, noise=0.1, random_state=1)
    model = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=1).fit(X, y)

    expl_shap = shap.TreeExplainer(model)
    expl_pg = PGTreeExplainer(model)

    np.testing.assert_allclose(expl_pg.expected_value, expl_shap.expected_value, atol=1e-10)

    X_test = X[:20]
    sv_shap = expl_shap.shap_values(X_test)
    sv_pg = expl_pg.shap_values(X_test)

    np.testing.assert_allclose(sv_pg, sv_shap, atol=1e-6, rtol=1e-6)


def test_e2e_decision_tree_classifier_matches_shap():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    expl_shap = shap.TreeExplainer(model)
    expl_pg = PGTreeExplainer(model)

    np.testing.assert_allclose(expl_pg.expected_value, expl_shap.expected_value, atol=1e-10)

    X_test = X[:10]
    sv_shap = expl_shap.shap_values(X_test)
    sv_pg = expl_pg.shap_values(X_test)

    np.testing.assert_allclose(sv_pg, sv_shap, atol=1e-6, rtol=1e-6)
