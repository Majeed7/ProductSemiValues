import numpy as np
import pytest

from pgshapley.treeshap.sklearn import sklearn_to_unified


def _predict_unified(ens, X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    out = np.zeros((X.shape[0], ens.n_outputs), dtype=np.float64)
    for tree, w in zip(ens.trees, ens.tree_weights):
        for i in range(X.shape[0]):
            node = 0
            while True:
                left = int(tree.children_left[node])
                right = int(tree.children_right[node])
                if left == -1 and right == -1:
                    break
                f = int(tree.feature[node])
                thr = float(tree.threshold[node])
                node = left if X[i, f] <= thr else right
            out[i] += w * tree.values[node]
    return out


def test_decision_tree_regressor_conversion_roundtrip():
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression(n_samples=200, n_features=7, random_state=0)
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)

    ens = sklearn_to_unified(model)
    assert ens.n_features == X.shape[1]
    assert ens.n_outputs == 1
    assert len(ens.trees) == 1
    assert ens.tree_weights.shape == (1,)
    np.testing.assert_allclose(ens.tree_weights, [1.0])

    ut = ens.trees[0]
    # Structural equality with sklearn tree arrays
    np.testing.assert_array_equal(ut.children_left, model.tree_.children_left)
    np.testing.assert_array_equal(ut.children_right, model.tree_.children_right)
    np.testing.assert_array_equal(ut.feature, model.tree_.feature)
    np.testing.assert_allclose(ut.threshold, model.tree_.threshold)

    # Predictions should match
    pred_skl = model.predict(X[:20]).reshape(-1, 1)
    pred_uni = _predict_unified(ens, X[:20])
    np.testing.assert_allclose(pred_uni, pred_skl, atol=1e-12)


def test_decision_tree_classifier_conversion_roundtrip():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    ens = sklearn_to_unified(model)
    assert ens.n_features == X.shape[1]
    assert ens.n_outputs == len(model.classes_)
    assert len(ens.trees) == 1

    proba_skl = model.predict_proba(X[:10])
    proba_uni = _predict_unified(ens, X[:10])
    np.testing.assert_allclose(proba_uni, proba_skl, atol=1e-12)


def test_random_forest_weights_and_predictions():
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression(n_samples=300, n_features=6, random_state=0)
    model = RandomForestRegressor(n_estimators=7, max_depth=4, random_state=0).fit(X, y)

    ens = sklearn_to_unified(model)
    assert len(ens.trees) == model.n_estimators
    np.testing.assert_allclose(ens.tree_weights.sum(), 1.0)

    pred_skl = model.predict(X[:15]).reshape(-1, 1)
    pred_uni = _predict_unified(ens, X[:15])
    np.testing.assert_allclose(pred_uni, pred_skl, atol=1e-12)
