"""Run the C++ explain kernel in a tight loop for perf profiling."""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from quadrashap import TreeExplainer

n_leaves = 3000
n_samples = max(500, n_leaves * 5)
X, y = make_regression(n_samples=n_samples, n_features=10, random_state=42)
model = DecisionTreeRegressor(max_leaf_nodes=n_leaves, random_state=42).fit(X, y)

expl = TreeExplainer(model)
X_test = X[:10]

# warmup
expl.shap_values(X_test, check_additivity=False)

# tight loop for ~10 seconds of samples
for _ in range(500):
    expl.shap_values(X_test, check_additivity=False)
