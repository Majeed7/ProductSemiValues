from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import TreeShapBackend
from .product_games import ProductGamesTreeShapBackend
from .sklearn import sklearn_to_unified


def _default_phi_matrix_fn(method: str = "numpy_prefix_scan"):
    """Select a default product-game SHAP implementation."""
    from pgshapley.product_games.shapley import ProductGamesShapleyNumpy, ProductGamesShapleyJax, JAX_AVAILABLE

    method = method.lower()
    if method == "numpy_prefix_scan":
        return ProductGamesShapleyNumpy().phi_matrix_prefix_scan
    if method == "numpy_logspace":
        return ProductGamesShapleyNumpy().phi_matrix_logspace
    if method == "jax_prefix_scan":
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available, cannot use jax_prefix_scan")
        return ProductGamesShapleyJax().phi_matrix_prefix_scan
    if method == "jax_logspace":
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available, cannot use jax_logspace")
        return ProductGamesShapleyJax().phi_matrix_logspace

    raise ValueError(
        f"Unknown method '{method}'. Expected one of: numpy_prefix_scan, numpy_logspace, jax_prefix_scan, jax_logspace."
    )


class TreeExplainer:
    def __init__(
        self,
        model: Any,
        data: Optional[np.ndarray] = None,
        model_output: str = "raw",
        feature_perturbation: str = "interventional",
        feature_names=None,
        **deprecated_options,
    ):
        # SHAP's API includes these deprecated options; accept and ignore.
        _ = deprecated_options

        self.model = model
        self.data = data
        self.model_output = model_output
        # Match SHAP's behavior: when no background data is provided, the
        # default "interventional" mode falls back to "tree_path_dependent".
        if feature_perturbation == "interventional" and data is None:
            feature_perturbation = "tree_path_dependent"
        self.feature_perturbation = feature_perturbation
        self.feature_names = feature_names

        # Extended options
        self.backend_method = deprecated_options.pop("backend_method", "numpy_prefix_scan")
        self.batch_size = deprecated_options.pop("batch_size", 256)
        self.m_q = deprecated_options.pop("m_q", None)

        if model_output != "raw":
            raise NotImplementedError(
                "Only model_output='raw' is currently supported."
            )

        if feature_perturbation not in ("tree_path_dependent", "interventional"):
            raise NotImplementedError(
                "Only feature_perturbation='tree_path_dependent' is supported."
            )

        if feature_perturbation == "interventional":
            raise NotImplementedError(
                "This explainer implements the tree_path_dependent semantics. Please pass feature_perturbation='tree_path_dependent'."
            )

        # Convert model to unified representation
        self._unified = sklearn_to_unified(model, feature_names=feature_names)

        phi_fn = _default_phi_matrix_fn(self.backend_method)
        self._backend: TreeShapBackend = ProductGamesTreeShapBackend(
            phi_matrix_fn=phi_fn,
            m_q=self.m_q,
            batch_size=self.batch_size,
        )
        prepared = self._backend.prepare(self._unified)

        self.expected_value = prepared.expected_value  # numpy.ndarray, (n_outputs,)

    def shap_values(
        self,
        X: np.ndarray,
        y=None,
        tree_limit=None,
        approximate=False,
        check_additivity=True,
    ):
        # Keep signature compatible; ignore unsupported params.
        _ = y, approximate

        sv = self._backend.explain(X, tree_limit=tree_limit)
        # sv: (n_samples, n_features, n_outputs)
        n_outputs = sv.shape[2]

        if check_additivity:
            pred = self._predict_unified(X)
            recon = self.expected_value.reshape(1, -1) + sv.sum(axis=1)
            max_err = np.max(np.abs(recon - pred))
            if not np.isfinite(max_err) or max_err > 1e-5:
                raise AssertionError(
                    f"Additivity check failed: max |expected + sum(phi) - pred| = {max_err}"
                )

        if n_outputs == 1:
            return sv[:, :, 0]
        return sv

    def _predict_unified(self, X: np.ndarray) -> np.ndarray:
        """Compute model predictions using the unified model (no missing features)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        ens = self._unified
        n_samples = X.shape[0]
        out = np.zeros((n_samples, ens.n_outputs), dtype=np.float64)

        for tree, w in zip(ens.trees, ens.tree_weights):
            for i in range(n_samples):
                node = 0
                while True:
                    left = int(tree.children_left[node])
                    right = int(tree.children_right[node])
                    if left == -1 and right == -1:
                        break
                    f = int(tree.feature[node])
                    thr = float(tree.threshold[node])
                    if X[i, f] <= thr:
                        node = left
                    else:
                        node = right
                out[i] += w * tree.values[node]
        return out

    def __call__(self, X: np.ndarray, *args, **kwargs):
        # SHAP's explainers are callable; return shap_values.
        return self.shap_values(X, *args, **kwargs)

