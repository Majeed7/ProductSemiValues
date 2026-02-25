from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import numpy as np


def _normalize_proba(counts: np.ndarray) -> np.ndarray:
    """Normalize class counts to probabilities.

    counts: (..., n_classes)
    """
    denom = counts.sum(axis=-1, keepdims=True)
    # Avoid divide by zero (should not happen for sklearn trees, but be safe)
    denom = np.where(denom == 0.0, 1.0, denom)
    return counts / denom


def _sklearn_tree_to_unified(tree, *, objective: str) -> "treeshap.unified.UnifiedTree":
    from .unified import UnifiedTree

    children_left = np.asarray(tree.children_left, dtype=np.int32)
    children_right = np.asarray(tree.children_right, dtype=np.int32)
    feature = np.asarray(tree.feature, dtype=np.int32)
    threshold = np.asarray(tree.threshold, dtype=np.float64)

    # Weighted sample counts per node
    if hasattr(tree, "weighted_n_node_samples"):
        node_weight = np.asarray(tree.weighted_n_node_samples, dtype=np.float64)
    else:
        # fallback
        node_weight = np.asarray(tree.n_node_samples, dtype=np.float64)

    raw_value = np.asarray(tree.value)

    # sklearn shapes:
    # - regression: (n_nodes, n_outputs, 1)
    # - classification: (n_nodes, n_outputs, n_classes)
    if raw_value.ndim != 3:
        raise ValueError(f"Unexpected sklearn tree_.value shape: {raw_value.shape}")

    if objective == "regression":
        # (n_nodes, n_outputs, 1) -> (n_nodes, n_outputs)
        values = raw_value[:, :, 0].astype(np.float64, copy=False)
        n_outputs = values.shape[1]
    elif objective == "classification":
        # Take probabilities
        # (n_nodes, 1, n_classes) -> (n_nodes, n_classes)
        # sklearn supports multi-output classification; we only support single-output.
        if raw_value.shape[1] != 1:
            raise NotImplementedError(
                "Multi-output classification is not supported in this implementation."
            )
        counts = raw_value[:, 0, :].astype(np.float64, copy=False)
        values = _normalize_proba(counts)
        n_outputs = values.shape[1]
    else:
        raise ValueError(f"Unknown objective: {objective}")

    n_features = int(tree.n_features)
    return UnifiedTree(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        values=values,
        node_weight=node_weight,
        n_features=n_features,
        n_outputs=n_outputs,
    )


def _infer_objective(model) -> str:
    """Infer objective (regression vs classification) for supported sklearn models."""
    # sklearn estimators define _estimator_type
    est_type = getattr(model, "_estimator_type", None)
    if est_type == "classifier":
        return "classification"
    if est_type == "regressor":
        return "regression"

    # fallback
    if hasattr(model, "predict_proba"):
        return "classification"
    return "regression"


def _iter_sklearn_trees(model) -> Tuple[List[object], np.ndarray]:
    """Return list of underlying sklearn trees and their weights in the ensemble output."""

    # Single tree
    if hasattr(model, "tree_"):
        return [model], np.array([1.0], dtype=np.float64)

    # RandomForest / ExtraTrees
    if hasattr(model, "estimators_"):
        estimators = model.estimators_
        # sklearn forests store as list
        if isinstance(estimators, (list, tuple)):
            trees = list(estimators)
        else:
            # Some ensembles use ndarray (e.g., GradientBoosting)
            trees = list(np.ravel(estimators))

        # Heuristics for supported ensembles
        cls_name = model.__class__.__name__.lower()
        if "randomforest" in cls_name or "extratrees" in cls_name:
            w = np.full(len(trees), 1.0 / max(1, len(trees)), dtype=np.float64)
            return trees, w

        # AdaBoost-style: estimator_weights_
        if hasattr(model, "estimator_weights_"):
            w = np.asarray(model.estimator_weights_, dtype=np.float64)
            if w.ndim != 1 or w.shape[0] != len(trees):
                raise ValueError("Unexpected estimator_weights_ shape.")
            # Some AdaBoost variants normalize weights implicitly; keep as-is.
            return trees, w

        raise NotImplementedError(
            f"Unsupported sklearn ensemble type for conversion: {model.__class__.__name__}"
        )

    raise NotImplementedError(f"Unsupported sklearn model type: {model.__class__.__name__}")


def sklearn_to_unified(model, feature_names: Optional[List[str]] = None) -> "treeshap.unified.UnifiedEnsemble":
    """Convert a supported scikit-learn tree model / ensemble into a UnifiedEnsemble."""

    from .unified import UnifiedEnsemble

    objective = _infer_objective(model)
    trees, tree_weights = _iter_sklearn_trees(model)

    unified_trees = []
    n_features: Optional[int] = None
    n_outputs: Optional[int] = None

    for est in trees:
        ut = _sklearn_tree_to_unified(est.tree_, objective=objective)
        unified_trees.append(ut)
        n_features = ut.n_features if n_features is None else n_features
        n_outputs = ut.n_outputs if n_outputs is None else n_outputs
        if ut.n_features != n_features:
            raise ValueError("Inconsistent n_features across trees")
        if ut.n_outputs != n_outputs:
            raise ValueError("Inconsistent n_outputs across trees")

    if n_features is None or n_outputs is None:
        raise RuntimeError("No trees found")

    if feature_names is None:
        # Best effort
        if hasattr(model, "feature_names_in_"):
            feature_names = list(map(str, model.feature_names_in_))
        else:
            feature_names = None

    return UnifiedEnsemble(
        trees=unified_trees,
        tree_weights=tree_weights.astype(np.float64, copy=False),
        n_features=n_features,
        n_outputs=n_outputs,
        model_type="sklearn",
        objective=objective,
        feature_names=feature_names,
    )
