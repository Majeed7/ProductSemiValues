"""Convert cached sklearn forests to an equivalent XGBoost JSON model.

This bridge exists only for benchmarking XGBoost's integrated GPU TreeSHAP
on exactly the same tree topology, leaf values, and path covers as the paper
models. XGBoost stores split thresholds and leaf values as float32, so those
two arrays are explicitly quantized during conversion.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np


def _xgboost_node_order(
    old_left: np.ndarray,
    old_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder nodes so every XGBoost child pair is consecutive."""
    old_by_new = [0]
    old_to_new = {0: 0}
    cursor = 0
    while cursor < len(old_by_new):
        old = old_by_new[cursor]
        if old_left[old] != -1:
            for child in (int(old_left[old]), int(old_right[old])):
                old_to_new[child] = len(old_by_new)
                old_by_new.append(child)
        cursor += 1

    order = np.asarray(old_by_new, dtype=np.int32)
    if order.size != old_left.size:
        raise ValueError("The sklearn tree contains unreachable nodes.")
    left = np.full(order.size, -1, dtype=np.int32)
    right = np.full(order.size, -1, dtype=np.int32)
    for new, old in enumerate(order):
        if old_left[old] != -1:
            left[new] = old_to_new[int(old_left[old])]
            right[new] = old_to_new[int(old_right[old])]
            if right[new] != left[new] + 1:
                raise AssertionError("XGBoost children must form a pair.")
    return order, left, right


def _tree_payload(
    estimator,
    tree_id: int,
    tree_weight: float,
    target: int | None,
) -> dict:
    tree = estimator.tree_
    old_left = np.asarray(tree.children_left, dtype=np.int32)
    old_right = np.asarray(tree.children_right, dtype=np.int32)
    order, left, right = _xgboost_node_order(old_left, old_right)
    feature = np.asarray(tree.feature, dtype=np.int32)[order]
    threshold = np.asarray(tree.threshold, dtype=np.float32)[order]
    cover = np.asarray(
        tree.weighted_n_node_samples,
        dtype=np.float32,
    )[order]
    raw_value = np.asarray(tree.value, dtype=np.float64)[order]
    if raw_value.ndim != 3 or raw_value.shape[1] != 1:
        raise NotImplementedError(
            "The XGBoost benchmark bridge supports single-output models only."
        )
    if raw_value.shape[2] == 1:
        node_value = raw_value[:, 0, 0]
    else:
        if target is None:
            raise ValueError(
                "A class target is required when converting a classifier."
            )
        counts = raw_value[:, 0, :]
        denominator = counts.sum(axis=1)
        denominator[denominator == 0.0] = 1.0
        node_value = counts[:, target] / denominator

    n_nodes = int(left.size)
    is_leaf = left == -1
    # sklearn routes x <= threshold left; XGBoost routes x < threshold left.
    # Advancing one float32 ULP makes the predicates equivalent for values as
    # represented by XGBoost's float32 DMatrix, including sparse text zeros.
    threshold[~is_leaf] = np.nextafter(
        threshold[~is_leaf],
        np.float32(np.inf),
    )
    parents = np.full(n_nodes, 2**31 - 1, dtype=np.int64)
    for parent in np.flatnonzero(~is_leaf):
        parents[int(left[parent])] = int(parent)
        parents[int(right[parent])] = int(parent)

    split_conditions = threshold.copy()
    leaf_values = np.asarray(
        node_value * tree_weight,
        dtype=np.float32,
    )
    split_conditions[is_leaf] = leaf_values[is_leaf]
    split_indices = feature.copy()
    split_indices[is_leaf] = 0

    # XGBoost's sparse page representation omits exact zeros. Route an omitted
    # zero to the same branch that sklearn takes for the split threshold.
    default_left = np.asarray(
        (~is_leaf) & (threshold > 0.0),
        dtype=np.int8,
    )
    return {
        "base_weights": np.where(is_leaf, leaf_values, 0.0)
        .astype(np.float32)
        .tolist(),
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": default_left.tolist(),
        "id": tree_id,
        "left_children": left.tolist(),
        "loss_changes": np.zeros(n_nodes, dtype=np.float32).tolist(),
        "parents": parents.tolist(),
        "right_children": right.tolist(),
        "split_conditions": split_conditions.tolist(),
        "split_indices": split_indices.tolist(),
        "split_type": np.zeros(n_nodes, dtype=np.int8).tolist(),
        "sum_hessian": cover.tolist(),
        "tree_param": {
            "num_deleted": "0",
            "num_feature": str(int(tree.n_features)),
            "num_nodes": str(n_nodes),
            "size_leaf_vector": "1",
        },
    }


def sklearn_forest_xgboost_json(
    model,
    *,
    target: int | None = None,
) -> dict:
    estimators = list(np.ravel(model.estimators_))
    if not estimators:
        raise ValueError("The forest contains no trees.")
    n_features = int(model.n_features_in_)
    tree_weight = 1.0 / len(estimators)
    trees = [
        _tree_payload(estimator, tree_id, tree_weight, target)
        for tree_id, estimator in enumerate(estimators)
    ]
    n_trees = len(trees)
    return {
        "learner": {
            "attributes": {
                "quadrashap_bridge": "sklearn_random_forest",
            },
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "model": {
                    "cats": {
                        "enc": [],
                        "feature_segments": [],
                        "sorted_idx": [],
                    },
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": str(n_trees),
                    },
                    "iteration_indptr": list(range(n_trees + 1)),
                    "tree_info": [0] * n_trees,
                    "trees": trees,
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": "[0E0]",
                "boost_from_average": "0",
                "num_class": "0",
                "num_feature": str(n_features),
                "num_target": "1",
            },
            "objective": {
                "name": "reg:squarederror",
                "reg_loss_param": {"scale_pos_weight": "1"},
            },
        },
        "version": [3, 1, 3],
    }


def sklearn_forest_to_xgboost(
    model,
    *,
    target: int | None = None,
    device: str = "cuda:0",
):
    """Return an XGBoost Booster with the converted forest."""
    import xgboost as xgb

    payload = sklearn_forest_xgboost_json(model, target=target)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as stream:
        json.dump(payload, stream, separators=(",", ":"))
        model_path = Path(stream.name)
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
    finally:
        model_path.unlink(missing_ok=True)
    booster.set_param({"device": device})
    return booster
