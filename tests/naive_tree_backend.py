from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pgshapley.treeshap.unified import UnifiedEnsemble, UnifiedTree


def _is_leaf(tree: UnifiedTree, node: int) -> bool:
    return tree.children_left[node] == -1 and tree.children_right[node] == -1


def _edge_weight(tree: UnifiedTree, parent: int, child: int) -> float:
    pw = float(tree.node_weight[parent])
    cw = float(tree.node_weight[child])
    if pw <= 0.0:
        return 0.5
    return float(cw / pw)


def _predict_tree_with_subset(tree: UnifiedTree, x: np.ndarray, S_mask: np.ndarray) -> np.ndarray:
    """Tree prediction when features not in S are treated as missing and routed by edge weights."""

    def rec(node: int) -> np.ndarray:
        if _is_leaf(tree, node):
            return tree.values[node]
        f = int(tree.feature[node])
        thr = float(tree.threshold[node])
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])

        if S_mask[f]:
            # feature is present/known
            if x[f] <= thr:
                return rec(left)
            else:
                return rec(right)
        else:
            # feature missing: weighted sum of both branches
            wl = _edge_weight(tree, node, left)
            wr = _edge_weight(tree, node, right)
            return wl * rec(left) + wr * rec(right)

    return rec(0)


def predict_with_subset(ensemble: UnifiedEnsemble, x: np.ndarray, S_mask: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
    n_trees = len(ensemble.trees)
    if tree_limit is not None:
        n_trees = min(n_trees, int(tree_limit))
    out = np.zeros((ensemble.n_outputs,), dtype=np.float64)
    for tree, w in zip(ensemble.trees[:n_trees], ensemble.tree_weights[:n_trees]):
        out += w * _predict_tree_with_subset(tree, x, S_mask)
    return out


def naive_tree_shap_values(ensemble: UnifiedEnsemble, X: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
    """Naive (exponential) TreeSHAP for path-dependent missingness."""

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_samples, n_features = X.shape
    n_outputs = ensemble.n_outputs

    out = np.zeros((n_samples, n_features, n_outputs), dtype=np.float64)

    # factorial weights
    fact = [math.factorial(i) for i in range(n_features + 1)]
    denom = fact[n_features]

    # Precompute all subset masks and their sizes (for re-use per feature)
    # Only feasible for small n_features.
    subset_masks = []
    subset_sizes = []
    for r in range(n_features + 1):
        for comb in itertools.combinations(range(n_features), r):
            mask = np.zeros((n_features,), dtype=bool)
            mask[list(comb)] = True
            subset_masks.append(mask)
            subset_sizes.append(r)

    subset_masks = np.stack(subset_masks, axis=0)
    subset_sizes = np.asarray(subset_sizes, dtype=np.int32)

    for s in range(n_samples):
        x = X[s]
        # Evaluate v(S) for all subsets once
        v = np.zeros((subset_masks.shape[0], n_outputs), dtype=np.float64)
        for idx, mask in enumerate(subset_masks):
            v[idx] = predict_with_subset(ensemble, x, mask, tree_limit=tree_limit)

        # Map mask -> index to look up v(S union {i}) quickly
        # Since n_features is tiny in tests, brute map is fine.
        mask_to_idx = {tuple(mask.tolist()): idx for idx, mask in enumerate(subset_masks)}

        for i in range(n_features):
            contrib = np.zeros((n_outputs,), dtype=np.float64)
            for idx, mask in enumerate(subset_masks):
                if mask[i]:
                    continue
                r = int(subset_sizes[idx])
                w = fact[r] * fact[n_features - r - 1] / denom
                mask_i = mask.copy()
                mask_i[i] = True
                idx_i = mask_to_idx[tuple(mask_i.tolist())]
                contrib += w * (v[idx_i] - v[idx])
            out[s, i] = contrib

    return out
