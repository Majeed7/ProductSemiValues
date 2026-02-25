from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from tree_explainer import (
    BackendPrecompute,
    TreeExplainerBackend,
    TreeEnsemble,
    Tree,
)

def _tree_predict_one(tree: Tree, x: np.ndarray) -> np.ndarray:
    node = 0
    while True:
        feat = int(tree.feature[node])
        if feat < 0:  # leaf
            return tree.value[node]

        v = x[feat]
        thr = tree.threshold[node]

        # Missing routing
        if np.isnan(v):
            nxt = int(tree.children_default[node])
        else:
            nxt = int(tree.children_left[node] if v <= thr else tree.children_right[node])

        assert nxt >= 0
        node = nxt


def _ensemble_predict_one(ens: TreeEnsemble, x: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
    n_trees_total = len(ens.trees)
    n_trees = n_trees_total if tree_limit is None else min(int(tree_limit), n_trees_total)

    out = ens.base_offset.astype(np.float64, copy=True)
    for t_idx in range(n_trees):
        t = ens.trees[t_idx]
        w = float(ens.tree_weights[t_idx])
        out += w * _tree_predict_one(t, x)
    return out


def _ensemble_predict_batch(ens: TreeEnsemble, X: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
    X = np.asarray(X)
    n = X.shape[0]
    k = ens.n_outputs
    out = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        out[i] = _ensemble_predict_one(ens, X[i], tree_limit=tree_limit)

    if k == 1:
        return out[:, 0]
    return out


@dataclass
class NaivePrecomputePayload:
    background: np.ndarray            # float32/float64, shape (Nbg, n_features)
    m: int                            # n_features
    v_empty: np.ndarray               # shape (n_outputs,) = E_z[f(z)]
    # weights_by_size[k] = k! * (m-k-1)! / m!
    weights_by_size: np.ndarray       # shape (m,)
    # cached list of masks by size for faster loops
    masks_by_size: Tuple[np.ndarray, ...]  # tuple length m+1, each array of masks (int)


class NaiveShapleyBackend(TreeExplainerBackend):
    name = "naive"

    def precompute(
        self,
        ensemble: TreeEnsemble,
        background: Optional[np.ndarray] = None,
        feature_perturbation: str = "auto",
        model_output: str = "raw",
    ) -> BackendPrecompute:
        m = int(ensemble.n_features)

        if background is None:
            # Fallback: deterministic but not a real interventional baseline.
            background = np.zeros((1, m), dtype=np.float32)
        else:
            background = np.asarray(background)
            if background.ndim != 2 or background.shape[1] != m:
                raise ValueError(f"background must be 2D with shape (N, {m}), got {background.shape}")
            # Keep as float32 for sklearn-like parity
            background = background.astype(np.float32, copy=False)

        # v(empty) = E[f(z)]
        preds_bg = _ensemble_predict_batch(ensemble, background)
        if ensemble.n_outputs == 1:
            v_empty = np.array([float(np.mean(preds_bg))], dtype=np.float64)
        else:
            v_empty = np.mean(preds_bg, axis=0).astype(np.float64)

        # Precompute Shapley combinatorial weights by subset size
        # weight for subsets S with |S|=k when adding feature i:
        #   k!(m-k-1)! / m!
        mfact = math.factorial(m) if m > 0 else 1
        weights_by_size = np.zeros((max(m, 1),), dtype=np.float64)
        if m > 0:
            for k in range(m):
                weights_by_size[k] = (math.factorial(k) * math.factorial(m - k - 1)) / mfact

        # Group masks by size (optional convenience)
        n_masks = 1 << m if m < 63 else None
        if n_masks is None:
            raise ValueError("m too large for bitmask enumeration in this naive backend.")
        masks_by_size = []
        for k in range(m + 1):
            masks = [mask for mask in range(n_masks) if mask.bit_count() == k]
            masks_by_size.append(np.asarray(masks, dtype=np.int64))
        payload = NaivePrecomputePayload(
            background=background,
            m=m,
            v_empty=v_empty,
            weights_by_size=weights_by_size,
            masks_by_size=tuple(masks_by_size),
        )
        return BackendPrecompute(payload=payload, backend_name=self.name, backend_version="1.0")

    def predict(
        self,
        X: np.ndarray,
        ensemble: TreeEnsemble,
        pre: Optional[BackendPrecompute] = None,
        *,
        tree_limit: Optional[int] = None,
    ) -> np.ndarray:
        X = np.asarray(X).astype(np.float32, copy=False)
        return _ensemble_predict_batch(ensemble, X, tree_limit=tree_limit)

    def explain(
        self,
        X: np.ndarray,
        ensemble: TreeEnsemble,
        pre: BackendPrecompute,
        *,
        interactions: bool = False,
        tree_limit: Optional[int] = None,
        check_additivity: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if interactions:
            raise NotImplementedError("Naive backend: interactions=True not implemented.")

        X = np.asarray(X).astype(np.float32, copy=False)
        if X.ndim != 2 or X.shape[1] != ensemble.n_features:
            raise ValueError(f"X must have shape (n, {ensemble.n_features}), got {X.shape}")

        payload: NaivePrecomputePayload = pre.payload
        bg = payload.background
        m = payload.m
        n = X.shape[0]
        k_out = ensemble.n_outputs

        # Output containers
        if k_out == 1:
            phi = np.zeros((n, m), dtype=np.float64)
            base_values = np.broadcast_to(payload.v_empty[0], (n,)).astype(np.float64, copy=False)
        else:
            phi = np.zeros((n, m, k_out), dtype=np.float64)
            base_values = np.broadcast_to(payload.v_empty.reshape(1, k_out), (n, k_out)).astype(np.float64, copy=False)

        # Enumerate subsets and compute v(S) for each sample
        # Cache v(S) for a single sample: shape (2^m, k_out)
        n_masks = 1 << m

        # Preallocate a mixing buffer for background edits to reduce allocations
        # We'll copy bg each time anyway (because we overwrite columns), but keep it explicit.
        for row_idx in range(n):
            x = X[row_idx]

            # v_cache[mask] = v_x(mask)
            v_cache = np.zeros((n_masks, k_out), dtype=np.float64)

            # Compute v(mask) for all masks
            # v(mask) = mean_{z in bg} f(x_S, z_{~S})
            for mask in range(n_masks):
                if mask == 0:
                    v_cache[mask] = payload.v_empty
                    continue

                Xm = bg.copy()  # (Nbg, m)

                # Set columns in S to x values
                # naive loop over bits
                mm = mask
                while mm:
                    bit = (mm & -mm)
                    j = (bit.bit_length() - 1)
                    Xm[:, j] = x[j]
                    mm ^= bit

                preds = _ensemble_predict_batch(ensemble, Xm, tree_limit=tree_limit)

                if k_out == 1:
                    v_cache[mask, 0] = float(np.mean(preds))
                else:
                    v_cache[mask] = np.mean(preds, axis=0)

            # Shapley aggregation:
            # phi_i = sum_{S not containing i} w(|S|) * (v(S ∪ {i}) - v(S))
            for i in range(m):
                acc = np.zeros((k_out,), dtype=np.float64)
                bit_i = 1 << i

                # Iterate subsets by size to use precomputed weights
                for size in range(m):
                    w = payload.weights_by_size[size]
                    if w == 0.0:
                        continue
                    masks = payload.masks_by_size[size]
                    # masks here include those that may contain i; skip those quickly
                    for mask in masks:
                        if mask & bit_i:
                            continue
                        vS = v_cache[mask]
                        vSi = v_cache[mask | bit_i]
                        acc += w * (vSi - vS)

                if k_out == 1:
                    phi[row_idx, i] = acc[0]
                else:
                    phi[row_idx, i, :] = acc

            if check_additivity:
                # Check: base + sum(phi) == f(x)
                pred_x = _ensemble_predict_one(ensemble, x, tree_limit=tree_limit)
                if k_out == 1:
                    lhs = payload.v_empty[0] + float(np.sum(phi[row_idx, :]))
                    rhs = float(pred_x[0])
                    if not np.isclose(lhs, rhs, rtol=1e-6, atol=1e-6):
                        raise ValueError(
                            f"Additivity check failed at row {row_idx}: "
                            f"base+sum(phi)={lhs}, pred={rhs}"
                        )
                else:
                    lhs = payload.v_empty + np.sum(phi[row_idx, :, :], axis=0)
                    rhs = pred_x
                    if not np.allclose(lhs, rhs, rtol=1e-6, atol=1e-6):
                        raise ValueError(
                            f"Additivity check failed at row {row_idx}: "
                            f"base+sum(phi)={lhs}, pred={rhs}"
                        )

        return phi, base_values
