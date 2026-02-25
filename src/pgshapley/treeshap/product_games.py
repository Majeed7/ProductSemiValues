from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .base import PreparedModel, TreeShapBackend
from .unified import UnifiedEnsemble, UnifiedTree


@dataclass
class _PreparedTree:
    """Precomputed rule/path metadata for a single tree."""

    max_path_features: int  # D

    # Per-leaf arrays, padded to (n_leaves, D)
    leaf_feature_ids: np.ndarray  # int32, -1 for padding
    leaf_lower: np.ndarray  # float64
    leaf_upper: np.ndarray  # float64
    leaf_inv_weight_prod: np.ndarray  # float64

    # (n_leaves, n_outputs) coefficient alpha = R_empty (already includes tree weight)
    leaf_alpha: np.ndarray


@dataclass
class PreparedProductGamesModel(PreparedModel):
    trees: List[_PreparedTree]


def _is_leaf(tree: UnifiedTree, node: int) -> bool:
    return tree.children_left[node] == -1 and tree.children_right[node] == -1


def _edge_weight(tree: UnifiedTree, parent: int, child: int) -> float:
    pw = float(tree.node_weight[parent])
    cw = float(tree.node_weight[child])
    if pw <= 0.0:
        # Degenerate; treat as uniform routing.
        return 0.5
    w = cw / pw
    # Numerical safety
    return float(np.clip(w, 0.0, 1.0))


@dataclass
class _LeafRules:
    """Root-to-leaf decision rules extracted from a tree.

    Each list has one entry per leaf.  For a leaf with *d* unique features
    on its root-to-leaf path the arrays have shape ``(d,)``.
    """

    feature_ids: List[np.ndarray]   # int32
    lower_bounds: List[np.ndarray]  # float64
    upper_bounds: List[np.ndarray]  # float64
    inv_weight_prod: List[np.ndarray]  # float64
    leaf_node_ids: List[int]


def _dfs_build_leaf_rules(tree: UnifiedTree) -> _LeafRules:
    """Enumerate root-to-leaf rules."""

    feature_ids_list: List[np.ndarray] = []
    lower_list: List[np.ndarray] = []
    upper_list: List[np.ndarray] = []
    invw_list: List[np.ndarray] = []
    leaf_nodes: List[int] = []

    # path state: dict feature -> (lower, upper, invw)
    def rec(node: int, state: dict[int, Tuple[float, float, float]]):
        if _is_leaf(tree, node):
            # Materialize in deterministic order (sorted by feature id)
            feats = sorted(state.keys())
            d = len(feats)
            feature_ids_list.append(np.asarray(feats, dtype=np.int32))
            lower_list.append(np.asarray([state[f][0] for f in feats], dtype=np.float64))
            upper_list.append(np.asarray([state[f][1] for f in feats], dtype=np.float64))
            invw_list.append(np.asarray([state[f][2] for f in feats], dtype=np.float64))
            leaf_nodes.append(node)
            return

        f = int(tree.feature[node])
        thr = float(tree.threshold[node])
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])

        # Edge weights from sample proportions
        w_left = _edge_weight(tree, node, left)
        w_right = _edge_weight(tree, node, right)

        # Left branch: x[f] <= thr
        st_left = dict(state)
        lo, hi, invw = st_left.get(f, (-np.inf, np.inf, 1.0))
        hi = min(hi, thr)
        invw = invw * (1.0 / max(w_left, 1e-300))
        st_left[f] = (lo, hi, invw)
        rec(left, st_left)

        # Right branch: x[f] > thr
        st_right = dict(state)
        lo, hi, invw = st_right.get(f, (-np.inf, np.inf, 1.0))
        lo = max(lo, thr)
        invw = invw * (1.0 / max(w_right, 1e-300))
        st_right[f] = (lo, hi, invw)
        rec(right, st_right)

    rec(0, {})
    return _LeafRules(
        feature_ids=feature_ids_list,
        lower_bounds=lower_list,
        upper_bounds=upper_list,
        inv_weight_prod=invw_list,
        leaf_node_ids=leaf_nodes,
    )


class ProductGamesTreeShapBackend(TreeShapBackend):
    """TreeSHAP backend that reduces tree explanation to product games.

    This backend follows the Linear TreeSHAP decision-rule decomposition:

        f_S(x) = sum_{leaves v} R^v_empty * prod_{j in S} q_{j,v}(x)

    and computes Shapley values for each rule as a product game.

    Parameters
    ----------
    phi_matrix_fn:
        A callable implementing ``phi_matrix(K, m_q)`` as in
        :class:`product_games.shapley.ProductGamesShapleyNumpy`.

    m_q:
        Quadrature size. If ``None``, uses ``max(1, (D+1)//2)`` per tree where
        ``D`` is the maximum number of unique features on any root-to-leaf path.

    batch_size:
        If set, evaluates leaf games in batches of at most this size.
    """

    def __init__(
        self,
        *,
        phi_matrix_fn: Callable[[np.ndarray, int], np.ndarray],
        m_q: Optional[int] = None,
        batch_size: int = 256,
    ):
        super().__init__()
        self._phi_matrix_fn = phi_matrix_fn
        self._m_q_user = m_q
        self._batch_size = int(batch_size) if batch_size is not None else 0

    def prepare(self, ensemble: UnifiedEnsemble) -> PreparedProductGamesModel:
        prepared_trees: List[_PreparedTree] = []
        expected_value = np.zeros((ensemble.n_outputs,), dtype=np.float64)

        for t_idx, (tree, t_weight) in enumerate(zip(ensemble.trees, ensemble.tree_weights)):
            rules = _dfs_build_leaf_rules(tree)
            if len(rules.leaf_node_ids) == 0:
                raise RuntimeError("Tree has no leaves?")

            # Determine per-tree D
            max_d = max(len(f) for f in rules.feature_ids) if rules.feature_ids else 0
            max_d = int(max_d)

            n_leaves = len(rules.leaf_node_ids)

            # Pad per-leaf arrays to (n_leaves, D)
            feat_ids = np.full((n_leaves, max_d), -1, dtype=np.int32)
            lower = np.full((n_leaves, max_d), -np.inf, dtype=np.float64)
            upper = np.full((n_leaves, max_d), np.inf, dtype=np.float64)
            invw = np.ones((n_leaves, max_d), dtype=np.float64)

            for i in range(n_leaves):
                d = len(rules.feature_ids[i])
                if d == 0:
                    continue
                feat_ids[i, :d] = rules.feature_ids[i]
                lower[i, :d] = rules.lower_bounds[i]
                upper[i, :d] = rules.upper_bounds[i]
                invw[i, :d] = rules.inv_weight_prod[i]

            # alpha = V * prod(w_e) = V * leaf_weight/root_weight
            root_w = float(tree.node_weight[0])
            if root_w <= 0.0:
                root_w = 1.0

            leaf_alpha = np.empty((n_leaves, ensemble.n_outputs), dtype=np.float64)
            for i, node in enumerate(rules.leaf_node_ids):
                leaf_w = float(tree.node_weight[node])
                prob = leaf_w / root_w
                leaf_val = tree.values[node].astype(np.float64, copy=False)  # (n_outputs,)
                leaf_alpha[i, :] = t_weight * prob * leaf_val

            expected_value += leaf_alpha.sum(axis=0)

            prepared_trees.append(
                _PreparedTree(
                    max_path_features=max_d,
                    leaf_feature_ids=feat_ids,
                    leaf_lower=lower,
                    leaf_upper=upper,
                    leaf_inv_weight_prod=invw,
                    leaf_alpha=leaf_alpha,
                )
            )

        out = PreparedProductGamesModel(
            ensemble=ensemble,
            expected_value=expected_value,
            trees=prepared_trees,
        )
        self.prepared = out
        return out

    def _m_q_for_tree(self, D: int) -> int:
        if self._m_q_user is not None:
            return int(self._m_q_user)
        return max(1, (int(D) + 1) // 2)

    def explain(self, X: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
        if self.prepared is None:
            raise RuntimeError("Backend is not prepared. Call prepare() first.")

        ensemble = self.prepared.ensemble
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != ensemble.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but model has {ensemble.n_features}."
            )

        n_samples = X.shape[0]
        out = np.zeros((n_samples, ensemble.n_features, ensemble.n_outputs), dtype=np.float64)

        n_trees = len(self.prepared.trees)
        if tree_limit is not None:
            n_trees = min(n_trees, int(tree_limit))

        for t in range(n_trees):
            pt = self.prepared.trees[t]
            D = pt.max_path_features
            if D == 0:
                continue
            m_q = self._m_q_for_tree(D)

            feat_ids_all = pt.leaf_feature_ids  # (n_leaves, D)
            lower_all = pt.leaf_lower
            upper_all = pt.leaf_upper
            invw_all = pt.leaf_inv_weight_prod
            alpha_all = pt.leaf_alpha  # (n_leaves, n_outputs)

            n_leaves = feat_ids_all.shape[0]
            bs = self._batch_size if self._batch_size and self._batch_size > 0 else n_leaves

            for s in range(n_samples):
                x = X[s]
                phi_s = out[s]  # (n_features, n_outputs)

                for start in range(0, n_leaves, bs):
                    end = min(n_leaves, start + bs)

                    feat_ids = feat_ids_all[start:end]  # (b, D)
                    lower = lower_all[start:end]
                    upper = upper_all[start:end]
                    invw = invw_all[start:end]
                    alpha = alpha_all[start:end]  # (b, n_outputs)
                    b = end - start

                    # Gather feature values for each (leaf, slot)
                    gather_ids = np.where(feat_ids >= 0, feat_ids, 0)
                    x_vals = x[gather_ids]  # (b, D)

                    satisfied = (x_vals > lower) & (x_vals <= upper)
                    # For padding entries, invw==1, bounds==(-inf, inf), so satisfied is True.

                    q = invw * satisfied.astype(np.float64)  # (b, D)
                    # But for padding, q should be 1 (not invw * True == 1) good.

                    # However, for features not present (feat_id == -1), we want q=1.
                    # Our construction ensures q==1 there because invw==1 and bounds are wide.

                    K = (q - 1.0).T  # (D, b)
                    Phi = self._phi_matrix_fn(K, m_q)  # (D, b)

                    # Multiply by per-leaf alpha (vector)
                    contrib = Phi[:, :, None] * alpha[None, :, :]  # (D, b, n_outputs)

                    # Scatter-add to feature dimension
                    for k in range(D):
                        f_idx = feat_ids[:, k]
                        mask = f_idx >= 0
                        if not np.any(mask):
                            continue
                        np.add.at(phi_s, f_idx[mask], contrib[k, mask, :])

        return out
