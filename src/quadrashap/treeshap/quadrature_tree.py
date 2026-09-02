"""O(m_q * L) TreeSHAP solver based on Gauss-Legendre quadrature.

This backend implements the two-pass algorithm described in ``paper.tex``
(section "Tree ensemble value functions and induced product structure",
algorithms "Optimal algorithm: first pass" and "Optimal algorithm: second pass").

For each sample, the method walks the tree once to populate per-node
accumulators ``G_u(x_r)`` and per-edge increments ``Delta F_e(x_r)``, and then
performs a bottom-up aggregation that yields all feature Shapley values in
``O(m_q * L)`` time per tree, where ``L`` is the number of leaves and ``m_q``
is the Gauss-Legendre quadrature order.

For exact Shapley values we pick ``m_q = ceil(D/2)``, where ``D`` is the number
of distinct features on the longest root-to-leaf path; smaller choices trade
accuracy for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import PreparedModel, TreeShapBackend
from .unified import UnifiedEnsemble, UnifiedTree
from quadrashap._cpp_ext import HAS_CPP_EXT


@dataclass
class _PreparedTreeQT:
    """Per-tree precomputed metadata for the quadrature tree backend."""

    tree: UnifiedTree
    tree_weight: float
    max_path_features: int  # D: longest distinct-feature path length

    # For each non-root node: weight of the edge from its parent.
    edge_weight: np.ndarray     # (n_nodes,) float64 (NaN for root)
    inv_edge_weight: np.ndarray  # (n_nodes,) float64 (NaN for root)

    # Post-order traversal (children before parent) for the bottom-up pass.
    postorder: np.ndarray  # (n_nodes,) int32

    # Gauss-Legendre quadrature on [0, 1].
    quad_x: np.ndarray  # (m_q,)
    quad_w: np.ndarray  # (m_q,)

    # Lazily-built C++ PreparedTreeQT (None until first C++ explain call).
    cpp_data: object = None


@dataclass
class PreparedQuadratureTreeModel(PreparedModel):
    trees: List[_PreparedTreeQT]


def _edge_weight(tree: UnifiedTree, parent: int, child: int) -> float:
    pw = float(tree.node_weight[parent])
    cw = float(tree.node_weight[child])
    if pw <= 0.0:
        return 0.5
    w = cw / pw
    return float(np.clip(w, 1e-300, 1.0))


class QuadratureTreeShapBackend(TreeShapBackend):
    """Two-pass O(m_q * L) TreeSHAP using Gauss-Legendre quadrature.

    Parameters
    ----------
    m_q:
        Quadrature order. If ``None`` (default), uses ``max(1, (D+1)//2)``
        per tree, where ``D`` is the longest distinct-feature root-to-leaf
        path. This gives Shapley values that are exact up to floating-point
        precision; smaller ``m_q`` trades accuracy for speed.
    use_cpp:
        If ``True``, dispatch to the native C++ kernel
        (``explain_trees_quadrature``) when the extension is available. If
        ``False``, force the pure-Python/numpy DFS path even when the C++
        kernel is built. Defaults to ``True``. The two paths are kept side
        by side so they can be benchmarked head-to-head.
    """

    def __init__(
        self,
        *,
        m_q: Optional[int] = None,
        use_cpp: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self._m_q_user = m_q
        self._use_cpp = bool(use_cpp)
        self._device = str(device).lower()
        self._cuda_solver = None
        if self._device not in ("cpu", "cuda"):
            raise ValueError("device must be either 'cpu' or 'cuda'")
        if self._device == "cpu" and self._use_cpp and not HAS_CPP_EXT:
            import warnings
            warnings.warn(
                "quadrashap: use_cpp=True requested but the C++ extension "
                "(_core) failed to load; falling back to the ~50-100x slower "
                "NumPy/Python DFS path. Reinstall with a C++ toolchain and "
                "pybind11 available (e.g. `uv pip install pybind11>=2.12 && "
                "uv pip install --no-build-isolation --reinstall --no-deps "
                "-e .`) to enable the fast kernel.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _m_q_for_tree(self, D: int) -> int:
        if self._m_q_user is not None:
            return int(self._m_q_user)
        return max(1, (int(D) + 1) // 2)

    def prepare(self, ensemble: UnifiedEnsemble) -> PreparedQuadratureTreeModel:
        prepared_trees: List[_PreparedTreeQT] = []
        expected_value = np.zeros((ensemble.n_outputs,), dtype=np.float64)

        for tree, t_weight in zip(ensemble.trees, ensemble.tree_weights):
            n_nodes = int(tree.children_left.shape[0])

            edge_weight = np.full(n_nodes, np.nan, dtype=np.float64)
            inv_edge_weight = np.full(n_nodes, np.nan, dtype=np.float64)

            postorder: List[int] = []

            max_d = 0
            path_feats: set[int] = set()

            def rec(u: int) -> None:
                nonlocal max_d
                l = int(tree.children_left[u])
                r = int(tree.children_right[u])
                if l == -1:
                    if len(path_feats) > max_d:
                        max_d = len(path_feats)
                    postorder.append(u)
                    return

                f = int(tree.feature[u])
                for child in (l, r):
                    w_e = _edge_weight(tree, u, child)
                    edge_weight[child] = w_e
                    inv_edge_weight[child] = 1.0 / w_e
                added = f not in path_feats
                if added:
                    path_feats.add(f)
                rec(l)
                rec(r)
                if added:
                    path_feats.remove(f)
                postorder.append(u)

            rec(0)

            m_q = self._m_q_for_tree(max_d)
            leg_x, leg_w = np.polynomial.legendre.leggauss(m_q)
            qx = (0.5 * (leg_x + 1.0)).astype(np.float64)
            qw = (0.5 * leg_w).astype(np.float64)

            # Expected value contribution for this tree.
            root_w = float(tree.node_weight[0])
            if root_w <= 0.0:
                root_w = 1.0
            for u in postorder:
                if int(tree.children_left[u]) == -1:
                    leaf_w = float(tree.node_weight[u])
                    prob = leaf_w / root_w
                    expected_value += t_weight * prob * tree.values[u].astype(np.float64, copy=False)

            prepared_trees.append(
                _PreparedTreeQT(
                    tree=tree,
                    tree_weight=float(t_weight),
                    max_path_features=int(max_d),
                    edge_weight=edge_weight,
                    inv_edge_weight=inv_edge_weight,
                    postorder=np.asarray(postorder, dtype=np.int32),
                    quad_x=qx,
                    quad_w=qw,
                )
            )

        out = PreparedQuadratureTreeModel(
            ensemble=ensemble,
            expected_value=expected_value,
            trees=prepared_trees,
        )
        self.prepared = out
        if self._device == "cuda":
            from .cuda import CudaQuadratureTreeSolver

            self._cuda_solver = CudaQuadratureTreeSolver(out)
        return out

    def explain(self, X: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
        if self.prepared is None:
            raise RuntimeError("Backend is not prepared. Call prepare() first.")

        ensemble = self.prepared.ensemble
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != ensemble.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but model has {ensemble.n_features}."
            )

        n_trees_total = len(self.prepared.trees)
        n_trees = n_trees_total if tree_limit is None else min(n_trees_total, int(tree_limit))

        if self._device == "cuda":
            if n_trees != n_trees_total:
                raise NotImplementedError(
                    "tree_limit is not yet supported by the CUDA backend"
                )
            return self._cuda_solver.explain(X)

        if self._use_cpp and HAS_CPP_EXT:
            return self._explain_cpp(X, n_trees)

        n_samples = X.shape[0]
        out = np.zeros((n_samples, ensemble.n_features, ensemble.n_outputs), dtype=np.float64)

        for t in range(n_trees):
            pt = self.prepared.trees[t]
            phi = self._explain_tree_batch(pt, X)
            if pt.tree_weight == 1.0:
                out += phi
            else:
                out += pt.tree_weight * phi

        return out

    def _build_cpp_data(self, pt: _PreparedTreeQT):
        """Build the per-tree pybind11 PreparedTreeQT struct, lazily."""
        from quadrashap._cpp_ext import PreparedTreeQT as CppPreparedTreeQT

        tree = pt.tree

        # The recursion fills NaN for the root edge weight; the C++ kernel
        # never reads that slot, but pass a finite value just in case to be
        # robust against -ffast-math weirdness.
        ew = pt.edge_weight.copy()
        if ew.size > 0 and not np.isfinite(ew[0]):
            ew[0] = 1.0

        td = CppPreparedTreeQT()
        td.children_left = np.ascontiguousarray(tree.children_left, dtype=np.int32)
        td.children_right = np.ascontiguousarray(tree.children_right, dtype=np.int32)
        td.feature = np.ascontiguousarray(tree.feature, dtype=np.int32)
        td.threshold = np.ascontiguousarray(tree.threshold, dtype=np.float64)
        td.edge_weight = np.ascontiguousarray(ew, dtype=np.float64)
        td.values = np.ascontiguousarray(tree.values, dtype=np.float64)
        td.postorder = np.ascontiguousarray(pt.postorder, dtype=np.int32)
        td.quad_x = np.ascontiguousarray(pt.quad_x, dtype=np.float64)
        td.quad_w = np.ascontiguousarray(pt.quad_w, dtype=np.float64)
        td.n_features = int(tree.n_features)
        td.n_outputs = int(tree.n_outputs)
        td.tree_weight = float(pt.tree_weight)
        return td

    def _explain_cpp(self, X: np.ndarray, n_trees: int) -> np.ndarray:
        from quadrashap._cpp_ext import explain_trees_quadrature

        ensemble = self.prepared.ensemble
        n_samples = int(X.shape[0])
        out = np.zeros(
            (n_samples, ensemble.n_features, ensemble.n_outputs),
            dtype=np.float64,
        )

        cpp_trees = []
        for t in range(n_trees):
            pt = self.prepared.trees[t]
            if pt.cpp_data is None:
                pt.cpp_data = self._build_cpp_data(pt)
            cpp_trees.append(pt.cpp_data)

        explain_trees_quadrature(
            np.ascontiguousarray(X, dtype=np.float64),
            out,
            cpp_trees,
            n_trees,
        )
        return out

    def _explain_tree_batch(self, pt: _PreparedTreeQT, X: np.ndarray) -> np.ndarray:
        """Compute Shapley contributions for a full batch of samples on one tree.

        Vectorized across samples: every numpy op processes all ``n_samples``
        in one call, amortizing Python/numpy dispatch overhead on the hot DFS
        path.

        Parameters
        ----------
        pt:
            Precomputed per-tree data.
        X:
            ``(n_samples, n_features)`` float64 array.

        Returns
        -------
        phi:
            ``(n_samples, n_features, n_outputs)`` array, not yet scaled by
            ``pt.tree_weight``.
        """

        tree = pt.tree
        n_nodes = int(tree.children_left.shape[0])
        n_features = int(tree.n_features)
        n_outputs = int(tree.n_outputs)
        qx = pt.quad_x                 # (m_q,)
        one_m_qx = 1.0 - qx            # (m_q,)
        qw = pt.quad_w                 # (m_q,)
        m_q = int(qx.shape[0])
        n_samples = int(X.shape[0])

        # Precomputed tree arrays (used heavily in the hot loop).
        children_left = tree.children_left
        children_right = tree.children_right
        feat_arr = tree.feature
        thr_arr = tree.threshold

        # First-pass accumulators.
        # G_node[u, s, r]: G value at node u, sample s, quadrature node r.
        G_node = np.empty((n_nodes, n_samples, m_q), dtype=np.float64)
        # Delta_F[u, s, r]: ΔF for the edge into node u (root slot unused).
        Delta_F = np.zeros((n_nodes, n_samples, m_q), dtype=np.float64)

        # Persistent DFS state, all vectorized across samples.
        # q defaults to 1 off-path, so a_r(q) = 1 and F = 0.
        current_q = np.ones((n_samples, n_features), dtype=np.float64)
        current_F = np.zeros((n_samples, n_features, m_q), dtype=np.float64)

        # Current G vector as we descend the tree.
        G_cur = np.ones((n_samples, m_q), dtype=np.float64)

        def dfs1(u: int) -> None:
            nonlocal G_cur
            # Snapshot G at this node before descending.
            G_node[u] = G_cur

            l = int(children_left[u])
            if l == -1:
                return
            r = int(children_right[u])

            f = int(feat_arr[u])
            thr = float(thr_arr[u])

            q_old = current_q[:, f].copy()         # (n_samples,)
            F_old = current_F[:, f, :].copy()      # (n_samples, m_q)
            a_old = one_m_qx + qx * q_old[:, None] # (n_samples, m_q)

            # Route left/right branches on feature f for every sample.
            x_f = X[:, f]
            sat_left = (x_f <= thr).astype(np.float64)   # (n_samples,)
            sat_right = 1.0 - sat_left                   # (n_samples,)

            for child, sat in ((l, sat_left), (r, sat_right)):
                w_e = float(pt.edge_weight[child])
                inv_w = float(pt.inv_edge_weight[child])

                q_new = q_old * (inv_w * sat)             # (n_samples,)
                a_new = one_m_qx + qx * q_new[:, None]    # (n_samples, m_q)

                F_new = (q_new[:, None] - 1.0) / a_new    # (n_samples, m_q)
                Delta_F[child] = F_new - F_old

                G_saved = G_cur
                G_cur = G_cur * (w_e * (a_new / a_old))
                current_q[:, f] = q_new
                current_F[:, f, :] = F_new

                dfs1(child)

                G_cur = G_saved

            # Restore state for feature f at the parent level.
            current_q[:, f] = q_old
            current_F[:, f, :] = F_old

        dfs1(0)

        # Second pass: bottom-up accumulation of H and Shapley values.
        # H[u, s, r, k] = sum over leaves v in the subtree rooted at u of
        #                 V_v[k] * G_node[v, s, r].
        # For each edge (u -> v) splitting on feature f:
        #     phi[:, f, :] += sum_r w_r * H[v, :, r, :] * Delta_F[v, :, r]
        H = np.zeros((n_nodes, n_samples, m_q, n_outputs), dtype=np.float64)
        phi = np.zeros((n_samples, n_features, n_outputs), dtype=np.float64)

        # qw broadcast shape for the edge contraction: (1, m_q).
        qw_row = qw[None, :]

        for u_i in pt.postorder:
            u = int(u_i)
            l = int(children_left[u])
            if l == -1:
                V_u = tree.values[u].astype(np.float64, copy=False)  # (n_outputs,)
                # (n_samples, m_q, 1) * (1, 1, n_outputs) -> (n_samples, m_q, n_outputs)
                H[u] = G_node[u][:, :, None] * V_u[None, None, :]
                continue

            r = int(children_right[u])
            H[u] = H[l] + H[r]

            f = int(feat_arr[u])
            for child in (l, r):
                # H[child]: (n_samples, m_q, n_outputs)
                # Delta_F[child]: (n_samples, m_q)
                # sum_r w_r * H[child, s, r, :] * Delta_F[child, s, r]
                weight = qw_row * Delta_F[child]               # (n_samples, m_q)
                phi[:, f, :] += (H[child] * weight[:, :, None]).sum(axis=1)

        return phi
