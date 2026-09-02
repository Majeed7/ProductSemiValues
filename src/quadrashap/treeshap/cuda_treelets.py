"""Host planning for the checkpointed CUDA treelet solver.

The planner partitions the internal nodes of every tree into connected
components.  A component is small enough for one warp to reconstruct all of
its sample-dependent state in shared memory.  Only component-root K/G values
are kept in global memory between bottom-up and top-down launches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


TREELET_SIZE = 15
TREELET_MAX_PORTALS = 10
TREELET_MAX_Q = 16


class CompactHostPlan(Protocol):
    """The subset of ``cuda._HostPlan`` consumed by this module."""

    n_internal: int
    m_q: int
    internal_roots: np.ndarray
    compact_left_ref: np.ndarray
    compact_right_ref: np.ndarray
    compact_feature: np.ndarray
    compact_m_q: np.ndarray
    compact_left_weight: np.ndarray
    compact_right_weight: np.ndarray
    compact_old_lower: np.ndarray
    compact_old_upper: np.ndarray
    compact_threshold: np.ndarray
    leaf_value: np.ndarray


@dataclass(frozen=True)
class TreeletLevel:
    """One dependency level in the component forest."""

    treelets: np.ndarray
    used_features: np.ndarray
    feature_offsets: np.ndarray
    n_parents: int


@dataclass(frozen=True)
class TreeletPlan:
    """Packed arrays consumed by the fixed-size CUDA treelet kernels."""

    supported: bool
    reason: str | None
    treelet_size: int
    max_portals: int
    headers: np.ndarray
    q_offsets: np.ndarray
    int_record: np.ndarray
    float_record: np.ndarray
    leaf_record: np.ndarray
    portals: np.ndarray
    roots: np.ndarray
    levels: tuple[TreeletLevel, ...]
    n_treelets: int
    total_q_states: int
    max_level_parents: int
    node_to_treelet: np.ndarray
    treelet_nodes: tuple[np.ndarray, ...]

    @property
    def workspace_values_per_sample(self) -> int:
        return self.total_q_states + self.max_level_parents


def _unsupported(
    h: CompactHostPlan,
    reason: str,
    treelet_size: int,
    max_portals: int,
) -> TreeletPlan:
    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    return TreeletPlan(
        supported=False,
        reason=reason,
        treelet_size=treelet_size,
        max_portals=max_portals,
        headers=empty_i.reshape(0, 3),
        q_offsets=empty_i,
        int_record=empty_i.reshape(0, treelet_size, 5),
        float_record=empty_f.reshape(0, treelet_size, 5),
        leaf_record=empty_f.reshape(0, treelet_size, 2),
        portals=empty_i.reshape(0, max_portals),
        roots=empty_i,
        levels=(),
        n_treelets=0,
        total_q_states=0,
        max_level_parents=0,
        node_to_treelet=np.full(h.n_internal, -1, dtype=np.int32),
        treelet_nodes=(),
    )


def build_treelet_plan(
    h: CompactHostPlan,
    *,
    treelet_size: int = TREELET_SIZE,
    max_portals: int = TREELET_MAX_PORTALS,
) -> TreeletPlan:
    """Build a bottom-up-greedy connected-component plan.

    Each recursive subtree returns one open component.  A parent merges both
    open children when they fit.  Otherwise it closes the larger child
    component until the remainder fits.  Thus every non-root component has at
    least half of ``TREELET_SIZE`` nodes, apart from degenerate tiny trees.
    """

    if h.n_internal == 0:
        return _unsupported(
            h, "the ensemble has no internal nodes", treelet_size, max_portals
        )
    if h.m_q > TREELET_MAX_Q:
        return _unsupported(
            h,
            f"quadrature order {h.m_q} exceeds treelet limit "
            f"{TREELET_MAX_Q}",
            treelet_size,
            max_portals,
        )

    left = np.asarray(h.compact_left_ref, dtype=np.int32)
    right = np.asarray(h.compact_right_ref, dtype=np.int32)
    closed: list[list[int]] = []

    def open_component(parent: int) -> list[int]:
        parts = [
            open_component(int(ref)) if ref >= 0 else []
            for ref in (left[parent], right[parent])
        ]
        size = 1 + len(parts[0]) + len(parts[1])
        if size > treelet_size:
            for side in sorted(
                range(2), key=lambda i: len(parts[i]), reverse=True
            ):
                if size <= treelet_size:
                    break
                if parts[side]:
                    closed.append(parts[side])
                    size -= len(parts[side])
                    parts[side] = []
        return [parent, *parts[0], *parts[1]]

    try:
        for root in np.asarray(h.internal_roots, dtype=np.int32):
            closed.append(open_component(int(root)))
    except RecursionError:
        return _unsupported(
            h,
            "tree depth exceeds the host treelet planner recursion limit",
            treelet_size,
            max_portals,
        )

    if sum(map(len, closed)) != h.n_internal:
        return _unsupported(
            h,
            "treelet partition did not cover every node",
            treelet_size,
            max_portals,
        )

    n_treelets = len(closed)
    node_to_treelet = np.full(h.n_internal, -1, dtype=np.int32)
    for tile, nodes in enumerate(closed):
        if len(nodes) > treelet_size:
            return _unsupported(
                h,
                "treelet partition exceeded fixed size",
                treelet_size,
                max_portals,
            )
        if np.any(node_to_treelet[nodes] >= 0):
            return _unsupported(
                h,
                "an internal node occurs in two treelets",
                treelet_size,
                max_portals,
            )
        node_to_treelet[nodes] = tile
    if np.any(node_to_treelet < 0):
        return _unsupported(
            h,
            "an internal node is missing from treelets",
            treelet_size,
            max_portals,
        )

    headers = np.zeros((n_treelets, 3), dtype=np.int32)
    int_record = np.zeros(
        (n_treelets, treelet_size, 5), dtype=np.int32
    )
    float_record = np.zeros(
        (n_treelets, treelet_size, 5), dtype=np.float64
    )
    leaf_record = np.zeros(
        (n_treelets, treelet_size, 2), dtype=np.float64
    )
    portals = np.full(
        (n_treelets, max_portals), -1, dtype=np.int32
    )
    treelet_parent = np.full(n_treelets, -1, dtype=np.int32)
    treelet_children: list[list[int]] = [[] for _ in range(n_treelets)]

    leaf_value = np.asarray(h.leaf_value, dtype=np.float64)
    if leaf_value.ndim == 2:
        leaf_value = leaf_value[:, 0]
    elif leaf_value.ndim != 1:
        return _unsupported(
            h,
            "treelet path requires scalar leaf values",
            treelet_size,
            max_portals,
        )

    for tile, nodes in enumerate(closed):
        local_of = {node: local for local, node in enumerate(nodes)}
        portal_of: dict[int, int] = {}
        q_count = int(h.compact_m_q[nodes[0]])
        headers[tile, :2] = (len(nodes), q_count)
        for local, parent in enumerate(nodes):
            if int(h.compact_m_q[parent]) != q_count:
                return _unsupported(
                    h,
                    "a treelet crosses quadrature orders",
                    treelet_size,
                    max_portals,
                )
            child_refs: list[int] = []
            for child in (int(left[parent]), int(right[parent])):
                if child < 0:
                    child_refs.append(child)
                elif child in local_of:
                    child_refs.append(local_of[child])
                else:
                    child_tile = int(node_to_treelet[child])
                    portal = portal_of.get(child_tile)
                    if portal is None:
                        portal = len(portal_of)
                        if portal >= max_portals:
                            return _unsupported(
                                h,
                                "a treelet needs more than "
                                f"{max_portals} portals",
                                treelet_size,
                                max_portals,
                            )
                        portal_of[child_tile] = portal
                        portals[tile, portal] = child_tile
                        treelet_children[tile].append(child_tile)
                        if (
                            treelet_parent[child_tile] >= 0
                            and treelet_parent[child_tile] != tile
                        ):
                            return _unsupported(
                                h,
                                "a treelet has multiple parents",
                                treelet_size,
                                max_portals,
                            )
                        treelet_parent[child_tile] = tile
                    child_refs.append(treelet_size + portal)

            int_record[tile, local, :4] = (
                child_refs[0],
                child_refs[1],
                parent,
                int(h.compact_feature[parent]),
            )
            float_record[tile, local] = (
                float(h.compact_left_weight[parent]),
                float(h.compact_right_weight[parent]),
                float(h.compact_old_lower[parent]),
                float(h.compact_old_upper[parent]),
                float(h.compact_threshold[parent]),
            )
            if left[parent] < 0:
                leaf_record[tile, local, 0] = leaf_value[-left[parent] - 1]
            if right[parent] < 0:
                leaf_record[tile, local, 1] = leaf_value[-right[parent] - 1]
        headers[tile, 2] = len(portal_of)

    roots = np.flatnonzero(treelet_parent < 0).astype(np.int32)
    expected_roots = node_to_treelet[
        np.asarray(h.internal_roots, dtype=np.int32)
    ]
    if set(roots.tolist()) != set(expected_roots.tolist()):
        return _unsupported(
            h,
            "component roots do not match tree roots",
            treelet_size,
            max_portals,
        )

    depth = np.full(n_treelets, -1, dtype=np.int32)
    frontier = roots.tolist()
    current_depth = 0
    while frontier:
        next_frontier: list[int] = []
        for tile in frontier:
            if depth[tile] >= 0:
                return _unsupported(
                    h,
                    "cycle in treelet dependency graph",
                    treelet_size,
                    max_portals,
                )
            depth[tile] = current_depth
            next_frontier.extend(treelet_children[tile])
        frontier = next_frontier
        current_depth += 1
    if np.any(depth < 0):
        return _unsupported(
            h,
            "disconnected treelet dependency graph",
            treelet_size,
            max_portals,
        )

    levels: list[TreeletLevel] = []
    max_level_parents = 0
    compact_feature = np.asarray(h.compact_feature, dtype=np.int32)
    for level_depth in range(int(depth.max()) + 1):
        level_tiles = np.flatnonzero(depth == level_depth).astype(np.int32)
        entries: list[tuple[int, int, int, int]] = []
        for tile in level_tiles:
            for local, parent in enumerate(closed[int(tile)]):
                entries.append(
                    (
                        int(compact_feature[parent]),
                        int(parent),
                        int(tile),
                        local,
                    )
                )
        entries.sort(key=lambda item: (item[0], item[1]))
        for slot, (_, _, tile, local) in enumerate(entries):
            int_record[tile, local, 4] = slot

        if entries:
            sorted_features = np.fromiter(
                (item[0] for item in entries),
                dtype=np.int32,
                count=len(entries),
            )
            used_features, counts = np.unique(
                sorted_features, return_counts=True
            )
            offsets = np.empty(len(used_features) + 1, dtype=np.int32)
            offsets[0] = 0
            np.cumsum(counts, dtype=np.int32, out=offsets[1:])
        else:
            used_features = np.empty(0, dtype=np.int32)
            offsets = np.zeros(1, dtype=np.int32)
        n_parents = len(entries)
        max_level_parents = max(max_level_parents, n_parents)
        levels.append(
            TreeletLevel(
                treelets=np.ascontiguousarray(level_tiles),
                used_features=np.ascontiguousarray(
                    used_features, dtype=np.int32
                ),
                feature_offsets=np.ascontiguousarray(offsets),
                n_parents=n_parents,
            )
        )

    q_offsets = np.empty(n_treelets + 1, dtype=np.int32)
    q_offsets[0] = 0
    np.cumsum(headers[:, 1], dtype=np.int32, out=q_offsets[1:])

    return TreeletPlan(
        supported=True,
        reason=None,
        treelet_size=treelet_size,
        max_portals=max_portals,
        headers=np.ascontiguousarray(headers),
        q_offsets=np.ascontiguousarray(q_offsets[:-1]),
        int_record=np.ascontiguousarray(int_record),
        float_record=np.ascontiguousarray(float_record),
        leaf_record=np.ascontiguousarray(leaf_record),
        portals=np.ascontiguousarray(portals),
        roots=np.ascontiguousarray(roots),
        levels=tuple(levels),
        n_treelets=n_treelets,
        total_q_states=int(q_offsets[-1]),
        max_level_parents=max_level_parents,
        node_to_treelet=node_to_treelet,
        treelet_nodes=tuple(
            np.asarray(nodes, dtype=np.int32) for nodes in closed
        ),
    )
