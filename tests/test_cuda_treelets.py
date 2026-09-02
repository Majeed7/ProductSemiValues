import numpy as np


def _make_host_plan():
    from quadrashap.treeshap.cuda import _build_host_plan
    from quadrashap.treeshap.quadrature_tree import QuadratureTreeShapBackend
    from quadrashap.treeshap.sklearn import sklearn_to_unified
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    X, y = make_regression(
        n_samples=500,
        n_features=12,
        n_informative=9,
        random_state=13,
    )
    model = RandomForestRegressor(
        n_estimators=5,
        max_depth=9,
        random_state=17,
    ).fit(X, y)
    backend = QuadratureTreeShapBackend(device="cpu")
    prepared = backend.prepare(sklearn_to_unified(model))
    return _build_host_plan(prepared)


def test_treelet_plan_partitions_and_schedules_every_internal_node():
    from quadrashap.treeshap.cuda_treelets import build_treelet_plan

    host = _make_host_plan()
    plan = build_treelet_plan(host, treelet_size=7, max_portals=8)
    assert plan.supported, plan.reason

    nodes = np.concatenate(plan.treelet_nodes)
    np.testing.assert_array_equal(
        np.sort(nodes), np.arange(host.n_internal, dtype=np.int32)
    )
    assert max(map(len, plan.treelet_nodes)) <= 7
    assert plan.q_offsets[-1] + plan.headers[-1, 1] == plan.total_q_states

    tile_depth = np.full(plan.n_treelets, -1, dtype=np.int32)
    for depth, level in enumerate(plan.levels):
        tile_depth[level.treelets] = depth
        slots = []
        slot_features = []
        for tile in level.treelets:
            count = int(plan.headers[tile, 0])
            slots.extend(plan.int_record[tile, :count, 4].tolist())
            slot_features.extend(
                zip(
                    plan.int_record[tile, :count, 4].tolist(),
                    plan.int_record[tile, :count, 3].tolist(),
                )
            )
        assert sorted(slots) == list(range(level.n_parents))
        ordered_features = np.asarray(
            [feature for _, feature in sorted(slot_features)],
            dtype=np.int32,
        )
        expanded = np.repeat(
            level.used_features, np.diff(level.feature_offsets)
        )
        np.testing.assert_array_equal(ordered_features, expanded)

    assert np.all(tile_depth >= 0)
    for tile, tile_nodes in enumerate(plan.treelet_nodes):
        count, _, portal_count = plan.headers[tile]
        assert count == len(tile_nodes)
        assert portal_count <= 8
        for local in range(int(count)):
            for ref in plan.int_record[tile, local, :2]:
                if 0 <= ref < 7:
                    # Root-first order guarantees children are reconstructed
                    # before parents by the reverse local traversal.
                    assert ref > local
                elif ref >= 7:
                    portal = int(ref - 7)
                    assert portal < portal_count
                    child_tile = plan.portals[tile, portal]
                    assert tile_depth[child_tile] == tile_depth[tile] + 1
