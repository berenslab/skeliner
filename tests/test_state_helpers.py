import numpy as np

from skeliner._state import (
    SkeletonState,
    compact_state,
    remap_edges,
    rebuild_vert2node,
    swap_nodes,
)


def _make_state(with_node2verts: bool = True) -> SkeletonState:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    radii = {
        "median": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "mean": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    }
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    if with_node2verts:
        node2verts = [
            np.array([0], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([], dtype=np.int64),
        ]
        vert2node = {0: 0, 1: 1, 2: 1, 3: 2}
    else:
        node2verts = None
        vert2node = None
    return SkeletonState(
        nodes=nodes,
        radii=radii,
        edges=edges,
        node2verts=node2verts,
        vert2node=vert2node,
    )


def test_compact_state_slices_all_arrays_and_builds_mapping():
    state = _make_state()
    keep_mask = np.array([True, False, True, False], dtype=bool)

    new_state, old2new = compact_state(state, keep_mask, return_old2new=True)

    assert np.allclose(new_state.nodes, state.nodes[[0, 2]])
    for key in state.radii:
        assert np.allclose(new_state.radii[key], state.radii[key][[0, 2]])

    # All edges touched dropped nodes, so nothing should survive.
    assert new_state.edges.shape == (0, 2)

    assert new_state.node2verts is not None
    assert len(new_state.node2verts) == 2
    assert np.array_equal(new_state.node2verts[0], np.array([0], dtype=np.int64))
    assert np.array_equal(new_state.node2verts[1], np.array([3], dtype=np.int64))
    assert new_state.vert2node == {0: 0, 3: 1}

    assert np.array_equal(old2new, np.array([0, -1, 1, -1]))


def test_compact_state_handles_absent_node2verts():
    state = _make_state(with_node2verts=False)
    keep_mask = np.array([True, True, False, False], dtype=bool)

    new_state, _ = compact_state(state, keep_mask, return_old2new=True)

    assert new_state.node2verts is None
    assert new_state.vert2node is None
    assert new_state.edges.shape == (1, 2)


def test_remap_edges_deduplicates_and_drops_removed_nodes():
    edges = np.array([[0, 1], [1, 0], [2, 3]], dtype=np.int64)
    old2new = np.array([0, -1, 1, 2], dtype=np.int64)

    remapped = remap_edges(edges, old2new)

    assert np.array_equal(remapped, np.array([[1, 2]], dtype=np.int64))


def test_rebuild_vert2node_handles_empty_lists():
    mapping = rebuild_vert2node(
        [
            np.array([], dtype=np.int64),
            np.array([5, 7], dtype=np.int64),
        ]
    )

    assert mapping == {5: 1, 7: 1}
    assert rebuild_vert2node(None) is None


def test_swap_nodes_updates_all_linked_structures():
    state = _make_state()
    original_nodes = state.nodes.copy()
    original_radii = {k: v.copy() for k, v in state.radii.items()}
    swap_nodes(state, 0, 2)

    assert np.allclose(state.nodes[0], original_nodes[2])
    assert np.allclose(state.nodes[2], original_nodes[0])
    for key in state.radii:
        assert state.radii[key][0] == original_radii[key][2]
        assert state.radii[key][2] == original_radii[key][0]

    assert np.array_equal(state.node2verts[0], np.array([3], dtype=np.int64))
    assert np.array_equal(state.node2verts[2], np.array([0], dtype=np.int64))
    assert state.vert2node[3] == 0
    assert state.vert2node[0] == 2
