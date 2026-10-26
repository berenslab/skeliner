import numpy as np
import pytest

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


def test_rebuild_vert2node_lists_keys_by_node_and_lets_the_last_node_win():
    # vertex 7 is in nodes 0 and 2: it keeps its first place and maps to 2
    mapping = rebuild_vert2node(
        [
            np.array([7, 3], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([1, 7], dtype=np.int32),
        ]
    )

    assert list(mapping.items()) == [(7, 2), (3, 0), (1, 2)]
    assert all(type(k) is int and type(v) is int for k, v in mapping.items())
    assert rebuild_vert2node([]) == {}


def test_swap_nodes_keeps_the_vertex_order():
    state = _make_state()
    swap_nodes(state, 1, 2)

    assert list(state.vert2node.items()) == [(0, 0), (1, 2), (2, 2), (3, 1)]


def _dict_of_bins(bins):
    """vert2node as the dict it was before VertexMap."""
    verts = [np.asarray(v, dtype=np.int64).ravel() for v in bins]
    nids = np.repeat(np.arange(len(verts)), [v.size for v in verts])
    flat = np.concatenate(verts) if verts else np.empty(0, np.int64)
    return dict(zip(flat.tolist(), nids.tolist()))


def test_a_vertex_map_is_the_dict_it_replaces():
    import copy
    import pickle

    import skeliner as sk
    from skeliner import _rs
    from skeliner._state import VertexMap, rebuild_vert2node
    from skeliner.python import dataclass as python_dataclass

    from ._parity import assert_parity

    rng = np.random.default_rng(83)
    for trial in range(60):
        n_bins = int(rng.integers(0, 30))
        bins = [
            rng.integers(0, 80, size=int(rng.integers(0, 6))) for _ in range(n_bins)
        ]
        want = _dict_of_bins(bins)
        for path in ("rust", "python"):
            with sk.backend(path):
                vm = rebuild_vert2node(bins)
            assert type(vm) is VertexMap
            assert list(vm) == list(want) and list(vm.items()) == list(want.items())
            assert list(vm.values()) == list(want.values()) and len(vm) == len(want)
            assert vm == want and dict(vm) == want
        ptr = np.zeros(n_bins + 1, dtype=np.int64)
        np.cumsum([b.size for b in bins], out=ptr[1:])
        idx = np.concatenate(bins) if bins else np.empty(0, np.int64)
        for n in (None, 90):
            assert_parity(
                _rs.dataclass.vertex_map(ptr, idx, n),
                python_dataclass.vertex_map(ptr, idx, n),
            )
    # lookups and edits as a dict's
    vm, d = rebuild_vert2node([[3, 0], [1, 1, 4], [3]]), {3: 2, 0: 0, 1: 1, 4: 1}
    assert vm == d and list(vm) == [3, 0, 1, 4]
    assert vm.get(9) is None and vm.get(9, -5) == -5 and vm[np.int64(3)] == 2
    assert 9 not in vm and -1 not in vm and "x" not in vm
    with pytest.raises(KeyError):
        vm[7]
    vm[10] = 4  # a new key goes last; an old one keeps its place
    vm[0] = 3
    d[10], d[0] = 4, 3
    assert list(vm.items()) == list(d.items())
    del vm[1]
    del d[1]
    assert list(vm.items()) == list(d.items())
    with pytest.raises(KeyError):
        del vm[1]
    with pytest.raises(ValueError):
        vm[2] = -1
    for twin in (vm.copy(), copy.deepcopy(vm), pickle.loads(pickle.dumps(vm))):
        assert list(twin.items()) == list(d.items())
    twin = vm.copy()
    twin[11] = 0
    assert 11 not in vm
    # the arrays, and the whole-map edits
    assert vm.lut().tolist() == [3, -1, -1, 2, 1, -1, -1, -1, -1, -1, 4]
    assert vm.lut(12)[-1] == -1
    with pytest.raises(ValueError, match="names vertex 10"):
        vm.lut(5)
    vm.swap_nodes(2, 4)
    assert list(vm.items()) == [(3, 4), (0, 3), (4, 1), (10, 2)]
    old2new = np.array([0, -1, 1, 2, 5])
    assert list(vm.remapped(old2new).items()) == [(3, 5), (0, 2), (10, 1)]


def test_a_skeleton_keeps_vert2node_as_a_vertex_map():
    from skeliner import Skeleton, Soma
    from skeliner._state import VertexMap

    skel = Skeleton(
        soma=Soma.from_sphere(np.zeros(3), 1.0, verts=None),
        nodes=np.zeros((2, 3)),
        radii={"median": np.ones(2)},
        edges=np.array([[0, 1]]),
        ntype=None,
        node2verts=[np.array([0, 1]), np.array([2])],
        vert2node={0: 0, 1: 0, 2: 1},
    )
    assert type(skel.vert2node) is VertexMap and skel.vert2node == {0: 0, 1: 0, 2: 1}


def test_neighbour_lists_follow_the_edges():
    from skeliner import Skeleton, Soma

    rng = np.random.default_rng(89)
    for _ in range(100):
        n = int(rng.integers(1, 30))
        edges = rng.integers(0, n, size=(int(rng.integers(0, 40)), 2))
        skel = Skeleton(
            soma=Soma.from_sphere(np.zeros(3), 1.0, verts=None),
            nodes=np.zeros((n, 3)),
            radii={"median": np.ones(n)},
            edges=edges,
            ntype=None,
        )
        want = [[] for _ in range(n)]
        for u, v in edges.tolist():
            want[u].append(v)
            want[v].append(u)
        got = skel._ensure_node_neighbors()
        assert [g.tolist() for g in got] == want
        assert all(g.dtype == np.int64 for g in got)
