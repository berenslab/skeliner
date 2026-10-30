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
    from skeliner import Skeleton, Soma, _rs
    from skeliner.python import dataclass as python_dataclass

    from ._parity import assert_parity

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
        assert_parity(
            _rs.dataclass.node_neighbors(edges, n),
            python_dataclass.node_neighbors(edges, n),
        )
    for kernel in (_rs.dataclass, python_dataclass):
        with pytest.raises(ValueError, match="edge names node 3, not one of 3"):
            kernel.node_neighbors(np.array([[0, 1], [3, 0]]), 3)


# -----------------------------------------------------------------------------
# the dataclasses: Skeleton, Soma, MeshComponents
# -----------------------------------------------------------------------------


def _tree(rng, n: int, *, unit: str | None = "nm"):
    """A random tree of *n* nodes around a fitted soma (with a nucleus),
    with a centreline radius and its distances kept, as `skeletonize` keeps
    them."""
    from skeliner import Skeleton, Soma

    parent = [int(rng.integers(0, i)) for i in range(1, n)]
    nodes = np.zeros((n, 3))
    for i, p in enumerate(parent, start=1):
        nodes[i] = nodes[p] + rng.normal(size=3) * 1500.0
    soma = Soma.fit(rng.normal(size=(40, 3)) * [3000.0, 2000.0, 1200.0])
    soma.verts = np.arange(5, dtype=np.int64)
    soma.nucleus = {
        "center": soma.center + 100.0,
        "peak_r": 900.0,
        "z_range": (-700.0, 800.0),
        "slices": rng.normal(size=(6, 4)) * 1000.0,
    }
    return Skeleton(
        soma=soma,
        nodes=nodes,
        radii={
            "median": rng.uniform(50, 400, n),
            "centerline": rng.uniform(50, 400, n),
        },
        edges=np.sort(np.array([[p, i] for i, p in enumerate(parent, 1)]), axis=1),
        ntype=None,
        meta={} if unit is None else {"unit": unit},
        extra={
            "cl_dist_vids": np.arange(4, dtype=np.int64),
            "cl_dist_vals": np.array([150.0, 270.0, 1001.0, 3.0]),
        },
    )


def test_convert_unit_scales_every_length():
    import copy

    rng = np.random.default_rng(7)
    skel = _tree(rng, 30)
    before = copy.deepcopy(skel)
    inside = skel.soma.contains(skel.nodes)
    assert inside[0] and not inside.all()  # the soma holds some nodes, not all

    skel.convert_unit("um")
    assert skel.meta["unit"] == "um"
    # nm -> um divides by 1000, so 1001 nm is 1.001 um, not 1.0010000000000001
    assert np.array_equal(skel.nodes, before.nodes / 1000.0)
    for k in before.radii:
        assert np.array_equal(skel.radii[k], before.radii[k] / 1000.0)
    assert np.array_equal(skel.extra["cl_dist_vals"], [0.15, 0.27, 1.001, 0.003])
    assert np.array_equal(skel.extra["cl_dist_vids"], before.extra["cl_dist_vids"])
    soma, was = skel.soma, before.soma
    assert np.array_equal(soma.center, was.center / 1000.0)
    assert np.array_equal(soma.axes, was.axes / 1000.0)
    assert np.array_equal(soma.R, was.R) and np.array_equal(soma.verts, was.verts)
    nuc, nuc_was = soma.nucleus, was.nucleus
    assert np.array_equal(nuc["center"], nuc_was["center"] / 1000.0)
    assert np.array_equal(nuc["slices"], nuc_was["slices"] / 1000.0)
    assert nuc["peak_r"] == 0.9 and nuc["z_range"] == (-0.7, 0.8)
    # the soma holds the same nodes, at a thousandth of the distance
    assert np.array_equal(soma.contains(skel.nodes), inside)
    np.testing.assert_allclose(
        soma.distance_to_surface(skel.nodes),
        was.distance_to_surface(before.nodes) / 1000.0,
        rtol=1e-9,
        atol=1e-12,
    )

    # and back: um -> nm multiplies by exactly 1000 (one rounding, so the
    # round trip may still be an ulp off)
    skel.convert_unit("nm")
    assert skel.meta["unit"] == "nm"
    um = np.array([0.15, 0.27, 1.001, 0.003])
    assert np.array_equal(skel.extra["cl_dist_vals"], um * 1000.0)
    one = _tree(rng, 2)
    one.nodes = np.array([[1.0, 2.0, 3.0], [0.001, 7.0, 1e-3]])
    one.meta["unit"] = "um"
    one.convert_unit("nm")
    assert one.nodes.tolist() == [[1000.0, 2000.0, 3000.0], [1.0, 7000.0, 1.0]]
    assert one._get_unit_conversion_factor("µm", "nm") == 1000.0
    assert one._get_unit_conversion_factor("mm", "μm") == 1000.0
    assert one._get_unit_conversion_factor("nm", "um") == 0.001


def test_convert_unit_records_a_unit_it_need_not_convert_to():
    rng = np.random.default_rng(8)
    skel = _tree(rng, 5, unit=None)
    nodes = skel.nodes.copy()
    with pytest.raises(ValueError, match="current_unit must be specified"):
        skel.convert_unit("nm")
    skel.convert_unit("nm", current_unit="nm")
    assert skel.meta["unit"] == "nm" and np.array_equal(skel.nodes, nodes)
    skel.convert_unit("nanometer")
    assert skel.meta["unit"] == "nanometer" and np.array_equal(skel.nodes, nodes)
    with pytest.raises(ValueError, match="Unsupported unit conversion"):
        skel.convert_unit("inch")


def test_convert_unit_leaves_skeletons_sharing_its_arrays_alone():
    import copy

    rng = np.random.default_rng(9)
    skel = _tree(rng, 12)
    other = copy.copy(skel)  # the same arrays, the same soma, the same dicts
    nodes, axes = skel.nodes.copy(), skel.soma.axes.copy()
    radii = {k: v.copy() for k, v in skel.radii.items()}
    skel.convert_unit("mm")
    assert np.array_equal(other.nodes, nodes)
    assert np.array_equal(other.soma.axes, axes)
    assert all(np.array_equal(other.radii[k], v) for k, v in radii.items())
    assert other.extra["cl_dist_vals"].tolist() == [150.0, 270.0, 1001.0, 3.0]
    assert other.meta["unit"] == "nm" and skel.meta["unit"] == "mm"
    # integer coordinates convert too
    ints = _tree(rng, 3)
    ints.nodes = np.array([[0, 0, 0], [1000, 0, 0], [0, 2000, 0]])
    ints.convert_unit("um")
    assert ints.nodes.tolist() == [[0, 0, 0], [1, 0, 0], [0, 2, 0]]


def _without_caches(skel):
    import copy

    fresh = copy.copy(skel)  # the edits below replace arrays, never write them
    fresh._invalidate_spatial_index()
    return fresh


def test_the_skeleton_caches_follow_its_nodes_and_edges():
    import skeliner as sk

    rng = np.random.default_rng(10)
    skel = _tree(rng, 80)
    points = skel.nodes[rng.choice(80, 50)] + rng.normal(size=(50, 3)) * 800

    def after(edit):
        s = _without_caches(skel)
        sk.dx.distance(s, points)  # builds the caches
        edit(s)
        got = sk.dx.distance(s, points)
        assert np.array_equal(got, sk.dx.distance(_without_caches(s), points))
        return s

    leaves = np.flatnonzero(np.bincount(skel.edges.ravel(), minlength=80) == 1)
    u, v = (int(x) for x in skel.edges[np.flatnonzero(skel.edges[:, 1] == 40)[0]])
    pruned = after(lambda s: sk.post.prune(s, kind="twigs", num_nodes=2))
    assert len(pruned.nodes) < 80  # the prune dropped nodes
    assert len(after(lambda s: sk.post.clip(s, u, v, drop_orphans=True)).nodes) < 80
    after(lambda s: sk.post.prune(s, kind="nodes", nodes=leaves[:5]))
    after(lambda s: sk.post.clip(s, u, v))
    after(lambda s: sk.post.graft(s, int(leaves[0]), int(leaves[1])))
    after(lambda s: setattr(s, "nodes", s.nodes + 5000.0))
    after(lambda s: setattr(s, "edges", s.edges[: len(s.edges) // 2]))


def test_soma_containment_follows_its_axes():
    from skeliner import Soma

    rng = np.random.default_rng(11)
    soma = Soma.fit(rng.normal(size=(30, 3)) * [3.0, 2.0, 1.0])
    points = soma.center + rng.normal(size=(500, 3)) * 4.0
    for edit in (
        lambda s: setattr(s, "axes", s.axes * 2.0),
        lambda s: s.axes.__imul__(0.5),
        lambda s: setattr(s, "R", s.R[:, [0, 2, 1]] @ np.diag([1.0, 1.0, -1.0])),
    ):
        edit(soma)
        same = Soma(soma.center.copy(), soma.axes.copy(), soma.R.copy())
        assert np.array_equal(soma.contains(points), same.contains(points))
        assert soma.contains(points).any() and not soma.contains(points).all()


def test_soma_survives_a_pickle_from_before_w_was_derived():
    import pickle

    from skeliner import Soma

    soma = Soma.from_sphere(np.array([1.0, 2.0, 3.0]), 2.0, verts=np.arange(3))
    state = (None, {f: getattr(soma, f) for f in Soma.__dataclass_fields__})
    state[1]["_W"] = np.eye(3) / 2.0  # an old pickle carried the cached map
    old = Soma.__new__(Soma)
    old.__setstate__(state)
    assert np.array_equal(old.verts, soma.verts) and old.contains(soma.center)
    again = pickle.loads(pickle.dumps(old))
    assert np.array_equal(again.center, soma.center) and again.nucleus is None


def test_soma_remap_keeps_the_nucleus():
    rng = np.random.default_rng(12)
    soma = _tree(rng, 2).soma
    remapped = soma.remap(np.array([-1, 0, 1, -1, 2, 3], dtype=np.int64))
    assert remapped.verts.tolist() == [0, 1, 2]
    assert remapped.nucleus is not None
    assert remapped.nucleus["peak_r"] == soma.nucleus["peak_r"]
    assert np.array_equal(remapped.nucleus["slices"], soma.nucleus["slices"])


def test_soma_fit_is_a_rotation():
    from skeliner import Soma

    rng = np.random.default_rng(13)
    flipped = 0
    for _ in range(100):
        pts = rng.normal(size=(int(rng.integers(5, 60)), 3)) * rng.uniform(1, 9, 3)
        soma = Soma.fit(pts)
        R = soma.R
        assert np.dot(np.cross(R[:, 0], R[:, 1]), R[:, 2]) > 0  # right-handed
        evals, evecs = np.linalg.eigh(np.cov(pts - pts.mean(axis=0), rowvar=False))
        raw = Soma(soma.center, soma.axes, evecs[:, ::-1])  # as fit made it
        flipped += not np.array_equal(raw.R, soma.R)
        # the same ellipsoid: every measure squares the body coordinates
        x = soma.center + rng.normal(size=(40, 3)) * 5.0
        assert np.array_equal(raw.contains(x), soma.contains(x))
        assert np.array_equal(raw.distance_to_surface(x), soma.distance_to_surface(x))
    assert 20 < flipped < 100  # the eigenvectors come both ways


def test_a_skeleton_leaves_its_inputs_alone_and_may_have_no_nodes():
    from skeliner import Skeleton, Soma

    soma = Soma.from_sphere(np.zeros(3), 1.0, verts=None)
    ntype = np.array([3, 3, 0], dtype=np.int8)
    skel = Skeleton(
        soma=soma,
        nodes=np.zeros((3, 3)),
        radii={"median": np.ones(3)},
        edges=np.array([[0, 1], [1, 2]]),
        ntype=ntype,
    )
    assert ntype.tolist() == [3, 3, 0] and skel.ntype.tolist() == [-1, 3, 0]
    for ntype in (None, np.zeros(0, dtype=np.int8)):
        empty = Skeleton(
            soma=soma,
            nodes=np.zeros((0, 3)),
            radii={"median": np.zeros(0)},
            edges=np.zeros((0, 2), dtype=np.int64),
            ntype=ntype,
        )
        assert empty.ntype.dtype == np.int8 and empty.ntype.shape == (0,)
    # no edges, as an empty list makes them (a one-node SWC)
    one = Skeleton(
        soma=soma,
        nodes=np.zeros((1, 3)),
        radii={"median": np.ones(1)},
        edges=np.asarray([]),
        ntype=None,
    )
    assert one.edges.shape == (0, 2) and one.edges.dtype == np.int64
    assert one.ntype.tolist() == [-1]
    with pytest.raises(ValueError, match="nodes must be"):
        Skeleton(
            soma=soma, nodes=np.zeros((2, 2)), radii={}, edges=one.edges, ntype=None
        )


def test_recommend_radius_names_a_radius_the_skeleton_has():
    from skeliner import Skeleton, Soma

    soma = Soma.from_sphere(np.zeros(3), 1.0, verts=None)
    for radii, want in (
        ({"mean": np.ones(2)}, "mean"),
        ({"median": np.ones(2)}, "median"),
        ({"trim": np.ones(2), "calibrated": np.ones(2)}, "trim"),
        ({"mean": np.ones(2), "centerline": np.ones(2)}, "centerline"),
    ):
        skel = Skeleton(
            soma=soma,
            nodes=np.zeros((2, 3)),
            radii=radii,
            edges=np.array([[0, 1]]),
            ntype=None,
        )
        assert skel.recommend_radius()[0] == want
        assert skel.r is radii[want]


def test_rescue_discarded_moves_each_fragment_named_once():
    from skeliner.dataclass import Discarded, MeshComponents, Neurites, Organelles

    def components():
        return MeshComponents(
            soma=None,
            organelles=Organelles.empty(10),
            neurites=Neurites([np.arange(5)]),
            discarded=Discarded([np.array([i]) for i in range(4)]),
        )

    def lists(c):
        return [x.tolist() for x in c.neurites], [x.tolist() for x in c.discarded]

    for named, moved in (
        ([1, 1], [1]),
        ([3, -1], [3]),
        ([-4, 2, 0], [0, 2]),
        (np.int64(2), [2]),
    ):
        c = components()
        c.rescue_discarded(named)
        assert lists(c) == (
            [[0, 1, 2, 3, 4], *([i] for i in moved)],
            [[i] for i in range(4) if i not in moved],
        )
    for bad in ([2, -9], [0, 4]):
        c = components()
        with pytest.raises(IndexError):
            c.rescue_discarded(bad)
        assert lists(c) == ([[0, 1, 2, 3, 4]], [[0], [1], [2], [3]])  # untouched


def _somas_and_points(rng):
    """Somas with points to measure: fitted ones with points everywhere,
    and axis-aligned ones (spheres and spheroids too) with points on their
    axes and planes, the centre, and points at the surface — where the
    distance takes its degenerate branches."""
    from skeliner import Soma

    for _ in range(25):
        soma = Soma.fit(rng.normal(size=(30, 3)) * rng.uniform(0.5, 8.0, 3))
        n = int(rng.choice([1, 2, 7, 300, 1500]))
        yield soma, soma.center + rng.normal(size=(n, 3)) * soma.axes.max()
    # enough inside and out for the pool: a third of a ball 1.5 times the soma
    ball = rng.normal(size=(3000, 3))
    ball *= (
        1.5 * rng.random((3000, 1)) ** (1 / 3) / np.linalg.norm(ball, axis=1)[:, None]
    )
    yield soma, soma.center + (ball * soma.axes) @ soma.R.T
    for axes in ([3.0, 2.0, 1.0], [2.0, 2.0, 1.0], [3.0, 1.0, 1.0], [2.0, 2.0, 2.0]):
        for R in (
            np.eye(3),
            np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
        ):
            soma = Soma(np.zeros(3), axes, R)
            body = rng.uniform(-1.2, 1.2, size=(600, 3)) * axes
            body[rng.random(body.shape) < 0.3] = 0.0  # on axes and planes
            body[:8] = 0.0
            body[8:20] = np.round(body[8:20] / np.asarray(axes), 1) * axes
            on = rng.normal(size=(20, 3))
            body[20:40] = on / np.linalg.norm(on / axes, axis=1)[:, None]
            yield soma, body @ R.T


def test_soma_measures_on_both_paths():
    from skeliner import _rs
    from skeliner.python import dataclass as python_dataclass

    from ._parity import assert_parity

    rng = np.random.default_rng(14)
    big = None
    for soma, x in _somas_and_points(rng):
        args = (soma.center, soma.axes, soma.R, x)
        for limit in (1.0, 0.25):
            assert_parity(
                _rs.dataclass.soma_contains(*args, limit),
                python_dataclass.soma_contains(*args, limit),
            )
        assert_parity(
            _rs.dataclass.soma_distance_to_center(soma.center, x),
            python_dataclass.soma_distance_to_center(soma.center, x),
        )
        for atol, max_iter in ((1e-9, 64), (1e-3, 5), (1e-9, 0), (1e-9, -2)):
            rust = _rs.dataclass.soma_distance_to_surface(*args, atol, max_iter)
            assert_parity(
                rust, python_dataclass.soma_distance_to_surface(*args, atol, max_iter)
            )
            if max_iter == 64:
                inside, outside = (rust < 0).sum(), (rust > 0).sum()
                big = big or (inside >= 256 and outside >= 1024)
    assert big  # the points went to the pool, inside and out, at least once
    with_bad = np.zeros((2, 2))
    for kernel in (_rs.dataclass, python_dataclass):
        with pytest.raises(
            ValueError, match=r"expected \(N, 3\) array, got shape \[2, 2\]"
        ):
            kernel.soma_distance_to_center(np.zeros(3), with_bad)


def test_soma_distance_inside_is_euclidean():
    """Inside, no point of the surface is nearer than the distance, and one
    is about as near; outside as before. The distance along the ray from
    the centre (as it was) is never nearer, and on an elongated soma much
    farther."""
    import skeliner as sk
    from skeliner import Soma

    rng = np.random.default_rng(15)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 1200), np.linspace(0, np.pi, 600))
    unit = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1)
    unit = unit.reshape(-1, 3)
    for axes in ([3.0, 2.0, 1.0], [5.0, 1.5, 1.0], [2.0, 2.0, 2.0]):
        R = Soma.fit(rng.normal(size=(20, 3))).R
        soma = Soma(np.array([1.0, -2.0, 0.5]), axes, R)
        surface = soma.center + (unit * axes) @ R.T
        body = rng.uniform(-1, 1, size=(400, 3))
        body = body[np.linalg.norm(body, axis=1) < 0.98][:40] * axes
        x = soma.center + body @ R.T
        for path in ("rust", "python"):
            with sk.backend(path):
                d = soma.distance_to_surface(x)
            assert (d < 0).all()
            for p, depth in zip(x, -d):
                near = np.sqrt(((surface - p) ** 2).sum(axis=1)).min()
                assert near >= depth - 1e-9 and near - depth < 5e-3, (near, depth)
            s = np.linalg.norm(body / axes, axis=1)
            radial = np.linalg.norm(body / s[:, None] - body, axis=1)
            assert (radial >= -d - 1e-12).all()
            if axes[0] / axes[2] >= 3:
                assert (radial - -d).max() > 0.5


def test_remapped_edges_on_both_paths():
    from skeliner import _rs
    from skeliner.python import dataclass as python_dataclass

    from ._parity import assert_parity

    rng = np.random.default_rng(16)
    kept = 0
    for _ in range(200):
        n = int(rng.integers(1, 25))
        edges = rng.integers(0, n, size=(int(rng.integers(0, 40)), 2))
        old2new = np.where(rng.random(n) < 0.3, -1, rng.permutation(n))
        rust = _rs.dataclass.remap_edges(edges, old2new)
        assert_parity(rust, python_dataclass.remap_edges(edges, old2new))
        if len(edges):
            assert np.array_equal(rust, remap_edges(edges, old2new))
        kept += len(rust)
    assert kept > 500
    for kernel in (_rs.dataclass, python_dataclass):
        with pytest.raises(ValueError, match="edge names node 4, not one of 3"):
            kernel.remap_edges(np.array([[0, 1], [4, 2]]), np.arange(3))


def test_vert2node_from_bins_of_any_shape():
    from skeliner._state import VertexMap

    want = {3: 2, 0: 0, 4: 1, 1: 2}  # keys first seen, the last bin wins
    for bins in (
        [np.array([3, 0]), np.array([4], dtype=np.int32), [1, 3]],
        [np.array([[3], [0]]), np.array([4]), np.array([[1, 3]])],
        [np.array([3, 0]), np.int64(4), (1, 3)],
    ):
        vm = VertexMap.from_bins(bins)
        assert list(vm.items()) == list(want.items())
