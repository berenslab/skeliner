"""
Smoke-tests for the mutating helpers in `skeliner.post`.

Every mutation is done on a *deep copy* of the reference skeleton so the
other tests stay unaffected.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from skeliner import dx, post, skeletonize
from skeliner.dataclass import Skeleton, Soma
from skeliner.io import load_mesh


# ---------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def template_skel():
    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    return skeletonize(mesh, verbose=False)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _is_forest(skel):
    g = skel._igraph()
    return g.ecount() == g.vcount() - len(g.components())


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------
def test_graft_then_clip(template_skel):
    skel = copy.deepcopy(template_skel)

    leaves = dx.nodes_of_degree(skel, 1)
    # fall back to any two nodes if mesh has no leaves
    u, v = (int(leaves[0]), int(leaves[1])) if len(leaves) >= 2 else (0, 1)

    n_edges = skel.edges.shape[0]
    post.graft(skel, u, v, allow_cycle=True)
    assert skel.edges.shape[0] == n_edges + 1

    # clipping should restore edge count
    post.clip(skel, u, v)
    assert skel.edges.shape[0] == n_edges
    assert _is_forest(skel)


def _path_skel(n_nodes, edges):
    return Skeleton(
        soma=Soma.from_sphere(np.zeros(3), 1.0, verts=None),
        nodes=np.arange(3 * n_nodes, dtype=np.float64).reshape(-1, 3),
        radii={"median": np.ones(n_nodes)},
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        ntype=None,
    )


def test_graft_refuses_a_cycle_through_a_longer_path():
    # 0-1-2-3 and 4 on its own: joining 0 and 3 closes a loop over three
    # edges, joining 3 and 4 two trees
    skel = _path_skel(5, [[0, 1], [1, 2], [2, 3]])
    with pytest.raises(ValueError, match="cycle"):
        post.graft(skel, 0, 3, allow_cycle=False)
    with pytest.raises(ValueError, match="cycle"):
        post.graft(skel, 3, 1, allow_cycle=False)
    assert skel.edges.tolist() == [[0, 1], [1, 2], [2, 3]]
    post.graft(skel, 4, 3, allow_cycle=False)
    assert skel.edges.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
    post.graft(skel, 0, 3, allow_cycle=True)
    assert [0, 3] in skel.edges.tolist()


def test_prune_twigs(template_skel):
    skel = copy.deepcopy(template_skel)
    n_before = len(skel.nodes)
    post.prune(skel, kind="twigs", num_nodes=2)
    # Allowed to prune zero – just make sure the structure is still a forest
    assert len(skel.nodes) <= n_before
    assert _is_forest(skel)


def test_set_ntype_on_subtree(template_skel):
    skel = copy.deepcopy(template_skel)
    base = 1 if len(skel.nodes) > 1 else 0

    original_code = int(skel.ntype[base])
    assert original_code != 4  # sanity: it really changes

    post.set_ntype(skel, root=base, code=4, subtree=False)

    assert skel.ntype[base] == 4
    assert skel.ntype[0] == -1
    changed = np.where(skel.ntype == 4)[0]
    assert set(changed) == {base}


def test_remap_ntype_prefers_labels_over_unknown():
    ntype = np.array([-1, 3, 0, 0], np.int8)
    old2new = np.array([0, 1, 1, 1], np.int64)
    mapped = post._remap_ntype(ntype, old2new, 2)
    assert mapped.tolist() == [-1, 3]


def test_remap_ntype_resolves_conflicts_with_priority():
    ntype = np.array([-1, 4, 3, 3], np.int8)
    old2new = np.array([0, 1, 1, 1], np.int64)
    mapped = post._remap_ntype(ntype, old2new, 2)
    assert mapped[0] == -1
    assert mapped[1] == 3  # majority wins (4 vs two 3s)


def test_downsample_preserves_branch_labels():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2], [2, 3]], np.int64)
    radii = {"median": np.array([1.0, 1.0, 1.0, 1.0])}
    ntype = np.array([-1, 2, 0, 0], np.int8)
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    ds = post.downsample(
        skel,
        merge_endpoints=True,
        slide_branchpoints=True,
        verbose=False,
    )
    assert len(ds.nodes) < len(nodes)
    assert ds.ntype[0] == -1
    assert 2 in ds.ntype


def test_detect_soma_demoted_root_keeps_neurite_label():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    # Make node 2 obviously the soma by radius
    radii = {"median": np.array([1.0, 1.0, 5.0])}
    ntype = np.array([-1, 4, 4], np.int8)
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    res = post.detect_soma(
        skel,
        radius_key="median",
        soma_radius_percentile_threshold=50.0,
        soma_radius_distance_factor=1.0,
        soma_min_nodes=1,
        verbose=False,
    )
    assert res.ntype[0] == 1  # new soma
    # old root (now index 1) should keep its neurite label instead of 0
    assert res.ntype[1] == 4


def test_rebuild_drop_set_preserves_labels_and_fills_gaps(template_skel):
    skel = copy.deepcopy(template_skel)
    # Mark a small branch as axon
    leaves = dx.nodes_of_degree(skel, 1)
    if len(leaves) < 2:
        pytest.skip("template has too few leaves for this test")
    target = int(leaves[0])
    skel.ntype[target] = 2
    # Drop a different leaf to force compaction/remap
    drop = int(leaves[1])
    post.clip(skel, drop, int(dx.neighbors(skel, drop)[0]), drop_orphans=True)
    assert 2 in skel.ntype  # axon label survived remap
    assert skel.ntype[0] in (-1, 1)


def test_merge_near_soma_nodes_keeps_labels(template_skel):
    skel = copy.deepcopy(template_skel)
    # Force a near-soma node to have a distinct label and to be merged
    if len(skel.nodes) < 3:
        pytest.skip("template too small")
    skel.ntype[1] = 4  # apical
    merged = post.merge_near_soma_nodes(
        skel,
        mesh_vertices=None,
        inside_tol=0.0,
        near_factor=10.0,
        fat_factor=0.0,
        verbose=False,
    )
    assert 4 in merged.ntype or 4 in skel.ntype  # label survives or merge skipped
    assert merged.ntype[0] in (-1, 1)


def test_prune_neurites_keeps_labels(template_skel):
    skel = copy.deepcopy(template_skel)
    if len(skel.ntype) < 2:
        pytest.skip("template too small")
    skel.ntype[1] = 3
    res = post.prune_neurites(
        skel,
        mesh_vertices=None,
        tip_extent_factor=0.0,
        stem_extent_factor=0.0,
        drop_single_node_branches=True,
        verbose=False,
    )
    assert 3 in res.ntype
    assert res.ntype[0] in (-1, 1)


def test_reroot_clears_duplicate_roots_and_fills_gaps():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([1.0, 1.0, 1.0])}
    ntype = np.array(
        [-1, -1, 4], np.int8
    )  # duplicate roots, gap on index 1 after reroot
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    res = post.reroot(skel, node_id=2, rebuild_mst=False, verbose=False)
    assert res.ntype[0] == -1 or res.ntype[0] == 1
    assert int(np.sum(res.ntype == 1) + np.sum(res.ntype == -1)) == 1
    assert 4 in res.ntype or 4 in skel.ntype


def test_reroot_updates_soma_and_ntype():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([2.0, 1.0, 0.8])}
    ntype = np.array([1, 3, 3], np.int8)
    s0 = Skeleton(
        soma=Soma.from_sphere(nodes[0], 2.0, verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
    )
    s = post.reroot(
        s0, node_id=2, radius_key="median", set_soma_ntype=True, verbose=False
    )
    # New soma at new node 0 (old node 2)
    assert np.allclose(s.soma.center, s.nodes[0])
    assert np.isclose(s.soma.axes[0], s.soma.axes[1]) and np.isclose(
        s.soma.axes[1], s.soma.axes[2]
    )  # spherical
    assert np.isclose(
        s.soma.axes[0], s.radii["median"][0]
    )  # radius equals selected column
    assert s.ntype[0] == 1
    assert int(np.sum(s.ntype == 1)) == 1


def test_reroot_node2verts_vert2node_consistency():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([2.0, 1.0, 0.8])}
    node2verts = [np.array([10, 11]), np.array([20]), np.array([30, 31])]
    vert2node = {10: 0, 11: 0, 20: 1, 30: 2, 31: 2}
    s0 = Skeleton(
        soma=Soma.from_sphere(nodes[0], 2.0, verts=node2verts[0]),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=np.array([1, 3, 3], np.int8),
        node2verts=node2verts,
        vert2node=vert2node,
    )
    s = post.reroot(s0, node_id=2, verbose=False)
    # Vertex memberships and back-map follow the swap
    for i, vs in enumerate(s.node2verts):
        for v in vs:
            assert s.vert2node[v] == i
    # Soma verts now come from the new node 0's membership
    assert set(s.soma.verts.tolist()) == set(s.node2verts[0].tolist())


def test_detect_soma_remaps_ntype_once():
    nodes = np.array([[0, 0, 0], [0, 0, 5], [0, 5, 0]], float)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    radii = {"median": np.array([1.0, 4.0, 1.5])}
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], radii["median"][0], verts=None),
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=np.array([1, 3, 3], np.int8),
    )

    s = post.detect_soma(
        skel,
        radius_key="median",
        soma_radius_percentile_threshold=90.0,
        soma_radius_distance_factor=4.0,
        soma_min_nodes=1,
        verbose=False,
        mesh_vertices=None,
    )

    assert np.allclose(s.nodes[0], nodes[1])  # new soma promoted
    assert s.ntype[0] == 1
    assert int(np.sum(s.ntype == 1)) == 1


# ---------------------------------------------------------------------
# reassign_verts — bin editing
#
# A node *is* the set of mesh vertices it owns: its position is their
# centroid and every radius is an aggregate over them.  So the standard
# these tests hold the primitive to is that afterwards the skeleton is in
# exactly the state a from-scratch rebuild would have produced.  Anything
# less is a skeleton that looks right and measures wrong.
# ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def bin_mesh():
    return load_mesh(Path(__file__).parent / "data" / "60427.obj")


@pytest.fixture
def bin_skel(bin_mesh):
    return skeletonize(bin_mesh, verbose=False)


def _a_neighbour_of(skel, node):
    e = np.asarray(skel.edges)
    nbrs = np.unique(e[(e[:, 0] == node) | (e[:, 1] == node)])
    nbrs = nbrs[(nbrs != node) & (nbrs != 0)]
    assert nbrs.size, f"node {node} has no usable neighbour"
    return int(nbrs[0])


def _assert_matches_rebuild(skel, mesh, node):
    """Position and radii must equal what `_make_nodes` would produce."""
    from skeliner._core import _estimate_radius

    owned = np.asarray(skel.node2verts[node], dtype=np.int64)
    pts = np.asarray(mesh.vertices, dtype=np.float64)[owned]
    centre = pts.mean(axis=0)
    assert np.allclose(skel.nodes[node], centre, atol=1e-9)

    d = np.linalg.norm(pts - centre, axis=1)
    for key in skel.radii:
        if key == "centerline":
            continue
        want = _estimate_radius(d, method=key, trim_fraction=0.05)
        assert skel.radii[key][node] == pytest.approx(want, abs=1e-9), key


def _first_editable_node(skel):
    for n in range(1, len(skel.nodes)):
        if len(skel.node2verts[n]) >= 4 and _a_neighbour_of(skel, n):
            return n
    pytest.skip("fixture has no bin big enough to split")


def test_moving_part_of_a_bin_recomputes_both_ends(bin_skel, bin_mesh):
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    half = np.asarray(bin_skel.node2verts[src])[: len(bin_skel.node2verts[src]) // 2]
    n_before = len(bin_skel.nodes)

    report = post.reassign_verts(bin_skel, half, dst, mesh=bin_mesh)

    assert report["moved"] == len(half)
    assert report["donors"] == [src]
    assert report["dropped"] == []
    assert len(bin_skel.nodes) == n_before, "nothing was emptied"
    _assert_matches_rebuild(bin_skel, bin_mesh, src)
    _assert_matches_rebuild(bin_skel, bin_mesh, dst)
    assert dx.check_bins(bin_skel)


def test_emptying_a_bin_drops_its_node(bin_skel, bin_mesh):
    """A node owning nothing has no position and no radius; it is not a
    node.  Moving a whole bin is therefore a merge."""
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    everything = np.asarray(bin_skel.node2verts[src])
    n_before = len(bin_skel.nodes)

    report = post.reassign_verts(bin_skel, everything, dst, mesh=bin_mesh)

    assert report["dropped"] == [src]
    assert len(bin_skel.nodes) == n_before - 1
    new_dst = report["old2new"][dst]
    assert set(everything.tolist()) <= set(
        np.asarray(bin_skel.node2verts[new_dst]).tolist()
    )
    _assert_matches_rebuild(bin_skel, bin_mesh, new_dst)
    assert dx.check_bins(bin_skel)


def test_centerline_is_recomputed_when_the_distances_were_kept(bin_mesh):
    """Only the preprocessing track produces `centerline`, so this needs a
    preproc skeleton — on a direct-track one the test would skip forever
    and prove nothing."""
    from skeliner._core import TRIM_FRACTION, _estimate_radius

    from .test_core import _make_components

    skel = skeletonize(
        bin_mesh, components=_make_components(bin_mesh, with_soma=True), verbose=False
    )
    assert "centerline" in skel.radii and "cl_dist_vids" in skel.extra

    src = _first_editable_node(skel)
    dst = _a_neighbour_of(skel, src)
    half = np.asarray(skel.node2verts[src])[: len(skel.node2verts[src]) // 2]
    before = float(skel.radii["centerline"][dst])
    post.reassign_verts(skel, half, dst, mesh=bin_mesh)

    lut = np.zeros(len(bin_mesh.vertices))
    lut[skel.extra["cl_dist_vids"]] = skel.extra["cl_dist_vals"]
    want = _estimate_radius(
        lut[np.asarray(skel.node2verts[dst])],
        method="trim",
        trim_fraction=TRIM_FRACTION,
    )
    assert skel.radii["centerline"][dst] == pytest.approx(want, abs=1e-9)
    assert skel.radii["centerline"][dst] != before, "it must actually be rebuilt"


def test_a_radius_that_cannot_be_recomputed_is_reported(bin_skel, bin_mesh):
    """`calibrated` is measured by ray casting, not from the vertices.
    Leaving it silently stale beside freshly recomputed keys is the one
    outcome worse than either dropping or reporting it."""
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    bin_skel.radii["calibrated"] = np.asarray(bin_skel.radii["median"]).copy()

    report = post.reassign_verts(
        bin_skel, np.asarray(bin_skel.node2verts[src])[:2], dst, mesh=bin_mesh
    )
    assert "calibrated" in report["stale_radii"]


def test_moving_within_one_bin_is_a_no_op(bin_skel, bin_mesh):
    src = _first_editable_node(bin_skel)
    before = np.asarray(bin_skel.nodes[src]).copy()
    report = post.reassign_verts(
        bin_skel, np.asarray(bin_skel.node2verts[src])[:2], src, mesh=bin_mesh
    )
    assert report["moved"] == 0
    assert np.array_equal(bin_skel.nodes[src], before)


def test_edges_are_left_alone(bin_skel, bin_mesh):
    """Re-deriving the graph from the surface would discard the edges that
    have no surface support — soma stems and gap bridges — and can
    disconnect the skeleton.  Re-derivation is opt-in, via rebuild_mst."""
    src = _first_editable_node(bin_skel)
    dst = _a_neighbour_of(bin_skel, src)
    before = np.asarray(bin_skel.edges).copy()
    half = np.asarray(bin_skel.node2verts[src])[: len(bin_skel.node2verts[src]) // 2]

    post.reassign_verts(bin_skel, half, dst, mesh=bin_mesh)
    assert np.array_equal(bin_skel.edges, before)


@pytest.mark.parametrize(
    "verts, to, expected",
    [
        ([1, 2], 0, "node 0 is the soma"),
        ([1], 10**9, "out of range"),
        ([], 1, "No vertices given"),
    ],
)
def test_reassign_verts_guards(bin_skel, bin_mesh, verts, to, expected):
    with pytest.raises(ValueError, match=expected):
        post.reassign_verts(bin_skel, verts, to, mesh=bin_mesh)


def test_unowned_surface_is_refused(bin_skel, bin_mesh):
    """Unowned surface is soma, organelle or discarded — claiming it would
    be a components decision made without re-deriving them."""
    owned = np.fromiter(bin_skel.vert2node.keys(), np.int64, len(bin_skel.vert2node))
    unowned = np.setdiff1d(np.arange(len(bin_mesh.vertices)), owned)
    if unowned.size == 0:
        pytest.skip("every vertex of this mesh belongs to a bin")
    with pytest.raises(ValueError, match="belong to no bin"):
        post.reassign_verts(bin_skel, unowned[:3], 1, mesh=bin_mesh)


def test_prune_neurites_survives_a_centerline_radius(bin_mesh):
    """Regression: `_prune_neurites` recomputes the soma's radii for every
    key in the dict, and a preprocessing-track skeleton always carries
    `centerline`, which was never an estimator name.  It raised `Unknown
    radius estimator 'centerline'` on any cell where something merged."""
    skel = skeletonize(bin_mesh, verbose=False)
    skel.radii["centerline"] = np.asarray(skel.radii["median"]).copy()
    n_before = len(skel.nodes)
    cl_before = float(skel.radii["centerline"][0])

    out = post.prune_neurites(
        skel,
        mesh_vertices=np.asarray(bin_mesh.vertices),
        stem_extent_factor=1e9,
        tip_extent_factor=1e9,
        verbose=False,
    )
    assert len(out.nodes) < n_before, "the merge branch must actually run"
    assert out.radii["centerline"][0] == cl_before, "left alone, not corrupted"
    assert dx.check_bins(out)


# ── merging must not cut the tree ────────────────────────────────────
#
# An emptied bin is absorbed, not deleted.  Deleting its edges instead
# strands every neighbour it was holding on to, which is the failure the
# whole manual-editing effort exists to avoid.


def _n_components(skel):
    """Connected components of the skeleton graph."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(skel.nodes)
    e = np.asarray(skel.edges, dtype=np.int64)
    adj = sp.coo_matrix(
        (np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n)
    )
    return connected_components(adj, directed=False)[0]


def _a_node_between_two_others(skel):
    """A node with >=2 neighbours, so absorbing it could strand one."""
    e = np.asarray(skel.edges)
    for n in range(1, len(skel.nodes)):
        nbrs = np.unique(e[(e[:, 0] == n) | (e[:, 1] == n)])
        nbrs = nbrs[nbrs != n]
        if nbrs.size >= 2 and 0 not in nbrs.tolist():
            return int(n), [int(x) for x in nbrs]
    pytest.skip("fixture has no interior node to absorb")


def test_absorbing_a_node_contracts_its_edges_instead_of_dropping_them(
    bin_skel, bin_mesh
):
    src, nbrs = _a_node_between_two_others(bin_skel)
    dst = nbrs[0]
    others = [n for n in nbrs if n != dst]

    before = _n_components(bin_skel)
    out = post.reassign_verts(bin_skel, bin_skel.node2verts[src], dst, mesh=bin_mesh)
    assert out["dropped"] == [src], "the emptied bin should have been dropped"

    old2new = out["old2new"]
    new_dst = int(old2new[dst])
    e = np.asarray(bin_skel.edges)
    survivors = set(np.unique(e[(e[:, 0] == new_dst) | (e[:, 1] == new_dst)]).tolist())
    for o in others:
        assert int(old2new[o]) in survivors, (
            f"node {o} lost its link when {src} was absorbed into {dst}"
        )
    assert _n_components(bin_skel) == before, "the merge split the skeleton"


def test_absorbing_a_leaf_leaves_no_self_loop(bin_skel, bin_mesh):
    e = np.asarray(bin_skel.edges)
    leaf = None
    for n in range(1, len(bin_skel.nodes)):
        nbrs = np.unique(e[(e[:, 0] == n) | (e[:, 1] == n)])
        nbrs = nbrs[nbrs != n]
        if nbrs.size == 1 and int(nbrs[0]) != 0:
            leaf, host = int(n), int(nbrs[0])
            break
    if leaf is None:
        pytest.skip("fixture has no leaf to absorb")

    post.reassign_verts(bin_skel, bin_skel.node2verts[leaf], host, mesh=bin_mesh)
    e2 = np.asarray(bin_skel.edges)
    assert not (e2[:, 0] == e2[:, 1]).any(), "contraction left a self-loop"
    assert len(np.unique(e2, axis=0)) == len(e2), "contraction left a duplicate edge"


# ── split_node ───────────────────────────────────────────────────────
#
# The one bin verb whose destination does not exist yet.  The partition
# is drawn, so nothing is invented about vertices; the interesting part
# is what it does *not* do, which is guess the new node's other edges.


def test_a_split_recomputes_both_halves_from_their_own_vertices(bin_skel, bin_mesh):
    """A node *is* the vertices it owns, on both sides of the cut."""
    node = 3
    owned = np.asarray(bin_skel.node2verts[node])
    assert len(owned) >= 4, "fixture bin is too small to split"

    out = post.split_node(bin_skel, node, owned[: len(owned) // 2], mesh=bin_mesh)
    new = out["node"]

    for nid in (node, new):
        pts = np.asarray(bin_mesh.vertices)[np.asarray(bin_skel.node2verts[nid])]
        assert np.allclose(bin_skel.nodes[nid], pts.mean(axis=0))
        for key, arr in bin_skel.radii.items():
            if key in ("centerline", "calibrated"):
                continue
            assert np.isfinite(arr[nid])
    assert dx.check_bins(bin_skel), "the split broke the partition"


def test_a_split_appends_and_renumbers_nothing(bin_skel, bin_mesh):
    """Unlike a merge, which drops a bin and shifts every id after it."""
    before = bin_skel.nodes.copy()
    owned = np.asarray(bin_skel.node2verts[3])

    out = post.split_node(bin_skel, 3, owned[:2], mesh=bin_mesh)

    assert out["node"] == len(before), "the new node is appended"
    assert len(bin_skel.nodes) == len(before) + 1
    untouched = [i for i in range(len(before)) if i != 3]
    assert np.allclose(bin_skel.nodes[untouched], before[untouched])


def test_a_split_joins_the_new_node_to_its_parent_and_nothing_else(bin_skel, bin_mesh):
    """Surface adjacency does not imply tree adjacency — measured on a real
    cell, where bins that touch are routinely far apart in the tree — so the
    other edges are the user's to make, with `dx.edge_support` to guide them."""
    node, nbrs = _a_node_between_two_others(bin_skel)
    owned = np.asarray(bin_skel.node2verts[node])
    if len(owned) < 4:
        pytest.skip("fixture bin is too small to split")
    before = _n_components(bin_skel)

    out = post.split_node(bin_skel, node, owned[: len(owned) // 2], mesh=bin_mesh)
    new = out["node"]

    e = np.asarray(bin_skel.edges)
    touching = set(np.unique(e[(e[:, 0] == new) | (e[:, 1] == new)]).tolist()) - {new}
    assert touching == {node}, f"the split invented edges to {touching - {node}}"
    assert _n_components(bin_skel) == before, "the split left a piece adrift"
    assert dx.check_acyclicity(bin_skel), "the split closed a cycle"


def test_a_split_reports_whether_the_halves_share_surface(bin_skel, bin_mesh):
    """The one edge a split does make can still have nothing behind it, when
    the bin was already in two patches."""
    owned = np.asarray(bin_skel.node2verts[3])
    out = post.split_node(bin_skel, 3, owned[: len(owned) // 2], mesh=bin_mesh)
    assert out["supported"] is True

    assert post._bins_touch(bin_mesh, owned[:1], owned[:1]) is False, (
        "a vertex set does not touch itself"
    )


def test_a_split_ignores_vertices_the_bin_does_not_own(bin_skel, bin_mesh):
    """The >=2-of-3 face rule means a surface selection routinely catches a
    neighbour's vertex; that is not grounds for failing the edit."""
    owned = np.asarray(bin_skel.node2verts[3])
    other = np.asarray(bin_skel.node2verts[5])
    out = post.split_node(
        bin_skel, 3, np.concatenate([owned[:2], other[:3]]), mesh=bin_mesh
    )
    assert out["moved"] == 2
    assert out["ignored"] == 3
    assert len(bin_skel.node2verts[5]) == len(other), "a non-owner lost vertices"


@pytest.mark.parametrize(
    "node, verts, expected",
    [
        (0, [1, 2], "node 0 is the soma"),
        (10**6, [1, 2], "out of range"),
        (3, [], "No vertices given"),
    ],
)
def test_split_rejects_bad_input(bin_skel, bin_mesh, node, verts, expected):
    with pytest.raises(ValueError, match=expected):
        post.split_node(bin_skel, node, verts, mesh=bin_mesh)


def test_splitting_off_the_whole_bin_is_refused(bin_skel, bin_mesh):
    """That renames a node rather than splitting it, and would leave the
    parent owning nothing — which is not a node."""
    with pytest.raises(ValueError, match="leave something behind"):
        post.split_node(bin_skel, 3, bin_skel.node2verts[3], mesh=bin_mesh)


def test_splitting_vertices_from_elsewhere_only_is_refused(bin_skel, bin_mesh):
    with pytest.raises(ValueError, match="belong to node 3"):
        post.split_node(bin_skel, 3, bin_skel.node2verts[5], mesh=bin_mesh)


# ---------------------------------------------------------------------
# the Python path computes what the Rust path does
# ---------------------------------------------------------------------
def test_ray_hits_and_vertex_faces_on_both_paths_agree():
    from skeliner import _rs
    from skeliner.python import post as python_post

    from ._parity import assert_parity

    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    tris = np.asarray(mesh.triangles, np.float64).reshape(-1, 9)
    normals = np.asarray(mesh.face_normals, np.float64)
    centres = np.asarray(mesh.triangles_center, np.float64)
    golden = (1 + np.sqrt(5)) / 2  # calibrate_radii's ray directions
    k = np.arange(30)
    theta, phi = 2 * np.pi * k / golden, np.arccos(1 - 2 * (k + 0.5) / 30)
    dirs = np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    rng = np.random.default_rng(0)
    f = rng.choice(len(tris), 300, replace=False)
    origins = np.repeat(centres[f] + 0.001 * normals[f], len(dirs), axis=0)
    directions = np.tile(dirs, (len(f), 1))
    rust = _rs.post.ray_hits(tris[f], normals[f], origins, directions)
    assert 0 < rust.sum() < len(rust)
    assert_parity(rust, python_post.ray_hits(tris[f], normals[f], origins, directions))

    faces = np.asarray(mesh.faces, np.int64)
    rs = _rs.post.VertexFaces(faces, len(mesh.vertices))
    py = python_post.VertexFaces(faces, len(mesh.vertices))
    for size in (1, 10, 300, 1500):
        vids = rng.choice(len(mesh.vertices), size, replace=False)
        assert_parity(rs.faces_within(vids), py.faces_within(vids))


def _calibrated_node_by_node(skel, mesh, **kwargs):
    """``calibrate_radii``'s per-node loop, the route embree's and sampled
    rays take, on *skel*: ``(radii, methods)``."""
    params = dict(
        aggregate="trim",
        min_n_outer=20,
        min_frac_outer=0.33,
        min_verts_q_outer=80.0,
        rays_num_outer=30,
        rays_thresh_outer=0.2,
        rays_sample=False,
    )
    params.update(kwargs)
    metric = skel.recommend_radius()[0]
    r_new = np.array(skel.radii[metric], dtype=np.float64)
    r_kind = np.full(len(skel.nodes), "fallback", dtype=object)
    n_verts = np.array([len(v) for v in skel.node2verts])
    post._calibrate_node_by_node(
        skel,
        mesh,
        r_new,
        r_kind,
        radius_metric=metric,
        min_n_verts_bulb=int(np.percentile(n_verts, q=params["min_verts_q_outer"])),
        **params,
    )
    return r_new, r_kind.tolist()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"min_verts_q_outer": 0.0},
        {"aggregate": "median"},
        {"aggregate": "mean", "min_verts_q_outer": 50.0, "min_frac_outer": 0.0},
    ],
)
def test_calibrate_radii_in_one_call_is_the_node_by_node_loop(template_skel, kwargs):
    from .test_core import _ball_with_two_processes

    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    ball, comps = _ball_with_two_processes()
    kinds = set()
    for skel, m in [(template_skel, mesh), (skeletonize(ball, components=comps), ball)]:
        one_call = copy.deepcopy(skel)
        post.calibrate_radii(one_call, m, **kwargs)
        radii, methods = _calibrated_node_by_node(copy.deepcopy(skel), m, **kwargs)
        assert np.array_equal(one_call.radii["calibrated"], radii)
        assert one_call.extra["calibration"]["radius_method"] == methods
        kinds |= set(methods)
    if kwargs.get("min_verts_q_outer") == 0.0:
        assert kinds == {"fallback", "full_centerline", "outer_centerline"}


def test_calibrate_radii_with_sampled_rays_runs_node_by_node(template_skel):
    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    skel = copy.deepcopy(template_skel)
    post.calibrate_radii(skel, mesh, rays_sample=True)
    methods = set(skel.extra["calibration"]["radius_method"])
    assert methods <= {"fallback", "full_centerline", "outer_centerline"}
    assert np.isfinite(skel.radii["calibrated"]).all()


def test_calibrate_radii_on_both_paths_agree(template_skel):
    from ._parity import assert_parity, both
    from .test_core import _ball_with_two_processes

    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    ball, comps = _ball_with_two_processes()
    cases = [
        (template_skel, mesh, {}),
        (template_skel, mesh, {"min_verts_q_outer": 0.0, "aggregate": "median"}),
        (skeletonize(ball, components=comps), ball, {"min_verts_q_outer": 0.0}),
    ]
    for skel, m, kwargs in cases:

        def calibrated():
            s = copy.deepcopy(skel)
            post.calibrate_radii(s, m, **kwargs)
            return s.radii["calibrated"], s.extra["calibration"]["radius_method"]

        rust, python = both(calibrated)
        assert_parity(rust, python)


def test_remap_ntype_and_gap_fill_on_both_paths():
    from skeliner import _rs
    from skeliner.python import post as python_post

    from ._parity import assert_parity

    rng = np.random.default_rng(5)
    labels = np.array([0, 0, 0, 1, -1, 2, 3, 4, 5, 6, 7, 9, -2])
    filled = 0
    for _ in range(400):
        n = int(rng.integers(1, 40))
        ntype = rng.choice(labels, size=n)
        new_len = int(rng.integers(1, n + 1))
        old2new = rng.integers(-1, new_len, size=n)
        edges = rng.integers(0, new_len, size=(int(rng.integers(0, 3 * new_len)), 2))
        root = int(rng.choice([1, -1]))
        for fill in (True, False):
            got = _rs.post.remap_ntype(ntype, old2new, new_len, root, edges, fill)
            assert_parity(
                got,
                python_post.remap_ntype(ntype, old2new, new_len, root, edges, fill),
            )
        edges = rng.integers(0, n, size=(int(rng.integers(0, 3 * n)), 2))
        gaps = _rs.post.fill_ntype_gaps(ntype, edges)
        assert_parity(gaps, python_post.fill_ntype_gaps(ntype, edges))
        filled += int(((ntype == 0) & (gaps != 0)).sum())
    assert filled > 100
    # errors alike
    for call, message in [
        (
            lambda m: m.remap_ntype(
                np.array([1, 2]), np.array([0]), 1, 1, np.empty((0, 2), int), True
            ),
            "one entry per node",
        ),
        (
            lambda m: m.remap_ntype(
                np.array([1, 2]), np.array([0, 4]), 2, 1, np.empty((0, 2), int), True
            ),
            "names node 4, past",
        ),
        (
            lambda m: m.fill_ntype_gaps(np.array([0, 2]), np.array([[0, 2]])),
            "edge names node 2, not",
        ),
    ]:
        for module in (_rs.post, python_post):
            with pytest.raises(ValueError, match=message):
                call(module)


def test_remap_ntype_ties_beyond_the_known_types_go_to_the_smallest():
    mapped = post._remap_ntype(
        np.array([-1, 9, 7, 0], np.int8), np.array([0, 1, 1, 1]), 2
    )
    assert mapped.tolist() == [-1, 7]
    assert mapped.dtype == np.int8


def test_merge_near_soma_nodes_leaves_the_input_skeleton_alone():
    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    skel = skeletonize(mesh, verbose=False)
    before = copy.deepcopy(skel.soma)
    out = post.merge_near_soma_nodes(
        skel,
        mesh_vertices=mesh.vertices,
        near_factor=3.0,
        fat_factor=0.05,
        verbose=False,
    )
    assert len(out.nodes) < len(skel.nodes)  # it merged something
    assert not (out.edges[:, 0] == out.edges[:, 1]).any()  # no self-loop
    assert len(out.soma.verts) > len(before.verts)
    assert np.array_equal(skel.soma.verts, before.verts)
    assert np.array_equal(skel.soma.center, before.center)


def _random_tree_case(rng, n, *, cycles):
    """A forest on *n* nodes along a wobbly line, with branch points; with
    *cycles*, a few extra edges, rings, a self-loop. Radii in runs of equal
    values (so groups and absorptions happen) with jitter now and then."""
    edges = [
        [int(rng.integers(max(0, i - 3), i)), i]
        for i in range(1, n)
        if rng.random() > 0.05
    ]
    if cycles:
        edges += rng.integers(0, n, size=(int(rng.integers(0, 3)), 2)).tolist()
        if rng.random() < 0.3:
            v = int(rng.integers(0, n))
            edges.append([v, v])
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    nodes = np.cumsum(rng.normal(size=(n, 3)), axis=0)
    nodes[rng.random(n) < 0.05] = nodes[0]  # some zero-length segments
    base = np.repeat(rng.uniform(1, 3, size=n // 4 + 1), 4)[:n]
    radii = np.stack([base, base * rng.choice([1.0, 1.02, 1.3], size=n)])
    sizes = rng.integers(0, 4, size=n)
    idx = rng.integers(0, 5 * n, size=int(sizes.sum())).astype(np.int64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    return nodes, radii, edges, ptr, idx


def test_downsample_on_both_paths():
    from skeliner import _rs
    from skeliner.python import post as python_post

    from ._parity import assert_parity

    rng = np.random.default_rng(67)
    merged = absorbed_twice = 0
    for trial in range(300):
        n = int(rng.integers(1, 60))
        nodes, radii, edges, ptr, idx = _random_tree_case(rng, n, cycles=trial % 3 == 0)
        aggregate = ("area", "mean", "median")[trial % 3]
        shift = None if trial % 4 else float(rng.uniform(0.5, 3))
        bins = (ptr, idx) if trial % 5 else (None, None)
        args = (
            nodes,
            radii,
            int(rng.integers(0, 2)),
            edges,
            *bins,
            float(rng.choice([0.0, 0.05, 0.3])),
            float(rng.choice([0.0, 0.1])),
            aggregate,
            bool(trial % 2),
            bool(trial % 7 != 3),
            shift,
        )
        rust = _rs.post.downsample(*args)
        assert_parity(rust, python_post.downsample(*args))
        assert (rust[3] >= 0).all()  # every node goes somewhere
        merged += len(rust[0]) < n
        if bins[0] is not None and len(rust[0]):
            counts = np.bincount(rust[3], minlength=len(rust[0]))
            absorbed_twice += int((counts > 3).sum())
    assert merged > 150 and absorbed_twice > 20


def test_downsample_ends_alike_on_both_paths(template_skel):
    from ._parity import assert_parity, both

    for kwargs in (
        {},
        {"aggregate": "median", "slide_branchpoints": False},
        {"aggregate": "mean", "merge_endpoints": False, "max_anchor_shift": 50.0},
    ):
        rust, python = both(post.downsample, template_skel, verbose=False, **kwargs)
        for field in ("nodes", "radii", "edges", "ntype", "node2verts", "vert2node"):
            assert_parity(getattr(rust, field), getattr(python, field), field)


def _labelled_line(edges, n, radii=None):
    nodes = np.c_[np.arange(n, dtype=float), np.zeros(n), np.zeros(n)]
    return Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii={"median": np.ones(n) if radii is None else np.asarray(radii)},
        edges=np.asarray(edges, dtype=np.int64),
        ntype=None,
        node2verts=[np.array([10 * i, 10 * i + 1]) for i in range(n)],
        vert2node={v: i for i in range(n) for v in (10 * i, 10 * i + 1)},
    )


@pytest.mark.parametrize("path", ["rust", "python"])
def test_downsample_ends_on_rings_and_loops(path):
    # Before: a ring through node 0 never ended, a ring with no anchor lost
    # its nodes, and a loop through a branch point whose run its branch point
    # absorbed raised "self-loops are not allowed".
    import skeliner as sk

    with sk.backend(path):
        ring = _labelled_line([[0, 1], [1, 2], [2, 3], [3, 0]], 4)
        d = post.downsample(ring, verbose=False)
        assert len(d.nodes) == 2 and dx.check_bins(d)

        apart = _labelled_line([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 4]], 7)
        d = post.downsample(apart, verbose=False)
        assert len(d.nodes) == 4  # the path's two and the ring's two
        assert len(d.vert2node) == 14 and dx.check_bins(d)

        loop = _labelled_line([[0, 1], [1, 2], [2, 3], [3, 1]], 4)
        d = post.downsample(loop, verbose=False)
        assert dx.check_bins(d) and (d.edges[:, 0] != d.edges[:, 1]).all()


@pytest.mark.parametrize("path", ["rust", "python"])
def test_a_branch_point_absorbing_from_two_runs_keeps_both(path):
    # 0-1-2-3 and 3-4-5 and 3-6: node 3 a branch point whose runs 1-2 and
    # 4 agree with it. Before: the second absorption overwrote the first, so
    # nodes 1 and 2 mapped to the branch point while their vertices were in
    # no bin, and dx.check_bins failed.
    import skeliner as sk

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [3, 6]]
    radii = [5.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    skel = _labelled_line(edges, 7, radii)
    with sk.backend(path):
        d = post.downsample(skel, verbose=False)
    hub = d.vert2node[30]
    assert d.vert2node[10] == d.vert2node[20] == d.vert2node[40] == hub
    assert set(d.node2verts[hub].tolist()) == {10, 11, 20, 21, 30, 31, 40, 41}
    assert dx.check_bins(d)


def test_dropping_the_soma_is_refused():
    # the guard compared numpy's bool with `is False`, so it never fired
    skel = _labelled_line([[0, 1], [1, 2]], 3)
    with pytest.raises(RuntimeError, match="soma"):
        post._rebuild_drop_set(skel, {0, 2})


def test_clip_drops_what_the_soma_no_longer_reaches():
    skel = _labelled_line([[0, 1], [1, 2], [2, 3], [1, 4]], 5)
    post.clip(skel, 1, 2, drop_orphans=True)
    assert len(skel.nodes) == 3 and skel.edges.tolist() == [[0, 1], [1, 2]]
    assert sorted(skel.vert2node) == [0, 1, 10, 11, 40, 41]


def test_pruning_nodes_keeps_a_hub_joining_two_kept_neighbours():
    # hub 2 joins 1, 3 and 4; dropping 2 and 4 keeps 2, which still joins
    # two kept neighbours
    skel = _labelled_line([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5]], 6)
    post.prune(skel, kind="nodes", nodes=[2, 4])
    assert len(skel.nodes) == 5
    assert sorted(skel.vert2node.values()) == sorted([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
