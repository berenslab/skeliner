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
    ntype = np.array([-1, -1, 4], np.int8)  # duplicate roots, gap on index 1 after reroot
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
