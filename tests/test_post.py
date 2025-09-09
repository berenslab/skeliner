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
from skeliner.core import Skeleton, Soma
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
    assert skel.ntype[0] == 1
    changed = np.where(skel.ntype == 4)[0]
    assert set(changed) == {base}


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
