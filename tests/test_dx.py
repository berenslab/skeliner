"""
Smoke-check every public diagnostic helper in `skeliner.dx`.

No hard-coded biology – we just compare against ground-truth values
computed on-the-fly with igraph so the test is independent of the mesh
content.
"""
import copy
from pathlib import Path

import igraph as ig
import numpy as np
import pytest

from skeliner import dx, skeletonize
from skeliner.io import load_mesh


# ---------------------------------------------------------------------
# shared fixture: skeleton of the reference mesh
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def skel():
    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    return skeletonize(mesh, verbose=False)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _igraph(skel):
    return skel._igraph()


# ---------------------------------------------------------------------
# individual tests
# ---------------------------------------------------------------------
def test_connectivity_and_acyclicity(skel):
    assert dx.connectivity(skel)
    assert dx.acyclicity(skel) is True
    # acyclicity(..., return_cycles=True) must return a boolean when acyclic
    assert dx.acyclicity(skel, return_cycles=True) is True


def test_degree_and_neighbors_match_igraph(skel):
    g = _igraph(skel)
    degrees_ref = g.degree()
    # vector query
    assert np.array_equal([dx.degree(skel, node_id=n) for n in range(len(skel.nodes))], degrees_ref)
    # scalar query + neighbors
    nid = 0  # arbitrary but deterministic
    assert dx.degree(skel, nid) == degrees_ref[nid]
    assert set(dx.neighbors(skel, nid)) == set(g.neighbors(nid))


def test_nodes_of_degree(skel):
    g = _igraph(skel)
    deg = np.asarray(g.degree())
    for k in (0, 1, 2, 3):      # 0 included on purpose – should be empty
        expected = {int(i) for i in np.where(deg == k)[0] if i != 0}
        got = set(dx.nodes_of_degree(skel, k))
        assert got == expected


def test_branches_and_twigs_lengths(skel):
    # We do not assume k actually exists – just assert path lengths.
    for k in (1, 2, 3):
        for path in dx.branches_of_length(skel, k):
            assert len(path) == k
        for twig in dx.twigs_of_length(skel, k):
            assert len(twig) == k


def test_suspicious_tips_are_leaves(skel):
    tips = dx.suspicious_tips(skel)  # may be empty
    if not tips:
        return
    g = _igraph(skel)
    deg = np.asarray(g.degree())
    assert all(deg[t] == 1 and t != 0 for t in tips)
