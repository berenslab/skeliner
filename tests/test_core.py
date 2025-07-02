"""
Core pipeline smoke-test.

Runs `skeletonize()` on the reference mesh and checks a handful of
topological / numerical invariants so that regressions blow up early.
"""
from pathlib import Path

import numpy as np
import pytest

from skeliner import skeletonize
from skeliner.io import load_mesh


def _assert_skeleton_valid(skel):
    """A few cheap invariants that *must* hold for every result."""
    # ----- basic shapes -------------------------------------------------
    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.nodes.shape[0] > 0, "no nodes produced"
    assert (skel.r > 0).all(), "non-positive radii"

    # ----- edges are sorted & acyclic (forest) --------------------------
    assert (skel.edges[:, 0] < skel.edges[:, 1]).all(), "edges not sorted"

    g = skel._igraph()
    n_components = len(g.components())
    # for every forest: |E| = |V| − #components
    assert skel.edges.shape[0] == skel.nodes.shape[0] - n_components

    # ----- soma is node 0 ----------------------------------------------
    assert skel.ntype[0] == 1, "node 0 not marked as soma"


@pytest.fixture(scope="session")
def reference_mesh():
    data_dir = Path(__file__).parent / "data"
    mesh_path = data_dir / "60427.obj"
    return load_mesh(mesh_path)


def test_skeletonize_smoke(reference_mesh):
    skel = skeletonize(reference_mesh, verbose=False)
    _assert_skeleton_valid(skel)
