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


def test_distance_point_queries(skel):
    unit = skel.meta.get("unit", "nm")
    soma = skel.nodes[0]
    # distance to a node should be zero irrespective of units
    assert dx.distance(skel, soma, point_unit=unit) == pytest.approx(0.0, abs=1e-9)

    # take edge midpoint and move it away along a perpendicular direction
    u, v = map(int, skel.edges[0])
    edge_vec = skel.nodes[v] - skel.nodes[u]
    mid = 0.5 * (skel.nodes[u] + skel.nodes[v])

    # robust perpendicular
    perp = None
    for axis in np.eye(3):
        candidate = np.cross(edge_vec, axis)
        norm = np.linalg.norm(candidate)
        if norm > 1e-9:
            perp = candidate / norm
            break
    if perp is None:  # degenerate edge, fall back to arbitrary axis
        perp = np.array([1.0, 0.0, 0.0])

    offset_nm = perp * 500.0  # 500 nm away from the edge
    point_nm = mid + offset_nm

    radius_key = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_key], dtype=float)

    def brute_centerline(point_nm_space: np.ndarray) -> float:
        """Distance to the centreline (no radii)."""
        d_nodes = np.linalg.norm(skel.nodes - point_nm_space, axis=1).min()
        d_edges = np.inf
        for a, b in skel.edges:
            d_edges = min(
                d_edges,
                dx._point_segment_distance(point_nm_space, skel.nodes[a], skel.nodes[b]),
            )
        return min(d_nodes, d_edges)

    def brute_surface(point_nm_space: np.ndarray) -> float:
        """Distance to the capsule envelope (clamped to zero inside)."""
        d_nodes = float(
            np.min(np.linalg.norm(skel.nodes - point_nm_space, axis=1) - radii)
        )
        d_edges = np.inf
        for a, b in skel.edges:
            d_edges = min(
                d_edges,
                dx._point_segment_capsule_distance(
                    point_nm_space,
                    skel.nodes[a],
                    skel.nodes[b],
                    radii[a],
                    radii[b],
                ),
            )
        return max(min(d_nodes, d_edges), 0.0)

    expected_center_nm = brute_centerline(point_nm)
    expected_surface_nm = brute_surface(point_nm)

    # --- centreline mode -------------------------------------------------
    d_center_nm = dx.distance(skel, point_nm, point_unit="nm", mode="centerline")
    assert d_center_nm == pytest.approx(expected_center_nm, rel=1e-6)

    point_um = point_nm * 1e-3
    d_center_um = dx.distance(skel, point_um, point_unit="um", mode="centerline")
    assert d_center_um == pytest.approx(expected_center_nm * 1e-3, rel=1e-6)

    # --- surface mode ----------------------------------------------------
    d_surface_nm = dx.distance(skel, point_nm, point_unit="nm", mode="surface")
    assert d_surface_nm == pytest.approx(expected_surface_nm, rel=1e-6)

    point_um_surface = dx.distance(skel, point_um, point_unit="um", mode="surface")
    assert point_um_surface == pytest.approx(expected_surface_nm * 1e-3, rel=1e-6)

    # vectorised query mixes modes
    arr_nm = np.vstack([point_nm, mid])
    distances_surface = dx.distance(skel, arr_nm, point_unit="nm", mode="surface")
    assert distances_surface.shape == (2,)
    assert distances_surface[0] == pytest.approx(expected_surface_nm, rel=1e-6)
    assert distances_surface[1] == pytest.approx(0.0, abs=1e-9)

    distances_center = dx.distance(skel, arr_nm, point_unit="nm", mode="centerline")
    assert distances_center.shape == (2,)
    assert distances_center[0] == pytest.approx(expected_center_nm, rel=1e-6)
    assert distances_center[1] == pytest.approx(0.0, abs=1e-9)
