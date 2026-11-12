"""
Tests for skeliner.pre – mesh preprocessing utilities.

Uses synthetic meshes to exercise detection and removal functions
without depending on large biological data.
"""

from collections import Counter

import numpy as np
import pytest
import trimesh

import skeliner as sk
from skeliner import pre
from skeliner.dataclass import (
    Discarded,
    Mesh,
    MeshComponents,
    Neurites,
    Organelles,
    Soma,
)
from skeliner.pre import _non_degenerate

from ._parity import assert_parity, both


#: Tests of a deprecated function, which warns on every call
deprecated = pytest.mark.filterwarnings(
    r"ignore:sk\.pre\.\w+\(\) is deprecated:DeprecationWarning"
)


def _live_faces(mesh):
    """Count non-degenerate faces (degenerate = removed by _rebuild_mesh)."""
    return int(_non_degenerate(mesh.faces).sum())


# ── Mesh builders ────────────────────────────────────────────────────


def _icosphere(subdivisions=2, radius=100.0):
    """Watertight icosphere centered at origin."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def _open_mesh(n_remove=3):
    """Icosphere with a few faces removed → creates a hole."""
    mesh = _icosphere()
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[:n_remove] = False
    mesh = mesh.submesh([np.where(keep)[0]], append=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_with_island(n_island_faces=2):
    """Main icosphere + a small disconnected island.

    When n_island_faces=1, the island is a single triangle.
    When n_island_faces=2, two triangles sharing an edge (one component).
    """
    main = _icosphere()
    n_v = len(main.vertices)

    offset = np.array([500.0, 0.0, 0.0])
    v0 = offset + np.array([0.0, 0.0, 0.0])
    v1 = offset + np.array([1.0, 0.0, 0.0])
    v2 = offset + np.array([0.5, 1.0, 0.0])
    island_verts = [v0, v1, v2]
    island_tri = [[n_v, n_v + 1, n_v + 2]]

    if n_island_faces >= 2:
        # Add a second triangle sharing edge v0-v1
        v3 = offset + np.array([0.5, -1.0, 0.0])
        island_verts.append(v3)
        island_tri.append([n_v, n_v + 1, n_v + 3])

    # Add more if needed (each shares v1-v2)
    for i in range(2, n_island_faces):
        vi = offset + np.array([1.0 + i * 0.5, 0.5, 0.0])
        base = n_v + len(island_verts)
        island_verts.append(vi)
        island_tri.append([n_v + 1, n_v + 2, base])

    verts = np.vstack([main.vertices, np.array(island_verts)])
    faces = np.vstack([main.faces, np.array(island_tri)])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _mesh_with_fin():
    """Icosphere with a dangling fin face attached by one edge.

    A fin has 2+ boundary edges (only 1 of its 3 edges shared with
    another face).
    """
    main = _icosphere()
    n_v = len(main.vertices)

    # Pick an edge from face 0 and hang a triangle off it
    f0 = main.faces[0]
    v0, v1 = int(f0[0]), int(f0[1])

    # New vertex sticking outward
    mid = (main.vertices[v0] + main.vertices[v1]) / 2
    normal = mid / np.linalg.norm(mid)  # radial direction
    new_pt = mid + normal * 50.0

    verts = np.vstack([main.vertices, new_pt.reshape(1, 3)])
    fin_face = np.array([[v0, v1, n_v]])
    faces = np.vstack([main.faces, fin_face])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _two_cylinders(separation=200.0):
    """Two elongated cylinders separated by a gap — for gap detection.

    Elongated shapes (PCA ratio > 5) avoid being filtered as organelle-like
    blobs by find_disconnected. Separation > 1000 avoids distance filter.
    """
    c1 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
    c2 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
    # Shift c2 along x beyond the organelle proximity filter (1000nm)
    c2.vertices[:, 0] += 200.0 + separation

    n_v = len(c1.vertices)
    verts = np.vstack([c1.vertices, c2.vertices])
    faces = np.vstack([c1.faces, c2.faces + n_v])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _cylinder_at(radius, height, z_center, sections):
    """Cylinder along z, centred at ``z_center``."""
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    c.vertices[:, 2] += z_center
    return c


def _subdivided_tube(sections=16, height=1000.0):
    """A tube with several face rings along z.

    ``_cylinder_at`` puts the whole side wall in one ring of triangles,
    so there is no band across the tube to cut.
    """
    return _cylinder_at(50.0, height, 0.0, sections).subdivide().subdivide()


def _open_tube(z_center, open_at, sections=16):
    """``_subdivided_tube``'s tube at *z_center*, its end at z = *open_at*
    left open: a neurite the segmentation cut (``4 * sections`` edges of
    one face)."""
    t = _cylinder_at(50.0, 1000.0, z_center, sections).subdivide().subdivide()
    on_end = np.isclose(t.vertices[t.faces][:, :, 2], open_at).all(axis=1)
    t = trimesh.Trimesh(t.vertices, t.faces[~on_end], process=False)
    t.remove_unreferenced_vertices()
    return t


def _edge_uses(mesh):
    """How many faces each edge of *mesh* has (its non-degenerate faces)."""
    f = np.asarray(mesh.faces)
    f = np.sort(f[_non_degenerate(f)], axis=1)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
    return np.unique(e, axis=0, return_counts=True)[1]


def _band(mesh, lo, hi):
    """Faces whose centroid z falls in [lo, hi] — an annulus around a tube."""
    z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    return {int(i) for i in np.nonzero((z >= lo) & (z <= hi))[0]}


def _combine(parts):
    verts, faces, n = [], [], 0
    for p in parts:
        verts.append(p.vertices)
        faces.append(p.faces + n)
        n += len(p.vertices)
    return trimesh.Trimesh(
        vertices=np.vstack(verts), faces=np.vstack(faces), process=False
    )


def _main_debris_piece():
    """main — tiny debris — a real piece, collinear along z.

    Distances: debris↔main 101, debris↔piece 43, piece↔main 200.
    Bboxes:    main ~2000, piece ~603, debris ~61.

    Prim's attaches the debris first, because 101 is the cheapest edge
    in the graph — and the size-aware cap, which keys on the *smaller*
    side's bbox, refuses it (101 > 61).  Nothing retries a refused tree
    edge, so the debris is left with no bridge even though its 43 nm
    edge to the piece is well inside the same cap.
    """
    return _combine(
        [
            _cylinder_at(20, 2000, 0, 48),  # main,   z -1000..1000
            _cylinder_at(4, 60, 1130, 8),  # debris, z  1100..1160
            _cylinder_at(20, 600, 1500, 32),  # piece,  z  1200..1800
        ]
    )


def _main_relay_piece():
    """main — small relay — a real piece, collinear along z.

    Distances: relay↔main 100, piece↔relay 100, piece↔main 300.
    The relay is 32 faces: two bridges land on it, and ``remove_gaps``
    peels a tip patch off each end of every bridge, so it is erased and
    the chain routed through it breaks.
    """
    return _combine(
        [
            _cylinder_at(20, 2000, 0, 48),  # main,  z -1000..1000
            _cylinder_at(10, 100, 1150, 8),  # relay, z  1100..1200 (32f)
            _cylinder_at(20, 600, 1600, 32),  # piece, z  1300..1900
        ]
    )


def _sphere_with_internal():
    """Large sphere with a small inverted sphere inside.

    The inner sphere has normals pointing inward (toward center),
    simulating an isolated organelle.
    """
    outer = _icosphere(subdivisions=2, radius=100.0)
    inner = _icosphere(subdivisions=1, radius=20.0)

    # Flip inner normals by reversing face winding
    inner.faces = inner.faces[:, ::-1]

    n_v = len(outer.vertices)
    verts = np.vstack([outer.vertices, inner.vertices])
    faces = np.vstack([outer.faces, inner.faces + n_v])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# ── _face_edge_components ────────────────────────────────────────────


class TestFaceEdgeComponents:
    def test_single_component(self):
        mesh = _icosphere()
        labels, main = pre._face_edge_components(mesh)
        assert labels.shape == (len(mesh.faces),)
        # All faces in one component
        assert np.unique(labels).size == 1
        assert main == 0

    def test_two_components(self):
        mesh = _two_cylinders()
        labels, main = pre._face_edge_components(mesh)
        unique = np.unique(labels)
        assert len(unique) == 2
        # Main component is the largest (both are equal here, so either is fine)
        main_size = (labels == main).sum()
        for cid in unique:
            assert (labels == cid).sum() <= main_size

    def test_with_island(self):
        mesh = _mesh_with_island(n_island_faces=2)
        labels, main = pre._face_edge_components(mesh)
        unique = np.unique(labels)
        # Main icosphere + 1 disconnected 2-face island
        assert len(unique) == 2


# ── find_holes / fill_holes ──────────────────────────────────────────


@deprecated
class TestFindHoles:
    def test_watertight_has_no_holes(self):
        mesh = _icosphere()
        holes = pre.find_holes(mesh)
        assert holes == []

    def test_open_mesh_has_hole(self):
        mesh = _open_mesh(n_remove=3)
        holes = pre.find_holes(mesh)
        assert len(holes) >= 1
        # Each hole is a loop of at least 3 vertices
        for loop in holes:
            assert len(loop) >= 3

    def test_hole_vertices_are_valid(self):
        mesh = _open_mesh(n_remove=1)
        holes = pre.find_holes(mesh)
        assert len(holes) >= 1
        for loop in holes:
            for vid in loop:
                assert 0 <= vid < len(mesh.vertices)


def _tube_with_a_pocket(around=63, rings=30, radius=10.0):
    """A tube round the z axis, edges about 1 long, faces wound outwards,
    open at both ends; on its side a pocket: the faces in a window of it
    are organelle, the vertices only they use pushed 3 in."""
    j, i = np.meshgrid(np.arange(rings), np.arange(around), indexing="ij")
    t = 2 * np.pi * (i + 0.5 * (j % 2)) / around
    verts = np.c_[radius * np.cos(t).ravel(), radius * np.sin(t).ravel(), j.ravel()]
    faces = []
    for jj in range(rings - 1):
        for ii in range(around):
            a, b = jj * around + ii, jj * around + (ii + 1) % around
            c, d = a + around, b + around
            if jj % 2 == 0:
                faces += [[a, b, c], [b, d, c]]
            else:
                faces += [[a, b, d], [a, d, c]]
    faces = np.array(faces)
    centre = verts[faces].mean(axis=1)
    pocket = (np.abs(np.arctan2(centre[:, 1], centre[:, 0])) < 0.35) & (
        np.abs(centre[:, 2] - rings / 2) < 3
    )
    inner = np.setdiff1d(faces[pocket], faces[~pocket])
    verts[inner, :2] *= 1 - 3 / radius
    z = np.zeros(len(faces), dtype=bool)
    org = Organelles(pocket=pocket, isolated=z.copy(), expanded=z.copy())
    return Mesh(verts, faces), org


def _open_edges(faces) -> int:
    """Edges of *faces* that only one of them has."""
    e = np.sort(np.asarray(faces)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, n = np.unique(e, axis=0, return_counts=True)
    return int((n == 1).sum())


class TestFillHoles:
    def test_a_pocket_mouth_is_patched_on_the_tube(self):
        mesh, org = _tube_with_a_pocket()
        n = len(mesh.faces)
        out = pre.fill_holes(mesh, organelles=org)
        # the input's ids name what they did
        assert np.array_equal(out.faces[:n], mesh.faces)
        new = out.faces[n:]
        assert len(new) > 0
        # closed but for the tube's two ends, 63 edges round each
        assert _open_edges(np.r_[mesh.faces[~org.pocket], new]) == 2 * 63
        # on the tube, not across it: a flat lid's middle would lie 0.6 in
        added = out.vertices[len(mesh.vertices) :]
        assert len(added) > 0
        assert np.abs(np.hypot(added[:, 0], added[:, 1]) - 10.0).max() < 0.2

    def test_mesh_stats_keep_up(self):
        mesh, org = _tube_with_a_pocket()
        stats = pre.compute_mesh_stats(mesh)
        out = pre.fill_holes(mesh, organelles=org, mesh_stats=stats)
        assert len(stats.outward_dots) == len(out.faces)

    def test_the_python_path_runs_it_in_rust(self):
        mesh, org = _tube_with_a_pocket()
        rust = pre.fill_holes(mesh, organelles=org)
        with sk.backend("python"):
            python = pre.fill_holes(mesh, organelles=org)
        assert np.array_equal(rust.faces, python.faces)
        assert np.array_equal(rust.vertices, python.vertices)

    def test_a_hole_the_soma_touches_stays_open(self):
        mesh, org = _tube_with_a_pocket()
        rim = np.intersect1d(mesh.faces[org.pocket], mesh.faces[~org.pocket])
        soma = Soma.from_sphere(np.zeros(3), 1.0, verts=rim[:1])
        out = pre.fill_holes(mesh, organelles=org, soma=soma)
        assert len(out.faces) == len(mesh.faces)
        assert np.array_equal(out.vertices, mesh.vertices)

    def test_preprocess_fills_unless_told_not_to(self, monkeypatch):
        mesh, org = _tube_with_a_pocket()
        monkeypatch.setattr(pre, "find_organelles", lambda *_a, **_k: org)
        filled, _ = pre.preprocess(mesh)
        kept, _ = pre.preprocess(mesh, fill=False)
        assert len(filled.faces) > len(kept.faces) == len(mesh.faces)


# ── find_fragments / remove_fragments ────────────────────────────────


@deprecated
class TestFindFragments:
    def test_clean_mesh_no_fragments(self):
        mesh = _icosphere()
        mask = pre.find_fragments(mesh)
        assert mask.shape == (len(mesh.faces),)
        assert mask.sum() == 0

    def test_island_detected(self):
        mesh = _mesh_with_island(n_island_faces=2)
        mask = pre.find_fragments(mesh, min_faces=3)
        # The 2-face island should be detected (< min_faces=3)
        assert mask.sum() == 2

    def test_min_faces_affects_island_detection(self):
        """min_faces controls island threshold (fins are always detected)."""
        mesh = _mesh_with_island(n_island_faces=2)
        # With min_faces=3, island is flagged as island (2 < 3)
        mask_strict = pre.find_fragments(mesh, min_faces=3)
        # With min_faces=1, island not flagged as island (2 >= 1)
        # but may still be flagged as fin (2 boundary edges each)
        mask_lax = pre.find_fragments(mesh, min_faces=1)
        # Both should detect the same count here since they're also fins
        assert mask_strict.sum() == mask_lax.sum() == 2

    def test_fin_detected(self):
        mesh = _mesh_with_fin()
        mask = pre.find_fragments(mesh)
        # The dangling fin face should be detected
        assert mask.sum() >= 1


@deprecated
class TestRemoveFragments:
    def test_removes_islands(self):
        mesh = _mesh_with_island(n_island_faces=2)
        n_before = _live_faces(mesh)
        clean = pre.remove_fragments(mesh, min_faces=3)
        assert _live_faces(clean) < n_before
        assert _live_faces(clean) == n_before - 2

    def test_removes_fins(self):
        mesh = _mesh_with_fin()
        n_before = _live_faces(mesh)
        clean = pre.remove_fragments(mesh)
        assert _live_faces(clean) < n_before

    def test_clean_mesh_unchanged(self):
        mesh = _icosphere()
        clean = pre.remove_fragments(mesh)
        assert _live_faces(clean) == _live_faces(mesh)

    def test_precomputed_mask(self):
        mesh = _mesh_with_island(n_island_faces=2)
        mask = pre.find_fragments(mesh, min_faces=3)
        clean = pre.remove_fragments(mesh, fragments=mask)
        assert _live_faces(clean) == _live_faces(mesh) - 2


# ── remove_islands / remove_fins ─────────────────────────────────────


@deprecated
class TestRemoveIslands:
    def test_removes_small_components(self):
        mesh = _mesh_with_island(n_island_faces=2)
        clean = pre.remove_islands(mesh, min_faces=3)
        assert _live_faces(clean) == _live_faces(mesh) - 2

    def test_keeps_large_enough_components(self):
        mesh = _mesh_with_island(n_island_faces=2)
        clean = pre.remove_islands(mesh, min_faces=2)
        assert _live_faces(clean) == _live_faces(mesh)

    def test_noop_on_clean_mesh(self):
        mesh = _icosphere()
        clean = pre.remove_islands(mesh)
        assert _live_faces(clean) == _live_faces(mesh)


@deprecated
class TestRemoveFins:
    def test_removes_dangling_faces(self):
        mesh = _mesh_with_fin()
        clean = pre.remove_fins(mesh)
        assert _live_faces(clean) < _live_faces(mesh)

    def test_noop_on_clean_mesh(self):
        mesh = _icosphere()
        clean = pre.remove_fins(mesh)
        assert _live_faces(clean) == _live_faces(mesh)


# ── find_disconnected ────────────────────────────────────────────────


class TestFindDisconnected:
    def test_single_component_returns_empty(self):
        mesh = _icosphere()
        comps = pre.find_disconnected(mesh)
        assert comps == []

    def test_two_cylinders_detected(self):
        mesh = _two_cylinders(separation=1500.0)
        comps = pre.find_disconnected(mesh)
        assert len(comps) == 1  # one non-main component
        # The component should have the face count of one cylinder
        cyl_faces = len(
            trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16).faces
        )
        assert len(comps[0]) == cyl_faces

    def test_min_faces_filter(self):
        mesh = _mesh_with_island(n_island_faces=2)
        # Island has 2 faces, min_faces=100 → excluded
        comps = pre.find_disconnected(mesh)
        assert comps == []

    def test_sorted_largest_first(self):
        """Three elongated components of different sizes → sorted largest-first."""
        c1 = trimesh.creation.cylinder(radius=10.0, height=300.0, sections=16)
        c2 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
        c3 = trimesh.creation.cylinder(radius=10.0, height=100.0, sections=8)

        c2.vertices[:, 0] += 2000.0
        c3.vertices[:, 0] += 4000.0

        n1 = len(c1.vertices)
        n2 = n1 + len(c2.vertices)
        verts = np.vstack([c1.vertices, c2.vertices, c3.vertices])
        faces = np.vstack([c1.faces, c2.faces + n1, c3.faces + n2])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        comps = pre.find_disconnected(mesh)
        assert len(comps) == 2
        assert len(comps[0]) >= len(comps[1])

    def test_the_main_component_is_the_somas_not_the_largest(self):
        """A soma cut from its arbor by the segmentation: the arbor is the
        larger piece, and still the one bridged to the soma."""
        ball = trimesh.creation.icosphere(subdivisions=3, radius=3000.0)
        tube = trimesh.creation.capsule(height=20000.0, radius=400.0, count=[32, 32])
        tube.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )
        tube.vertices[:, 0] += 3000.0 + 300.0 + 10400.0
        n = len(ball.vertices)
        mesh = trimesh.Trimesh(
            np.vstack([ball.vertices, tube.vertices]),
            np.vstack([ball.faces, tube.faces + n]),
            process=False,
        )
        assert len(tube.faces) > len(ball.faces)
        soma = Soma.fit(ball.vertices, verts=np.arange(n))
        labels, largest = pre._face_edge_components(mesh)
        assert largest == labels[len(ball.faces)]
        assert pre._main_component(mesh, labels, largest, soma) == labels[0]
        assert pre._main_component(mesh, labels, largest, None) == largest

        comps = pre.find_disconnected(mesh, soma=soma)
        assert [len(c) for c in comps] == [len(tube.faces)]
        gaps = pre.find_gaps(mesh, soma=soma, **NO_CAP)
        assert [sorted(g[3:5]) for g in gaps] == [[-1, 0]]


# ── find_gaps / remove_gaps ──────────────────────────────────────────

# Disable the size-aware bridge cap so the gap-finding mechanics (MST,
# tip selection, sorting, stitching) can be tested on the synthetic
# cylinder fixtures, whose small pieces + large separations are exactly
# the "floating fragment" case the cap is designed to drop.  The cap
# itself is covered by test_size_cap_drops_far_fragment.
NO_CAP = dict(max_bridge_ratio=float("inf"), max_bridge_dist=float("inf"))

# Pin the bridge cap to the size-ratio term alone, so the fixtures below
# depend only on their own bboxes and not on the mesh's median edge:
#   bridge_cap(a, b) = min(bbox_a, bbox_b)
PINNED_CAP = dict(min_bridge_gap=0.0, max_bridge_dist=5000.0)


class TestFindGaps:
    def test_is_the_same_on_both_paths(self):
        from pathlib import Path

        from skeliner.io import load_mesh

        reference = load_mesh(Path(__file__).parent / "data" / "60427.obj")
        cyl = _two_cylinders(separation=300.0)
        first = np.arange(len(cyl.faces) // 2)
        # the first cylinder's faces nearest the gap, posing as a fusion
        near = first[np.argsort(-cyl.triangles_center[first, 0])[:6]].tolist()
        cases = [
            (reference, {}),
            (_two_cylinders(separation=1500.0), NO_CAP),
            (cyl, NO_CAP),
            (cyl, {"fusions": [near], **NO_CAP}),
            (cyl, {"fusions": [near], "fusion_clearance": 1e4, **NO_CAP}),
            (_main_debris_piece(), PINNED_CAP),
            (_main_relay_piece(), {"relay_min_faces": 0, **PINNED_CAP}),
            (_main_relay_piece(), PINNED_CAP),
        ]
        n_gaps = 0
        for mesh, kwargs in cases:
            rust, python = both(pre.find_gaps, mesh, **kwargs)
            assert_parity(rust, python)
            n_gaps += len(rust)
        assert n_gaps > 10

    def test_single_component_no_gaps(self):
        mesh = _icosphere()
        gaps = pre.find_gaps(mesh)
        assert gaps == []

    def test_two_cylinders_one_gap(self):
        mesh = _two_cylinders(separation=1500.0)
        gaps = pre.find_gaps(mesh, **NO_CAP)
        assert len(gaps) == 1
        faces_a, faces_b, dist, *_ = gaps[0]
        assert len(faces_a) > 0
        assert len(faces_b) > 0
        assert dist > 0

    def test_gap_distance_positive(self):
        mesh = _two_cylinders(separation=1500.0)
        gaps = pre.find_gaps(mesh, **NO_CAP)
        assert len(gaps) == 1
        _, _, dist, *_ = gaps[0]
        assert dist > 100  # should be roughly the separation

    def test_sorted_by_distance(self):
        """Three cylinders with different gaps → sorted by distance."""
        c1 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
        c2 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)
        c3 = trimesh.creation.cylinder(radius=10.0, height=200.0, sections=16)

        c2.vertices[:, 0] += 2000.0  # gap ~1800 from c1
        c3.vertices[:, 0] += 5000.0  # gap ~2800 from c2

        n1, n2 = len(c1.vertices), len(c1.vertices) + len(c2.vertices)
        verts = np.vstack([c1.vertices, c2.vertices, c3.vertices])
        faces = np.vstack([c1.faces, c2.faces + n1, c3.faces + n2])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        gaps = pre.find_gaps(mesh, **NO_CAP)
        assert len(gaps) >= 2
        for i in range(len(gaps) - 1):
            assert gaps[i][2] <= gaps[i + 1][2]

    def test_size_cap_drops_far_fragment(self):
        # Small pieces far apart: gap (~1680) dwarfs each piece's bbox
        # (~202), ratio ≈ 8 — a floating fragment, not a real break.
        mesh = _two_cylinders(separation=1500.0)
        # default caps drop it
        assert pre.find_gaps(mesh) == []
        # both caps disabled → the gap is found
        assert len(pre.find_gaps(mesh, **NO_CAP)) == 1
        # size-ratio gate (absolute ceiling lifted)
        assert (
            len(
                pre.find_gaps(mesh, max_bridge_ratio=10.0, max_bridge_dist=float("inf"))
            )
            == 1
        )
        assert (
            pre.find_gaps(mesh, max_bridge_ratio=1.0, max_bridge_dist=float("inf"))
            == []
        )
        # absolute ceiling gate (ratio lifted)
        assert pre.find_gaps(mesh, max_bridge_ratio=float("inf")) == []
        # small-gap floor forces a bridge regardless of ratio
        assert (
            len(
                pre.find_gaps(mesh, min_bridge_gap=5000.0, max_bridge_dist=float("inf"))
            )
            == 1
        )

    def test_stranded_piece_is_rebridged(self):
        """A refused MST edge must not leave a component with no bridge.

        The cheapest edge in the graph is the one Prim's picks for the
        debris, and the cap refuses it.  Without the repair pass nothing
        retries, and the debris is silently discarded downstream despite
        having an acceptable edge to the piece.
        """
        mesh = _main_debris_piece()
        gaps = pre.find_gaps(mesh, **PINNED_CAP)
        pairs = {frozenset(g[3:5]) for g in gaps}
        # disc0 = the 128f piece, disc1 = the 32f debris, -1 = main
        assert frozenset((-1, 0)) in pairs, "piece must reach main"
        assert frozenset((0, 1)) in pairs, "debris must be re-bridged"
        assert {round(g[2]) for g in gaps} == {200, 40}

    def test_chain_does_not_route_through_a_small_relay(self):
        """A component too small to survive two bridges is not a relay.

        ``remove_gaps`` peels a tip patch off each end of every bridge,
        so a fragment used as an intermediate hop is erased and the chain
        through it breaks.  The piece must bridge to main directly even
        though the hop through the relay is shorter (100 vs 300).

        The kiss penalty is pinned off.  It is not what this test is
        about, and the relay is short enough (100 long, radius 10) that a
        ``kiss_radius`` neighbourhood spans its whole length and reads as
        two-sided, so its flank score sits on a knife edge and moves with
        the triangulation trimesh happens to generate.  That flipped the
        control assertion on CI while passing locally.
        """
        mesh = _main_relay_piece()

        # relay_min_faces=0 restores the old behaviour: chain through it
        old = pre.find_gaps(mesh, relay_min_faces=0, kiss_penalty=0.0, **PINNED_CAP)
        assert {(int(g[3]), int(g[4])) for g in old} == {(-1, 1), (0, 1)}

        # default: the piece bridges straight to main
        new = pre.find_gaps(mesh, kiss_penalty=0.0, **PINNED_CAP)
        assert {(int(g[3]), int(g[4])) for g in new} == {(-1, 1), (-1, 0)}
        assert any(round(g[2]) == 300 for g in new)


class TestRemoveGaps:
    def test_is_the_same_on_both_paths(self):
        added = 0
        cases = [
            (_two_cylinders(separation=1500.0), NO_CAP),
            (_two_cylinders(separation=300.0), NO_CAP),
            (_main_debris_piece(), PINNED_CAP),
            (_main_relay_piece(), {"relay_min_faces": 0, **PINNED_CAP}),
        ]
        for mesh, kwargs in cases:
            gaps = pre.find_gaps(mesh, **kwargs)
            rust, python = both(pre.remove_gaps, mesh, gaps=gaps)
            assert_parity(np.asarray(rust.vertices), np.asarray(python.vertices))
            assert_parity(np.asarray(rust.faces), np.asarray(python.faces))
            added += len(rust.faces) - len(mesh.faces)
        assert added > 0

    def test_noop_single_component(self):
        mesh = _icosphere()
        result = pre.remove_gaps(mesh)
        assert len(result.faces) == len(mesh.faces)

    def test_bridges_two_cylinders(self):
        mesh = _two_cylinders(separation=1500.0)
        labels_before, _ = pre._face_edge_components(mesh)
        n_comps_before = len(set(labels_before) - {-2})
        assert n_comps_before == 2

        gaps = pre.find_gaps(mesh, **NO_CAP)
        result = pre.remove_gaps(mesh, gaps=gaps)
        labels_after, _ = pre._face_edge_components(result)
        n_comps_after = len(set(labels_after) - {-2})
        # After bridging, should be a single connected component
        assert n_comps_after < n_comps_before

    def test_precomputed_gaps(self):
        mesh = _two_cylinders(separation=1500.0)
        gaps = pre.find_gaps(mesh, **NO_CAP)
        result = pre.remove_gaps(mesh, gaps=gaps)
        labels, _ = pre._face_edge_components(result)
        n_comps = len(set(labels) - {-2})
        # Should still work with precomputed gaps
        assert n_comps < 2 or _live_faces(result) != _live_faces(mesh)

    def test_a_loop_is_never_paired_with_itself(self):
        assert "same rim" in pre._validate_loop_pair([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        # pieces glued at the rim share a few vertices; that alone is fine
        assert pre._validate_loop_pair([1, 2, 3, 4, 5], [4, 5, 6, 7, 8]) is None

    def test_tips_side_by_side_are_filled_not_tubed_on_both_paths(self):
        # Two tips on one surface whose rims share vertices: they are glued
        # along those vertices (a fusion join).  A tube from rim to rim
        # would fold over the glue; removing both and closing the one hole
        # they leave keeps the surface manifold.
        from skeliner import _rs

        mesh = _subdivided_tube()
        topo = _rs.mesh.Topology(np.asarray(mesh.faces, np.int64), len(mesh.vertices))
        centre_z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
        fa = [int(np.argmin(np.abs(centre_z)))]
        sel_a, rim_a = pre._expand_tip_to_good_rim(fa, topo, None)
        # a second tip three faces away grows a patch beside the first
        adj = pre._face_adjacency(mesh)
        ring, seen = set(fa), set(fa)
        for _ in range(3):
            ring = {g for f in ring for g in adj[f]} - seen
            seen |= ring
        for f in sorted(ring):
            sel_b, rim_b = pre._expand_tip_to_good_rim([f], topo, None)
            if len(set(rim_a) & set(rim_b)) >= 2 and not sel_a & sel_b:
                fb = [f]
                break
        else:
            pytest.fail("no side-by-side tip on the fixture")
        out, python = both(pre.remove_gaps, mesh, gaps=[(fa, fb, 1.0, -1, 0)])
        assert_parity(np.asarray(out.vertices), np.asarray(python.vertices))
        assert_parity(np.asarray(out.faces), np.asarray(python.faces))

        def edge_uses(m):
            f = np.asarray(m.faces)
            f = np.sort(f[_non_degenerate(f)], axis=1)
            e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
            return np.unique(e, axis=0, return_counts=True)[1]

        assert _live_faces(out) != _live_faces(mesh)
        assert len(out.vertices) == len(mesh.vertices) + 1  # the fan's centre
        assert (edge_uses(out) <= 2).all()
        assert (edge_uses(out) == 1).sum() == (edge_uses(mesh) == 1).sum()

    def test_tips_on_open_ends_are_bridged_round_the_ends_on_both_paths(self):
        # Two tubes end to end, the facing ends open (a neurite the
        # segmentation cut): a tip there takes faces off the end, and its
        # border with the other faces runs open from hole to hole.  Its rim
        # takes in the rest of the end, so the bridge joins the two ends
        # and closes both.
        from skeliner import _rs

        mesh = _combine([_open_tube(-600.0, -100.0), _open_tube(600.0, 100.0)])
        assert (_edge_uses(mesh) == 1).sum() == 2 * 64
        gaps = pre.find_gaps(mesh, **NO_CAP)
        assert len(gaps) == 1
        topo = _rs.mesh.Topology(np.asarray(mesh.faces, np.int64), len(mesh.vertices))
        open_at = {
            int(v)
            for v in np.flatnonzero(np.isclose(np.abs(mesh.vertices[:, 2]), 100.0))
        }
        for tip in gaps[0][:2]:
            assert not topo.border_loops(tip)
            assert len(topo.rim_loops(tip)) == 1
            # no ring is peeled: the tip as found, the end round it its rim
            sel, rim = pre._expand_tip_to_good_rim(tip, topo, None)
            assert sel == set(tip)
            assert len(set(rim) & open_at) > 32
        out, python = both(pre.remove_gaps, mesh, gaps=gaps)
        assert_parity(np.asarray(out.vertices), np.asarray(python.vertices))
        assert_parity(np.asarray(out.faces), np.asarray(python.faces))
        labels, _ = pre._face_edge_components(out)
        assert len(set(labels.tolist()) - {-2}) == 1
        assert (_edge_uses(out) <= 2).all()
        assert (_edge_uses(out) == 1).sum() == 0

    def test_a_rim_round_an_open_end_pairs_with_a_capped_tip_on_both_paths(self):
        # An open end facing a capped tube: the open end's rim is the whole
        # end, the cap's — grown to its first rim of six vertices — more
        # than five times shorter than a pair takes.  The capped side is
        # grown on until its rim reaches a fifth of the end's, and the pair
        # is bridged.
        from skeliner import _rs
        from skeliner.python import pre as python_pre

        capped = _cylinder_at(50.0, 1000.0, 600.0, 16).subdivide().subdivide()
        mesh = _combine([_open_tube(-600.0, -100.0, sections=32), capped])
        gaps = pre.find_gaps(mesh, **NO_CAP)
        assert len(gaps) == 1
        faces = np.asarray(mesh.faces, np.int64)
        topo = _rs.mesh.Topology(faces, len(mesh.vertices))
        tips = gaps[0][:2]
        peeled = [pre._expand_tip_to_good_rim(t, topo, None) for t in tips]
        short, long_ = sorted(len(rim) for _, rim in peeled)
        assert long_ > 5 * short
        as_dict = pre._edge_to_faces(mesh)
        adj = pre._face_adjacency(mesh, as_dict)
        pair = python_pre.paired(tips, peeled, as_dict, adj, faces=faces)
        assert min(len(rim) for _, rim in pair) * 5 >= long_
        assert pre._validate_loop_pair(pair[0][1], pair[1][1]) is None
        out, python = both(pre.remove_gaps, mesh, gaps=gaps)
        assert_parity(np.asarray(out.vertices), np.asarray(python.vertices))
        assert_parity(np.asarray(out.faces), np.asarray(python.faces))
        labels, _ = pre._face_edge_components(out)
        assert len(set(labels.tolist()) - {-2}) == 1
        assert (_edge_uses(out) <= 2).all()

    def test_a_bridge_left_with_nothing_to_meet_yields_on_both_paths(self):
        # Two bridges landing side by side on main's flank, each decided on
        # its own: the long tube's patch on main takes every face along the
        # stub's loop there, so the stub's tube would end on nothing.  The
        # bridge winning back more (the long tube's) is built; the stub's
        # is not, the stub left apart rather than hung on a dangling tube.
        main = _subdivided_tube()
        turn = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        long_ = _cylinder_at(30.0, 600.0, 0.0, 16).subdivide()
        long_.apply_transform(turn)
        long_.vertices[:, 0] += 400.0
        stub = _cylinder_at(10.0, 100.0, 0.0, 8).subdivide()
        stub.apply_transform(turn)
        stub.vertices[:, 0] += 110.0
        stub.vertices[:, 1] += 45.0
        mesh = _combine([main, long_, stub])
        n_main, n_long = len(main.faces), len(long_.faces)
        centres = mesh.triangles_center

        def near(point, lo, hi, k):
            d = np.linalg.norm(centres[lo:hi] - np.asarray(point), axis=1)
            return (lo + np.argsort(d, kind="stable")[:k]).tolist()

        # the stub's tip on main one face; the long tube's the three rings
        # of faces round it, which hold the stub's patch and the faces
        # round that
        stub_tip = near((50.0, 0.0, 0.0), 0, n_main, 1)
        adj = pre._face_adjacency(mesh)
        long_tip = set(stub_tip)
        for _ in range(3):
            long_tip |= {g for f in long_tip for g in adj[f]}
        gaps = [
            (
                sorted(long_tip),
                near((100.0, 0.0, 0.0), n_main, n_main + n_long, 8),
                50.0,
                -1,
                0,
            ),
            (
                stub_tip,
                near((60.0, 45.0, 0.0), n_main + n_long, len(mesh.faces), 6),
                20.0,
                -1,
                1,
            ),
        ]
        out, python = both(pre.remove_gaps, mesh, gaps=gaps)
        assert_parity(np.asarray(out.vertices), np.asarray(python.vertices))
        assert_parity(np.asarray(out.faces), np.asarray(python.faces))
        labels, _ = pre._face_edge_components(out)
        stub_faces = np.arange(n_main + n_long, len(mesh.faces))
        # the long tube joined to main, the stub whole and apart
        assert labels[0] == labels[n_main]
        assert len(set(labels[stub_faces].tolist())) == 1
        assert labels[0] != labels[stub_faces[0]]
        assert len(set(labels[labels >= 0].tolist())) == 2

    def test_with_the_fusions_a_bridge_wins_back_what_their_walls_cut_off(
        self, monkeypatch
    ):
        # Two halves of one tube glued only by a band of fusion faces: as
        # one face-edge component, a bridge between them wins back nothing;
        # split along the fusion wall, as find_gaps bridged them, it wins
        # back the smaller half — what remove_fusions would cut off.
        mesh = _subdivided_tube()
        band = sorted(_band(mesh, -250.0, 0.0))  # a whole row of faces
        assert band
        seen = []

        def spy(_mesh, _gaps, labels):
            seen.append(labels)
            return None, 0, 0, 0

        monkeypatch.setattr(pre, "_bridge_gaps", spy)
        z = mesh.triangles_center[:, 2]
        fa, fb = int(np.argmax(z)), int(np.argmin(z))
        gaps = [([fa], [fb], 1.0, -1, 0)]
        pre.remove_gaps(mesh, gaps=gaps)
        pre.remove_gaps(mesh, gaps=gaps, fusions=[band])
        unsplit, split = seen
        assert unsplit is None  # labelled in the kernel, on first need
        raw, _ = pre._face_edge_components(mesh)
        assert pre._rescue_size(raw, Counter(raw.tolist()), fa, fb) == 0
        assert (split[band] == -3).all()
        halves = Counter(split[split >= 0].tolist())
        assert len(halves) == 2
        won = pre._rescue_size(split, Counter(split.tolist()), fa, fb)
        assert won == min(halves.values()) > 0

    def test_tips_that_grow_onto_the_same_rim_are_not_stitched(self):
        # Both sides of a gap end on one rim when the pieces are still glued
        # together; stitching it to itself would fold a zero-length tube of
        # non-manifold edges into the rim.
        mesh = _two_cylinders(separation=1500.0)
        fa, _, dist, ca, _ = pre.find_gaps(mesh, **NO_CAP)[0]
        result = pre.remove_gaps(mesh, gaps=[(fa, list(fa), dist, ca, ca)])
        assert np.array_equal(result.faces, mesh.faces)
        assert len(result.vertices) == len(mesh.vertices)


class TestMergeSelectedFaces:
    def test_a_lasso_over_two_open_ends_joins_them_on_both_paths(self):
        # The viewer's merge on two tubes whose facing ends are open, half
        # of each end lassoed: a group's border with the other faces runs
        # open from hole to hole, but it takes faces off the end, so its rim
        # is the end, and the two are stitched end to end.
        mesh = _combine([_open_tube(-600.0, -100.0), _open_tube(600.0, 100.0)])
        tri = mesh.vertices[mesh.faces]
        on_end = np.isclose(np.abs(tri[:, :, 2]), 100.0).any(axis=1)
        lasso = np.flatnonzero(on_end & (tri[:, :, 0].mean(axis=1) > 0.0)).tolist()
        assert not pre._trace_border_loops(set(lasso), pre._edge_to_faces(mesh))
        out, python = both(pre.merge_selected_faces, mesh, lasso)
        assert_parity(np.asarray(out.vertices), np.asarray(python.vertices))
        assert_parity(np.asarray(out.faces), np.asarray(python.faces))
        labels, _ = pre._face_edge_components(out)
        assert len(set(labels[labels >= 0].tolist())) == 1
        assert (_edge_uses(out) <= 2).all()
        assert (_edge_uses(out) == 1).sum() == 0


# ── _sever_cost ───────────────────────────────────────────────────────


class TestSeverCost:
    """What a severing stitch costs, against what it rescues.

    ``_removal_would_sever`` says only that a removal patch is an annulus
    and so *could* cut the surface.  The cost varies by orders of
    magnitude — on 564241053 such a patch stranded 13,438 f to rescue 54,
    on 554656742 a patch of the same shape stranded 4 and would have
    rescued 4,160 — so the loop count alone cannot decide.
    """

    def _adj(self, mesh):
        return pre._face_adjacency(mesh, pre._edge_to_faces(mesh))

    def test_a_piece_crossing_the_budget_is_walked_whole(self):
        # A path of seven faces; removing the two ends leaves one piece of
        # five, over the budget and the only piece, so nothing strands.  A
        # walk that stops at the face crossing the budget without expanding
        # it returns the faces beyond as a phantom second piece.
        faces = np.array([[k, k + 1, k + 2] for k in range(7)])
        vertices = np.random.default_rng(0).random((9, 3))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        assert pre._sever_cost({0, 6}, self._adj(mesh), 2) == 0

    def test_band_across_a_tube_strands_the_shorter_side(self):
        mesh = _subdivided_tube()
        sel = _band(mesh, 100.0, 300.0)
        assert sel, "no band selected"

        cost = pre._sever_cost(sel, self._adj(mesh), len(mesh.faces))
        survivor = len(mesh.faces) - len(sel) - cost
        assert 0 < cost < survivor

    def test_cap_at_the_end_strands_nothing(self):
        """A patch at a tube's end is a cap: removing it leaves one piece."""
        mesh = _subdivided_tube()
        z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
        sel = {int(i) for i in np.nonzero(z >= z.max() - 1e-6)[0]}
        assert sel
        assert pre._sever_cost(sel, self._adj(mesh), len(mesh.faces)) == 0

    def test_patch_is_everything(self):
        mesh = _subdivided_tube()
        sel = set(range(len(mesh.faces)))
        assert pre._sever_cost(sel, self._adj(mesh), 10) == 0

    def test_budget_caps_the_answer(self):
        """Over budget the exact cost is irrelevant, only that it compares
        as larger — that comparison is the rule the guard applies."""
        mesh = _subdivided_tube()
        adj = self._adj(mesh)
        sel = _band(mesh, 100.0, 300.0)
        full = pre._sever_cost(sel, adj, len(mesh.faces))
        assert full > 2

        assert pre._sever_cost(sel, adj, full - 1) > full - 1
        assert pre._sever_cost(sel, adj, full + 5) == full


# ── _rescue_size ──────────────────────────────────────────────────────


class TestRescueSize:
    """The other half of the guard: what a stitch wins back.

    ``_sever_cost`` prices the removal, ``_rescue_size`` prices the
    bridge, and ``remove_gaps`` skips only when the first exceeds the
    second.
    """

    def test_separate_components_rescue_the_smaller(self):
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        counts = Counter(int(x) for x in labels)
        assert pre._rescue_size(labels, counts, 0, 3) == 2
        assert pre._rescue_size(labels, counts, 3, 0) == 2

    def test_same_component_is_a_fusion_join(self):
        """Both tips in one raw component: the sides are still glued, so
        connectivity cannot size the far side.  Claim nothing."""
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        counts = Counter(int(x) for x in labels)
        assert pre._rescue_size(labels, counts, 0, 2) == 0

    def test_matches_the_component_it_would_strand(self):
        """On a real two-piece mesh the rescue is the smaller piece."""
        mesh = _two_cylinders(separation=1500.0)
        labels, _ = pre._face_edge_components(mesh)
        counts = Counter(int(x) for x in labels)
        comps = sorted(set(int(x) for x in labels) - {-2})
        assert len(comps) == 2

        fa = int(np.nonzero(labels == comps[0])[0][0])
        fb = int(np.nonzero(labels == comps[1])[0][0])
        smaller = min(counts[comps[0]], counts[comps[1]])
        assert pre._rescue_size(labels, counts, fa, fb) == smaller


# ── find_nucleus_center ───────────────────────────────


class TestFindNucleusCenter:
    def test_grid_and_slab_follow_the_mesh(self):
        # a hollow 3 um ball on 450 nm edges, like hemibrain's meshes: on a
        # 200 nm grid its outline stays open and a short void near its
        # bottom wins; on the median edge's grid the whole hollow is found
        ball = trimesh.creation.icosphere(subdivisions=3, radius=3000.0)
        assert pre._median_edge(ball) > 400.0
        found = pre.find_nucleus_center(ball)
        assert np.abs(found["center"]).max() < 200.0
        assert len(found["slices"]) > 20
        fixed = pre.find_nucleus_center(ball, grid_res=200.0, z_tol=150.0)
        assert fixed is None or np.abs(fixed["center"]).max() > 1000.0

    def test_full_resolution_keeps_200_and_150(self):
        ball = trimesh.creation.icosphere(subdivisions=5, radius=3000.0)
        assert pre._median_edge(ball) < 200.0
        found = pre.find_nucleus_center(ball)
        fixed = pre.find_nucleus_center(ball, grid_res=200.0, z_tol=150.0)
        assert np.array_equal(found["center"], fixed["center"])
        assert np.array_equal(found["slices"], fixed["slices"])

    def test_a_void_standing_out_of_no_other_is_no_nucleus(self):
        # two hollow neurites 2.2 um thick lying flat, apart: a fixed 500 nm
        # passes their lumens, but neither stands out of the other's; a
        # hollow 3 um ball in place of one stands out of it
        def tube(y, z):
            t = trimesh.creation.cylinder(radius=1100.0, height=15000.0, sections=16)
            t = t.subdivide_to_size(max_edge=500.0)
            t.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
            )
            t.vertices[:, 1:] += [y, z]
            return t

        twins = trimesh.util.concatenate([tube(0.0, 0.0), tube(20000.0, 8000.0)])
        assert len(pre.find_nucleus_center(twins, min_void_r=500.0)["slices"]) > 2
        assert pre.find_nucleus_center(twins) is None
        ball = trimesh.creation.icosphere(subdivisions=4, radius=3000.0)
        body = trimesh.util.concatenate([ball, tube(20000.0, 8000.0)])
        found = pre.find_nucleus_center(body)
        assert np.abs(found["center"]).max() < 500.0

    def test_a_point_picks_the_chain_nearest_it_on_both_paths(self):
        # a hollow column broken by a gap with no vertices: two runs past
        # the void minimum, the longer below; a point on the upper one
        # picks it, a point on the lower one changes nothing
        a = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)

        def rings(z0, n):
            return np.concatenate(
                [
                    np.c_[r * np.cos(a), r * np.sin(a), np.full(len(a), z0 + 100.0 * k)]
                    for k in range(n)
                    for r in (3000.0, 3100.0)
                ]
            )

        cloud = Mesh(np.r_[rings(0.0, 30), rings(3800.0, 20)], np.empty((0, 3)))
        alone = both(pre.find_nucleus_center, cloud)
        up = both(pre.find_nucleus_center, cloud, near=[0.0, 3000.0, 4600.0])
        down = both(pre.find_nucleus_center, cloud, near=(0.0, 3000.0, 1500.0))
        for found in (*alone, *down):
            assert found["z_range"][1] < 3000.0, found["z_range"]
        for found in up:
            assert found["z_range"][0] > 3600.0, found["z_range"]
        for pair in (alone, up, down):
            assert_parity(pair[0]["slices"], pair[1]["slices"])

    def test_a_long_nucleus_is_one_body_on_both_paths(self):
        # two overlapping hollows 3.6 um apart, the left one wider below,
        # the right one above: the deepest point changes end half-way up
        # and the chain breaks there, but the upper part's void lies over
        # the lower part's — the same nucleus, no void elsewhere to beat
        a = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)

        def lobes(z0, radii):
            out = []
            centres = (-1800.0, 1800.0)
            for k in range(15):
                for w in (0.0, 100.0):
                    for side in (0, 1):
                        r, ro = radii[side] + w, radii[1 - side] + w
                        x = centres[side] + r * np.cos(a)
                        y = r * np.sin(a)
                        keep = np.hypot(x - centres[1 - side], y) >= ro
                        z = np.full(keep.sum(), z0 + 100.0 * k)
                        out.append(np.c_[x[keep], y[keep], z])
            return np.concatenate(out)

        v = np.r_[lobes(0.0, (2200.0, 1800.0)), lobes(1500.0, (1800.0, 2150.0))]
        raw = pre._z_scan(v, retry_broken=False)[0]
        # the fixture: the deepest point at either end, within a cell
        left = max(r[3] for r in raw if r[3] > 0 and r[1] < 0)
        right = max(r[3] for r in raw if r[3] > 0 and r[1] > 0)
        assert 0 < left - right <= 200.0, (left, right)
        cloud = Mesh(v, np.empty((0, 3)))
        found = both(pre.find_nucleus_center, cloud)
        for f in found:
            assert np.abs(f["center"][:2]).max() < 3600.0, f["center"]
        assert_parity(found[0]["slices"], found[1]["slices"])
        assert_parity(*both(pre._z_scan, v))


# ── find_soma ─────────────────────────────────────────


class TestFindSoma:
    def test_no_fragments_returns_none(self):
        mesh = _icosphere()
        soma = pre.find_soma(mesh)
        assert soma is None

    def test_a_soma_only_organelle_faces_hold_is_none(self):
        """Every vertex it claims is let go, and an ellipsoid holding no
        vertex is no soma."""
        ball = trimesh.creation.icosphere(subdivisions=3, radius=3000.0)
        n = len(ball.faces)
        assert len(pre.find_soma(ball).verts) > 4
        every = Organelles(
            pocket=np.ones(n, bool),
            isolated=np.zeros(n, bool),
            expanded=np.zeros(n, bool),
        )
        assert both(pre.find_soma, ball, organelles=every) == (None, None)

    def test_a_point_keeps_the_soma_it_points_at_and_no_other(self):
        """A click on the membrane, on the nucleus or beside the soma finds
        it as without the click; a soma farther from the click than it is
        long is none."""
        ball = trimesh.creation.icosphere(subdivisions=3, radius=3000.0)
        alone = pre.find_soma(ball)
        assert alone is not None, "fixture: a soma to find"
        length = 2.0 * alone.axes.max()
        for point in (ball.vertices[7], np.zeros(3), [0.9 * length, 0.0, 0.0]):
            for got in both(pre.find_soma, ball, near=point):
                assert np.array_equal(got.verts, alone.verts)
        far = [1.1 * length, 0.0, 0.0]
        assert both(pre.find_soma, ball, near=far) == (None, None)

    def test_the_rings_grow_on_the_outline_not_on_a_surface_inside_it(self):
        """A slice whose seeds lie mostly on a surface inside the membrane,
        not joined to it, as in EyeWire 1's somas: the membrane is the
        outline, and only its seeds are kept."""
        from skeliner.dataclass import Mesh

        t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.concatenate(
            [np.c_[10 * np.cos(t), 10 * np.sin(t), np.full(12, z)] for z in (0, 1)]
        )
        i = np.arange(12)
        j = (i + 1) % 12
        band = np.r_[np.c_[i, j, i + 12], np.c_[j, j + 12, i + 12]]
        g = np.linspace(-2, 2, 7)
        inside = np.array([(x, y, 0.5) for y in g for x in g])
        a = (np.arange(6)[:, None] * 7 + np.arange(6)).ravel() + 24
        patch = np.r_[np.c_[a, a + 1, a + 8], np.c_[a, a + 8, a + 7]]
        mesh = Mesh(np.r_[ring, inside], np.r_[band, patch])
        labels, _ = pre._face_edge_components(mesh)
        faces = np.asarray(mesh.faces)
        seeds = np.arange(len(mesh.vertices))
        # two components, the inner one holding most of the seeds
        assert labels[0] != labels[len(band)] and len(inside) > len(ring)
        kept = pre._outline_seeds(mesh.vertices, faces, labels, seeds)
        assert np.array_equal(kept, np.arange(24))


@pytest.mark.parametrize(
    "name, instead",
    [
        ("find_holes", None),
        ("find_fragments", None),
        ("remove_fragments", None),
        ("remove_islands", None),
        ("remove_fins", None),
        ("find_pocket_mouths", None),
        ("find_soma_via_neurite_exclusion", "find_soma"),
        ("find_soma_via_geodesic", "find_soma"),
        ("find_soma_via_z_contour", "find_soma"),
        ("find_soma_via_ring_cutoff", "find_soma"),
    ],
)
def test_a_deprecated_function_warns_at_the_line_that_calls_it(name, instead):
    pattern = rf"sk\.pre\.{name}\(\) is deprecated"
    with pytest.warns(DeprecationWarning, match=pattern) as caught:
        try:
            getattr(pre, name)(_icosphere())
        except Exception:
            pass  # what the call does is not the point
    (warning,) = [w for w in caught if name in str(w.message)]
    assert warning.filename == __file__
    if instead:
        assert f"use {instead}()" in str(warning.message)


# ── find_organelles (isolated) ───────────────────────────────────────


class TestFindIsolatedOrganelles:
    def test_clean_mesh_no_organelles(self):
        mesh = _icosphere()
        mask = pre.find_isolated_organelles(mesh, radius=50.0)
        assert mask.shape == (len(mesh.faces),)
        assert mask.sum() == 0

    def test_inner_sphere_detected(self):
        mesh = _sphere_with_internal()
        mask = pre.find_isolated_organelles(mesh, radius=50.0)
        inner_n = len(_icosphere(subdivisions=1, radius=20.0).faces)
        # Should detect the inner sphere faces
        assert mask.sum() == inner_n


class TestFindOrganelles:
    def test_returns_organelles(self):
        mesh = _icosphere()
        org = pre.find_organelles(mesh, radius=50.0)
        assert org.pocket.shape == (len(mesh.faces),)
        assert org.isolated.shape == (len(mesh.faces),)
        assert org.pocket.dtype == bool
        assert org.isolated.dtype == bool
        assert org.mask.shape == (len(mesh.faces),)

    def test_masks_are_disjoint(self):
        mesh = _sphere_with_internal()
        org = pre.find_organelles(mesh, radius=50.0)
        overlap = org.pocket & org.isolated
        assert overlap.sum() == 0


# ── remove_organelles ────────────────────────────────────────────────


class TestRemoveOrganelles:
    def test_clean_mesh_unchanged(self):
        mesh = _icosphere()
        clean = pre.remove_organelles(mesh, radius=50.0)
        assert len(clean.faces) == len(mesh.faces)

    def test_inner_sphere_removed(self):
        mesh = _sphere_with_internal()
        n_before = _live_faces(mesh)
        clean = pre.remove_organelles(mesh, radius=50.0)
        assert _live_faces(clean) < n_before
        # Should only have the outer sphere faces (approximately)
        outer_n = _live_faces(_icosphere(subdivisions=2, radius=100.0))
        assert _live_faces(clean) == outer_n


# ── find_fusions ─────────────────────────────────────────────────────


class TestFindFusions:
    def test_clean_mesh_no_fusions(self):
        mesh = _icosphere()
        clusters = pre.find_fusions(mesh, radius=50.0)
        assert clusters == []

    def test_nonmanifold_pinch_vertex(self):
        """_find_nonmanifold_fusions detects two tetrahedra sharing a vertex."""
        from skeliner.pre import _find_nonmanifold_fusions

        v0 = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0.5, 1, 0],
                [0.5, 0.5, 1],
            ],
            dtype=np.float64,
        )
        f0 = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        v1 = np.array([[-1, 0, 0], [-0.5, 1, 0], [-0.5, 0.5, 1]], dtype=np.float64)
        f1 = np.array([[0, 4, 5], [0, 4, 6], [4, 5, 6], [0, 5, 6]])
        mesh = trimesh.Trimesh(
            vertices=np.vstack([v0, v1]),
            faces=np.vstack([f0, f1]),
            process=False,
        )
        clusters = _find_nonmanifold_fusions(mesh, radius=2.0)
        assert len(clusters) >= 1

    def test_pinch_vertex_detected(self):
        """Two tetrahedra sharing a single vertex = fan vertex fusion."""
        # Tetrahedron 1
        v0 = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0.5, 1, 0],
                [0.5, 0.5, 1],
            ],
            dtype=np.float64,
        )
        f0 = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [1, 2, 3],
                [0, 2, 3],
            ]
        )
        # Tetrahedron 2 sharing vertex 0
        v1 = np.array(
            [
                [-1, 0, 0],
                [-0.5, 1, 0],
                [-0.5, 0.5, 1],
            ],
            dtype=np.float64,
        )
        f1 = np.array(
            [
                [0, 4, 5],
                [0, 4, 6],
                [4, 5, 6],
                [0, 5, 6],
            ]
        )
        verts = np.vstack([v0, v1])
        faces = np.vstack([f0, f1])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        clusters = pre.find_fusions(mesh, radius=2.0)
        assert len(clusters) >= 1


# ── remove_fusions ───────────────────────────────────────────────────


class TestRemoveFusions:
    def test_clean_mesh_unchanged(self):
        mesh = _icosphere()
        clean = pre.remove_fusions(mesh, radius=50.0)
        # Should still have the same number of faces (no fusions to remove)
        assert len(clean.faces) == len(mesh.faces)


# ── _filter_small_clusters ───────────────────────────────────────────


class TestFilterSmallClusters:
    def test_removes_small_clusters(self):
        mesh = _mesh_with_island(n_island_faces=2)
        # Mark the 2 island faces
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[-2:] = True  # the island faces are appended at the end
        filtered = pre._filter_small_clusters(mesh, mask, min_cluster_size=3)
        # 2-face cluster < min_size=3 → removed
        assert filtered.sum() == 0

    def test_keeps_large_clusters(self):
        mesh = _icosphere()
        # Mark all faces
        mask = np.ones(len(mesh.faces), dtype=bool)
        filtered = pre._filter_small_clusters(mesh, mask, min_cluster_size=3)
        assert filtered.sum() == len(mesh.faces)


# ── soma_face_mask ───────────────────────────────────────────────────


class TestSomaFaceMask:
    def test_majority_rule(self):
        # one triangle per row; soma verts are {0, 1}
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 0, 5]])
        mask = pre.soma_face_mask(faces, np.array([0, 1]))
        # 2 of 3 → soma; 1 of 3 → not; 0 of 3 → not
        assert mask.tolist() == [True, False, False, True]

    def test_no_soma(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        assert not pre.soma_face_mask(faces, None).any()
        assert not pre.soma_face_mask(faces, np.array([], dtype=int)).any()

    def test_matches_reference_loop(self):
        mesh = _icosphere()
        rng = np.random.default_rng(0)
        soma_verts = rng.choice(len(mesh.vertices), size=60, replace=False)
        soma_set = set(soma_verts.tolist())
        want = np.array(
            [
                sum(1 for v in f if int(v) in soma_set) >= 2
                for f in np.asarray(mesh.faces)
            ],
            dtype=bool,
        )
        got = pre.soma_face_mask(mesh.faces, soma_verts)
        assert np.array_equal(got, want)
        assert want.any(), "fixture should produce some soma faces"


# ── _face_components_fast ────────────────────────────────────────────


def _slow_components(mesh, fi):
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[fi] = True
    ef = pre._build_edge_to_faces(np.asarray(mesh.faces), mask)
    return pre._face_components(np.asarray(mesh.faces), ef, np.asarray(fi))


class TestFaceComponentsFast:
    def test_matches_slow_on_island_mesh(self):
        mesh = _mesh_with_island(n_island_faces=2)
        fi = np.arange(len(mesh.faces))
        fast = pre._face_components_fast(mesh.faces, fi)
        slow = _slow_components(mesh, fi)
        assert len(fast) == len(slow)
        for a, b in zip(fast, slow, strict=True):
            assert np.array_equal(np.sort(a), np.sort(b))

    def test_ties_break_on_lowest_face_id(self):
        # two identical cylinders → two components of equal size, so the
        # order is decided purely by the tie-break
        mesh = _two_cylinders()
        fi = np.arange(len(mesh.faces))
        fast = pre._face_components_fast(mesh.faces, fi)
        assert len(fast) == 2
        assert len(fast[0]) == len(fast[1])
        assert fast[0].min() < fast[1].min()
        slow = _slow_components(mesh, fi)
        for a, b in zip(fast, slow, strict=True):
            assert np.array_equal(np.sort(a), np.sort(b))

    def test_largest_first(self):
        mesh = _mesh_with_island(n_island_faces=2)
        comps = pre._face_components_fast(mesh.faces, np.arange(len(mesh.faces)))
        assert [len(c) for c in comps] == sorted((len(c) for c in comps), reverse=True)

    def test_empty(self):
        mesh = _icosphere()
        assert pre._face_components_fast(mesh.faces, np.array([], dtype=int)) == []

    def test_returns_faces_ascending(self):
        mesh = _icosphere()
        comps = pre._face_components_fast(mesh.faces, np.arange(len(mesh.faces)))
        for c in comps:
            assert np.array_equal(c, np.sort(c))


# ── _component_effects ───────────────────────────────────────────────


def _components(neurites, discarded, n_faces=64):
    """A MeshComponents carrying only the face lists the diff reads."""
    z = np.zeros(n_faces, dtype=bool)
    return MeshComponents(
        soma=None,
        organelles=Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy()),
        neurites=Neurites([np.asarray(a) for a in neurites]),
        discarded=Discarded([np.asarray(a) for a in discarded]),
    )


def _effects(before, after, n_faces=64):
    return dict(pre._component_effects(before, after, n_faces))


class TestComponentEffects:
    def test_no_change_reports_nothing(self):
        c = _components([[0, 1, 2, 3]], [])
        assert _effects(c, _components([[0, 1, 2, 3]], [])) == {}

    def test_split(self):
        before = _components([[0, 1, 2, 3]], [])
        after = _components([[0, 1], [2, 3]], [])
        assert _effects(before, after)["neurite 0"] == "split into 2"

    def test_merge(self):
        before = _components([[0, 1], [2, 3]], [])
        after = _components([[0, 1, 2, 3]], [])
        got = _effects(before, after)
        assert got["neurite 0"] == "merged with neurite 1"
        assert got["neurite 1"] == "merged with neurite 0"

    def test_grown_and_shrunk(self):
        before = _components([[0, 1, 2, 3]], [])
        assert _effects(before, _components([[0, 1, 2, 3, 4]], []))["neurite 0"] == (
            "grown +1f"
        )
        assert _effects(before, _components([[0, 1, 2]], []))["neurite 0"] == (
            "shrunk -1f"
        )

    def test_dissolved(self):
        before = _components([[0, 1], [2, 3]], [])
        after = _components([[2, 3]], [])
        assert _effects(before, after)["neurite 0"] == (
            "dissolved into soma/organelles"
        )

    def test_reclassified_between_neurite_and_discarded(self):
        before = _components([[0, 1, 2, 3]], [[4]])
        after = _components([[4]], [[0, 1, 2, 3]])
        got = _effects(before, after)
        assert got["neurite 0"] == "reclassified as discarded 0"
        assert got["discarded 0"] == "reclassified as neurite 0"

    def test_new_component(self):
        before = _components([[0, 1]], [])
        after = _components([[0, 1]], [[8, 9]])
        assert _effects(before, after)["discarded 0"] == "new (2f)"

    def test_pure_renumbering_is_not_reported(self):
        # same two components, swapped positions — ids shift on every
        # re-derive and reporting that would bury the real changes
        before = _components([[0, 1, 2], [4, 5]], [])
        after = _components([[4, 5], [0, 1, 2]], [])
        assert _effects(before, after) == {}


# ── preview_reassignment / apply_reassignment ────────────────────────


def _tube_state():
    """A plain tube: one neurite, no soma, no organelles."""
    mesh = _subdivided_tube()
    z = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
    return mesh, pre.break_up_mesh(mesh, None, org)


def _tube_with_soma():
    """A tube whose middle band is soma, split either side of it.

    The components are built here rather than by ``break_up_mesh``.  On a
    bare tube each half is bounded *only* by soma faces, so the
    absorption pass — which reclaims any component whose boundary is more
    than half soma — would pull the whole mesh into the soma and leave no
    arbor to reassign.  Real neurites escape that because their organelle
    pockets give them plenty of non-soma boundary.
    """
    mesh = _subdivided_tube()
    band = np.array(sorted(_band(mesh, -100.0, 100.0)))
    sv = np.unique(np.asarray(mesh.faces)[band])
    soma = Soma.fit(mesh.vertices[sv], verts=sv)
    z = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
    arbor = np.flatnonzero(~pre.soma_face_mask(mesh.faces, sv))
    components = MeshComponents(
        soma=soma,
        organelles=org,
        neurites=Neurites(pre._face_components_fast(mesh.faces, arbor)),
        discarded=Discarded([]),
    )
    return mesh, components, band


class TestPreviewReassignment:
    def test_rejects_unknown_target(self):
        mesh, comp = _tube_state()
        with pytest.raises(ValueError, match="to must be"):
            pre.preview_reassignment(mesh, comp, [0], to="neurite 1")

    def test_rejects_empty_selection(self):
        mesh, comp = _tube_state()
        with pytest.raises(ValueError, match="no faces selected"):
            pre.preview_reassignment(mesh, comp, [], to="organelle")

    def test_rejects_out_of_range_faces(self):
        mesh, comp = _tube_state()
        with pytest.raises(ValueError, match="face ids must lie"):
            pre.preview_reassignment(mesh, comp, [len(mesh.faces)], to="organelle")

    def test_soma_target_without_a_soma_makes_the_selection_the_soma(self):
        # find_soma found none: the faces lassoed on one side of the body
        # become the soma, its ellipsoid fitted to them; the re-derive takes
        # in the rest of the body and leaves the process a neurite
        mesh, _, org = _ball_with_stub(height=160.0)
        comp = pre.break_up_mesh(mesh, None, org)
        assert comp.soma is None
        f = np.asarray(mesh.faces)
        # the fixture puts the ball's faces first
        ball = np.arange(len(trimesh.creation.icosphere(subdivisions=4).faces))
        top = ball[mesh.vertices[f[ball]].mean(axis=1)[:, 2] > 0]
        r = pre.preview_reassignment(mesh, comp, top, to="soma")
        soma = r.components.soma
        assert soma is not None and soma.nucleus is None
        assert set(np.unique(f[ball]).tolist()) <= set(soma.verts.tolist())
        (neurite,) = r.components.neurites
        far = np.flatnonzero(mesh.vertices[f].mean(axis=1)[:, 2] > 170.0)
        assert set(far.tolist()) <= set(neurite.tolist())
        assert not np.isin(neurite, ball).any()
        pre.apply_reassignment(comp, r)
        assert comp.soma is soma

    def test_a_soma_made_by_hand_is_one_body_not_the_stray_faces(self):
        # a lasso drawn through the mesh also caught faces far off on the
        # process: the soma is the selection's largest connected piece
        mesh, _, org = _ball_with_stub(height=160.0)
        comp = pre.break_up_mesh(mesh, None, org)
        f = np.asarray(mesh.faces)
        n_ball = len(trimesh.creation.icosphere(subdivisions=4).faces)
        cz = mesh.vertices[f].mean(axis=1)[:, 2]
        top = np.flatnonzero((np.arange(len(f)) < n_ball) & (cz > 0))
        stray = np.flatnonzero(cz > 230.0)[:6]
        r = pre.preview_reassignment(mesh, comp, np.r_[top, stray], to="soma")
        assert not np.isin(np.unique(f[stray]), r.components.soma.verts).any()
        assert np.isin(stray, np.concatenate(list(r.components.neurites))).all()

    def test_soma_target_refuses_a_selection_spanning_no_soma(self):
        mesh, comp = _tube_state()
        cap = np.array(sorted(_band(mesh, 499.9, 500.1)))
        assert len(cap) > 4
        for sel in ([0], cap):
            with pytest.raises(ValueError, match="span no soma"):
                pre.preview_reassignment(mesh, comp, sel, to="soma")

    def test_organelle_target_sets_the_manual_mask(self):
        mesh, comp = _tube_state()
        sel = np.array(sorted(_band(mesh, -100.0, 100.0)))
        r = pre.preview_reassignment(mesh, comp, sel, to="organelle")
        manual = r.components.organelles.manual
        assert manual[sel].all()
        assert manual.sum() == len(sel)
        # and the detected masks are left alone
        assert not r.components.organelles.pocket.any()

    def test_preview_does_not_mutate_the_input(self):
        mesh, comp = _tube_state()
        before_n = [c.copy() for c in comp.neurites]
        before_org = comp.organelles.mask.copy()
        sel = np.array(sorted(_band(mesh, -100.0, 100.0)))
        pre.preview_reassignment(mesh, comp, sel, to="organelle")
        assert np.array_equal(comp.organelles.mask, before_org)
        assert len(comp.neurites) == len(before_n)
        for a, b in zip(comp.neurites, before_n, strict=True):
            assert np.array_equal(a, b)

    def test_apply_matches_a_from_scratch_break_up_mesh(self):
        mesh, comp = _tube_state()
        sel = np.array(sorted(_band(mesh, -100.0, 100.0)))
        r = pre.preview_reassignment(mesh, comp, sel, to="organelle")
        pre.apply_reassignment(comp, r)

        scratch = pre.break_up_mesh(mesh, comp.soma, comp.organelles)
        assert len(comp.neurites) == len(scratch.neurites)
        assert len(comp.discarded) == len(scratch.discarded)
        for a, b in zip(comp.neurites, scratch.neurites, strict=True):
            assert np.array_equal(np.sort(a), np.sort(b))

    def test_soma_target_grows_the_vertex_set(self):
        mesh, comp, _ = _tube_with_soma()
        before = set(comp.soma.verts.tolist())
        sel = comp.neurites[0][:64]
        r = pre.preview_reassignment(mesh, comp, sel, to="soma")
        after = set(r.components.soma.verts.tolist())
        # every vertex of the selection joins the soma, and nothing leaves
        assert before <= after
        assert set(np.unique(np.asarray(mesh.faces)[sel]).tolist()) <= after

    @staticmethod
    def _cell():
        """The ball with a process standing out of it, broken up: a soma,
        and the process a neurite."""
        mesh, soma, org = _ball_with_stub(height=160.0)
        comp = pre.break_up_mesh(mesh, soma, org)
        assert comp.soma is not None and len(comp.neurites) == 1
        return mesh, comp

    @staticmethod
    def _fitted(mesh, soma):
        """Whether *soma*'s ellipsoid is the one its vertices give."""
        ref = Soma.fit(mesh.vertices[soma.verts])
        return np.allclose(soma.center, ref.center) and np.allclose(soma.axes, ref.axes)

    def test_faces_given_to_the_soma_refit_its_ellipsoid(self):
        mesh, comp = self._cell()
        f = np.asarray(mesh.faces)
        cz = mesh.vertices[f].mean(axis=1)[:, 2]
        (process,) = comp.neurites
        base = process[(cz[process] > 105.0) & (cz[process] < 150.0)]
        r = pre.preview_reassignment(mesh, comp, base, to="soma")
        soma = r.components.soma
        assert np.isin(np.unique(f[base]), soma.verts).all()
        assert self._fitted(mesh, soma)
        # it reaches up the process it took a piece of
        assert soma.center[2] > comp.soma.center[2]
        assert soma.nucleus is comp.soma.nucleus

    def test_faces_released_from_the_soma_refit_its_ellipsoid(self):
        mesh, comp = self._cell()
        f = np.asarray(mesh.faces)
        cz = mesh.vertices[f].mean(axis=1)[:, 2]
        sv = comp.soma.verts
        on_soma = np.flatnonzero(pre.soma_face_mask(f, sv))
        bottom = on_soma[cz[on_soma] < -60.0]
        r = pre.preview_reassignment(
            mesh, comp, bottom, to="remainder", released=bottom
        )
        soma = r.components.soma
        assert not np.isin(np.unique(f[bottom]), soma.verts).any()
        assert self._fitted(mesh, soma)
        # what is left sits higher
        assert soma.center[2] > comp.soma.center[2]

    def test_a_soma_released_whole_is_none(self):
        mesh, comp = self._cell()
        f = np.asarray(mesh.faces)
        every = np.flatnonzero(np.isin(f, comp.soma.verts).any(axis=1))
        r = pre.preview_reassignment(mesh, comp, every, to="remainder", released=every)
        assert r.components.soma is None

    def test_an_organelle_assignment_keeps_the_ellipsoid(self):
        # the soma's vertices are what they were: its ellipsoid is too
        mesh, comp = self._cell()
        on_soma = np.flatnonzero(pre.soma_face_mask(mesh.faces, comp.soma.verts))
        r = pre.preview_reassignment(mesh, comp, on_soma[:40], to="organelle")
        soma = r.components.soma
        np.testing.assert_array_equal(soma.center, comp.soma.center)
        np.testing.assert_array_equal(soma.axes, comp.soma.axes)

    def test_leaving_includes_the_majority_rule_fringe(self):
        mesh, comp, _ = _tube_with_soma()
        sel = comp.neurites[0][:64]
        r = pre.preview_reassignment(mesh, comp, sel, to="soma")
        # the selection leaves the arbor, and drags with it every
        # neighbour left sharing two of its vertices
        assert set(sel.tolist()) <= set(r.leaving.tolist())
        assert len(r.leaving) > len(sel)
        assert len(r.entering) == 0

    def test_remainder_target_releases_organelle_faces(self):
        mesh, comp = _tube_state()
        sel = np.array(sorted(_band(mesh, -100.0, 100.0)))
        r = pre.preview_reassignment(mesh, comp, sel, to="organelle")
        pre.apply_reassignment(comp, r)
        assert comp.organelles.manual[sel].all()

        back = pre.preview_reassignment(mesh, comp, sel, to="remainder")
        assert not back.components.organelles.manual[sel].any()
        assert set(sel.tolist()) <= set(back.entering.tolist())


# ── claiming a component as arbor ────────────────────────────────────
#
# `rescued` names faces the user has declared arbor, and the claim beats
# all three ways a component can stop being a neurite: the size
# threshold, absorption into the soma, and organelle trapping.


def _ball_with_stub(height=40.0):
    """A soma-sized ball with a process *height* long standing on it.

    The process's far end is a closed cap, so the junction with the ball
    is nearly all the boundary it has and the absorption rule reads it as
    soma.  At the default it ends inside the soma's fitted ellipsoid — a
    truncated stump, absorbed; at 160 it stands out of the soma like a
    neurite and stays one.
    """
    ball = trimesh.creation.icosphere(subdivisions=4, radius=100.0)
    stub = trimesh.creation.cylinder(radius=12.0, height=height, sections=24)
    stub = stub.subdivide().subdivide()
    stub.apply_translation([0.0, 0.0, 90.0 + height / 2])
    mesh = _combine([ball, stub])
    z = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
    sv = np.flatnonzero(np.linalg.norm(mesh.vertices, axis=1) <= 105.0)
    return mesh, Soma.fit(mesh.vertices[sv], verts=sv), org


def _ball_with_arbor(subdivisions):
    """:func:`_ball_with_stub`'s shape with a longer process, cut into more
    faces the more it is subdivided: fewer than the soma has at 0, more at
    2. Its boundary is soma faces all round either way — for the kernel's
    size test; ``break_up_mesh`` leaves it out by its shape at both."""
    ball = trimesh.creation.icosphere(subdivisions=2, radius=100.0)
    arbor = trimesh.creation.cylinder(radius=12.0, height=400.0, sections=24)
    for _ in range(subdivisions):
        arbor = arbor.subdivide()
    arbor.apply_translation([0.0, 0.0, 290.0])
    mesh = _combine([ball, arbor])
    z = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
    sv = np.flatnonzero(np.linalg.norm(mesh.vertices, axis=1) <= 105.0)
    return mesh, Soma.fit(mesh.vertices[sv], verts=sv), org


def _tube_cut_by_an_organelle_band(lo=0.0, hi=450.0):
    """A capped tube ringed by organelle faces all the way round, for z in
    (*lo*, *hi*); by default four rows of the side, no smaller than the cap
    they strand.

    A partial band leaves a way past and does not trap anything, so the
    band has to close for this fixture to be about trapping at all.
    """
    mesh = _subdivided_tube(sections=16, height=1000.0)
    cz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    band = (cz > lo) & (cz < hi)
    z = np.zeros(len(mesh.faces), dtype=bool)
    return mesh, Organelles(pocket=band, isolated=z.copy(), expanded=z.copy())


def _kernel_kinds(mesh, soma, org):
    """What the kernel makes of :func:`_arbor_pieces`, before
    ``break_up_mesh``'s shape test."""
    faces = np.asarray(mesh.faces)
    usable = pre._usable_face_mask(mesh)
    soma_face = pre.soma_face_mask(faces, None if soma is None else soma.verts)
    claimed = np.zeros(len(faces), dtype=bool)
    _, kinds, _ = pre._break_up_pieces(
        faces,
        mesh.vertices,
        usable,
        org.mask,
        soma_face,
        claimed,
        soma is not None,
        centers=mesh.triangles_center,
        normals=mesh.face_normals,
    )
    return list(kinds)


def _arbor_pieces(mesh, soma, org):
    """The components break_up_mesh is about to classify."""
    faces = np.asarray(mesh.faces)
    remain = (
        pre._usable_face_mask(mesh)
        & ~pre.soma_face_mask(faces, None if soma is None else soma.verts)
        & ~org.mask
    )
    return pre._face_components_fast(faces, np.flatnonzero(remain))


class TestSomaAbsorption:
    """What the rule does unaided — the premise the override exists for."""

    def test_a_process_on_the_soma_is_absorbed(self):
        mesh, soma, org = _ball_with_stub()
        assert len(_arbor_pieces(mesh, soma, org)) == 1, (
            "fixture must leave exactly the process to classify"
        )
        c = pre.break_up_mesh(mesh, soma, org)
        assert len(c.neurites) == 0
        assert len(c.discarded) == 0, "absorbed, not discarded — a different rule"

    def test_absorbing_it_grows_the_soma_around_it(self):
        # Absorption is not a relabel: the verts join soma.verts and the
        # ellipsoid is re-fit, so a wrong absorption also reshapes the soma —
        # stretched towards the stump.
        mesh, soma, org = _ball_with_stub()
        stub = _arbor_pieces(mesh, soma, org)[0]
        absorbed = pre.break_up_mesh(mesh, soma, org)
        kept = pre.break_up_mesh(mesh, soma, org, rescued=stub[:1])
        assert len(absorbed.soma.verts) > len(kept.soma.verts)
        assert len(kept.soma.verts) == len(soma.verts)
        assert kept.soma.axes[0] < absorbed.soma.axes[0]

    def test_a_piece_bigger_than_the_soma_is_never_absorbed(self):
        # an organelle-free arbor all one piece reads as soma by its boundary
        # at any length; to the kernel, its size against the soma's decides
        for subdivisions, absorbed in ((0, True), (2, False)):
            mesh, soma, org = _ball_with_arbor(subdivisions)
            (piece,) = _arbor_pieces(mesh, soma, org)
            faces = np.asarray(mesh.faces)
            soma_faces = pre._usable_face_mask(mesh) & pre.soma_face_mask(
                faces, soma.verts
            )
            assert bool(len(piece) > soma_faces.sum()) != absorbed
            kind = pre._PIECE_SOMA if absorbed else pre._PIECE_CANDIDATE
            assert _kernel_kinds(mesh, soma, org) == [kind]

    def test_a_process_standing_out_like_a_neurite_is_not_absorbed(self):
        # one process, a stump's length and a neurite's: both read as soma
        # by their boundary; how far it stands out of the soma decides
        for height, absorbed in ((40.0, True), (160.0, False)):
            mesh, soma, org = _ball_with_stub(height)
            (stub,) = _arbor_pieces(mesh, soma, org)
            assert _kernel_kinds(mesh, soma, org) == [pre._PIECE_SOMA]
            c = pre.break_up_mesh(mesh, soma, org)
            assert (len(c.neurites) == 0) == absorbed
            if not absorbed:
                assert np.array_equal(np.sort(c.neurites[0]), np.sort(stub))
                assert np.array_equal(c.soma.verts, soma.verts)

    def test_a_full_organelle_band_traps_what_is_behind_it(self):
        mesh, org = _tube_cut_by_an_organelle_band()
        pieces = _arbor_pieces(mesh, None, org)
        assert len(pieces) == 2, "the band must close, or nothing is trapped"
        assert len(pieces[1]) <= int(org.mask.sum())
        c = pre.break_up_mesh(mesh, None, org)
        assert int(c.organelles.expanded.sum()) == len(pieces[1])
        assert len(c.neurites) == 1

    def test_a_neurite_cut_off_by_a_thin_band_is_joined_back_across_it(self):
        # one row of the side strands the rest of the tube: bigger than the
        # band and looking out of the cell, arbor the band severed, not a
        # fold's inside — the band, touching both, joins it back, and is
        # no organelle any more
        mesh, org = _tube_cut_by_an_organelle_band(100.0, 190.0)
        pieces = _arbor_pieces(mesh, None, org)
        assert len(pieces) == 2 and len(pieces[1]) > int(org.mask.sum())
        c = pre.break_up_mesh(mesh, None, org)
        assert not c.organelles.mask.any()
        (neurite,) = c.neurites
        band = np.flatnonzero(org.mask)
        assert set(neurite.tolist()) == set(np.concatenate([*pieces, band]).tolist())
        # an organelle drawn by hand stays one, and the piece stays apart
        drawn = Organelles(
            pocket=org.pocket,
            isolated=org.isolated,
            expanded=org.expanded,
            manual=org.mask.copy(),
        )
        kept = pre.break_up_mesh(mesh, None, drawn)
        assert np.array_equal(kept.organelles.mask, org.mask)
        assert sorted(len(x) for x in kept.neurites) == sorted(len(x) for x in pieces)


# ── rescuing a discarded fragment ────────────────────────────────────


def _tube_with_speck():
    """A tube plus a speck small enough for the 95% rule to discard.

    The threshold keeps components until the cumulative face count
    reaches 95% of the total, so the speck has to be under 1/19th of the
    tube for this to be a rescue test rather than a no-op.
    """
    tube = _cylinder_at(50.0, 1000.0, 0.0, 64).subdivide()
    speck = trimesh.creation.box(extents=[8.0, 8.0, 8.0])
    speck.apply_translation([500.0, 0.0, 0.0])
    mesh = _combine([tube, speck])
    z = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
    return mesh, org


def test_pieces_the_size_of_a_kept_neurite_are_neurites():
    """One big piece and ten of a size: the 95% cut would land among the
    ten and drop two of them; all ten are neurites, the speck is not."""
    big = _cylinder_at(50.0, 1000.0, 0.0, 64).subdivide().subdivide().subdivide()
    small = []
    for k in range(10):
        t = _cylinder_at(50.0, 300.0, 0.0, 8).subdivide().subdivide()
        t.apply_translation([400.0 * (k + 1), 0.0, 0.0])
        small.append(t)
    speck = trimesh.creation.box(extents=[8.0, 8.0, 8.0])
    speck.apply_translation([0.0, 600.0, 0.0])
    mesh = _combine([big, *small, speck])
    sizes = sorted([len(big.faces)] + [len(t.faces) for t in small], reverse=True)
    covered = np.cumsum(sizes) >= 0.95 * (sum(sizes) + len(speck.faces))
    assert int(np.argmax(covered)) + 1 < len(sizes), "the 95% cut must fall among them"
    z = np.zeros(len(mesh.faces), dtype=bool)
    c = pre.break_up_mesh(
        mesh, None, Organelles(pocket=z, isolated=z.copy(), expanded=z.copy())
    )
    assert len(c.neurites) == 11
    assert [len(d) for d in c.discarded] == [len(speck.faces)]


class TestRescued:
    def test_the_speck_is_discarded_without_an_override(self):
        mesh, org = _tube_with_speck()
        c = pre.break_up_mesh(mesh, None, org)
        assert len(c.neurites) == 1
        assert len(c.discarded) == 1

    def test_an_override_keeps_it(self):
        mesh, org = _tube_with_speck()
        speck = pre.break_up_mesh(mesh, None, org).discarded[0]
        c = pre.break_up_mesh(mesh, None, org, rescued=speck)
        assert len(c.discarded) == 0
        assert len(c.neurites) == 2

    def test_one_face_claims_the_whole_fragment(self):
        # The threshold discards whole components, so naming any face of
        # one is enough to keep all of it.
        mesh, org = _tube_with_speck()
        speck = pre.break_up_mesh(mesh, None, org).discarded[0]
        c = pre.break_up_mesh(mesh, None, org, rescued=speck[:1])
        assert len(c.discarded) == 0
        assert set(c.neurites[1].tolist()) == set(speck.tolist())

    def test_a_rescued_fragment_is_an_ordinary_neurite(self):
        # Nothing on the result records that it was ever discarded.
        mesh, org = _tube_with_speck()
        speck = pre.break_up_mesh(mesh, None, org).discarded[0]
        c = pre.break_up_mesh(mesh, None, org, rescued=speck)
        assert not hasattr(c, "rescued")
        assert not hasattr(c.neurites, "rescued")

    def test_neurites_stay_sorted_by_size(self):
        mesh, org = _tube_with_speck()
        speck = pre.break_up_mesh(mesh, None, org).discarded[0]
        c = pre.break_up_mesh(mesh, None, org, rescued=speck)
        sizes = [len(x) for x in c.neurites]
        assert sizes == sorted(sizes, reverse=True)

    def test_an_override_naming_nothing_discarded_changes_nothing(self):
        mesh, org = _tube_with_speck()
        plain = pre.break_up_mesh(mesh, None, org)
        c = pre.break_up_mesh(mesh, None, org, rescued=plain.neurites[0][:5])
        assert [len(x) for x in c.neurites] == [len(x) for x in plain.neurites]
        assert [len(x) for x in c.discarded] == [len(x) for x in plain.discarded]

    def test_out_of_range_face_ids_are_ignored(self):
        mesh, org = _tube_with_speck()
        nF = len(mesh.faces)
        c = pre.break_up_mesh(mesh, None, org, rescued=[nF + 10, -1])
        assert len(c.discarded) == 1

    def test_rescue_discarded_moves_it_but_does_not_make_it_stick(self):
        # The dataclass method is a plain list move; durability is the
        # caller's job, via the override.  Pinning that down because the
        # difference is exactly what used to be lost.
        mesh, org = _tube_with_speck()
        c = pre.break_up_mesh(mesh, None, org)
        c.rescue_discarded(0)
        assert len(c.neurites) == 2 and len(c.discarded) == 0
        again = pre.break_up_mesh(mesh, None, org)
        assert len(again.discarded) == 1

    def test_rescue_discarded_keeps_neurites_sorted(self):
        c = _components([[0, 1, 2, 3], [4, 5, 6]], [[7, 8], [9]])
        c.rescue_discarded([0, 1])
        sizes = [len(x) for x in c.neurites]
        assert sizes == sorted(sizes, reverse=True)
        assert len(c.discarded) == 0

    def test_a_reassignment_re_derive_does_not_undo_a_rescue(self):
        # preview_reassignment runs the real break_up_mesh, which is
        # where a rescue used to vanish.
        mesh, org = _tube_with_speck()
        speck = pre.break_up_mesh(mesh, None, org).discarded[0]
        c = pre.break_up_mesh(mesh, None, org, rescued=speck)

        # Same edit, nowhere near the speck; the override is the only
        # difference between the two.
        elsewhere = [int(c.neurites[0][0])]
        lost = pre.preview_reassignment(mesh, c, elsewhere, to="organelle")
        assert len(lost.components.discarded) == 1

        kept = pre.preview_reassignment(
            mesh, c, elsewhere, to="organelle", rescued=speck
        )
        assert len(kept.components.discarded) == 0

    def test_a_claim_outranks_the_soma_absorption_rule(self):
        # The rule reads a process on the soma as soma, because a process
        # capped at its far end has the junction for nearly all of its
        # boundary.  A claim is how the user says otherwise.
        mesh, soma, org = _ball_with_stub()
        stub = _arbor_pieces(mesh, soma, org)[0]
        c = pre.break_up_mesh(mesh, soma, org, rescued=stub[:1])
        assert len(c.neurites) == 1
        assert set(c.neurites[0].tolist()) == set(stub.tolist())

    def test_a_claim_outranks_organelle_trapping(self):
        mesh, org = _tube_cut_by_an_organelle_band()
        far = _arbor_pieces(mesh, None, org)[1]
        c = pre.break_up_mesh(mesh, None, org, rescued=far[:1])
        assert int(c.organelles.expanded.sum()) == 0
        assert sorted(len(x) for x in c.neurites) == sorted(
            len(p) for p in _arbor_pieces(mesh, None, org)
        )

    def test_the_floor_applies_to_a_lasso_claim_and_not_to_a_rescue(self):
        # The two differ only in how well the user could aim, which is
        # exactly why they are separate arguments.
        mesh, org = _tube_with_speck()
        speck = pre.break_up_mesh(mesh, None, org).discarded[0]
        assert len(speck) < 16, "fixture speck must sit under the floor"

        lassoed = pre.break_up_mesh(mesh, None, org, released=speck, claim_min_faces=16)
        assert len(lassoed.discarded) == 1

        picked = pre.break_up_mesh(mesh, None, org, rescued=speck, claim_min_faces=16)
        assert len(picked.discarded) == 0

    def test_the_floor_does_not_hand_a_claim_back_to_the_soma(self):
        # The floor decides neurite vs discarded and nothing else.  Under
        # it the piece must stay out of the soma — escaping absorption is
        # the half of the claim that is not size-dependent.
        mesh, soma, org = _ball_with_stub()
        stub = _arbor_pieces(mesh, soma, org)[0]
        c = pre.break_up_mesh(
            mesh, soma, org, released=stub[:1], claim_min_faces=len(stub) + 1
        )
        assert len(c.soma.verts) == len(soma.verts), "the soma must not have grown"
        assert sum(len(x) for x in c.neurites) + sum(
            len(x) for x in c.discarded
        ) == len(stub), "the piece is still on the arbor side, whichever list it is in"

    def test_the_floor_only_gates_what_the_threshold_discarded(self):
        # It is a filter on the promotion, not a minimum neurite size: a
        # piece the 95% rule keeps by itself is a neurite either way.
        mesh, soma, org = _ball_with_stub()
        stub = _arbor_pieces(mesh, soma, org)[0]
        c = pre.break_up_mesh(
            mesh, soma, org, released=stub[:1], claim_min_faces=len(stub) + 1
        )
        assert [len(x) for x in c.neurites] == [len(stub)]
        assert len(c.discarded) == 0

    def test_a_neurite_standing_out_of_the_soma_is_kept_at_any_size(self):
        # The 95% cut is by size alone: beside a process holding nearly all
        # the arbor's faces, a stub standing out of the soma falls under it
        # (and under a quarter of the smallest kept).  It is a neurite by
        # the test that keeps a stub out of the soma; a speck apart from the
        # soma stays discarded.
        ball_stub, _, _ = _ball_with_stub(height=160.0)
        long = trimesh.creation.cylinder(radius=12.0, height=3000.0, sections=24)
        for _ in range(5):
            long = long.subdivide()
        long.apply_translation([0.0, 0.0, -(90.0 + 1500.0)])
        speck = trimesh.creation.box(extents=[8.0, 8.0, 8.0])
        speck.apply_translation([2000.0, 0.0, 0.0])
        mesh = _combine([ball_stub, long, speck])
        z = np.zeros(len(mesh.faces), dtype=bool)
        org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
        sv = np.flatnonzero(np.linalg.norm(mesh.vertices, axis=1) <= 105.0)
        soma = Soma.fit(mesh.vertices[sv], verts=sv)

        c = mesh.triangles_center
        pieces = _arbor_pieces(mesh, soma, org)
        near = [p for p in pieces if c[p, 0].max() < 1000]  # not the speck
        (stub,) = [p for p in near if c[p, 2].mean() > 0]
        (arbor,) = [p for p in near if c[p, 2].mean() < 0]
        sizes = len(stub) + len(arbor) + len(speck.faces)
        assert len(arbor) >= 0.95 * sizes and len(stub) < len(arbor) / 4, (
            "fixture: the size rule alone must drop the stub"
        )

        got = pre.break_up_mesh(mesh, soma, org)
        kept = [set(x.tolist()) for x in got.neurites]
        assert set(stub.tolist()) in kept and set(arbor.tolist()) in kept
        assert [len(x) for x in got.discarded] == [len(speck.faces)]

    def test_a_full_preprocess_does_not_undo_a_rescue(self):
        # Same hazard one level up: preprocess ends in break_up_mesh, so a
        # one-click re-run would re-discard what the user kept.
        mesh, org = _tube_with_speck()
        speck = pre.break_up_mesh(mesh, None, org).discarded[0]

        _, lost = pre.preprocess(mesh)
        assert len(lost.discarded) == 1, "fixture must have something to lose"

        _, kept = pre.preprocess(mesh, rescued=speck)
        assert len(kept.discarded) == 0
        assert len(kept.neurites) == 2

    def test_a_full_preprocess_takes_a_soma_given_by_hand(self, monkeypatch):
        # a soma assigned where find_soma found none is input to the next
        # run, not a casualty of it: find_soma is not asked again
        mesh, soma, _ = _ball_with_stub(height=160.0)

        def no(*_a, **_k):
            raise AssertionError("find_soma ran over a given soma")

        monkeypatch.setattr(pre, "find_soma", no)
        _, got = pre.preprocess(mesh, organelles=False, soma=soma)
        assert np.isin(soma.verts, got.soma.verts).all()
        assert len(got.neurites) == 1

    def test_a_full_preprocess_told_there_is_no_soma_runs_as_when_none_is_found(
        self, monkeypatch
    ):
        # soma=False takes the path find_soma's None takes, without asking
        # find_soma: a cell whose soma lies outside the volume is not forced
        # one out of a swelling of its arbor
        mesh, soma, _ = _ball_with_stub(height=160.0)
        _, given = pre.preprocess(mesh, organelles=False, soma=soma)
        assert given.soma is not None, "fixture: the soma must change the run"

        monkeypatch.setattr(pre, "find_soma", lambda *_a, **_k: None)
        want_mesh, want = pre.preprocess(mesh, organelles=False)

        def no(*_a, **_k):
            raise AssertionError("find_soma ran though the cell has no soma")

        monkeypatch.setattr(pre, "find_soma", no)
        got_mesh, got = pre.preprocess(mesh, organelles=False, soma=False)
        assert got.soma is None
        assert np.array_equal(got_mesh.vertices, want_mesh.vertices)
        assert np.array_equal(got_mesh.faces, want_mesh.faces)
        assert [x.tolist() for x in got.neurites] == [x.tolist() for x in want.neurites]
        assert [x.tolist() for x in got.discarded] == [
            x.tolist() for x in want.discarded
        ]

    def test_a_full_preprocess_looks_for_the_soma_at_a_point(self):
        # an annotation: the soma found there is kept, and none is made of
        # a point no soma found holds
        ball = trimesh.creation.icosphere(subdivisions=3, radius=3000.0)
        _, found = pre.preprocess(ball, organelles=False)
        assert found.soma is not None, "fixture: a soma to find"
        _, at = pre.preprocess(ball, organelles=False, soma=ball.vertices[0])
        assert np.array_equal(at.soma.verts, found.soma.verts)
        _, off = pre.preprocess(ball, organelles=False, soma=(30000.0, 0.0, 0.0))
        assert off.soma is None

    def test_the_record_says_how_the_soma_was_settled(self):
        ball = trimesh.creation.icosphere(subdivisions=3, radius=3000.0)
        point = [float(x) for x in ball.vertices[0]]
        far = [30000.0, 0.0, 0.0]
        mesh, soma, _ = _ball_with_stub(height=160.0)
        for m, kw, want in [
            (ball, {}, {"asked": "find", "point": None, "found": True}),
            (ball, {"soma": point}, {"asked": "point", "point": point, "found": True}),
            (ball, {"soma": far}, {"asked": "point", "point": far, "found": False}),
            (mesh, {"soma": soma}, {"asked": "given", "point": None, "found": True}),
            (mesh, {"soma": False}, {"asked": "off", "point": None, "found": False}),
        ]:
            _, c = pre.preprocess(m, organelles=False, **kw)
            assert c.record["soma"] == want, kw

    def test_the_record_places_each_bridge(self, monkeypatch):
        # the gaps find_gaps hands preprocess, one entry each: where the two
        # tips are, the gap, the pieces joined (None the body), and the
        # smaller piece's box diagonal — find_gaps' own measure of its size
        mesh = _two_cylinders(separation=1500.0)
        real, seen = pre.find_gaps, []

        def find_gaps(m, **kw):
            seen.extend(real(m, **kw, **NO_CAP))
            return seen

        monkeypatch.setattr(pre, "find_gaps", find_gaps)
        _, c = pre.preprocess(mesh, organelles=False, soma=False)
        assert len(seen) == 1, "fixture: one gap to bridge"
        fa, fb, d, ia, ib = seen[0]
        (b,) = c.record["bridges"]
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        np.testing.assert_allclose(b["a"], v[f[fa]].reshape(-1, 3).mean(0))
        np.testing.assert_allclose(b["b"], v[f[fb]].reshape(-1, 3).mean(0))
        assert b["length"] == d
        half = len(mesh.faces) // 2
        assert sorted(b["pieces"], key=lambda x: x is None) == [half, None]
        far = v[f[half:]].reshape(-1, 3)
        assert b["diag"] == pytest.approx(np.linalg.norm(far.max(0) - far.min(0)))
        assert c.record["disconnected"] == {"pieces": 1, "faces": half}
        assert c.record["median_edge"] == pytest.approx(
            float(np.median(trimesh.Trimesh(v, f, process=False).edges_unique_length))
        )

    def test_the_record_is_plain_json_and_survives_compaction(self):
        import json

        mesh, org = _tube_with_speck()
        _, c = pre.preprocess(mesh)
        _, cc = pre.preprocess(mesh, compact=True)
        assert json.loads(json.dumps(c.record)) == c.record
        assert cc.record == c.record
        assert c.record["input"] == {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
        }
        assert c.record["fill"]["asked"] is True
        assert c.record["overrides"] == {"rescued": 0, "released": 0}
        _, kept = pre.preprocess(mesh, rescued=c.discarded[0])
        assert kept.record["overrides"]["rescued"] == len(c.discarded[0])

    def test_a_full_preprocess_refuses_a_soma_it_cannot_read(self):
        mesh, _, _ = _ball_with_stub()
        for what in ("none", [1.0, 2.0]):
            with pytest.raises(TypeError, match="soma is a Soma"):
                pre.preprocess(mesh, soma=what)

    def test_preprocess_without_organelles_takes_none_out(self):
        """For a mesh that traces none: the surface find_organelles would
        call one is left to the split into neurites and discards."""
        mesh = _sphere_with_internal()
        _, found = pre.preprocess(mesh)
        assert found.organelles.mask.sum() > 0, "fixture must have an organelle"

        _, none = pre.preprocess(mesh, organelles=False)
        assert not none.organelles.mask.any()
        pieces = [*none.neurites, *none.discarded]
        assert sum(len(p) for p in pieces) == len(mesh.faces)

    def test_break_up_mesh_takes_none_for_no_organelles(self):
        mesh, soma, _ = _ball_with_stub()
        given = pre.break_up_mesh(mesh, soma, None)
        empty = pre.break_up_mesh(mesh, soma, Organelles.empty(len(mesh.faces)))
        assert not given.organelles.mask.any()
        assert [len(x) for x in given.neurites] == [len(x) for x in empty.neurites]
        assert np.array_equal(given.soma.verts, empty.soma.verts)


# ── Smoke test on real data ──────────────────────────────────────────


@pytest.fixture(scope="session")
def reference_mesh():
    from pathlib import Path

    from skeliner.io import load_mesh

    mesh_path = Path(__file__).parent / "data" / "60427.obj"
    if not mesh_path.exists():
        pytest.skip("Reference mesh not available")
    return load_mesh(mesh_path)


class TestRealMesh:
    @deprecated
    def test_find_holes_returns_list(self, reference_mesh):
        holes = pre.find_holes(reference_mesh)
        assert isinstance(holes, list)

    @deprecated
    def test_find_fragments_returns_mask(self, reference_mesh):
        mask = pre.find_fragments(reference_mesh)
        assert mask.shape == (len(reference_mesh.faces),)
        assert mask.dtype == bool

    def test_fill_holes_appends_and_keeps_every_id(self, reference_mesh):
        org = pre.find_organelles(reference_mesh)
        out = pre.fill_holes(reference_mesh, organelles=org)
        n, nv = len(reference_mesh.faces), len(reference_mesh.vertices)
        assert np.array_equal(out.faces[:n], reference_mesh.faces)
        assert len(out.faces) > n, "fixture must have a hole to patch"
        assert out.faces[n:].max() < len(out.vertices)
        # a moved vertex moves by about an edge, not across the cell
        moved = np.linalg.norm(out.vertices[:nv] - reference_mesh.vertices, axis=1)
        assert moved.max() < 10 * pre._median_edge(reference_mesh)

    def test_find_disconnected_returns_list(self, reference_mesh):
        comps = pre.find_disconnected(reference_mesh)
        assert isinstance(comps, list)

    @deprecated
    def test_remove_fragments_preserves_main(self, reference_mesh):
        clean = pre.remove_fragments(reference_mesh)
        # Should keep at least 90% of faces (main component)
        assert _live_faces(clean) > _live_faces(reference_mesh) * 0.5


def test_compaction_remaps_names_rather_than_dropping_them():
    """Compaction reindexes faces and leaves the partition alone, so it is
    not a re-derive and not a reason to forget what a neurite was called."""
    mesh, soma, org = _ball_with_stub()
    stub = _arbor_pieces(mesh, soma, org)[0]
    comp = pre.break_up_mesh(mesh, soma, org, rescued=stub[:1])
    comp.neurites.name(0, "axon")

    _, out = pre.compact_mesh(mesh, comp)
    assert out.neurites.labels == ["axon"]
    assert out.neurites.swc_types == [2]


def test_compaction_leaves_unnamed_neurites_unnamed():
    mesh, soma, org = _ball_with_stub()
    stub = _arbor_pieces(mesh, soma, org)[0]
    comp = pre.break_up_mesh(mesh, soma, org, rescued=stub[:1])
    _, out = pre.compact_mesh(mesh, comp)
    assert out.neurites.named is False


class TestNearest:
    def test_exact_ties_go_to_the_lowest_index(self):
        from scipy.spatial import KDTree

        pts = np.array([[1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 5]], dtype=float)
        tree = KDTree(
            pts[::-1].copy()
        )  # reversed, so index order is not insertion order
        d, i = pre._nearest(tree, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]))
        assert i.tolist() == [1, 0]
        assert d.tolist() == [1.0, 1.0]

    def test_a_single_point_tree(self):
        from scipy.spatial import KDTree

        d, i = pre._nearest(KDTree(np.zeros((1, 3))), np.ones((2, 3)))
        assert i.tolist() == [0, 0]
