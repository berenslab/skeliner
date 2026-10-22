"""The Rust core: conversions, the dict and Topology paths agreeing, and
behaviour the fixtures pin down."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import trimesh

from skeliner import _rs, pre
from skeliner.io import load_mesh


def assert_identical(a, b):
    np.testing.assert_array_equal(a, b, strict=True)


def _dirty_mesh() -> trimesh.Trimesh:
    """A closed sphere spoiled the ways CAVE meshes are.

    A fin on one edge and a duplicated face make non-manifold edges; two
    faces repeat a vertex; one triangle touches the sphere at a vertex only;
    a separate pair of triangles floats beside it.
    """
    sphere = trimesh.creation.icosphere(subdivisions=1)
    v = np.asarray(sphere.vertices)
    f = np.asarray(sphere.faces)
    n = len(v)
    a, b, c = (int(x) for x in f[0])
    extra_v = np.array(
        [
            v[a] * 1.5,  # n: fin apex on edge (a, b)
            v[c] * 1.5,  # n+1, n+2: triangle on vertex c only
            v[c] * 1.6 + 0.1,
            [5.0, 0.0, 0.0],  # n+3 .. n+5: floating pair
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],
            [5.0, 1.0, 1.0],
        ]
    )
    extra_f = np.array(
        [
            [a, b, n],  # fin: (a, b) now has three faces
            [b, a, c],  # duplicate of face 0, reversed
            [c, n + 1, n + 2],  # touches the sphere at c only
            [a, a, b],  # degenerate
            [n + 3, n + 4, n + 4],  # degenerate
            [n + 3, n + 4, n + 5],
            [n + 4, n + 6, n + 5],
        ]
    )
    return trimesh.Trimesh(
        vertices=np.vstack([v, extra_v]), faces=np.vstack([f, extra_f]), process=False
    )


def _random_soup(seed: int, n_faces: int = 400, n_vertices: int = 40):
    """Random triangles over few vertices: many edges shared by 3+ faces."""
    rng = np.random.default_rng(seed)
    faces = rng.integers(0, n_vertices, size=(n_faces, 3))
    vertices = rng.random((n_vertices, 3))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _edge_face_counts(mesh) -> Counter:
    return Counter(len(fs) for fs in pre._edge_to_faces(mesh).values())


@pytest.fixture(scope="module")
def dirty_mesh():
    mesh = _dirty_mesh()
    counts = _edge_face_counts(mesh)
    assert counts[1] > 0 and counts[2] > 0 and max(counts) >= 3
    assert (~pre._non_degenerate(np.asarray(mesh.faces))).sum() == 2
    return mesh


@pytest.fixture(scope="module")
def reference_mesh():
    path = Path(__file__).parent / "data" / "60427.obj"
    if not path.exists():
        pytest.skip("Reference mesh not available")
    return load_mesh(path)


def _meshes(dirty_mesh, reference_mesh):
    soups = [_random_soup(s) for s in range(5)]
    for soup in soups:
        assert max(_edge_face_counts(soup)) >= 3
        assert not pre._non_degenerate(np.asarray(soup.faces)).all()
    return [dirty_mesh, reference_mesh, *soups]


class TestTopology:
    def test_reads_like_the_python_edge_map(self, dirty_mesh, reference_mesh):
        for mesh in _meshes(dirty_mesh, reference_mesh):
            faces = np.asarray(mesh.faces, dtype=np.int64)
            topo = _rs.Topology(faces, len(mesh.vertices))
            expected = pre._edge_to_faces(mesh)
            assert list(topo.as_dict().items()) == list(expected.items())
            assert list(topo.items()) == list(expected.items())
            assert topo.keys() == list(expected.keys())
            assert topo.values() == list(expected.values())
            assert list(topo) == list(expected)
            assert len(topo) == len(expected)

    def test_lookups(self, dirty_mesh):
        faces = np.asarray(dirty_mesh.faces, dtype=np.int64)
        topo = _rs.Topology(faces, len(dirty_mesh.vertices))
        expected = pre._edge_to_faces(dirty_mesh)
        a, b = next(k for k, fs in expected.items() if len(fs) >= 3)
        assert topo[(a, b)] == expected[(a, b)]
        assert (a, b) in topo
        assert topo.get((a, b)) == expected[(a, b)]
        missing = (len(dirty_mesh.vertices), len(dirty_mesh.vertices) + 1)
        assert missing not in topo
        assert topo.get(missing) is None
        assert topo.get(missing, []) == []
        with pytest.raises(KeyError):
            topo[missing]

    def test_csr_matches_the_lookups(self, dirty_mesh):
        faces = np.asarray(dirty_mesh.faces, dtype=np.int64)
        topo = _rs.Topology(faces, len(dirty_mesh.vertices))
        ptr, fs = topo.edge_faces_csr()
        edges = topo.edges()
        assert len(ptr) == len(edges) + 1 == topo.n_edges + 1
        for i, (a, b) in enumerate(edges):
            assert fs[ptr[i] : ptr[i + 1]].tolist() == topo[(int(a), int(b))]

    def test_refuses_an_index_past_the_vertex_count(self, dirty_mesh):
        faces = np.asarray(dirty_mesh.faces, dtype=np.int64)
        with pytest.raises(ValueError, match="names vertex"):
            _rs.Topology(faces, int(faces.max()))


class TestFaceEdgeComponents:
    def test_the_fixture_has_several_components(self, dirty_mesh):
        labels, main = pre._face_edge_components(dirty_mesh)
        assert (labels == -2).sum() == 2
        assert len(set(labels[labels >= 0].tolist())) >= 3
        assert main == 0


class TestFaceComponentsFast:
    def test_degenerate_faces_still_connect_through_their_edges(self, dirty_mesh):
        faces = np.asarray(dirty_mesh.faces)
        degenerate = np.flatnonzero(~pre._non_degenerate(faces))
        comps = pre._face_components_fast(faces, np.arange(len(faces)))
        # [a, a, b] shares (a, b) with the sphere, so it is not alone.
        alone = {int(c[0]) for c in comps if len(c) == 1}
        assert int(degenerate[0]) not in alone


@pytest.fixture
def square_pair():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]], dtype=np.int64)
    return vertices, faces


class TestRoundTrip:
    def test_returns_what_it_was_given(self, square_pair):
        vertices, faces = square_pair
        v, f = _rs.mesh_roundtrip(vertices, faces)
        assert_identical(v, vertices)
        assert_identical(f, faces)

    def test_accepts_non_contiguous_input(self, square_pair):
        vertices, faces = square_pair
        v, f = _rs.mesh_roundtrip(np.asfortranarray(vertices), np.asfortranarray(faces))
        assert_identical(v, vertices)
        assert_identical(f, faces)

    def test_refuses_a_face_naming_a_missing_vertex(self, square_pair):
        vertices, _ = square_pair
        with pytest.raises(ValueError, match="names vertex 5"):
            _rs.mesh_roundtrip(vertices, np.array([[0, 1, 5]], dtype=np.int64))

    def test_refuses_a_negative_index(self, square_pair):
        vertices, _ = square_pair
        with pytest.raises(ValueError, match="names vertex -1"):
            _rs.mesh_roundtrip(vertices, np.array([[0, -1, 2]], dtype=np.int64))

    def test_refuses_the_wrong_shape(self, square_pair):
        vertices, faces = square_pair
        with pytest.raises(ValueError, match="expected"):
            _rs.mesh_roundtrip(vertices[:, :2].copy(), faces)


class TestRefineComponentsWithFusions:
    def test_fusion_faces_are_walls_on_the_dirty_mesh(self, dirty_mesh):
        labels, main = pre._face_edge_components(dirty_mesh)
        fusions = pre.find_fusions(dirty_mesh)
        assert fusions
        fusion = {f for c in fusions for f in c}
        new, new_main = pre._refine_components_with_fusions(
            dirty_mesh, labels, main, fusions
        )
        assert set(np.flatnonzero(new == -3).tolist()) == fusion
        kept = new[new >= 0]
        assert kept.min() > labels.max()
        sizes = Counter(kept.tolist())
        assert sizes[new_main] == max(sizes.values())


def _patches(mesh, rng, n=12):
    """Selections that exercise border tracing: grown patches, scattered
    faces, and patches that swallow a degenerate face."""
    adj = pre._face_adjacency(mesh)
    n_faces = len(mesh.faces)
    for _ in range(n):
        seed = int(rng.integers(n_faces))
        sel, frontier = {seed}, [seed]
        target = int(rng.integers(1, max(2, n_faces // 3)))
        while frontier and len(sel) < target:
            nxt = []
            for f in frontier:
                for g in sorted(adj.get(f, ())):
                    if g not in sel:
                        sel.add(g)
                        nxt.append(g)
            frontier = nxt
        yield sel
    yield set(rng.choice(n_faces, size=max(1, n_faces // 4), replace=False).tolist())
    degenerate = np.flatnonzero(~pre._non_degenerate(np.asarray(mesh.faces)))
    if len(degenerate):
        yield {int(degenerate[0]), *range(min(40, n_faces))}


class TestBorderLoops:
    def test_topology_and_dict_agree(self, dirty_mesh, reference_mesh):
        rng = np.random.default_rng(3)
        n_loops = 0
        for mesh in _meshes(dirty_mesh, reference_mesh):
            faces = np.asarray(mesh.faces)
            topo = _rs.Topology(faces.astype(np.int64), len(mesh.vertices))
            as_dict = pre._edge_to_faces(mesh)
            for sel in _patches(mesh, rng):
                loops = pre._trace_border_loops(sel, topo, faces)
                n_loops += len(loops)
                assert loops == pre._trace_border_loops(sel, as_dict, faces)
                assert loops == pre._trace_border_loops(sel, as_dict)
                assert pre._removal_would_sever(
                    sel, topo, faces
                ) == pre._removal_would_sever(sel, as_dict, faces)
        assert n_loops > 20

    def test_tip_expansion_topology_and_dict_agree(self, dirty_mesh, reference_mesh):
        rng = np.random.default_rng(5)
        for mesh in _meshes(dirty_mesh, reference_mesh):
            faces = np.asarray(mesh.faces)
            topo = _rs.Topology(faces.astype(np.int64), len(mesh.vertices))
            as_dict = pre._edge_to_faces(mesh)
            adj = pre._face_adjacency(mesh, as_dict)
            for _ in range(10):
                tip = rng.choice(len(faces), size=3, replace=False).tolist()
                assert pre._expand_tip_to_good_rim(
                    tip, topo, None, faces=faces
                ) == pre._expand_tip_to_good_rim(tip, as_dict, adj, faces=faces)


class TestGapStitchSelection:
    def test_sever_cost_topology_and_dict_agree(self, dirty_mesh, reference_mesh):
        from .test_pre import _band, _subdivided_tube

        tube = _subdivided_tube()
        meshes = [tube, *_meshes(dirty_mesh, reference_mesh)]
        rng = np.random.default_rng(13)
        costs = set()
        for mesh in meshes:
            topo = _rs.Topology(np.asarray(mesh.faces, np.int64), len(mesh.vertices))
            adj = pre._face_adjacency(mesh, pre._edge_to_faces(mesh))
            sels = list(_patches(mesh, rng, n=6))
            if mesh is tube:
                sels.append(set(np.flatnonzero(_band(mesh, -40.0, 40.0)).tolist()))
            for sel in sels:
                for budget in (0, 5, 100, 10**9):
                    cost = pre._sever_cost(sel, None, budget, topology=topo)
                    assert cost == pre._sever_cost(sel, adj, budget)
                    costs.add(cost)
        assert len(costs) > 3

    def test_expand_selection_per_group_topology_and_dict_agree(
        self, dirty_mesh, reference_mesh
    ):
        rng = np.random.default_rng(17)
        for mesh in _meshes(dirty_mesh, reference_mesh):
            faces = np.asarray(mesh.faces)
            topo = _rs.Topology(faces.astype(np.int64), len(mesh.vertices))
            as_dict = pre._edge_to_faces(mesh)
            adj = pre._face_adjacency(mesh, as_dict)
            for _ in range(6):
                sel = rng.choice(len(faces), size=4, replace=False).tolist()
                assert pre._expand_selection_per_group(
                    sel, topo, None, faces=faces
                ) == pre._expand_selection_per_group(sel, as_dict, adj, faces=faces)


class TestFusions:
    def test_removing_a_fusion_splits_the_fan_it_leaves(self):
        # two triangles meeting at vertex 0, held together by face 2
        bowtie = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 1, 0], [-1, 0, 0]],
            faces=[[0, 1, 2], [0, 3, 4], [0, 2, 3]],
            process=False,
        )
        out = pre.remove_fusions(bowtie, fusions=[[2]])
        faces = np.asarray(out.faces)
        assert len(out.vertices) == len(bowtie.vertices) + 1
        assert faces[2].tolist() == [0, 0, 0]
        assert not set(faces[0].tolist()) & set(faces[1].tolist())
        assert_identical(out.vertices[faces[1]], bowtie.vertices[[0, 3, 4]])


class TestKdTree:
    def _grid_points(self, rng, n):
        # integer coordinates, as on a voxel grid: exact distance ties abound
        return rng.integers(0, 12, size=(n, 3)).astype(np.float64)

    def test_nearest_matches_scipy_with_lowest_index_ties(self):
        from scipy.spatial import KDTree

        rng = np.random.default_rng(31)
        points = np.unique(self._grid_points(rng, 600), axis=0)
        queries = self._grid_points(rng, 2000) + 0.5
        d_ref, i_ref = pre._nearest(KDTree(points), queries)
        d, i = _rs.KdTree(points).query(queries)
        assert_identical(d, d_ref)
        assert_identical(i, i_ref)
        _, i_scipy = KDTree(points).query(queries)
        assert (i_scipy != i_ref).any(), "the fixture should contain ties"

    def test_ball_queries_match_scipy(self):
        from scipy.spatial import KDTree

        rng = np.random.default_rng(37)
        points = self._grid_points(rng, 3000)
        queries = rng.random((300, 3)) * 12
        ref = KDTree(points).query_ball_point(queries, 2.0, return_sorted=True)
        ours = _rs.KdTree(points).query_ball_point(queries, 2.0)
        assert [h.tolist() for h in ours] == [list(r) for r in ref]
        assert sum(len(r) for r in ref) > 1000


def _islands():
    """A main sphere and small spheres placed to hit every rule of
    ``find_disconnected``; returns the mesh and the face range of each."""
    parts = {
        "main": trimesh.creation.icosphere(subdivisions=3, radius=10.0),
        "enclosed": trimesh.creation.icosphere(subdivisions=1, radius=1.0),
        "outside": trimesh.creation.icosphere(subdivisions=1, radius=1.0),
        "far": trimesh.creation.icosphere(subdivisions=1, radius=1.0),
        "in_soma": trimesh.creation.icosphere(subdivisions=1, radius=1.0),
        "organelle": trimesh.creation.icosphere(subdivisions=1, radius=1.0),
    }
    offsets = {
        "enclosed": (0, 0, 8.5),
        "outside": (0, 0, 12.0),
        "far": (50, 0, 0),
        "in_soma": (30, 0, 0),
        "organelle": (0, 30, 0),
    }
    for name, off in offsets.items():
        parts[name].vertices += off
    speck = trimesh.Trimesh(
        vertices=[[0, -40, 0], [1, -40, 0], [0, -39, 0]], faces=[[0, 1, 2]]
    )
    ranges, start = {}, 0
    for name, m in [*parts.items(), ("speck", speck)]:
        ranges[name] = (start, start + len(m.faces))
        start += len(m.faces)
    mesh = trimesh.util.concatenate([*parts.values(), speck])
    return mesh, ranges


class TestFindDisconnected:
    def test_every_rule_fires(self):
        from skeliner.dataclass import Organelles, Soma

        mesh, ranges = _islands()
        n = len(mesh.faces)
        org_mask = np.zeros(n, dtype=bool)
        org_mask[slice(*ranges["organelle"])] = True
        org = Organelles(
            pocket=org_mask, isolated=np.zeros(n, bool), expanded=np.zeros(n, bool)
        )
        soma = Soma(
            center=np.array([30.0, 0, 0]), axes=np.array([5.0, 4.0, 3.0]), R=np.eye(3)
        )
        found = pre.find_disconnected(mesh, soma=soma, organelles=org)
        kept = {next(k for k, (a, b) in ranges.items() if a <= c[0] < b) for c in found}
        assert kept == {"outside", "far"}
        # without the soma and organelles, those come back
        found = pre.find_disconnected(mesh, soma=None, organelles=None)
        kept = {next(k for k, (a, b) in ranges.items() if a <= c[0] < b) for c in found}
        assert {"outside", "far"} <= kept and "enclosed" not in kept


def _hollow_soma(gap_above: float = 1000.0) -> np.ndarray:
    """Vertices of a sphere's surface cut to |z| <= 1500 nm, in rings.

    Rings with |z| > *gap_above* miss a 40 degree arc, so the Z-scan's first
    pass finds no void there and the hull-guided second pass has work.
    """
    pts = []
    for z in np.arange(-1500.0, 1501.0, 70.0):
        rho = np.sqrt(3000.0**2 - z**2)
        n = int(2 * np.pi * rho / 80.0)
        for a in np.arange(n) * (2 * np.pi / n):
            if abs(z) > gap_above and a < np.radians(40.0):
                continue
            pts.append([np.round(rho * np.cos(a)), np.round(rho * np.sin(a)), z])
    return np.array(pts, dtype=np.float64)


class TestZScan:
    def test_the_hull_pass_lengthens_the_chain(self):
        v = _hollow_soma()
        both_passes = pre._z_scan(v)
        first_only = pre._z_scan(v, retry_broken=False)
        # the second pass recovers the broken levels
        assert len(both_passes[1]) > len(first_only[1]) >= 5

    def test_no_chain(self):
        v = _hollow_soma()
        v = v[np.abs(v[:, 2]) < 200]
        assert pre._z_scan(v)[1] is None

    def test_find_nucleus_center(self):
        mesh = trimesh.Trimesh(vertices=_hollow_soma(), process=False)
        assert len(pre.find_nucleus_center(mesh)["slices"]) >= 5


def _revolved(profile, spacing=180.0):
    """A surface of revolution about z through ``(r, z)`` rings of vertices
    about *spacing* apart, consecutive rings zipped by angle; open at both
    ends, with integer coordinates like CAVE's."""
    verts, rings = [], []
    for r, z in profile:
        n = max(8, int(round(2 * np.pi * r / spacing)))
        a = np.arange(n) * (2 * np.pi / n)
        rings.append(list(range(len(verts), len(verts) + n)))
        verts.extend(np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n, z)]))
    faces = []
    for ra, rb in zip(rings, rings[1:]):
        na, nb = len(ra), len(rb)
        i = j = 0
        while i < na or j < nb:
            if j >= nb or (i < na and (i + 1) / na <= (j + 1) / nb):
                faces.append([ra[i % na], ra[(i + 1) % na], rb[j % nb]])
                i += 1
            else:
                faces.append([ra[i % na], rb[(j + 1) % nb], rb[j % nb]])
                j += 1
    return np.round(np.array(verts)), np.array(faces)


def _soma_cell():
    """A 3 um soma tapering into two neurites, with a small sphere inside
    marked as an organelle: ``(mesh, organelles)``."""
    prof = []
    for z in np.arange(-15000.0, 15001.0, 150.0):
        r = np.sqrt(max(3000.0**2 - z**2, 0.0))
        prof.append((max(r, 450.0 if z < 0 else 300.0), z))
    v, f = _revolved(prof)
    inner = trimesh.creation.icosphere(subdivisions=2, radius=300.0)
    n_outer = len(f)
    f = np.vstack([f, inner.faces + len(v)])
    v = np.vstack([v, np.round(inner.vertices + [800.0, 0.0, 0.0])])
    mesh = trimesh.Trimesh(v, f, process=False)
    org = pre.find_organelles(mesh)
    org.isolated[n_outer:] = True
    return mesh, org


class TestFindSomaViaRingCutoff:
    def test_finds_the_soma_and_leaves_out_the_organelle(self):
        mesh, org = _soma_cell()
        soma = pre.find_soma_via_ring_cutoff(mesh, organelles=org)
        assert soma is not None and soma.nucleus is not None
        assert np.abs(soma.center).max() < 100.0
        # the 3 um body, not the neurites tapering away from it
        assert np.abs(mesh.vertices[soma.verts][:, 2]).max() <= 3000.0
        inner = np.unique(np.asarray(mesh.faces)[org.isolated])
        assert not np.isin(soma.verts, inner).any()

    def test_no_nucleus(self):
        # every cross-section's hollow is under the 500 nm void minimum
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=300.0)
        assert pre.find_soma_via_ring_cutoff(mesh) is None


class TestBreakUpMesh:
    @staticmethod
    def _cut(mesh, org):
        """An organelle band around the upper neurite, which strands its tip
        behind organelle faces."""
        z = mesh.triangles_center[:, 2]
        band = (z > 9000) & (z < 9400)
        org.isolated[band] = True
        return int(band.sum())

    def test_the_stranded_tip_is_trapped_unless_rescued(self):
        mesh, org = _soma_cell()
        assert self._cut(mesh, org) > 0
        soma = pre.find_soma_via_ring_cutoff(mesh, organelles=org)
        found = []
        for kwargs in (
            {},
            {"rescued": [int(np.argmax(mesh.triangles_center[:, 2]))]},
            {"released": np.arange(len(mesh.faces)), "claim_min_faces": 50},
        ):
            for s in (soma, None):
                comps = pre.break_up_mesh(mesh, s, org, **kwargs)
                found.append(int(comps.organelles.expanded.sum()))
        # the stranded tip is trapped unless a claim releases it
        assert found[0] > 0 and found[2] == 0


def _capped_tube_x(x_open, x_cap, side, *, turn=0.0, n=48):
    """A tube along x, open at *x_open*, ending in a flat disk at *x_cap*
    whose normal points along *side*, joined to the tube by a tilted
    annulus; *turn* rotates the vertices by that fraction of a step."""
    r_tube, r_disk, tilt = 600.0, 450.0, 40.0
    ang = (np.arange(n) + turn) * (2 * np.pi / n)
    rings = [(x, r_tube) for x in np.linspace(x_open, x_cap - side * tilt, 12)]
    rings += [(x_cap, r_disk * k / 6) for k in range(6, 0, -1)]
    verts, ids = [], []
    for x, r in rings:
        ids.append(list(range(len(verts), len(verts) + n)))
        verts.extend([[x, r * np.cos(a), r * np.sin(a)] for a in ang])
    center = len(verts)
    verts.append([x_cap, 0.0, 0.0])
    faces = []
    for ra, rb in zip(ids, ids[1:]):
        for i in range(n):
            j = (i + 1) % n
            faces += [[ra[i], ra[j], rb[i]], [ra[j], rb[j], rb[i]]]
    faces += [[ids[-1][i], ids[-1][(i + 1) % n], center] for i in range(n)]
    v, f = np.round(np.array(verts)), np.array(faces)
    # orient: the disk along *side*, the tube away from the axis
    c = v[f].mean(axis=1)
    out = np.zeros_like(c)
    on_cap = np.abs(c[:, 0] - x_cap) < tilt + 1
    out[on_cap, 0] = side
    out[~on_cap, 1:] = c[~on_cap, 1:]
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = (nrm * out).sum(axis=1) < 0
    f[flip] = f[flip][:, ::-1]
    return v, f


def _sandwich():
    """Two tubes whose capped ends face each other across x = 0, the
    marching-cubes double layer at a chunk boundary, plus a flap folded
    flat onto itself at x = 0."""
    va, fa = _capped_tube_x(-3000.0, -1.0, +1)
    vb, fb = _capped_tube_x(3000.0, 1.0, -1, turn=0.5)
    flap_v = np.array([[0, 5000, 0], [0, 6000, 0], [0, 5500, 800], [0, 5500, 700]])
    flap_f = np.array([[0, 1, 2], [0, 3, 1]])
    parts = [(va, fa), (vb, fb), (flap_v.astype(float), flap_f)]
    verts, faces, off = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)


def _has_fold(mesh, patch) -> bool:
    """Does an up face of *patch* share an edge with a down face?"""
    up, down = set(patch["up_faces"]), set(patch["down_faces"])
    sides: dict[tuple[int, int], set[str]] = {}
    for fi in patch["faces"]:
        a, b, c = (int(v) for v in mesh.faces[fi])
        for e in ((a, b), (b, c), (a, c)):
            e = (min(e), max(e))
            sides.setdefault(e, set()).add(
                "up" if fi in up else "down" if fi in down else ""
            )
    return any({"up", "down"} <= s for s in sides.values())


class TestParallelPatches:
    def test_the_sandwich_is_found_and_zipped(self):
        mesh = _sandwich()
        patches = pre.find_parallel_patches(mesh, boundaries={0: np.array([0])})
        assert len(patches) == 3
        assert any(_has_fold(mesh, p) for p in patches)
        out = pre.remove_parallel_patches(mesh, patches=patches)
        # the caps are gone and the rims zipped: one tube
        labels, _ = pre._face_edge_components(out)
        assert len(set(labels.tolist()) - {-2}) == 1
        assert len(out.faces) > len(mesh.faces)
