"""Contact sites between two meshes."""

import numpy as np
import pytest
import trimesh

import skeliner as sk
from skeliner import pair


def _tube(
    axis,
    center,
    radius=200.0,
    length=4000.0,
    step=25.0,
    around=64,
    cap=False,
    bead=None,
):
    """A tube along unit vector *axis* through *center*, finely meshed, faces
    wound outward; open, or with *cap* a flat disc closing its +axis end.
    *bead* ``(peak, width)`` swells it at *center*: the radius rises to
    *peak* there and falls back over *width* (a Gaussian)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ang = np.arange(around) * (2 * np.pi / around)
    unit = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)
    zs = np.arange(-length / 2, length / 2 + step / 2, step)
    radii = np.full(len(zs), radius)
    if bead is not None:
        peak, width = bead
        radii += (peak - radius) * np.exp(-((zs / width) ** 2))
    verts = np.vstack([r * unit + z * axis for r, z in zip(radii, zs)]) + center
    faces = []
    for k in range(len(zs) - 1):
        for i in range(around):
            j = (i + 1) % around
            a, b = k * around + i, k * around + j
            c, d = a + around, b + around
            faces += [[a, b, d], [a, d, c]]
    n_side = len(faces)
    if cap:
        last = (len(zs) - 1) * around
        rings = [r * radius for r in np.linspace(1, 0, 9)[1:-1]]
        prev = list(range(last, last + around))
        for r in rings:
            ids = list(range(len(verts), len(verts) + around))
            verts = np.vstack([verts, r * unit + zs[-1] * axis + center])
            for i in range(around):
                j = (i + 1) % around
                faces += [[prev[i], prev[j], ids[j]], [prev[i], ids[j], ids[i]]]
            prev = ids
        tip = len(verts)
        verts = np.vstack([verts, zs[-1] * axis + center])
        faces += [[prev[i], prev[(i + 1) % around], tip] for i in range(around)]
    mesh = trimesh.Trimesh(verts, np.array(faces), process=False)
    out = mesh.triangles_center - center
    out -= np.outer(out @ axis, axis)
    out[n_side:] = axis
    flip = (mesh.face_normals * out).sum(1) < 0
    if flip.any():
        f = np.array(mesh.faces)
        f[flip] = f[flip][:, ::-1]
        mesh = trimesh.Trimesh(verts, f, process=False)
    return mesh


@pytest.fixture(scope="module")
def tube_z():
    return _tube([0, 0, 1], [0.0, 0.0, 0.0])


def test_the_fixture_faces_outward(tube_z):
    radial = tube_z.triangles_center.copy()
    radial[:, 2] = 0
    assert ((tube_z.face_normals * radial).sum(1) > 0).all()


def test_crossing_tubes_touch_at_one_small_site(tube_z):
    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    sites = pair.scan_contacts(tube_z, crossing, tol=20.0)
    assert len(sites.faces_A) == 1
    assert len(sites.faces_A[0]) and len(sites.faces_B[0])
    # a kiss: a patch of a few hundred nm across, not a band
    assert sites.area_mean[0] < 0.01 * tube_z.area
    assert np.all(np.isfinite(sites.seeds_A)) and np.all(np.isfinite(sites.bbox))


def test_tubes_side_by_side_touch_along_their_length(tube_z):
    beside = _tube([0, 0, 1], [405.0, 0.0, 0.0])
    sites = pair.compute_contact_stats(
        tube_z, beside, pair.scan_contacts(tube_z, beside, tol=20.0)
    )
    assert len(sites.faces_A) == 1
    assert sites.stats_A["aspect"][0] > 3
    assert sites.stats_pair["normal_opposition_dot"][0] < -0.9


def test_tubes_apart_have_no_sites(tube_z):
    far = _tube([1, 0, 0], [0.0, 1000.0, 0.0])
    sites = pair.scan_contacts(tube_z, far, tol=20.0, return_pairs=True)
    assert sites.faces_A == [] and sites.faces_B == [] and sites.pairs_AB == []
    assert sites.area_mean.shape == (0,) and sites.bbox.shape == (0, 2, 3)


def test_pairs_name_faces_of_their_site(tube_z):
    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    sites = pair.scan_contacts(tube_z, crossing, tol=20.0, return_pairs=True)
    (pairs,) = sites.pairs_AB
    assert len(pairs)
    fa = set(sites.faces_A[0].tolist())
    fb = set(sites.faces_B[0].tolist())
    a_side = [int(a) in fa or int(b) in fb for a, b, _ in pairs]
    assert all(a_side)
    # every touching face of either side is in a pair
    assert fa <= set(pairs[:, 0].astype(int).tolist())
    assert fb <= set(pairs[:, 1].astype(int).tolist())


def test_seeded_pairs_stay_with_their_seeds(tube_z):
    # seeds at two crossings, interleaved: the seeds come back cluster by
    # cluster, out of order, and each seed's pairs once went to whichever
    # seed sat in its slot
    lo, hi = (_tube([1, 0, 0], [0.0, 405.0, z]) for z in (-1000.0, 1000.0))
    crossings = trimesh.Trimesh(
        np.vstack([lo.vertices, hi.vertices]),
        np.vstack([lo.faces, hi.faces + len(lo.vertices)]),
        process=False,
    )
    seeds = np.array([[0.0, 202.5, z] for z in (-1000.0, 1000.0) * 2])
    sites = pair.map_contact_sites(tube_z, crossings, seeds, return_pairs=True)
    assert len(sites.faces_A) == 2
    for fa, fb, pairs in zip(sites.faces_A, sites.faces_B, sites.pairs_AB):
        assert len(pairs)
        fa, fb = set(fa.tolist()), set(fb.tolist())
        assert all(int(a) in fa or int(b) in fb for a, b, _ in pairs)


def _labelled(a, b):
    sites = pair.scan_contacts(a, b, tol=20.0)
    assert len(sites.faces_A) == 1, len(sites.faces_A)
    return sites


def _geometry(sites):
    return {k: float(v[0]) for k, v in sites.geometry.items()}


def test_crossing_branches_pass_straight_through_at_right_angles(tube_z):
    sites = _labelled(tube_z, _tube([1, 0, 0], [0.0, 405.0, 0.0]))
    g = _geometry(sites)
    assert sites.kind.tolist() == ["crossing"]
    assert g["exits_A"] == 2 and g["exits_B"] == 2
    assert g["crossing_deg"] > 85
    assert abs(g["thickness_A"] - 400) < 20 and abs(g["thickness_B"] - 400) < 20
    assert g["ball_A"] == pytest.approx(3 * g["thickness_A"])
    assert abs(g["swelling_A"] - 1) < 0.1 and abs(g["swelling_B"] - 1) < 0.1


def test_a_swelling_where_it_touches_is_a_blob(tube_z):
    bead = _tube([1, 0, 0], [0.0, 505.0, 0.0], radius=100.0, bead=(300.0, 200.0))
    sites = _labelled(tube_z, bead)
    g = _geometry(sites)
    assert sites.kind.tolist() == ["blob"]
    # measured across the bead, against the thin tube it is on
    assert abs(g["thickness_B"] - 600) < 40
    assert g["swelling_B"] > 2.5 and abs(g["swelling_A"] - 1) < 0.1
    assert g["exits_B"] == 2


def test_branches_at_a_shallow_angle_are_not_a_crossing(tube_z):
    tilt = np.radians(20.0)
    beside = _tube([0, np.sin(tilt), np.cos(tilt)], [405.0, 0.0, 0.0])
    sites = _labelled(tube_z, beside)
    g = _geometry(sites)
    assert g["exits_A"] == 2 and g["exits_B"] == 2
    assert abs(g["crossing_deg"] - 20) < 3
    assert sites.kind.tolist() == ["other"]


def test_a_branch_ending_at_the_site_leaves_its_ball_once(tube_z):
    tip = _tube([0, -1, 0], [0.0, 1205.0, 0.0], length=2000.0, cap=True)
    g = _geometry(_labelled(tube_z, tip))
    assert g["exits_A"] == 2 and g["exits_B"] == 1
    assert np.isnan(g["crossing_deg"])


def test_unlabelled_sites_have_no_kind(tube_z):
    sites = pair.scan_contacts(
        tube_z, _tube([1, 0, 0], [0.0, 405.0, 0.0]), tol=20.0, label=False
    )
    assert sites.kind is None and sites.geometry is None


def test_relabelling_reuses_the_geometry_from_the_same_ball(tube_z):
    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    sites = pair.scan_contacts(tube_z, crossing, tol=20.0)
    geometry = sites.geometry
    pair.label_contacts(tube_z, crossing, sites, crossing_min_deg=95.0)
    assert sites.geometry is geometry
    assert sites.kind.tolist() == ["other"]
    assert sites.meta["label"]["crossing_min_deg"] == 95.0
    pair.label_contacts(tube_z, crossing, sites, ball_scale=2.0)
    assert sites.geometry is not geometry
    assert sites.geometry["ball_A"][0] == pytest.approx(
        2 * sites.geometry["thickness_A"][0]
    )
    assert sites.kind.tolist() == ["crossing"]


def test_sites_without_geometry_are_other(tube_z):
    far = _tube([1, 0, 0], [0.0, 1000.0, 0.0])
    sites = pair.scan_contacts(tube_z, far, tol=20.0)
    assert sites.kind.shape == (0,)
    assert all(v.shape == (0,) for v in sites.geometry.values())


def test_picking_sites_keeps_their_faces_kinds_and_stats(tube_z):
    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    beside = _tube([0, 0, 1], [405.0, 0.0, 0.0])
    both = trimesh.util.concatenate([crossing, beside])
    sites = pair.compute_contact_stats(
        tube_z, both, pair.scan_contacts(tube_z, both, tol=20.0)
    )
    assert sorted(sites.kind.tolist()) == ["crossing", "other"]
    picked = pair.pick_contacts(sites, sites.kind == "other")
    i = int(np.flatnonzero(sites.kind == "other")[0])
    assert len(picked.faces_A) == 1 and picked.kind.tolist() == ["other"]
    assert np.array_equal(picked.faces_A[0], sites.faces_A[i])
    assert picked.stats_A["aspect"][0] == sites.stats_A["aspect"][i]
    assert picked.geometry["swelling_A"][0] == sites.geometry["swelling_A"][i]


def test_kinds_survive_a_round_trip_and_old_files_load(tube_z, tmp_path):
    from skeliner.io import load_contact_sites_npz

    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    sites = pair.scan_contacts(tube_z, crossing, tol=20.0)
    sites.to_npz(tmp_path / "labelled.npz")
    back = load_contact_sites_npz(tmp_path / "labelled.npz")
    assert back.kind.tolist() == sites.kind.tolist()
    assert back.geometry.keys() == sites.geometry.keys()
    assert back.geometry["swelling_A"][0] == sites.geometry["swelling_A"][0]
    plain = pair.scan_contacts(tube_z, crossing, tol=20.0, label=False)
    plain.to_npz(tmp_path / "plain.npz")
    back = load_contact_sites_npz(tmp_path / "plain.npz")
    assert back.kind is None and back.geometry is None


# ---------------------------------------------------------------------
# the Python path computes what the Rust path does
# ---------------------------------------------------------------------
def test_contact_faces_on_both_paths_agree(tube_z):
    from skeliner import _rs
    from skeliner.python import pair as python_pair

    from ._parity import assert_parity

    other = _tube([1, 0, 0], [0.0, 405.0, 0.0])  # crosses tube_z, touching it
    near = np.flatnonzero(
        np.linalg.norm(tube_z.triangles_center - [0.0, 200.0, 0.0], axis=1) < 300.0
    )
    assert len(near) > 100
    args = (
        np.asarray(tube_z.triangles_center[near], np.float64),
        np.asarray(tube_z.face_normals[near], np.float64),
        np.asarray(other.triangles, np.float64).reshape(-1, 9),
        np.asarray(other.face_normals, np.float64),
    )
    for tol, max_dot in ((20.0, -0.2), (60.0, 1.0)):
        rust = _rs.pair.contact_faces(*args, tol, max_dot)
        assert len(rust[0]) > 0
        assert_parity(rust, python_pair.contact_faces(*args, tol, max_dot))


def test_first_hits_on_both_paths_agree(tube_z):
    from skeliner import _rs
    from skeliner.python import pair as python_pair

    from ._parity import assert_parity

    tris = np.asarray(tube_z.triangles, np.float64).reshape(-1, 9)
    normals = np.asarray(tube_z.face_normals, np.float64)
    faces = np.arange(0, len(tris), 37)
    rays = (tube_z.triangles_center[faces], -normals[faces], faces, 1e-6)
    rust = _rs.pair.TriangleBvh(tris, normals).first_hits(*rays)
    assert np.isfinite(rust).sum() > len(faces) // 2  # inward rays cross the tube
    assert_parity(rust, python_pair.TriangleBvh(tris, normals).first_hits(*rays))


def _ridge(n=8, step=10.0, slope=0.5):
    """An n x n grid of squares, two triangles each, faces up, folded into a
    ridge along x = n * step / 2, each side falling *slope* per unit x."""
    xs = np.arange(n + 1) * step
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    gz = -slope * np.abs(gx - n * step / 2)
    verts = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b, c, d = a + n + 1, a + 1, a + n + 2
            faces += [[a, b, d], [a, d, c]]
    return verts, np.array(faces)


def test_nearest_points_on_both_paths_agree():
    from skeliner import _rs
    from skeliner.python import pair as python_pair

    from ._parity import assert_parity

    verts, faces = _ridge()
    normals = sk.Mesh(verts, faces).face_normals
    rng = np.random.default_rng(1)
    on_grid = verts[rng.choice(len(verts), 40)]
    edges = (verts[faces[:, 0]] + verts[faces[:, 1]]) / 2
    # ties: above vertices and edges (six faces and two), on the surface,
    # under it, astride the ridge, and anywhere
    pts = np.vstack(
        [
            on_grid + [0.0, 0.0, 3.0],
            edges[:40] + [0.0, 0.0, 3.0],
            on_grid,
            edges[40:80] - [0.0, 0.0, 3.0],
            np.c_[
                40.0 + rng.uniform(-2, 2, 40), rng.uniform(0, 80, 40), np.full(40, 5.0)
            ],
            rng.uniform([-10, -10, -30], [90, 90, 10], (200, 3)),
        ]
    )
    args = (verts, faces, normals, pts)
    rust = _rs.pair.closest_points(*args)
    assert_parity(rust, python_pair.closest_points(*args))
    # astride the ridge, the side the point leans to
    lean = slice(160, 200)
    assert ((normals[rust[2][lean]] @ [1.0, 0, 0] > 0) == (pts[lean, 0] > 40)).all()
    for closest_points in (_rs.pair.closest_points, python_pair.closest_points):
        with pytest.raises(ValueError, match="no faces"):
            closest_points(verts, faces[:0], normals[:0], pts)
        with pytest.raises(ValueError, match="one normal per face"):
            closest_points(verts, faces, normals[1:], pts)
        with pytest.raises(ValueError, match="points must be finite"):
            closest_points(verts, faces, normals, pts + [np.nan, 0.0, 0.0])


def test_nearest_points_are_trimeshs():
    from trimesh.proximity import ProximityQuery

    from skeliner import _rs
    from skeliner.python import pair as python_pair

    from ._parity import assert_parity

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=100.0)
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(400, 3))
    pts *= rng.uniform(60.0, 160.0, (400, 1)) / np.linalg.norm(pts, axis=1)[:, None]
    ref_close, ref_dist, ref_face = ProximityQuery(sphere).on_surface(pts)
    verts, faces = np.asarray(sphere.vertices), np.asarray(sphere.faces)
    normals = np.asarray(sphere.face_normals)
    # every face's squared distance: a nearest within 1e-8 of the next ties
    tris = verts[faces]
    q = python_pair.closest_point_trimesh(
        np.repeat(tris[None], len(pts), 0).reshape(-1, 3, 3),
        np.repeat(pts, len(tris), 0),
    )
    d2 = np.sort(((np.repeat(pts, len(tris), 0) - q) ** 2).sum(1).reshape(len(pts), -1))
    clear = d2[:, 1] - d2[:, 0] >= 1e-8
    assert 0 < clear.sum() < len(pts)

    def facing(face, close):
        v = pts - close
        return (normals[face] * v).sum(1) / np.linalg.norm(v, axis=1)

    for closest_points in (_rs.pair.closest_points, python_pair.closest_points):
        close, dist, face = closest_points(verts, faces, normals, pts)
        assert (face[clear] == ref_face[clear]).all()
        assert_parity((close[clear], dist[clear]), (ref_close[clear], ref_dist[clear]))
        # a tie goes to the face facing the point best, of all that tie
        assert np.allclose(dist, ref_dist, rtol=0, atol=1e-9)
        tied = ~clear
        assert (
            facing(face, close)[tied] >= facing(ref_face, ref_close)[tied] - 1e-12
        ).all()


def test_a_faces_reach_on_both_paths_agrees():
    from skeliner import _rs
    from skeliner.python import pair as python_pair

    verts, faces = _ridge()
    # one large face, its corners far from its centre
    big = [[-500.0, -500.0, -300.0], [600.0, -500.0, -300.0], [0.0, 600.0, -300.0]]
    verts = np.vstack([verts, big])
    faces = np.vstack([faces, len(verts) - 3 + np.arange(3)])
    tris = verts[faces]
    centers = tris.mean(1)
    expect = np.linalg.norm(tris - centers[:, None], axis=2).max()
    for face_reach in (_rs.pair.face_reach, python_pair.face_reach):
        assert np.isclose(face_reach(verts, faces), expect, rtol=1e-12)
        assert face_reach(verts, faces[:0]) == 0.0
        nan = verts.copy()
        nan[faces[3, 1], 0] = np.nan
        assert np.isnan(face_reach(nan, faces))
    assert _rs.pair.face_reach(verts, faces) == python_pair.face_reach(verts, faces)


def test_seeds_project_as_on_the_whole_mesh():
    # the seeded path projects its seeds on the faces near them, found with
    # the tree over the face centroids, and gets what the whole mesh gives
    from skeliner.pair import _build_face_kdt, _nearest_on_surface

    from ._parity import assert_parity

    verts, faces = _ridge()
    big = [[-500.0, -500.0, -300.0], [600.0, -500.0, -300.0], [0.0, 600.0, -300.0]]
    verts = np.vstack([verts, big])
    mesh = sk.Mesh(verts, np.vstack([faces, len(verts) - 3 + np.arange(3)]))
    rng = np.random.default_rng(3)
    pts = np.vstack(
        [
            rng.uniform([-10, -10, -30], [90, 90, 10], (200, 3)),
            verts[:40],  # on the surface, where faces tie
            rng.uniform(-2000, 2000, (20, 3)),  # far, nearest the large face
        ]
    )
    for path in ("rust", "python"):
        with sk.backend(path):
            kdt = _build_face_kdt(mesh)[0]
            near = _nearest_on_surface(mesh, pts, kdt)
            assert_parity(near, _nearest_on_surface(mesh, pts))
            with pytest.raises(ValueError, match="must be finite"):
                _nearest_on_surface(mesh, pts + [np.nan, 0.0, 0.0], kdt)
    assert (near[2] == len(faces)).sum() > 10


def test_contact_sites_on_both_paths_agree():
    from ._parity import assert_parity, both

    # small and coarse, for the Python path: still ~7 nm apart where they cross
    small = dict(length=1200.0, step=40.0, around=32)
    along_z = _tube([0, 0, 1], [0.0, 0.0, 0.0], **small)
    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0], **small)

    def sites():
        s = pair.scan_contacts(along_z, crossing, tol=20.0)
        return [np.concatenate(f) for f in (s.faces_A, s.faces_B)], list(s.kind)

    rust, python = both(sites)
    assert len(rust[0][0])
    assert_parity(rust, python)


def test_the_seeded_path_takes_skeletons():
    # find_contact_seeds and approximate_contact_sites take skeletons; the
    # mesh conversion once applied to every A and B broke them
    from skeliner import Skeleton, Soma, pair

    def line(y):
        nodes = np.c_[np.arange(10.0), np.full(10, y), np.zeros(10)]
        return Skeleton(
            soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
            nodes=nodes,
            radii={"median": np.full(10, 0.6)},
            edges=np.c_[np.arange(9), np.arange(1, 10)],
            ntype=None,
            meta={"unit": "nm"},
        )

    a, b = line(0.0), line(1.0)
    seeds = pair.find_contact_seeds(a, b, delta=0.1)
    assert len(seeds.idx_a) == 9 and np.allclose(seeds.center_gap, -0.2)
    pair.approximate_contact_sites(a, b, seeds)


def _skeleton(nodes, radii):
    from skeliner import Skeleton, Soma

    nodes = np.asarray(nodes, float)
    return Skeleton(
        soma=Soma.from_sphere(nodes[0], float(radii[0]), verts=None),
        nodes=nodes,
        radii={"median": np.asarray(radii, float)},
        edges=np.c_[np.zeros(len(nodes) - 1, int), np.arange(1, len(nodes))],
        ntype=None,
        meta={"unit": "nm"},
    )


def test_seeds_reach_from_thick_nodes_and_ties_go_low():
    # B: a thin line along x; A: a thick node just past B's box, which the
    # box prefilter once dropped (it left out the node's own radius), and a
    # node as near two B nodes, which goes to the lower
    b = _skeleton(np.c_[np.arange(11.0), np.zeros((11, 2))], np.full(11, 0.1))
    a = _skeleton([[5.0, 5.0, 0.0], [5.0, 1.4, 0.0], [2.5, 0.15, 0.0]], [1.0, 1.3, 0.5])
    seeds = pair.find_contact_seeds(a, b, delta=0.1, radius_key="median")
    assert seeds.idx_a.tolist() == [1, 2]
    assert seeds.idx_b.tolist() == [5, 2]
