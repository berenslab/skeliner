"""Contact sites between two meshes."""

import numpy as np
import pytest
import trimesh

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
