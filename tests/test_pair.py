"""Contact sites between two meshes."""

import numpy as np
import pytest
import trimesh

from skeliner import pair


def _tube(axis, center, radius=200.0, length=2000.0, step=25.0, around=64, cap=False):
    """A tube along unit vector *axis* through *center*, finely meshed, faces
    wound outward; open, or with *cap* a flat disc closing its +axis end."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ang = np.arange(around) * (2 * np.pi / around)
    ring = radius * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v))
    zs = np.arange(-length / 2, length / 2 + step / 2, step)
    verts = np.vstack([ring + z * axis for z in zs]) + center
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
            verts = np.vstack([verts, r * (ring / radius) + zs[-1] * axis + center])
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


def _geometry(a, b):
    return {k: float(v[0]) for k, v in _labelled(a, b).geometry.items()}


def test_a_crossing_reads_as_two_flanks_at_right_angles(tube_z):
    g = _geometry(tube_z, _tube([1, 0, 0], [0.0, 405.0, 0.0]))
    assert g["crossing_deg"] > 80
    assert g["axis_normal_A"] < 0.2 and g["axis_normal_B"] < 0.2
    assert g["elongation_A"] > 2 and g["elongation_B"] > 2
    assert abs(g["radius_A"] - 200) < 20 and abs(g["radius_B"] - 200) < 20
    assert 0.3 < g["conformity"] < 3


def test_tubes_side_by_side_read_as_parallel_and_conforming(tube_z):
    g = _geometry(tube_z, _tube([0, 0, 1], [405.0, 0.0, 0.0]))
    crossing = _geometry(tube_z, _tube([1, 0, 0], [0.0, 405.0, 0.0]))
    assert g["crossing_deg"] < 10
    assert g["conformity"] > 3 * crossing["conformity"]


def test_a_tube_ending_on_a_flank_reads_as_a_tip(tube_z):
    g = _geometry(tube_z, _tube([0, -1, 0], [0.0, 1205.0, 0.0], cap=True))
    assert g["axis_normal_B"] > 0.9 and g["axis_normal_A"] < 0.2


def test_every_site_is_kept_and_labelled(tube_z):
    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    beside = _tube([0, 0, 1], [405.0, 0.0, 0.0])
    tip = _tube([0, -1, 0], [0.0, 1205.0, 0.0], cap=True)
    for other, kind in ((crossing, "crossing"), (beside, "parallel"), (tip, "tip")):
        assert _labelled(tube_z, other).kind.tolist() == [kind]


def test_unlabelled_sites_have_no_kind(tube_z):
    sites = pair.scan_contacts(
        tube_z, _tube([1, 0, 0], [0.0, 405.0, 0.0]), tol=20.0, label=False
    )
    assert sites.kind is None and sites.geometry is None


def test_relabelling_reuses_the_geometry(tube_z):
    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    sites = pair.scan_contacts(tube_z, crossing, tol=20.0)
    geometry = sites.geometry
    pair.label_contacts(tube_z, crossing, sites, crossing_min_deg=95.0)
    assert sites.geometry is geometry
    assert sites.kind.tolist() == ["other"]
    assert sites.meta["label"]["crossing_min_deg"] == 95.0


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
    assert sorted(sites.kind.tolist()) == ["crossing", "parallel"]
    picked = pair.pick_contacts(sites, sites.kind == "parallel")
    i = int(np.flatnonzero(sites.kind == "parallel")[0])
    assert len(picked.faces_A) == 1 and picked.kind.tolist() == ["parallel"]
    assert np.array_equal(picked.faces_A[0], sites.faces_A[i])
    assert picked.stats_A["aspect"][0] == sites.stats_A["aspect"][i]
    assert picked.geometry["crossing_deg"][0] == sites.geometry["crossing_deg"][i]


def test_kinds_survive_a_round_trip_and_old_files_load(tube_z, tmp_path):
    from skeliner.io import load_contact_sites_npz

    crossing = _tube([1, 0, 0], [0.0, 405.0, 0.0])
    sites = pair.scan_contacts(tube_z, crossing, tol=20.0)
    sites.to_npz(tmp_path / "labelled.npz")
    back = load_contact_sites_npz(tmp_path / "labelled.npz")
    assert back.kind.tolist() == sites.kind.tolist()
    assert back.geometry.keys() == sites.geometry.keys()
    assert back.geometry["conformity"][0] == sites.geometry["conformity"][0]
    plain = pair.scan_contacts(tube_z, crossing, tol=20.0, label=False)
    plain.to_npz(tmp_path / "plain.npz")
    back = load_contact_sites_npz(tmp_path / "plain.npz")
    assert back.kind is None and back.geometry is None
