"""
IO round-trip smoke tests.

* load `.obj`
* run skeletonize
* save to SWC & NPZ
* reload and compare a few coarse features
"""

import json
from pathlib import Path

import numpy as np
import pytest

import skeliner as sk
from skeliner import Skeleton, Soma, dx, skeletonize
from skeliner._backend import PATHS
from skeliner.dataclass import MeshStats, Organelles
from skeliner.io import (
    load_mesh,
    load_skeleton_npz,
    load_soma_npz,
    load_skeleton_swc,
    save_soma_npz,
    save_organelles_npz,
    load_organelles_npz,
    save_mesh_stats_npz,
    load_mesh_stats_npz,
    save_mesh,
)

SAMPLES_DIR = Path(__file__).parent / "data"

SAMPLE_SWCS = [
    "60427.swc",
]


@pytest.fixture(scope="session")
def reference_mesh():
    mesh_path = SAMPLES_DIR / "60427.obj"
    return load_mesh(mesh_path)


def test_io_roundtrip(reference_mesh, tmp_path):
    skel = skeletonize(reference_mesh, verbose=False)

    # warm up KD-tree cache (and verify it exists)
    dx.distance(skel, skel.nodes[0], point_unit=skel.meta.get("unit", "nm"))
    assert skel._nodes_kdtree is not None

    # --- write ---------------------------------------------------------
    swc_path = tmp_path / "60427_test.swc"
    npz_path = tmp_path / "60427_test.npz"
    skel.to_swc(swc_path)
    skel.to_npz(npz_path)

    assert swc_path.exists()
    assert npz_path.exists()

    # --- read back -----------------------------------------------------
    skel_from_swc = load_skeleton_swc(swc_path)
    skel_from_npz = load_skeleton_npz(npz_path)
    assert skel_from_npz._nodes_kdtree is not None
    assert skel_from_npz._node_neighbors is not None

    # --- very coarse equivalence checks --------------------------------
    # (Exact float equality is not expected; topology & sizes should match.)
    assert skel_from_swc.nodes.shape[0] == skel.nodes.shape[0]
    assert skel_from_npz.edges.shape == skel.edges.shape
    assert np.isclose(
        skel_from_npz.soma.equiv_radius,
        skel.soma.equiv_radius,
        rtol=1e-4,
    )


# ------------------------------------------------------------------------
#  Helper
# ------------------------------------------------------------------------
def _edges_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """
    True iff two undirected edge lists connect the same pairs of
    vertices (row order may differ).
    """
    a = np.sort(a, axis=1)
    b = np.sort(b, axis=1)
    if a.shape != b.shape:
        return False
    # sort rows to make order irrelevant
    a = a[np.lexsort(a.T[::-1])]
    b = b[np.lexsort(b.T[::-1])]
    return np.array_equal(a, b)  # exact – they are integers


# ------------------------------------------------------------------------
#  Parametrised smoke-test
# ------------------------------------------------------------------------
@pytest.mark.parametrize("fname", SAMPLE_SWCS)
def test_swc_roundtrip_exact(fname: str, tmp_path: Path):
    src_path = SAMPLES_DIR / fname
    assert src_path.exists(), f"missing sample file {src_path}"

    # 1 · load the reference skeleton
    skel_ref = load_skeleton_swc(src_path)

    # 2 · write it back
    out_path = tmp_path / fname
    skel_ref.to_swc(out_path)
    assert out_path.exists()

    # 3 · read what we just wrote
    skel_rt = load_skeleton_swc(out_path)

    # 4 · compare ­­­—­­ geometry ------------------------------------------------
    assert np.allclose(skel_rt.nodes, skel_ref.nodes, rtol=1e-6, atol=0.0), (
        "XYZ coordinates changed"
    )

    for k in skel_ref.radii:
        assert np.allclose(skel_rt.radii[k], skel_ref.radii[k], rtol=1e-6, atol=0.0), (
            f"radius column '{k}' changed"
        )

    # topology ----------------------------------------------------------
    assert _edges_equal(skel_rt.edges, skel_ref.edges), "edge list changed"

    # node-type labels --------------------------------------------------
    assert np.array_equal(skel_rt.ntype, skel_ref.ntype), "ntype vector changed"

    # soma geometry (allow rounding) ------------------------------------
    assert np.isclose(
        skel_rt.soma.equiv_radius, skel_ref.soma.equiv_radius, rtol=1e-6
    ), "soma radius changed"

    assert np.allclose(skel_rt.soma.center, skel_ref.soma.center, rtol=1e-6), (
        "soma center changed"
    )


def test_skeleton_roundtrip(tmp_path, reference_mesh):
    skel0 = skeletonize(reference_mesh, verbose=False)
    out = tmp_path / "rt.swc"
    skel0.to_swc(out)
    skel1 = load_skeleton_swc(out)

    assert np.allclose(skel0.nodes, skel1.nodes, rtol=1e-6)
    assert _edges_equal(skel0.edges, skel1.edges)
    assert np.allclose(skel0.r, skel1.r, rtol=1e-6)


# ------------------------------------------------------------------------
#  Soma NPZ round-trips
# ------------------------------------------------------------------------
def test_soma_npz_roundtrip_with_verts(tmp_path):
    """Full ellipsoid soma with verts survives save/load."""
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    soma = Soma(
        center=[10, 20, 30],
        axes=[5, 4, 3],
        R=R,
        verts=np.array([0, 7, 42, 999], dtype=np.int64),
    )

    path = tmp_path / "soma_verts"
    soma.to_npz(path)

    loaded = Soma.from_npz(path.with_suffix(".npz"))
    assert np.allclose(loaded.center, soma.center)
    assert np.allclose(loaded.axes, soma.axes)
    assert np.allclose(loaded.R, soma.R)
    assert np.array_equal(loaded.verts, soma.verts)


def test_soma_npz_roundtrip_no_verts(tmp_path):
    """Soma without verts round-trips cleanly (verts stays None)."""
    soma = Soma(center=[0, 0, 0], axes=[1, 1, 1], R=np.eye(3))

    path = tmp_path / "soma_noverts.npz"
    soma.to_npz(path)

    loaded = Soma.from_npz(path)
    assert loaded.verts is None
    assert np.allclose(loaded.center, soma.center)
    assert np.allclose(loaded.axes, soma.axes)


def test_soma_npz_functional_api(tmp_path):
    """save_soma_npz / load_soma_npz work directly."""
    soma = Soma.from_sphere([1, 2, 3], radius=5.0, verts=np.arange(10))

    path = tmp_path / "soma_func"
    save_soma_npz(soma, path)

    loaded = load_soma_npz(path.with_suffix(".npz"))
    assert np.isclose(loaded.equiv_radius, soma.equiv_radius)
    assert np.array_equal(loaded.verts, soma.verts)


def test_soma_npz_derived_fields(tmp_path):
    """Derived field _W is recomputed on load (not stored)."""
    soma = Soma(center=[1, 2, 3], axes=[6, 4, 2], R=np.eye(3))
    path = tmp_path / "soma_derived.npz"
    soma.to_npz(path)

    loaded = Soma.from_npz(path)
    # _W should be recomputed via __post_init__
    pt = np.array([[2, 3, 4]], dtype=np.float64)
    assert np.allclose(soma.contains(pt), loaded.contains(pt))


def test_soma_npz_nucleus_roundtrip(tmp_path):
    """Full nucleus dict survives round-trip."""
    nucleus = {
        "center": np.array([100.0, 200.0, 300.0]),
        "peak_r": 1500.0,
        "z_range": (280.0, 340.0),
        "slices": np.array(
            [
                [280, 101, 201, 1200],
                [290, 102, 202, 1400],
                [300, 100, 200, 1500],
                [310, 99, 199, 1300],
                [340, 98, 198, 1000],
            ],
            dtype=np.float64,
        ),
    }
    soma = Soma(center=[10, 20, 30], axes=[5, 4, 3], R=np.eye(3), nucleus=nucleus)
    path = tmp_path / "soma_nuc.npz"
    save_soma_npz(soma, path)

    loaded = load_soma_npz(path)
    assert np.allclose(loaded.nucleus["center"], nucleus["center"])
    assert loaded.nucleus["peak_r"] == nucleus["peak_r"]
    assert loaded.nucleus["z_range"] == nucleus["z_range"]
    assert np.array_equal(loaded.nucleus["slices"], nucleus["slices"])


def test_soma_npz_nucleus_none(tmp_path):
    """Soma without nucleus loads as None (backward compat)."""
    soma = Soma(center=[0, 0, 0], axes=[1, 1, 1], R=np.eye(3))
    path = tmp_path / "soma_no_nuc.npz"
    save_soma_npz(soma, path)

    loaded = load_soma_npz(path)
    assert loaded.nucleus is None


# ------------------------------------------------------------------------
#  Skeleton classmethod round-trips
# ------------------------------------------------------------------------
def test_skeleton_from_swc():
    """Skeleton.from_swc matches load_skeleton_swc."""
    src = SAMPLES_DIR / "60427.swc"
    skel_func = load_skeleton_swc(src)
    skel_cls = Skeleton.from_swc(src)

    assert np.allclose(skel_cls.nodes, skel_func.nodes)
    assert _edges_equal(skel_cls.edges, skel_func.edges)
    assert np.array_equal(skel_cls.ntype, skel_func.ntype)


def test_skeleton_from_npz(tmp_path, reference_mesh):
    """Skeleton.from_npz matches load_skeleton_npz."""
    skel = skeletonize(reference_mesh, verbose=False)
    path = tmp_path / "skel_cls.npz"
    skel.to_npz(path)

    loaded = Skeleton.from_npz(path)
    assert loaded.nodes.shape == skel.nodes.shape
    assert loaded.edges.shape == skel.edges.shape
    assert np.allclose(loaded.soma.center, skel.soma.center)
    assert np.allclose(loaded.soma.axes, skel.soma.axes)


# ------------------------------------------------------------------------
#  Organelles NPZ round-trips
# ------------------------------------------------------------------------
def test_organelles_npz_full_roundtrip(tmp_path):
    """All masks survive round-trip."""
    nF = 1000
    pocket = np.zeros(nF, dtype=bool)
    pocket[:100] = True
    isolated = np.zeros(nF, dtype=bool)
    isolated[200:250] = True
    expanded = np.zeros(nF, dtype=bool)
    expanded[500:520] = True

    org = Organelles(pocket=pocket, isolated=isolated, expanded=expanded)

    path = tmp_path / "org_full.npz"
    save_organelles_npz(org, path)

    loaded = load_organelles_npz(path)
    assert np.array_equal(loaded.pocket, org.pocket)
    assert np.array_equal(loaded.isolated, org.isolated)
    assert np.array_equal(loaded.expanded, org.expanded)
    # .mask property
    assert np.array_equal(loaded.mask, pocket | isolated | expanded)


def test_organelles_npz_backward_compat(tmp_path):
    """Old files with 'expanded' missing or with embedded mesh_stats still load."""
    nF = 100
    pocket = np.ones(nF, dtype=bool)
    isolated = np.zeros(nF, dtype=bool)
    # Simulate old format: no expanded key, with leftover mesh_stats fields
    path = tmp_path / "org_old.npz"
    face_comp = np.zeros(nF, dtype=np.int64)
    np.savez_compressed(
        path,
        pocket=pocket,
        isolated=isolated,
        outward_dots=np.zeros(nF),
        face_comp=face_comp,
        main_ci=np.array(0),
    )

    loaded = load_organelles_npz(path)
    assert np.array_equal(loaded.pocket, pocket)
    assert np.array_equal(loaded.expanded, np.zeros(nF, dtype=bool))


# ------------------------------------------------------------------------
#  MeshStats NPZ round-trip
# ------------------------------------------------------------------------
def test_mesh_stats_npz_roundtrip(tmp_path):
    """outward_dots / face_comp / main_ci survive a round-trip."""
    nF = 500
    rng = np.random.default_rng(42)
    stats = MeshStats(
        outward_dots=rng.uniform(-1, 1, nF).astype(np.float32),
        face_comp=rng.integers(0, 5, nF).astype(np.int64),
        main_ci=2,
    )

    path = tmp_path / "mesh_stats.npz"
    save_mesh_stats_npz(stats, path)

    loaded = load_mesh_stats_npz(path)
    assert np.allclose(loaded.outward_dots, stats.outward_dots)
    assert np.array_equal(loaded.face_comp, stats.face_comp)
    assert loaded.main_ci == stats.main_ci


def test_mesh_stats_npz_face_count_validation(tmp_path):
    """load_mesh_stats_npz raises when face count disagrees with mesh."""
    import trimesh

    nF = 50
    stats = MeshStats(
        outward_dots=np.zeros(nF, dtype=np.float32),
        face_comp=np.zeros(nF, dtype=np.int64),
        main_ci=0,
    )
    path = tmp_path / "mesh_stats.npz"
    save_mesh_stats_npz(stats, path)

    # Build a tetrahedron mesh (4 faces) — does not match nF=50
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    with pytest.raises(ValueError, match="stale"):
        load_mesh_stats_npz(path, mesh=mesh)

    # Without the mesh argument it should still load
    loaded = load_mesh_stats_npz(path)
    assert loaded.main_ci == 0


# ── provenance: which track and which parameters produced a skeleton ──
#
# Written into every export, so it cannot be added retroactively — a
# skeleton saved without it can never say what made it.  The two tracks
# disagree about the same cell by construction, which is exactly why the
# file has to name the one that ran.


class TestSkeletonProvenance:
    def test_each_track_names_itself(self, reference_mesh):
        from skeliner import pre

        direct = skeletonize(reference_mesh, verbose=False)
        assert direct.meta["track"] == "direct"

        empty = np.zeros(len(reference_mesh.faces), dtype=bool)
        comps = pre.break_up_mesh(
            reference_mesh,
            direct.soma,
            Organelles(
                pocket=empty.copy(), isolated=empty.copy(), expanded=empty.copy()
            ),
            verbose=False,
        )
        preproc = skeletonize(reference_mesh, components=comps, verbose=False)
        assert preproc.meta["track"] == "preproc"

    def test_a_track_records_only_the_parameters_it_used(self, reference_mesh):
        """The preprocessing track ignores soma detection, bridging and
        pruning — they happen upstream — so claiming them would describe a
        run that did not happen."""
        from skeliner import pre

        direct = skeletonize(reference_mesh, verbose=False)
        empty = np.zeros(len(reference_mesh.faces), dtype=bool)
        comps = pre.break_up_mesh(
            reference_mesh,
            direct.soma,
            Organelles(
                pocket=empty.copy(), isolated=empty.copy(), expanded=empty.copy()
            ),
            verbose=False,
        )
        preproc = skeletonize(reference_mesh, components=comps, verbose=False)

        for ignored in ("bridge_gaps", "detect_soma", "prune_tiny_neurites"):
            assert ignored in direct.meta["params"]
            assert ignored not in preproc.meta["params"]
        # and the one they share is recorded by both
        assert "geodesic_shell_count" in preproc.meta["params"]
        assert "geodesic_shell_count" in direct.meta["params"]

    def test_a_changed_parameter_shows_up(self, reference_mesh):
        a = skeletonize(reference_mesh, geodesic_shell_count=1000, verbose=False)
        b = skeletonize(reference_mesh, geodesic_shell_count=400, verbose=False)
        assert a.meta["params"]["geodesic_shell_count"] == 1000
        assert b.meta["params"]["geodesic_shell_count"] == 400

    def test_numpy_arguments_do_not_break_swc_export(self, reference_mesh, tmp_path):
        """``to_swc`` writes meta as one JSON line with a plain ``json.dumps``
        and no fallback encoder, so an ndarray left in the record would raise
        on export rather than where it was recorded."""
        skel = skeletonize(
            reference_mesh,
            soma_seed_point=np.array([1.0, 2.0, 3.0]),
            verbose=False,
        )
        assert skel.meta["params"]["soma_seed_point"] == [1.0, 2.0, 3.0]
        json.dumps(skel.meta)  # the export path, directly

        path = tmp_path / "s.swc"
        skel.to_swc(path)
        assert load_skeleton_swc(path).meta["params"]["soma_seed_point"] == [
            1.0,
            2.0,
            3.0,
        ]

    @pytest.mark.parametrize("fmt", ["npz", "swc"])
    def test_provenance_survives_the_round_trip(self, reference_mesh, tmp_path, fmt):
        skel = skeletonize(reference_mesh, verbose=False)
        path = tmp_path / f"s.{fmt}"
        (skel.to_npz if fmt == "npz" else skel.to_swc)(path)
        back = (load_skeleton_npz if fmt == "npz" else load_skeleton_swc)(path)
        assert back.meta["track"] == skel.meta["track"]
        assert back.meta["params"] == skel.meta["params"]
        assert back.meta["skeliner_version"] == skel.meta["skeliner_version"]


# ---------------------------------------------------------------------
# mesh readers: the Rust and the Python path read OBJ as trimesh does, and
# leave anything else to it
# ---------------------------------------------------------------------
def _trimesh_arrays(path):
    import trimesh

    m = trimesh.load_mesh(path, process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def _obj_files(reference_mesh, tmp_path):
    """OBJ files both readers read."""
    files = [Path(__file__).parent / "data" / "60427.obj"]
    written = tmp_path / "written.obj"
    save_mesh(reference_mesh, written)  # trimesh's export, with its header
    files.append(written)
    loose = tmp_path / "loose.obj"  # a vertex no face names, and CRLF
    loose.write_bytes(b"v 0 0 0\r\nv 9 9 9\r\nv 1 0 0\r\nv 0 1 0\r\n\r\nf 1 3 4\r\n")
    files.append(loose)
    numbers = tmp_path / "numbers.obj"  # decimals as writers spell them
    numbers.write_bytes(
        b"v 1e3 -2.5E-2 +3.\nv .5 -0 0.1000000000000000055511151231257827\n"
        b"v 515088.00000 510594.00000 46431.00000\n"
        b"v 1.7976931348623157e308 4.9e-324 -123456789.123456789\n"
        b"f 1 2 3\nf 2 3 4\n"
    )
    files.append(numbers)
    return files


def test_mesh_readers_read_as_trimesh_on_both_paths(reference_mesh, tmp_path):
    from skeliner import _rs
    from skeliner.python import io as python_io

    from ._parity import assert_parity, both

    for path in _obj_files(reference_mesh, tmp_path):
        want = _trimesh_arrays(path)
        rust = _rs.io.read_obj(str(path))
        python = python_io.read_obj(str(path))
        assert rust is not None, path.name
        assert_parity(rust, python, path.name)
        for got in (rust[0], rust[1], python[0], python[1]):
            assert got.flags.c_contiguous
        # bit for bit, signed zeros included
        assert rust[0].tobytes() == want[0].tobytes(), path.name
        assert np.array_equal(rust[1], want[1]), path.name
        for mesh in both(load_mesh, path):
            assert mesh.vertices.tobytes() == want[0].tobytes(), path.name
            assert np.array_equal(mesh.faces, want[1]), path.name
    # the vertex no face names is dropped, as trimesh drops it
    assert len(load_mesh(tmp_path / "loose.obj").vertices) == 3


def test_what_the_readers_do_not_read_goes_to_trimesh(reference_mesh, tmp_path):
    import trimesh

    from skeliner import _rs
    from skeliner.python import io as python_io

    from ._parity import both

    tri = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
    declined = {
        "normals.obj": tri + b"vn 0 0 1\nf 1//1 2//1 3//1\n",
        "quad.obj": tri + b"f 1 2 4 3\n",
        "objects.obj": b"o a\n" + tri + b"f 1 2 3\no b\nf 1 3 2\n",
        "relative.obj": tri + b"f -4 -3 -2\n",
        "colours.obj": b"v 0 0 0 1 0 0\nv 1 0 0 1 0 0\nv 0 1 0 1 0 0\nf 1 2 3\n",
    }
    for name, data in declined.items():
        (tmp_path / name).write_bytes(data)
        assert _rs.io.read_obj(str(tmp_path / name)) is None, name
        assert python_io.read_obj(str(tmp_path / name)) is None, name
    # formats skeliner has no reader for go to trimesh directly
    others = ["mesh.ply", "mesh.stl", "mesh.off"]
    for name in others:
        reference_mesh.export(tmp_path / name)
    for name in [*declined, *others]:
        path = tmp_path / name
        want = trimesh.load_mesh(path, process=False)
        for mesh in both(load_mesh, path):
            assert type(mesh) is type(want), name
            assert np.array_equal(mesh.vertices, want.vertices), name
            assert np.array_equal(mesh.faces, want.faces), name


def test_a_ctm_the_reader_does_not_read_goes_to_trimesh(tmp_path):
    import trimesh

    from skeliner import _rs
    from skeliner.python import io as python_io

    path = tmp_path / "other.ctm"
    path.write_bytes(b"not a CTM file")
    assert _rs.io.read_ctm(str(path)) is None
    assert python_io.read_ctm(str(path)) is None
    # whatever trimesh makes of it (here: no python-openctm), load_mesh does
    with pytest.raises(Exception) as trimesh_error:
        trimesh.load_mesh(path, process=False)
    for name in PATHS:
        with sk.backend(name), pytest.raises(trimesh_error.type):
            load_mesh(path)


def test_a_missing_mesh_file_raises_as_open_does_on_both_paths(tmp_path):
    from skeliner import _rs
    from skeliner.python import io as python_io

    missing = str(tmp_path / "missing.obj")
    for read in (_rs.io.read_obj, _rs.io.read_ctm):
        with pytest.raises(FileNotFoundError) as rust:
            read(missing)
        with pytest.raises(FileNotFoundError) as python:
            getattr(python_io, read.__name__)(missing)
        assert str(rust.value) == str(python.value)
        assert rust.value.filename == missing
    with pytest.raises(IsADirectoryError):
        _rs.io.read_obj(str(tmp_path))


# ---------------------------------------------------------------------
# OpenCTM, for EyeWire 1's meshes: files built here, as the format lays
# them out, read on both paths
# ---------------------------------------------------------------------
def _ctm_block(words) -> bytes:
    """A packed CTM block of 32-bit *words* ``(count, size)``: four byte
    planes, most significant first, each component by component, LZMA1
    packed after its five property bytes."""
    import lzma
    import struct

    w = np.asarray(words, dtype=np.uint32).reshape(len(words), -1)
    planes = np.stack([(w >> s).astype(np.uint8) for s in (24, 16, 8, 0)])
    alone = lzma.compress(planes.transpose(0, 2, 1).tobytes(), format=lzma.FORMAT_ALONE)
    # an "alone" stream: properties, unpacked size, data; CTM keeps no size
    return struct.pack("<I", len(alone) - 13) + alone[:5] + alone[13:]


def _ctm_index_deltas(faces) -> np.ndarray:
    """MG1 / MG2 triangle deltas, as the reference encoder makes them."""
    w = np.asarray(faces, dtype=np.int64).copy()
    for i in range(len(w) - 1, -1, -1):
        if i >= 1 and w[i, 0] == w[i - 1, 0]:
            w[i, 1] -= w[i - 1, 1]
        else:
            w[i, 1] -= w[i, 0]
        w[i, 2] -= w[i, 0]
        if i >= 1:
            w[i, 0] -= w[i - 1, 0]
    return w & 0xFFFFFFFF


def _ctm_file(method: str, vertices, faces) -> tuple[bytes, np.ndarray, np.ndarray]:
    """An OpenCTM v5 file of *vertices* and *faces*, and what it decodes to:
    ``(data, vertices, faces)``."""
    import struct

    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    head = b"OCTM" + struct.pack("<I", 5) + method.encode().ljust(4, b"\0")
    head += struct.pack("<5I", len(v), len(f), 0, 0, 0) + struct.pack("<I", 4) + b"note"
    if method == "RAW":
        body = b"INDX" + f.astype("<u4").tobytes() + b"VERT" + v.astype("<f4").tobytes()
        return head + body, v.astype(np.float64), f
    if method == "MG1":
        body = b"INDX" + _ctm_block(_ctm_index_deltas(f))
        body += b"VERT" + _ctm_block(v.view(np.uint32))
        return head + body, v.astype(np.float64), f
    # MG2: each vertex on a 4 x 4 x 4 grid, in steps of the precision from
    # its cell's corner; vertices sorted by cell, x a delta within one
    precision, div = np.float32(0.5), np.array([4, 4, 4])
    lo, hi = v.min(axis=0) - 1, v.max(axis=0) + 1
    size = (hi - lo) / div.astype(np.float32)
    g = np.minimum(((v - lo) / size).astype(np.int64), div - 1)
    cell = g[:, 0] + div[0] * (g[:, 1] + div[1] * g[:, 2])
    origin = g.astype(np.float32) * size + lo
    ints = np.round((v - origin) / precision).astype(np.int64)
    order = np.lexsort((ints[:, 0], cell))
    cell, ints, origin = cell[order], ints[order], origin[order]
    new_id = np.empty(len(v), dtype=np.int64)
    new_id[order] = np.arange(len(v))
    f = new_id[f]
    x = ints[:, 0].copy()
    same = np.r_[False, cell[1:] == cell[:-1]]
    x[1:][same[1:]] -= ints[:-1, 0][same[1:]]
    stored = np.column_stack([x, ints[:, 1], ints[:, 2]])
    mg2h = b"MG2H" + struct.pack("<8f", precision, 0.0, *lo, *hi)
    mg2h += struct.pack("<3I", *div)
    body = mg2h + b"VERT" + _ctm_block(stored)
    body += b"GIDX" + _ctm_block(np.diff(cell, prepend=0)[:, None])
    body += b"INDX" + _ctm_block(_ctm_index_deltas(f))
    expected = precision * ints.astype(np.float32) + origin
    return head + body, expected.astype(np.float64), f


@pytest.mark.parametrize("method", ["RAW", "MG1", "MG2"])
def test_eyewire_ctm_reads_on_both_paths(tmp_path, method):
    import trimesh

    from skeliner import _rs
    from skeliner.python import io as python_io

    from ._parity import assert_parity, both

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=40.0)
    vertices = sphere.vertices + [100.0, 50.0, 7.0]
    data, want_v, want_f = _ctm_file(method, vertices, sphere.faces)
    path = tmp_path / f"{method}.ctm"
    path.write_bytes(data)
    rust = _rs.io.read_ctm(str(path))
    assert_parity(rust, python_io.read_ctm(str(path)))
    assert np.array_equal(rust[0], want_v) and np.array_equal(rust[1], want_f)
    # the same triangles; MG2's corners within half its precision (0.5)
    tol = 0.25 + 1e-4 if method == "MG2" else 1e-4
    assert np.abs(rust[0][rust[1]] - vertices[sphere.faces]).max() <= tol
    for mesh in both(load_mesh, path):
        assert np.array_equal(mesh.vertices, want_v)
        assert np.array_equal(mesh.faces, want_f)


def test_eyewire_ctm_refuses_the_same_broken_files_on_both_paths(tmp_path):
    import lzma

    import trimesh

    from skeliner import _rs
    from skeliner.python import io as python_io

    sphere = trimesh.creation.icosphere(subdivisions=1)
    data, _, _ = _ctm_file("MG2", sphere.vertices, sphere.faces)
    gidx = data.index(b"GIDX")
    vert = data.index(b"VERT")
    cases = {
        "short.ctm": (data[:-7], ValueError, "ends early"),
        "unordered.ctm": (
            data[:gidx] + b"XXXX" + data[gidx + 4 :],
            ValueError,
            "missing or out of place",
        ),
        # VERT's first LZMA property byte past the largest valid one
        "packed.ctm": (
            data[: vert + 8] + b"\xff" + data[vert + 9 :],
            lzma.LZMAError,
            None,
        ),
    }
    for name, (blob, error, match) in cases.items():
        path = tmp_path / name
        path.write_bytes(blob)
        with pytest.raises(error, match=match) as rust:
            _rs.io.read_ctm(str(path))
        with pytest.raises(error, match=match) as python:
            python_io.read_ctm(str(path))
        assert str(rust.value) == str(python.value), name
