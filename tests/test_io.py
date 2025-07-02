"""
IO round-trip smoke tests.

* load `.obj`
* run skeletonize
* save to SWC & NPZ
* reload and compare a few coarse features
"""
from pathlib import Path

import numpy as np
import pytest

from skeliner import skeletonize
from skeliner.io import load_mesh, load_npz, load_swc

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

    # --- write ---------------------------------------------------------
    swc_path = tmp_path / "60427_test.swc"
    npz_path = tmp_path / "60427_test.npz"
    skel.to_swc(swc_path)
    skel.to_npz(npz_path)

    assert swc_path.exists()
    assert npz_path.exists()

    # --- read back -----------------------------------------------------
    skel_from_swc = load_swc(swc_path)
    skel_from_npz = load_npz(npz_path)

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
    return np.array_equal(a, b)          # exact – they are integers


# ------------------------------------------------------------------------
#  Parametrised smoke-test
# ------------------------------------------------------------------------
@pytest.mark.parametrize("fname", SAMPLE_SWCS)
def test_swc_roundtrip_exact(fname: str, tmp_path: Path):
    src_path = SAMPLES_DIR / fname
    assert src_path.exists(), f"missing sample file {src_path}"

    # 1 · load the reference skeleton
    skel_ref = load_swc(src_path)

    # 2 · write it back
    out_path = tmp_path / fname
    skel_ref.to_swc(out_path)
    assert out_path.exists()

    # 3 · read what we just wrote
    skel_rt = load_swc(out_path)

    # 4 · compare ­­­—­­ geometry ------------------------------------------------
    assert np.allclose(
        skel_rt.nodes, skel_ref.nodes, rtol=1e-6, atol=0.0
    ), "XYZ coordinates changed"

    for k in skel_ref.radii:
        assert np.allclose(
            skel_rt.radii[k], skel_ref.radii[k], rtol=1e-6, atol=0.0
        ), f"radius column '{k}' changed"

    # topology ----------------------------------------------------------
    assert _edges_equal(skel_rt.edges, skel_ref.edges), "edge list changed"

    # node-type labels --------------------------------------------------
    assert np.array_equal(
        skel_rt.ntype, skel_ref.ntype
    ), "ntype vector changed"

    # soma geometry (allow rounding) ------------------------------------
    assert np.isclose(
        skel_rt.soma.equiv_radius, skel_ref.soma.equiv_radius, rtol=1e-6
    ), "soma radius changed"

    assert np.allclose(
        skel_rt.soma.center, skel_ref.soma.center, rtol=1e-6
    ), "soma center changed"

def test_skeleton_roundtrip(tmp_path, reference_mesh):
    skel0 = skeletonize(reference_mesh, verbose=False)
    out = tmp_path / "rt.swc"
    skel0.to_swc(out)
    skel1 = load_swc(out)

    assert np.allclose(skel0.nodes,  skel1.nodes,  rtol=1e-6)
    assert _edges_equal(skel0.edges, skel1.edges)
    assert np.allclose(skel0.r, skel1.r, rtol=1e-6)