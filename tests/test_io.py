"""
IO round-trip smoke tests.

* load `.ctm`
* run skeletonize
* save to SWC & NPZ
* reload and compare a few coarse features
"""
from pathlib import Path

import numpy as np
import pytest

from skeliner import skeletonize
from skeliner.io import load_mesh, load_npz, load_swc


@pytest.fixture(scope="session")
def reference_mesh():
    mesh_path = Path(__file__).parent / "data" / "60427.ctm"
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
