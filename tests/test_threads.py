"""``sk.threads``: how many threads the Rust core's parallel loops run on."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import trimesh

import skeliner as sk
from skeliner import _rs

# one parallel loop in Rust, for the subprocess checks
_LOOP = """
import numpy as np
import trimesh
from skeliner import _rs

_m = trimesh.creation.icosphere(subdivisions=2)
_n = np.asarray(_m.face_normals, np.float64)
_bvh = _rs.pair.TriangleBvh(np.asarray(_m.triangles, np.float64).reshape(-1, 9), _n)


def loop():
    faces = np.arange(len(_n), dtype=np.int64)
    _bvh.first_hits(np.asarray(_m.triangles_center), -_n, faces, 1e-6)
"""


def _python(code: str, **env) -> list[str]:
    """Run *code* after ``_LOOP`` in a fresh interpreter, with
    ``RAYON_NUM_THREADS`` only as given; its printed words."""
    full = {k: v for k, v in os.environ.items() if k != "RAYON_NUM_THREADS"}
    full.update(env)
    out = subprocess.run(
        [sys.executable, "-c", _LOOP + textwrap.dedent(code)],
        env=full,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


@pytest.fixture(autouse=True)
def _default_pool():
    sk.threads(None)
    yield
    sk.threads(None)


def test_set_report_and_reset():
    default = sk.threads.get()
    assert default >= 1
    sk.threads(3)
    assert sk.threads.get() == 3 and _rs.threads.thread_override() == 3
    sk.threads(None)
    assert sk.threads.get() == default and _rs.threads.thread_override() is None


def test_a_block_puts_back_what_was_set_before():
    default = sk.threads.get()
    with sk.threads(3):
        assert sk.threads.get() == 3
        with sk.threads(2):
            assert sk.threads.get() == 2
        assert sk.threads.get() == 3
    assert sk.threads.get() == default
    sk.threads(4)
    with sk.threads(1):
        assert sk.threads.get() == 1
    assert sk.threads.get() == 4


def test_a_block_puts_it_back_after_an_error():
    with pytest.raises(KeyError), sk.threads(2):
        raise KeyError
    assert _rs.threads.thread_override() is None


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "4"])
def test_refuses_what_is_not_a_positive_integer(bad):
    with pytest.raises(ValueError):
        sk.threads(bad)
    assert _rs.threads.thread_override() is None


def test_results_do_not_depend_on_the_thread_count():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=100.0)
    tri = np.asarray(mesh.triangles, np.float64).reshape(-1, 9)
    nrm = np.asarray(mesh.face_normals, np.float64)
    faces = np.arange(len(nrm), dtype=np.int64)

    def run():
        hits = _rs.pair.TriangleBvh(tri, nrm).first_hits(
            np.asarray(mesh.triangles_center), -nrm, faces, 1e-6
        )
        return hits, sk.pre.compute_mesh_stats(mesh).outward_dots

    with sk.threads(1):
        one = run()
    with sk.threads(3):
        three = run()
    assert np.isfinite(one[0]).all() and np.isfinite(one[1]).any()
    for a, b in zip(one, three):
        assert np.array_equal(a, b, equal_nan=True)


def test_the_variable_sets_the_default_and_threads_overrides_it():
    out = _python(
        """
        import skeliner as sk
        print(sk.threads.get())
        with sk.threads(3):
            loop()
            print(sk.threads.get())
        loop()
        print(sk.threads.get())
        """,
        RAYON_NUM_THREADS="2",
    )
    assert out == ["2", "3", "2"]


def test_the_variable_counts_until_the_first_parallel_loop():
    out = _python(
        """
        import os
        import skeliner as sk
        os.environ["RAYON_NUM_THREADS"] = "2"
        print(sk.threads.get())  # asking does not start the default pool
        loop()
        os.environ["RAYON_NUM_THREADS"] = "5"
        print(sk.threads.get())
        """
    )
    assert out == ["2", "2"]


def test_the_loops_run_on_that_many_threads():
    pytest.importorskip("psutil")
    out = _python(
        """
        import psutil
        import skeliner as sk
        proc = psutil.Process()
        before = proc.num_threads()
        with sk.threads(3):
            loop()
        print(proc.num_threads() - before)
        """
    )
    assert out == ["3"]
