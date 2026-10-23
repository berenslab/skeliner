"""``sk.backend``: which path skeliner runs."""

import pytest

import skeliner as sk
from skeliner._backend import kernels


def test_rust_is_the_default_and_the_python_path_can_be_chosen():
    assert sk.backend.get() == "rust"
    with sk.backend("python"):
        assert sk.backend.get() == "python"
        assert kernels.dx.segment_distances.__module__ == "skeliner.python.dx"
    assert sk.backend.get() == "rust"
    assert kernels.dx.segment_distances is sk._rs.dx.segment_distances


def test_a_call_switches_until_switched_back():
    try:
        sk.backend("python")
        assert sk.backend.get() == "python"
    finally:
        sk.backend("rust")
    assert sk.backend.get() == "rust"


def test_blocks_nest_and_put_back_what_was_set():
    with sk.backend("python"):
        with sk.backend("rust"):
            assert sk.backend.get() == "rust"
        assert sk.backend.get() == "python"
    assert sk.backend.get() == "rust"


def test_only_the_two_paths():
    with pytest.raises(ValueError, match="python"):
        sk.backend("cython")
    assert sk.backend.get() == "rust"


def test_a_function_without_rust_twin_runs_on_the_python_path(monkeypatch):
    import skeliner.python.dx as python_dx

    monkeypatch.setattr(python_dx, "only_in_python", lambda: "python", raising=False)
    assert kernels.dx.only_in_python() == "python"
