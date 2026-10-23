"""Compare the Rust path's results with the Python path's.

Exact on the platform the two paths were checked against each other on
(macOS on arm64, numpy on Accelerate). Elsewhere BLAS and libm round
differently, so floats there are compared within a tolerance; integers,
booleans, shapes and dtypes always exactly.
"""

import os
import platform

import numpy as np

import skeliner as sk

EXACT = platform.system() == "Darwin" and platform.machine() == "arm64"
# floats off the reference platform; set from CI's first measured differences
RTOL, ATOL = 1e-9, 1e-12

#: Off the reference platform, the largest float difference each test saw
#: between the two paths, ``{test: [max |a - b|, max |a - b| / |b|]}``,
#: which ``conftest.py`` prints at the end of the session.
DIFFERENCES: dict[str, list[float]] = {}


def _record(a, b) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_finite = np.isfinite(a) & np.isfinite(b)
    if not both_finite.any():
        return
    d = np.abs(a - b)[both_finite]
    rel = d / np.maximum(np.abs(b[both_finite]), np.finfo(np.float64).tiny)
    test = os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]
    worst = DIFFERENCES.setdefault(test, [0.0, 0.0])
    worst[0] = max(worst[0], float(d.max()))
    worst[1] = max(worst[1], float(rel.max()))


def both(fn, *args, **kwargs):
    """``(rust, python)``: one call to *fn* on each path."""
    with sk.backend("rust"):
        rust = fn(*args, **kwargs)
    with sk.backend("python"):
        python = fn(*args, **kwargs)
    return rust, python


def assert_parity(a, b, path="result", *, rtol=RTOL, atol=ATOL):
    """*a* and *b* equal — exactly on the reference platform, floats within
    *rtol* / *atol* elsewhere — recursing into tuples, lists, dicts and
    arrays; NaN equals NaN."""
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return
        if not EXACT:
            _record(a, b)
        assert a == b or (not EXACT and np.isclose(a, b, rtol, atol)), (
            f"{path}: {a!r} != {b!r}"
        )
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, f"{path}: shape {a.shape} != {b.shape}"
        assert a.dtype.kind == b.dtype.kind, f"{path}: dtype {a.dtype} != {b.dtype}"
        if a.dtype.kind == "f" and not EXACT:
            _record(a, b)
        if a.dtype.kind in "fc" and not EXACT:
            assert np.allclose(a, b, rtol, atol, equal_nan=True), f"{path}: values"
        else:
            equal_nan = a.dtype.kind in "fc"
            assert np.array_equal(a, b, equal_nan=equal_nan), f"{path}: values"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys differ"
        for k in a:
            assert_parity(a[k], b[k], f"{path}[{k!r}]", rtol=rtol, atol=atol)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_parity(x, y, f"{path}[{i}]", rtol=rtol, atol=atol)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"
