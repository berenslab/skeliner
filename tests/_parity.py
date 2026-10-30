"""Compare the Rust path's results with the Python path's.

Exact on the platform the two paths were checked against each other on
(macOS on arm64, numpy on Accelerate). Elsewhere BLAS and libm round
differently, so floats there need only agree to rounding; integers,
booleans, shapes and dtypes always exactly.
"""

import os
import platform

import numpy as np

import skeliner as sk

EXACT = platform.system() == "Darwin" and platform.machine() == "arm64"
#: Off the reference platform finite floats agree to within ``RTOL`` of the
#: largest magnitude among the values compared: a rounding difference, also
#: for values near zero, which carry the rounding of the large ones they
#: were computed from (a difference of two coordinates in nanometres).
RTOL = 1e-12

#: Off the reference platform, the largest float difference each test saw
#: between the two paths, ``{test: [max |a - b|, max |a - b| / scale]}``
#: (*scale* as in :func:`_close`), which ``conftest.py`` prints at the end
#: of the session.
DIFFERENCES: dict[str, list[float]] = {}


def _close(a, b, rtol: float, scale: float | None = None) -> bool:
    """NaN where NaN is and the same infinities; the finite values within
    *rtol* of *scale* — the magnitude their rounding comes from, by default
    the largest among them — which is recorded."""
    a, b = np.asarray(a), np.asarray(b)
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    if not np.array_equal(a[~fa], b[~fb], equal_nan=True):
        return False
    if not fa.any():
        return True
    x, y = a[fa], b[fb]
    d = float(np.abs(x - y).max())
    if scale is None:
        scale = float(max(np.abs(x).max(), np.abs(y).max()))
    test = os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]
    worst = DIFFERENCES.setdefault(test, [0.0, 0.0])
    worst[0] = max(worst[0], d)
    worst[1] = max(worst[1], d / scale if scale else 0.0)
    return d <= rtol * scale


def both(fn, *args, **kwargs):
    """``(rust, python)``: one call to *fn* on each path."""
    with sk.backend("rust"):
        rust = fn(*args, **kwargs)
    with sk.backend("python"):
        python = fn(*args, **kwargs)
    return rust, python


def assert_parity(a, b, path="result", *, rtol=RTOL, scale=None):
    """*a* and *b* equal — exactly on the reference platform, floats to
    rounding elsewhere (:data:`RTOL` of *scale*, the magnitude of what they
    were computed from when the values alone can be all rounding, by
    default the largest among them) — recursing into tuples, lists, dicts
    and arrays; NaN equals NaN."""
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return
        assert a == b or (not EXACT and _close(a, b, rtol, scale)), (
            f"{path}: {a!r} != {b!r}"
        )
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, f"{path}: shape {a.shape} != {b.shape}"
        assert a.dtype.kind == b.dtype.kind, f"{path}: dtype {a.dtype} != {b.dtype}"
        if a.dtype.kind in "fc" and not EXACT:
            assert _close(a, b, rtol, scale), f"{path}: values"
        else:
            equal_nan = a.dtype.kind in "fc"
            assert np.array_equal(a, b, equal_nan=equal_nan), f"{path}: values"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys differ"
        for k in a:
            assert_parity(a[k], b[k], f"{path}[{k!r}]", rtol=rtol, scale=scale)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_parity(x, y, f"{path}[{i}]", rtol=rtol, scale=scale)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"
