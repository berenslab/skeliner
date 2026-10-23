"""Which path skeliner runs: the Rust core or the Python path.

Every function that computes exists twice, with the same name, arguments and
result: in Rust (``skeliner/rust/<module>/``, loaded as
``skeliner._rs.<module>``) and in Python (``skeliner/python/<module>.py``),
which reads like the algorithm and runs on its own. The public modules
(``sk.pre``, ``sk.post``, …) turn meshes and skeletons into arrays, call the
active path's function through :data:`kernels`, and turn the result back.
"""

from __future__ import annotations

import importlib
import os

PATHS = ("rust", "python")


def _from_env() -> str:
    name = os.environ.get("SKELINER_BACKEND", "rust")
    if name not in PATHS:
        raise ValueError(f"SKELINER_BACKEND must be one of {PATHS}, not {name!r}")
    return name


_active = _from_env()


class backend:
    """
    Choose the path skeliner runs: ``"rust"`` (the default) or ``"python"``.

    ``sk.backend("python")`` switches from then on; ``with
    sk.backend("python"): ...`` only for the block, putting back what was
    set before. ``sk.backend.get()`` is the active path, and
    ``SKELINER_BACKEND`` the one a process starts with. The choice is
    process-wide.

    The Python path computes what the Rust core does — exactly on the
    platform the two were checked on, within each function's stated
    tolerance elsewhere, where BLAS and libm round differently — and is
    there to read, step through and learn from; the Rust core is the fast
    one.
    """

    def __init__(self, name: str):
        global _active
        if name not in PATHS:
            raise ValueError(f"backend takes one of {PATHS}, not {name!r}")
        self._previous = _active
        _active = name
        self.name = name

    def __enter__(self) -> backend:
        return self

    def __exit__(self, *_exc) -> None:
        global _active
        _active = self._previous

    def __repr__(self) -> str:
        return f"skeliner.backend({self.name!r})"

    @staticmethod
    def get() -> str:
        """The active path: ``"rust"`` or ``"python"``."""
        return _active


class _Module:
    """One module's functions on the active path. On the Rust path a
    function the Rust core does not have yet is the Python path's."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, function: str):
        if _active == "rust":
            from skeliner import _rs

            found = getattr(getattr(_rs, self._name, None), function, None)
            if found is not None:
                return found
        python = importlib.import_module(f"skeliner.python.{self._name}")
        return getattr(python, function)


class _Kernels:
    """The active path's functions, by module: ``kernels.pre.find_gaps`` is
    ``skeliner._rs.pre.find_gaps`` or ``skeliner.python.pre.find_gaps``."""

    def __getattr__(self, module: str) -> _Module:
        return _Module(module)


kernels = _Kernels()
