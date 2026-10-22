"""How many threads skeliner's Rust core runs its parallel loops on."""

from __future__ import annotations

from numbers import Integral

from . import _rs


class threads:
    """
    Set how many threads skeliner's Rust core runs its parallel loops on.

    ``sk.threads(4)`` sets it from then on; ``with sk.threads(4): ...`` only
    for the block, putting back what was set before; ``sk.threads(None)``
    goes back to the default — every core the process may use, or
    ``RAYON_NUM_THREADS`` when that is set. ``sk.threads.get()`` is the
    number the next parallel loop runs on.

    It works at any time, unlike ``RAYON_NUM_THREADS``, which is read once,
    when the first parallel loop starts the default pool. The two do not
    conflict: the variable sizes that default pool, and while
    ``sk.threads(n)`` is set skeliner runs its loops on a pool of *n* threads
    of its own instead. The setting is process-wide, for every Python thread
    at once, and leaves numpy's own threads alone.
    """

    def __init__(self, n: int | None):
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
                raise ValueError(f"threads takes a positive integer or None, not {n!r}")
            n = int(n)
        self._previous = _rs.thread_override()
        _rs.set_threads(n)
        self.n = n

    def __enter__(self) -> threads:
        return self

    def __exit__(self, *_exc) -> None:
        _rs.set_threads(self._previous)

    def __repr__(self) -> str:
        return f"skeliner.threads({self.n!r})"

    @staticmethod
    def get() -> int:
        """How many threads the next parallel loop runs on."""
        return _rs.current_threads()
