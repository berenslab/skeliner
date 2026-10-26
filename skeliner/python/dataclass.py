"""``skeliner.dataclass`` on the Python path: the arrays a skeleton keeps in
step (``skeliner._state``)."""

from __future__ import annotations

import numpy as np


def vertex_map(
    ptr, idx, n_vertices: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``vert2node`` from the bins (a CSR *ptr*, *idx*, nodes in order):
    ``(keys, node)`` — each vertex the first time a bin lists it, and by
    vertex the node of the last bin listing it (-1 for none), *n_vertices*
    long (one past the highest listed when not given). The dict
    ``zip(vertices, nodes)`` makes, as arrays."""
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if not (
        ptr.size and ptr[0] == 0 and (np.diff(ptr) >= 0).all() and ptr[-1] == idx.size
    ):
        raise ValueError("the bins must be a CSR (ptr, idx)")
    if (idx < 0).any():
        raise ValueError("vertex ids must not be negative")
    if n_vertices is None:
        n_vertices = int(idx.max()) + 1 if idx.size else 0
    elif idx.size and idx.max() >= n_vertices:
        v = int(idx[np.argmax(idx >= n_vertices)])
        raise ValueError(
            f"the bins name vertex {v}, but there are {n_vertices} vertices"
        )
    nodes = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    mapping = dict(zip(idx.tolist(), nodes.tolist()))
    keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    node = np.full(n_vertices, -1, dtype=np.int64)
    node[keys] = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    return keys, node
