"""Reusable skeleton helpers shared across the package."""

from __future__ import annotations

from collections import deque
from typing import List

import igraph as ig
import numpy as np

from .dataclass import Skeleton, Soma

__all__ = [
    "_find_soma",
    "_bfs_parents",
    "_build_mst",
]


def _find_soma(
    nodes: np.ndarray,
    radii: np.ndarray,
    *,
    pct_large: float = 99.9,
    dist_factor: float = 3.0,
    min_keep: int = 2,
) -> tuple[Soma, np.ndarray, bool]:
    """
    Geometry-only soma heuristic shared by the core pipeline and post-processing.
    """
    if nodes.shape[0] == 0:
        raise ValueError("empty skeleton")

    large_thresh = np.percentile(radii, pct_large)
    cand_idx = np.where(radii >= large_thresh)[0]
    if cand_idx.size == 0:
        raise RuntimeError(
            f"no nodes above the {pct_large:g}-th percentile (try lowering pct_large)"
        )

    idx_max = int(np.argmax(radii))
    R_max = radii[idx_max]

    dists = np.linalg.norm(nodes[cand_idx] - nodes[idx_max], axis=1)
    keep = dists <= dist_factor * R_max
    soma_idx = cand_idx[keep]
    has_soma = soma_idx.size >= min_keep

    if not has_soma:
        return Soma.from_sphere(nodes[idx_max], R_max, None), soma_idx, has_soma

    soma = Soma.from_sphere(
        center=nodes[soma_idx].mean(0),
        radius=R_max,
        verts=None,
    )
    return soma, soma_idx, has_soma


def _bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> List[int]:
    """Return parent[] array of BFS tree from *root* given an undirected edge list."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    parent = [-1] * n_nodes
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v != root and parent[v] == -1:
                parent[v] = u
                q.append(v)
    return parent


def _build_mst(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the edge list of the global minimum-spanning tree."""
    g = ig.Graph(
        n=len(nodes), edges=[tuple(map(int, e)) for e in edges], directed=False
    )
    g.es["weight"] = [float(np.linalg.norm(nodes[a] - nodes[b])) for a, b in edges]
    mst = g.spanning_tree(weights="weight")
    return np.asarray(
        sorted(tuple(sorted(e)) for e in mst.get_edgelist()), dtype=np.int64
    )