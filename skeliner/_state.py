"""Utility helpers that keep skeleton arrays in sync."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "SkeletonState",
    "compact_state",
    "remap_edges",
    "rebuild_vert2node",
    "swap_nodes",
]


@dataclass
class SkeletonState:
    """Container bundling all skeleton arrays that must stay in lock-step."""

    nodes: np.ndarray
    radii: dict[str, np.ndarray]
    edges: np.ndarray
    node2verts: list[np.ndarray] | None = None
    vert2node: dict[int, int] | None = None

    def ensure_node2verts(self) -> list[np.ndarray]:
        """Create empty `node2verts` shells when the input mesh is unavailable."""
        if self.node2verts is None:
            self.node2verts = [
                np.empty(0, dtype=np.int64) for _ in range(len(self.nodes))
            ]
        return self.node2verts

    def clone(self) -> "SkeletonState":
        """Deep-copy all arrays so mutations on the clone stay isolated."""
        node2verts = (
            [np.asarray(v, dtype=np.int64).copy() for v in self.node2verts]
            if self.node2verts is not None
            else None
        )
        vert2node = dict(self.vert2node) if self.vert2node is not None else None
        return SkeletonState(
            nodes=self.nodes.copy(),
            radii={k: v.copy() for k, v in self.radii.items()},
            edges=self.edges.copy(),
            node2verts=node2verts,
            vert2node=vert2node,
        )


def _build_old2new(keep_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep_indices, old2new) for a boolean mask."""
    if keep_mask.dtype != bool:
        raise ValueError("keep_mask must be boolean.")
    keep_idx = np.flatnonzero(keep_mask)
    old2new = -np.ones(keep_mask.size, dtype=np.int64)
    old2new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    return keep_idx, old2new


def remap_edges(edges: np.ndarray, old2new: np.ndarray) -> np.ndarray:
    """
    Drop edges that touch removed nodes and return the deduplicated remap.
    """
    if edges.size == 0:
        return edges.copy()

    remapped = old2new[edges]
    keep = (remapped[:, 0] >= 0) & (remapped[:, 1] >= 0)
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.int64)

    remapped = remapped[keep]
    remapped = remapped[remapped[:, 0] != remapped[:, 1]]
    if remapped.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    remapped.sort(axis=1)
    return np.unique(remapped, axis=0)


def rebuild_vert2node(node2verts: list[np.ndarray] | None) -> dict[int, int] | None:
    """Invert node2verts (if present) into a vertex→node dictionary."""
    if node2verts is None:
        return None
    mapping: dict[int, int] = {}
    for nid, verts in enumerate(node2verts):
        if verts.size == 0:
            continue
        for vid in np.asarray(verts, dtype=np.int64):
            mapping[int(vid)] = nid
    return mapping


def compact_state(
    state: SkeletonState,
    keep_mask: np.ndarray,
    *,
    rebuild_map: bool = True,
    return_old2new: bool = False,
) -> tuple[SkeletonState, np.ndarray | None]:
    """
    Slice all node-aligned arrays down to keep_mask == True.
    """
    keep_idx, old2new = _build_old2new(keep_mask)

    nodes = state.nodes[keep_mask].copy()
    radii = {k: v[keep_mask].copy() for k, v in state.radii.items()}
    node2verts = (
        [state.node2verts[i] for i in keep_idx]
        if state.node2verts is not None
        else None
    )

    edges = remap_edges(state.edges, old2new)
    vert2node = rebuild_vert2node(node2verts) if rebuild_map else state.vert2node

    new_state = SkeletonState(
        nodes=nodes,
        radii=radii,
        edges=edges,
        node2verts=node2verts,
        vert2node=vert2node,
    )
    payload = old2new if return_old2new else None
    return new_state, payload


def swap_nodes(
    state: SkeletonState,
    a: int,
    b: int,
    *,
    update_vert2node: bool = True,
    touched_vertices: Iterable[int] | None = None,
) -> None:
    """
    Swap two node indices while keeping auxiliary data structures consistent.
    """
    if a == b:
        return
    state.nodes[[a, b]] = state.nodes[[b, a]]
    for arr in state.radii.values():
        arr[[a, b]] = arr[[b, a]]
    if state.node2verts is not None:
        state.node2verts[a], state.node2verts[b] = (
            state.node2verts[b],
            state.node2verts[a],
        )
    if update_vert2node and state.vert2node is not None:
        if touched_vertices is None:
            touched = list(state.vert2node.keys())
        else:
            touched = touched_vertices
        for vid in touched:
            nid = state.vert2node.get(int(vid))
            if nid == a:
                state.vert2node[int(vid)] = b
            elif nid == b:
                state.vert2node[int(vid)] = a
