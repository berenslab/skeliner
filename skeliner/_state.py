"""Utility helpers that keep skeleton arrays in sync."""

from __future__ import annotations

import operator
from collections.abc import ItemsView, Mapping, MutableMapping, ValuesView
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

__all__ = [
    "SkeletonState",
    "VertexMap",
    "as_vertex_map",
    "compact_state",
    "remap_edges",
    "rebuild_vert2node",
    "swap_nodes",
    "vertex_arrays",
    "vertex_lut",
]


class VertexMap(MutableMapping):
    """``vert2node``: mesh vertex → skeleton node, with a dict's behaviour —
    keys in the order they arrived, ``get``, ``items``, assignment — kept as
    two arrays: the keys in order, and each vertex's node by vertex (-1 for
    none). The arrays come out without building a dict: :meth:`lut`,
    :meth:`keys_array`, :meth:`values_array`.

    Keys are vertex ids and values node ids, both non-negative ints.
    """

    __slots__ = ("_keys", "_node", "_added")

    def __init__(self, mapping: Mapping[int, int] | None = None):
        self._keys = np.empty(0, dtype=np.int64)
        self._node = np.empty(0, dtype=np.int64)
        self._added: list[int] = []
        if mapping:
            keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
            vals = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
            if (keys < 0).any() or (vals < 0).any():
                raise ValueError("vertex and node ids must not be negative")
            self._keys = keys
            self._node = np.full(int(keys.max()) + 1, -1, dtype=np.int64)
            self._node[keys] = vals

    @classmethod
    def _of_arrays(cls, keys: np.ndarray, node: np.ndarray) -> "VertexMap":
        """A map of *keys* in order and *node* by vertex, taken as they are."""
        out = cls.__new__(cls)
        out._keys = keys
        out._node = node
        out._added = []
        return out

    @classmethod
    def from_bins(cls, node2verts, n_vertices: int | None = None) -> "VertexMap":
        """The map *node2verts* implies: each vertex keyed the first time a
        bin lists it, mapped to the last bin listing it."""
        from ._backend import kernels

        bins = [np.asarray(v, dtype=np.int64).reshape(-1) for v in node2verts]
        ptr = np.zeros(len(bins) + 1, dtype=np.int64)
        np.cumsum([b.size for b in bins], out=ptr[1:])
        idx = np.concatenate(bins) if bins else np.empty(0, dtype=np.int64)
        return cls._of_arrays(*kernels.dataclass.vertex_map(ptr, idx, n_vertices))

    def _settle(self) -> None:
        if self._added:
            self._keys = np.concatenate(
                [self._keys, np.asarray(self._added, dtype=np.int64)]
            )
            self._added = []

    # -- the mapping ------------------------------------------------------
    def __getitem__(self, vertex) -> int:
        try:
            v = operator.index(vertex)
        except TypeError:
            raise KeyError(vertex) from None
        if 0 <= v < len(self._node) and self._node[v] >= 0:
            return int(self._node[v])
        raise KeyError(vertex)

    def __contains__(self, vertex) -> bool:
        try:
            v = operator.index(vertex)
        except TypeError:
            return False
        return 0 <= v < len(self._node) and bool(self._node[v] >= 0)

    def get(self, vertex, default=None):
        return self[vertex] if vertex in self else default

    def __setitem__(self, vertex, node) -> None:
        v, n = operator.index(vertex), operator.index(node)
        if v < 0 or n < 0:
            raise ValueError("vertex and node ids must not be negative")
        if v >= len(self._node):
            grown = np.full(max(v + 1, 2 * len(self._node)), -1, dtype=np.int64)
            grown[: len(self._node)] = self._node
            self._node = grown
        if self._node[v] < 0:
            self._added.append(v)
        self._node[v] = n

    def __delitem__(self, vertex) -> None:
        if vertex not in self:
            raise KeyError(vertex)
        v = operator.index(vertex)
        self._settle()
        self._node[v] = -1
        self._keys = self._keys[self._keys != v]

    def __iter__(self) -> Iterator[int]:
        self._settle()
        return iter(self._keys.tolist())

    def __len__(self) -> int:
        return len(self._keys) + len(self._added)

    def values(self) -> ValuesView:
        return _Values(self)

    def items(self) -> ItemsView:
        return _Items(self)

    def __eq__(self, other) -> bool:
        if isinstance(other, VertexMap):
            return np.array_equal(self.keys_array(), other.keys_array()) and (
                np.array_equal(self.values_array(), other.values_array())
            )
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    __hash__ = None  # mutable, as a dict is

    def __repr__(self) -> str:
        return f"VertexMap({len(self):,} vertices)"

    def __reduce__(self):
        self._settle()
        return (VertexMap._of_arrays, (self._keys, self._node))

    def copy(self) -> "VertexMap":
        self._settle()
        return VertexMap._of_arrays(self._keys.copy(), self._node.copy())

    # -- as arrays ----------------------------------------------------------
    def keys_array(self) -> np.ndarray:
        """``(K,)`` int64 the vertices, in key order."""
        self._settle()
        return self._keys.copy()

    def values_array(self) -> np.ndarray:
        """``(K,)`` int64 each key's node, in key order."""
        self._settle()
        return self._node[self._keys]

    def lut(self, n_vertices: int | None = None) -> np.ndarray:
        """``(n_vertices,)`` int64 each vertex's node, -1 for none
        (*n_vertices* at least one past the highest vertex keyed; one past it
        by default)."""
        self._settle()
        top = int(self._keys.max()) + 1 if len(self._keys) else 0
        n = top if n_vertices is None else int(n_vertices)
        if n < top:
            raise ValueError(
                f"vert2node names vertex {top - 1:,}, but there are {n:,} vertices"
            )
        out = np.full(n, -1, dtype=np.int64)
        m = min(n, len(self._node))
        out[:m] = self._node[:m]
        return out

    # -- whole-map edits ------------------------------------------------------
    def swap_nodes(self, a: int, b: int) -> None:
        """Nodes *a* and *b* trade every vertex, keys in place."""
        node = self._node
        self._node = np.where(node == a, b, np.where(node == b, a, node))

    def remapped(self, old2new: np.ndarray) -> "VertexMap":
        """Each vertex to ``old2new[node]``; a vertex whose node has none
        there (-1, or past its end) dropped; key order kept."""
        self._settle()
        old2new = np.asarray(old2new, dtype=np.int64)
        vals = self._node[self._keys]
        mapped = np.full(len(vals), -1, dtype=np.int64)
        known = vals < len(old2new)
        mapped[known] = old2new[vals[known]]
        keep = mapped >= 0
        keys = self._keys[keep]
        node = np.full(len(self._node), -1, dtype=np.int64)
        node[keys] = mapped[keep]
        return VertexMap._of_arrays(keys, node)


class _Values(ValuesView):
    def __iter__(self):
        return iter(self._mapping.values_array().tolist())


class _Items(ItemsView):
    def __iter__(self):
        m = self._mapping
        return zip(m.keys_array().tolist(), m.values_array().tolist())


def as_vertex_map(v2n) -> VertexMap:
    """A ``vert2node`` as a :class:`VertexMap` (itself when it is one)."""
    return v2n if isinstance(v2n, VertexMap) else VertexMap(v2n)


def vertex_arrays(v2n) -> tuple[np.ndarray, np.ndarray]:
    """A ``vert2node`` (a :class:`VertexMap` or a dict) as its keys and
    values, int64 arrays in key order."""
    if isinstance(v2n, VertexMap):
        return v2n.keys_array(), v2n.values_array()
    keys = np.fromiter(v2n.keys(), dtype=np.int64, count=len(v2n))
    vals = np.fromiter(v2n.values(), dtype=np.int64, count=len(v2n))
    return keys, vals


def vertex_lut(v2n, n_vertices: int) -> np.ndarray:
    """A ``vert2node`` as each vertex's node, -1 for none, over
    *n_vertices*."""
    if isinstance(v2n, VertexMap):
        return v2n.lut(n_vertices)
    keys, vals = vertex_arrays(v2n)
    lut = np.full(n_vertices, -1, dtype=np.int64)
    lut[keys] = vals
    return lut


@dataclass
class SkeletonState:
    """Container bundling all skeleton arrays that must stay in lock-step."""

    nodes: np.ndarray
    radii: dict[str, np.ndarray]
    edges: np.ndarray
    node2verts: list[np.ndarray] | None = None
    vert2node: VertexMap | dict[int, int] | None = None

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
        vert2node = None if self.vert2node is None else self.vert2node.copy()
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


def rebuild_vert2node(node2verts: list[np.ndarray] | None) -> VertexMap | None:
    """Invert node2verts (if present) into a vertex→node map.

    Keys come in node order, each node's vertices in the order it lists
    them; a vertex listed by several nodes maps to the last of them.
    """
    if node2verts is None:
        return None
    return VertexMap.from_bins(node2verts)


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
        v2n = state.vert2node
        if touched_vertices is None and isinstance(v2n, VertexMap):
            v2n.swap_nodes(a, b)
            return
        if touched_vertices is None:
            # values only, so the keys keep their order
            for vid, nid in v2n.items():
                if nid == a:
                    v2n[vid] = b
                elif nid == b:
                    v2n[vid] = a
            return
        for vid in touched_vertices:
            nid = v2n.get(int(vid))
            if nid == a:
                v2n[int(vid)] = b
            elif nid == b:
                v2n[int(vid)] = a
