"""The ``mesh`` crate on the Python path: a mesh's edge → face incidence and
the loops along the border of a face set, and a KD-tree queried as the
Rust one is."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree as _KDTree

# The three vertex-column pairs that make a face's edges.
EDGE_COLS = ((0, 1), (1, 2), (0, 2))


def rows3(name: str, a) -> np.ndarray:
    """*a* as an ``(name, 3)`` float64 array."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected ({name}, 3) array, got shape {list(a.shape)}")
    return a


def node_pairs(edges, n: int) -> np.ndarray:
    """Skeleton *edges* as an ``(E, 2)`` int64 array checked against the node
    count *n*."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"expected (E, 2) edges, got shape {list(edges.shape)}")
    bad = edges[(edges < 0) | (edges >= n)]
    if bad.size:
        raise ValueError(f"edge names node {bad[0]}, not one of {n}")
    return edges


def checked_faces(faces, n_vertices: int | None = None) -> tuple[np.ndarray, int]:
    """*faces* as an ``(F, 3)`` int64 array checked against the vertex count,
    which defaults to one past the highest index; ``(faces, count)``."""
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"expected (F, 3) array, got shape {faces.shape}")
    faces = faces.astype(np.int64, copy=False)
    if n_vertices is None:
        n_vertices = int(faces.max()) + 1 if faces.size and faces.max() >= 0 else 0
    bad = np.argwhere((faces < 0) | (faces >= n_vertices))
    if len(bad):
        fi, col = bad[0]
        raise ValueError(
            f"face {fi} names vertex {faces[fi, col]}, but the mesh has"
            f" {n_vertices} vertices"
        )
    return faces, int(n_vertices)


def mesh_roundtrip(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """*vertices* and *faces*, checked and returned unchanged."""
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"expected (V, 3) array, got shape {vertices.shape}")
    faces, _ = checked_faces(faces, len(vertices))
    return (
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )


def unique_edges(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Every distinct edge of *faces* as ``[low, high]``, ordered by ``high``
    and then ``low``: trimesh's ``edges_unique``, which keeps a degenerate
    face's edges too — ``[v, v]`` for its repeated vertex."""
    ends = np.asarray(faces, dtype=np.int64)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    width = max(int(n_vertices), 1)
    key = np.unique(ends.max(axis=1) * width + ends.min(axis=1))
    return np.stack([key % width, key // width], axis=1)


def edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Length of each edge, ``sqrt((dx² + dy²) + dz²)``: trimesh's
    ``edges_unique_length``, whose BLAS dot of the squares with ones adds
    them in that order."""
    d = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    return np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])


def row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product of each row of two ``(n, 3)`` arrays, summed left to right.

    ``np.einsum("ij,ij->i", ...)`` follows no single per-row order, so its
    last bit — and the sign of a dot near zero — is not reproducible; this
    order is the Rust core's.
    """
    return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) + a[:, 2] * b[:, 2]


def non_degenerate(faces: np.ndarray) -> np.ndarray:
    """Boolean mask of faces with 3 distinct vertices."""
    return (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )


def edge_to_faces(faces) -> dict[tuple[int, int], list[int]]:
    """Edge → faces sharing it, skipping faces with a repeated vertex.

    Keys are ``(low, high)`` vertex pairs, in the order a pass over the
    column pairs, each over all faces, first meets them.
    """
    faces = np.asarray(faces)
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    fi_idx = np.where(non_degenerate(faces))[0]
    gf = faces[fi_idx]
    for col_a, col_b in EDGE_COLS:
        va = gf[:, col_a].astype(np.intp)
        vb = gf[:, col_b].astype(np.intp)
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        for k in range(len(fi_idx)):
            result[(int(lo[k]), int(hi[k]))].append(int(fi_idx[k]))
    return result


def face_adjacency(
    edge_to_faces: dict[tuple[int, int], list[int]],
) -> dict[int, set[int]]:
    """Face → the faces sharing an edge with it."""
    adj: dict[int, set[int]] = defaultdict(set)
    for fis in edge_to_faces.values():
        for i in range(len(fis)):
            for j in range(i + 1, len(fis)):
                adj[fis[i]].add(fis[j])
                adj[fis[j]].add(fis[i])
    return adj


def border_edges(
    sel: set[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    faces: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Edges shared by a face in *sel* and a face outside it, sorted.

    Without *faces* the whole map is scanned.  With *faces* — the array
    *edge_to_faces* was built from — only the selection's own edges are
    examined, so the cost follows the selection rather than the mesh.
    Both return the same list.
    """
    if faces is None:
        return sorted(
            e
            for e, fis in edge_to_faces.items()
            if any(fi in sel for fi in fis) and any(fi not in sel for fi in fis)
        )
    found: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    for fi in sel:
        f = faces[fi]
        for a, b in EDGE_COLS:
            u, v = int(f[a]), int(f[b])
            e = (u, v) if u < v else (v, u)
            if e in seen:
                continue
            seen.add(e)
            fis = edge_to_faces.get(e)
            if fis and any(x in sel for x in fis) and any(x not in sel for x in fis):
                found.add(e)
    return sorted(found)


def border_loops(
    sel: set[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    faces: np.ndarray | None = None,
) -> list[list[int]]:
    """Closed loops along the border between *sel* and the other faces.

    Each loop starts at its smallest vertex and, wherever the border offers
    a choice, steps to the smallest unvisited neighbour, so the loops depend
    only on which edges are on the border.  *faces* as in
    :func:`border_edges`.
    """
    edges = border_edges(sel, edge_to_faces, faces)
    if len(edges) < 3:
        return []

    adj: dict[int, set[int]] = defaultdict(set)
    for v1, v2 in edges:
        adj[v1].add(v2)
        adj[v2].add(v1)

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in sorted(adj):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current, prev = start, -1
        while True:
            nbs = [n for n in adj[current] if n != prev]
            if not nbs:
                break
            closed = False
            for n in nbs:
                if n == start and len(loop) > 2:
                    loops.append(loop)
                    closed = True
                    break
            if closed:
                break
            nxt = min((n for n in nbs if n not in visited), default=None)
            if nxt is None:
                break
            visited.add(nxt)
            loop.append(nxt)
            prev, current = current, nxt

    return loops


class Topology:
    """Edge → face incidence of a mesh, read like the dict
    :func:`edge_to_faces` builds — which it is.

    Keys are ``(low, high)`` vertex pairs, values the faces sharing that
    edge, and iteration follows the dict's insertion order.  Faces with a
    repeated vertex have no edges.
    """

    def __init__(self, faces, n_vertices: int | None = None):
        self._faces, self._n_vertices = checked_faces(faces, n_vertices)
        self._map = dict(edge_to_faces(self._faces))
        self._adj: dict[int, set[int]] | None = None

    @property
    def n_faces(self) -> int:
        return len(self._faces)

    @property
    def n_vertices(self) -> int:
        return self._n_vertices

    @property
    def n_edges(self) -> int:
        return len(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key) -> bool:
        return tuple(key) in self._map

    def __getitem__(self, key) -> list[int]:
        return list(self._map[tuple(key)])

    def get(self, key, default=None):
        """Faces of edge *key*, or *default* if no face has that edge."""
        found = self._map.get(tuple(key))
        return default if found is None else list(found)

    def keys(self) -> list[tuple[int, int]]:
        """Edge keys, in the dict's insertion order."""
        return list(self._map)

    def values(self) -> list[list[int]]:
        """Face lists, in the dict's insertion order."""
        return [list(v) for v in self._map.values()]

    def items(self) -> list[tuple[tuple[int, int], list[int]]]:
        """``(key, faces)`` pairs, in the dict's insertion order."""
        return [(k, list(v)) for k, v in self._map.items()]

    def __iter__(self):
        return iter(self.keys())

    def as_dict(self) -> dict[tuple[int, int], list[int]]:
        """The same mapping as a plain dict."""
        return {k: list(v) for k, v in self._map.items()}

    def edges(self) -> np.ndarray:
        """Distinct edges as an ``(E, 2)`` array, in ascending order."""
        return np.array(sorted(self._map), dtype=np.int64).reshape(-1, 2)

    def edge_faces_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Edge → faces as CSR arrays ``(offsets, faces)``, edges in
        ascending order."""
        lists = [self._map[e] for e in sorted(self._map)]
        offsets = np.zeros(len(lists) + 1, dtype=np.int64)
        np.cumsum([len(fis) for fis in lists], out=offsets[1:])
        flat = [f for fis in lists for f in fis]
        return offsets, np.array(flat, dtype=np.int64)

    def face_adjacency(self) -> dict[int, set[int]]:
        """Face → the faces sharing an edge with it (built once)."""
        if self._adj is None:
            self._adj = face_adjacency(self._map)
        return self._adj

    def border_edges(self, sel) -> list[tuple[int, int]]:
        """Edges between a face of *sel* and a face outside it, ascending."""
        return border_edges(set(int(f) for f in sel), self._map, self._faces)

    def border_loops(self, sel) -> list[list[int]]:
        """Closed loops along the border of the faces in *sel*."""
        return border_loops(set(int(f) for f in sel), self._map, self._faces)

    def expand_tip(self, tip, target_verts=6, max_iters=20, max_eat_factor=30):
        """``skeliner.pre._expand_tip_to_good_rim`` on this mesh:
        ``(selection, longest loop or None)``."""
        from . import pre

        return pre.expand_tip_to_good_rim(
            list(tip),
            self._map,
            self.face_adjacency(),
            target_verts=target_verts,
            max_iters=max_iters,
            max_eat_factor=max_eat_factor,
            faces=self._faces,
        )

    def expand_selection_per_group(
        self, sel, target_verts=6, max_iters=20, max_eat_factor=30
    ):
        """``skeliner.pre._expand_selection_per_group`` on this mesh:
        ``(selection, loops)``."""
        from . import pre

        return pre.expand_selection_per_group(
            sel,
            self._map,
            self.face_adjacency(),
            target_verts=target_verts,
            max_iters=max_iters,
            max_eat_factor=max_eat_factor,
            faces=self._faces,
        )

    def sever_cost(self, sel, budget: int) -> int:
        """``skeliner.pre._sever_cost`` on this mesh."""
        from . import pre

        return pre.sever_cost(set(int(f) for f in sel), self.face_adjacency(), budget)


def nearest(tree: _KDTree, points) -> tuple[np.ndarray, np.ndarray]:
    """Nearest point of *tree* to each of *points*: ``(distances, indices)``.

    Exact ties go to the lowest index.  scipy's own pick among equally near
    points depends on how its tree was built, and on these grid-snapped
    meshes such ties are common — 2% of face-centroid queries on one cell.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if tree.n < 2 or len(points) == 0:
        return tree.query(points)
    d2, i2 = tree.query(points, k=2)
    d, i = d2[:, 0].copy(), i2[:, 0].copy()
    tied = np.flatnonzero(d2[:, 1] <= d2[:, 0])
    if len(tied):
        reach = d[tied] * (1 + 1e-9) + 1e-12
        data = tree.data
        for row, cand in zip(tied, tree.query_ball_point(points[tied], reach)):
            cand = np.asarray(cand)
            sq = ((data[cand] - points[row]) ** 2).sum(axis=1)
            i[row] = cand[sq == sq.min()].min()
    return d, i


class KdTree:
    """A KD-tree over ``(N, 3)`` points: scipy's, with exact ties in
    :meth:`query` going to the lowest index.

    *fused* is the Rust tree's choice of how squared distances are summed;
    scipy sums them its own one way, so it is not read here.
    """

    def __init__(self, points, fused: bool = False):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected (N, 3) array, got shape {points.shape}")
        self._tree = _KDTree(points)
        del fused

    def __len__(self) -> int:
        return self._tree.n

    def query_ball_point(self, x, r: float) -> list[np.ndarray]:
        """Indices within *r* of each query point, ascending, one array per
        point."""
        found = self._tree.query_ball_point(
            np.asarray(x, dtype=np.float64), r, return_sorted=True
        )
        return [np.asarray(h, dtype=np.int64) for h in found]

    def query(self, x) -> tuple[np.ndarray, np.ndarray]:
        """``(distances, indices)`` of each query point's nearest point."""
        if self._tree.n == 0:
            raise ValueError("the tree is empty")
        d, i = nearest(self._tree, x)
        return np.asarray(d, dtype=np.float64), np.asarray(i, dtype=np.int64)
