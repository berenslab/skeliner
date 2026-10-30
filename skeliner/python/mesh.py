"""The ``mesh`` crate on the Python path: a mesh's edge → face incidence and
the loops along the border of a face set, and a KD-tree queried as the
Rust one is."""

from __future__ import annotations

import math
import struct
import sys
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.spatial import KDTree as _KDTree

# The three vertex-column pairs that make a face's edges.
EDGE_COLS = ((0, 1), (1, 2), (0, 2))


def rows3(name: str, a) -> np.ndarray:
    """*a* as an ``(name, 3)`` float64 array."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected ({name}, 3) array, got shape {list(a.shape)}")
    return a


def total_key(x: float) -> int:
    """A sort key that orders floats as Rust's ``f64::total_cmp`` does:
    -NaN, -inf, …, -0.0, +0.0, …, +inf, +NaN."""
    bits = struct.unpack("<q", struct.pack("<d", x))[0]
    return bits ^ ((bits >> 63) & 0x7FFF_FFFF_FFFF_FFFF)


def convex_hull(points) -> list[tuple[float, float]]:
    """The convex hull of 2D *points*, counter-clockwise, by the monotone
    chain: the points sorted by x then y (as ``f64::total_cmp`` orders
    them), repeats dropped, collinear points on the hull left out. Fewer
    than three distinct points come back as they are, sorted."""
    pts = sorted(
        ((float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)),
        key=lambda q: (total_key(q[0]), total_key(q[1])),
    )
    p = [q for i, q in enumerate(pts) if i == 0 or q != pts[i - 1]]
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull: list[tuple[float, float]] = []
    for chain in (p, p[::-1]):
        start = len(hull)
        for q in chain:
            while len(hull) >= start + 2 and cross(hull[-2], hull[-1], q) <= 0.0:
                hull.pop()
            hull.append(q)
        hull.pop()
    return hull


def qhull_flat(points, hull) -> bool:
    """Whether qhull, as scipy's ``ConvexHull`` runs it, refuses *points*
    (their *hull* from :func:`convex_hull`) as flat: fewer than three
    distinct points, or the triangle it starts from — the hull's points of
    least and greatest x and the one farthest from their line — with its
    centroid within qhull's roundoff of that line (``DISTround``, from the
    largest coordinate)."""
    if len(hull) < 3:
        return True
    maxabs = maxsum = 0.0
    for x, y in np.asarray(points, dtype=np.float64).tolist():
        maxabs = max(maxabs, abs(x), abs(y))
        maxsum = max(maxsum, abs(x) + abs(y))
    distsum = min(math.sqrt(2.0) * maxabs, maxsum)
    roundoff = sys.float_info.epsilon * (2.0 * distsum * 1.01 + maxabs)
    order = sorted(hull, key=lambda q: (total_key(q[0]), total_key(q[1])))
    a, b = order[0], order[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.sqrt(dx * dx + dy * dy)
    height = 0.0
    for q in hull:
        height = max(height, abs((q[0] - a[0]) * dy - (q[1] - a[1]) * dx) / length)
    return height / 3.0 <= roundoff


def hull_measures(points) -> tuple[float, float]:
    """``(area, perimeter)`` of the convex hull of 2D *points* — scipy's
    ``ConvexHull`` ``volume`` and ``area`` — by the shoelace formula about
    the hull's first point (far from the origin, the products otherwise
    cancel) and the edge lengths; both 0 where qhull refuses the points as
    flat."""
    hull = convex_hull(points)
    if qhull_flat(points, hull):
        return 0.0, 0.0
    o = hull[0]
    s = perimeter = 0.0
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        s += (a[0] - o[0]) * (b[1] - o[1]) - (b[0] - o[0]) * (a[1] - o[1])
        dx, dy = b[0] - a[0], b[1] - a[1]
        perimeter += math.sqrt(dx * dx + dy * dy)
    return 0.5 * abs(s), perimeter


def hull_equations(points) -> np.ndarray | None:
    """The edges of the convex hull of 2D *points* as scipy's
    ``ConvexHull.equations`` gives them: per edge ``(n0, n1, offset)``, the
    unit normal pointing out, ``n . p + offset <= 0`` inside, counter-
    clockwise from the lowest point; None where qhull refuses the points
    as flat."""
    hull = convex_hull(points)
    if qhull_flat(points, hull):
        return None
    rows = []
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(dx * dx + dy * dy)
        n0, n1 = dy / length, -dx / length
        rows.append((n0, n1, -(n0 * a[0] + n1 * a[1])))
    return np.array(rows, dtype=np.float64)


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


def triangles(vertices, faces) -> np.ndarray:
    """``sk.Mesh.triangles``: each face's three vertices, ``(F, 3, 3)``."""
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    return vertices[faces]


def submesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """trimesh's ``submesh(..., append=True)`` of *faces* ``(F, 3)``
    (``sk.Mesh.submesh``): the vertices they name, ascending, and the faces
    renumbered to them, in order."""
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    used = np.unique(faces)
    renumber = np.zeros(len(vertices), dtype=np.int64)
    renumber[used] = np.arange(len(used))
    return vertices[used], renumber[faces]


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


def seam_joined(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Each vertex's first vertex at its position that a seam joins it to,
    itself when none comes before it (positions equal as floats are: −0.0 is
    0.0, a NaN equals nothing). Two vertices at one position are joined when
    an edge of each ends at one position too, at one vertex or at two more
    joined so: a seam is a chain of edges both sides of it repeat. Vertices
    that only touch stay apart — ``remove_fusions`` cuts a vertex shared by
    branches that share no edge into such copies, and joining them would
    fuse the branches again. Edges are those of faces of three vertices; a
    face naming one vertex twice joins nothing."""
    from scipy.sparse import coo_array
    from scipy.sparse.csgraph import connected_components

    n = len(verts)
    first = np.arange(n, dtype=np.int64)
    number = np.flatnonzero(~np.isnan(verts).any(axis=1))
    if len(number):
        # adding 0.0 turns −0.0 into 0.0; np.unique's rows compare as bytes
        _, where, inverse = np.unique(
            verts[number] + 0.0, axis=0, return_index=True, return_inverse=True
        )
        first[number] = number[where[inverse.reshape(-1)]]
    shared = np.zeros(n, dtype=bool)
    copy = first != np.arange(n)
    shared[copy] = True
    shared[first[copy]] = True
    if not shared.any():
        return first
    whole = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    ends = faces[whole][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    at = first[ends]
    use = (at[:, 0] != at[:, 1]) & shared[ends].any(axis=1)
    ends, at = ends[use], at[use]
    # each edge from its lower position to its higher, keyed by the two
    flip = at[:, 0] > at[:, 1]
    ends[flip] = ends[flip, ::-1]
    at[flip] = at[flip, ::-1]
    _, edge_of, key = np.unique(at, axis=0, return_index=True, return_inverse=True)
    there = ends[edge_of[key.reshape(-1)]]
    graph = coo_array(
        (np.ones(2 * len(ends)), (ends.ravel(), there.ravel())), shape=(n, n)
    )
    _, label = connected_components(graph, directed=False)
    low = np.full(label.max() + 1, n, dtype=np.int64)
    np.minimum.at(low, label, np.arange(n))
    return low[label]


def merge_coincident_vertices(vertices, faces) -> tuple[np.ndarray, int, int]:
    """*faces* with every vertex replaced by the first vertex at its position
    that a seam joins it to (:func:`seam_joined`), and each face the merge
    made a copy of another collapsed to ``[0, 0, 0]`` (None when no vertex is
    merged): ``(faces, copies, collapsed)``, the number of vertices merged
    into an earlier one and of faces collapsed.

    Of the faces with the same three vertices, those the merge left alone
    all stay (a duplicate the mesh had already is the fusions' evidence), and
    when it changed every one of them the first stays.
    """
    verts = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(verts))
    joined = seam_joined(verts, faces)
    copies = int((joined != np.arange(len(verts))).sum())
    if copies == 0:
        return None, 0, 0
    merged = joined[faces]
    changed = (merged != faces).any(axis=1)
    _, group = np.unique(np.sort(merged, axis=1), axis=0, return_inverse=True)
    group = group.reshape(-1)
    face_ids = np.arange(len(faces))
    first_face = np.full(group.max() + 1, len(faces), dtype=np.int64)
    np.minimum.at(first_face, group, face_ids)
    any_unchanged = np.zeros(group.max() + 1, dtype=bool)
    any_unchanged[group[~changed]] = True
    collapse = changed & (any_unchanged[group] | (first_face[group] != face_ids))
    merged[collapse] = 0
    return merged, copies, int(collapse.sum())


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


#: trimesh's zero for a cross product's norm.
TOL_ZERO = np.finfo(np.float64).resolution * 100


def face_crosses(vertices, faces) -> np.ndarray:
    """trimesh's ``triangles_cross``: ``(t1 - t0) x (t2 - t1)`` per face,
    each component numpy's product less a product."""
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    t = vertices[faces]
    u = t[:, 1] - t[:, 0]
    w = t[:, 2] - t[:, 1]
    cross = np.empty((len(faces), 3), dtype=np.float64)
    cross[:, 0] = u[:, 1] * w[:, 2] - u[:, 2] * w[:, 1]
    cross[:, 1] = u[:, 2] * w[:, 0] - u[:, 0] * w[:, 2]
    cross[:, 2] = u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
    return cross


def _cross_norm(c: np.ndarray) -> np.ndarray:
    """Each cross product's norm, ``sqrt((x² + y²) + z²)``: trimesh's BLAS dot
    of the squares with ones."""
    return np.sqrt((c[:, 0] * c[:, 0] + c[:, 1] * c[:, 1]) + c[:, 2] * c[:, 2])


def face_normals(crosses) -> np.ndarray:
    """trimesh's ``face_normals`` from the *crosses*: each times the
    reciprocal of its norm, zero where the norm is not above ``TOL_ZERO``."""
    c = rows3("F", crosses)
    norm = _cross_norm(c)
    valid = norm > TOL_ZERO
    out = np.zeros_like(c)
    out[valid] = c[valid] * (1.0 / norm[valid])[:, None]
    return out


def face_areas(crosses) -> np.ndarray:
    """trimesh's ``area_faces`` from the *crosses*: half each one's norm."""
    return _cross_norm(rows3("F", crosses)) / 2.0


def face_centers(vertices, faces) -> np.ndarray:
    """trimesh's ``triangles_center``: each face's vertices summed in order
    from zero, as numpy's mean sums them, over three."""
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    t = vertices[faces]
    return (((0.0 + t[:, 0]) + t[:, 1]) + t[:, 2]) / 3.0


def adjacent_faces(faces, n_vertices: int) -> np.ndarray:
    """trimesh's ``face_adjacency``: for every edge two face edges share — by
    its higher end, then its lower — the two faces, lower first; an edge
    three or more face edges share has none, and one a face shares with
    itself (a repeated vertex) is left out."""
    faces, n = checked_faces(faces, n_vertices)
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    key = e[:, 1] * max(n, 1) + e[:, 0]
    order = np.lexsort((owner, key))
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sizes = np.diff(np.r_[starts, len(key)])
    two = starts[sizes == 2]
    pairs = np.stack([owner[order[two]], owner[order[two + 1]]], axis=1)
    return pairs[pairs[:, 0] != pairs[:, 1]]


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


def _finite(what: str, points) -> np.ndarray:
    """*points* as floats, refused when a coordinate is NaN or infinite."""
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError(f"{what} must be finite")
    return points


class KdTree:
    """A KD-tree over ``(N, 3)`` points: scipy's, with exact ties in
    :meth:`query` going to the lowest index.

    *fused* is the Rust tree's choice of how squared distances are summed;
    scipy sums them its own one way, so it is not read here.
    """

    def __init__(self, points, fused: bool | None = None):
        from scipy.spatial import KDTree as _KDTree

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected (N, 3) array, got shape {points.shape}")
        self._tree = _KDTree(points)
        del fused

    def __len__(self) -> int:
        return self._tree.n

    def query_ball_point(self, x, r) -> list[np.ndarray]:
        """Indices within *r* of each query point, ascending, one array per
        point; *r* one radius, or one per point."""
        x = _finite("query points", x)
        r = np.asarray(r, dtype=np.float64)
        if r.ndim and r.shape != x.shape[:1]:
            raise ValueError("r must be a radius or one per point")
        found = self._tree.query_ball_point(x, r, return_sorted=True)
        return [np.asarray(h, dtype=np.int64) for h in found]

    def query(self, x) -> tuple[np.ndarray, np.ndarray]:
        """``(distances, indices)`` of each query point's nearest point."""
        x = _finite("query points", x)
        if self._tree.n == 0:
            raise ValueError("the tree is empty")
        d, i = nearest(self._tree, x)
        return np.asarray(d, dtype=np.float64), np.asarray(i, dtype=np.int64)
