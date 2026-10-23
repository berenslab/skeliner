"""``skeliner.post`` on the Python path: ``calibrate_radii``'s ray cast and
the faces of a vertex set, and node types through edits."""

from __future__ import annotations

import numpy as np
import trimesh

from .mesh import checked_faces, node_pairs, rows3

#: The order a tie between node types goes: the soma, the root's other mark,
#: the neurite types, unlabelled.
NTYPE_PRIORITY = (1, -1, 2, 3, 4, 5, 6, 0)


def choose_label(labels) -> int:
    """The type of a node made of nodes typed *labels*: their most common
    non-zero type (the most common of all when every one is zero), a tie
    going by ``NTYPE_PRIORITY``, then to the smallest; 0 for none."""
    labels = [int(label) for label in labels]
    if len(labels) == 1:
        return labels[0]
    vals = [label for label in labels if label != 0] or labels
    if not vals:
        return 0
    uniq, counts = np.unique(vals, return_counts=True)
    tied = uniq[counts == counts.max()].tolist()
    return next((p for p in NTYPE_PRIORITY if p in tied), tied[0])


def ensure_root_label(labels: np.ndarray, root_label: int) -> np.ndarray:
    """Node 0 marked 1 when *root_label* is 1 and -1 otherwise, and every
    other node marked 1 or -1 unlabelled; in place, *labels* returned."""
    if labels.size == 0:
        return labels
    labels[0] = 1 if root_label == 1 else -1
    marked = (labels == 1) | (labels == -1)
    marked[0] = False
    labels[marked] = 0
    return labels


def mode_int(vals) -> int:
    """The most common of *vals* as ``np.bincount`` counts them after
    clipping to ``0..max`` (``max`` floored at 0): negative values count as
    0, and a tie goes to the smallest value."""
    vals = np.asarray(vals, dtype=np.int64)
    top = int(vals.max(initial=0))
    return int(np.bincount(np.clip(vals, 0, top)).argmax())


def fill_ntype_gaps(ntype, edges) -> np.ndarray:
    """Zero node types filled from their neighbours over the undirected
    *edges* ``(E, 2)``: sweeps over the nodes typed 0 at the sweep's start,
    in order, each taking the most common neighbouring type other than 0, 1
    and -1 (``mode_int``; a neighbour counts once per edge, and a fill
    earlier in the sweep is seen), until a sweep fills nothing, at most one
    sweep per node."""
    filled = np.array(ntype, dtype=np.int64)
    edges = node_pairs(edges, len(filled))
    n = len(filled)
    if n == 0 or len(edges) == 0:
        return filled
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges.tolist():
        if a != b:
            adj[a].append(b)
            adj[b].append(a)
    for _ in range(n):
        zeros = np.flatnonzero(filled == 0)
        if not len(zeros):
            break
        changed = False
        for z in zeros:
            if not adj[z]:
                continue
            around = filled[adj[z]]
            around = around[(around != 0) & (around != 1) & (around != -1)]
            if around.size == 0:
                continue
            label = mode_int(around)
            if label != 0:
                filled[z] = label
                changed = True
        if not changed:
            break
    return filled


def remap_ntype(ntype, old2new, new_len: int, root_label: int, edges, fill_gaps):
    """Node types after the nodes are renumbered by *old2new* (-1 for a node
    dropped) into *new_len* nodes: each new node takes ``choose_label`` of
    the old nodes mapped to it, the root is marked by *root_label*
    (``ensure_root_label``), and with *fill_gaps* the zero types are filled
    over *edges* (between new nodes) and the root marked again."""
    ntype = np.asarray(ntype, dtype=np.int64)
    old2new = np.asarray(old2new, dtype=np.int64)
    edges = node_pairs(edges, new_len)
    if len(old2new) != len(ntype):
        raise ValueError("old2new must have one entry per node")
    buckets: list[list[int]] = [[] for _ in range(new_len)]
    for label, new in zip(ntype.tolist(), old2new.tolist()):
        if new < 0:
            continue
        if new >= new_len:
            raise ValueError(f"old2new names node {new}, past the new nodes")
        buckets[new].append(label)
    mapped = np.array([choose_label(b) for b in buckets], dtype=np.int64)
    ensure_root_label(mapped, root_label)
    if fill_gaps:
        mapped = fill_ntype_gaps(mapped, edges)
        ensure_root_label(mapped, root_label)
    return mapped


#: Rays per call of trimesh's ray test.  Its broad phase pairs each ray with
#: every triangle whose box the ray's box crosses, so on a soma-sized patch a
#: ray meets thousands of triangles, and one call over all the rays ran out of
#: memory; a ray's hit does not depend on the other rays.
_RAY_CHUNK = 2048


def ray_hits(triangles, normals, origins, directions) -> np.ndarray:
    """Which rays hit any of the triangles, with trimesh's ``ray_triangle``
    test: per ray from ``origins[k]`` along ``directions[k]``. *triangles*
    are ``(F, 3, 3)`` flattened to ``(F, 9)``, *normals* their face normals.
    """
    tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    normals = np.asarray(normals, dtype=np.float64)
    if len(normals) != len(tris):
        raise ValueError("one normal per triangle")
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if len(origins) != len(directions):
        raise ValueError("one direction per origin")
    hit = np.zeros(len(origins), dtype=bool)
    if len(origins) and len(tris):
        tree = trimesh.triangles.bounds_tree(tris)
        for lo in range(0, len(origins), _RAY_CHUNK):
            _, rays, _ = trimesh.ray.ray_triangle.ray_triangle_id(
                tris,
                origins[lo : lo + _RAY_CHUNK],
                directions[lo : lo + _RAY_CHUNK],
                triangles_normal=normals,
                tree=tree,
                multiple_hits=False,
            )
            hit[lo + rays] = True
    return hit


class VertexFaces:
    """Faces by vertex, for cutting a mesh down to the faces of a vertex set."""

    def __init__(self, faces, n_vertices: int):
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must be (F, 3)")
        if faces.size and (faces.min() < 0 or faces.max() >= n_vertices):
            raise ValueError("face vertex index out of range")
        self._faces = faces
        self._n_vertices = int(n_vertices)

    def faces_within(self, vids) -> np.ndarray:
        """Faces with all three vertices among *vids*, ascending."""
        vids = np.asarray(vids, dtype=np.int64)
        if vids.size and (vids.min() < 0 or vids.max() >= self._n_vertices):
            raise ValueError("vertex index out of range")
        return np.flatnonzero(np.isin(self._faces, vids).all(axis=1))


#: What :func:`calibrate_radii` made of a node's radius.
FALLBACK, FULL_CENTERLINE, OUTER_CENTERLINE = 0, 1, 2


def outer_vertices(
    triangles, normals, centers, local_faces, directions, threshold: int
) -> np.ndarray:
    """The vertices (local ids, ascending) of the faces fewer than
    *threshold* rays hit: from each face's centre, nudged along its normal,
    every one of *directions* that points out of the face is cast against
    all the *triangles*; with no such ray at all, every face is outer."""
    n_faces = len(local_faces)
    origins = centers[:, None, :] + 0.001 * normals[:, None, :]
    origins = np.broadcast_to(origins, (n_faces, len(directions), 3))
    active = np.nonzero((directions @ normals.T).T > 0)
    if len(active[0]) == 0:
        return np.unique(local_faces.reshape(-1))
    hit = ray_hits(
        np.asarray(triangles, dtype=np.float64).reshape(-1, 9),
        normals,
        np.ascontiguousarray(origins[active]),
        np.ascontiguousarray(directions[active[1]]),
    )
    hit_count = np.zeros(n_faces, dtype=int)
    np.add.at(hit_count, active[0], hit)
    return np.unique(local_faces[np.nonzero(hit_count < threshold)[0]].reshape(-1))


def calibrate_radii(
    vertices,
    faces,
    centers,
    normals,
    nodes,
    edges,
    node2verts,
    skip,
    base,
    directions,
    min_n_verts_bulb: int,
    check_outer: bool,
    min_n_outer: int,
    min_frac_outer: float,
    rays_threshold: int,
    aggregate: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Each node's radius from the distances of its vertices to its
    centreline (the node and its edges): ``(radii, kinds)``.

    With *check_outer*, a node of *min_n_verts_bulb* vertices or more is
    cut out of the mesh (the faces with all three vertices among its own)
    and :func:`outer_vertices` finds which of its vertices are on the outer
    surface rather than on internal membranes; when at least
    *min_n_outer*, and *min_frac_outer* of them all, but not all are, its
    radius is the *aggregate* of their distances (kind
    :data:`OUTER_CENTERLINE`). Otherwise the aggregate over all its
    vertices replaces the *base* radius if smaller
    (:data:`FULL_CENTERLINE`). Nodes in *skip* or without vertices keep
    *base* (:data:`FALLBACK`). *centers* and *normals* are the mesh's face
    centres and normals, the same per face as a submesh's.
    """
    from .._core import _estimate_radius
    from .dx import segment_distances

    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    centers = rows3("F", centers)
    normals = rows3("F", normals)
    if len(centers) != len(faces) or len(normals) != len(faces):
        raise ValueError("centers and normals need one row per face")
    nodes = rows3("N", nodes)
    n = len(nodes)
    base = np.asarray(base, dtype=np.float64)
    skip = np.asarray(skip, dtype=bool)
    if len(base) != n or len(skip) != n or len(node2verts) != n:
        raise ValueError("one radius, skip flag and vertex list per node")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be (E, 2)")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edges name a node out of range")
    node2verts = [np.asarray(v, dtype=np.int64) for v in node2verts]
    for vids in node2verts:
        bad = (vids < 0) | (vids >= len(vertices))
        if bad.any():
            raise ValueError(f"vertex index {vids[bad][0]} is out of range")
    directions = rows3("R", directions)
    vertex_faces = VertexFaces(faces, len(vertices))

    neighbours: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges.tolist():
        neighbours[u].append(v)
        neighbours[v].append(u)

    radii = base.copy()
    kinds = np.full(n, FALLBACK, dtype=np.uint8)
    for i in range(n):
        if skip[i]:
            continue
        vids = node2verts[i]
        if vids.size == 0:
            continue
        segments = sorted({(min(i, nb), max(i, nb)) for nb in neighbours[i]})

        outer_success = False
        if check_outer and len(vids) >= min_n_verts_bulb:
            n_inner = len(vids)
            sub = vertex_faces.faces_within(vids)
            # the submesh's own vertex ids: a vertex's position in vids
            old_to_new = -np.ones(len(vertices), dtype=np.int64)
            old_to_new[vids] = np.arange(len(vids))
            outer = outer_vertices(
                vertices[faces[sub]],
                normals[sub],
                centers[sub],
                old_to_new[faces[sub]],
                directions,
                rays_threshold,
            )
            n_outer = len(outer)
            if (
                n_outer >= min_n_outer
                and n_outer >= min_frac_outer * n_inner
                and n_outer < n_inner
            ):
                d = segment_distances(vertices[vids[outer]], nodes, [i], segments)
                radii[i] = _estimate_radius(d, method=aggregate)
                kinds[i] = OUTER_CENTERLINE
                outer_success = True

        if not outer_success:
            d = segment_distances(vertices[vids], nodes, [i], segments)
            r = _estimate_radius(d, method=aggregate)
            if r < radii[i]:  # only if smaller
                radii[i] = r
                kinds[i] = FULL_CENTERLINE
    return radii, kinds
