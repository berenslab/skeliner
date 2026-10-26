"""``skeliner.post`` on the Python path: ``calibrate_radii``'s ray cast and
the faces of a vertex set, and node types through edits."""

from __future__ import annotations

import numpy as np

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
    import trimesh

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


# ---------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------
AGGREGATES = ("median", "mean", "area")


def _aggregate(vals: np.ndarray, how: str) -> float:
    """One radius for a merged node from its nodes' radii *vals*."""
    if how == "median":
        return float(np.median(vals))
    if how == "mean":
        return float(np.mean(vals))
    if how == "area":  # preserve the mean cross-section
        return float(np.sqrt(np.mean(vals * vals)))
    raise ValueError("aggregate must be 'median', 'mean', or 'area'")


def _centroid(nodes: np.ndarray, lines: list[list[int]], members: list[int]):
    """The length-weighted centroid of the polylines *lines* (node ids):
    segment midpoints weighted by segment length; the mean of the
    *members* when a length is not finite or every one is zero; the one
    member's position when there is one."""
    if len(members) <= 1:
        return nodes[members[0]].copy()
    xs = [nodes[line] for line in lines]
    seg = np.concatenate([x[1:] - x[:-1] for x in xs])
    L = np.linalg.norm(seg, axis=1)
    if not np.isfinite(L).all() or np.all(L == 0):
        return nodes[members].mean(axis=0)
    mids = np.concatenate([0.5 * (x[1:] + x[:-1]) for x in xs])
    return (mids * L[:, None]).sum(axis=0) / L.sum()


def _partition_by_radius(ids: list[int], r: np.ndarray, rtol: float, atol: float):
    """Greedy 1D segmentation along a path based on a running radius
    reference: a new group starts when the next radius deviates beyond
    tolerance."""
    if not ids:
        return []
    groups: list[list[int]] = []
    cur: list[int] = [ids[0]]
    r_ref = float(r[ids[0]])
    for nid in ids[1:]:
        ri = float(r[nid])
        tol = float(atol) + float(rtol) * max(abs(r_ref), abs(ri))
        if abs(ri - r_ref) <= tol:
            cur.append(nid)
            # running mean keeps the reference centered without exploding variance
            r_ref += (ri - r_ref) / len(cur)
        else:
            groups.append(cur)
            cur = [nid]
            r_ref = ri
    groups.append(cur)
    return groups


def downsample(
    nodes,
    radii,
    key: int,
    edges,
    ptr,
    idx,
    rtol: float,
    atol: float,
    aggregate: str,
    merge_endpoints: bool,
    slide_branchpoints: bool,
    max_anchor_shift: float | None,
):
    """``post.downsample`` on arrays: *nodes* ``(N, 3)``, *radii* ``(K, N)``
    (row *key* decides), *edges* ``(E, 2)``, the bins as a CSR (*ptr*,
    *idx*) or ``None``.

    Every run of degree-2 nodes between anchors — node 0 and the nodes of
    another degree, ascending, each one's neighbours ascending; then the
    lowest node of each ring with none — splits greedily into groups whose
    radii stay within ``atol + rtol * max(|r|)`` of their running mean, each
    group one node at its length-weighted centroid with aggregated radii and
    the union of its bins. A leaf (with *merge_endpoints*) or a branch
    point (with *slide_branchpoints*), node 0 never, absorbs the group
    beside it when that group's radius is within tolerance of its own and
    it moves no more than *max_anchor_shift*; an anchor absorbing from
    several runs takes them all — the centroid of every absorbed run, the
    radii of all its nodes, their bins. A node with no other node to join
    stays as it is.

    ``(nodes, radii (K, M), edges (P, 2) in the order they were made,
    old2new (N,) with -1 for none, ptr, idx)``; ``ptr``, ``idx`` ``None``
    without bins.
    """
    nodes = rows3("N", nodes)
    n = len(nodes)
    edges = node_pairs(edges, n)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1, n)
    if aggregate not in AGGREGATES:
        raise ValueError("aggregate must be 'median', 'mean', or 'area'")
    r = radii[key]
    bins = None
    if ptr is not None:
        ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if len(ptr) != n + 1 or ptr[0] != 0 or ptr[-1] != len(idx):
            raise ValueError("the bins must be a CSR (ptr, idx) over the nodes")
        bins = [idx[ptr[i] : ptr[i + 1]] for i in range(n)]

    deg = np.bincount(edges.ravel(), minlength=n)
    adj_sets: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges.tolist():
        if a != b:
            adj_sets[a].add(b)
            adj_sets[b].add(a)
    adj = [sorted(s) for s in adj_sets]
    anchors = [v for v in range(n) if v == 0 or deg[v] != 2]

    new_nodes: list[np.ndarray] = []
    new_radii: list[list[float]] = [[] for _ in range(len(radii))]
    new_bins: list[np.ndarray] | None = [] if bins is not None else None
    new_edges: list[tuple[int, int]] = []
    old2new: dict[int, int] = {}
    held: dict[int, list[int]] = {}  # an anchor's old nodes, in aggregate order
    runs: dict[int, list[list[int]]] = {}  # the runs it absorbed

    def merged_bins(ids: list[int]) -> np.ndarray:
        parts = [bins[j] for j in ids if bins[j].size]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)

    def add_anchor(oid: int) -> int:
        if oid in old2new:
            return old2new[oid]
        nid = len(new_nodes)
        new_nodes.append(nodes[oid].astype(np.float64))
        for k in range(len(radii)):
            new_radii[k].append(float(radii[k][oid]))
        if new_bins is not None:
            new_bins.append(np.asarray(bins[oid], dtype=np.int64))
        old2new[oid] = nid
        return nid

    def add_group(gids: list[int]) -> int:
        nid = len(new_nodes)
        new_nodes.append(_centroid(nodes, [gids], gids))
        for k in range(len(radii)):
            new_radii[k].append(_aggregate(radii[k][gids], aggregate))
        if new_bins is not None:
            new_bins.append(merged_bins(gids))
        for j in gids:
            old2new[j] = nid
        return nid

    def within_tol(ra: float, rb: float) -> bool:
        tol = float(atol) + float(rtol) * max(abs(ra), abs(rb))
        return abs(ra - rb) <= tol

    def absorbs(oid: int) -> bool:
        return oid != 0 and (
            (merge_endpoints and deg[oid] == 1)
            or (slide_branchpoints and deg[oid] >= 3)
        )

    def absorb(nid: int, oid: int, group: list[int], anchor_first: bool) -> bool:
        """*group* into anchor *nid* (old node *oid*), unless that moves it
        more than *max_anchor_shift*."""
        if not within_tol(float(r[oid]), _aggregate(r[group], aggregate)):
            return False
        line = [oid] + group if anchor_first else group + [oid]
        members = line if nid not in held else held[nid] + group
        lines = [line] if nid not in runs else runs[nid] + [line]
        pos = _centroid(nodes, lines, members)
        if max_anchor_shift is not None and not (
            np.linalg.norm(pos - nodes[oid]) <= max_anchor_shift
        ):
            return False
        new_nodes[nid] = pos
        for k in range(len(radii)):
            new_radii[k][nid] = _aggregate(radii[k][members], aggregate)
        if new_bins is not None:
            new_bins[nid] = merged_bins(members)
        held[nid], runs[nid] = members, lines
        for j in group:
            old2new[j] = nid
        return True

    visited: set[tuple[int, int]] = set()

    def walk_from(a: int) -> None:
        for b in adj[a]:
            if (min(a, b), max(a, b)) in visited:
                continue
            # walk a → ... → z along the corridor
            path = [a]
            prev, cur = a, b
            visited.add((min(a, b), max(a, b)))
            while deg[cur] == 2 and cur != a:
                path.append(cur)
                nxts = [x for x in adj[cur] if x != prev]
                if len(nxts) != 1:
                    break
                nxt = nxts[0]
                visited.add((min(cur, nxt), max(cur, nxt)))
                prev, cur = cur, nxt
            if cur != path[-1]:
                path.append(cur)

            a_id, z_id = path[0], path[-1]
            left = add_anchor(a_id)
            right = add_anchor(z_id)
            groups = _partition_by_radius(path[1:-1], r, rtol, atol)
            if groups and absorbs(a_id) and absorb(left, a_id, groups[0], True):
                groups = groups[1:]
            if groups and absorbs(z_id) and absorb(right, z_id, groups[-1], False):
                groups = groups[:-1]

            # wire: left → groups → right
            L = left
            for grp in groups:
                g_id = add_group(grp)
                new_edges.append((min(L, g_id), max(L, g_id)))
                L = g_id
            if L != right:
                new_edges.append((min(L, right), max(L, right)))

    for a in anchors:
        walk_from(a)
    for v in range(n):  # a ring of degree-2 nodes, from its lowest node
        if v not in old2new and deg[v] == 2 and adj[v]:
            walk_from(v)
    for v in range(n):  # a node with no other node to join
        if v not in old2new and not adj[v]:
            add_anchor(v)

    m = len(new_nodes)
    out_nodes = np.asarray(new_nodes, dtype=np.float64).reshape(m, 3)
    out_radii = np.asarray(new_radii, dtype=np.float64).reshape(len(radii), m)
    out_edges = np.asarray(new_edges, dtype=np.int64).reshape(-1, 2)
    old2new_arr = np.full(n, -1, dtype=np.int64)
    for old, new in old2new.items():
        old2new_arr[old] = new
    if new_bins is None:
        return out_nodes, out_radii, out_edges, old2new_arr, None, None
    out_ptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum([b.size for b in new_bins], out=out_ptr[1:])
    out_idx = np.concatenate(new_bins) if new_bins else np.empty(0, np.int64)
    return (
        out_nodes,
        out_radii,
        out_edges,
        old2new_arr,
        out_ptr,
        out_idx.astype(np.int64),
    )
