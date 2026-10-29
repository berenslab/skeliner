"""``skeliner.dx`` on the Python path: distances from points to a skeleton,
its volume and cable, its graph's checks and walks, and its bins against the
mesh."""

from __future__ import annotations

import igraph as ig
import numpy as np
from scipy.spatial import KDTree

from .mesh import checked_faces, node_pairs, rows3


def point_segment_distance(point, start, end) -> float:
    """Euclidean distance from *point* to the segment [*start*, *end*]."""
    vec = end - start
    seg_len2 = float(np.dot(vec, vec))
    if seg_len2 <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, vec) / seg_len2)
    t = min(1.0, max(0.0, t))  # a NaN becomes 0
    closest = start + t * vec
    return float(np.linalg.norm(point - closest))


def point_segment_capsule_distance(point, start, end, r_start, r_end) -> float:
    """Signed distance from *point* to the capsule around the segment
    [*start*, *end*] whose radius goes from *r_start* to *r_end* (negative
    inside)."""
    vec = end - start
    seg_len2 = float(np.dot(vec, vec))
    if seg_len2 <= 0.0:
        return float(np.linalg.norm(point - start)) - max(float(r_start), float(r_end))
    t = float(np.dot(point - start, vec) / seg_len2)
    t = min(1.0, max(0.0, t))  # a NaN becomes 0
    closest = start + t * vec
    dist = float(np.linalg.norm(point - closest))
    return dist - ((1.0 - t) * float(r_start) + t * float(r_end))


def _k_nearest(tree, nodes: np.ndarray, p: np.ndarray, k: int):
    """The *k* nearest nodes to *p*, nearest first, among equally near ones
    the lowest index first: ``(indices, distances)``, the distances scipy's.
    scipy's own pick among ties is arbitrary, so it is asked for a few
    more, and for every node when ties reach past those."""
    n = len(nodes)
    m = min(n, k + 8)
    d, i = tree.query(p, k=m)
    d, i = np.atleast_1d(d), np.atleast_1d(i).astype(np.int64)
    if m < n and d[-1] <= d[k - 1]:
        d, i = tree.query(p, k=n)
        d, i = np.atleast_1d(d), np.atleast_1d(i).astype(np.int64)
    order = np.lexsort((i, d))[:k]
    return i[order], d[order]


class SkeletonIndex:
    """A skeleton's *nodes* ``(N, 3)``, their neighbours (a CSR *ptr*, *idx*)
    and a KD-tree over them: what ``skeliner.dx.distance`` measures against,
    built once per skeleton."""

    def __init__(self, nodes, ptr, idx):
        nodes = rows3("N", nodes)
        n = len(nodes)
        if n == 0:
            raise ValueError("Skeleton has no nodes; cannot compute distances.")
        ptr = np.asarray(ptr, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        if (
            len(ptr) != n + 1
            or ptr[0] != 0
            or np.any(np.diff(ptr) < 0)
            or ptr[-1] != len(idx)
            or np.any((idx < 0) | (idx >= n))
        ):
            raise ValueError("the neighbours must be a CSR over the nodes")
        self._nodes = nodes
        self._adjacency = [idx[ptr[i] : ptr[i + 1]].tolist() for i in range(n)]
        self._tree = KDTree(nodes)

    def distance(
        self, points, radii, k, centres, segments, scale_in, scale_out
    ) -> np.ndarray:
        """Per point of *points* ``(M, 3)``, scaled by *scale_in*, the distance
        to the nearest node or edge — to the surface given *radii* (then never
        below 0), to the centreline otherwise — scaled by *scale_out*: against
        the *k* nearest nodes (ties to the lowest index) and every edge at one
        of them, or, given *centres* and *segments* (node ids, node-id
        pairs), against those for every point."""
        points = rows3("P", points)
        nodes = self._nodes
        n = len(nodes)
        if radii is not None:
            radii = np.asarray(radii, dtype=np.float64)
            if len(radii) != n:
                raise ValueError("one radius per node")
        whitelist = centres is not None or segments is not None
        if whitelist:
            centres = list(centres or [])
            segments = [(int(a), int(b)) for a, b in (segments or [])]
            if any(i >= n for i in [*centres, *(i for s in segments for i in s)]):
                raise ValueError("node index out of range")
        elif k == 0:
            raise ValueError("k_nearest must be at least 1.")
        else:
            k = min(k, n)

        out = np.empty(len(points), dtype=np.float64)
        for j, q in enumerate(points):
            p = q * scale_in
            if whitelist:
                if centres:
                    dc = np.linalg.norm(nodes[centres] - p, axis=1)
                    best = float(
                        np.min(dc - radii[centres] if radii is not None else dc)
                    )
                else:
                    best = float("inf")
                segs = segments
            else:
                near, nd = _k_nearest(self._tree, nodes, p, k)
                best = float(np.min(nd - radii[near] if radii is not None else nd))
                segs = sorted(
                    {
                        (min(i, nb), max(i, nb))
                        for i in near.tolist()
                        for nb in self._adjacency[i]
                    }
                )
            for a, b in segs:
                if radii is not None:
                    d = point_segment_capsule_distance(
                        p, nodes[a], nodes[b], radii[a], radii[b]
                    )
                else:
                    d = point_segment_distance(p, nodes[a], nodes[b])
                if d < best:
                    best = d
            if radii is not None:
                best = max(best, 0.0)
            out[j] = best * scale_out
        return out


def segment_distances(points, nodes, centres, segments) -> np.ndarray:
    """Per point, the distance to the nearest of the *centres* (node ids) or
    of the *segments* (node-id pairs); infinity when there are neither.

    ``skeliner.dx.distance`` in centreline mode against a whitelist.
    """
    points = np.asarray(points, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    ids = [*centres, *(i for pair in segments for i in pair)]
    if any(i >= len(nodes) for i in ids):
        raise ValueError("node index out of range")
    out = np.empty(len(points), dtype=np.float64)
    for k, p in enumerate(points):
        best = float("inf")
        for c in centres:
            d = float(np.linalg.norm(nodes[c] - p))
            if d < best:
                best = d
        for a, b in segments:
            d = point_segment_distance(p, nodes[a], nodes[b])
            if d < best:
                best = d
        out[k] = best
    return out


def voxel_union_count(nodes, radii, edges, lo, hi, h, shape, soma, include_soma) -> int:
    """``dx.volume``'s voxel union: how many voxels of the grid from *lo* in
    steps of *h*, *shape* voxels per axis, have their centre in an edge's
    frustum (the radius linear along it) or, with *include_soma*, in the
    *soma*'s ellipsoid ``(center, axes, R.T, lo, hi)``; without it the
    soma's voxels are carved out after. Boxes are clipped to [*lo*, *hi*]."""
    nodes = rows3("N", nodes)
    radii = np.asarray(radii, dtype=np.float64)
    if len(radii) != len(nodes):
        raise ValueError("one radius per node")
    edges = node_pairs(edges, len(nodes))
    if not h > 0:
        raise ValueError("voxel_size must be positive")
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    nx, ny, nz = (int(n) for n in shape)
    xs = lo[0] + (np.arange(nx) + 0.5) * h
    ys = lo[1] + (np.arange(ny) + 0.5) * h
    zs = lo[2] + (np.arange(nz) + 0.5) * h
    occ = np.zeros((nx, ny, nz), dtype=bool)

    def _range(d, x0, x1):
        n = (nx, ny, nz)[d]
        i0 = int(max(0, np.floor((x0 - lo[d]) / h)))
        i1 = int(min(n - 1, np.floor((x1 - lo[d]) / h)))
        return i0, i1

    soma_slice = soma_mask = None
    if soma is not None:
        (cx, cy, cz), (ax, ay, az), rt, slo, shi = soma
        rt = np.asarray(rt, dtype=np.float64)
        slo = np.maximum(np.asarray(slo, dtype=np.float64), lo)
        shi = np.minimum(np.asarray(shi, dtype=np.float64), hi)
        (i0, i1), (j0, j1), (k0, k1) = (_range(d, slo[d], shi[d]) for d in range(3))
        if i1 >= i0 and j1 >= j0 and k1 >= k0:
            dx = xs[i0 : i1 + 1][:, None, None] - cx
            dy = ys[j0 : j1 + 1][None, :, None] - cy
            dz = zs[k0 : k1 + 1][None, None, :] - cz
            ux = rt[0, 0] * dx + rt[0, 1] * dy + rt[0, 2] * dz
            uy = rt[1, 0] * dx + rt[1, 1] * dy + rt[1, 2] * dz
            uz = rt[2, 0] * dx + rt[2, 1] * dy + rt[2, 2] * dz
            soma_mask = (ux * ux) / (ax * ax) + (uy * uy) / (ay * ay) + (uz * uz) / (
                az * az
            ) <= 1.0
            soma_slice = (slice(i0, i1 + 1), slice(j0, j1 + 1), slice(k0, k1 + 1))
    if include_soma and soma_mask is not None:
        occ[soma_slice] |= soma_mask

    for i, j in edges.tolist():
        a, b = nodes[i], nodes[j]
        r0, r1 = float(radii[i]), float(radii[j])
        rmax = max(r0, r1)
        if not np.isfinite(rmax) or rmax < 0.0:
            continue
        lo_e = np.maximum(np.minimum(a, b) - rmax, lo)
        hi_e = np.minimum(np.maximum(a, b) + rmax, hi)
        if np.any(lo_e > hi_e):
            continue
        (i0, i1), (j0, j1), (k0, k1) = (_range(d, lo_e[d], hi_e[d]) for d in range(3))
        if i1 < i0 or j1 < j0 or k1 < k0:
            continue
        dx = xs[i0 : i1 + 1][:, None, None] - a[0]
        dy = ys[j0 : j1 + 1][None, :, None] - a[1]
        dz = zs[k0 : k1 + 1][None, None, :] - a[2]
        v = b - a
        l2 = float(v @ v)
        if l2 <= 1e-24:  # a ball of the larger radius at the start
            mask = dx * dx + dy * dy + dz * dz <= rmax * rmax
        else:
            s = (dx * v[0] + dy * v[1] + dz * v[2]) / l2
            np.clip(s, 0.0, 1.0, out=s)
            rx, ry, rz = dx - s * v[0], dy - s * v[1], dz - s * v[2]
            r = r0 + s * (r1 - r0)
            mask = rx * rx + ry * ry + rz * rz <= r * r
        occ[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1] |= mask

    if not include_soma and soma_mask is not None:
        occ[soma_slice] &= ~soma_mask
    return int(occ.sum())


def segment_box_length(a, b, lo, hi) -> float:
    """How much of the segment from *a* to *b* lies inside the box [*lo*,
    *hi*] (Liang–Barsky clipping); 0 for a degenerate segment."""
    v = b - a
    L = float(np.linalg.norm(v))
    if L <= 0.0:
        return 0.0
    t0, t1 = 0.0, 1.0
    for d in range(3):
        vd = float(v[d])
        ad = float(a[d])
        eps = 1e-12 * max(1.0, abs(v[0]), abs(v[1]), abs(v[2]))
        if abs(vd) < eps:
            # parallel to this slab: inside it or not at all
            if ad < lo[d] or ad > hi[d]:
                return 0.0
            continue
        t_enter = (lo[d] - ad) / vd
        t_exit = (hi[d] - ad) / vd
        if t_enter > t_exit:
            t_enter, t_exit = t_exit, t_enter
        t0 = max(t0, t_enter)
        t1 = min(t1, t_exit)
        if t0 > t1:
            return 0.0
    return L * max(0.0, t1 - t0)


def path_length_in_box(nodes, edges, lo, hi) -> tuple[float, int]:
    """``total_path_length`` within a box: the length of the skeleton
    *edges* ``(E, 2)`` inside [*lo*, *hi*], summed edge by edge in order, and
    how many edges reach in."""
    nodes = rows3("N", nodes)
    edges = node_pairs(edges, len(nodes))
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    total, hit = 0.0, 0
    for u, v in edges.tolist():
        ell = segment_box_length(nodes[u], nodes[v], lo, hi)
        if ell > 0.0:
            total += ell
            hit += 1
    return total, hit


def arm_cables(nodes, edges, min_degree: int):
    """``_arm_cables``: every node with *min_degree* edges or more that is not
    a component's root, split into its arms — ``(junctions, degrees,
    proximal, distal_ptr, distal)``, each junction's distal cables largest
    first. Each component is rooted at its lowest node (node 0 for the
    soma's) and walked breadth first, a node's neighbours in edge order; the
    cable strictly below each node sums up in reverse of that order. An arm
    through child *c* carries the cable below *c* plus the edge to it; the
    parent's side, the component's cable less the cable below the node."""
    nodes = rows3("N", nodes)
    n = len(nodes)
    edges = node_pairs(edges, n)
    if n == 0 or edges.size == 0:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.float64),
            np.zeros(1, np.int64),
            np.empty(0, np.float64),
        )
    lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)

    # CSR-style adjacency: neighbour ids and the length of the edge used.
    deg = np.bincount(edges.reshape(-1), minlength=n)
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=start[1:])
    fill = start[:-1].copy()
    adj = np.empty(2 * len(edges), dtype=np.int64)
    adj_len = np.empty(2 * len(edges), dtype=np.float64)
    for (a, b), ln in zip(edges, lengths, strict=True):
        adj[fill[a]] = b
        adj_len[fill[a]] = ln
        fill[a] += 1
        adj[fill[b]] = a
        adj_len[fill[b]] = ln
        fill[b] += 1

    parent = np.full(n, -1, dtype=np.int64)
    parent_len = np.zeros(n, dtype=np.float64)
    order = np.empty(n, dtype=np.int64)  # BFS order, roots first
    seen = np.zeros(n, dtype=bool)
    k = 0
    for root in range(n):  # node 0 first, so the soma roots its own component
        if seen[root]:
            continue
        seen[root] = True
        order[k] = root
        k += 1
        head = k - 1
        while head < k:
            v = order[head]
            head += 1
            for j in range(start[v], start[v + 1]):
                w = adj[j]
                if not seen[w]:
                    seen[w] = True
                    parent[w] = v
                    parent_len[w] = adj_len[j]
                    order[k] = w
                    k += 1

    # Post-order accumulation: walk the BFS order backwards, so every child is
    # finished before its parent is read.
    sub = np.zeros(n, dtype=np.float64)
    for idx in range(n - 1, -1, -1):
        v = order[idx]
        p = parent[v]
        if p >= 0:
            sub[p] += sub[v] + parent_len[v]

    # Total cable of the component each node belongs to — the proximal arm is
    # measured against its own component, not the whole file.
    comp_total = np.zeros(n, dtype=np.float64)
    comp_of = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        v = order[idx]
        comp_of[v] = v if parent[v] < 0 else comp_of[parent[v]]
    for v in range(n):
        comp_total[v] = sub[comp_of[v]]

    junctions, degrees, proximal, ptr, distal = [], [], [], [0], []
    for v in range(n):
        if deg[v] < min_degree or parent[v] < 0:
            # A component root has no proximal arm, so "one arm holds the
            # soma and the others do not" is undefined there.  The soma is
            # legitimately high-degree anyway.
            continue
        arms = [
            float(sub[adj[j]] + adj_len[j])
            for j in range(start[v], start[v + 1])
            if adj[j] != parent[v]
        ]
        arms.sort(reverse=True)
        junctions.append(v)
        degrees.append(int(deg[v]))
        proximal.append(float(comp_total[v] - sub[v]))
        distal.extend(arms)
        ptr.append(len(distal))
    return (
        np.asarray(junctions, dtype=np.int64),
        np.asarray(degrees, dtype=np.int64),
        np.asarray(proximal, dtype=np.float64),
        np.asarray(ptr, dtype=np.int64),
        np.asarray(distal, dtype=np.float64),
    )


# ---------------------------------------------------------------------
# the graph: checks and walks, as igraph walks it
# ---------------------------------------------------------------------
def _graph(edges, n: int) -> ig.Graph:
    """The *edges* ``(E, 2)`` on *n* nodes as an undirected igraph graph."""
    edges = node_pairs(edges, n)
    return ig.Graph(n=n, edges=edges.tolist(), directed=False)


def _node_id(v, n: int, what: str) -> int:
    v = int(v)
    if not 0 <= v < n:
        raise ValueError(f"{what} {v} is not one of {n} nodes")
    return v


def components(edges, n: int) -> np.ndarray:
    """Each of the *n* nodes' component over the *edges* ``(E, 2)``, numbered
    in the order of their lowest node, as igraph numbers them."""
    return np.asarray(_graph(edges, n).components().membership, dtype=np.int64)


def hops(edges, n: int, source: int) -> np.ndarray:
    """Hops from *source* to each of the *n* nodes over the *edges* ``(E, 2)``,
    infinite where it does not reach."""
    g = _graph(edges, n)
    source = _node_id(source, n, "source")
    return np.asarray(g.distances(source=[source])[0], dtype=np.float64)


def cycle_basis(edges, n: int) -> list[np.ndarray]:
    """igraph's ``minimum_cycle_basis`` of the *edges* ``(E, 2)`` on *n*
    nodes: each cycle's edge ids in the order they run round it, shortest
    first."""
    basis = _graph(edges, n).minimum_cycle_basis()
    return [np.asarray(c, dtype=np.int64) for c in basis]


def branches(edges, n: int, k: int, include_endpoints: bool) -> list[list[int]]:
    """``branches_of_length``: every maximal path whose ends have a degree
    other than 2 (node 0 always an end) and whose inner nodes have degree 2,
    kept when it has *k* nodes (*k* inner nodes without
    *include_endpoints*); walked from the ends in ascending order, each
    end's neighbours in turn, a pair of nodes walked once."""
    g = _graph(edges, n)
    deg = np.asarray(g.degree(), dtype=np.int64)
    endpoints = {i for i, d in enumerate(deg.tolist()) if d != 2}
    if n:
        endpoints.add(0)

    visited_edges: set[tuple[int, int]] = set()
    out: list[list[int]] = []
    for ep in sorted(endpoints):
        for nb in g.neighbors(ep):
            if (min(ep, nb), max(ep, nb)) in visited_edges:
                continue
            path = [ep]
            prev, curr = ep, nb
            while True:
                path.append(curr)
                visited_edges.add((min(prev, curr), max(prev, curr)))
                if curr in endpoints:
                    break
                # an inner node (degree 2): on along its other edge
                nxts = [v for v in g.neighbors(curr) if v != prev]
                if not nxts:
                    break
                prev, curr = curr, nxts[0]
            length = len(path) if include_endpoints else len(path) - 2
            if length == k:
                out.append([int(v) for v in path])
    return out


def twigs(edges, n: int, k: int, include_branching_node: bool) -> list[list[int]]:
    """``twigs_of_length``: from every leaf (degree 1, node 0 aside),
    ascending, the chain up the breadth-first tree from node 0 through nodes
    of degree 2, up to the first node that is not (node 0 always ends it);
    kept when it has *k* nodes, proximal first, that node in front with
    *include_branching_node*. A leaf node 0 does not reach has none."""
    g = _graph(edges, n)
    if n == 0:
        return []
    deg = np.asarray(g.degree(), dtype=np.int64)
    # igraph's parents: -1 for node 0, -2 for a node it does not reach
    parent = g.bfs(0, mode="ALL")[2]

    out: list[list[int]] = []
    for leaf in (v for v in range(1, n) if deg[v] == 1):
        chain = [leaf]
        curr = leaf
        while True:
            par = parent[curr]
            if par < 0:
                break
            if deg[par] == 2 and par != 0:
                chain.append(par)
                curr = par
                continue
            # par is the branching point (degree other than 2, or node 0)
            if len(chain) == k:
                if include_branching_node:
                    chain.append(par)
                out.append(chain[::-1])
            break
    return out


def _distal_children(g: ig.Graph, edges: np.ndarray) -> list[list[int]]:
    """Each node's children with every edge pointing away from node 0, in
    the order the *edges* come: an edge between two nodes as many hops from
    node 0, or two it does not reach, points nowhere."""
    dists = np.asarray(g.distances(source=[0])[0], dtype=np.float64)
    children: list[list[int]] = [[] for _ in range(g.vcount())]
    for a, b in edges.tolist():
        da, db = dists[a], dists[b]
        if da == db:
            continue
        parent, child = (a, b) if da < db else (b, a)
        children[parent].append(child)
    return children


def _subtree(children: list[list[int]], root: int, include_root: bool) -> list[int]:
    """*root* and every node below it in *children*, depth first."""
    out: list[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        if v != root or include_root:
            out.append(v)
        stack.extend(children[v])
    return out


def neurite(edges, n: int, root: int, include_root: bool) -> np.ndarray:
    """``extract_neurites``: *root* and everything distal to it from node 0,
    ascending (a node reached twice, on a cycle, listed twice)."""
    edges = node_pairs(edges, n)
    g = _graph(edges, n)
    root = _node_id(root, n, "root")
    out = sorted(_subtree(_distal_children(g, edges), root, include_root))
    return np.asarray(out, dtype=np.int64)


def neurites_out_of_bounds(nodes, edges, lo, hi, include_root: bool) -> np.ndarray:
    """``neurites_out_of_bounds``: the nodes distal to (and with
    *include_root*, at) each first node outside the box ``[lo, hi]`` on the
    way out from node 0 — a node whose breadth-first parent is inside, or
    that has none — ascending, once each."""
    nodes = rows3("N", nodes)
    n = len(nodes)
    edges = node_pairs(edges, n)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    outside = np.any((nodes < lo) | (nodes > hi), axis=1)
    if not outside.any():
        return np.empty(0, dtype=np.int64)
    g = _graph(edges, n)
    # igraph gives node 0 the parent -1 and a node it does not reach -2, so
    # each of those starts a subtree of its own
    parents = g.bfs(0, mode="ALL")[2]
    children = _distal_children(g, edges)
    targets: set[int] = set()
    for nid in np.flatnonzero(outside).tolist():
        par = parents[nid]
        if par >= 0 and outside[par]:
            continue
        targets.update(_subtree(children, nid, include_root))
    return np.asarray(sorted(targets), dtype=np.int64)


def path_lengths(edges, n: int, weights, sources, target: int) -> np.ndarray:
    """The shortest path from each of *sources* to *target* over the *edges*
    ``(E, 2)`` of *n* nodes weighted by *weights* (one per edge, none NaN or
    negative), as igraph's Dijkstra sums it from the source; infinite where
    there is none."""
    g = _graph(edges, n)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != g.ecount():
        raise ValueError("one weight per edge")
    if np.isnan(weights).any():
        raise ValueError("edge lengths must not be NaN")
    if (weights < 0).any():
        raise ValueError("edge lengths must not be negative")
    sources = [_node_id(v, n, "source") for v in np.asarray(sources).tolist()]
    target = _node_id(target, n, "target")
    if not sources:
        return np.empty(0, dtype=np.float64)
    d = g.distances(source=sources, target=[target], weights=weights.tolist())
    return np.asarray(d, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------
# the bins against vert2node and the mesh
# ---------------------------------------------------------------------
def _bins(ptr, idx, n_vertices: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The bins as a CSR (*ptr* ``(K + 1,)`` from 0, never decreasing,
    *idx*), every vertex id ``0 <= id < n_vertices`` (``0 <= id`` without a
    count)."""
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if not (
        ptr.size and ptr[0] == 0 and (np.diff(ptr) >= 0).all() and ptr[-1] == idx.size
    ):
        raise ValueError("the bins must be a CSR (ptr, idx)")
    bad = (idx < 0) if n_vertices is None else (idx < 0) | (idx >= n_vertices)
    if bad.any():
        v = int(idx[np.argmax(bad)])
        if n_vertices is None:
            raise ValueError("vertex ids must not be negative")
        raise ValueError(
            f"the bins name vertex {v}, but there are {n_vertices} vertices"
        )
    return ptr, idx


def check_bins(ptr, idx, keys, vals):
    """``check_bins`` on the bins as a CSR (*ptr*, *idx*; the first is the
    soma's ``verts``, not a bin) and ``vert2node`` as *keys* → *vals*:
    ``(n_owned, duplicated, mismatched, empty_bins, soma_overlap)`` — the
    vertices a bin past the first owns, those listed more than once there,
    those whose ``vert2node`` entry is not the last bin listing them (-1 for
    none on either side), the nodes with an empty bin, and the distinct
    vertices of the first bin another bin owns too."""
    ptr, idx = _bins(ptr, idx, None)
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    vals = np.asarray(vals, dtype=np.int64).reshape(-1)
    if (keys < 0).any():
        raise ValueError("vertex ids must not be negative")
    if len(vals) != len(keys):
        raise ValueError("one node per vertex")
    n_bins = len(ptr) - 1

    owned = idx[ptr[1] :] if n_bins else idx[:0]
    uniq, counts = np.unique(owned, return_counts=True)
    duplicated = uniq[counts > 1]

    # `vert2node` must be what rebuild_vert2node() would produce: ascending
    # node order, so an arbor bin overwrites node 0 on the shared boundary.
    size = 1 + int(max(idx.max(initial=-1), keys.max(initial=-1)))
    expect = np.full(size, -1, dtype=np.int64)
    for nid in range(n_bins):
        b = idx[ptr[nid] : ptr[nid + 1]]
        if b.size:
            expect[b] = nid
    actual = np.full(size, -1, dtype=np.int64)
    actual[keys] = vals
    mismatched = np.flatnonzero(expect != actual).astype(np.int64)

    soma = idx[ptr[0] : ptr[1]] if n_bins else idx[:0]
    soma_overlap = np.intersect1d(soma, uniq).size if soma.size else 0
    empty_bins = [i for i in range(n_bins) if ptr[i + 1] == ptr[i]]
    return int(uniq.size), duplicated, mismatched, empty_bins, int(soma_overlap)


def fragmented_bins(ptr, idx, edges, n_vertices: int):
    """The bins (a CSR *ptr*, *idx* over a mesh of *n_vertices*; the first is
    the soma's ``verts``, not a bin) in more than one patch of surface over
    the mesh *edges* ``(M, 2)``: ``(nodes, patches)``, ascending by node.

    One pass over the whole mesh rather than a BFS per bin: keep only the
    mesh edges whose ends share an owner (the last bin listing a vertex),
    label the connected components of what is left, then count distinct
    labels within each bin."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ptr, idx = _bins(ptr, idx, n_vertices)
    e = node_pairs(edges, n_vertices)
    owner = np.full(n_vertices, -1, dtype=np.int64)
    for nid in range(1, len(ptr) - 1):  # node 0 is soma.verts, not a bin
        owner[idx[ptr[nid] : ptr[nid + 1]]] = nid

    keep = (owner[e[:, 0]] == owner[e[:, 1]]) & (owner[e[:, 0]] >= 1)
    e = e[keep]
    adj = coo_matrix(
        (np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])),
        shape=(n_vertices, n_vertices),
    )
    _, labels = connected_components(adj, directed=False)

    nodes, pieces = [], []
    for nid in range(1, len(ptr) - 1):
        b = idx[ptr[nid] : ptr[nid + 1]]
        if b.size > 1:
            n_pieces = np.unique(labels[b]).size
            if n_pieces > 1:
                nodes.append(nid)
                pieces.append(n_pieces)
    return np.asarray(nodes, dtype=np.int64), np.asarray(pieces, dtype=np.int64)


def face_owner(faces, owner) -> np.ndarray:
    """``face_owner``: per face of *faces* ``(F, 3)``, the node *owner* (one per
    vertex, -1 for none) gives at least two of its three vertices, or -1."""
    owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    faces, _ = checked_faces(faces, len(owner))
    per_face = np.sort(owner[faces], axis=1)
    return np.where(
        per_face[:, 0] == per_face[:, 1],
        per_face[:, 0],
        np.where(per_face[:, 1] == per_face[:, 2], per_face[:, 1], -1),
    )
