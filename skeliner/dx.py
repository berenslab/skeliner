"""skeliner.dx – graph‑theoretic diagnostics for a single Skeleton"""

from typing import Any, Dict, List, Sequence, Set, Tuple

import igraph as ig
import numpy as np

__skeleton__ = [
    "connectivity",
    "acyclicity",
    "degree",
    "neighbors",
    "nodes_of_degree",
    "branches_of_length",
    "twigs_of_length",
    "suspicious_tips",
    "distance",
    "node_summary",
    "extract_neurites",
    "neurites_out_of_bounds",
    "volume",
    "total_path_length",
]

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _graph(skel) -> ig.Graph:
    """Return the undirected *igraph* view of the skeleton."""
    return skel._igraph()


# -----------------------------------------------------------------------------
# 1. connectivity & cycles
# -----------------------------------------------------------------------------


def connectivity(skel, *, return_isolated: bool = False):
    """Verify that **every** node is reachable from the soma (vertex 0).

    Parameters
    ----------
    skel
        A :class:`skeliner.Skeleton` instance.
    return_isolated
        When *True* return a list of orphan node indices instead of a boolean.
    """
    g = _graph(skel)
    order, _, _ = g.bfs(0, mode="ALL")  # order[i] == -1 ⇔ unreachable
    reachable = {v for v in order if v != -1}
    if return_isolated:
        return [i for i in range(g.vcount()) if i not in reachable]
    return len(reachable) == g.vcount()


def acyclicity(skel, *, return_cycles: bool = False):
    """Check that the skeleton is a *forest* (|E| = |V| − components).

    If a cycle exists and ``return_cycle`` is *True*, a representative list of
    (u, v) edges forming the cycle is returned.
    """
    g = _graph(skel)
    n_comp = len(g.components())
    acyclic = g.ecount() == g.vcount() - n_comp
    if acyclic or not return_cycles:
        return acyclic
    cyc = g.cycle_basis()[0]  # list of vertex ids
    return [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]


# -----------------------------------------------------------------------------
# 2. degree-related helpers
# -----------------------------------------------------------------------------


def degree(skel, node_id: int | Sequence[int]):
    """Return the degree(s) of one node *or* a sequence of nodes."""
    g = _graph(skel)
    if isinstance(node_id, (list, tuple, np.ndarray)):
        return np.asarray(g.degree(node_id))
    return int(g.degree(node_id))


def neighbors(skel, node_id: int) -> List[int]:
    """Neighbour vertex IDs of *node_id* (undirected)."""
    g = _graph(skel)
    return [int(v) for v in g.neighbors(node_id)]


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Return Euclidean distance from *point* to the segment [start, end]."""
    vec = end - start
    seg_len2 = float(np.dot(vec, vec))
    if seg_len2 <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, vec) / seg_len2)
    t = min(1.0, max(0.0, t))
    closest = start + t * vec
    return float(np.linalg.norm(point - closest))


def _point_segment_capsule_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    r_start: float,
    r_end: float,
) -> float:
    """
    Return signed distance from *point* to the capsule defined by
    segment [start, end] with radii r_start → r_end.

    Negative values mean the point falls inside the interpolated radius.
    """
    vec = end - start
    seg_len2 = float(np.dot(vec, vec))
    if seg_len2 <= 0.0:
        radius = max(float(r_start), float(r_end))
        return float(np.linalg.norm(point - start)) - radius

    t = float(np.dot(point - start, vec) / seg_len2)
    t = min(1.0, max(0.0, t))
    closest = start + t * vec
    dist = float(np.linalg.norm(point - closest))
    radius = (1.0 - t) * float(r_start) + t * float(r_end)
    return dist - radius


def distance(
    skel,
    point: Sequence[float] | np.ndarray,
    *,
    point_unit: str | None = None,
    k_nearest: int = 4,
    radius_metric: str | None = None,
    mode: str = "surface",
) -> float | np.ndarray:
    """
    Distance from an arbitrary point (or collection of points) to the skeleton.

    Parameters
    ----------
    skel
        :class:`skeliner.Skeleton` instance.
    point
        3-vector or array of shape (M, 3) giving query locations.
    point_unit
        Unit of the input coordinates and the returned distance. If ``None`` or
        identical to ``skel.meta['unit']``, no conversion is performed.
    k_nearest
        Number of nearest skeleton nodes considered when refining the distance
        against neighbouring edges (≥ 1).
    radius_metric
        Which column of ``skel.radii`` to use. Defaults to the recommended estimator.
        Only consulted when *mode* is ``'surface'``.
    mode
        ``'surface'`` (default) returns the distance to the radius-aware capsule
        envelope (values inside clamp to ``0``). ``'centerline'`` measures distance
        to the centreline alone.

    Returns
    -------
    float or ndarray
        Minimum distance(s) in the same unit as *point_unit*. With
        ``mode='surface'`` the envelope distance is returned (0 inside); otherwise
        the pure centreline distance.
    """
    if mode not in {"surface", "centerline"}:
        raise ValueError("mode must be either 'surface' or 'centerline'")
    surface = mode == "surface"

    pts = np.asarray(point, dtype=np.float64)
    if pts.ndim == 1:
        if pts.shape[0] != 3:
            raise ValueError("point must be a 3-vector or an array of shape (M, 3)")
        pts = pts[None, :]
        single_input = True
    elif pts.ndim == 2 and pts.shape[1] == 3:
        single_input = False
    else:
        raise ValueError("point must be a 3-vector or an array of shape (M, 3)")

    if skel.nodes.size == 0:
        raise ValueError("Skeleton has no nodes; cannot compute distances.")
    if k_nearest < 1:
        raise ValueError("k_nearest must be at least 1.")

    tree = skel._ensure_nodes_kdtree()
    neighbours = skel._ensure_node_neighbors()

    skel_unit = skel.meta.get("unit")
    if point_unit is None or skel_unit is None or point_unit == skel_unit:
        scale_in = 1.0
        scale_out = 1.0
    else:
        scale_in = skel._get_unit_conversion_factor(point_unit, skel_unit)
        scale_out = skel._get_unit_conversion_factor(skel_unit, point_unit)

    if surface:
        if radius_metric is None:
            radius_metric = skel.recommend_radius()[0]
        if radius_metric not in skel.radii:
            raise ValueError(
                f"radius_metric '{radius_metric}' not found in skel.radii "
                f"(available keys: {tuple(skel.radii)})"
            )
        radii = np.asarray(skel.radii[radius_metric], dtype=np.float64)
    else:
        radii = None

    distances = np.empty(len(pts), dtype=np.float64)
    max_k = min(int(k_nearest), len(skel.nodes))
    nodes = skel.nodes

    for i, p in enumerate(pts):
        p_skel = p * scale_in

        nn_dist, nn_idx = tree.query(p_skel, k=max_k)
        nn_idx_arr = np.atleast_1d(nn_idx).astype(np.int64, copy=False)
        nn_dist_arr = np.atleast_1d(nn_dist)
        if surface:
            best = float(np.min(nn_dist_arr - radii[nn_idx_arr]))
        else:
            best = float(nn_dist_arr.min())

        # Collect unique edges incident to the nearest nodes
        candidates: set[tuple[int, int]] = set()
        for nid in nn_idx_arr:
            for nb in neighbours[nid]:
                if nid < nb:
                    candidates.add((nid, nb))
                else:
                    candidates.add((nb, nid))

        if candidates:
            for a_idx, b_idx in candidates:
                if surface:
                    d = _point_segment_capsule_distance(
                        p_skel,
                        nodes[a_idx],
                        nodes[b_idx],
                        radii[a_idx],
                        radii[b_idx],
                    )
                else:
                    d = _point_segment_distance(p_skel, nodes[a_idx], nodes[b_idx])
                if d < best:
                    best = d

        if surface:
            distances[i] = max(best, 0.0) * scale_out
        else:
            distances[i] = best * scale_out

    return float(distances[0]) if single_input else distances


def node_summary(
    skel,
    node_id: int,
    *,
    radius_metric: str | None = None,
) -> Dict[str, Any]:
    """Rich information about a single vertex.

    Returned dict structure::

        {
            "degree": int,
            "radius": float,
            "neighbors": [
                {"id": j, "degree": int, "radius": float},
                ...
            ]
        }
    """
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    deg_root = degree(skel, node_id)
    r_root = float(skel.radii[radius_metric][node_id])

    out = {
        "degree": deg_root,
        "radius": r_root,
        "neighbors": [],
    }
    for nb in neighbors(skel, node_id):
        out["neighbors"].append(
            {
                "id": int(nb),
                "degree": degree(skel, nb),
                "radius": float(skel.radii[radius_metric][nb]),
            }
        )
    return out


# -----------------------------------------------------------------------------
# 3. degree distribution with optional detailed map
# -----------------------------------------------------------------------------


def degree_distribution(
    skel,
    *,
    high_deg_percentile: float = 99.5,
    detailed: bool = False,
    radius_metric: str | None = None,
) -> Dict[str, Any]:
    """Histogram + outliers; optionally attach neighbour radii/deg info.

    Parameters
    ----------
    high_deg_percentile
        Percentile threshold that defines *high-degree* nodes.
    detailed
        When *True* each high-degree node is expanded to include its
        neighbours' IDs, degrees and radii.
    radius_metric
        Which radius column to report. Default = the estimator recommended
        by :py:meth:`Skeleton.recommend_radius`.
    """
    g = _graph(skel)
    deg = np.asarray(g.degree())

    hist = np.bincount(deg)
    thresh = np.percentile(deg, high_deg_percentile)
    high = np.where(deg > thresh)[0]

    high_dict: Dict[int, Any] = {}
    for idx in high:
        high_dict[int(idx)] = int(deg[idx])
        if detailed:
            high_dict[int(idx)] = node_summary(
                skel, int(idx), radius_metric=radius_metric
            )

    return {
        "degree": np.arange(hist.size)[1:],
        "counts": hist[1:],
        "threshold": float(thresh),
        "high_degree_nodes": high_dict,
    }


def nodes_of_degree(skel, k: int):
    """Return *all* node IDs whose degree == *k* (soma excluded).

    Examples
    --------
    >>> leaves = sk.dx.degree_eq(skel, 1)
    >>> hubs   = sk.dx.degree_eq(skel, 4)
    """
    if k < 0:
        raise ValueError("k must be non‑negative")
    g = _graph(skel)
    deg = np.asarray(g.degree())
    idx = np.where(deg == k)[0]
    if k == deg[0]:  # avoid returning the soma
        idx = idx[idx != 0]
    return idx.astype(int)


def branches_of_length(
    skel,
    k: int,
    *,
    include_endpoints: bool = True,
) -> List[List[int]]:
    """Return every *branch* (sequence of degree‑2 vertices) whose length == k.

    Definition of a *branch*
    ------------------------
    A maximal simple path **P** such that:
    * the two endpoints have degree ≠ 2 (soma, bifurcation, or leaf), and
    * every interior vertex (if any) has degree == 2.

    Example – degree pattern ``1‑2‑2‑3``::

        0‑1‑2‑3
        ^   ^   ^
        |   |   +—— endpoint (deg != 2)
        |   +—— interior (deg == 2)
        +—— endpoint (leaf)

    ``branches_of_length(skel, k=3)`` would return ``[[0,1,2]]``.

    Parameters
    ----------
    k
        Desired branch length *in number of nodes* (``len(path)``).
    include_endpoints
        If *True* endpoints are counted as part of the path and therefore
        contribute to *k*.  If *False* only the *interior* degree‑2 vertices
        are counted.
    """
    g = _graph(skel)
    deg = np.asarray(g.degree())
    N = len(deg)

    # Mark endpoints = vertices with degree != 2 OR soma (0) even if deg==2
    endpoints: Set[int] = {i for i, d in enumerate(deg) if d != 2}
    endpoints.add(0)

    visited_edges: Set[Tuple[int, int]] = set()
    branches: List[List[int]] = []

    for ep in endpoints:
        for nb in g.neighbors(ep):
            edge = tuple(sorted((ep, nb)))
            if edge in visited_edges:
                continue

            path = [ep]
            prev, curr = ep, nb
            while True:
                path.append(curr)
                visited_edges.add(
                    (min(int(prev), int(curr)), max(int(prev), int(curr)))
                )
                if curr in endpoints:
                    break
                # internal vertex (deg==2) → continue straight
                nxts = [v for v in g.neighbors(curr) if v != prev]
                if not nxts:
                    break  # should not happen in a well‑formed tree
                prev, curr = curr, nxts[0]

            length = len(path) if include_endpoints else len(path) - 2
            if length == k:
                branches.append([int(v) for v in path])

    return branches


def twigs_of_length(
    skel,
    k: int,
    *,
    include_branching_node: bool = False,
) -> List[List[int]]:
    """
    Return every *terminal twig* whose **chain length** == k.

    *Twig length* counts the leaf (deg==1) and all intermediate deg==2
    vertices **up to but NOT including** the branching point (deg>2 or soma).

    Parameters
    ----------
    k  : int
        Number of vertices in the terminal chain *excluding* the branching
        node.  Example::

            soma-B-1-2-L        # degrees  >2-2-2-1
                 └─┬──────      k = 3   (1-2-L)
                   `- returned path length is 3 or 4
                      depending on include_branching_node
    include_branching_node : bool, default ``False``
        If *True*, the branching node is prepended to each returned path.

    Returns
    -------
    list[list[int]]
        Each sub-list is ordered **proximal ➜ leaf**.
        * Length == k              when include_branching_node=False
        * Length == k + 1          when include_branching_node=True
    """
    g = _graph(skel)
    deg = np.asarray(g.degree())

    twigs: List[List[int]] = []

    # candidates = all leaves (deg==1, exclude soma)
    leaves = [v for v in range(1, len(deg)) if deg[v] == 1]
    parent = g.bfs(0, mode="ALL")[2]

    for leaf in leaves:
        chain = [leaf]
        curr = leaf
        while True:
            par = parent[curr]
            if par == -1:
                break  # should not happen – disconnected
            if deg[par] == 2 and par != 0:
                chain.append(par)
                curr = par
                continue
            # par is branching point (deg!=2 or soma)
            if len(chain) == k:
                if include_branching_node:
                    chain.append(par)
                twigs.append(chain[::-1])  # proximal➜distal order
            break

    return twigs


# -----------------------------------------------------------------------------
# 3. leaf depths (BFS distance in *edges* from soma)
# -----------------------------------------------------------------------------


def leaf_depths(skel) -> np.ndarray:
    """Depth (in *edges*) of every leaf node relative to the soma."""
    g = _graph(skel)
    deg = np.asarray(g.degree())
    leaves = np.where((deg == 1) & (np.arange(len(deg)) != 0))[0]
    if leaves.size == 0:
        return np.empty(0, dtype=int)
    depths = np.asarray(g.shortest_paths(source=0))[0]
    return depths[leaves].astype(int)


def suspicious_tips(
    skel,
    *,
    near_factor: float = 1.2,
    path_ratio_thresh: float = 2.0,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[int, Dict[str, float]]]:
    r"""Identify *tip* nodes suspiciously close to the soma.

    A *tip* is a node with graph degree = 1 (i.e. a leaf) and **not** the soma
    itself.  A leaf *i* is flagged when

    1. Its Euclidean distance to the soma center is *small*::

           d\_euclid(i) \le near\_factor × max(soma.axes)

    2. Yet the shortest‑path length along the skeleton is *long*::

           d\_graph(i) / d\_euclid(i) \ge path\_ratio\_thresh

    Parameters
    ----------
    skel
        A fully‑constructed :class:`skeliner.Skeleton` instance.
    near_factor
        Multiplicative factor applied to the largest soma semi‑axis to set the
        *Euclidean* proximity threshold.
    path_ratio_thresh
        Minimum ratio between graph‑path length and straight‑line distance for
        a leaf to be considered suspicious.
    return_stats
        If *True*, a per‑node diagnostic dictionary is returned in addition to
        the sorted list of suspicious node IDs.

    Returns
    -------
    suspicious
        ``list[int]`` – tip node indices, sorted by decreasing *path/straight*
        ratio (most suspicious first).
    stats
        *Optional* ``dict[int, dict]`` where each entry contains::

            {"d_center", "d_surface", "path_len", "ratio"}
    """
    if skel.nodes.size == 0 or skel.edges.size == 0:
        return [] if not return_stats else ([], {})

    soma_c = skel.soma.center.astype(np.float64)
    r_max = float(skel.soma.axes.max())
    near_thr = near_factor * r_max

    # Build an igraph view with edge‑length weights (Euclidean)
    g: ig.Graph = skel._igraph()
    g.es["weight"] = [
        float(np.linalg.norm(skel.nodes[a] - skel.nodes[b])) for a, b in skel.edges
    ]

    # Tip detection – degree = 1, excluding the soma (node 0)
    deg = np.bincount(skel.edges.flatten(), minlength=len(skel.nodes))
    tips = np.where(deg == 1)[0]
    tips = tips[tips != 0]  # exclude soma itself
    if tips.size == 0:
        return [] if not return_stats else ([], {})

    # Shortest path (edge‑weighted) length to soma for every tip
    path_d = np.asarray(
        g.distances(source=list(tips), target=[0], weights="weight"),
        dtype=np.float64,
    ).reshape(-1)

    # Straight‑line metrics
    eucl_d = np.linalg.norm(skel.nodes[tips] - soma_c, axis=1)
    surf_d = skel.soma.distance_to_surface(skel.nodes[tips])

    # Robust guard against division by zero (very unlikely)
    ratio = path_d / np.maximum(eucl_d, 1e-9)

    sus_mask = (eucl_d <= near_thr) & (ratio >= path_ratio_thresh)
    suspicious = tips[sus_mask]

    if not return_stats:
        # sort by descending ratio (most egregious first)
        return sorted(
            map(int, suspicious), key=lambda nid: -ratio[np.where(tips == nid)[0][0]]
        )

    stats: Dict[int, Dict[str, float]] = {
        int(nid): {
            "d_center": float(eucl_d[i]),
            "d_surface": float(surf_d[i]),
            "path_len": float(path_d[i]),
            "ratio": float(ratio[i]),
        }
        for i, nid in enumerate(tips)
        if sus_mask[i]
    }

    suspicious_sorted = sorted(suspicious, key=lambda nid: -stats[int(nid)]["ratio"])
    return suspicious_sorted, stats


def extract_neurites(
    skel,
    root: int,
    *,
    include_root: bool = True,
) -> List[int]:
    """Return the full *neurite subtree* emerging distally from ``root``.

    The routine uses *graph distance* to the soma (node 0) to orient edges:
    for every edge ``(u, v)`` the direction is from the **closer** vertex to
    soma → **further** vertex.  All vertices whose shortest path to soma passes
    through ``root`` (including any downstream bifurcations) are returned.

        The skeleton is assumed to be a tree (acyclic).  *Distal* means all
        descendants of ``root`` when the soma (vertex 0) is treated as the
        root.

        Examples
        --------
        >>> skel.dx.extract_neurite(skel, 2)
        [2, 3, 4, 5, ...]   # entire subtree starting at 2
        >>> skel.dx.extract_neurite(skel, 0, include_root=False)
        list(range(1, len(skel.nodes)))  # every non‑soma node

        Parameters
        ----------
        skel
            A :class:`skeliner.Skeleton` instance.
        root
            Index of the *proximal* node that defines the neurite base.
        include_root : bool, default ``True``
            Whether ``root`` itself should be included in the returned list.

        Returns
        -------
        list[int]
            Sorted vertex IDs belonging to the neurite.
    """
    N = len(skel.nodes)
    if root < 0 or root >= N:
        raise ValueError("root is out of range")

    # 1. shortest‑path distance from soma to EVERY node (unweighted graph)
    g = skel._igraph()
    dists = np.asarray(g.shortest_paths(source=[0])[0], dtype=int)

    # 2. build children[]: edge directed along *increasing* distance
    children: List[List[int]] = [[] for _ in range(N)]
    for a, b in skel.edges:
        da, db = dists[a], dists[b]
        if da == db:
            # should not happen in a tree, but guard anyway
            continue
        parent, child = (a, b) if da < db else (b, a)
        children[parent].append(child)

    # 3. DFS from root collecting all downstream vertices
    out: List[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        if v != root or include_root:
            out.append(v)
        stack.extend(children[v])

    return sorted(out)


def neurites_out_of_bounds(
    skel,
    bounds: tuple[np.ndarray, np.ndarray] | tuple[Sequence[float], Sequence[float]],
    *,
    include_root: bool = True,
) -> list[int]:
    """
    Return all node IDs that belong to a *distal* subtree whose **root is the
    first node that leaves the axis-aligned bounding box** ``bounds``.

    Parameters
    ----------
    bounds
        ``(lo, hi)`` – each a 3-vector.  A node is inside iff
        ``lo <= coord <= hi`` component-wise.
    include_root
        Whether the very first out-of-bounds node should be included in the
        output.  (Default: ``True``.)

    Notes
    -----
    * Works on acyclic skeletons (trees).
    * Uses only igraph helpers; no custom BFS routine.
    """
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bounds must be two length-3 vectors")

    coords = skel.nodes
    outside = np.any((coords < lo) | (coords > hi), axis=1)
    if not outside.any():
        return []

    # ------------------------------------------------------------------
    # one igraph shortest-path pass from soma (vertex 0)
    # ------------------------------------------------------------------
    g = skel._igraph()
    dists = np.asarray(g.shortest_paths(source=[0])[0], dtype=int)

    # For every edge (u,v) orient from proximal→distal
    children: list[list[int]] = [[] for _ in range(len(coords))]
    for u, v in skel.edges:
        parent, child = (u, v) if dists[u] < dists[v] else (v, u)
        children[parent].append(child)

    # ------------------------------------------------------------------
    # Treat each *first* out-of-bounds node as a neurite root
    # ------------------------------------------------------------------
    targets: set[int] = set()
    for nid in np.where(outside)[0]:
        # ensure nid is indeed the *first* outside node on its path
        par = g.bfs(0, mode="ALL")[2][nid]
        if par != -1 and outside[par]:
            continue  # ancestor already outside – skip
        targets.update(extract_neurites(skel, int(nid), include_root=include_root))

    return sorted(targets)


# -----------------------------------------------------------------------------
# volume helpers
# -----------------------------------------------------------------------------


def _parse_bbox(bbox) -> tuple[np.ndarray, np.ndarray] | None:
    """Accepts [xmin, xmax, ymin, ymax, zmin, zmax] or ((xlo,ylo,zlo),(xhi,yhi,zhi))."""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 6:
        lo = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float64)
        hi = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float64)
        if not np.all(lo <= hi):
            raise ValueError("bbox must satisfy lo <= hi in each axis")
        return lo, hi
    if isinstance(bbox, (list, tuple)) and len(bbox) == 2:
        lo = np.asarray(bbox[0], dtype=np.float64).reshape(3)
        hi = np.asarray(bbox[1], dtype=np.float64).reshape(3)
        if not np.all(lo <= hi):
            raise ValueError("bbox must satisfy lo <= hi in each axis")
        return lo, hi
    raise ValueError("bbox must be [xmin,xmax,ymin,ymax,zmin,zmax] or (lo, hi)")


def _ellipsoid_aabb(soma) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a rotated ellipsoid."""
    # For x = c + R diag(a) u, u ∈ unit sphere, the extreme along axis i is:
    #   c_i ± sum_j |R_{ij}| * a_j
    R = soma.R
    a = soma.axes
    extents = np.abs(R) @ a
    lo = soma.center - extents
    hi = soma.center + extents
    return lo, hi


def _choose_voxel_size(
    radii: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    target_voxels: float = 1e8,
    user_voxel: float | None = None,
    min_voxels_across_diam: int = 24,
) -> tuple[float, tuple[int, int, int]]:
    if user_voxel is not None and user_voxel <= 0:
        raise ValueError("voxel_size must be positive")

    if user_voxel is not None:
        base = float(user_voxel)
    else:
        pos = radii[radii > 0]
        if pos.size:
            base = float(np.percentile(pos, 25)) / 3.0
            base = max(base, 1e-6)
            r_ref = float(np.percentile(pos, 10))
            if r_ref > 0:
                base = min(base, (2.0 * r_ref) / float(min_voxels_across_diam))
        else:
            span = float(np.max(hi - lo))
            base = max(span / 256.0, 1e-6)

    span = hi - lo
    n_est = np.ceil(span / base).astype(int)
    est_total = float(n_est[0] * n_est[1] * n_est[2])

    if est_total > target_voxels:
        scale = (est_total / target_voxels) ** (1.0 / 3.0)
        base *= scale
        n_est = np.ceil(span / base).astype(int)

    n_est = np.maximum(n_est, 1)
    return float(base), (int(n_est[0]), int(n_est[1]), int(n_est[2]))


def _voxelize_union(
    skel,
    radii: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    voxel_size: float | None,
    include_soma: bool,
):
    """
    Boolean occupancy grid for the union of edge frusta and (optionally) soma,
    inside [lo, hi].  Returns (occ, h, (nx,ny,nz), lo, hi).
    """
    h, (nx, ny, nz) = _choose_voxel_size(radii, lo, hi, user_voxel=voxel_size)
    xs = lo[0] + (np.arange(nx) + 0.5) * h
    ys = lo[1] + (np.arange(ny) + 0.5) * h
    zs = lo[2] + (np.arange(nz) + 0.5) * h

    occ = np.zeros((nx, ny, nz), dtype=bool)
    nodes = skel.nodes.astype(np.float64, copy=False)
    edges = skel.edges.astype(np.int64, copy=False)

    # ---------------- helpers ----------------
    def _idx_range(lo_e, hi_e):
        """Index range [i0,i1] (clipped) for an axis."""
        i0 = int(max(0, np.floor((lo_e - lo) / h)))
        i1 = int(min([nx - 1, ny - 1, nz - 1]))  # overwritten per-axis
        return i0, i1

    def _range_x(x0, x1):
        i0 = int(max(0, np.floor((x0 - lo[0]) / h)))
        i1 = int(min(nx - 1, np.floor((x1 - lo[0]) / h)))
        return i0, i1

    def _range_y(y0, y1):
        j0 = int(max(0, np.floor((y0 - lo[1]) / h)))
        j1 = int(min(ny - 1, np.floor((y1 - lo[1]) / h)))
        return j0, j1

    def _range_z(z0, z1):
        k0 = int(max(0, np.floor((z0 - lo[2]) / h)))
        k1 = int(min(nz - 1, np.floor((z1 - lo[2]) / h)))
        return k0, k1

    # --------- precompute soma mask once (if there is a soma) ------------
    soma_slice = None
    soma_mask = None
    if getattr(skel, "soma", None) is not None:
        slo, shi = _ellipsoid_aabb(skel.soma)
        # clip to global bbox
        slo = np.maximum(slo, lo)
        shi = np.minimum(shi, hi)
        i0, i1 = _range_x(slo[0], shi[0])
        j0, j1 = _range_y(slo[1], shi[1])
        k0, k1 = _range_z(slo[2], shi[2])

        if (i1 >= i0) and (j1 >= j0) and (k1 >= k0):
            # broadcasted coordinate slabs (no big meshgrid; uses ogrid)
            xi = xs[i0 : i1 + 1][:, None, None]
            yj = ys[j0 : j1 + 1][None, :, None]
            zk = zs[k0 : k1 + 1][None, None, :]

            cx, cy, cz = skel.soma.center
            Rt = skel.soma.R.T  # 3x3
            ax, ay, az = skel.soma.axes
            ax2, ay2, az2 = ax * ax, ay * ay, az * az

            dx = xi - cx
            dy = yj - cy
            dz = zk - cz

            # rotate into soma body-frame: u = R^T (x - c)
            ux = Rt[0, 0] * dx + Rt[0, 1] * dy + Rt[0, 2] * dz
            uy = Rt[1, 0] * dx + Rt[1, 1] * dy + Rt[1, 2] * dz
            uz = Rt[2, 0] * dx + Rt[2, 1] * dy + Rt[2, 2] * dz

            soma_mask = (ux * ux) / ax2 + (uy * uy) / ay2 + (uz * uz) / az2 <= 1.0
            soma_slice = (slice(i0, i1 + 1), slice(j0, j1 + 1), slice(k0, k1 + 1))

    # If soma is to be included, OR it now.
    if include_soma and soma_mask is not None:
        occ[soma_slice] |= soma_mask

    # ---------------- rasterize every edge (broadcasted) -----------------
    for i, j in edges:
        a = nodes[i]
        b = nodes[j]
        r0 = float(radii[i])
        r1 = float(radii[j])

        rmax = max(r0, r1)
        if not np.isfinite(rmax) or rmax < 0.0:
            continue

        # edge AABB padded by rmax, clipped to [lo,hi]
        lo_e = np.maximum(np.minimum(a, b) - rmax, lo)
        hi_e = np.minimum(np.maximum(a, b) + rmax, hi)
        if np.any(lo_e > hi_e):
            continue

        ii0, ii1 = _range_x(lo_e[0], hi_e[0])
        jj0, jj1 = _range_y(lo_e[1], hi_e[1])
        kk0, kk1 = _range_z(lo_e[2], hi_e[2])
        if (ii1 < ii0) or (jj1 < jj0) or (kk1 < kk0):
            continue

        xi = xs[ii0 : ii1 + 1][:, None, None]
        yj = ys[jj0 : jj1 + 1][None, :, None]
        zk = zs[kk0 : kk1 + 1][None, None, :]

        v = b - a
        L2 = float(v @ v)
        if L2 <= 1e-24:
            # degenerate: paint a ball of radius rmax at 'a'
            dx = xi - a[0]
            dy = yj - a[1]
            dz = zk - a[2]
            d2 = dx * dx + dy * dy + dz * dz
            mask = d2 <= (rmax * rmax)
            occ[ii0 : ii1 + 1, jj0 : jj1 + 1, kk0 : kk1 + 1] |= mask
            continue

        vx, vy, vz = v
        # projection parameter s (broadcasted), then clamp to [0,1]
        dx = xi - a[0]
        dy = yj - a[1]
        dz = zk - a[2]
        s = (dx * vx + dy * vy + dz * vz) / L2
        # clip in-place to save a temporary
        np.clip(s, 0.0, 1.0, out=s)

        # distance from voxel center to closest point on segment
        rx = dx - s * vx
        ry = dy - s * vy
        rz = dz - s * vz
        d2 = rx * rx + ry * ry + rz * rz

        # linear radius along the frustum
        r = r0 + s * (r1 - r0)
        mask = d2 <= (r * r)

        occ[ii0 : ii1 + 1, jj0 : jj1 + 1, kk0 : kk1 + 1] |= mask

    # If soma is to be excluded, carve it out once (reuses precomputed mask).
    if (not include_soma) and (soma_mask is not None):
        occ[soma_slice] &= ~soma_mask

    return occ, float(h), (int(nx), int(ny), int(nz)), lo, hi


def volume(
    skel,
    bbox: list[float] | tuple[Sequence[float], Sequence[float]] | None = None,
    *,
    radius_metric: str | None = None,
    voxel_size: float | None = None,
    include_soma: bool = True,
    return_details: bool = False,
):
    """
    Estimate the morphology volume, optionally restricted to an axis-aligned bbox.

    Robust union via voxelization inside the bbox: fills voxels that lie
    inside any edge frustum or the soma ellipsoid. Correctly handles
    branch overlaps and bbox clipping. Accuracy controlled by `voxel_size`
    (defaults to ~1/3 of a thin-branch radius, auto-scaled to keep the grid
    under ~60M voxels unless you pass voxel_size explicitly).

    Parameters
    ----------
    bbox
        None (whole neuron) or [xmin, xmax, ymin, ymax, zmin, zmax] or (lo, hi).
    radius_metric
        Which `skel.radii[metric]` column to use; defaults to the
        choice from `skel.recommend_radius()`.
    voxel_size
        Edge length of voxels. If None, a size is chosen
        from radii and capped so the grid stays reasonably small.
    include_soma
        Whether to include the soma ellipsoid in the volume.
    return_details
        If True, returns (V, details_dict) with diagnostic info for debugging.

    Returns
    -------
    float or (float, dict)
        Estimated volume (in the cube of your skeleton units). If
        `return_details=True`, also returns a small diagnostics dict.

    Notes
    -----
    * 'frustum' is blazing-fast and good for whole-cell summaries. It trims
      soma overlap on edges but **does not** de-overlap at branch junctions.
    * 'voxel' is the accurate union (no double counting) and is recommended
      whenever `bbox` is used or precise union is important.
    """
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_metric], dtype=np.float64).reshape(-1)
    if radii.shape[0] != skel.nodes.shape[0]:
        raise ValueError("radius_metric array length must match number of nodes")

    # dispatch
    lo_hi = _parse_bbox(bbox)

    if lo_hi is None:
        # Auto bbox that tightly encloses neuron + radii + soma extents
        lo_nodes = (skel.nodes - radii[:, None]).min(axis=0)
        hi_nodes = (skel.nodes + radii[:, None]).max(axis=0)
        if include_soma and skel.soma is not None:
            slo, shi = _ellipsoid_aabb(skel.soma)
            lo = np.minimum(lo_nodes, slo)
            hi = np.maximum(hi_nodes, shi)
        else:
            lo, hi = lo_nodes, hi_nodes
    else:
        lo, hi = lo_hi

    occ, h, (nx, ny, nz), lo, hi = _voxelize_union(
        skel, radii, lo, hi, voxel_size=voxel_size, include_soma=include_soma
    )
    count = int(occ.sum())
    V = float(count) * (h**3)
    if not return_details:
        return V
    return V, {
        "voxel_size": h,
        "grid_shape": (nx, ny, nz),
        "bbox_lo": lo,
        "bbox_hi": hi,
        "filled_voxels": count,
    }


# -----------------------------------------------------------------------------
# path length (sum of edge lengths) with optional bbox clipping
# -----------------------------------------------------------------------------


def _segment_aabb_clip_length(
    a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """
    Return the length of the line segment [a,b] that lies inside the
    axis-aligned box [lo, hi].  Zero if there is no intersection.
    """
    v = b - a
    L = float(np.linalg.norm(v))
    if L <= 0.0:
        # Degenerate edge → contributes nothing (even if "inside").
        return 0.0

    # Liang–Barsky style parametric clipping in 3D.
    t0, t1 = 0.0, 1.0
    for d in range(3):
        vd = float(v[d])
        ad = float(a[d])
        eps = 1e-12 * max(1.0, abs(v[0]), abs(v[1]), abs(v[2]))
        if abs(vd) < eps:
            # Segment is parallel to this slab; reject if outside.
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


def total_path_length(
    skel,
    bbox: list[float] | tuple[Sequence[float], Sequence[float]] | None = None,
    *,
    return_details: bool = False,
):
    """
    Sum of Euclidean edge lengths, optionally **clipped** to an axis-aligned bbox.

    Parameters
    ----------
    bbox
        None → full skeleton length (no clipping); or
        [xmin, xmax, ymin, ymax, zmin, zmax]; or ((xlo,ylo,zlo), (xhi,yhi,zhi)).
    return_details
        When True, also returns a small diagnostics dict.

    Returns
    -------
    float or (float, dict)
        Total path length in the same units as your coordinates.

    Notes
    -----
    * This is purely geometric path length over the graph edges. It does **not**
      subtract portions running inside the soma ellipsoid.
    * Complexity O(|E|). Numerically robust to nearly-parallel edges.
    """
    nodes = skel.nodes.astype(np.float64)
    edges = np.asarray(skel.edges, dtype=int)

    if bbox is None:
        # Fast vectorized sum with no clipping.
        if edges.size == 0:
            return (
                (
                    0.0,
                    {
                        "bbox_lo": None,
                        "bbox_hi": None,
                        "edges_total": 0,
                        "edges_intersected": 0,
                        "clipped": False,
                    },
                )
                if return_details
                else 0.0
            )
        seg = nodes[edges[:, 1]] - nodes[edges[:, 0]]
        L = float(np.linalg.norm(seg, axis=1).sum())
        if not return_details:
            return L
        return L, {
            "bbox_lo": None,
            "bbox_hi": None,
            "edges_total": int(edges.shape[0]),
            "edges_intersected": int(edges.shape[0]),
            "clipped": False,
        }

    lo, hi = _parse_bbox(bbox)

    total = 0.0
    n_hit = 0
    for u, v in edges:
        ell = _segment_aabb_clip_length(nodes[u], nodes[v], lo, hi)
        if ell > 0.0:
            total += ell
            n_hit += 1

    if not return_details:
        return float(total)

    return float(total), {
        "bbox_lo": lo,
        "bbox_hi": hi,
        "edges_total": int(edges.shape[0]),
        "edges_intersected": n_hit,
        "clipped": True,
    }
