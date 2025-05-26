"""skeliner.dx – graph‑theoretic diagnostics for a single Skeleton
"""
from typing import Any, Dict, List, Sequence, Tuple, Union

import igraph as ig
import numpy as np
from scipy.stats import spearmanr

__all__ = [
    "connectivity",
    "acyclicity",
    "degree_distribution",
    "leaf_depths",
    "path_length_monotonicity",
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


def acyclicity(skel, *, return_cycle: bool = False):
    """Check that the skeleton is a *forest* (|E| = |V| − components).

    If a cycle exists and ``return_cycle`` is *True*, a representative list of
    (u, v) edges forming the cycle is returned.
    """
    g = _graph(skel)
    n_comp = len(g.components())
    acyclic = g.ecount() == g.vcount() - n_comp
    if acyclic or not return_cycle:
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
        out["neighbors"].append({
            "id": int(nb),
            "degree": degree(skel, nb),
            "radius": float(skel.radii[radius_metric][nb]),
        })
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
                skel, int(idx), radius_metric=radius_metric)

    return {
        "hist": hist,
        "threshold": float(thresh),
        "high_degree_nodes": high_dict,
    }

def degree_eq(skel, k: int):
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
                visited_edges.add(tuple(sorted((prev, curr))) )
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


def terminal_branches_of_length(
    skel,
    k: int,
    *,
    include_endpoints: bool = True,
) -> List[List[int]]:
    """Return *terminal* branches whose node count == ``k``.

    A terminal branch is a simple path that starts at a bifurcation (or the
    soma) and ends at a **leaf** (degree == 1).  Interior vertices (if any)
    have degree == 2.

    Parameters
    ----------
    skel : Skeleton
        Input skeleton.
    k : int
        Desired length in **nodes**.  Endpoints are included when
        ``include_endpoints`` is *True*.
    include_endpoints : bool, default ``True``
        Whether endpoints contribute to *k*.

    Returns
    -------
    list[list[int]]
        Ordered node‑ID paths of every matching branch.
    """
    all_branches = branches_of_length(
        skel, k, include_endpoints=include_endpoints)

    g = _graph(skel)
    deg = np.asarray(g.degree())

    term = []
    for path in all_branches:
        # exactly one endpoint must be a leaf (degree==1)
        leaf_count = int(deg[path[0]] == 1) + int(deg[path[-1]] == 1)
        if leaf_count == 1:
            term.append(path)
    return term


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

# -----------------------------------------------------------------------------
# 4. path‑length monotonicity vs. node id order
# -----------------------------------------------------------------------------

def path_length_monotonicity(
    skel,
    *,
    return_pvalue: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Spearman ρ between node *id* and metric *path length* from soma.

    Nodes are typically created in roughly increasing distance order, so a
    high positive correlation (ρ ≫ 0) is expected.  Values ≲ 0.6 often hint
    at mis‑ordered nodes produced by disconnected components or aggressive
    bridging.

    Returns
    -------
    ρ  — or *(ρ, p‑value)* when ``return_pvalue=True``.
    """
    g = _graph(skel)
    coords = skel.nodes.view(np.ndarray)
    weights = [float(np.linalg.norm(coords[a] - coords[b])) for a, b in skel.edges]
    dists = np.asarray(g.shortest_paths(source=0, weights=weights))[0]

    ids = np.arange(len(dists))
    rho, p = spearmanr(ids, dists)
    return (float(rho), float(p)) if return_pvalue else float(rho)
