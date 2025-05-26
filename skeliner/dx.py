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


# def degree_distribution(
#     skel,
#     *,
#     high_deg_percentile: float = 99.5,
# ) -> Dict[str, Union[np.ndarray, Dict[int, int]]]:
#     """Return degree histogram and *high‑degree* outliers.

#     The 99.5‑th percentile is flagged by default; override the percentile to
#     tune sensitivity.
#     """
#     g = _graph(skel)
#     deg = np.asarray(g.degree())

#     hist = np.bincount(deg)
#     thresh = np.percentile(deg, high_deg_percentile)
#     high_nodes = {int(i): int(deg[i]) for i in np.where(deg > thresh)[0]}

#     return {
#         "hist": hist,
#         "threshold": float(thresh),
#         "high_degree_nodes": high_nodes,
#     }

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
