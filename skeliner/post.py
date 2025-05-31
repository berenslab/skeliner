"""skeliner.post – post-processing functions for skeletons.
"""
from typing import Iterable, Set

import igraph as ig
import numpy as np

from . import dx

__skeleton__ = [
    # editing edges
    "graft",
    "clip",
    "prune",
    # editing ntype
    "set_ntype",
]

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _norm_edge(u: int, v: int) -> tuple[int, int]:
    """Return *sorted* vertex pair as tuple."""
    if u == v:
        raise ValueError("self-loops are not allowed")
    return (u, v) if u < v else (v, u)


def _refresh_igraph(skel) -> ig.Graph:  # type: ignore[valid-type]
    """Build an igraph view from current edge list."""
    return ig.Graph(n=len(skel.nodes), edges=[tuple(map(int, e)) for e in skel.edges], directed=False)

# -----------------------------------------------------------------------------
# editing edges: graft / clip
# -----------------------------------------------------------------------------

def graft(skel, u: int, v: int, *, allow_cycle: bool = True) -> None:
    """Insert an undirected edge *(u,v)*.

    Parameters
    ----------
    allow_cycle
        When *False* the function refuses to create a cycle and raises
        ``ValueError`` if *u* and *v* are already connected via another path.
    """
    u, v = int(u), int(v)
    if u == v:
        raise ValueError("Cannot graft a self-edge (u == v)")

    new_edge = _norm_edge(u, v)
    if any((skel.edges == new_edge).all(1)):
        return  # already present – no-op

    if not allow_cycle:
        g = _refresh_igraph(skel)
        if g.are_connected(u, v):
            raise ValueError("graft would introduce a cycle; set allow_cycle=True to override")

    skel.edges = np.sort(
        np.vstack([skel.edges, np.asarray(new_edge, dtype=np.int64)]), axis=1
    )
    skel.edges = np.unique(skel.edges, axis=0)


def clip(skel, u: int, v: int, *, drop_orphans: bool = False) -> None:
    """Remove the undirected edge *(u,v)* if it exists.

    Parameters
    ----------
    drop_orphans
        After clipping, remove any node(s) that become unreachable from the
        soma (vertex 0).  This also drops their incident edges and updates all
        arrays.
    """
    u, v = _norm_edge(int(u), int(v))
    mask = ~((skel.edges[:, 0] == u) & (skel.edges[:, 1] == v))
    if mask.all():
        return  # edge not present – no-op
    skel.edges = skel.edges[mask]

    if drop_orphans:
        # Build connectivity mask from soma (0)
        g = _refresh_igraph(skel)
        order, _, _ = g.bfs(0, mode="ALL")
        reachable: Set[int] = {v for v in order if v != -1}
        if len(reachable) == len(skel.nodes):
            return  # nothing to drop

        _rebuild_keep_subset(skel, reachable)

def prune(
    skel,
    *,
    kind: str= "twigs",
    num_nodes: int | None = None,
    nodes: Iterable[int] | None = None,
) -> None:
    """Rule-based removal of sub-trees or hubs.

    Parameters
    ----------
    kind : {"twigs", "nodes"}
        * ``"twigs"``  – delete all terminal branches (twigs) whose node count
          ≤ ``max_nodes``.
        * ``"nodes"`` – delete all specified nodes along with their incident edges.
    max_nodes
        Threshold for *twigs* pruning (ignored otherwise).
    nodes:
        Iterable of node indices to prune (ignored for *twigs* pruning).
    """
    if kind == "twigs":
        if num_nodes is None:
            raise ValueError("num_nodes must be given for kind='twigs'")
        _prune_twigs(skel, num_nodes=num_nodes)
    elif kind == "nodes":
        if nodes is None:
            raise ValueError("nodes must be given for kind='nodes'")
        _prune_nodes(skel, nodes=nodes)
    else:
        raise ValueError(f"Unknown kind '{kind}'")

def _collect_twig_nodes(skel, num_nodes: int) -> Set[int]:
    """
    Vertices to drop when pruning *twigs* ≤ num_nodes.

    *Always* keeps the branching node.
    """
    nodes: Set[int] = set()

    for k in range(1, num_nodes + 1):
        for twig in dx.twigs_of_length(skel, k, include_branching_node=True):
            # twig[0] is the branching node when include_branching_node=True
            nodes.update(twig[1:])          # drop only the true twig
    return nodes


def _prune_twigs(skel, *, num_nodes: int):
    drop = _collect_twig_nodes(skel, num_nodes=num_nodes)
    if not drop:
        return  # nothing to do
    _rebuild_drop_set(skel, drop)

def _prune_nodes(
    skel,
    nodes: Iterable[int],
) -> None:
    drop = set(int(n) for n in nodes if n != 0)  # never drop soma
    if not drop:
        return

    g = skel._igraph()
    deg = np.asarray(g.degree())
    for n in list(drop):
        if deg[n] <= 2:
            continue
        neigh = [v for v in g.neighbors(n) if v not in drop]
        if len(neigh) >= 2:
            drop.remove(n)

    _rebuild_drop_set(skel, drop)

# -----------------------------------------------------------------------------
#  array rebuild utilities
# -----------------------------------------------------------------------------

def _rebuild_drop_set(skel, drop: Iterable[int]):
    """Compact skeleton arrays after dropping a set of vertices."""
    drop_set = set(map(int, drop))
    keep_mask = np.ones(len(skel.nodes), dtype=bool)
    for i in drop_set:
        keep_mask[i] = False
    if keep_mask[0] is False:
        raise RuntimeError("Attempted to drop the soma (vertex 0)")

    remap = -np.ones(len(keep_mask), dtype=np.int64)
    remap[keep_mask] = np.arange(keep_mask.sum(), dtype=np.int64)

    skel.nodes = skel.nodes[keep_mask]
    skel.node2verts = [skel.node2verts[i] for i in np.where(keep_mask)[0]] if skel.node2verts is not None else None
    skel.radii = {k: v[keep_mask] for k, v in skel.radii.items()}

    # update vert2node mapping
    if skel.vert2node is not None:
        skel.vert2node = {int(v): int(remap[n]) for v, n in skel.vert2node.items() if keep_mask[n]}

    # rebuild edges
    new_edges = []
    for a, b in skel.edges:
        if keep_mask[a] and keep_mask[b]:
            new_edges.append((remap[a], remap[b]))
    skel.edges = np.sort(np.asarray(new_edges, dtype=np.int64), axis=1)


def _rebuild_keep_subset(skel, keep_set: Set[int]):
    """Compact arrays keeping only *keep_set* vertices."""
    keep_mask = np.zeros(len(skel.nodes), dtype=bool)
    keep_mask[list(keep_set)] = True
    _rebuild_drop_set(skel, np.where(~keep_mask)[0])

# -----------------------------------------------------------------------------
#  ntype editing
# -----------------------------------------------------------------------------

def set_ntype(
    skel,
    *,
    root: int | Iterable[int] | None = None,
    node_ids: Iterable[int] | None = None,
    code: int = 3,
    subtree: bool = True,
    include_root: bool = True,
) -> None:
    """
    Label nodes with SWC *code*.

    Exactly one of ``root`` or ``node_ids`` must be provided.

    Parameters
    ----------
    root
        Base node(s) whose neurite(s) will be coloured.  Requires
        ``node_ids is None``.  If *subtree* is True (default) every base
        node is expanded with :pyfunc:`dx.extract_neurites`.
    node_ids
        Explicit collection of node indices to label.  Requires
        ``root is None``; no expansion is performed.
    code
        SWC integer code to assign (2 = axon, 3 = dendrite, …).
    subtree, include_root
        Control how the neurite expansion behaves (ignored when
        ``node_ids`` is given).
    """
   # ----------------------------------------------------------- #
    # argument sanity                                             #
    # ----------------------------------------------------------- #
    if (root is None) == (node_ids is None):
        raise ValueError("supply exactly one of 'root' or 'node_ids'")

    # ----------------------------------------------------------- #
    # gather the target set                                       #
    # ----------------------------------------------------------- #
    if node_ids is not None:
        target = set(map(int, node_ids))
    else:
        bases = np.atleast_1d(root).astype(int)
        target: set[int] = set()
        if subtree:
            for nid in bases:
                target.update(
                    dx.extract_neurites(skel, int(nid), include_root=include_root)
                )
        else:
            target.update(bases)

    target.discard(0)          # never overwrite soma
    if not target:
        return

    # ----------------------------------------------------------- #
    # fast vectorised assignment                                  #
    # ----------------------------------------------------------- #
    skel.ntype[np.fromiter(target, dtype=int)] = int(code)