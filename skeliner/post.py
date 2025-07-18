"""skeliner.post – post-processing functions for skeletons.
"""
from typing import Iterable, Set, cast

import igraph as ig
import numpy as np
from numpy.typing import ArrayLike

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
    skel.node2verts = [skel.node2verts[i] for i in np.where(keep_mask)[0]] if skel.node2verts is not None and len(skel.node2verts) > 0 else None
    skel.radii = {k: v[keep_mask] for k, v in skel.radii.items()}

    # update vert2node mapping
    if skel.vert2node is not None and len(skel.vert2node) > 0:
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
        bases_arr = np.atleast_1d(
            cast(ArrayLike, root)
        ).astype(int)

        bases: set[int] = set(bases_arr)
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

    skel.ntype[np.fromiter(target, dtype=int)] = int(code)

# -----------------------------------------------------------------------------
# Re-detect Soma
# -----------------------------------------------------------------------------

def _find_soma(
    nodes: np.ndarray,
    radii: np.ndarray,
    *,
    pct_large: float = 99.9,
    dist_factor: float = 3.0,
    min_keep: int = 2,
):
    """
    Geometry-only soma heuristic used by both the core pipeline and
    :pyfunc:`detect_soma`.

    Returns
    -------
    soma        –  *Soma* instance (sphere model – no surface verts)
    soma_idx    –  1-D int64 array of node IDs judged to belong to the soma
    has_soma    –  True when ≥ `min_keep` nodes qualified
    """
    from .core import Soma

    if nodes.shape[0] == 0:
        raise ValueError("empty skeleton")

    # 1. radius threshold – pick the fattest ~0.1 % of nodes
    large_thresh = np.percentile(radii, pct_large)
    cand_idx = np.where(radii >= large_thresh)[0]

    if cand_idx.size == 0:
        return Soma.from_sphere(nodes[0], radii[0], verts=None), cand_idx, False

    # 2. anchor = single largest node
    idx_max = int(np.argmax(radii))
    R_max   = float(radii[idx_max])

    # 3. keep candidates that cluster around the anchor
    d = np.linalg.norm(nodes[cand_idx] - nodes[idx_max], axis=1)
    soma_idx = cand_idx[d <= dist_factor * R_max]
    has_soma = soma_idx.size >= min_keep

    soma_est = Soma.from_sphere(
        center=nodes[soma_idx].mean(0) if has_soma else nodes[idx_max],
        radius=R_max,
        verts=None,
    )
    return soma_est, soma_idx, has_soma


def detect_soma(          
    skel,
    *,
    radius_key: str = "median",
    soma_radius_percentile_threshold: float = 99.9,
    soma_radius_distance_factor: float = 4.0,
    soma_min_nodes: int = 3,
    verbose: bool = True,
):
    """
    Post-hoc soma detection **on an existing Skeleton**.

    Examples
    --------
    >>> import skeliner as sk
    >>> s = sk.core.skeletonize(mesh, detect_soma=False)  # soma missed
    >>> s2 = sk.post.detect_soma(s, verbose=True)         # re-root to soma

    Parameters
    ----------
    radius_key
        Which radius estimator column to use for node “fatness”.
    pct_large, dist_factor, min_keep
        Hyper-parameters forwarded to the internal :pyfunc:`_find_soma`.
    merge
        When *True* (default) every node classified as soma is **collapsed**
        into a single centroid that becomes vertex 0.  When *False* only the
        fattest soma node is promoted to root and the others stay, simply
        re-connected to it.
    verbose
        Print a concise log of what happened.

    Returns
    -------
    Skeleton
        *Either* the original instance (no change was necessary) *or* a new
        skeleton whose node 0 is the freshly detected soma centroid.
    """
    from .core import Skeleton, Soma, _build_mst
    if radius_key not in skel.radii:
        raise KeyError(
            f"radius_key '{radius_key}' not found in skel.radii "
            f"(available keys: {tuple(skel.radii)})"
        )
    if len(skel.nodes) <= 1:
        return skel                 # trivial graph → nothing to do


    has_node2verts = skel.node2verts is not None and len(skel.node2verts) > 0
    has_vert2node = skel.vert2node is not None and len(skel.vert2node) > 0
    # ------------------------------------------------------------------ 
    # A. re-detect the soma cluster
    # ------------------------------------------------------------------
    soma_est, soma_idx, has_soma = _find_soma(
        skel.nodes, skel.radii[radius_key],
        pct_large=soma_radius_percentile_threshold,
        dist_factor=soma_radius_distance_factor,
        min_keep=soma_min_nodes,
    )

    # Already fine?
    if (not has_soma) or set(map(int, soma_idx)) == {0}:
        if verbose:
            print("[skeliner] detect_soma – existing soma kept unchanged.")
        return skel

    # Which node will be the *new* root?
    new_root_old = int(
        soma_idx[np.argmax(skel.radii[radius_key][soma_idx])]
    )
    drop_nodes = {int(i) for i in soma_idx if i != new_root_old}

    # ------------------------------------------------------------------ 
    # B. clone arrays so we do not mutate the caller’s object
    # ------------------------------------------------------------------
    nodes       = skel.nodes.copy()
    radii       = {k: v.copy() for k, v in skel.radii.items()}
    edges       = skel.edges.copy()
    node2verts  = [vs.copy() for vs in skel.node2verts] if has_node2verts else None
    vert2node   = dict(skel.vert2node)         if has_vert2node else None
    ntype       = skel.ntype.copy()            if skel.ntype is not None else None

    # ------------------------------------------------------------------ 
    # C. **collapse** of multiple soma nodes
    # ------------------------------------------------------------------
    if drop_nodes:
        #
        # 1. move geometric centre to the keeper (new_root_old)
        #
        nodes[new_root_old] = nodes[list(drop_nodes) + [new_root_old]].mean(0)

        #
        # 2. merge vertex memberships + radii (tolerate missing node2verts)
        #
        for idx in drop_nodes:
            if has_node2verts:
                # auto-extend the mapping list if it is shorter than needed
                if idx >= len(node2verts):
                    node2verts.extend(
                        [np.empty(0, dtype=np.int64) for _ in range(idx + 1 - len(node2verts))]
                    )
                if new_root_old >= len(node2verts):
                    node2verts.extend(
                        [np.empty(0, dtype=np.int64) for _ in range(new_root_old + 1 - len(node2verts))]
                    )
                node2verts[new_root_old] = np.concatenate(
                    (node2verts[new_root_old], node2verts[idx])
                )

            for k in radii:
                radii[k][new_root_old] = max(radii[k][new_root_old], radii[k][idx])

        #
        # 3. RE-WIRE: connect every neighbour of a soon-to-be-dropped node
        #    directly to the keeper so the skeleton stays in one piece.
        #
        drop_set = set(drop_nodes)
        extra_edges = []
        for a, b in edges:
            if a in drop_set and b not in drop_set:
                extra_edges.append((new_root_old, b))
            elif b in drop_set and a not in drop_set:
                extra_edges.append((new_root_old, a))

        if extra_edges:
            edges = np.vstack([edges, np.asarray(extra_edges, dtype=np.int64)])
            # row-wise sort then deduplicate
            edges = np.unique(np.sort(edges, axis=1), axis=0)

    # ------------------------------------------------------------------ 
    # D. build keep-mask & remap after the (optional) merge
    # ------------------------------------------------------------------
    keep_mask = np.ones(len(nodes), bool)
    keep_mask[list(drop_nodes)] = False
    remap = -np.ones(len(nodes), np.int64)
    remap[np.where(keep_mask)[0]] = np.arange(keep_mask.sum(), dtype=np.int64)

    nodes = nodes[keep_mask]
    radii = {k: v[keep_mask] for k, v in radii.items()}
    if ntype is not None:
        ntype = ntype[keep_mask]
    if has_node2verts:
        node2verts = [node2verts[i] for i in np.where(keep_mask)[0]]
    if has_vert2node:
        vert2node = {int(v): remap[int(n)]
                     for v, n in vert2node.items() if keep_mask[n]}

    # edges – remap & de-duplicate
    edges = np.asarray(
        [(remap[a], remap[b]) for a, b in edges if keep_mask[a] and keep_mask[b]],
        dtype=np.int64,
    )
    if edges.size:
        edges = np.unique(np.sort(edges, axis=1), axis=0)

    new_root = remap[new_root_old]

    # ------------------------------------------------------------------ 
    # E. enforce: soma → vertex 0
    # ------------------------------------------------------------------
    if new_root != 0:
        swap = new_root

        nodes[[0, swap]] = nodes[[swap, 0]]
        for k in radii:
            radii[k][[0, swap]] = radii[k][[swap, 0]]
        if ntype is not None:
            ntype[[0, swap]] = ntype[[swap, 0]]
        if has_node2verts:
            node2verts[0], node2verts[swap] = node2verts[swap], node2verts[0]
        if has_vert2node:
            for v, n in list(vert2node.items()):
                if n == 0:
                    vert2node[v] = swap
                elif n == swap:
                    vert2node[v] = 0

        a0, a1 = edges == 0, edges == swap
        edges[a0] = swap
        edges[a1] = 0
        edges = np.unique(np.sort(edges, axis=1), axis=0)

    # ------------------------------------------------------------------ 
    # F. rebuild the Soma object (sphere model – no mesh available)
    # ------------------------------------------------------------------
    r0 = float(radii[radius_key][0])
    soma_new = Soma.from_sphere(nodes[0], r0,
                                verts=node2verts[0] if node2verts is not None and len(node2verts) > 0 else None)

    if ntype is not None:
        ntype[0] = 1                       # SWC code for soma

    if verbose:
        centre_txt = ", ".join(f"{c:7.1f}" for c in nodes[0])
        merged = len(drop_nodes)
        what = f"merged {merged} node{'s' if merged != 1 else ''}"
        print(f"[skeliner] detect_soma – {what} → soma @ [{centre_txt}], r ≈ {r0:.1f}")

    # ------------------------------------------------------------------ 
    # G. return the **new** skeleton object
    # ------------------------------------------------------------------
    new_skel = Skeleton(
        soma=soma_new,
        nodes=nodes,
        radii=radii,
        edges=_build_mst(nodes, edges),
        ntype=ntype,
        node2verts=node2verts,
        vert2node=vert2node,
        meta={**skel.meta},       # shallow copies are fine
        extra={**skel.extra},
    )

    new_skel.prune(num_nodes=1)  # remove any remaining twigs
    return new_skel

