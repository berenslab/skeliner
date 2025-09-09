"""skeliner.post – post-processing functions for skeletons."""

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
    "downsample",
    # editing ntype
    "set_ntype",
    # reroot / redetect soma
    "reroot",
    "detect_soma",
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
    return ig.Graph(
        n=len(skel.nodes),
        edges=[tuple(map(int, e)) for e in skel.edges],
        directed=False,
    )


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
            raise ValueError(
                "graft would introduce a cycle; set allow_cycle=True to override"
            )

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
    kind: str = "twigs",
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
            nodes.update(twig[1:])  # drop only the true twig
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
    skel.node2verts = (
        [skel.node2verts[i] for i in np.where(keep_mask)[0]]
        if skel.node2verts is not None and len(skel.node2verts) > 0
        else None
    )
    skel.radii = {k: v[keep_mask] for k, v in skel.radii.items()}

    # update vert2node mapping
    if skel.vert2node is not None and len(skel.vert2node) > 0:
        skel.vert2node = {
            int(v): int(remap[n]) for v, n in skel.vert2node.items() if keep_mask[n]
        }

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
        bases_arr = np.atleast_1d(cast(ArrayLike, root)).astype(int)

        bases: set[int] = set(bases_arr)
        target: set[int] = set()
        if subtree:
            for nid in bases:
                target.update(
                    dx.extract_neurites(skel, int(nid), include_root=include_root)
                )
        else:
            target.update(bases)

    target.discard(0)  # never overwrite soma
    if not target:
        return

    skel.ntype[np.fromiter(target, dtype=int)] = int(code)


# -----------------------------------------------------------------------------
# Reroot skeleton (re-assign a new soma node)
# -----------------------------------------------------------------------------


def _axis_index(axis: str) -> int:
    try:
        return {"x": 0, "y": 1, "z": 2}[axis.lower()]
    except KeyError:
        raise ValueError("axis must be one of {'x','y','z'}")


def _degrees_from_edges(n: int, edges: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.int64)
    if edges.size:
        for a, b in edges:
            deg[int(a)] += 1
            deg[int(b)] += 1
    return deg


def _extreme_node(
    skel,
    *,
    axis: str = "z",
    mode: str = "min",  # {"min","max"}
    prefer_leaves: bool = True,
) -> int:
    """
    Pick a node index at an *extreme* along `axis`, based on skeleton coords only.

    If `prefer_leaves=True`, restrict to degree-1 nodes when any exist; otherwise
    search all nodes. Returns an index in [0..N-1].
    """
    ax = _axis_index(axis)
    xs = np.asarray(skel.nodes, dtype=np.float64)[:, ax]
    n = xs.shape[0]
    if n == 0:
        raise ValueError("empty skeleton")

    deg = _degrees_from_edges(n, np.asarray(skel.edges, dtype=np.int64))
    cand = np.where(deg == 1)[0] if prefer_leaves and np.any(deg == 1) else np.arange(n)
    if cand.size == 0:
        cand = np.arange(n)

    vals = xs[cand]
    idx = int(cand[np.argmin(vals) if mode == "min" else np.argmax(vals)])
    return idx


def reroot(
    skel,
    node_id: int | None = None,
    *,
    axis: str = "z",
    mode: str = "min",
    prefer_leaves: bool = True,
    radius_key: str = "median",
    set_soma_ntype: bool = True,
    rebuild_mst: bool = False,
    verbose: bool = True,
):
    """
    Re-root so that node 0 becomes `node_id` (or an axis-extreme among leaves).

    Pure reindexing: swaps indices 0 ↔ target and remaps edges and mappings.
    Geometry and radii are unchanged. Ideal prep for `detect_soma()`.
    """
    import numpy as np

    from .core import Skeleton, Soma, _build_mst

    N = int(len(skel.nodes))
    if N <= 1:
        return skel

    tgt = (
        int(node_id)
        if node_id is not None
        else int(_extreme_node(skel, axis=axis, mode=mode, prefer_leaves=prefer_leaves))
    )
    if tgt < 0 or tgt >= N:
        raise ValueError(f"reroot: node_id {tgt} out of bounds [0,{N - 1}]")
    if tgt == 0:
        if verbose:
            print("[skeliner] reroot – already rooted at 0.")
        return skel

    # Clone arrays
    nodes = skel.nodes.copy()
    radii = {k: v.copy() for k, v in skel.radii.items()}
    edges = skel.edges.copy()
    ntype = skel.ntype.copy() if skel.ntype is not None else None

    has_node2verts = skel.node2verts is not None and len(skel.node2verts) > 0
    has_vert2node = skel.vert2node is not None and len(skel.vert2node) > 0
    node2verts = [vs.copy() for vs in skel.node2verts] if has_node2verts else None
    vert2node = dict(skel.vert2node) if has_vert2node else None

    # Swap 0 ↔ tgt
    swap = tgt
    nodes[[0, swap]] = nodes[[swap, 0]]
    for k in radii:
        radii[k][[0, swap]] = radii[k][[swap, 0]]
    if ntype is not None:
        ntype[[0, swap]] = ntype[[swap, 0]]
    if node2verts is not None:
        node2verts[0], node2verts[swap] = node2verts[swap], node2verts[0]
    if vert2node is not None:
        for v, n in list(vert2node.items()):
            if n == 0:
                vert2node[v] = swap
            elif n == swap:
                vert2node[v] = 0

    # Remap edges with a permutation
    perm = np.arange(N, dtype=np.int64)
    perm[[0, swap]] = perm[[swap, 0]]
    edges = perm[edges]
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)

    if rebuild_mst:
        edges = _build_mst(nodes, edges)

    if radius_key not in radii:
        raise KeyError(
            f"radius_key '{radius_key}' not found in skel.radii "
            f"(available keys: {tuple(radii)})"
        )
    r0 = float(radii[radius_key][0])
    new_soma = Soma.from_sphere(
        center=nodes[0],
        radius=r0,
        verts=node2verts[0]
        if node2verts is not None and node2verts[0].size > 0
        else None,
    )

    if set_soma_ntype and ntype is not None:
        ntype[0] = 1

    new_skel = Skeleton(
        soma=new_soma,
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
        node2verts=node2verts,
        vert2node=vert2node,
        meta={**skel.meta},
        extra={**skel.extra},
    )

    if verbose:
        src = (
            f"node_id={node_id}"
            if node_id is not None
            else f"extreme({axis.lower()},{mode}, prefer_leaves={prefer_leaves})"
        )
        print(f"[skeliner] reroot – 0 ↔ {swap} ({src}); rebuild_mst={rebuild_mst}")

    return new_skel


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
    R_max = float(radii[idx_max])

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
        return skel  # trivial graph → nothing to do

    has_node2verts = skel.node2verts is not None and len(skel.node2verts) > 0
    has_vert2node = skel.vert2node is not None and len(skel.vert2node) > 0
    # ------------------------------------------------------------------
    # A. re-detect the soma cluster
    # ------------------------------------------------------------------
    soma_est, soma_idx, has_soma = _find_soma(
        skel.nodes,
        skel.radii[radius_key],
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
    new_root_old = int(soma_idx[np.argmax(skel.radii[radius_key][soma_idx])])
    drop_nodes = {int(i) for i in soma_idx if i != new_root_old}

    # ------------------------------------------------------------------
    # B. clone arrays so we do not mutate the caller’s object
    # ------------------------------------------------------------------
    nodes = skel.nodes.copy()
    radii = {k: v.copy() for k, v in skel.radii.items()}
    edges = skel.edges.copy()
    node2verts = [vs.copy() for vs in skel.node2verts] if has_node2verts else None
    vert2node = dict(skel.vert2node) if has_vert2node else None
    ntype = skel.ntype.copy() if skel.ntype is not None else None

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
                        [
                            np.empty(0, dtype=np.int64)
                            for _ in range(idx + 1 - len(node2verts))
                        ]
                    )
                if new_root_old >= len(node2verts):
                    node2verts.extend(
                        [
                            np.empty(0, dtype=np.int64)
                            for _ in range(new_root_old + 1 - len(node2verts))
                        ]
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
        vert2node = {
            int(v): remap[int(n)] for v, n in vert2node.items() if keep_mask[n]
        }

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
    soma_new = Soma.from_sphere(
        nodes[0],
        r0,
        verts=node2verts[0] if node2verts is not None and len(node2verts) > 0 else None,
    )

    if ntype is not None:
        ntype[0] = 1  # SWC code for soma

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
        meta={**skel.meta},  # shallow copies are fine
        extra={**skel.extra},
    )

    new_skel.prune(num_nodes=1)  # remove any remaining twigs
    return new_skel


# -----------------------------------------------------------------------------
# Radii-aware subsampling
# -----------------------------------------------------------------------------


def _mode_int(vals: np.ndarray, default: int = 3) -> int:
    """Fast integer mode with a sane default when empty."""
    vals = np.asarray(vals, dtype=np.int64)
    if vals.size == 0:
        return int(default)
    mx = int(vals.max(initial=0))
    if mx < 0:
        return int(default)
    return int(np.bincount(np.clip(vals, 0, mx)).argmax())


def _adjacency_from_edges(n: int, edges: np.ndarray) -> list[list[int]]:
    """Build simple adjacency lists."""
    adj = [set() for _ in range(n)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return [list(s) for s in adj]


def _len_weighted_centroid(xs: np.ndarray) -> np.ndarray:
    """
    Length-weighted centroid of a polyline defined by node coordinates.
    Uses segment midpoints weighted by segment length.
    For a single point, returns that point.
    """
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) <= 1:
        return xs.reshape(-1, 3)[0]
    seg = xs[1:] - xs[:-1]
    L = np.linalg.norm(seg, axis=1)
    if not np.isfinite(L).all() or np.all(L == 0):
        return xs.mean(axis=0)
    mids = 0.5 * (xs[1:] + xs[:-1])
    return (mids * L[:, None]).sum(axis=0) / L.sum()


def _partition_by_radius(
    ids: list[int], r: np.ndarray, *, rtol: float, atol: float
) -> list[list[int]]:
    """
    Greedy 1D segmentation along a path based on a running radius reference.
    Starts a new group when the next radius deviates beyond tolerance.
    """
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
    skel,
    *,
    radius_key: str = "median",
    rtol: float = 0.05,
    atol: float = 0.0,
    aggregate: str = "area",  # {"median","mean", "area"} for radii aggregation
    merge_endpoints: bool = True,
    slide_branchpoints: bool = True,
    max_anchor_shift: float | None = None,  # (units of coords)
    verbose: bool = True,
):
    """
    Radii-aware downsampling that preserves topology.

    By default: only degree-2 runs are merged (anchors kept).
    Optional: absorb leaf endpoints and/or slide branchpoints into adjacent
    runs when |Δr| ≤ atol + rtol * max(r_anchor, r_group). Merging node 0 is
    never allowed.
    """
    from .core import Skeleton, _build_mst

    if radius_key not in skel.radii:
        raise KeyError(
            f"radius_key '{radius_key}' not found in skel.radii "
            f"(available keys: {tuple(skel.radii)})"
        )
    N = int(len(skel.nodes))
    if N <= 1:
        return skel

    nodes = skel.nodes
    radiiD = skel.radii
    r_dec = radiiD[radius_key]
    ntype0 = skel.ntype if skel.ntype is not None else np.full(N, 3, dtype=np.int8)

    has_node2verts = skel.node2verts is not None and len(skel.node2verts) > 0
    has_vert2node = skel.vert2node is not None and len(skel.vert2node) > 0

    node2verts0: list[np.ndarray] | None = None
    vert2node0: dict[int, int] | None = None
    if has_node2verts:
        node2verts0 = list(skel.node2verts)
        if len(node2verts0) < N:
            node2verts0 += [
                np.empty(0, dtype=np.int64) for _ in range(N - len(node2verts0))
            ]
    if has_vert2node:
        vert2node0 = dict(skel.vert2node)

    g = skel._igraph()
    deg = np.asarray(g.degree(), dtype=np.int64)
    anchors: set[int] = {i for i, d in enumerate(deg) if d != 2}
    anchors.add(0)

    adj = _adjacency_from_edges(N, skel.edges)

    new_nodes: list[np.ndarray] = []
    new_radii: dict[str, list[float]] = {k: [] for k in radiiD.keys()}
    new_ntype: list[int] = []
    new_node2verts: list[np.ndarray] | None = [] if has_node2verts else None
    new_edges: list[tuple[int, int]] = []
    old2new: dict[int, int] = {}

    def _add_anchor(old_id: int) -> int:
        oid = int(old_id)
        if oid in old2new:
            return old2new[oid]
        nid = len(new_nodes)
        new_nodes.append(nodes[oid].astype(np.float64))
        for k in new_radii:
            new_radii[k].append(float(radiiD[k][oid]))
        new_ntype.append(int(ntype0[oid]))
        if new_node2verts is not None:
            arr = (
                node2verts0[oid]
                if node2verts0 is not None
                else np.empty(0, dtype=np.int64)
            )
            new_node2verts.append(np.asarray(arr, dtype=np.int64))
        old2new[oid] = nid
        return nid

    def _compute_aggregate(vals, aggregate):
        if aggregate == "median":
            val = float(np.median(vals))
        elif aggregate == "mean":
            val = float(np.mean(vals))
        elif aggregate == "area":  # new: preserve mean cross‑section
            val = float(np.sqrt(np.mean(vals * vals)))
        else:
            raise ValueError("aggregate must be 'median', 'mean', or 'area'")

        return val

    def _add_group(group_ids: list[int]) -> int:
        gids = list(map(int, group_ids))
        nid = len(new_nodes)
        pos = _len_weighted_centroid(nodes[gids])
        new_nodes.append(pos.astype(np.float64))

        for k in new_radii:
            vals = radiiD[k][gids]
            val = _compute_aggregate(vals, aggregate)
            new_radii[k].append(val)

        new_ntype.append(_mode_int(ntype0[gids], default=3))

        if new_node2verts is not None:
            if node2verts0 is None:
                new_node2verts.append(np.empty(0, dtype=np.int64))
            else:
                parts = [
                    np.asarray(node2verts0[j], dtype=np.int64)
                    for j in gids
                    if j < len(node2verts0)
                ]
                merged = (
                    np.unique(np.concatenate(parts))
                    if parts
                    else np.empty(0, dtype=np.int64)
                )
                new_node2verts.append(merged)

        for j in gids:
            old2new[j] = nid
        return nid

    def _within_tol(ra: float, rb: float) -> bool:
        tol = float(atol) + float(rtol) * max(abs(ra), abs(rb))
        return abs(ra - rb) <= tol

    visited: set[tuple[int, int]] = set()

    for a in sorted(anchors):
        for b in adj[a]:
            e0 = _norm_edge(a, b)
            if e0 in visited:
                continue

            # Walk a → ... → z along the corridor
            path = [a]
            prev, cur = a, b
            visited.add(e0)
            while deg[cur] == 2:
                path.append(cur)
                nxts = [x for x in adj[cur] if x != prev]
                if len(nxts) != 1:
                    break
                nxt = nxts[0]
                visited.add(_norm_edge(cur, nxt))
                prev, cur = cur, nxt
            if cur != path[-1]:
                path.append(cur)

            a_id, z_id = path[0], path[-1]
            internal = path[1:-1]

            left = _add_anchor(a_id)
            right = _add_anchor(z_id)

            groups = _partition_by_radius(internal, r_dec, rtol=rtol, atol=atol)

            # --- Optional: absorb leftmost group into left anchor -----------
            if (
                groups
                and a_id != 0
                and (
                    (merge_endpoints and deg[a_id] == 1)
                    or (slide_branchpoints and deg[a_id] >= 3)
                )
            ):
                g0 = groups[0]
                ra = float(r_dec[a_id])
                rg = _compute_aggregate(r_dec[g0], aggregate)
                if _within_tol(ra, rg):
                    new_pos = _len_weighted_centroid(nodes[[a_id] + g0])
                    if (
                        max_anchor_shift is None
                        or np.linalg.norm(new_pos - new_nodes[left]) <= max_anchor_shift
                    ):
                        # mutate anchor accumulators
                        new_nodes[left] = new_pos
                        for k in new_radii:
                            vals = np.concatenate(([radiiD[k][a_id]], radiiD[k][g0]))
                            new_radii[k][left] = _compute_aggregate(vals, aggregate)

                        if new_node2verts is not None and node2verts0 is not None:
                            parts = [node2verts0[a_id]] + [node2verts0[j] for j in g0]
                            merged = (
                                np.unique(
                                    np.concatenate(
                                        [
                                            p
                                            for p in parts
                                            if p is not None and p.size > 0
                                        ]
                                    )
                                )
                                if any((p is not None and p.size > 0) for p in parts)
                                else np.empty(0, dtype=np.int64)
                            )
                            new_node2verts[left] = merged
                        for j in g0:
                            old2new[j] = left
                        groups = groups[1:]

            # --- Optional: absorb rightmost group into right anchor ----------
            if (
                groups
                and z_id != 0
                and (
                    (merge_endpoints and deg[z_id] == 1)
                    or (slide_branchpoints and deg[z_id] >= 3)
                )
            ):
                gL = groups[-1]
                rz = float(r_dec[z_id])
                rg = _compute_aggregate(r_dec[gL], aggregate)
                if _within_tol(rz, rg):
                    new_pos = _len_weighted_centroid(nodes[gL + [z_id]])
                    if (
                        max_anchor_shift is None
                        or np.linalg.norm(new_pos - new_nodes[right])
                        <= max_anchor_shift
                    ):
                        new_nodes[right] = new_pos
                        for k in new_radii:
                            vals = np.concatenate((radiiD[k][gL], [radiiD[k][z_id]]))
                            new_radii[k][right] = _compute_aggregate(vals, aggregate)
                        if new_node2verts is not None and node2verts0 is not None:
                            parts = [node2verts0[j] for j in gL] + [node2verts0[z_id]]
                            merged = (
                                np.unique(
                                    np.concatenate(
                                        [
                                            p
                                            for p in parts
                                            if p is not None and p.size > 0
                                        ]
                                    )
                                )
                                if any((p is not None and p.size > 0) for p in parts)
                                else np.empty(0, dtype=np.int64)
                            )
                            new_node2verts[right] = merged
                        for j in gL:
                            old2new[j] = right
                        groups = groups[:-1]

            # Wire: left → groups → right
            L = left
            for grp in groups:
                g_id = _add_group(grp)
                new_edges.append(_norm_edge(L, g_id))
                L = g_id
            new_edges.append(_norm_edge(L, right))

    # Handle isolated anchors (deg == 0)
    for a in sorted(anchors):
        if deg[a] == 0:
            _add_anchor(a)

    nodes_new = np.asarray(new_nodes, dtype=np.float64)
    radii_new = {k: np.asarray(v, dtype=np.float64) for k, v in new_radii.items()}
    ntype_new = np.asarray(new_ntype, dtype=np.int8)

    edges_arr = np.asarray(new_edges, dtype=np.int64)
    edges_arr = (
        np.unique(np.sort(edges_arr, axis=1), axis=0) if edges_arr.size else edges_arr
    )

    # vert2node
    if has_vert2node and vert2node0 is not None:
        vert2node_new = {
            int(v): int(old2new.get(int(n), old2new.get(0, 0)))
            for v, n in vert2node0.items()
            if int(n) in old2new
        }
    elif new_node2verts is not None:
        vert2node_new = {
            int(v): int(i)
            for i, vs in enumerate(new_node2verts)
            for v in np.asarray(vs, dtype=np.int64)
        }
    else:
        vert2node_new = None

    # Keep soma at index 0
    new_root = int(old2new.get(0, 0))
    if new_root != 0 and len(nodes_new):
        swap = new_root
        nodes_new[[0, swap]] = nodes_new[[swap, 0]]
        for k in radii_new:
            radii_new[k][[0, swap]] = radii_new[k][[swap, 0]]
        ntype_new[[0, swap]] = ntype_new[[swap, 0]]
        if new_node2verts is not None:
            new_node2verts[0], new_node2verts[swap] = (
                new_node2verts[swap],
                new_node2verts[0],
            )
        if vert2node_new is not None:
            for v, n in list(vert2node_new.items()):
                if n == 0:
                    vert2node_new[v] = swap
                elif n == swap:
                    vert2node_new[v] = 0
        # a0, a1 = edges_arr == 0, edges_arr == swap
        # edges_arr[a0] = swap
        # edges_arr[a1] = 0
        # edges_arr = np.unique(np.sort(edges_arr, axis=1), axis=0)
        perm = np.arange(len(nodes_new), dtype=np.int64)
        perm[[0, swap]] = perm[[swap, 0]]
        edges_arr = perm[edges_arr]  # apply to both columns at once

        edges_arr = np.sort(edges_arr, axis=1)
        edges_arr = edges_arr[edges_arr[:, 0] != edges_arr[:, 1]]  # drop self-loops
        edges_arr = np.unique(edges_arr, axis=0)

    g_check = ig.Graph(
        n=len(nodes_new), edges=[tuple(map(int, e)) for e in edges_arr], directed=False
    )
    if g_check.ecount() != g_check.vcount() - len(g_check.components()):
        # make it a spanning forest over your candidate edges (acyclic by construction)
        edges_arr = _build_mst(nodes_new, edges_arr)  # same helper you already import

    new_skel = Skeleton(
        soma=skel.soma,
        nodes=nodes_new,
        radii=radii_new,
        edges=_build_mst(nodes_new, edges_arr),
        ntype=ntype_new,
        node2verts=new_node2verts,
        vert2node=vert2node_new,
        meta={**skel.meta},
        extra={**skel.extra},
    )

    if verbose:
        print(
            f"[skeliner] downsample – nodes: {N} → {len(nodes_new)}; "
            f"rtol={rtol:g}, atol={atol:g}, key='{radius_key}', agg='{aggregate}', "
            f"merge_endpoints={merge_endpoints}, slide_branchpoints={slide_branchpoints}"
        )

    return new_skel
