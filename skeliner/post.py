"""skeliner.post – post-processing functions for skeletons."""

import time
from contextlib import contextmanager
from typing import Iterable, Set, cast

import igraph as ig
import numpy as np
from numpy.typing import ArrayLike

from . import dx
from ._core import (
    _bridge_gaps,
    _build_mst,
    _detect_soma,
    _merge_near_soma_nodes,
    _prune_neurites,
)
from ._state import (
    SkeletonState,
    compact_state,
    rebuild_vert2node,
    remap_edges,
    swap_nodes,
)
from .dataclass import Skeleton, Soma

__skeleton__ = [
    # editing edges
    "graft",
    "clip",
    "prune",
    "bridge_gaps",
    "merge_near_soma_nodes",
    "prune_neurites",
    "rebuild_mst",
    "downsample",
    # editing ntype
    "set_ntype",
    # reroot / redetect soma
    "reroot",
    "detect_soma",
]


@contextmanager
def _post_stage(label: str, *, verbose: bool):
    """Uniform verbose/timing helper matching ``skeletonize`` output."""
    if not verbose:
        yield lambda *_: None
        return

    PAD = 39  # keeps the ASCII arrow alignment consistent
    prefix = "[skeliner.post]"
    print(f"{prefix} {label:<{PAD}} …", end="", flush=True)
    t0 = time.perf_counter()
    _msgs: list[str] = []

    def log(msg: str) -> None:
        _msgs.append(str(msg))

    try:
        yield log
    finally:
        dt = time.perf_counter() - t0
        print(f" {dt:.2f} s")
        for msg in _msgs:
            print(f"      └─ {msg}")


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


def _remap_ntype(
    ntype: np.ndarray | None, old2new: np.ndarray, new_len: int
) -> np.ndarray | None:
    if ntype is None:
        return None
    mapped = np.full(new_len, 3, dtype=ntype.dtype)
    for old_idx, new_idx in enumerate(old2new):
        if new_idx >= 0:
            mapped[new_idx] = ntype[old_idx]
    if new_len:
        mapped[0] = 1
    return mapped


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
#  gap bridging
# -----------------------------------------------------------------------------


def bridge_gaps(
    skel,
    *,
    bridge_max_factor: float | None = None,
    bridge_recalc_after: int | None = None,
    rebuild_mst: bool = True,
    verbose: bool = True,
) -> None:
    """
    Connect disconnected skeleton components with synthetic edges, mirroring
    the automatic gap bridging performed during :func:`skeletonize`.

    Parameters
    ----------
    bridge_max_factor
        Optional ceiling for acceptable bridge length expressed as a multiple of
        the mean edge length. ``None`` (default) uses the adaptive heuristic from
        the core pipeline.
    bridge_recalc_after
        How often to recompute component-to-island distances. ``None`` (default)
        triggers the adaptive heuristic.
    rebuild_mst
        When *True* (default) rebuild the global minimum-spanning tree after
        adding the bridges to remove any short cycles.
    verbose
        When *True* print timing information and per-stage summaries.
    """
    edges = skel.edges
    with _post_stage("bridge skeleton gaps", verbose=verbose) as log:
        prev_edges = int(edges.shape[0])
        edges = _bridge_gaps(
            skel.nodes,
            edges,
            bridge_max_factor=bridge_max_factor,
            bridge_recalc_after=bridge_recalc_after,
        )
        added = int(edges.shape[0] - prev_edges)
        if added > 0:
            log(f"added {added} synthetic bridge{'s' if added != 1 else ''}")
        else:
            log("already connected; no bridges added")

        if rebuild_mst:
            before = int(edges.shape[0])
            edges = _build_mst(skel.nodes, edges)
            log(f"recomputed MST ({before} → {edges.shape[0]} edges)")

    skel.edges = edges
    skel._invalidate_spatial_index()


# -----------------------------------------------------------------------------
#  soma-adjacent cleanups
# -----------------------------------------------------------------------------


def merge_near_soma_nodes(
    skel,
    *,
    mesh_vertices: np.ndarray | None = None,
    radius_key: str = "median",
    inside_tol: float = 0.0,
    near_factor: float = 1.2,
    fat_factor: float = 0.20,
    verbose: bool = True,
):
    """
    Collapse nodes whose spheres overlap the soma **exactly** like stage 5 of
    :func:`skeliner.skeletonize`.

    Tests applied to every node (see :func:`_merge_near_soma_nodes` for details):

    • Inside test:     ``distance_to_surface < −inside_tol``
    • Near-and-fat:    ``distance_to_center < near_factor × r_soma`` and
      ``radius ≥ fat_factor × r_soma``.

    Nodes satisfying either condition are merged into node 0 along with their
    contributing mesh vertices, after which the soma is re-fitted using the
    expanded vertex set when mesh coordinates are provided.  If
    ``mesh_vertices`` is *None*, vertex bookkeeping is skipped and the soma
    falls back to a spherical approximation with a warning.
    verbose
        When *True* print timing information and messages from the merge routine.
    """
    node2verts_arr = (
        [np.asarray(v, dtype=np.int64).copy() for v in skel.node2verts]
        if skel.node2verts is not None
        else None
    )
    mesh_arr = (
        None if mesh_vertices is None else np.asarray(mesh_vertices, dtype=np.float64)
    )

    with _post_stage("merge redundant near-soma nodes", verbose=verbose) as log:
        (
            nodes_new,
            radii_new,
            node2verts_new,
            vert2node,
            edges_new,
            soma_new,
            old2new,
        ) = _merge_near_soma_nodes(
            np.asarray(skel.nodes, dtype=np.float64),
            {k: v.copy() for k, v in skel.radii.items()},
            np.asarray(skel.edges, dtype=np.int64),
            node2verts_arr,
            soma=skel.soma,
            radius_key=radius_key,
            mesh_vertices=mesh_arr,
            inside_tol=inside_tol,
            near_factor=near_factor,
            fat_factor=fat_factor,
            log=log,
        )

    ntype_new = _remap_ntype(skel.ntype, old2new, len(nodes_new))

    return Skeleton(
        soma=soma_new,
        nodes=nodes_new,
        radii=radii_new,
        edges=edges_new,
        ntype=ntype_new,
        node2verts=(
            [np.asarray(v, dtype=np.int64) for v in node2verts_new]
            if node2verts_new is not None
            else None
        ),
        vert2node=vert2node,
        meta={**skel.meta},
        extra={**skel.extra},
    )


def prune_neurites(
    skel,
    *,
    mesh_vertices: np.ndarray | None = None,
    tip_extent_factor: float = 1.2,
    stem_extent_factor: float = 3.0,
    drop_single_node_branches: bool = True,
    verbose: bool = True,
):
    """
    Remove tiny peri-soma neurites (stage 8 of :func:`skeliner.skeletonize`).

    Behaviour matches the original two-pass routine:

    1. **Extent pruning** – branches whose stem and tips never exceed the
       thresholds (multiples of the soma radius) collapse into the soma.
    2. **Single-node pruning** – optionally collapse degree-1 twigs whose parent
       has degree ≥ 3.

    The stage mirrors :func:`skeliner.skeletonize` even when ``mesh_vertices`` is
    unavailable. In that case radius/soma refits are skipped and a warning is
    emitted in the verbose log.
    verbose
        When *True* print timing information and messages from the pruning routine.
    """
    if not skel.node2verts:
        raise ValueError("prune_neurites requires node2verts data.")
    mesh_arr = (
        None if mesh_vertices is None else np.asarray(mesh_vertices, dtype=np.float64)
    )

    with _post_stage("prune tiny neurites", verbose=verbose) as log:
        (
            nodes_new,
            radii_new,
            node2verts_new,
            vert2node,
            edges_new,
            soma_new,
            old2new,
        ) = _prune_neurites(
            np.asarray(skel.nodes, dtype=np.float64),
            {k: v.copy() for k, v in skel.radii.items()},
            [np.asarray(v, dtype=np.int64).copy() for v in skel.node2verts],
            np.asarray(skel.edges, dtype=np.int64),
            soma=skel.soma,
            mesh_vertices=mesh_arr,
            tip_extent_factor=tip_extent_factor,
            stem_extent_factor=stem_extent_factor,
            drop_single_node_branches=drop_single_node_branches,
            log=log,
        )

    ntype_new = _remap_ntype(skel.ntype, old2new, len(nodes_new))

    return Skeleton(
        soma=soma_new,
        nodes=nodes_new,
        radii=radii_new,
        edges=edges_new,
        ntype=ntype_new,
        node2verts=[np.asarray(v, dtype=np.int64) for v in node2verts_new],
        vert2node=vert2node,
        meta={**skel.meta},
        extra={**skel.extra},
    )


def rebuild_mst(skel, *, verbose: bool = True):
    """Recompute the global MST to remove microscopic cycles (optionally verbosely)."""
    with _post_stage("build global minimum-spanning tree", verbose=verbose) as log:
        before = int(skel.edges.shape[0])
        state = SkeletonState(
            nodes=skel.nodes.copy(),
            radii={k: v.copy() for k, v in skel.radii.items()},
            edges=skel.edges.copy(),
            node2verts=(
                None
                if skel.node2verts is None
                else [np.asarray(v, dtype=np.int64).copy() for v in skel.node2verts]
            ),
            vert2node=None if skel.vert2node is None else dict(skel.vert2node),
        )
        edges_new = _build_mst(np.asarray(state.nodes, dtype=np.float64), state.edges)
        log(f"edges contracted {before} → {edges_new.shape[0]}")
    return Skeleton(
        soma=skel.soma,
        nodes=state.nodes,
        radii=state.radii,
        edges=edges_new,
        ntype=None if skel.ntype is None else skel.ntype.copy(),
        node2verts=state.node2verts,
        vert2node=state.vert2node,
        meta={**skel.meta},
        extra={**skel.extra},
    )


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

    state = SkeletonState(
        nodes=skel.nodes,
        radii=skel.radii,
        edges=skel.edges,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
    )
    new_state, _ = compact_state(state, keep_mask)
    skel.nodes = new_state.nodes
    skel.radii = new_state.radii
    skel.edges = new_state.edges
    skel.node2verts = new_state.node2verts
    skel.vert2node = new_state.vert2node


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
    state = SkeletonState(
        nodes=skel.nodes.copy(),
        radii={k: v.copy() for k, v in skel.radii.items()},
        edges=skel.edges.copy(),
        node2verts=(
            [np.asarray(v, dtype=np.int64).copy() for v in skel.node2verts]
            if skel.node2verts is not None
            else None
        ),
        vert2node=dict(skel.vert2node) if skel.vert2node is not None else None,
    )
    ntype = skel.ntype.copy() if skel.ntype is not None else None

    # Swap 0 ↔ tgt
    swap = tgt
    swap_nodes(state, 0, swap)
    if ntype is not None:
        ntype[[0, swap]] = ntype[[swap, 0]]

    # Remap edges with a permutation
    perm = np.arange(N, dtype=np.int64)
    perm[[0, swap]] = perm[[swap, 0]]
    edges = remap_edges(state.edges, perm)

    if rebuild_mst:
        edges = _build_mst(state.nodes, edges)

    if radius_key not in state.radii:
        raise KeyError(
            f"radius_key '{radius_key}' not found in skel.radii "
            f"(available keys: {tuple(state.radii)})"
        )
    r0 = float(state.radii[radius_key][0])
    new_soma = Soma.from_sphere(
        center=state.nodes[0],
        radius=r0,
        verts=state.node2verts[0]
        if state.node2verts is not None and state.node2verts[0].size > 0
        else None,
    )

    if set_soma_ntype and ntype is not None:
        ntype[0] = 1

    new_skel = Skeleton(
        soma=new_soma,
        nodes=state.nodes,
        radii=state.radii,
        edges=edges,
        ntype=ntype,
        node2verts=state.node2verts,
        vert2node=state.vert2node,
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


def detect_soma(
    skel,
    *,
    radius_key: str = "median",
    soma_radius_percentile_threshold: float = 99.9,
    soma_radius_distance_factor: float = 4.0,
    soma_min_nodes: int = 3,
    verbose: bool = True,
    mesh_vertices: np.ndarray | None = None,
):
    """
    Post-hoc soma detection **on an existing Skeleton**.

    Examples
    --------
    >>> import skeliner as sk
    >>> s = sk.skeletonize(mesh, detect_soma=False)  # soma missed
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
        When *True* emit timing and sub-messages using the unified post-processing logger.

    Returns
    -------
    Skeleton
        *Either* the original instance (no change was necessary) *or* a new
        skeleton whose node 0 is the freshly detected soma centroid.
    """

    if radius_key not in skel.radii:
        raise KeyError(
            f"radius_key '{radius_key}' not found in skel.radii "
            f"(available keys: {tuple(skel.radii)})"
        )
    if len(skel.nodes) <= 1:
        return skel

    has_node2verts = skel.node2verts is not None and len(skel.node2verts) > 0
    has_vert2node = skel.vert2node is not None and len(skel.vert2node) > 0

    state = SkeletonState(
        nodes=skel.nodes,
        radii=skel.radii,
        edges=skel.edges,
        node2verts=(
            [np.asarray(v, dtype=np.int64) for v in skel.node2verts]
            if has_node2verts and skel.node2verts is not None
            else None
        ),
        vert2node=dict(skel.vert2node) if has_vert2node else {},
    ).clone()
    nodes = state.nodes
    radii = state.radii
    node2verts = state.ensure_node2verts()
    vert2node = state.vert2node or {}

    with _post_stage(" post-skeletonization soma detection", verbose=verbose) as log:
        (
            nodes,
            radii,
            node2verts,
            vert2node,
            soma_new,
            has_soma,
            old2new,
        ) = _detect_soma(
            nodes,
            radii,
            node2verts,
            vert2node,
            soma_radius_percentile_threshold=soma_radius_percentile_threshold,
            soma_radius_distance_factor=soma_radius_distance_factor,
            soma_min_nodes=soma_min_nodes,
            detect_soma=True,
            radius_key=radius_key,
            mesh_vertices=(
                np.asarray(mesh_vertices, dtype=np.float64)
                if mesh_vertices is not None
                else None
            ),
            log=log,
        )

        if not has_soma:
            log("existing soma kept unchanged.")
            return skel

    edges = remap_edges(state.edges, old2new)
    ntype_new = _remap_ntype(skel.ntype, old2new, len(nodes))

    node2verts_ret = node2verts if has_node2verts else None
    vert2node_ret = vert2node if has_vert2node else None

    return Skeleton(
        soma=soma_new,
        nodes=nodes,
        radii=radii,
        edges=_build_mst(nodes, edges),
        ntype=ntype_new,
        node2verts=node2verts_ret,
        vert2node=vert2node_ret,
        meta={**skel.meta},
        extra={**skel.extra},
    )


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
        vert2node_new = rebuild_vert2node(new_node2verts)
    else:
        vert2node_new = None

    # Keep soma at index 0
    new_root = int(old2new.get(0, 0))
    if new_root != 0 and len(nodes_new):
        state = SkeletonState(
            nodes=nodes_new,
            radii=radii_new,
            edges=np.empty((0, 2), dtype=np.int64),
            node2verts=new_node2verts,
            vert2node=vert2node_new,
        )
        swap_nodes(state, 0, new_root)
        nodes_new = state.nodes
        radii_new = state.radii
        new_node2verts = state.node2verts
        vert2node_new = state.vert2node
        ntype_new[[0, new_root]] = ntype_new[[new_root, 0]]
        perm = np.arange(len(nodes_new), dtype=np.int64)
        perm[[0, new_root]] = perm[[new_root, 0]]
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
