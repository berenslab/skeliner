"""skeliner.post – post-processing functions for skeletons."""

import functools
import time
from contextlib import contextmanager
from typing import Iterable, Set, cast

import numpy as np
from numpy.typing import ArrayLike

from . import dx
from ._backend import kernels
from ._core import (
    _bridge_gaps,
    _build_mst,
    _detect_soma,
    _estimate_radius,
    _merge_near_soma_nodes,
    _prune_neurites,
    _radii_from_distances,
)
from ._state import (
    SkeletonState,
    as_vertex_map,
    compact_state,
    rebuild_vert2node,
    remap_edges,
    swap_nodes,
)
from .dataclass import Mesh, Skeleton, Soma, mesh_args
from .python import post as _python_post

__skeleton__ = [
    # editing bins
    "reassign_verts",
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
    "calibrate_radii",
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


def _fill_ntype_gaps(ntype: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Fill zero-labeled gaps by propagating neighboring non-zero (non-root/soma) labels.
    Iterative majority voting; ties keep the gap.
    """
    if ntype is None or ntype.size == 0 or edges is None or edges.size == 0:
        return ntype
    filled = kernels.post.fill_ntype_gaps(
        np.asarray(ntype, dtype=np.int64),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )
    return filled.astype(ntype.dtype, copy=False)


def _derive_root_label(ntype: np.ndarray) -> int:
    val0 = int(ntype[0]) if len(ntype) else -1
    if val0 == 1:
        return 1
    if val0 == -1:
        return -1
    return -1


_ensure_root_label = _python_post.ensure_root_label


def _remap_ntype(
    ntype: np.ndarray | None,
    old2new: np.ndarray,
    new_len: int,
    *,
    root_hint: int | None = None,
    edges: np.ndarray | None = None,
    fill_gaps: bool = True,
) -> np.ndarray | None:
    if ntype is None:
        return None
    root_label = int(root_hint) if root_hint in (-1, 1) else _derive_root_label(ntype)
    mapped = kernels.post.remap_ntype(
        np.asarray(ntype, dtype=np.int64),
        np.asarray(old2new, dtype=np.int64),
        int(new_len),
        root_label,
        np.empty((0, 2), dtype=np.int64)
        if edges is None
        else np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        bool(fill_gaps),
    )
    return mapped.astype(ntype.dtype, copy=False)


# -----------------------------------------------------------------------------
# editing bins: which mesh vertices each node owns
# -----------------------------------------------------------------------------


def _cl_dist_lut(skel, n_verts: int) -> np.ndarray | None:
    """Per-vertex centreline distance, or None when it was not kept.

    ``centerline`` is the one radius that is not a function of the vertices
    alone — it aggregates a perpendicular distance the second binning pass
    measured.  ``skeletonize`` keeps those distances precisely so a bin can be
    edited without leaving that key stale beside freshly recomputed ones.
    """
    vids = skel.extra.get("cl_dist_vids")
    vals = skel.extra.get("cl_dist_vals")
    if vids is None or vals is None or len(vids) == 0:
        return None
    lut = np.zeros(n_verts, dtype=np.float64)
    lut[np.asarray(vids, dtype=np.int64)] = np.asarray(vals, dtype=np.float64)
    return lut


def _recompute_nodes(skel, nodes: Iterable[int], mesh, cl_lut) -> list[str]:
    """Rebuild position and radii for *nodes* from the vertices they own.

    Returns the radius keys that could not be rebuilt.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    stale: list[str] = []
    for nid in nodes:
        owned = np.asarray(skel.node2verts[nid], dtype=np.int64)
        if owned.size == 0:
            continue
        pts = verts[owned]
        centre = pts.mean(axis=0)
        skel.nodes[nid] = centre
        d = np.linalg.norm(pts - centre, axis=1)
        stale = _radii_from_distances(
            skel.radii, nid, d, cl_d=None if cl_lut is None else cl_lut[owned]
        )
    return stale


@mesh_args()
def reassign_verts(skel, verts: ArrayLike, to: int, *, mesh, verbose: bool = False):
    """Move mesh vertices into bin *to*, and recompute what depends on them.

    A node *is* the set of mesh vertices it owns: its position is their
    centroid and every radius is an aggregate over them.  So this is the one
    primitive bin editing needs — nudging a bin boundary, merging two nodes and
    splitting one are all this call with different arguments.

    Everything downstream is recomputed, never interpolated.  A node left
    owning nothing is dropped, because a node with no vertices has no position
    and no radius.

    Parameters
    ----------
    skel
        Modified in place.
    verts
        Mesh vertex ids to move.  Every one must currently belong to an arbor
        bin: unowned surface is soma, organelle or discarded, and claiming it
        would be a components decision made without re-deriving them.
    to
        Destination node.  Node 0 is refused — its "bin" is ``soma.verts``,
        assigned wholesale by the soma stitch, and belongs to the components,
        not to the partition.
    mesh
        The mesh the skeleton was built from.  Positions and radii are
        measured from it.

    Returns
    -------
    dict
        ``moved``, ``donors``, ``dropped`` (nodes emptied and removed),
        ``old2new`` (only when something was dropped), and ``stale_radii`` —
        radius keys that could not be recomputed, e.g. ``calibrated``, which
        is measured by ray casting rather than from the vertices.

    Notes
    -----
    **Edges are left alone.** Moving vertices between adjacent bins almost
    never changes which bins touch, and re-deriving the whole graph from the
    surface would silently discard the edges that have no surface support —
    the soma stems and ``bridge_gaps`` bridges — and can disconnect the
    skeleton. Re-derive explicitly with :func:`rebuild_mst`, or re-skeletonize,
    when that is what you want.

    The one exception is a bin emptied by this call. It was absorbed into
    *to*, not deleted, so its edges are **contracted onto** *to* rather than
    dropped — otherwise merging a node would cut the tree at every neighbour
    that node was holding on to.
    """
    if skel.node2verts is None:
        raise ValueError("Skeleton carries no node2verts; nothing to reassign.")
    to = int(to)
    if not 0 <= to < len(skel.nodes):
        raise ValueError(f"node {to} out of range (0..{len(skel.nodes) - 1})")
    if to == 0:
        raise ValueError(
            "node 0 is the soma, not a bin — its vertices are Soma.verts and "
            "belong to the components. Reassign them on the mesh components."
        )

    moving = np.unique(np.asarray(verts, dtype=np.int64).ravel())
    if moving.size == 0:
        raise ValueError("No vertices given.")
    if moving.min() < 0 or moving.max() >= len(mesh.vertices):
        raise ValueError("vertex ids must lie within the mesh")

    # node 0 holds soma.verts, which is not a bin; a later bin wins a vertex
    bins = [np.asarray(v, dtype=np.int64).ravel() for v in skel.node2verts[1:]]
    owner = np.full(len(mesh.vertices), -1, dtype=np.int64)
    if bins:
        owner[np.concatenate(bins)] = np.repeat(
            np.arange(1, len(bins) + 1), [b.size for b in bins]
        )

    unowned = moving[owner[moving] < 0]
    if unowned.size:
        raise ValueError(
            f"{unowned.size:,} of {moving.size:,} vertices belong to no bin "
            "(soma, organelle or discarded surface) — the mesh components "
            "own those, not the skeleton."
        )

    donors = sorted({int(n) for n in np.unique(owner[moving])} - {to})
    if not donors:
        return {
            "moved": 0,
            "donors": [],
            "dropped": [],
            "stale_radii": [],
        }

    keep_set = set(moving.tolist())
    for nid in donors:
        owned = np.asarray(skel.node2verts[nid], dtype=np.int64)
        skel.node2verts[nid] = owned[~np.isin(owned, moving)]
    skel.node2verts[to] = np.unique(
        np.concatenate(
            [
                np.asarray(skel.node2verts[to], dtype=np.int64),
                np.asarray(sorted(keep_set)),
            ]
        )
    )

    dropped = [n for n in donors if skel.node2verts[n].size == 0]
    touched = [n for n in donors if n not in dropped] + [to]
    old2new = None

    if dropped:
        # A bin that gave everything to `to` was *absorbed* by it, not
        # deleted, so its edges are now `to`'s.  Contract them onto `to`
        # before compacting: `remap_edges` drops edges touching a removed
        # node, which is right for a deletion and would here cut the tree
        # at every neighbour the absorbed node was holding.
        contracted = np.asarray(skel.edges, dtype=np.int64).copy()
        if contracted.size:
            contracted[np.isin(contracted, dropped)] = to
            # remap_edges removes the resulting self-loops and dedups.
            skel.edges = contracted

        keep_mask = np.ones(len(skel.nodes), dtype=bool)
        keep_mask[dropped] = False
        state = SkeletonState(
            nodes=skel.nodes,
            radii=skel.radii,
            edges=skel.edges,
            node2verts=skel.node2verts,
            vert2node=skel.vert2node,
        )
        new_state, old2new = compact_state(state, keep_mask, return_old2new=True)
        skel.nodes = new_state.nodes
        skel.radii = new_state.radii
        skel.edges = new_state.edges
        skel.node2verts = new_state.node2verts
        skel.vert2node = new_state.vert2node
        skel.ntype = _remap_ntype(
            skel.ntype,
            old2new,
            len(new_state.nodes),
            edges=new_state.edges,
            fill_gaps=True,
        )
        touched = [int(old2new[n]) for n in touched]
    else:
        skel.vert2node = rebuild_vert2node(skel.node2verts)

    cl_lut = _cl_dist_lut(skel, len(mesh.vertices))
    stale = sorted(_recompute_nodes(skel, touched, mesh, cl_lut))
    skel._invalidate_spatial_index()
    if verbose:
        print(
            f"[skeliner.post] reassign_verts: {moving.size:,} verts → node {to}, "
            f"{len(donors)} donor(s), {len(dropped)} emptied"
        )
        if stale:
            print(f"      └─ not recomputable, now stale: {', '.join(stale)}")

    return {
        "moved": int(moving.size),
        "donors": donors,
        "dropped": dropped,
        "old2new": None if old2new is None else old2new.tolist(),
        "stale_radii": stale,
    }


def _bins_touch(mesh, a: ArrayLike, b: ArrayLike) -> bool:
    """Does any mesh edge run from a vertex of *a* to a vertex of *b*?

    The local form of the question :func:`skeliner.dx.edge_support` answers
    globally, without building the whole node-adjacency graph for it.
    """
    e = np.asarray(mesh.edges_unique, dtype=np.int64)
    in_a = np.isin(e, np.asarray(a, dtype=np.int64))
    in_b = np.isin(e, np.asarray(b, dtype=np.int64))
    return bool(((in_a[:, 0] & in_b[:, 1]) | (in_a[:, 1] & in_b[:, 0])).any())


@mesh_args()
def split_node(skel, node: int, verts: ArrayLike, *, mesh, verbose: bool = False):
    """Promote part of bin *node* to a node of its own.

    The remaining bin verb, and the only one whose destination does not exist
    yet.  The *partition* needs no invention — the caller draws it — so this
    is :func:`reassign_verts` into a node created for the purpose, and
    position and radii for both halves follow from the vertices as always.

    **The new node is attached to its parent and to nothing else**, which is
    the only edge here that can be defended.  The tempting alternative is to
    derive its edges from surface adjacency, giving ``a—k'—k—b`` where the
    geometry says so — but surface adjacency does not imply tree adjacency.
    Measured on 549190673, 147 of the 156 bin pairs that share surface while
    three or more tree hops apart lie *within one branch*, with the two bins
    overlapping in space: a densely packed axon tuft.  Deriving edges there
    would wire the new node to whatever branch happens to be touching.  So the
    split makes one honest edge and leaves the rest to :func:`graft` and
    :func:`clip`, where ``dx.edge_support`` can say which joins the surface
    actually backs.

    Node ids are stable: the new node is appended, and nothing is dropped, so
    no existing id changes.

    Parameters
    ----------
    node
        The bin to split.  Node 0 is refused, as everywhere — its "bin" is
        ``soma.verts``.
    verts
        The subset to promote.  Vertices *node* does not own are ignored and
        counted, because the ≥2-of-3 face rule means a selection drawn on the
        surface routinely catches a neighbour's.  Taking the whole bin is
        refused: that renames a node rather than splitting it.

    Returns
    -------
    dict
        ``node`` (the new id), ``parent``, ``moved``, ``ignored``,
        ``supported`` — whether the two halves actually share surface, false
        when the split cut a bin that was already in two patches — and
        ``stale_radii``.
    """
    if skel.node2verts is None:
        raise ValueError("Skeleton carries no node2verts; nothing to split.")
    node = int(node)
    if not 0 <= node < len(skel.nodes):
        raise ValueError(f"node {node} out of range (0..{len(skel.nodes) - 1})")
    if node == 0:
        raise ValueError(
            "node 0 is the soma, not a bin — its vertices are Soma.verts and "
            "belong to the components. Edit them on the mesh components."
        )

    owned = np.asarray(skel.node2verts[node], dtype=np.int64)
    picked = np.unique(np.asarray(verts, dtype=np.int64).ravel())
    if picked.size == 0:
        raise ValueError("No vertices given.")
    if picked.min() < 0 or picked.max() >= len(mesh.vertices):
        raise ValueError("vertex ids must lie within the mesh")

    moving = picked[np.isin(picked, owned)]
    ignored = int(picked.size - moving.size)
    if moving.size == 0:
        raise ValueError(f"none of those vertices belong to node {node}")
    if moving.size == len(owned):
        raise ValueError(
            f"that is all {len(owned):,} of node {node}'s vertices — a split "
            "has to leave something behind"
        )

    # Extend every node-indexed array by one.  The values are placeholders
    # that reassign_verts recomputes from the vertices; ntype is the exception
    # and is inherited, there being nothing to compute it from.
    new_id = len(skel.nodes)
    skel.nodes = np.vstack([skel.nodes, skel.nodes[node][None, :]])
    for key, arr in skel.radii.items():
        skel.radii[key] = np.append(arr, arr[node])
    if skel.ntype is not None:
        skel.ntype = np.append(skel.ntype, skel.ntype[node])
    skel.node2verts.append(np.empty(0, dtype=np.int64))
    skel.edges = np.vstack([np.asarray(skel.edges, dtype=np.int64), [[node, new_id]]])

    result = reassign_verts(skel, moving, new_id, mesh=mesh)
    supported = _bins_touch(mesh, skel.node2verts[node], skel.node2verts[new_id])

    if verbose:
        print(
            f"[skeliner.post] split_node: {moving.size:,} verts of node {node} "
            f"→ new node {new_id}"
        )
        if not supported:
            print("      └─ the two halves share no surface")

    return {
        "node": new_id,
        "parent": node,
        "moved": int(moving.size),
        "ignored": ignored,
        "supported": supported,
        "stale_radii": result["stale_radii"],
    }


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
        edges = np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2)
        if kernels.skeletonize.bfs_parents(edges, len(skel.nodes), u)[v] >= 0:
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
        # what the soma (0) still reaches
        edges = np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2)
        reached = np.isfinite(kernels.dx.hops(edges, len(skel.nodes), 0))
        if reached.all():
            return  # nothing to drop

        _rebuild_drop_set(skel, np.flatnonzero(~reached))


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

    deg = dx._degrees(skel)
    for n in sorted(drop):
        if deg[n] <= 2:
            continue
        # a hub still joining two or more kept neighbours stays
        neigh = [v for v in dx.neighbors(skel, n) if v not in drop]
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

    ntype_new = _remap_ntype(
        skel.ntype,
        old2new,
        len(nodes_new),
        edges=edges_new,
        fill_gaps=True,
    )

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

    ntype_new = _remap_ntype(
        skel.ntype,
        old2new,
        len(nodes_new),
        edges=edges_new,
        fill_gaps=True,
    )

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
            vert2node=None if skel.vert2node is None else skel.vert2node.copy(),
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

    keep_mask = np.ones(len(skel.nodes), dtype=bool)
    keep_mask[np.fromiter(map(int, drop), dtype=np.int64)] = False
    if not keep_mask[0]:
        raise RuntimeError("Attempted to drop the soma (vertex 0)")

    state = SkeletonState(
        nodes=skel.nodes,
        radii=skel.radii,
        edges=skel.edges,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
    )
    # request remapping to keep auxiliary arrays (including ntype) consistent
    new_state, old2new = compact_state(state, keep_mask, return_old2new=True)
    skel.nodes = new_state.nodes
    skel.radii = new_state.radii
    skel.edges = new_state.edges
    skel.node2verts = new_state.node2verts
    skel.vert2node = new_state.vert2node
    skel.ntype = _remap_ntype(
        skel.ntype, old2new, len(new_state.nodes), edges=new_state.edges, fill_gaps=True
    )


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
    return np.bincount(edges.reshape(-1), minlength=n).astype(np.int64)


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
    set_soma_ntype: bool = False,
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
        vert2node=skel.vert2node.copy() if skel.vert2node is not None else None,
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

    if ntype is not None:
        ntype[0] = 1 if set_soma_ntype else -1
        if len(ntype) > 1:  # remove all other somas and roots
            dupes = (ntype == 1) | (ntype == -1)
            dupes[0] = False
            ntype[dupes] = 0  # Default to unknown
        ntype = _fill_ntype_gaps(ntype, edges)
        ntype = _ensure_root_label(ntype, ntype[0])

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
        vert2node=skel.vert2node.copy() if has_vert2node else {},
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
    ntype_new = _remap_ntype(
        skel.ntype,
        old2new,
        len(nodes),
        root_hint=1,
        edges=edges,
        fill_gaps=True,
    )

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

    Runs of degree-2 nodes between anchors (node 0, leaves, branch points)
    split greedily into groups whose radii stay within
    ``atol + rtol * max(|r|)`` of their running mean; each group becomes one
    node at its length-weighted centroid, with aggregated radii and the
    union of its vertices. Optional: absorb leaf endpoints and/or slide
    branchpoints into adjacent runs when |Δr| ≤ atol + rtol * max(r_anchor,
    r_group); an anchor absorbing from several runs takes all of them.
    Merging node 0 is never allowed.
    """

    if radius_key not in skel.radii:
        raise KeyError(
            f"radius_key '{radius_key}' not found in skel.radii "
            f"(available keys: {tuple(skel.radii)})"
        )
    if aggregate not in _python_post.AGGREGATES:
        raise ValueError("aggregate must be 'median', 'mean', or 'area'")
    N = int(len(skel.nodes))
    if N <= 1:
        return skel

    keys = list(skel.radii)
    radii = np.stack([np.asarray(skel.radii[k], dtype=np.float64) for k in keys])
    ntype0 = skel.ntype if skel.ntype is not None else np.full(N, 0, dtype=np.int8)
    root_label = _derive_root_label(ntype0)

    has_node2verts = skel.node2verts is not None and len(skel.node2verts) > 0
    has_vert2node = skel.vert2node is not None and len(skel.vert2node) > 0
    ptr = idx = None
    if has_node2verts:
        bins = [np.asarray(v, dtype=np.int64).ravel() for v in skel.node2verts[:N]]
        bins += [np.empty(0, dtype=np.int64)] * (N - len(bins))
        ptr = np.zeros(N + 1, dtype=np.int64)
        np.cumsum([b.size for b in bins], out=ptr[1:])
        idx = np.concatenate(bins)

    nodes_new, radii_arr, edges_arr, old2new, new_ptr, new_idx = (
        kernels.post.downsample(
            np.asarray(skel.nodes, dtype=np.float64),
            radii,
            keys.index(radius_key),
            np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2),
            ptr,
            idx,
            float(rtol),
            float(atol),
            aggregate,
            bool(merge_endpoints),
            bool(slide_branchpoints),
            None if max_anchor_shift is None else float(max_anchor_shift),
        )
    )
    radii_new = {k: radii_arr[i] for i, k in enumerate(keys)}
    new_node2verts = (
        None
        if new_ptr is None
        else [new_idx[new_ptr[i] : new_ptr[i + 1]] for i in range(len(nodes_new))]
    )
    ntype_new = _remap_ntype(ntype0, old2new, len(nodes_new), root_hint=root_label)
    if edges_arr.size:
        edges_arr = np.unique(np.sort(edges_arr, axis=1), axis=0)

    # vert2node: each vertex to its node's new node, if it has one
    if has_vert2node:
        vert2node_new = as_vertex_map(skel.vert2node).remapped(old2new)
    elif new_node2verts is not None:
        vert2node_new = rebuild_vert2node(new_node2verts)
    else:
        vert2node_new = None

    # Keep soma at index 0
    new_root = int(old2new[0]) if old2new[0] >= 0 else 0
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
        if ntype_new is not None:
            ntype_new[[0, new_root]] = ntype_new[[new_root, 0]]
        perm = np.arange(len(nodes_new), dtype=np.int64)
        perm[[0, new_root]] = perm[[new_root, 0]]
        edges_arr = perm[edges_arr]  # apply to both columns at once

        edges_arr = np.sort(edges_arr, axis=1)
        edges_arr = edges_arr[edges_arr[:, 0] != edges_arr[:, 1]]  # drop self-loops
        edges_arr = np.unique(edges_arr, axis=0)
    if ntype_new is not None:
        ntype_new = _ensure_root_label(ntype_new, root_label)

    n_new = len(nodes_new)
    n_comp = int(kernels.dx.components(edges_arr, n_new).max()) + 1 if n_new else 0
    if len(edges_arr) != n_new - n_comp:  # not a forest: span one
        edges_arr = _build_mst(nodes_new, edges_arr)

    if ntype_new is not None:
        ntype_new = _fill_ntype_gaps(ntype_new, edges_arr)
        ntype_new = _ensure_root_label(ntype_new, root_label)

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


@mesh_args()
def submesh_by_vertices(
    mesh: Mesh,
    vertex_indices: np.ndarray,
    *,
    vertex_faces=None,
) -> Mesh:
    """
    Reduce mesh to a subset of vertices and corresponding faces.

    *vertex_faces*, a ``VertexFaces`` of *mesh* (either path's), finds the
    faces from the vertices' own instead of testing every face of the mesh.
    """
    vertex_indices = np.asarray(vertex_indices, dtype=np.int64)
    if vertex_faces is not None:
        faces = mesh.faces[vertex_faces.faces_within(vertex_indices)]
    else:
        used = np.isin(mesh.faces, vertex_indices).all(axis=1)
        faces = mesh.faces[used]

    # build mapping from old → new vertex indices
    old_to_new = -np.ones(len(mesh.vertices), dtype=np.int64)
    old_to_new[vertex_indices] = np.arange(len(vertex_indices))

    new_faces = old_to_new[faces]

    new_vertices = mesh.vertices[vertex_indices]

    return Mesh(new_vertices, new_faces)


@functools.cache
def _embree_intersector():
    """trimesh's embree ``RayMeshIntersector``, or None without embree.

    Asked once: a failing import is slow, and calibration asks per node.
    Without ``embreex`` or ``pyembree``, which trimesh's wrapper needs, the
    answer is None without importing trimesh (and with it scipy).
    """
    from importlib.util import find_spec

    if find_spec("embreex") is None and find_spec("pyembree") is None:
        return None
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError:
        return None
    return RayMeshIntersector


@mesh_args()
def filter_inner_surfaces_raycast(mesh, num_rays=20, thresh=0.2, sample=True):
    """
    Return vertex indices belonging to outer faces only.
    """
    # ---- Ray directions ----
    if sample:
        phi = np.random.uniform(0, 2 * np.pi, num_rays)
        theta = np.arccos(np.random.uniform(-1, 1, num_rays))
        directions = np.column_stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
    else:
        directions = _golden_spiral(num_rays)

    # ---- Precompute centroids & normals ----
    C = mesh.triangles_center
    N = mesh.face_normals
    F = len(C)

    # ---- Expand origins to (F, R, 3) ----
    origins = C[:, None, :] + 0.001 * N[:, None, :]
    origins = np.broadcast_to(origins, (F, num_rays, 3))

    # ---- Outward-facing ray mask (F, R) ----
    mask = (directions @ N.T).T > 0
    active_idx = np.nonzero(mask)

    if len(active_idx[0]) == 0:
        # All faces are considered "outer"
        return np.unique(mesh.faces.reshape(-1))

    # Flatten rays
    origins_flat = origins[active_idx]
    directions_flat = directions[active_idx[1]]

    # ---- Fast intersector ----
    embree = _embree_intersector()
    if embree is None:
        # trimesh's own `ray_triangle` test, on the Rust core
        hit_mask = kernels.post.ray_hits(
            np.asarray(mesh.triangles, dtype=np.float64).reshape(-1, 9),
            np.asarray(N, dtype=np.float64),
            np.ascontiguousarray(origins_flat, dtype=np.float64),
            np.ascontiguousarray(directions_flat, dtype=np.float64),
        )
    else:
        hits = embree(mesh.to_trimesh()).intersects_first(origins_flat, directions_flat)
        hit_mask = hits != -1

    # Count hits for each face
    hit_count = np.zeros(F, dtype=int)
    np.add.at(hit_count, active_idx[0], hit_mask)

    # Select outer faces
    threshold = int(num_rays * thresh)
    outer_faces = np.nonzero(hit_count < threshold)[0]

    # ---- Return vertex indices from these faces ----
    outer_vertices = mesh.faces[outer_faces].reshape(-1)
    return np.unique(outer_vertices)


#: :func:`calibrate_radii`'s methods, by the kernels' kind codes.
_RADIUS_KINDS = ("fallback", "full_centerline", "outer_centerline")


def _golden_spiral(n: int) -> np.ndarray:
    """*n* unit directions spread evenly over the sphere."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    idx = np.arange(n)
    theta = 2 * np.pi * idx / golden_ratio
    phi = np.arccos(1 - 2 * (idx + 0.5) / n)
    return np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )


def _calibrate_node_by_node(
    skel: Skeleton,
    mesh: Mesh,
    r_new: np.ndarray,
    r_kind: np.ndarray,
    *,
    radius_metric: str,
    aggregate: str,
    min_n_outer: int,
    min_frac_outer: float,
    min_verts_q_outer: float,
    min_n_verts_bulb: int,
    rays_num_outer: int,
    rays_thresh_outer: float,
    rays_sample: bool,
) -> None:
    """:func:`calibrate_radii`'s nodes one at a time, a trimesh submesh each,
    for the ray casts the kernels do not make: embree's, and fresh random
    rays per node.  Fills *r_new* and *r_kind* in place."""
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_verts = np.array([len(i) for i in skel.node2verts])
    vertex_faces = kernels.post.VertexFaces(
        np.asarray(mesh.faces, dtype=np.int64), len(mesh_vertices)
    )
    for i in range(len(r_new)):
        if skel.ntype[i] == 1 and skel.soma is not None:  # Ignore soma
            continue

        vids = (
            np.asarray(skel.node2verts[i], dtype=np.int64)
            if i < len(skel.node2verts)
            else np.empty(0, dtype=np.int64)
        )
        if vids.size == 0:
            continue

        outer_success = False
        if n_verts[i] >= min_n_verts_bulb and min_verts_q_outer < 100.0:
            n_inner = len(vids)
            mesh_i = submesh_by_vertices(mesh, vids, vertex_faces=vertex_faces)
            assert len(mesh_i.vertices) == len(vids)
            outer_vids = filter_inner_surfaces_raycast(
                mesh_i,
                num_rays=rays_num_outer,
                thresh=rays_thresh_outer,
                sample=rays_sample,
            )
            n_outer = len(outer_vids)

            if (
                (n_outer >= min_n_outer)
                and (n_outer >= min_frac_outer * n_inner)
                and (n_outer < n_inner)
            ):
                d_local = dx.distance(
                    skel,
                    mesh_vertices[vids[outer_vids]],
                    mode="centerline",
                    radius_metric=radius_metric,
                    allowed_nodes=[int(i)],
                )
                r_new[i] = _estimate_radius(d_local, method=aggregate)
                r_kind[i] = "outer_centerline"
                outer_success = True

        if not outer_success:
            d_local = dx.distance(
                skel,
                mesh_vertices[vids],
                mode="centerline",
                radius_metric=radius_metric,
                allowed_nodes=[int(i)],
            )
            r_new_i = _estimate_radius(d_local, method=aggregate)
            if r_new_i < r_new[i]:  # Update only if smaller
                r_new[i] = r_new_i
                r_kind[i] = "full_centerline"


@mesh_args()
def calibrate_radii(
    skel: Skeleton,
    mesh: Mesh,
    *,
    radius_metric: str | None = None,
    aggregate: str = "trim",
    min_n_outer: int = 20,
    min_frac_outer: float = 0.33,
    min_verts_q_outer: float = 80.0,
    rays_num_outer: int = 30,
    rays_thresh_outer: float = 0.2,
    rays_sample: bool = False,
    store_key: str = "calibrated",
    verbose: bool = False,
) -> None:
    """
    Refine per‑node radii by measuring distances from mesh vertices to the
    skeleton centreline, aggregated within each node's vertex bin.
    Tries to efficiently remove internal mesh structure like mitochondria using a combination of heuristics.

    Parameters
    ----------
    skel
        Skeleton whose radii dictionary will be updated in‑place.
    mesh : Mesh
        Mesh surface of the neuron. Must match the node2verts of the skeleton
    radius_metric
        Existing radius key to use as a fallback for nodes without vertex bins.
        Defaults to the recommended estimator.
    aggregate
        How to aggregate per‑vertex distances within a node: 'median' (default)
        or 'mean'.
    min_n_outer
        Minimum vertices of an outer shell as absolute number.
    min_frac_outer
        Minimum vertices of an outer shell as relative number.
    min_verts_q_outer
        Value between 0 and 100: Minimum vertices (as percentile over all nodes) to be tested for inner shell.
        This assumes that nodes with internal structure have more vertices.
        Set to zero to check all nodes.
    rays_num_outer
        Number of rays used to detect outer shell.
    rays_thresh_outer
        Value between 0 and 1: fraction of rays needed to hit a surface to be considered outer / inner shell.
    rays_sample
        If True, use randomly sampled cells instead of a grid.
    store_key
        Key under which the calibrated radii will be stored in ``skel.radii``.
    verbose
        When True (default), print a concise summary including how many nodes
        fell back to the prior radius and how many were clamped to it.

    Notes
    -----
    - Requires ``skel.node2verts``to compute refined radii.
    - Modifies the skeleton in place; returns None.
    """
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_total = int(len(skel.nodes))
    if n_total == 0:
        raise ValueError("Skeleton has no nodes; cannot calibrate radii.")

    # Ensure we have a fallback radius vector
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]
    if radius_metric not in skel.radii:
        raise ValueError(
            f"radius_metric '{radius_metric}' not found in skel.radii (available keys: {tuple(skel.radii)})"
        )

    # Aggregate distances for each node using per-call whitelist restriction
    r_new = np.array(skel.radii[radius_metric], dtype=np.float64, copy=True)
    r_kind = np.full(n_total, "fallback", dtype="object")

    n_verts = np.array([len(i) for i in skel.node2verts])
    min_n_verts_bulb = int(np.percentile(n_verts, q=min_verts_q_outer))

    with _post_stage("calibrate_radii", verbose=verbose) as log:
        log(
            f"Check for inner meshes in all nodes with min_n_verts_bulb>={min_n_verts_bulb};"
        )
        log(f"This is {np.mean(n_verts > min_n_verts_bulb):.0%} of all nodes")

    if not rays_sample and _embree_intersector() is None:
        # every node in one call on the active path, with the golden-spiral
        # rays and trimesh's own ray test the node-by-node loop uses then
        empty = np.empty(0, dtype=np.int64)
        radii, kinds = kernels.post.calibrate_radii(
            mesh_vertices,
            np.asarray(mesh.faces, dtype=np.int64),
            np.asarray(mesh.triangles_center, dtype=np.float64),
            np.asarray(mesh.face_normals, dtype=np.float64),
            np.asarray(skel.nodes, dtype=np.float64),
            np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2),
            [
                np.asarray(skel.node2verts[i], dtype=np.int64)
                if i < len(skel.node2verts)
                else empty
                for i in range(n_total)
            ],
            (np.asarray(skel.ntype) == 1) & (skel.soma is not None),
            r_new,
            _golden_spiral(rays_num_outer),
            min_n_verts_bulb,
            min_verts_q_outer < 100.0,
            min_n_outer,
            min_frac_outer,
            int(rays_num_outer * rays_thresh_outer),
            aggregate,
        )
        r_new = np.asarray(radii, dtype=np.float64)
        r_kind = np.array(_RADIUS_KINDS, dtype="object")[np.asarray(kinds)]
    else:
        _calibrate_node_by_node(
            skel,
            mesh,
            r_new,
            r_kind,
            radius_metric=radius_metric,
            aggregate=aggregate,
            min_n_outer=min_n_outer,
            min_frac_outer=min_frac_outer,
            min_verts_q_outer=min_verts_q_outer,
            min_n_verts_bulb=min_n_verts_bulb,
            rays_num_outer=rays_num_outer,
            rays_thresh_outer=rays_thresh_outer,
            rays_sample=rays_sample,
        )

    skel.radii[store_key] = r_new
    n_fallback = int(np.sum(r_kind == "fallback"))
    n_full_centerline = int(np.sum(r_kind == "full_centerline"))
    n_outer_centerline = int(np.sum(r_kind == "outer_centerline"))

    # annotate meta
    skel.extra["calibration"] = {
        "status": "computed",
        "aggregate": aggregate,
        "fallback_radius": radius_metric,
        "store_key": store_key,
        "radius_method": r_kind.tolist(),
        "nodes_total": int(n_total),
        "nodes_fallback": n_fallback,
        "nodes_full_centerline": n_full_centerline,
        "nodes_outer_centerline": n_outer_centerline,
    }

    with _post_stage("calibrate_radii summary", verbose=verbose) as log:
        log(f"n_total={n_total}")
        log(f"n_fallback={n_fallback}={n_fallback / n_total:.0%}, ")
        log(
            f"n_full_centerline={n_full_centerline}={n_full_centerline / n_total:.0%}, "
        )
        log(
            f"n_outer_centerline={n_outer_centerline}={n_outer_centerline / n_total:.0%}; "
        )
        log(f"store='{store_key}', base='{radius_metric}'")
