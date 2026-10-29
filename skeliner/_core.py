"""Reusable skeleton helpers shared across the package."""

from __future__ import annotations

import dataclasses
from typing import Callable, List

import numpy as np

from ._backend import kernels
from ._state import SkeletonState, rebuild_vert2node, swap_nodes
from .dataclass import Soma

__all__ = [
    "_find_soma",
    "_bfs_parents",
    "_build_mst",
    "_bridge_gaps",
    "_merge_near_soma_nodes",
    "_merge_single_node_branches",
    "_prune_neurites",
    "_detect_soma",
    "_estimate_radius",
]


def _find_soma(
    nodes: np.ndarray,
    radii: np.ndarray,
    *,
    pct_large: float = 99.9,
    dist_factor: float = 3.0,
    min_keep: int = 2,
) -> tuple[Soma, np.ndarray, bool]:
    """
    Geometry-only soma heuristic shared by the core pipeline and post-processing.
    """
    if nodes.shape[0] == 0:
        raise ValueError("empty skeleton")
    found = kernels.skeletonize.find_soma(
        np.asarray(nodes, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        float(pct_large),
        float(dist_factor),
        int(min_keep),
    )
    if found is None:
        raise RuntimeError(
            f"no nodes above the {pct_large:g}-th percentile (try lowering pct_large)"
        )
    soma_idx, idx_max, center, r_max, has_soma = found
    if not has_soma:
        return Soma.from_sphere(nodes[idx_max], r_max, None), soma_idx, has_soma
    soma = Soma.from_sphere(center=np.asarray(center), radius=r_max, verts=None)
    return soma, soma_idx, has_soma


def _bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> List[int]:
    """Return parent[] array of BFS tree from *root* given an undirected edge list."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return kernels.skeletonize.bfs_parents(edges, int(n_nodes), int(root)).tolist()


def _build_mst(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the edge list of the global minimum-spanning tree."""
    return kernels.skeletonize.build_mst(
        np.asarray(nodes, dtype=np.float64),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


#: How much of each tail ``trim`` discards.  Node radii are built with this
#: everywhere — ``_make_nodes``, the centreline pass, and every recomputation
#: after an edit — so that one estimator name means one thing across a
#: skeleton's whole radii array.
TRIM_FRACTION = 0.05


def _estimate_radius(
    d: np.ndarray,
    *,
    method: str = "median",
    trim_fraction: float = TRIM_FRACTION,
    q: float = 80.0,
) -> float:
    """Return one scalar radius from per-vertex distances *d*.

    The single implementation.  ``post`` used to carry a near-copy that
    differed only in its defaults, so ``trim`` and ``percentile`` quietly
    meant different things depending on which module you reached from.
    """
    if method == "median":
        return float(np.median(d))
    if method == "mean":
        return float(d.mean())
    if method == "max":
        return float(d.max())
    if method == "min":
        return float(d.min())
    if method == "percentile":
        return float(np.percentile(d, q=q))
    if method == "trim":
        lo, hi = np.quantile(d, [trim_fraction, 1.0 - trim_fraction])
        mask = (d >= lo) & (d <= hi)
    elif method == "trimlow":
        mask = d >= np.quantile(d, trim_fraction)
    elif method == "trimhigh":
        mask = d <= np.quantile(d, 1.0 - trim_fraction)
    else:
        raise ValueError(f"Unknown radius estimator '{method}'.")
    if not np.any(mask):
        return float(np.mean(d))
    return float(d[mask].mean())


#: Radius keys that are a plain aggregate over a node's vertex distances, and
#: so can be rebuilt whenever its vertex set changes.
RECOMPUTABLE_RADII = frozenset(
    {"median", "mean", "max", "min", "percentile", "trim", "trimlow", "trimhigh"}
)


def _radii_from_distances(
    radii_dict: dict[str, np.ndarray],
    nid: int,
    d: np.ndarray,
    *,
    cl_d: np.ndarray | None = None,
) -> list[str]:
    """Rewrite node *nid*'s radii from the distances of the vertices it owns.

    Every place that changes which vertices a node owns has to do this, so it
    lives here rather than being spelled out at each of them.

    Not every key is a distance aggregate, and the ones that are not must be
    skipped rather than handed to :func:`_estimate_radius`: ``centerline``
    aggregates *perpendicular* distances (pass them as *cl_d*), and
    ``calibrated`` is measured by ray casting.  Feeding those through was a
    real crash — ``post.prune_neurites`` raised ``Unknown radius estimator
    'centerline'`` on any preprocessing-track skeleton that merged anything
    into the soma.

    Returns the keys it could not rebuild, which are now stale.
    """
    stale: list[str] = []
    for key, arr in radii_dict.items():
        if key == "centerline":
            if cl_d is None:
                stale.append(key)
            else:
                arr[nid] = _estimate_radius(cl_d, method="trim")
        elif key in RECOMPUTABLE_RADII:
            arr[nid] = _estimate_radius(d, method=key)
        else:
            stale.append(key)
    return stale


def _pack(nodes, radii_dict: dict, node2verts) -> tuple:
    """A skeleton's radii and vertex lists as ``_core``'s stages take them:
    ``(keys, radii (K, N), ptr, idx)``, the vertices as a CSR."""
    n = len(nodes)
    keys = list(radii_dict)
    radii = np.empty((len(keys), n), dtype=np.float64)
    for k, key in enumerate(keys):
        radii[k] = radii_dict[key]
    lens = [len(v) for v in node2verts]
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(lens)
    idx = (
        np.concatenate([np.asarray(v, dtype=np.int64).ravel() for v in node2verts])
        if n
        else np.empty(0, dtype=np.int64)
    )
    return keys, radii, ptr, idx


def _unpack(arrays, keys: list, radii_like: dict) -> tuple:
    """``(nodes, radii dict, node2verts, edges)`` back from a stage's arrays,
    each radius in the dtype it came in."""
    nodes, radii, ptr, idx, edges = arrays
    radii_dict = {
        key: radii[k].astype(np.asarray(radii_like[key]).dtype)
        for k, key in enumerate(keys)
    }
    node2verts = [idx[ptr[i] : ptr[i + 1]] for i in range(len(nodes))]
    return nodes, radii_dict, node2verts, edges


def _merge_near_soma_nodes(
    nodes: np.ndarray,
    radii_dict: dict[str, np.ndarray],
    edges: np.ndarray,
    node2verts: list[np.ndarray] | None,
    *,
    soma: Soma,
    radius_key: str,
    mesh_vertices: np.ndarray | None,
    inside_tol: float = 0.0,
    near_factor: float = 1.2,
    fat_factor: float = 0.20,
    log: Callable | None = None,
    with_vert2node: bool = True,
):
    """
    Collapse every skeleton node whose sphere overlaps the soma (stage 5).

    The routine performs two geometric tests on each node:

    • Inside test:     d_surface < −inside_tol
    • Near-and-fat:    d_center < near_factor·r_soma  AND  radius ≥ fat_factor·r_soma

    Nodes that satisfy either test are merged into node 0, bringing along their
    contributing mesh vertices so the soma can be re-fitted afterwards.  Any
    dangling leaves created by the merge are re-linked back to the soma.

    Parameters
    ----------
    nodes, radii_dict, node2verts, edges
        Skeleton arrays from :func:`skeletonize`.  ``node2verts`` may be *None*
        when mesh data is unavailable.
    soma
        Current spherical/ellipsoidal soma model (node 0).
    radius_key
        Which entry of ``radii_dict`` to use when checking the “fat” criterion.
    mesh_vertices
        Original mesh coordinates; required to re-fit the soma after merging.
        If *None*, vertex bookkeeping is skipped and the soma falls back to a
        spherical approximation with a warning.
    inside_tol, near_factor, fat_factor
        Dimensionless thresholds controlling the tests described above.
    log
        Optional callback used for verbose progress output.
    with_vert2node
        Whether to rebuild ``vert2node``; None in its place otherwise (for a
        caller that rebuilds it later anyway).

    Returns
    -------
    nodes_new, radii_new, node2verts_new, vert2node, edges_new, soma_new, old2new
        Updated arrays and a mapping from *old* node indices to the survivors.
        ``node2verts_new`` and ``vert2node`` are *None* when mesh data is
        unavailable.
    """
    if radius_key not in radii_dict:
        raise KeyError(radius_key)
    n = len(nodes)
    lists = node2verts if node2verts is not None else [np.empty(0, np.int64)] * n
    keys, radii, ptr, idx = _pack(nodes, radii_dict, lists)
    arrays, old2new, soma_verts, n_merged = kernels.skeletonize.merge_near_soma_nodes(
        np.asarray(nodes, dtype=np.float64),
        keys,
        radii,
        ptr,
        idx,
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        keys.index(radius_key),
        np.asarray(soma.center, dtype=np.float64),
        np.asarray(soma.axes, dtype=np.float64),
        np.asarray(soma.R, dtype=np.float64),
        None if soma.verts is None else np.asarray(soma.verts, dtype=np.int64),
        None if mesh_vertices is None else np.asarray(mesh_vertices, np.float64),
        float(inside_tol),
        float(near_factor),
        float(fat_factor),
    )
    if log:
        log(f"{n_merged} nodes merged into soma")
    nodes_keep, radii_keep, node2_keep, edges_out = _unpack(arrays, keys, radii_dict)
    if node2verts is None:
        node2_keep = None

    vert2node = (
        rebuild_vert2node(node2_keep)
        if node2_keep is not None and with_vert2node
        else None
    )

    if mesh_vertices is None:
        if log:
            log(
                "mesh_vertices not provided; cannot re-fit soma, "
                "falling back to spherical approximation."
            )
        soma = Soma.from_sphere(soma.center, soma.spherical_radius, verts=soma_verts)
    elif soma_verts is not None and soma_verts.size:
        try:
            soma = Soma.fit(mesh_vertices[soma_verts], verts=soma_verts)
        except ValueError:
            if log:
                log("Soma fitting failed, using spherical approximation instead.")
            soma = Soma.from_sphere(
                soma.center, soma.spherical_radius, verts=soma_verts
            )
    else:
        soma = dataclasses.replace(
            soma,
            center=soma.center.copy(),
            axes=soma.axes.copy(),
            R=soma.R.copy(),
            verts=soma_verts,
        )

    nodes_keep[0] = soma.center
    r_soma = soma.spherical_radius
    for k in radii_keep:
        radii_keep[k][0] = r_soma

    if log:
        centre_txt = ", ".join(f"{c:7.1f}" for c in soma.center)
        radii_txt = ",".join(f"{c:7.1f}" for c in soma.axes)
        log(f"Moved soma to [{centre_txt}]")
        log(f"(r = {radii_txt})")

    return (
        nodes_keep,
        radii_keep,
        node2_keep,
        vert2node,
        edges_out,
        soma,
        old2new,
    )


def _prune_neurites(
    nodes: np.ndarray,
    radii_dict: dict[str, np.ndarray],
    node2verts: list[np.ndarray],
    edges: np.ndarray,
    *,
    soma: Soma,
    mesh_vertices: np.ndarray | None,
    tip_extent_factor: float = 1.1,
    stem_extent_factor: float = 5.0,
    drop_single_node_branches: bool = True,
    log: Callable | None = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    np.ndarray,
    Soma,
    np.ndarray,
]:
    """
    Collapse obvious mesh-artefact branches into the soma (node 0) in
    two independent passes.
    """
    if mesh_vertices is None and log:
        log(
            "warning: mesh_vertices not provided; skipping soma/radius refits "
            "and reusing existing estimates."
        )
    keys, radii, ptr, idx = _pack(nodes, radii_dict, node2verts)
    arrays, old2new, soma_verts, n_merged = kernels.skeletonize.prune_neurites(
        np.asarray(nodes, dtype=np.float64),
        keys,
        radii,
        ptr,
        idx,
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(soma.center, dtype=np.float64),
        float(soma.spherical_radius),
        None if mesh_vertices is None else np.asarray(mesh_vertices, np.float64),
        float(tip_extent_factor),
        float(stem_extent_factor),
        bool(drop_single_node_branches),
    )
    if n_merged and log:
        log(f"Merged {n_merged} peri-soma nodes into soma ")

    if mesh_vertices is not None and soma_verts.size:
        pts = mesh_vertices[soma_verts]
        try:
            soma = soma.fit(pts, verts=soma_verts)
        except ValueError:
            if log:
                log("Soma fitting failed, using previous parameters instead.")
            soma = Soma(
                center=soma.center.copy(),
                axes=soma.axes.copy(),
                R=soma.R.copy(),
                verts=soma_verts if soma_verts.size else None,
            )
    else:
        # keep the existing ellipsoid parameters; verts already updated above
        soma = Soma(
            center=soma.center.copy(),
            axes=soma.axes.copy(),
            R=soma.R.copy(),
            verts=soma_verts if soma_verts.size else None,
        )

    if log:
        centre_txt = ", ".join(f"{c:7.1f}" for c in soma.center)
        radii_txt = ",".join(f"{c:7.1f}" for c in soma.axes)
        log(f"Moved soma to [{centre_txt}]")
        log(f"(r = {radii_txt})")

    nodes_new, radii_new, node2verts_new, edges_new = _unpack(arrays, keys, radii_dict)
    vert2node = rebuild_vert2node(node2verts_new)
    return nodes_new, radii_new, node2verts_new, vert2node, edges_new, soma, old2new


def _bridge_gaps(
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    bridge_max_factor: float | None = None,
    bridge_recalc_after: int | None = None,
) -> np.ndarray:
    """
    Bridge all disconnected surface components of a neuron mesh **back to the
    soma component** by inserting synthetic edges.

    The routine works in four logical stages:

    1.  **Component analysis** – build an undirected graph of the mesh,
        identify connected components, and mark the one that contains the
        soma (vertex 0) as the *island*.
    2.  **Gap prioritisation** – for every *foreign* component find the
        geodesically closest vertex pair (component ↔ island) and push the
        tuple ``(gap_distance, cid, idx_comp, idx_island)`` into a
        min-heap.
        If *bridge_max_factor* is *None* we estimate a conservative upper
        bound from the initial gap distribution:

        ``factor = clip( 55-th percentile(gaps) / ⟨edge⟩ , [6 ×, 12 ×] )``

        This filters out pathologically long jumps right from the start.
    3.  **Greedy growth** – repeatedly pop the nearest component from the
        heap and connect it with **one** synthetic edge (the cached closest
        pair).  After each merge the island KD-tree is rebuilt and the heap
        entries are refreshed every *bridge_recalc_after* merges
        (auto-chosen if *None*; ≈ 5 % of remaining gaps, capped at 32).
        A stall counter and a gentle *relax_factor* (1.5) guarantee
        termination even on meshes with extremely uneven gap sizes.
    4.  **Finish** – return the original edges plus all new bridges,
        sorted and de-duplicated.

    Notes
    -----
    * Only **one** edge per foreign component is added; the global MST step
      later will prune any redundant cycles that could arise.
    * Complexity is dominated by KD-tree queries:
      *O((|V_island| + Σ|V_comp|) log |V|)* in practice.
    * The heuristic defaults trade a few hundred ms of runtime for a markedly
      lower rate of “long-jump” bridges.  Power users can override
      *bridge_max_factor* or *bridge_recalc_after* if desired.

    Parameters
    ----------
    nodes
        ``(N, 3)`` float64 array of mesh-vertex coordinates.
    edges
        ``(E, 2)`` int64 array of **undirected, sorted** mesh edges.
    bridge_max_factor
        Optional hard ceiling for acceptable bridge length expressed as a
        multiple of the mean mesh-edge length.  If *None* an adaptive value
        (see above) is chosen.
    bridge_recalc_after
        How many successful merges to perform before all component-to-island
        distances are recomputed.  If *None* an adaptive value based on the
        number of gaps is used.

    Returns
    -------
    np.ndarray
        ``(E′, 2)`` int64 undirected edge list containing the original mesh
        edges **plus** one synthetic edge for every formerly disconnected
        component.  The array is sorted row-wise and de-duplicated.
    """

    return kernels.skeletonize.bridge_gaps(
        np.asarray(nodes, dtype=np.float64),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        None if bridge_max_factor is None else float(bridge_max_factor),
        None if bridge_recalc_after is None else int(bridge_recalc_after),
    )


def _merge_single_node_branches(
    nodes: np.ndarray,
    radii_dict: dict[str, np.ndarray],
    node2verts: list[np.ndarray],
    edges: np.ndarray,
    *,
    mesh_vertices: np.ndarray | None,
    min_parent_degree: int = 3,
    return_mapping: bool = False,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    np.ndarray,
    np.ndarray | None,
]:
    """
    Iteratively merge every leaf whose *parent* has degree ≥ `min_parent_degree`.
    Terminates when no such leaves are left.
    """
    keys, radii, ptr, idx = _pack(nodes, radii_dict, node2verts)
    arrays, kept = kernels.skeletonize.merge_single_node_branches(
        np.asarray(nodes, dtype=np.float64),
        keys,
        radii,
        ptr,
        idx,
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        None if mesh_vertices is None else np.asarray(mesh_vertices, np.float64),
        int(min_parent_degree),
    )
    nodes_new, radii_new, node2verts_new, edges_new = _unpack(arrays, keys, radii_dict)
    vert2node = rebuild_vert2node(node2verts_new)
    return (
        nodes_new,
        radii_new,
        node2verts_new,
        vert2node,
        edges_new,
        kept if return_mapping else None,
    )


def _detect_soma(
    nodes: np.ndarray,
    radii: dict[str, np.ndarray],
    node2verts: list[np.ndarray],
    vert2node: dict[int, int],
    *,
    soma_radius_percentile_threshold: float,
    soma_radius_distance_factor: float,
    soma_min_nodes: int,
    detect_soma: bool,
    radius_key: str,
    mesh_vertices: np.ndarray | None,
    log: Callable[[str], None] | None = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    Soma,
    bool,
    np.ndarray,
]:
    """
    Detect the soma cluster and enforce node 0 as its centroid/root.

    Returns
    -------
    nodes, radii, node2verts, vert2node, soma, has_soma, old2new
        Updated arrays (same objects passed in) plus the permutation that maps
        old node IDs → new node IDs (identity when unchanged).
    """
    n_nodes = len(nodes)
    if not node2verts:
        node2verts[:] = [np.empty(0, dtype=np.int64) for _ in range(n_nodes)]
    vert2node = vert2node or {}

    state = SkeletonState(
        nodes=nodes,
        radii=radii,
        edges=np.empty((0, 2), dtype=np.int64),
        node2verts=node2verts,
        vert2node=vert2node,
    )

    def _identity() -> np.ndarray:
        return np.arange(n_nodes, dtype=np.int64)

    def _soma_verts(idx: int) -> np.ndarray | None:
        if idx < len(node2verts) and node2verts[idx].size:
            return node2verts[idx]
        return None

    if not detect_soma:
        verts0 = _soma_verts(0)
        soma = Soma.from_sphere(
            nodes[0],
            radii[radius_key][0],
            verts=verts0 if verts0 is not None and verts0.size else None,
        )
        return nodes, radii, node2verts, vert2node, soma, False, _identity()

    soma_est, soma_nodes, has_soma = _find_soma(
        nodes,
        radii[radius_key],
        pct_large=soma_radius_percentile_threshold,
        dist_factor=soma_radius_distance_factor,
        min_keep=soma_min_nodes,
    )
    if not has_soma:
        if log:
            log("no soma detected → keeping old root")
        verts0 = _soma_verts(0)
        soma = Soma.from_sphere(
            nodes[0],
            radii[radius_key][0],
            verts=verts0 if verts0 is not None and verts0.size else None,
        )
        return nodes, radii, node2verts, vert2node, soma, False, _identity()

    old2new = _identity()

    # the soma's vertices, before a soma node moves to node 0
    lists = [np.asarray(node2verts[int(i)], dtype=np.int64) for i in soma_nodes]
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(v) for v in lists])
    soma_vert_ids = kernels.skeletonize.soma_vertices(
        np.asarray(nodes[soma_nodes], dtype=np.float64),
        ptr,
        np.concatenate(lists) if lists else np.empty(0, dtype=np.int64),
        np.asarray(soma_est.center, dtype=np.float64),
        float(soma_est.spherical_radius),
    )

    if 0 not in soma_nodes:
        new_root = int(soma_nodes[np.argmax(radii[radius_key][soma_nodes])])
        swap_nodes(state, 0, new_root)
        old2new[0], old2new[new_root] = old2new[new_root], old2new[0]

    soma_est.verts = soma_vert_ids if soma_vert_ids.size else None

    nodes[0] = soma_est.center
    r_sphere = soma_est.spherical_radius
    for k in radii:
        radii[k][0] = r_sphere

    if mesh_vertices is None:
        if log:
            log(
                "mesh_vertices not provided; cannot re-fit soma, "
                "falling back to spherical approximation."
            )
        soma_est = Soma.from_sphere(soma_est.center, r_sphere, verts=soma_est.verts)
    elif soma_est.verts is not None and soma_est.verts.size:
        try:
            soma_est = Soma.fit(mesh_vertices[soma_est.verts], verts=soma_est.verts)
        except ValueError:
            if log:
                log("Soma fitting failed, using spherical approximation instead.")
            soma_est = Soma.from_sphere(soma_est.center, r_sphere, verts=soma_est.verts)

    if log:
        centre_txt = ", ".join(f"{c:7.1f}" for c in soma_est.center)
        radii_txt = ",".join(f"{c:7.1f}" for c in soma_est.axes)
        log(f"Found soma at [{centre_txt}]")
        log(f"(r = {radii_txt})")

    return nodes, radii, node2verts, vert2node, soma_est, True, old2new
