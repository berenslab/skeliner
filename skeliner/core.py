"""Reusable skeleton helpers shared across the package."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Callable, List

import igraph as ig
import numpy as np
from scipy.spatial import KDTree

from .dataclass import Soma

__all__ = [
    "_find_soma",
    "_bfs_parents",
    "_build_mst",
    "_bridge_gaps",
    "_merge_near_soma_nodes",
    "_merge_single_node_branches",
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

    large_thresh = np.percentile(radii, pct_large)
    cand_idx = np.where(radii >= large_thresh)[0]
    if cand_idx.size == 0:
        raise RuntimeError(
            f"no nodes above the {pct_large:g}-th percentile (try lowering pct_large)"
        )

    idx_max = int(np.argmax(radii))
    R_max = radii[idx_max]

    dists = np.linalg.norm(nodes[cand_idx] - nodes[idx_max], axis=1)
    keep = dists <= dist_factor * R_max
    soma_idx = cand_idx[keep]
    has_soma = soma_idx.size >= min_keep

    if not has_soma:
        return Soma.from_sphere(nodes[idx_max], R_max, None), soma_idx, has_soma

    soma = Soma.from_sphere(
        center=nodes[soma_idx].mean(0),
        radius=R_max,
        verts=None,
    )
    return soma, soma_idx, has_soma


def _bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> List[int]:
    """Return parent[] array of BFS tree from *root* given an undirected edge list."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    parent = [-1] * n_nodes
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v != root and parent[v] == -1:
                parent[v] = u
                q.append(v)
    return parent


def _build_mst(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the edge list of the global minimum-spanning tree."""
    g = ig.Graph(
        n=len(nodes), edges=[tuple(map(int, e)) for e in edges], directed=False
    )
    g.es["weight"] = [float(np.linalg.norm(nodes[a] - nodes[b])) for a, b in edges]
    mst = g.spanning_tree(weights="weight")
    return np.asarray(
        sorted(tuple(sorted(e)) for e in mst.get_edgelist()), dtype=np.int64
    )


def _estimate_radius(
    d: np.ndarray,
    *,
    method: str = "median",
    trim_fraction: float = 0.05,
    q: float = 0.90,
) -> float:
    """Return one scalar radius according to *method*."""
    if method == "median":
        return float(np.median(d))
    if method == "mean":
        return float(d.mean())
    if method == "max":
        return float(d.max())
    if method == "min":
        return float(d.min())
    if method == "trim":
        lo, hi = np.quantile(d, [trim_fraction, 1.0 - trim_fraction])
        mask = (d >= lo) & (d <= hi)
        if not np.any(mask):
            return float(np.mean(d))
        return float(d[mask].mean())
    raise ValueError(f"Unknown radius estimator '{method}'.")


def _merge_near_soma_nodes(
    nodes: np.ndarray,
    radii_dict: dict[str, np.ndarray],
    edges: np.ndarray,
    node2verts: list[np.ndarray],
    *,
    soma: Soma,
    radius_key: str,
    mesh_vertices: np.ndarray,
    inside_tol: float = 0.0,
    near_factor: float = 1.2,
    fat_factor: float = 0.20,
    log: Callable | None = None,
):
    """Shared collapse of near-soma nodes (ported from skeletonize)."""
    r_soma = soma.spherical_radius
    r_primary = radii_dict[radius_key]
    d2s = soma.distance(nodes, to="surface")
    d2c = soma.distance(nodes, to="center")
    inside = d2s < -inside_tol
    near = d2c < near_factor * r_soma
    fat = r_primary > fat_factor * r_soma

    keep_mask = ~(inside | (near & fat))
    keep_mask[0] = True
    merged_idx = np.where(~keep_mask)[0]

    if log:
        log(f"{merged_idx.size} nodes merged into soma")

    old2new = -np.ones(len(nodes), np.int64)
    old2new[np.where(keep_mask)[0]] = np.arange(keep_mask.sum())

    a, b = edges.T
    both_keep = keep_mask[a] & keep_mask[b]
    edges_out = np.unique(np.sort(old2new[edges[both_keep]], 1), axis=0)

    leaves = {
        int(old2new[u if keep_mask[u] else v])
        for u, v in edges
        if keep_mask[u] ^ keep_mask[v]
    }
    if leaves:
        edges_out = np.vstack(
            [edges_out, np.array([[0, i] for i in sorted(leaves)], dtype=np.int64)]
        )

    nodes_keep = nodes[keep_mask].copy()
    radii_keep = {k: v[keep_mask].copy() for k, v in radii_dict.items()}
    node2_keep = [node2verts[i] for i in np.where(keep_mask)[0]]
    vert2node = {}

    if merged_idx.size:
        w = np.array(
            [len(node2verts[0]), *[len(node2verts[i]) for i in merged_idx]],
            dtype=np.float64,
        )
        nodes_keep[0] = np.average(
            np.vstack([nodes[0], nodes[merged_idx]]), axis=0, weights=w
        )
        for idx in merged_idx:
            vidx = node2verts[idx]
            d_local = soma.distance_to_surface(mesh_vertices[vidx])
            close = d_local < near_factor * r_soma
            if np.any(close):
                soma.verts = (
                    np.concatenate((soma.verts, vidx[close]))
                    if soma.verts is not None
                    else vidx[close]
                )
                node2_keep[0] = np.concatenate((node2_keep[0], vidx[close]))

    for nid, verts in enumerate(node2_keep):
        for v in verts:
            vert2node[int(v)] = nid

    soma.verts = (
        np.unique(soma.verts).astype(np.int64) if soma.verts is not None else None
    )
    try:
        soma = Soma.fit(mesh_vertices[soma.verts], verts=soma.verts)
    except ValueError:
        if log:
            log("Soma fitting failed, using spherical approximation instead.")
        soma = Soma.from_sphere(soma.center, soma.spherical_radius, verts=soma.verts)

    nodes_keep[0] = soma.center
    r_soma = soma.spherical_radius
    for k in radii_keep:
        radii_keep[k][0] = r_soma

    if log:
        centre_txt = ", ".join(f"{c:7.1f}" for c in soma.center)
        radii_txt = ",".join(f"{c:7.1f}" for c in soma.axes)
        log(f"Moved soma to [{centre_txt}]")
        log(f"(r = {radii_txt})")

    return (nodes_keep, radii_keep, node2_keep, vert2node, edges_out, soma)


def _bridge_gaps(
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    bridge_max_factor: float | None = None,
    bridge_recalc_after: int | None = None,
) -> np.ndarray:
    """Shared helper to bridge disconnected components."""

    def _auto_bridge_max(
        gaps: list[float],
        edge_mean: float,
        *,
        pct: float = 55.0,
        lo: float = 6.0,
        hi: float = 12.0,
    ) -> float:
        raw = np.percentile(gaps, pct) / edge_mean
        return float(np.clip(raw, lo, hi))

    def _auto_recalc_after(n_gaps: int) -> int:
        if n_gaps <= 10:
            return 2
        if n_gaps <= 50:
            return 4
        if n_gaps <= 200:
            return max(4, n_gaps // 20)
        return 32

    g = ig.Graph(
        n=len(nodes), edges=[tuple(map(int, e)) for e in edges], directed=False
    )
    comps = [set(c) for c in g.components()]
    comp_idx = {
        cid: np.fromiter(verts, dtype=np.int64) for cid, verts in enumerate(comps)
    }
    soma_cid = g.components().membership[0]
    if len(comps) == 1:
        return edges

    edge_len_mean = np.linalg.norm(
        nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1
    ).mean()

    island = set(comps[soma_cid])
    island_idx = np.fromiter(island, dtype=np.int64)
    island_tree = KDTree(nodes[island_idx])

    def closest_pair(cid: int) -> tuple[float, int, int]:
        pts = nodes[comp_idx[cid]]
        dists, idx_is = island_tree.query(pts, k=1, workers=-1)
        best = int(np.argmin(dists))
        return float(dists[best]), best, int(idx_is[best])

    gaps = []
    pq = []
    for cid in range(len(comps)):
        if cid == soma_cid:
            continue
        gap, best_c, best_i = closest_pair(cid)
        gaps.append(gap)
        pq.append((gap, cid, best_c, best_i))

    if not gaps:
        return edges

    if bridge_max_factor is None:
        bridge_max_factor = _auto_bridge_max(gaps, edge_len_mean)
    else:
        bridge_max_factor = float(bridge_max_factor)

    if bridge_recalc_after is None:
        recalc_after = _auto_recalc_after(len(comps) - 1)
    else:
        recalc_after = int(bridge_recalc_after)

    heapq.heapify(pq)
    edges_new: list[tuple[int, int]] = []
    merges_since_recalc = 0
    stall = 0
    relax_factor = 1.5
    max_stall = 3 * len(pq)
    current_max = bridge_max_factor * edge_len_mean

    while pq:
        _, cid, _, _ = heapq.heappop(pq)
        gap, best_c, best_i = closest_pair(cid)
        if gap > current_max:
            heapq.heappush(pq, (gap, cid, best_c, best_i))
            stall += 1
            if stall >= 2 * max_stall:
                pq = [
                    (g2, cid2, bc, bi)
                    for _, cid2, _, _ in pq
                    for g2, bc, bi in (closest_pair(cid2),)
                ]
                heapq.heapify(pq)
                current_max *= relax_factor
                stall = 0
            continue

        stall = 0
        verts_idx = comp_idx[cid]
        u = int(verts_idx[best_c])
        v = int(island_idx[best_i])
        edges_new.append((u, v))

        island |= comps[cid]
        island_idx = np.concatenate([island_idx, verts_idx])
        island_tree = KDTree(nodes[island_idx])

        merges_since_recalc += 1
        if merges_since_recalc >= recalc_after:
            pq = [
                (gap, cid, b_comp, b_is)
                for _, cid, _, _ in pq
                for gap, b_comp, b_is in (closest_pair(cid),)
            ]
            heapq.heapify(pq)
            merges_since_recalc = 0

    if edges_new:
        edges_aug = np.vstack([edges, np.asarray(edges_new, dtype=np.int64)])
    else:
        edges_aug = edges

    return np.unique(edges_aug, axis=0)


def _merge_single_node_branches(
    nodes: np.ndarray,
    radii_dict: dict[str, np.ndarray],
    node2verts: list[np.ndarray],
    edges: np.ndarray,
    *,
    mesh_vertices: np.ndarray,
    min_parent_degree: int = 3,
    estimate_radius: Callable[..., float] | None = None,
) -> tuple[
    np.ndarray, dict[str, np.ndarray], list[np.ndarray], dict[int, int], np.ndarray
]:
    """Shared helper to merge single-node branches (ported from skeletonize)."""
    if estimate_radius is None:
        estimate_radius = _estimate_radius
    while True:
        deg = np.zeros(len(nodes), dtype=int)
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1

        leaves = [i for i, d in enumerate(deg) if d == 1 and i != 0]
        parent = _bfs_parents(edges, len(nodes), root=0)
        singles = [leaf for leaf in leaves if deg[parent[leaf]] >= min_parent_degree]

        if not singles:
            break

        to_drop = set()
        for leaf in singles:
            par = parent[leaf]
            node2verts[par] = np.concatenate((node2verts[par], node2verts[leaf]))
            pts = mesh_vertices[node2verts[par]]
            d = np.linalg.norm(pts - nodes[par], axis=1)
            for k in radii_dict:
                radii_dict[k][par] = estimate_radius(d, method=k)
            to_drop.add(leaf)

        keep = np.ones(len(nodes), bool)
        keep[list(to_drop)] = False
        remap = {old: new for new, old in enumerate(np.where(keep)[0])}

        nodes = nodes[keep]
        node2verts = [node2verts[i] for i in np.where(keep)[0]]
        radii_dict = {k: v[keep].copy() for k, v in radii_dict.items()}
        edges = np.asarray(
            [(remap[a], remap[b]) for a, b in edges if keep[a] and keep[b]],
            dtype=np.int64,
        )

    vert2node = {int(v): i for i, vs in enumerate(node2verts) for v in vs}
    return nodes, radii_dict, node2verts, vert2node, edges
