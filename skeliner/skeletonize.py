import heapq
import time
from contextlib import contextmanager
from importlib import metadata as _metadata
from typing import Any, Callable, Dict, List, Tuple

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

from .core import _bfs_parents, _build_mst, _find_soma
from .dataclass import Skeleton, Soma

_SKELINER_VERSION = _metadata.version("skeliner")


__all__ = [
    "skeletonize",
]


# -----------------------------------------------------------------------------
#  Graph helpers
# -----------------------------------------------------------------------------


def _surface_graph(mesh: trimesh.Trimesh) -> ig.Graph:
    """Return an edge‑weighted triangle‑adjacency graph.

    The graph has one vertex per mesh‑vertex and an undirected edge for every
    unique mesh edge.  Edge weights are the Euclidean lengths which later serve
    as geodesic distances.
    """
    edges = [tuple(map(int, e)) for e in mesh.edges_unique]
    g = ig.Graph(n=len(mesh.vertices), edges=edges, directed=False)
    g.es["weight"] = mesh.edges_unique_length.astype(float).tolist()
    return g


def _dist_vec_for_component(
    gsurf: ig.Graph,
    verts: np.ndarray,  # 1-D int64 array of vertex IDs (one component)
    seed_vid: int,  # mesh-vertex ID, must be in *verts*
) -> np.ndarray:
    """
    Return the distance vector *d[verts[i]]* from *seed_vid* to every
    vertex in this component, **without touching the rest of the mesh**.
    """
    # Build a dedicated sub-graph (much smaller than gsurf)
    sub = gsurf.induced_subgraph(verts, implementation="create_from_scratch")

    # Map the seed’s mesh-vertex ID → its local index in *sub*
    root_idx = int(np.where(verts == seed_vid)[0][0])

    # igraph returns shape (1, |verts|); squeeze to 1-D
    return sub.distances(
        source=[root_idx],
        weights="weight",
    )[0]


def _geodesic_bins(dist_dict: Dict[int, float], step: float) -> List[List[int]]:
    """Bucket mesh vertices into concentric geodesic shells."""
    if not dist_dict:
        return []

    # --- vectorise keys & distances ------------------------------------
    vids = np.fromiter(dist_dict.keys(), dtype=np.int64)
    dists = np.fromiter(dist_dict.values(), dtype=np.float64)

    # --- construct right-open bin edges --------------------------------
    edges = np.arange(0.0, dists.max() + step, step, dtype=np.float64)
    if edges[-1] <= dists.max():  # ensure last edge is strictly greater
        edges = np.append(edges, edges[-1] + step)

    # --- assign each vertex to a shell ---------------------------------
    idx = np.digitize(dists, edges) - 1  # 0-based indices
    idx[idx == len(edges) - 1] -= 1  # clip the “equal-max” case

    # --- build the bins -------------------------------------------------
    bins = [[] for _ in range(len(edges) - 1)]
    for vid, b in zip(vids, idx):
        bins[b].append(int(vid))

    return bins


def _split_comp_if_elongated(
    comp_idx: np.ndarray,
    v: np.ndarray,
    *,
    aspect_thr: float = 2.0,  # “acceptable” λ1 / λ2
    min_shell_vertices: int = 6,
    max_vertices_per_slice: int | None = None,
):
    """
    Yield 1–k vertex arrays after optional PCA-based splitting.

    •  If λ1/λ2 ≤ aspect_thr  → keep the component intact.
    •  Otherwise slice it into ⌈λ1/λ2 / aspect_thr⌉ roughly equal chunks.

    The automatic rule guarantees that **every resulting slice will have
    an aspect ratio ≤ aspect_thr** (plus a small safety margin).
    """

    if comp_idx.size < min_shell_vertices:
        yield comp_idx
        return

    # ── fast 3-D PCA ----------------------------------------------------
    pts = v[comp_idx].astype(np.float64)
    cov = np.cov(pts, rowvar=False)
    evals, vec = np.linalg.eigh(cov)  # ascending order
    elong = evals[-1] / (evals[-2] + 1e-9)

    if elong <= aspect_thr:
        yield comp_idx
        return

    # ── how many slices?  automatic & bounded  --------------------------
    n_split = int(np.ceil(elong / aspect_thr))

    # 1. never make more slices than vertices allow
    n_split = min(n_split, comp_idx.size // min_shell_vertices)

    # 2. optional extra guard: cap by absolute slice size
    if max_vertices_per_slice is not None:
        n_split = min(n_split, int(np.ceil(comp_idx.size / max_vertices_per_slice)))

    if n_split <= 1:
        yield comp_idx
        return

    # ── 1-D k-means via quantile cuts  ----------------------------------
    axis = vec[:, -1]  # major axis (unit vector)
    proj = pts @ axis  # scalar coordinate
    cuts = np.quantile(proj, np.linspace(0, 1, n_split + 1))

    for lo, hi in zip(cuts[:-1], cuts[1:]):
        m = (proj >= lo) & (proj <= hi)
        if m.sum() >= min_shell_vertices:
            yield comp_idx[m]


def _bin_geodesic_shells(
    mesh: trimesh.Trimesh,
    gsurf: ig.Graph,
    *,
    soma: Soma,
    step_size: float | None = None,
    target_shell_count: int = 500,
    min_shell_vertices: int = 6,
    max_shell_width_factor: float = 50.0,
    # -- split elongated shells (optional) ------------------------
    split_elongated_shells: bool = True,
    split_aspect_thr: float = 3.0,  # λ1 / λ2
    split_min_shell_vertices: int = 50,  # minimum size of a cluster to split
    split_max_vertices_per_slice: int | None = None,  # max size of a slice
) -> List[List[np.ndarray]]:
    """
    Cluster every connected surface patch into sets of *geodesic shells*.

    The function reproduces the logic that used to live inline in
    :pyfunc:`skeletonize`, but is now reusable and unit-testable.

    Parameters
    ----------
    mesh
        The watertight neuron mesh.
    gsurf
        Undirected triangle-adjacency graph of the mesh (from
        :pyfunc:`_surface_graph`).
    c_soma
        3-vector of the soma centroid **chosen earlier**.
        (Its exact origin depends on `detect_soma`.)
    soma_verts
        Set of mesh-vertex IDs that belong to the detected soma patch (may be a
        singleton `{seed_vid}` if soma detection is deferred or disabled).
    target_shell_count
        Requested number of shells per connected component.  The actual width
        is adapted to mesh resolution.
    min_shell_vertices
        Discard clusters smaller than this size; they usually represent noise.
    max_shell_width_factor
        Upper limit for the shell width expressed as a multiple of the mean
        mesh-edge length.  Prevents *very* sparse meshes from producing a
        single giant shell.

    Returns
    -------
    List[List[np.ndarray]]
        Outer list = shells ordered by growing distance;
        inner list = connected vertex clusters inside that shell;
        each cluster is a 1-D ``int64`` array of mesh-vertex IDs.

        The structure is exactly what stage 2 of *skeletonize()* expects.
    """
    v = mesh.vertices.view(np.ndarray)
    e_m = float(mesh.edges_unique_length.mean())  # mean mesh-edge length

    c_soma = soma.center
    soma_verts = set() if soma.verts is None else set(map(int, soma.verts))
    soma_vids = np.fromiter(soma_verts, dtype=np.int64)

    # ------------------------------------------------------------------
    # build a vertex list for every connected surface patch
    # ------------------------------------------------------------------
    comp_vertices = [np.asarray(c, dtype=np.int64) for c in gsurf.components()]
    all_shells: List[List[np.ndarray]] = []

    for cid, verts in enumerate(comp_vertices):
        # --------------------------------------------------------------
        # choose one seed *per component* – deterministic but cheap
        # --------------------------------------------------------------
        if np.intersect1d(verts, soma_vids).size:
            # component that contains (part of) the soma ➜
            # pick the *furthest* soma vertex from the centroid to avoid
            # degeneracy when the soma spans many shells
            seed_vid = int(
                soma_vids[np.argmax(np.linalg.norm(v[soma_vids] - c_soma, axis=1))]
            )
        else:
            # foreign island ➜ pick a pseudo-random, yet deterministic vertex
            seed_vid = int(verts[hash(cid) % len(verts)])

        # --------------------------------------------------------------
        # geodesic distance of *all* vertices in this component
        # --------------------------------------------------------------
        dist_vec = _dist_vec_for_component(gsurf, verts, seed_vid)
        dist_sub = {int(vid): float(d) for vid, d in zip(verts, dist_vec)}

        if not dist_sub:
            continue

        # shell width: max(edge × 2, arc_len / target_shell_count)
        if step_size is None:
            arc_len = max(dist_sub.values())
            step = max(e_m * 2.0, arc_len / target_shell_count)
        else:
            step = float(step_size)

        # ------------------------------------------------------------------
        # increase the step until we get at least one non-empty shell
        # (avoids pathological meshes with *too* fine resolution)
        # ------------------------------------------------------------------
        shells: List[List[int]] = []
        while not any(shells) and step < e_m * max_shell_width_factor:
            shells = _geodesic_bins(dist_sub, step)
            step *= 1.5

        # --------------------------------------------------------------
        # for every shell: split it into connected sub-clusters
        # --------------------------------------------------------------
        for shell_verts in shells:
            # exclude explicit soma vertices to keep the center clean
            inner = [vid for vid in shell_verts if vid not in soma_verts]
            if not inner:
                continue

            sub = gsurf.induced_subgraph(inner)
            comps = []
            for comp in sub.components():
                if len(comp) < min_shell_vertices:
                    continue  # too small ➜ ignore
                comp_idx = np.fromiter((inner[i] for i in comp), dtype=np.int64)
                if (
                    split_elongated_shells and len(comp) < 1500
                ):  # hard-coded for now, if too large, might be a soma
                    for part in _split_comp_if_elongated(
                        comp_idx,
                        v,
                        aspect_thr=split_aspect_thr,
                        min_shell_vertices=split_min_shell_vertices,
                        max_vertices_per_slice=split_max_vertices_per_slice,
                    ):
                        comps.append(part)
                else:
                    comps.append(comp_idx)

            all_shells.append(comps)

    return all_shells


def _estimate_radius(
    d: np.ndarray,
    *,
    method: str = "median",
    trim_fraction: float = 0.05,
    q: float = 0.90,
) -> float:
    """Return one scalar radius according to *method*.

    Methods
    -------
    median          : 50‑th percentile (robust default)
    mean            : arithmetic mean (biased by tails)
    trim            : mean after trimming *trim_fraction* at both ends
    qXX / pXX       : upper quantile XX given as integer, e.g. "q90" or "p75"
    """
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
        return float(d[(d >= lo) & (d <= hi)].mean())
    raise ValueError(f"Unknown radius estimator '{method}'.")


def _edges_from_mesh(
    edges_unique: np.ndarray,  # (E, 2) int64
    v2n: dict[int, int],  # mesh-vertex id -> skeleton node id
    n_mesh_verts: int,
) -> np.ndarray:
    """
    Vectorised remap of mesh edges -> skeleton edges.
    """
    # 1. build an int64 lookup table  mesh_vid -> node_id  (-1 if absent)
    lut = np.full(n_mesh_verts, -1, dtype=np.int64)
    lut[list(v2n.keys())] = list(v2n.values())

    # 2. map both columns in one shot
    a, b = edges_unique.T  # views, no copy
    na, nb = lut[a], lut[b]  # vectorised gather

    # 3. keep edges whose *both* endpoints exist and are different
    mask = (na >= 0) & (nb >= 0) & (na != nb)
    na, nb = na[mask], nb[mask]

    edges = np.vstack([na, nb]).T
    edges = np.sort(edges, axis=1)  # canonical order
    edges = np.unique(edges, axis=0)  # drop duplicates
    return edges.astype(np.int64)  # copy to new array


# -----------------------------------------------------------------------
#  Post-processing helpers
# -----------------------------------------------------------------------


def _merge_near_soma_nodes(
    nodes: np.ndarray,
    radii_dict: dict[str, np.ndarray],
    edges: np.ndarray,
    node2verts: list[np.ndarray],
    *,
    soma: Soma,
    radius_key: str,
    mesh_vertices: np.ndarray,
    # ---- new, all relative to the fitted spherical radius -------------
    inside_tol: float = 0.0,  # anything with d < 0 − tol is *inside*
    near_factor: float = 1.2,  # “near” if d < near_factor × r_soma
    fat_factor: float = 0.20,  # “fat” if r ≥ fat_factor  × r_soma
    log: Callable | None = None,
):
    """
    Collapse every skeleton node whose sphere would overlap the soma
    **according to strictly geometric tests based on `Soma.distance_`.**

    *Inside test*
        `d < −inside_tol`  (negative ➜ center is strictly inside)

    *Near-and-fat test*
        `d <  near_factor × r_soma   AND   r ≥ fat_factor × r_soma`

    All factors are *dimensionless* and therefore unit-safe.
    """
    # ------------------------------------------------------------------
    r_soma = soma.spherical_radius
    r_primary = radii_dict[radius_key]
    d2s = soma.distance(nodes, to="surface")  # signed distance
    d2c = soma.distance(nodes, to="center")  # unsigned distance
    inside = d2s < -inside_tol
    near = d2c < near_factor * r_soma
    fat = r_primary > fat_factor * r_soma

    keep_mask = ~(inside | (near & fat))
    keep_mask[0] = True  # never drop the soma vertex
    merged_idx = np.where(~keep_mask)[0]

    if log:
        log(f"{merged_idx.size} nodes merged into soma")

    # ---- remainder is identical to your original routine -------------
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
            # keep only the vertex subset that is *really* close to the soma
            vidx = node2verts[idx]
            d_local = soma.distance_to_surface(mesh_vertices[vidx])
            close = d_local < near_factor * r_soma  # same factor as for center test
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
        # fallback to spherical approximation
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

    def _auto_bridge_max(
        gaps: list[float],
        edge_mean: float,
        *,
        pct: float = 55.0,
        lo: float = 6.0,
        hi: float = 12.0,
    ) -> float:
        """
        Choose a bridge_max_factor from the initial gap distribution. Default is the
        55th percentile of the gap distribution, clipped to [6, 20] times the mean edge
        """
        raw = np.percentile(gaps, pct) / edge_mean
        return float(np.clip(raw, lo, hi))

    def _auto_recalc_after(n_gaps: int) -> int:
        """Return a suitable recalc period for the given number of gaps."""
        if n_gaps <= 10:  # tiny: update often
            return 2
        if n_gaps <= 50:  # small: every 3–4 merges
            return 4
        if n_gaps <= 200:  # medium: every ~5 % of gaps
            return max(4, n_gaps // 20)
        # giant meshes: cap to 32 so we never starve
        return 32

    # -- 0. quick exit if already connected ---------------------------------
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

    # -- 1. build one KD-tree per component ---------------------------------
    edge_len_mean = np.linalg.norm(
        nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1
    ).mean()

    # -- 2. grow the island using a distance-ordered priority queue ---------
    island = set(comps[soma_cid])
    island_idx = np.fromiter(island, dtype=np.int64)
    island_tree = KDTree(nodes[island_idx])

    # helper to compute the *current* closest gap of a component
    def closest_pair(cid: int) -> tuple[float, int, int]:
        pts = nodes[comp_idx[cid]]
        dists, idx_is = island_tree.query(pts, k=1, workers=-1)
        best = int(np.argmin(dists))
        return float(dists[best]), best, np.asarray(idx_is, dtype=np.int64)[best]

    # priority queue of (gap_distance, cid, best_comp_idx, best_island_idx)
    pq = []
    gap_samples = []
    for cid in range(len(comps)):
        if cid == soma_cid:
            continue
        gap, b_comp, b_is = closest_pair(cid)
        pq.append((gap, cid, b_comp, b_is))
        gap_samples.append(gap)
    heapq.heapify(pq)

    # -- heuristic hyperparameters if not given -------------------------------
    if bridge_max_factor is None:
        bridge_max_factor = _auto_bridge_max(gap_samples, edge_len_mean)

    if bridge_recalc_after is None:
        gaps = len(comps) - 1
        recalc_after = _auto_recalc_after(gaps)
    else:
        recalc_after = bridge_recalc_after

    edges_new: list[tuple[int, int]] = []
    merges_since_recalc = 0

    stall = 0
    relax_factor = 1.5
    max_stall = 3 * len(pq)
    current_max = bridge_max_factor * edge_len_mean

    while pq:
        _, cid, _, _ = heapq.heappop(pq)
        # postpone if the component is still too far away
        gap, best_c, best_i = closest_pair(cid)

        if gap > current_max:
            heapq.heappush(pq, (gap, cid, best_c, best_i))  # still too far, re-queue
            stall += 1
            if stall >= 2 * max_stall:
                # after too many futile tries, do a *forced* heap rebuild
                pq = [
                    (g, cid2, bc, bi)
                    for _, cid2, _, _ in pq
                    for g, bc, bi in (closest_pair(cid2),)
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

        # merge component into island and rebuild KD-tree
        island |= comps[cid]
        island_idx = np.concatenate([island_idx, verts_idx])
        island_tree = KDTree(nodes[island_idx])

        merges_since_recalc += 1
        if merges_since_recalc >= recalc_after:
            # distances of *every* remaining component may have changed
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
) -> tuple[
    np.ndarray, dict[str, np.ndarray], list[np.ndarray], dict[int, int], np.ndarray
]:
    """
    Iteratively merge every leaf whose *parent* has degree ≥ `min_parent_degree`.
    Terminates when no such leaves are left.
    """
    while True:
        # --- current degree -------------------------------------------
        deg = np.zeros(len(nodes), dtype=int)
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1

        # --- leaves that qualify --------------------------------------
        leaves = [i for i, d in enumerate(deg) if d == 1 and i != 0]
        parent = _bfs_parents(edges, len(nodes), root=0)
        singles = [leaf for leaf in leaves if deg[parent[leaf]] >= min_parent_degree]

        if not singles:
            break  # fixed-point reached – nothing more to merge

        # --- merge all singles in *one* shot ---------------------------
        to_drop = set()
        for leaf in singles:
            par = parent[leaf]

            # move vertices to parent
            node2verts[par] = np.concatenate((node2verts[par], node2verts[leaf]))

            # re-fit radii
            pts = mesh_vertices[node2verts[par]]
            d = np.linalg.norm(pts - nodes[par], axis=1)
            for k in radii_dict:
                radii_dict[k][par] = _estimate_radius(d, method=k)

            to_drop.add(leaf)

        # rebuild arrays after this sweep
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

    # final vert-to-node map
    vert2node = {int(v): i for i, vs in enumerate(node2verts) for v in vs}
    return nodes, radii_dict, node2verts, vert2node, edges


def _prune_neurites(
    nodes: np.ndarray,
    radii_dict: dict[str, np.ndarray],
    node2verts: list[np.ndarray],
    edges: np.ndarray,
    *,
    soma: Soma,
    mesh_vertices: np.ndarray,
    tip_extent_factor: float = 1.1,
    stem_extent_factor: float = 5.0,
    drop_single_node_branches: bool = True,
    log: Callable | None = None,
):
    """
    Collapse obvious mesh-artefact branches into the soma (node 0) in
    two independent passes.

    Parameters
    ----------
    tip_extent_factor : float, default 1.2
        Maximum allowed *tip* distance from the soma **surface**,
        expressed in multiples of the fitted spherical soma radius
        ``r_soma``.  For every branch grown from a soma-adjacent node
        (“stem”), we compute

            ``max(d_tip_to_surface)``

        over all nodes in that branch.  If the value never exceeds
        ``tip_extent_factor * r_soma`` we classify the whole branch as a
        peri-soma artefact and merge it into the soma.

    stem_extent_factor : float, default 3.0
        Companion threshold applied *only* to the stem node itself.
        This reproduces legacy behaviour where anything whose very first
        segment hugs the soma is considered expendable, even if its
        children wander further out.  Set to a smaller number to be more
        permissive (keep short stems), or larger to remove more.

    drop_single_node_branches : bool, default True
        After extent-based pruning, merge every remaining leaf whose
        branch consists of a **single node** (degree-1 child hanging off
        a parent with degree ≥ 3).  Disable if one-node twigs are
        meaningful for your downstream pipeline.

    Returns
    -------
    nodes_new, radii_new, node2verts_new, vert2node, edges_new, soma
        Skeleton arrays with the selected branches collapsed into
        the soma; see original function for exact formats.

    Notes
    -----
    * Both extent thresholds are relative to the soma *surface*,
      **not** its center.
    * Setting both extent factors ≤ 1 collapses any branch that never
      leaves the soma sphere at all.
    * The hard-coded ``min_parent_degree`` for one-node pruning is 3—
      adjust inside :func:`_merge_single_node_branches` if needed.
    """
    # ------------------------------------------------------------------
    # A. legacy extent-based pruning (unchanged)
    # ------------------------------------------------------------------
    r_soma = soma.spherical_radius
    d2c = np.asarray(soma.distance(nodes, to="center"))

    N = len(nodes)
    parent = _bfs_parents(edges, N, root=0)

    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    merge2soma, visited = set(), np.zeros(N, bool)

    for child in adj[0]:  # one neurite at a time
        if visited[child]:
            continue

        comp = []
        stack, prev = [child], 0
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            comp.append(v)
            for nb in adj[v]:
                if nb != prev and (parent[nb] == v or parent[v] == nb):
                    stack.append(nb)
            prev = v

        max_d = d2c[comp].max()
        thr = (stem_extent_factor if parent[child] == 0 else tip_extent_factor) * r_soma
        if max_d <= thr:
            merge2soma.update(comp)
    # ------------------------------------------------------------------
    # B. merge the selected branches *into* soma (node 0)
    # ------------------------------------------------------------------
    if merge2soma:
        # 1. move vertices
        for nid in merge2soma:
            node2verts[0] = np.concatenate((node2verts[0], node2verts[nid]))

        node2verts[0] = np.unique(node2verts[0])  # dedup

        # 2. recompute soma radii from the enlarged vertex set
        d0 = np.linalg.norm(mesh_vertices[node2verts[0]] - nodes[0], axis=1)
        for k in radii_dict:
            radii_dict[k][0] = _estimate_radius(d0, method=k)

        # (optional) keep the verbose trail intact
        if log:
            log(f"Merged {len(merge2soma)} peri-soma nodes into soma ")

    # ------------------------------------------------------------------
    # C. build the keep-mask (drop the now-empty helper nodes)
    # ------------------------------------------------------------------

    keep = np.ones(N, dtype=bool)
    if merge2soma:
        keep[list(merge2soma)] = False
    keep[0] = True  # never drop soma

    remap = {old: new for new, old in enumerate(np.where(keep)[0])}

    nodes_new = nodes[keep]
    node2verts_new = [node2verts[i] for i in np.where(keep)[0]]
    radii_new = {k: v[keep].copy() for k, v in radii_dict.items()}
    edges_new = np.asarray(
        [(remap[a], remap[b]) for a, b in edges if keep[a] and keep[b]], dtype=np.int64
    )
    vert2node = {int(v): i for i, vs in enumerate(node2verts_new) for v in vs}

    soma.verts = np.unique(node2verts_new[0]).astype(np.int64)
    pts = mesh_vertices[soma.verts]
    soma = soma.fit(pts, verts=soma.verts)

    if log:
        centre_txt = ", ".join(f"{c:7.1f}" for c in soma.center)
        radii_txt = ",".join(f"{c:7.1f}" for c in soma.axes)
        log(f"Moved soma to [{centre_txt}]")
        log(f"(r = {radii_txt})")

    # ------------------------------------------------------------------
    # C. optional one-node branch merge (runs on the rebuilt skeleton)
    # ------------------------------------------------------------------
    if drop_single_node_branches:
        before_n, before_e = len(nodes_new), edges_new.shape[0]

        (nodes_new, radii_new, node2verts_new, vert2node, edges_new) = (
            _merge_single_node_branches(
                nodes_new,
                radii_new,
                node2verts_new,
                edges_new,
                mesh_vertices=mesh_vertices,
                min_parent_degree=3,
            )
        )

        merged_n = before_n - len(nodes_new)
        merged_e = before_e - edges_new.shape[0]
        if log and merged_n:
            log(f"Merged {merged_n} single-node branches ({merged_e} edges).")

    return nodes_new, radii_new, node2verts_new, vert2node, edges_new, soma


def _extreme_vertex(mesh: trimesh.Trimesh, axis: str = "z", mode: str = "min") -> int:
    """
    Return the mesh-vertex index with either the minimal or maximal coordinate
    along *axis* (“x”, “y” or “z”).

    Examples
    --------
    >>> vid = _extreme_vertex(mesh, axis="x", mode="max")   # right-most tip
    >>> vid = _extreme_vertex(mesh, axis="z")               # lowest-z (default)
    """
    ax_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    coords = mesh.vertices[:, ax_idx]
    return int(np.argmin(coords) if mode == "min" else np.argmax(coords))


def _merge_nested_nodes(
    nodes: np.ndarray,
    radii: np.ndarray,  # primary estimator (e.g. "median")
    node2verts: list[np.ndarray],
    *,
    inside_frac: float = 0.9,  # 1.0 = 100 % (strict), 0.99 ≈ 99 %, …
    keep_root: bool = True,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Collapse node *j* into node *i* when at least ``inside_frac`` of *j*'s
    radius lies inside *i*'s radius:

        ‖cᵢ – cⱼ‖ + inside_frac · rⱼ  ≤  rᵢ  + tol

    The “keeper” (larger sphere) inherits *j*’s vertex IDs.

    Returns
    -------
    keep_mask      – Boolean mask to apply to all node-wise arrays.
    node2verts_new – Updated mapping (same order as keep_mask==True).
    old2new        – Vector mapping old → new node IDs (-1 if dropped).
    """
    if not (0.0 < inside_frac <= 1.0):
        raise ValueError("inside_frac must be in (0, 1].")

    N = len(nodes)
    order = np.argsort(-radii)  # big → small
    tree = KDTree(nodes)

    keep_mask = np.ones(N, bool)
    old2new = np.arange(N, dtype=np.int64)

    for i in order:
        if keep_root and i == 0:
            continue  # never drop soma
        if not keep_mask[i]:
            continue  # already swallowed

        # neighbours that *might* fit: distance ≤ rᵢ + r_max
        cand_idx = tree.query_ball_point(nodes[i], radii[i] + radii.max())
        for j in cand_idx:
            if j == i or not keep_mask[j] or radii[j] > radii[i]:
                continue  # j is larger or gone; skip

            dist = np.linalg.norm(nodes[i] - nodes[j])
            # modified containment test
            if dist + inside_frac * radii[j] <= radii[i] + tol:
                node2verts[i] = np.concatenate((node2verts[i], node2verts[j]))
                keep_mask[j] = False
                old2new[j] = old2new[i]

    # compact node2verts into surviving order
    node2verts_new = [node2verts[k] for k in np.where(keep_mask)[0]]
    return keep_mask, node2verts_new, old2new


def _make_nodes(
    all_shells: list[list[np.ndarray]],
    vertices: np.ndarray,
    *,
    radius_estimators: list[str],
    merge_nested: bool = True,
    merge_kwargs: dict | None = None,
) -> tuple[
    np.ndarray,  # nodes_arr
    dict[str, np.ndarray],  # radii_dict
    list[np.ndarray],  # node2verts
    dict[int, int],  # vert2node
]:
    """
    Convert geodesic bins into skeleton nodes **and** run the optional
    `_merge_nested_nodes()` clean-up.

    Parameters
    ----------
    all_shells
        Output of `_bin_geodesic_shells()`.
    vertices
        `mesh.vertices` as `(N,3) float64`.
    radius_estimators
        Names understood by `_estimate_radius()`.
    merge_nested
        Whether to collapse fully nested spheres afterwards.
    merge_kwargs
        Passed straight through to `_merge_nested_nodes()`.

    Returns
    -------
    nodes_arr, radii_dict, node2verts, vert2node
    """
    if merge_kwargs is None:
        merge_kwargs = {}

    nodes: list[np.ndarray] = []
    node2verts: list[np.ndarray] = []
    radii_dict: dict[str, np.ndarray] = {k: np.array([]) for k in radius_estimators}
    vert2node: dict[int, int] = {}

    next_id = 0
    for shells in all_shells:  # outer = distance order
        for bin_ids in shells:  # inner = connected patch
            pts = vertices[bin_ids]
            center = pts.mean(axis=0)

            d = np.linalg.norm(pts - center, axis=1)  # distances → radii
            for est in radius_estimators:
                radii_dict[est] = np.append(
                    radii_dict[est], _estimate_radius(d, method=est, trim_fraction=0.05)
                )

            nodes.append(center.astype(np.float64))
            node2verts.append(bin_ids)
            for vid in bin_ids:
                vert2node[int(vid)] = next_id
            next_id += 1

    nodes_arr = np.asarray(nodes, dtype=np.float64)
    radii_dict = {k: np.asarray(v) for k, v in radii_dict.items()}

    # ---- optional containment-based merge ----------------------------
    if merge_nested and len(nodes_arr):
        keep_mask, node2verts, _ = _merge_nested_nodes(
            nodes_arr,
            np.asanyarray(radii_dict[radius_estimators[0]]),
            node2verts,
            **merge_kwargs,
        )
        nodes_arr = nodes_arr[keep_mask]
        for k in radii_dict:
            radii_dict[k] = np.asanyarray(radii_dict[k][keep_mask])

        # rebuild vert2node using final indices
        vert2node = {
            int(v): int(new_id)
            for new_id, verts in enumerate(node2verts)
            for v in verts
        }

    return nodes_arr, radii_dict, node2verts, vert2node


# ------------------------------------------------------------------
#  Step-3 helper – refine soma & root swap
# ------------------------------------------------------------------
def _detect_soma(
    nodes,
    radii,
    node2verts,
    vert2node,
    soma_radius_percentile_threshold,
    soma_radius_distance_factor,
    soma_min_nodes,
    *,
    detect_soma,
    radius_key,
    mesh_vertices,
    log=None,
):
    """
    Now returns (nodes, radii, node2verts, vert2node, soma, has_soma)
    where `soma` is a *Soma* instance.
    """
    # ------------------------------------------------------------------
    # A. skip?
    # ------------------------------------------------------------------
    if not detect_soma:
        soma = Soma.from_sphere(nodes[0], radii[radius_key][0], verts=node2verts[0])
        return nodes, radii, node2verts, vert2node, soma, False

    # ------------------------------------------------------------------
    # B. geometry-based detection
    # ------------------------------------------------------------------
    soma, soma_nodes, has_soma = _find_soma(
        nodes,
        radii[radius_key],
        pct_large=soma_radius_percentile_threshold,
        dist_factor=soma_radius_distance_factor,
        min_keep=soma_min_nodes,
    )
    if not has_soma:
        if log:
            log("no soma detected → keeping old root")
        soma = Soma.from_sphere(nodes[0], radii[radius_key][0], verts=node2verts[0])
        return nodes, radii, node2verts, vert2node, soma, False

    # ------------------------------------------------------------------
    # C. ensure fattest soma node is root (logic unchanged)
    # ------------------------------------------------------------------
    if 0 not in soma_nodes:
        new_root = int(soma_nodes[np.argmax(radii[radius_key][soma_nodes])])
        nodes[[0, new_root]] = nodes[[new_root, 0]]
        for k in radii:
            radii[k][[0, new_root]] = radii[k][[new_root, 0]]
        node2verts[0], node2verts[new_root] = node2verts[new_root], node2verts[0]
        for vid, nid in list(vert2node.items()):
            vert2node[vid] = {0: new_root, new_root: 0}.get(nid, nid)

    # ------------------------------------------------------------------
    # D. collect surface vertices that belong to the soma envelope
    # ------------------------------------------------------------------
    # quick hack fix, might need a better solution within _find_soma()

    all_close = (
        soma.distance(nodes[soma_nodes], to="center") < soma.spherical_radius * 2
    )
    soma_vert_ids = np.unique(
        np.concatenate([node2verts[i] for i in soma_nodes[all_close]])
    ).astype(np.int64)
    soma.verts = soma_vert_ids

    # update centroid & spherical_radius written to node 0
    nodes[0] = soma.center
    r_sphere = soma.spherical_radius
    for k in radii:
        radii[k][0] = r_sphere

    try:
        soma = Soma.fit(
            mesh_vertices[soma.verts],
            verts=soma.verts,
        )
    except ValueError:
        if log:
            log("Soma fitting failed, using spherical approximation instead.")
        # fallback to spherical approximation
        soma = Soma.from_sphere(soma.center, r_sphere, verts=soma.verts)

    if log:
        centre_txt = ", ".join(f"{c:7.1f}" for c in soma.center)
        radii_txt = ",".join(f"{c:7.1f}" for c in soma.axes)
        log(f"Found soma at [{centre_txt}]")
        log(f"(r = {radii_txt})")

    return nodes, radii, node2verts, vert2node, soma, True


# -----------------------------------------------------------------------------
#  Skeletonization Public API
# -----------------------------------------------------------------------------


def skeletonize(
    mesh: trimesh.Trimesh,
    # --- radius estimation ---
    radius_estimators: list[str] = ["median", "mean", "trim"],
    # --- soma detection ---
    detect_soma: bool = True,
    soma_seed_point: np.ndarray | list | tuple | None = None,
    soma_radius_percentile_threshold: float = 99.9,
    soma_radius_distance_factor: float = 4,
    soma_min_nodes: int = 3,
    # -- for post-skeletonization soma detection only--
    soma_init_guess_axis: str = "z",  # "x" | "y" | "z"
    soma_init_guess_mode: str = "min",  # "min" | "max"
    # --- geodesic binning ---
    geodesic_step_size: float | None = None,
    geodesic_shell_count: int = 1000,  # higher = more bins, smaller bin size
    min_shell_vertices: int = 6,
    max_shell_width_factor: int = 50,
    split_elongated_shells: bool = False,
    split_aspect_thr: float = 3.0,  # λ1 / λ2
    split_min_shell_vertices: int = 15,
    split_max_vertices_per_slice: int | None = None,
    merge_nodes_overlap_fraction: float = 0.8,  # merge nested nodes if inside_frac ≥ this
    # --- bridging disconnected patches ---
    bridge_gaps: bool = True,
    bridge_max_factor: float | None = None,
    bridge_recalc_after: int | None = None,
    # -- post‑processing --
    # --- collapse soma-like nodes ---
    collapse_soma: bool = True,
    collapse_soma_dist_factor: float = 1.2,
    collapse_soma_radius_factor: float = 0.2,
    # --- prune tiny neurites ---
    prune_tiny_neurites: bool = True,
    prune_tip_extent_factor: float = 1.2,  # tip twigs (<–× r_soma)
    prune_stem_extent_factor: float = 3.0,  # stems touching soma
    prune_drop_single_node_branches: bool = True,
    # --- misc ---
    unit: str = "nm",
    id: str | int | None = None,
    verbose: bool = False,
) -> Skeleton:
    """Compute a center-line skeleton with radii of a neuronal mesh .

    The algorithm proceeds in eight conceptual stages:

      1. geodesic shell binning of every connected surface patch
      2. cluster each shell ⇒ interior node with local radius
      3. optional post-skeletonization soma detection
      4. project mesh edges ⇒ graph edges between nodes
      5. optional collapsing of soma-like/fat nodes near the centroid
      6. optional bridging of disconnected components
      7. minimum-spanning tree (global) to remove microscopic cycles
      8. optional pruning of tiny neurites sprouting directly from the soma


    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface mesh of the neuron in *arbitrary* units.
    target_shell_count : int, default ``500``
        Rough number of geodesic shells to produce per component.  The actual
        shell width is adapted to mesh resolution.
    bridge_gaps : bool, default ``True``
        If the mesh contains disconnected islands (breaks, imaging artefacts),
        attempt to connect them back to the soma with synthetic edges.
    bridge_k : int, default ``1``
        How many candidate node pairs to test when bridging a foreign island.
    prune_tiny_neurites : bool, default ``True``
        Remove sub-trees with fewer than ``min_branch_nodes`` that attach
        *directly* to the soma and do not extend beyond
        ``min_branch_extent_factor × r_soma``.
    collapse_soma : bool, default ``True``
        Merge centroids that sit well inside the soma or have very fat radii.
    verbose : bool, default ``False``
        Print progress messages.

    Returns
    -------
    Skeleton
        The (acyclic) skeleton with vertex 0 at the soma centroid.
    """
    # ------------------------------------------------------------------
    #  helpers for verbose timing
    # ------------------------------------------------------------------
    if verbose:
        _global_start = time.perf_counter()
        print(
            f"[skeliner] starting skeletonisation ({len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} faces)"
        )
        soma_ms = 0.0  # soma detection time
        post_ms = 0.0  # post-processing time

    @contextmanager
    def _timed(label: str, *, verbose: bool = verbose):  # keep the signature you like
        """
        Context manager that prints

            ↳  <label padded to width> … <elapsed> s
                └─ <sub-message 1>
                └─ <sub-message 2>
                …

        Use the yielded `log()` callback to record any number of sub-messages.
        """
        if not verbose:
            yield lambda *_: None
            return

        PAD = 47  # keeps the old alignment
        print(f" {label:<{PAD}} …", end="", flush=True)
        t0 = time.perf_counter()
        _msgs: list[str] = []

        def log(msg: str) -> None:
            _msgs.append(str(msg))

        try:
            yield log  # the `with`-body gets this function
        finally:
            dt = time.perf_counter() - t0
            print(f" {dt:.2f} s")  # finish first line

            for m in _msgs:  # then all sub-messages, nicely indented
                print(f"      └─ {m}")

    # 0. soma vertices ---------------------------------------------------
    with _timed("↳  build surface graph", verbose=verbose):
        gsurf = _surface_graph(mesh)

    # 1. binning surface vertices by geodesic distance ----------------------------------
    with _timed("↳  bin surface vertices by geodesic distance", verbose=verbose):
        mesh_vertices = mesh.vertices.view(np.ndarray)

        # pseudo-random soma seed point for kick-starting the binning
        if soma_seed_point is not None:
            seed_vid = int(
                np.argmin(
                    np.linalg.norm(mesh_vertices - np.asarray(soma_seed_point), axis=1)
                )
            )
        else:
            seed_vid = _extreme_vertex(
                mesh, axis=soma_init_guess_axis, mode=soma_init_guess_mode
            )

        avg_edge = float(mesh.edges_unique_length.mean())
        soma = Soma.from_sphere(
            mesh_vertices[seed_vid],
            radius=avg_edge,
            verts=np.asarray([int(seed_vid)], dtype=np.int64),
        )

        all_shells = _bin_geodesic_shells(
            mesh,
            gsurf,
            soma=soma,
            step_size=geodesic_step_size,
            target_shell_count=geodesic_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=max_shell_width_factor,
            split_elongated_shells=split_elongated_shells,
            split_aspect_thr=split_aspect_thr,
            split_min_shell_vertices=split_min_shell_vertices,
            split_max_vertices_per_slice=split_max_vertices_per_slice,
        )

    # 2. create skeleton nodes ------------------------------------------
    with _timed("↳  compute bin centroids and radii", verbose=verbose):
        (nodes_arr, radii_dict, node2verts, vert2node) = _make_nodes(
            all_shells,
            mesh_vertices,
            radius_estimators=radius_estimators,
            merge_nested=True,
            merge_kwargs={
                "inside_frac": merge_nodes_overlap_fraction
            },  # tune `inside_frac`/`keep_root` here if needed
        )

    # 3. soma detection (optional) -----------------------------------
    _t0 = time.perf_counter()
    with _timed("↳  post-skeletonization soma detection") as log:
        (nodes_arr, radii_dict, node2verts, vert2node, soma, has_soma) = _detect_soma(
            nodes_arr,
            radii_dict,
            node2verts,
            vert2node,
            soma_radius_percentile_threshold=soma_radius_percentile_threshold,
            soma_radius_distance_factor=soma_radius_distance_factor,
            soma_min_nodes=soma_min_nodes,
            detect_soma=detect_soma,
            mesh_vertices=mesh_vertices,
            radius_key=radius_estimators[0],
            log=log,
        )
        soma_ms = time.perf_counter() - _t0

    # 4. edges from mesh connectivity -----------------------------------
    with _timed("↳  map mesh faces to skeleton edges", verbose=verbose):
        edges_arr = _edges_from_mesh(
            mesh.edges_unique,
            vert2node,
            n_mesh_verts=len(mesh.vertices),
        )

    # 5. collapse soma‑like / fat nodes ---------------------------
    if has_soma and collapse_soma:
        _t0 = time.perf_counter()

        with _timed("↳  merge redundant near-soma nodes", verbose=verbose) as log:
            (nodes_arr, radii_dict, node2verts, vert2node, edges_arr, soma) = (
                _merge_near_soma_nodes(
                    nodes_arr,
                    radii_dict,
                    edges_arr,
                    node2verts,
                    soma=soma,
                    radius_key=radius_estimators[0],
                    mesh_vertices=mesh_vertices,
                    fat_factor=collapse_soma_radius_factor,
                    near_factor=collapse_soma_dist_factor,
                    log=log,
                )
            )

        if verbose:
            post_ms += time.perf_counter() - _t0

    # 6. Connect all components ------------------------------
    if bridge_gaps:
        _t0 = time.perf_counter()
        with _timed("↳  bridge skeleton gaps", verbose=verbose) as log:
            edges_arr = _bridge_gaps(
                nodes_arr,
                edges_arr,
                bridge_max_factor=bridge_max_factor,
                bridge_recalc_after=bridge_recalc_after,
            )
        if verbose:
            post_ms += time.perf_counter() - _t0

    # 7. global minimum-spanning tree ------------------------------------
    with _timed("↳  build global minimum-spanning tree", verbose=verbose):
        edges_mst = _build_mst(nodes_arr, edges_arr)
        if verbose:
            post_ms += time.perf_counter() - _t0

    # 8. prune tiny sub-trees near the soma
    if has_soma and prune_tiny_neurites:
        _t0 = time.perf_counter()
        with _timed("↳  prune tiny neurites", verbose=verbose) as log:
            (nodes_arr, radii_dict, node2verts, vert2node, edges_mst, soma) = (
                _prune_neurites(
                    nodes_arr,
                    radii_dict,
                    node2verts,
                    edges_mst,
                    soma=soma,
                    mesh_vertices=mesh_vertices,
                    tip_extent_factor=prune_tip_extent_factor,
                    stem_extent_factor=prune_stem_extent_factor,
                    drop_single_node_branches=prune_drop_single_node_branches,
                    log=log,
                )
            )
        if verbose:
            post_ms += time.perf_counter() - _t0

    if verbose:
        total_ms = time.perf_counter() - _global_start
        core_ms = total_ms - soma_ms - post_ms

        if post_ms > 1e-6:  # at least one optional stage ran
            print(
                f"{'TOTAL (soma + core + post)':<49}"
                f"… {total_ms:.2f} s "
                f"({soma_ms:.2f} + {core_ms:.2f} + {post_ms:.2f})"
            )
            print(f"({len(nodes_arr):,} nodes, {edges_mst.shape[0]:,} edges)")
        else:  # no post-processing at all
            print(
                f"{'TOTAL (soma + core)':<49}"
                f"… {total_ms:.2f} s "
                f"({soma_ms:.2f} + {core_ms:.2f})"
            )

    return Skeleton(
        nodes=nodes_arr,
        radii=radii_dict,
        edges=edges_mst,
        ntype=None,
        soma=soma,
        node2verts=node2verts,
        vert2node=vert2node,
        meta={
            "skeliner_version": _SKELINER_VERSION,
            "skeletonized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "unit": unit,
            "id": id,
        },
    )
