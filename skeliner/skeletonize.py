import time
from contextlib import contextmanager
from importlib import metadata as _metadata
from typing import Dict, List

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

from ._core import (
    _bridge_gaps,
    _build_mst,
    _detect_soma,
    _estimate_radius,
    _merge_near_soma_nodes,
    _prune_neurites,
)
from ._state import rebuild_vert2node
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

        vert2node = rebuild_vert2node(node2verts) or {}

    return nodes_arr, radii_dict, node2verts, vert2node


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
    postprocess: bool = True,
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
    postprocess : bool, default ``True``
        When ``False`` the optional post-processing stages (soma detection,
        near-soma merging, gap bridging, MST rebuild, neurite pruning) are
        skipped so that you can rerun them later via the corresponding
        :mod:`skeliner.post` helpers.

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

    run_mst = True
    if not postprocess:
        detect_soma = False
        collapse_soma = False
        bridge_gaps = False
        prune_tiny_neurites = False
        run_mst = False

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
        (
            nodes_arr,
            radii_dict,
            node2verts,
            vert2node,
            soma,
            has_soma,
            _,
        ) = _detect_soma(
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
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_arr,
                soma,
                _,
            ) = _merge_near_soma_nodes(
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
    if run_mst:
        _t0 = time.perf_counter()
        with _timed("↳  build global minimum-spanning tree", verbose=verbose):
            edges_mst = _build_mst(nodes_arr, edges_arr)
        if verbose:
            post_ms += time.perf_counter() - _t0
    else:
        edges_mst = edges_arr

    # 8. prune tiny sub-trees near the soma
    if has_soma and prune_tiny_neurites:
        _t0 = time.perf_counter()
        with _timed("↳  prune tiny neurites", verbose=verbose) as log:
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_mst,
                soma,
                _,
            ) = _prune_neurites(
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
