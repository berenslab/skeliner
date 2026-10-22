import time
from collections import deque
from contextlib import contextmanager, nullcontext
from importlib import metadata as _metadata
from typing import Any, Dict, List

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

from . import _rs
from ._core import (
    TRIM_FRACTION,
    _bridge_gaps,
    _build_mst,
    _detect_soma,
    _estimate_radius,
    _merge_near_soma_nodes,
    _prune_neurites,
)
from ._state import rebuild_vert2node
from .dataclass import MeshComponents, Skeleton, Soma

_SKELINER_VERSION = _metadata.version("skeliner")


__all__ = [
    "skeletonize",
]


def _jsonable(v):
    """Coerce a parameter value to something ``json.dumps`` accepts.

    ``to_swc`` writes ``meta`` as a single ``# meta {...}`` line with a plain
    ``json.dumps`` and no fallback encoder, so a numpy scalar or array left in
    here would raise on *export* — long after the run that recorded it.
    ``soma_seed_point`` is the reachable case.
    """
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _skel_meta(track: str, unit: str, id, params: dict, mesh=None) -> dict:
    """Provenance for one skeletonize run.

    *track* and *params* are recorded by the assembler that ran, not by the
    dispatcher, so what lands is what actually took effect.  The preprocessing
    track ignores most of :func:`skeletonize`'s arguments — soma detection,
    gap bridging and pruning all happen upstream — and copying them in would
    describe a run that did not happen.

    Without this a saved skeleton cannot say which track produced it, and the
    two tracks disagree about the same cell by construction.  It is written
    into every export, so it cannot be added retroactively.

    ``mesh`` records the surface ``node2verts`` indexes into.  A skeleton is
    only meaningful beside that mesh — the bins name vertex ids, and every
    radius was measured over them — but nothing in the vertex ids themselves
    says which mesh they belong to.  Pairing a skeleton with a *larger*
    unrelated mesh keeps every index in range, so bins resolve to real faces
    and read as a correct answer.  Counts settle it: they are exact, they
    survive the ``.obj``/``.ply`` round trip that perturbs coordinates in the
    ninth decimal, and they cost two integers.
    """
    meta = {
        "skeliner_version": _SKELINER_VERSION,
        "skeletonized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "unit": unit,
        "id": _jsonable(id),
        "track": track,
        "params": {k: _jsonable(v) for k, v in params.items()},
    }
    if mesh is not None:
        meta["mesh"] = {
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
        }
    return meta


# -----------------------------------------------------------------------------
#  Verbose timing helper
# -----------------------------------------------------------------------------


@contextmanager
def _timed(label: str, *, verbose: bool):
    """Context manager printing  ``↳  label … N.NN s``.

    Use the yielded ``log()`` callback for sub-messages::

        with _timed("stage", verbose=True) as log:
            log("detail 1")
    """
    if not verbose:
        yield lambda *_: None
        return

    PAD = 47
    print(f" {label:<{PAD}} …", end="", flush=True)
    t0 = time.perf_counter()
    _msgs: list[str] = []

    def log(msg: str) -> None:
        _msgs.append(str(msg))

    try:
        yield log
    finally:
        dt = time.perf_counter() - t0
        print(f" {dt:.2f} s")
        for m in _msgs:
            print(f"      └─ {m}")


def _no_stage(_label: str):
    """Report nothing — the default when a caller wants no progress."""
    return nullcontext()


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


def _surface_graph_rs(mesh: trimesh.Trimesh):
    """:func:`_surface_graph` as a :class:`skeliner._rs.SurfaceGraph`."""
    return _rs.SurfaceGraph(
        len(mesh.vertices),
        np.asarray(mesh.edges_unique, dtype=np.int64),
        np.asarray(mesh.edges_unique_length, dtype=np.float64),
    )


def _dist_vec_for_component(
    gsurf: ig.Graph,
    verts: np.ndarray,  # 1-D int64 array of vertex IDs (one component)
    seed_vid: int | np.ndarray,  # single or multiple mesh-vertex IDs
) -> np.ndarray:
    """
    Return the distance vector *d[verts[i]]* from *seed_vid* to
    every vertex in this component, **without touching the rest of
    the mesh**.

    When *seed_vid* is an array of vertex IDs (multi-seed), the
    returned distance is the minimum over all seeds (multi-source
    shortest path).
    """
    # Build a dedicated sub-graph (much smaller than gsurf)
    sub = gsurf.induced_subgraph(verts, implementation="create_from_scratch")

    seed_arr = np.atleast_1d(np.asarray(seed_vid, dtype=np.int64))

    # Map seed mesh-vertex IDs → local indices in *sub*.  Order does not
    # matter: either there is one seed, or they are all joined to a
    # single virtual vertex below.
    root_idxs = np.flatnonzero(np.isin(verts, seed_arr)).tolist()
    if not root_idxs:
        raise ValueError("no seed vertex lies in this component")

    if len(root_idxs) > 1:
        # Join every seed to one virtual vertex by a zero-weight edge:
        # a single Dijkstra from it is the min over all seeds, exactly.
        # Passing the seeds to igraph as `source=` instead runs one
        # Dijkstra per seed and builds a (len(seeds), |verts|) matrix
        # only to take its minimum — seeding a 740k-vertex neurite from
        # a 78-vertex soma ring costs 11.4 s and 0.46 GB that way,
        # against 0.3 s here for the same distances.
        sub.add_vertex()
        virtual = sub.vcount() - 1
        sub.add_edges(
            [(virtual, i) for i in root_idxs],
            attributes={"weight": [0.0] * len(root_idxs)},
        )
        root_idxs = [virtual]

    dist = np.asarray(sub.distances(source=root_idxs, weights="weight")[0])
    # drop the virtual vertex, which igraph appended last
    return dist[: len(verts)]


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


def _neighbour_groups(
    comp_verts: np.ndarray,
    gsurf: ig.Graph,
    owner_of: dict[int, tuple[int, int]],
    own: tuple[int, int],
) -> list[list[int]]:
    """Group the vertices of other bins that touch this one, by tube.

    Like :func:`_is_ring`, but returns the groups instead of stopping at
    two, and asks which *bin* a neighbour belongs to rather than whether
    it is in the component's vertex set.  A cross-section has exactly two
    groups, one on each side; more than two means the bin touches several
    tubes at once.

    Touching one bin along two separate patches is still one tube, so
    patches are merged when they share a bin.  Without that, a bin that
    merely dips into a neighbour twice looks like a branch point and gets
    cut into slivers.
    """
    ext: set[int] = set()
    for vid in comp_verts:
        for e in gsurf.incident(int(vid)):
            src, tgt = gsurf.es[e].source, gsurf.es[e].target
            nbr = tgt if src == int(vid) else src
            key = owner_of.get(nbr)
            if key is not None and key != own:
                ext.add(nbr)

    patches: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(ext):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        grp = [start]
        while queue:
            u = queue.popleft()
            for e in gsurf.incident(u):
                src, tgt = gsurf.es[e].source, gsurf.es[e].target
                nbr = tgt if src == u else src
                if nbr in ext and nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
                    grp.append(nbr)
        patches.append(grp)

    # union patches that touch the same neighbouring bin
    parent = list(range(len(patches)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen: dict[tuple[int, int], int] = {}
    for pi, grp in enumerate(patches):
        for v in grp:
            key = owner_of[v]
            other = first_seen.setdefault(key, pi)
            ra, rb = _find(pi), _find(other)
            if ra != rb:
                parent[rb] = ra

    merged: dict[int, list[int]] = {}
    for pi, grp in enumerate(patches):
        merged.setdefault(_find(pi), []).extend(grp)
    return [merged[k] for k in sorted(merged)]


def _split_branch_band(
    comp: np.ndarray,
    groups: list[list[int]],
    gsurf: ig.Graph,
) -> list[np.ndarray]:
    """Cut a band that spans a branch point into one piece per neighbour.

    Grows a region from each neighbouring group simultaneously, so every
    vertex joins the tube it is nearest to *across the surface*.  Each
    piece then wraps a single tube and its centroid lands inside that
    tube.  Vertices no group reaches stay with the largest piece.
    """
    vset = set(int(v) for v in comp)
    label: dict[int, int] = {}
    queue: deque[int] = deque()
    for gi, grp in enumerate(groups):
        for u in sorted(grp):
            for e in gsurf.incident(u):
                src, tgt = gsurf.es[e].source, gsurf.es[e].target
                nbr = tgt if src == u else src
                if nbr in vset and nbr not in label:
                    label[nbr] = gi
                    queue.append(nbr)

    while queue:
        u = queue.popleft()
        gi = label[u]
        for e in gsurf.incident(u):
            src, tgt = gsurf.es[e].source, gsurf.es[e].target
            nbr = tgt if src == u else src
            if nbr in vset and nbr not in label:
                label[nbr] = gi
                queue.append(nbr)

    parts: list[list[int]] = [[] for _ in groups]
    orphans: list[int] = []
    for v in comp:
        gi = label.get(int(v))
        if gi is None:
            orphans.append(int(v))
        else:
            parts[gi].append(int(v))

    out = [np.asarray(p, dtype=np.int64) for p in parts if p]
    if not out:
        return [np.asarray(comp, dtype=np.int64)]
    if orphans:
        big = max(range(len(out)), key=lambda i: len(out[i]))
        out[big] = np.concatenate([out[big], np.asarray(orphans, dtype=np.int64)])
    return out


def _is_ring(
    comp_verts: np.ndarray,
    gsurf: ig.Graph,
    neurite_set: set[int],
) -> bool:
    """Test whether a vertex component is a topological ring.

    A ring wraps around the tube so that removing it separates the
    surface into two sides.  Equivalently, the external neighbors
    (neurite vertices adjacent to the component but not in it)
    form 2+ disconnected groups.  A non-ring patch has all its
    external neighbors in one connected group.
    """
    vset = set(int(v) for v in comp_verts)

    # Collect external neighbors
    ext: set[int] = set()
    for vid in comp_verts:
        for e in gsurf.incident(int(vid)):
            src, tgt = gsurf.es[e].source, gsurf.es[e].target
            nbr = tgt if src == int(vid) else src
            if nbr not in vset and nbr in neurite_set:
                ext.add(nbr)

    if len(ext) < 2:
        return False

    # Count connected components among external neighbors
    visited: set[int] = set()
    n_groups = 0
    for start in ext:
        if start in visited:
            continue
        n_groups += 1
        if n_groups >= 2:
            return True
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)
            for e in gsurf.incident(u):
                src, tgt = gsurf.es[e].source, gsurf.es[e].target
                nbr = tgt if src == u else src
                if nbr in ext and nbr not in visited:
                    queue.append(nbr)

    return n_groups >= 2


def _bin_one_component(
    gsurf: ig.Graph,
    verts: np.ndarray,
    seed_vid: int | np.ndarray,
    *,
    mesh_vertices: np.ndarray,
    mean_edge_len: float,
    soma_verts: set[int] | None = None,
    step_size: float | None = None,
    target_shell_count: int = 500,
    min_shell_vertices: int = 6,
    max_shell_width_factor: float = 50.0,
    split_elongated_shells: bool = True,
    split_aspect_thr: float = 3.0,
    split_min_shell_vertices: int = 50,
    split_max_vertices_per_slice: int | None = None,
) -> List[List[np.ndarray]]:
    """Bin one connected surface component into geodesic shells.

    Parameters
    ----------
    gsurf
        Full surface graph of the mesh.
    verts
        1-D int64 array of vertex IDs in this component.
    seed_vid
        Mesh-vertex ID(s) to start the geodesic from.  A single
        ``int`` or a 1-D array for multi-source seeding.
    mesh_vertices
        ``mesh.vertices`` as ``(N, 3) float64``.
    mean_edge_len
        Mean mesh-edge length (for step-size fallback).
    soma_verts
        Vertex IDs to exclude from shells.  ``None`` or
        empty set means no exclusion.

    Returns
    -------
    List[List[np.ndarray]]
        Shells for this component — same structure as the
        return of :func:`_bin_geodesic_shells`.
    """
    v = mesh_vertices
    e_m = mean_edge_len
    exclude = soma_verts or set()
    vert_set = set(int(x) for x in verts)

    # -- geodesic distance from seed to every vertex in component --
    dist_vec = _dist_vec_for_component(gsurf, verts, seed_vid)
    dist_sub = {int(vid): float(d) for vid, d in zip(verts, dist_vec)}
    if not dist_sub:
        return []

    # -- shell width ---------------------------------------------------
    if step_size is None:
        arc_len = max(dist_sub.values())
        step = max(e_m * 2.0, arc_len / target_shell_count)
    else:
        step = float(step_size)

    # increase step until at least one non-empty shell
    shells: List[List[int]] = []
    while not any(shells) and step < e_m * max_shell_width_factor:
        shells = _geodesic_bins(dist_sub, step)
        step *= 1.5

    # -- split each shell into connected sub-clusters ------------------
    component_shells: List[List[np.ndarray]] = []
    pending_frags: List[np.ndarray] = []
    for shell_verts in shells:
        inner = [vid for vid in shell_verts if vid not in exclude]
        if not inner:
            continue

        sub = gsurf.induced_subgraph(inner)
        comps = []
        for comp in sub.components():
            comp_idx = np.fromiter((inner[i] for i in comp), dtype=np.int64)
            if len(comp) < min_shell_vertices:
                # Too thin to stand as its own bin — but its vertices are
                # still surface, so they go to the fragment merge below
                # instead of being dropped.  Dropping them severs the
                # skeleton: `_edges_from_mesh` joins two nodes only when a
                # mesh edge has *both* endpoints binned, so a discarded band
                # leaves no edge across it however continuous the surface is.
                # At a narrow neck the sub-components are small for several
                # consecutive shells, which is exactly where the band forms.
                pending_frags.append(comp_idx)
                continue
            if split_elongated_shells and len(comp) < 1500:  # hard-coded, might be soma
                for part in _split_comp_if_elongated(
                    comp_idx,
                    v,
                    aspect_thr=split_aspect_thr,
                    min_shell_vertices=(split_min_shell_vertices),
                    max_vertices_per_slice=(split_max_vertices_per_slice),
                ):
                    comps.append(part)
            else:
                comps.append(comp_idx)

        # -- separate rings from non-ring fragments --
        rings = []
        frags = []
        for c in comps:
            if _is_ring(c, gsurf, vert_set):
                rings.append(c)
            else:
                frags.append(c)
        if frags:
            pending_frags.extend(frags)
        comps = rings

        if comps:
            component_shells.append(comps)

    # -- merge pending fragments via mesh-edge connectivity --
    if pending_frags:
        # Build vertex → (band, comp) lookup for all rings
        vid_to_ring: dict[int, tuple[int, int]] = {}
        for bi, band in enumerate(component_shells):
            for ci, ring in enumerate(band):
                for vid in ring:
                    vid_to_ring[int(vid)] = (bi, ci)

        for frag in pending_frags:
            # Count mesh-edge connections to each ring
            votes: dict[tuple[int, int], int] = {}
            for vid in frag:
                for e in gsurf.incident(int(vid)):
                    nbr = int(
                        gsurf.es[e].target
                        if gsurf.es[e].source == int(vid)
                        else gsurf.es[e].source
                    )
                    key = vid_to_ring.get(nbr)
                    if key is not None:
                        votes[key] = votes.get(key, 0) + 1
            if votes:
                best = max(votes, key=votes.get)
                bi, ci = best
                component_shells[bi][ci] = np.concatenate(
                    [component_shells[bi][ci], frag]
                )
                # Update lookup for merged verts
                for vid in frag:
                    vid_to_ring[int(vid)] = best

    return component_shells


def _bin_geodesic_shells(
    mesh: trimesh.Trimesh,
    gsurf: ig.Graph,
    *,
    seed_vid: int,
    soma_verts: set[int] | None = None,
    step_size: float | None = None,
    target_shell_count: int = 500,
    min_shell_vertices: int = 6,
    max_shell_width_factor: float = 50.0,
    split_elongated_shells: bool = True,
    split_aspect_thr: float = 3.0,
    split_min_shell_vertices: int = 50,
    split_max_vertices_per_slice: int | None = None,
) -> List[List[np.ndarray]]:
    """Cluster every connected surface component into geodesic
    shells.

    Iterates over connected components of *gsurf*, picks a
    seed per component, and delegates to
    :func:`_bin_one_component`.

    Parameters
    ----------
    seed_vid
        Mesh-vertex ID used as the geodesic origin in the
        component that contains it.  Other components fall
        back to a deterministic pseudo-random vertex.
    soma_verts
        Vertex IDs to exclude from shells (e.g. soma surface
        vertices).  ``None`` means no exclusion.

    Returns
    -------
    List[List[np.ndarray]]
        Outer list = shells ordered by growing distance;
        inner list = connected vertex clusters inside that
        shell; each cluster is a 1-D ``int64`` array.
    """
    v = mesh.vertices.view(np.ndarray)
    e_m = float(mesh.edges_unique_length.mean())

    comp_vertices = [np.asarray(c, dtype=np.int64) for c in gsurf.components()]
    all_shells: List[List[np.ndarray]] = []

    for cid, verts in enumerate(comp_vertices):
        # one seed per component
        if np.any(verts == seed_vid):
            comp_seed = seed_vid
        else:
            comp_seed = int(verts[hash(cid) % len(verts)])

        shells = _bin_one_component(
            gsurf,
            verts,
            comp_seed,
            mesh_vertices=v,
            mean_edge_len=e_m,
            soma_verts=soma_verts,
            step_size=step_size,
            target_shell_count=target_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=max_shell_width_factor,
            split_elongated_shells=split_elongated_shells,
            split_aspect_thr=split_aspect_thr,
            split_min_shell_vertices=(split_min_shell_vertices),
            split_max_vertices_per_slice=(split_max_vertices_per_slice),
        )
        all_shells.extend(shells)

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
                    radii_dict[est],
                    _estimate_radius(d, method=est, trim_fraction=TRIM_FRACTION),
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


def _branch_donations(
    nodes_arr: np.ndarray,
    adj: dict[int, set[int]],
    degree: np.ndarray,
    node2verts: list[np.ndarray],
    mesh_vertices: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """For each branch node's bin, which vertices face which neighbour: each
    vertex goes to the neighbour its offset projects onto most."""
    donated: dict[tuple[int, int], list[int]] = {}
    for b in range(len(nodes_arr)):
        if degree[b] < 3:
            continue
        bv = np.asarray(node2verts[b], dtype=np.int64)
        if len(bv) < 3:
            continue
        nbrs = sorted(adj[b])
        dirs = nodes_arr[nbrs] - nodes_arr[b]
        ln = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = np.divide(dirs, ln, out=np.zeros_like(dirs), where=ln > 1e-6)
        proj = (mesh_vertices[bv] - nodes_arr[b]) @ dirs.T
        best = np.argmax(proj, axis=1)
        for k, nb in enumerate(nbrs):
            donated[(b, nb)] = bv[best == k].tolist()
    return donated


def _perpendicular_rebin_rs(
    nodes_arr: np.ndarray,
    edges_mst: np.ndarray,
    node2verts: list[np.ndarray],
    mesh_vertices: np.ndarray,
    radius_estimators: list[str],
    radii_dict: dict[str, np.ndarray],
):
    """:func:`_perpendicular_rebin` with the chains on the Rust core; the
    branch donations stay numpy's (their projection goes through BLAS)."""
    n_skel = len(nodes_arr)
    if n_skel < 2 or edges_mst.size == 0:
        return _perpendicular_rebin(
            nodes_arr,
            edges_mst,
            node2verts,
            mesh_vertices,
            radius_estimators,
            radii_dict,
        )
    adj: dict[int, set[int]] = {i: set() for i in range(n_skel)}
    for a, b in edges_mst:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    degree = np.array([len(adj[i]) for i in range(n_skel)])
    donated = _branch_donations(nodes_arr, adj, degree, node2verts, mesh_vertices)
    out, vids, dists = _rs.rebin_chains(
        np.asarray(nodes_arr, dtype=np.float64),
        np.asarray(edges_mst, dtype=np.int64).reshape(-1, 2),
        [np.asarray(v, dtype=np.int64).tolist() for v in node2verts],
        mesh_vertices,
        np.asarray(radii_dict[radius_estimators[0]], dtype=np.float64),
        [(b, nb, v) for (b, nb), v in donated.items()],
    )
    shells = [[v] for v in out if len(v)]
    return shells, dict(zip(vids.tolist(), dists.tolist()))


def _perpendicular_rebin(
    nodes_arr: np.ndarray,
    edges_mst: np.ndarray,
    node2verts: list[np.ndarray],
    mesh_vertices: np.ndarray,
    radius_estimators: list[str],
    radii_dict: dict[str, np.ndarray],
) -> tuple[
    list[list[np.ndarray]],
    dict[int, int],
]:
    """Re-assign vertices along skeleton chains for perpendicular
    bins.

    For every degree-2 path (chain) between branch points / tips,
    project all chain vertices onto the piecewise-linear skeleton
    path and assign each vertex to the nearest node by arc-length
    distance (radius-weighted so thick nodes claim proportionally
    more territory).

    Returns new shells (same structure as ``_bin_one_component``)
    and an updated ``vert2node`` mapping.
    """
    n_skel = len(nodes_arr)
    if n_skel < 2 or edges_mst.size == 0:
        return [[n2v] for n2v in node2verts], {
            int(v): i for i, n2v in enumerate(node2verts) for v in n2v
        }

    # -- skeleton adjacency & degree --
    adj: dict[int, set[int]] = {i: set() for i in range(n_skel)}
    for a, b in edges_mst:
        a, b = int(a), int(b)
        adj[a].add(b)
        adj[b].add(a)

    degree = np.array([len(adj[i]) for i in range(n_skel)])
    anchors = set(i for i in range(n_skel) if degree[i] != 2)

    # -- dissolve branch-point bins --
    donated = _branch_donations(nodes_arr, adj, degree, node2verts, mesh_vertices)

    # -- extract chains (degree-2 paths between anchors) --
    chains: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()
    for anchor in sorted(anchors):
        for nbr in sorted(adj[anchor]):
            edge = (min(anchor, nbr), max(anchor, nbr))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            chain = [anchor, nbr]
            cur, prev = nbr, anchor
            while cur not in anchors:
                nexts = [n for n in adj[cur] if n != prev]
                if len(nexts) != 1:
                    break
                prev, cur = cur, nexts[0]
                visited_edges.add((min(prev, cur), max(prev, cur)))
                chain.append(cur)
            chains.append(chain)

    # -- per-chain: project vertices, assign by arc-length --
    # Use first radius estimator for weighting
    r_key = radius_estimators[0]
    r_ref = radii_dict[r_key]

    new_node2verts: list[list[int]] = [[] for _ in range(n_skel)]
    centerline_dists: list[list[float]] = [[] for _ in range(n_skel)]
    assigned = np.zeros(len(mesh_vertices), dtype=bool)

    for chain in chains:
        # Collect all vertices along the chain
        all_vids_set: set[int] = set()
        for pos, ni in enumerate(chain):
            key = None
            if degree[ni] >= 3 and (pos == 0 or pos == len(chain) - 1):
                key = (ni, chain[1] if pos == 0 else chain[-2])
            if key is not None and key in donated:
                all_vids_set.update(donated[key])
            else:
                all_vids_set.update(int(v) for v in node2verts[ni])
        all_vids = [v for v in sorted(all_vids_set) if not assigned[v]]
        if not all_vids:
            continue

        all_vids_arr = np.asarray(all_vids, dtype=np.int64)
        all_pts = mesh_vertices[all_vids_arr]

        # Piecewise-linear skeleton path
        path_pts = np.array([nodes_arr[n] for n in chain])
        seg_lens = np.linalg.norm(np.diff(path_pts, axis=0), axis=1)
        arc_cum = np.concatenate(([0.0], np.cumsum(seg_lens)))

        # Project each vertex onto the path
        best_dist = np.full(len(all_vids), np.inf)
        best_t = np.zeros(len(all_vids))
        for si in range(len(chain) - 1):
            p0, p1 = path_pts[si], path_pts[si + 1]
            s = seg_lens[si]
            if s < 1e-10:
                continue
            sd = (p1 - p0) / s
            df = all_pts - p0
            tp = np.clip(df @ sd, 0, s)
            d = np.linalg.norm(
                all_pts - (p0 + np.outer(tp, sd)),
                axis=1,
            )
            arc_at = arc_cum[si] + tp
            closer = d < best_dist
            best_dist[closer] = d[closer]
            best_t[closer] = arc_at[closer]

        # Radius-weighted assignment
        node_arcs = np.array([arc_cum[i] for i in range(len(chain))])
        r_floor = np.median(r_ref[r_ref > 0]) * 0.01
        node_radii = np.array(
            [max(r_ref[chain[i]], r_floor) for i in range(len(chain))]
        )

        arc_diffs = np.abs(best_t[:, None] - node_arcs[None, :])
        weighted = arc_diffs / node_radii[None, :]
        nearest_idx = np.argmin(weighted, axis=1)

        for vi, ci_idx in enumerate(nearest_idx):
            ni = chain[ci_idx]
            new_node2verts[ni].append(all_vids[vi])
            centerline_dists[ni].append(best_dist[vi])
            assigned[all_vids[vi]] = True

    # Assign any remaining unassigned anchor vertices
    for ni in sorted(anchors):
        for v in node2verts[ni]:
            if not assigned[int(v)]:
                new_node2verts[ni].append(int(v))
                assigned[int(v)] = True

    # Convert to shells format (one band per node, one
    # component per band) for re-use with _is_ring / _make_nodes
    shells: list[list[np.ndarray]] = []
    for ni in range(n_skel):
        vids = new_node2verts[ni]
        if vids:
            shells.append([np.asarray(vids, dtype=np.int64)])

    # Per-vertex centerline distance map
    vid_cl_dist: dict[int, float] = {}
    for ni in range(n_skel):
        for v, d in zip(new_node2verts[ni], centerline_dists[ni]):
            vid_cl_dist[int(v)] = d

    return shells, vid_cl_dist


def _skeletonize_component(
    mesh: trimesh.Trimesh,
    gsurf: ig.Graph,
    vert_ids: np.ndarray,
    *,
    seed_vid: int | np.ndarray,
    soma_verts: set[int] | None = None,
    radius_estimators: list[str],
    merge_nested: bool = True,
    merge_kwargs: dict | None = None,
    step_size: float | None = None,
    target_shell_count: int = 500,
    min_shell_vertices: int = 6,
    max_shell_width_factor: float = 50.0,
    split_elongated_shells: bool = True,
    split_aspect_thr: float = 3.0,
    split_min_shell_vertices: int = 50,
    split_max_vertices_per_slice: int | None = None,
    second_pass: bool = True,
    stage=_no_stage,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    np.ndarray,
    dict[int, float],
]:
    """Bin, build nodes, and extract edges for one component.

    Combines :func:`_bin_one_component`, :func:`_make_nodes`,
    and :func:`_edges_from_mesh` into a single call that
    produces a complete sub-skeleton for one connected vertex
    set.

    Parameters
    ----------
    mesh
        The neuron mesh (needed for ``edges_unique`` and
        ``vertices``).
    gsurf
        Full surface graph of the mesh.
    vert_ids
        1-D int64 array — vertex IDs for this component.
    seed_vid
        Where to start the geodesic.  A single ``int`` or a
        1-D array for multi-source seeding.
    soma_verts
        Vertices to exclude from shells (``None`` = none).
    radius_estimators
        Passed to :func:`_make_nodes`.
    merge_nested, merge_kwargs
        Passed to :func:`_make_nodes`.

    Returns
    -------
    nodes_arr, radii_dict, node2verts, vert2node, edges_arr, vid_cl_dist

    ``vid_cl_dist`` maps mesh vertex -> perpendicular distance to the
    centreline, and is empty when the second pass does not run.  It is
    returned rather than consumed because it is the only input to the
    ``centerline`` radius that cannot be recovered from the vertices alone;
    keeping it lets that radius be recomputed after a bin is edited.
    """
    if isinstance(gsurf, _rs.SurfaceGraph):
        if soma_verts or merge_nested or split_elongated_shells:
            raise ValueError(
                "the Rust surface graph serves the preprocessing track only "
                "(no soma exclusion, nested merge or elongated-shell split)"
            )
        return _skeletonize_component_rs(
            mesh,
            gsurf,
            vert_ids,
            seed_vid=seed_vid,
            radius_estimators=radius_estimators,
            step_size=step_size,
            target_shell_count=target_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=max_shell_width_factor,
            second_pass=second_pass,
            stage=stage,
        )
    mesh_vertices = mesh.vertices.view(np.ndarray)
    e_m = float(mesh.edges_unique_length.mean())
    vid_cl_dist: dict[int, float] = {}

    with stage("bin vertices by geodesic distance"):
        all_shells = _bin_one_component(
            gsurf,
            vert_ids,
            seed_vid,
            mesh_vertices=mesh_vertices,
            mean_edge_len=e_m,
            soma_verts=soma_verts,
            step_size=step_size,
            target_shell_count=target_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=max_shell_width_factor,
            split_elongated_shells=split_elongated_shells,
            split_aspect_thr=split_aspect_thr,
            split_min_shell_vertices=(split_min_shell_vertices),
            split_max_vertices_per_slice=(split_max_vertices_per_slice),
        )

    with stage("compute bin centroids and radii"):
        nodes_arr, radii_dict, node2verts, vert2node = _make_nodes(
            all_shells,
            mesh_vertices,
            radius_estimators=radius_estimators,
            merge_nested=merge_nested,
            merge_kwargs=merge_kwargs,
        )

    with stage("derive edges from mesh edges"):
        edges_arr = _edges_from_mesh(
            mesh.edges_unique,
            vert2node,
            n_mesh_verts=len(mesh.vertices),
        )

    # -- second pass: perpendicular re-binning --------
    if second_pass and len(nodes_arr) > 1 and edges_arr.size:
        with stage("re-bin perpendicular to the centreline"):
            edges_mst = _build_mst(nodes_arr, edges_arr)

            rebin_shells, vid_cl_dist = _perpendicular_rebin(
                nodes_arr,
                edges_mst,
                node2verts,
                mesh_vertices,
                radius_estimators,
                radii_dict,
            )

        with stage("re-check rings, merge non-rings"):
            # Re-check rings, merge non-rings
            vert_set = set(int(x) for x in vert_ids)
            final_shells: List[List[np.ndarray]] = []
            pending: List[np.ndarray] = []
            for band in rebin_shells:
                rings = []
                frags = []
                for c in band:
                    if _is_ring(c, gsurf, vert_set):
                        rings.append(c)
                    else:
                        frags.append(c)
                if frags:
                    pending.extend(frags)
                if rings:
                    final_shells.append(rings)

            if pending:
                vid_to_ring: dict[int, tuple[int, int]] = {}
                for bi, band in enumerate(final_shells):
                    for ci, ring in enumerate(band):
                        for vid in ring:
                            vid_to_ring[int(vid)] = (bi, ci)
                for frag in pending:
                    votes: dict[tuple[int, int], int] = {}
                    for vid in frag:
                        for e in gsurf.incident(int(vid)):
                            src = gsurf.es[e].source
                            tgt = gsurf.es[e].target
                            nbr = tgt if src == int(vid) else src
                            key = vid_to_ring.get(nbr)
                            if key is not None:
                                votes[key] = votes.get(key, 0) + 1
                    if votes:
                        best = max(votes, key=votes.get)
                        bi, ci = best
                        final_shells[bi][ci] = np.concatenate(
                            [final_shells[bi][ci], frag]
                        )
                        for vid in frag:
                            vid_to_ring[int(vid)] = best

        with stage("reunite bins split across the surface"):
            # Arc-length assignment has no notion of the surface: a vertex
            # whose projection lands near a node is claimed by it even when
            # it is a micron away across the mesh.  That leaves bins in
            # several disconnected pieces and drags the node's centroid into
            # the empty space between them.  Keep each bin's largest
            # connected piece and hand every stray piece to a bin it
            # actually touches.  Connectivity is categorical — no threshold.
            owner_of: dict[int, tuple[int, int]] = {}
            for bi, band in enumerate(final_shells):
                for ci, comp in enumerate(band):
                    for vid in comp:
                        owner_of[int(vid)] = (bi, ci)

            def _pieces(comp):
                vset = set(int(v) for v in comp)
                seen: set[int] = set()
                out: list[list[int]] = []
                for start in sorted(vset):
                    if start in seen:
                        continue
                    queue = deque([start])
                    seen.add(start)
                    grp = [start]
                    while queue:
                        u = queue.popleft()
                        for e in gsurf.incident(u):
                            src = gsurf.es[e].source
                            tgt = gsurf.es[e].target
                            nb = tgt if src == u else src
                            if nb in vset and nb not in seen:
                                seen.add(nb)
                                queue.append(nb)
                                grp.append(nb)
                    out.append(grp)
                return sorted(out, key=len, reverse=True)

            # Donating a stray can disconnect the bin that receives it, so a
            # single pass does not converge — repeat until every bin is one
            # piece, refreshing ownership each round.
            for _ in range(8):
                changed = False
                for bi, band in enumerate(final_shells):
                    for ci, comp in enumerate(band):
                        if len(comp) < 2:
                            continue
                        parts = _pieces(comp)
                        if len(parts) < 2:
                            continue
                        band[ci] = np.asarray(parts[0], dtype=comp.dtype)
                        for vid in parts[0]:
                            owner_of[int(vid)] = (bi, ci)
                        for stray in parts[1:]:
                            votes: dict[tuple[int, int], int] = {}
                            for vid in stray:
                                for e in gsurf.incident(vid):
                                    src = gsurf.es[e].source
                                    tgt = gsurf.es[e].target
                                    nb = tgt if src == vid else src
                                    key = owner_of.get(nb)
                                    if key is not None and key != (bi, ci):
                                        votes[key] = votes.get(key, 0) + 1
                            if not votes:
                                band[ci] = np.concatenate(
                                    [band[ci], np.asarray(stray, dtype=np.int64)]
                                )
                                continue
                            tb, tc = max(votes, key=lambda k: votes[k])
                            final_shells[tb][tc] = np.concatenate(
                                [
                                    final_shells[tb][tc],
                                    np.asarray(stray, dtype=np.int64),
                                ]
                            )
                            for vid in stray:
                                owner_of[int(vid)] = (tb, tc)
                            changed = True
                if not changed:
                    break

        with stage("split bins that wrap a branch point"):
            # A bin that touches three or more separate neighbourhoods is not
            # a cross-section: it is the band wrapping a branch point, holding
            # the parent tube and both children at once.  Averaging points on
            # diverging tubes puts its node in the notch between them, on or
            # past the surface.  Split it so each piece wraps one tube.  The
            # number of neighbourhoods is a count, not a threshold.
            # One pass, not a loop: the split already sends every vertex to the
            # tube nearest it, so there is nothing left to cut.  Repeating it
            # only shaves slivers off, because the pieces of a split are bins
            # in their own right and each sibling then reads as another
            # neighbourhood.  For the same reason every decision is made
            # against a snapshot of ownership taken before any bin is cut, so
            # the result does not depend on the order bins are visited.
            snapshot = dict(owner_of)
            planned: list[tuple[int, int, list[np.ndarray]]] = []
            for bi, band in enumerate(final_shells):
                for ci, comp in enumerate(band):
                    if len(comp) < 3:
                        continue
                    groups = _neighbour_groups(comp, gsurf, snapshot, (bi, ci))
                    if len(groups) < 3:
                        continue
                    parts = _split_branch_band(comp, groups, gsurf)
                    if len(parts) > 1:
                        planned.append((bi, ci, parts))

            for bi, ci, parts in planned:
                band = final_shells[bi]
                band[ci] = parts[0]
                for vid in parts[0]:
                    owner_of[int(vid)] = (bi, ci)
                for extra in parts[1:]:
                    band.append(extra)
                    for vid in extra:
                        owner_of[int(vid)] = (bi, len(band) - 1)

        with stage("recompute centroids and centerline radii"):
            # Remake nodes and edges with centerline radii
            nodes_arr, radii_dict, node2verts, vert2node = _make_nodes(
                final_shells,
                mesh_vertices,
                radius_estimators=radius_estimators,
                merge_nested=merge_nested,
                merge_kwargs=merge_kwargs,
            )

            # Compute centerline radii (perpendicular distance
            # from each vertex to the skeleton path)
            cl_radii = np.empty(len(nodes_arr))
            for ni, n2v in enumerate(node2verts):
                dists = np.array([vid_cl_dist.get(int(v), 0.0) for v in n2v])
                if len(dists) > 0:
                    cl_radii[ni] = _estimate_radius(
                        dists,
                        method="trim",
                        trim_fraction=TRIM_FRACTION,
                    )
                else:
                    cl_radii[ni] = 0.0
            radii_dict["centerline"] = cl_radii

        with stage("re-derive edges from mesh edges"):
            edges_arr = _edges_from_mesh(
                mesh.edges_unique,
                vert2node,
                n_mesh_verts=len(mesh.vertices),
            )

    return (
        nodes_arr,
        radii_dict,
        node2verts,
        vert2node,
        edges_arr,
        vid_cl_dist,
    )


def _make_nodes_rs(shells, vertices, radius_estimators):
    """:func:`_make_nodes` without the nested merge, on the Rust core."""
    node2verts = [np.asarray(b, dtype=np.int64) for band in shells for b in band]
    if not node2verts:
        empty = {k: np.asarray(np.array([])) for k in radius_estimators}
        return np.asarray([], dtype=np.float64), empty, [], {}
    nodes, radii = _rs.skeleton_nodes(
        [[b.tolist() for b in band] for band in shells],
        vertices,
        list(radius_estimators),
    )
    vert2node: dict[int, int] = {}
    for nid, bin_ids in enumerate(node2verts):
        for vid in bin_ids.tolist():
            vert2node[vid] = nid
    return nodes, dict(zip(radius_estimators, radii)), node2verts, vert2node


def _skeletonize_component_rs(
    mesh: trimesh.Trimesh,
    sgraph,
    vert_ids: np.ndarray,
    *,
    seed_vid: int | np.ndarray,
    radius_estimators: list[str],
    step_size: float | None,
    target_shell_count: int,
    min_shell_vertices: int,
    max_shell_width_factor: float,
    second_pass: bool,
    stage,
):
    """:func:`_skeletonize_component` on the preprocessing track with the
    binning, the bin clean-up and the nodes in the Rust core; the spanning
    tree, the perpendicular re-binning and the edges stay as they are."""
    mesh_vertices = mesh.vertices.view(np.ndarray)
    e_m = float(mesh.edges_unique_length.mean())
    vid_cl_dist: dict[int, float] = {}
    verts = np.asarray(vert_ids, dtype=np.int64)

    with stage("bin vertices by geodesic distance"):
        all_shells = sgraph.bin_component(
            verts,
            np.atleast_1d(np.asarray(seed_vid, dtype=np.int64)),
            e_m,
            step_size,
            target_shell_count,
            min_shell_vertices,
            max_shell_width_factor,
        )

    with stage("compute bin centroids and radii"):
        nodes_arr, radii_dict, node2verts, vert2node = _make_nodes_rs(
            all_shells, mesh_vertices, radius_estimators
        )

    with stage("derive edges from mesh edges"):
        edges_arr = _edges_from_mesh(
            mesh.edges_unique, vert2node, n_mesh_verts=len(mesh.vertices)
        )

    if second_pass and len(nodes_arr) > 1 and edges_arr.size:
        with stage("re-bin perpendicular to the centreline"):
            edges_mst = _build_mst(nodes_arr, edges_arr)
            rebin_shells, vid_cl_dist = _perpendicular_rebin_rs(
                nodes_arr,
                edges_mst,
                node2verts,
                mesh_vertices,
                radius_estimators,
                radii_dict,
            )

        def _lists(shells):
            return [[b.tolist() for b in band] for band in shells]

        with stage("re-check rings, merge non-rings"):
            final_shells = sgraph.recheck_rings(_lists(rebin_shells), verts)

        with stage("reunite bins split across the surface"):
            final_shells = sgraph.reunite_bins(_lists(final_shells))

        with stage("split bins that wrap a branch point"):
            final_shells = sgraph.split_branch_bins(_lists(final_shells))

        with stage("recompute centroids and centerline radii"):
            nodes_arr, radii_dict, node2verts, vert2node = _make_nodes_rs(
                final_shells, mesh_vertices, radius_estimators
            )
            cl_radii = np.empty(len(nodes_arr))
            for ni, n2v in enumerate(node2verts):
                dists = np.array([vid_cl_dist.get(int(v), 0.0) for v in n2v])
                if len(dists) > 0:
                    cl_radii[ni] = _estimate_radius(
                        dists, method="trim", trim_fraction=TRIM_FRACTION
                    )
                else:
                    cl_radii[ni] = 0.0
            radii_dict["centerline"] = cl_radii

        with stage("re-derive edges from mesh edges"):
            edges_arr = _edges_from_mesh(
                mesh.edges_unique, vert2node, n_mesh_verts=len(mesh.vertices)
            )

    return nodes_arr, radii_dict, node2verts, vert2node, edges_arr, vid_cl_dist


# -----------------------------------------------------------------------------
#  Preprocessing track helpers
# -----------------------------------------------------------------------------


def _concat_radii(
    parts: list[dict[str, np.ndarray]],
    radius_estimators: list[str],
    *,
    lead_value: float | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate per-neurite radii, keeping keys beyond the estimators.

    The second binning pass writes a ``centerline`` entry that is not in
    ``radius_estimators``; collecting only the estimators silently drops it.

    A neurite too small for the second pass has no ``centerline`` of its
    own, so its nodes fall back to the first estimator — that keeps every
    radius array the same length as ``nodes`` and finite, which exporters
    and :meth:`Skeleton.recommend_radius` both rely on.

    Parameters
    ----------
    parts : list of dict
        One radii dict per sub-skeleton, in node order.
    radius_estimators : list of str
        Keys guaranteed to be present on every sub-skeleton.
    lead_value : float or None
        Radius for a single node prepended before the parts (the soma).
    """
    keys = list(radius_estimators)
    for radii in parts:
        for k in radii:
            if k not in keys:
                keys.append(k)
    fallback = radius_estimators[0]

    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    if lead_value is not None:
        for k in keys:
            out[k].append(np.array([lead_value], dtype=np.float64))
    for radii in parts:
        for k in keys:
            v = radii.get(k, radii[fallback])
            out[k].append(np.asarray(v, dtype=np.float64))
    return {
        k: (np.concatenate(v) if v else np.empty(0, dtype=np.float64))
        for k, v in out.items()
    }


def _stitch_to_soma(
    sub_skeletons: list[
        tuple[
            np.ndarray,
            dict[str, np.ndarray],
            list[np.ndarray],
            dict[int, int],
            np.ndarray,
            dict[int, float],
        ]
    ],
    soma: Soma,
    radius_estimators: list[str],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    np.ndarray,
]:
    """Combine per-neurite sub-skeletons with soma node 0.

    Each sub-skeleton's node IDs are offset so they sit after
    the soma and all earlier sub-skeletons.  One synthetic
    stem edge ``(0, root_i)`` is added per neurite, where
    ``root_i`` is the node closest to the soma centre.

    Returns
    -------
    nodes, radii, node2verts, vert2node, edges
    """
    r_soma = soma.spherical_radius

    # -- node 0 = soma --
    soma_verts_arr = soma.verts if soma.verts is not None else np.empty(0, np.int64)
    all_nodes = [soma.center.reshape(1, 3)]
    radii_parts: list[dict[str, np.ndarray]] = []
    all_n2v: list[np.ndarray] = [soma_verts_arr]
    all_edges: list[np.ndarray] = []

    offset = 1  # node 0 is soma
    for nodes, radii, n2v, _, edges, _cl in sub_skeletons:
        if len(nodes) == 0:
            continue

        all_nodes.append(nodes)
        radii_parts.append(radii)
        all_n2v.extend(n2v)

        # offset edge endpoints
        all_edges.append(edges + offset)

        # stem edge: soma → closest node
        dists = np.linalg.norm(nodes - soma.center, axis=1)
        root = int(np.argmin(dists)) + offset
        all_edges.append(np.array([[0, root]], dtype=np.int64))

        offset += len(nodes)

    nodes_arr = np.vstack(all_nodes) if all_nodes else np.empty((0, 3), np.float64)
    radii_dict = _concat_radii(radii_parts, radius_estimators, lead_value=r_soma)
    edges_arr = np.vstack(all_edges) if all_edges else np.empty((0, 2), np.int64)
    edges_arr = np.sort(edges_arr, axis=1)
    edges_arr = np.unique(edges_arr, axis=0)

    vert2node: dict[int, int] = {}
    for nid, verts in enumerate(all_n2v):
        for vid in verts:
            vert2node[int(vid)] = nid

    return (
        nodes_arr,
        radii_dict,
        all_n2v,
        vert2node,
        edges_arr,
    )


def _pick_neurite_seed(
    neurite_verts: np.ndarray,
    mesh_vertices: np.ndarray,
    soma: Soma | None,
) -> int | np.ndarray:
    """Choose a geodesic seed for one neurite.

    With soma: return the full boundary ring (all neurite
    vertices shared with the soma surface).  Multi-source
    seeding from the ring produces perpendicular shells.
    Without soma: return the single vertex closest to the
    neurite centroid.
    """
    if soma is not None and soma.verts is not None:
        boundary = np.intersect1d(neurite_verts, soma.verts)
        if boundary.size > 1:
            return boundary
        if boundary.size == 1:
            return int(boundary[0])
        # no shared verts — fall back to nearest
        dists = np.linalg.norm(
            mesh_vertices[neurite_verts] - soma.center,
            axis=1,
        )
        return int(neurite_verts[np.argmin(dists)])

    # no soma — seed at centroid of the neurite
    centroid = mesh_vertices[neurite_verts].mean(axis=0)
    dists = np.linalg.norm(mesh_vertices[neurite_verts] - centroid, axis=1)
    return int(neurite_verts[np.argmin(dists)])


def _concatenate_sub_skeletons(
    sub_skeletons: list[
        tuple[
            np.ndarray,
            dict[str, np.ndarray],
            list[np.ndarray],
            dict[int, int],
            np.ndarray,
            dict[int, float],
        ]
    ],
    radius_estimators: list[str],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    np.ndarray,
]:
    """Join per-neurite sub-skeletons without a soma node.

    Unlike :func:`_stitch_to_soma`, no node 0 is prepended
    and no stem edges are added.  The result is a forest
    (one tree per neurite).
    """
    all_nodes: list[np.ndarray] = []
    radii_parts: list[dict[str, np.ndarray]] = []
    all_n2v: list[np.ndarray] = []
    all_edges: list[np.ndarray] = []

    offset = 0
    for nodes, radii, n2v, _, edges, _cl in sub_skeletons:
        if len(nodes) == 0:
            continue
        all_nodes.append(nodes)
        radii_parts.append(radii)
        all_n2v.extend(n2v)
        if edges.size:
            all_edges.append(edges + offset)
        offset += len(nodes)

    nodes_arr = np.vstack(all_nodes) if all_nodes else np.empty((0, 3), np.float64)
    radii_dict = _concat_radii(radii_parts, radius_estimators)
    edges_arr = np.vstack(all_edges) if all_edges else np.empty((0, 2), np.int64)
    if edges_arr.size:
        edges_arr = np.sort(edges_arr, axis=1)
        edges_arr = np.unique(edges_arr, axis=0)

    vert2node: dict[int, int] = {}
    for nid, verts in enumerate(all_n2v):
        for vid in verts:
            vert2node[int(vid)] = nid

    return (
        nodes_arr,
        radii_dict,
        all_n2v,
        vert2node,
        edges_arr,
    )


def _skeletonize_preproc(
    mesh: trimesh.Trimesh,
    components: MeshComponents,
    *,
    radius_estimators: list[str],
    geodesic_step_size: float | None,
    geodesic_shell_count: int,
    min_shell_vertices: int,
    max_shell_width_factor: float,
    soma_init_guess_axis: str,
    soma_init_guess_mode: str,
    unit: str,
    id: str | int | None,
    verbose: bool,
    second_pass: bool = True,
) -> Skeleton:
    """Preprocessing track: per-neurite skeletonization.

    Each neurite in *components* is skeletonized via
    :func:`_skeletonize_component`, then all sub-skeletons
    are grafted onto the precomputed soma.

    Skips: soma detection, gap bridging, neurite pruning,
    near-soma collapse, nested-node merge, elongated-shell
    splitting (all unnecessary on clean preprocessed
    neurites).
    """
    # Snapshot the arguments before anything else is bound, so the recorded
    # parameters cannot drift from the signature the way a hand-kept list
    # would.  A stale provenance record is worse than none.
    _params = {
        k: v
        for k, v in locals().items()
        if k not in ("mesh", "components", "unit", "id", "verbose")
    }

    soma = components.soma
    has_soma = soma is not None

    if verbose:
        _global_start = time.perf_counter()
        soma_tag = "with soma" if has_soma else "no soma"
        print(
            f"[skeliner] skeletonize from preprocessed mesh components "
            f"({len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} faces, "
            f"{len(components.neurites)} neurites, "
            f"{soma_tag})"
        )

    with _timed("↳  build surface graph", verbose=verbose):
        gsurf = _surface_graph_rs(mesh)

    mesh_vertices = mesh.vertices.view(np.ndarray)

    # Each stage of the per-neurite work reports itself as a top-level
    # step, the way `_skeletonize_direct` does.  Wrapping the loop in one
    # `_timed` instead would report a single number for the slowest part
    # of the run, and `_timed` holds its sub-messages until the block
    # ends, so nothing would appear while it was actually working.
    n_neurites = len(components.neurites)
    sub_skeletons = []
    for i, face_idx in enumerate(components.neurites):
        neurite_verts = np.unique(mesh.faces[face_idx].ravel()).astype(np.int64)

        seed_vid = _pick_neurite_seed(
            neurite_verts,
            mesh_vertices,
            soma,
        )

        tag = f"neurite {i}: " if n_neurites > 1 else ""

        def stage(label, _tag=tag):
            return _timed(f"↳  {_tag}{label}", verbose=verbose)

        sub = _skeletonize_component(
            mesh,
            gsurf,
            neurite_verts,
            seed_vid=seed_vid,
            radius_estimators=radius_estimators,
            merge_nested=False,
            step_size=geodesic_step_size,
            target_shell_count=geodesic_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=(max_shell_width_factor),
            split_elongated_shells=False,
            second_pass=second_pass,
            stage=stage,
        )
        sub_skeletons.append(sub)
        if verbose:
            print(
                f"      └─ neurite {i}: {len(neurite_verts):,} verts "
                f"→ {sub[0].shape[0]:,} nodes"
            )

    # -- assemble skeleton -----------------------------------------
    if has_soma:
        with _timed(
            "↳  stitch neurites to soma",
            verbose=verbose,
        ):
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_arr,
            ) = _stitch_to_soma(sub_skeletons, soma, radius_estimators)
    else:
        with _timed(
            "↳  concatenate neurites",
            verbose=verbose,
        ):
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_arr,
            ) = _concatenate_sub_skeletons(sub_skeletons, radius_estimators)

    # -- keep the per-vertex centreline distances --
    # The `centerline` radius is an aggregate of these, so keeping them is
    # what lets it be recomputed when a bin changes; every other radius is
    # already recoverable from the vertices alone.  Vertex ids are global,
    # so the per-neurite maps merge without any offsetting.
    cl_vids: list[int] = []
    cl_dists: list[float] = []
    for *_rest, cl in sub_skeletons:
        cl_vids.extend(cl.keys())
        cl_dists.extend(cl.values())

    # -- global MST --
    with _timed(
        "↳  build global minimum-spanning tree",
        verbose=verbose,
    ):
        if len(nodes_arr) > 1 and edges_arr.size:
            edges_mst = _build_mst(nodes_arr, edges_arr)
        else:
            edges_mst = edges_arr

    if verbose:
        total = time.perf_counter() - _global_start
        print(f"{'TOTAL':<49}… {total:.2f} s")
        print(f"({len(nodes_arr):,} nodes, {edges_mst.shape[0]:,} edges)")

    ntype = np.zeros(len(nodes_arr), np.int8)
    if has_soma:
        ntype[0] = 1
    elif len(nodes_arr):
        ntype[0] = -1
        # pick a deterministic root like the direct track
        root_vid = _extreme_vertex(
            mesh,
            axis=soma_init_guess_axis,
            mode=soma_init_guess_mode,
        )
        if root_vid in vert2node:
            root_nid = vert2node[root_vid]
        else:
            # extreme vertex not in any neurite — pick
            # the closest skeleton node instead
            dists = np.linalg.norm(
                nodes_arr - mesh_vertices[root_vid],
                axis=1,
            )
            root_nid = int(np.argmin(dists))
        if root_nid != 0:
            # swap node 0 and root_nid
            nodes_arr[[0, root_nid]] = nodes_arr[[root_nid, 0]]
            for k in radii_dict:
                radii_dict[k][[0, root_nid]] = radii_dict[k][[root_nid, 0]]
            node2verts[0], node2verts[root_nid] = (
                node2verts[root_nid],
                node2verts[0],
            )
            m0 = edges_mst == 0
            mi = edges_mst == root_nid
            edges_mst[m0] = root_nid
            edges_mst[mi] = 0
            edges_mst = np.sort(edges_mst, axis=1)
            vert2node = rebuild_vert2node(node2verts)
        # placeholder soma at the root
        r0 = float(list(radii_dict.values())[0][0])
        soma = Soma.from_sphere(nodes_arr[0], r0, verts=None)

    # -- carry the neurites' hand-given identity onto the skeleton --
    # Read through `vert2node` rather than by re-deriving the node ranges:
    # the assembly lays sub-skeletons down in order, but skips any that came
    # out empty, so an index into `components.neurites` is not an index into
    # the node blocks.  Ownership is exact and needs no such assumption.
    #
    # Node 0 is excluded throughout: under the >=2-of-3 face rule a face at
    # the soma junction carries soma vertices, so a neurite's vertex set
    # reaches node 0, and stamping it would retype the soma as a dendrite.
    #
    # Three things are recorded, because `ntype` alone loses information:
    # it collapses "dendrite 0" and "dendrite 1" into the same code 3, and
    # SWC has no field for a name.  The labels go in `meta`, which both the
    # npz and the SWC header carry, and the per-node owner goes in `extra`,
    # which is what lets a name be resolved back to nodes.
    skel_meta = _skel_meta("preproc", unit, id, _params, mesh)
    skel_extra: dict[str, Any] = (
        {
            "cl_dist_vids": np.asarray(cl_vids, dtype=np.int64),
            "cl_dist_vals": np.asarray(cl_dists, dtype=np.float64),
        }
        if cl_vids
        else {}
    )

    if components.neurites.named:
        node2neurite = np.full(len(nodes_arr), -1, dtype=np.int32)
        for ni, (face_idx, code) in enumerate(
            zip(components.neurites.components, components.neurites.swc_types)
        ):
            nids = {
                vert2node[int(v)]
                for v in np.unique(mesh.faces[face_idx].ravel())
                if int(v) in vert2node
            }
            nids.discard(0)
            if not nids:
                continue
            owned = np.fromiter(nids, dtype=np.int64, count=len(nids))
            node2neurite[owned] = ni
            if code:
                ntype[owned] = code
        skel_meta["neurite_labels"] = list(components.neurites.labels)
        skel_meta["neurite_swc_types"] = [int(c) for c in components.neurites.swc_types]
        skel_extra["node2neurite"] = node2neurite

    return Skeleton(
        nodes=nodes_arr,
        radii=radii_dict,
        edges=edges_mst,
        ntype=ntype,
        soma=soma,
        node2verts=node2verts,
        vert2node=vert2node,
        meta=skel_meta,
        extra=skel_extra,
    )


# -----------------------------------------------------------------------------
#  Skeletonization Public API
# -----------------------------------------------------------------------------


def _skeletonize_direct(
    mesh: trimesh.Trimesh,
    # --- radius estimation ---
    radius_estimators: list[str] = ["median", "mean", "trim"],
    # --- soma detection ---
    detect_soma: bool = True,
    soma_seed_point: np.ndarray | list | tuple | None = None,
    soma_radius_percentile_threshold: float = 99.9,
    soma_radius_distance_factor: float = 4,
    soma_min_nodes: int = 3,
    # -- soma seed heuristic for the geodesic origin --
    soma_init_guess: str | None = None,  # "nucleus" | "<axis>-<mode>"
    soma_init_guess_axis: str = "z",  # legacy, when soma_init_guess is None
    soma_init_guess_mode: str = "min",  # legacy, when soma_init_guess is None
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
    """Direct track: full pipeline from raw mesh.

    Stages: surface graph, geodesic binning, node creation,
    soma detection, edge mapping, near-soma collapse, gap
    bridging, MST, neurite pruning.
    """
    # See _skeletonize_preproc: snapshot the arguments before anything else
    # is bound, so the record cannot drift from the signature.
    _params = {
        k: v for k, v in locals().items() if k not in ("mesh", "unit", "id", "verbose")
    }

    # ------------------------------------------------------------------
    #  Direct track: helpers for verbose timing
    # ------------------------------------------------------------------
    if verbose:
        _global_start = time.perf_counter()
        # Named for what it consumes, and matching how the other track names
        # itself: which of the two ran decides which parameters were
        # consulted, so a header saying only "starting" cannot answer it.
        print(
            f"[skeliner] skeletonize from raw mesh "
            f"({len(mesh.vertices):,} vertices, "
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

    # 0. build surface graph -----------------------------------------------
    with _timed("↳  build surface graph", verbose=verbose):
        gsurf = _surface_graph(mesh)

    # 1. binning surface vertices by geodesic distance ----------------------------------
    with _timed("↳  bin surface vertices by geodesic distance", verbose=verbose):
        mesh_vertices = mesh.vertices.view(np.ndarray)

        # -- resolve soma_init_guess → soma_seed_point ----
        if soma_seed_point is None and soma_init_guess is not None:
            if soma_init_guess == "nucleus":
                from .pre import find_nucleus_center

                nuc = find_nucleus_center(mesh, verbose=verbose)
                if nuc is not None:
                    soma_seed_point = nuc["center"]
            elif "-" in soma_init_guess:
                ax, md = soma_init_guess.split("-", 1)
                soma_init_guess_axis = ax
                soma_init_guess_mode = md

        # -- pick seed vertex -----------------------------
        if soma_seed_point is not None:
            seed_vid = int(
                np.argmin(
                    np.linalg.norm(
                        mesh_vertices - np.asarray(soma_seed_point),
                        axis=1,
                    )
                )
            )
        else:
            seed_vid = _extreme_vertex(
                mesh,
                axis=soma_init_guess_axis,
                mode=soma_init_guess_mode,
            )

        all_shells = _bin_geodesic_shells(
            mesh,
            gsurf,
            seed_vid=seed_vid,
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
    with _timed("↳  post-skeletonization soma detection", verbose=verbose) as log:
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

    ntype = np.zeros(len(nodes_arr), np.int8)
    ntype[0] = 1 if has_soma else -1

    return Skeleton(
        nodes=nodes_arr,
        radii=radii_dict,
        edges=edges_mst,
        ntype=ntype,
        soma=soma,
        node2verts=node2verts,
        vert2node=vert2node,
        meta=_skel_meta("direct", unit, id, _params, mesh),
    )


def skeletonize(
    mesh: trimesh.Trimesh,
    # --- preprocessing track ---
    components: MeshComponents | None = None,
    # --- radius estimation ---
    radius_estimators: list[str] = ["median", "mean", "trim"],
    # --- soma detection ---
    detect_soma: bool = True,
    soma_seed_point: np.ndarray | list | tuple | None = None,
    soma_radius_percentile_threshold: float = 99.9,
    soma_radius_distance_factor: float = 4,
    soma_min_nodes: int = 3,
    # -- soma seed heuristic for the geodesic origin --
    soma_init_guess: str | None = None,  # "nucleus" | "<axis>-<mode>"
    soma_init_guess_axis: str = "z",  # legacy, when soma_init_guess is None
    soma_init_guess_mode: str = "min",  # legacy, when soma_init_guess is None
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
    second_pass: bool = True,
) -> Skeleton:
    """Compute a center-line skeleton with radii from a neuronal mesh.

    Two tracks are available, named for what they read and selected by
    the *components* parameter:

    **From a raw mesh** (``components=None``, the default):
      Full pipeline — geodesic binning, post-skel soma detection,
      gap bridging, MST, neurite pruning.  Works on any raw mesh.

    **From preprocessed mesh components**
    (``components=MeshComponents(...)``):
      Per-neurite skeletonization using the soma and neurite
      partition from :func:`~skeliner.pre.preprocess` or
      :func:`~skeliner.pre.break_up_mesh`.  Skips soma detection,
      gap bridging, and pruning (all handled upstream).  Supports
      meshes with or without a soma.

    Which one ran is the first line of the verbose log, because it
    decides which of the parameters below were consulted at all.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface mesh of the neuron.
    components : MeshComponents or None
        If provided, skeletonize from preprocessed mesh components.
        The soma and neurite partition are taken from *components*;
        most other parameters (bridging, pruning, soma detection)
        are accepted and then ignored, so setting one against the
        wrong track is silent rather than refused.
    soma_init_guess : str or None
        Soma-seed heuristic when skeletonizing from a raw mesh.
        ``"nucleus"``
        runs :func:`~skeliner.pre.find_nucleus_center` to locate
        the soma before binning.  ``"<axis>-<mode>"`` (e.g.
        ``"z-min"``) selects an extreme vertex.  ``None`` falls
        back to *soma_init_guess_axis* / *soma_init_guess_mode*.
    verbose : bool, default ``False``
        Print per-stage timing.
    second_pass : bool, default ``True``
        Run the perpendicular re-binning pass: the first-pass bins
        build a rough skeleton, then every degree-2 chain is re-binned
        by projecting its vertices onto that path, and a ``centerline``
        radius is recorded.  Roughly halves median bin tilt.
        **Read only when skeletonizing from preprocessed mesh
        components** — ignored when ``components`` is not given.

    Returns
    -------
    Skeleton
        Acyclic skeleton.  Node 0 is the soma centroid (when
        detected) or a deterministic root vertex.
    """
    # ------------------------------------------------------------------
    #  Dispatch to the appropriate track
    # ------------------------------------------------------------------
    if components is not None:
        return _skeletonize_preproc(
            mesh,
            components,
            radius_estimators=radius_estimators,
            geodesic_step_size=geodesic_step_size,
            geodesic_shell_count=geodesic_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=max_shell_width_factor,
            soma_init_guess_axis=soma_init_guess_axis,
            soma_init_guess_mode=soma_init_guess_mode,
            unit=unit,
            id=id,
            verbose=verbose,
            second_pass=second_pass,
        )

    return _skeletonize_direct(
        mesh,
        radius_estimators=radius_estimators,
        detect_soma=detect_soma,
        soma_seed_point=soma_seed_point,
        soma_radius_percentile_threshold=soma_radius_percentile_threshold,
        soma_radius_distance_factor=soma_radius_distance_factor,
        soma_min_nodes=soma_min_nodes,
        soma_init_guess=soma_init_guess,
        soma_init_guess_axis=soma_init_guess_axis,
        soma_init_guess_mode=soma_init_guess_mode,
        geodesic_step_size=geodesic_step_size,
        geodesic_shell_count=geodesic_shell_count,
        min_shell_vertices=min_shell_vertices,
        max_shell_width_factor=max_shell_width_factor,
        split_elongated_shells=split_elongated_shells,
        split_aspect_thr=split_aspect_thr,
        split_min_shell_vertices=split_min_shell_vertices,
        split_max_vertices_per_slice=split_max_vertices_per_slice,
        merge_nodes_overlap_fraction=merge_nodes_overlap_fraction,
        bridge_gaps=bridge_gaps,
        bridge_max_factor=bridge_max_factor,
        bridge_recalc_after=bridge_recalc_after,
        collapse_soma=collapse_soma,
        collapse_soma_dist_factor=collapse_soma_dist_factor,
        collapse_soma_radius_factor=collapse_soma_radius_factor,
        prune_tiny_neurites=prune_tiny_neurites,
        prune_tip_extent_factor=prune_tip_extent_factor,
        prune_stem_extent_factor=prune_stem_extent_factor,
        prune_drop_single_node_branches=prune_drop_single_node_branches,
        unit=unit,
        id=id,
        verbose=verbose,
        postprocess=postprocess,
    )
