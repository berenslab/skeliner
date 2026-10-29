"""``skeliner.skeletonize`` on the Python path: geodesic binning of a mesh
into skeleton nodes, as the Rust crate ``skeletonize`` runs it.

The surface is an igraph graph with one vertex per mesh vertex and one edge
per unique mesh edge, weighted by its length; every walk over it visits a
vertex's neighbours in igraph's order, ascending.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List

import igraph as ig
import numpy as np
from scipy.spatial import KDTree

from .._core import RECOMPUTABLE_RADII, TRIM_FRACTION, _estimate_radius
from .._state import remap_edges
from ..dataclass import Mesh, Soma
from .mesh import (
    KdTree,
    checked_faces,
    edge_lengths,
    node_pairs,
    rows3,
    unique_edges,
)

#: Where a bin sits: shell index, then index within the shell.
Key = tuple[int, int]


def _vertex_ids(ids, n: int, what: str = "vertex index") -> np.ndarray:
    """*ids* as a flat int64 array, each checked to lie in ``[0, n)``."""
    ids = np.asarray(ids, dtype=np.int64).ravel()
    bad = (ids < 0) | (ids >= n)
    if bad.any():
        raise ValueError(f"{what} {ids[bad][0]} is out of range")
    return ids


def _shells(shells, n: int) -> list[list[np.ndarray]]:
    """*shells* as new lists of int64 bins, every vertex checked against *n*."""
    return [[_vertex_ids(b, n) for b in band] for band in shells]


def surface_graph(mesh: Mesh) -> ig.Graph:
    """Return an edge‑weighted triangle‑adjacency graph.

    The graph has one vertex per mesh‑vertex and an undirected edge for every
    unique mesh edge.  Edge weights are the Euclidean lengths which later serve
    as geodesic distances.
    """
    edges = [tuple(map(int, e)) for e in mesh.edges_unique]
    g = ig.Graph(n=len(mesh.vertices), edges=edges, directed=False)
    g.es["weight"] = mesh.edges_unique_length.astype(float).tolist()
    return g


def dist_vec_for_component(
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


def geodesic_bins(dist_dict: Dict[int, float], step: float) -> List[List[int]]:
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


def split_comp_if_elongated(
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


def is_ring(
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
        for nbr in gsurf.neighbors(int(vid)):
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
            for nbr in gsurf.neighbors(u):
                if nbr in ext and nbr not in visited:
                    queue.append(nbr)

    return n_groups >= 2


def owners(shells: list[list[np.ndarray]]) -> dict[int, Key]:
    """Each vertex's bin."""
    return {
        int(v): (bi, ci)
        for bi, band in enumerate(shells)
        for ci, comp in enumerate(band)
        for v in comp
    }


def merge_fragments(
    gsurf: ig.Graph,
    shells: list[list[np.ndarray]],
    pending: list[np.ndarray],
) -> None:
    """Merge each fragment into the ring its vertices touch most (the first
    among equals), in order; a fragment touching no ring is dropped."""
    if not pending:
        return
    owner = owners(shells)
    for frag in pending:
        votes: dict[Key, int] = {}
        for vid in frag:
            for nbr in gsurf.neighbors(int(vid)):
                key = owner.get(nbr)
                if key is not None:
                    votes[key] = votes.get(key, 0) + 1
        if votes:
            best = max(votes, key=votes.get)
            bi, ci = best
            shells[bi][ci] = np.concatenate([shells[bi][ci], frag])
            for vid in frag:
                owner[int(vid)] = best


def pieces(gsurf: ig.Graph, comp) -> list[list[int]]:
    """Connected pieces of *comp*, each in breadth-first order from its
    lowest vertex, the largest first (stable)."""
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
            for nb in gsurf.neighbors(u):
                if nb in vset and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
                    grp.append(nb)
        out.append(grp)
    return sorted(out, key=len, reverse=True)


def neighbour_groups(
    comp_verts: np.ndarray,
    gsurf: ig.Graph,
    owner_of: dict[int, Key],
    own: Key,
) -> list[list[int]]:
    """Group the vertices of other bins that touch this one, by tube.

    Like :func:`is_ring`, but returns the groups instead of stopping at
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
        for nbr in gsurf.neighbors(int(vid)):
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
            for nbr in gsurf.neighbors(u):
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

    first_seen: dict[Key, int] = {}
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


def split_branch_band(
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
            for nbr in gsurf.neighbors(u):
                if nbr in vset and nbr not in label:
                    label[nbr] = gi
                    queue.append(nbr)

    while queue:
        u = queue.popleft()
        gi = label[u]
        for nbr in gsurf.neighbors(u):
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


def bin_one_component(
    gsurf: ig.Graph,
    verts: np.ndarray,
    seed_vid: int | np.ndarray,
    *,
    mesh_vertices: np.ndarray | None,
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
        1-D int64 array of vertex IDs in this component, ascending.
    seed_vid
        Mesh-vertex ID(s) to start the geodesic from.  A single
        ``int`` or a 1-D array for multi-source seeding.
    mesh_vertices
        ``mesh.vertices`` as ``(N, 3) float64``; read only to split
        elongated shells.
    mean_edge_len
        Mean mesh-edge length (for step-size fallback).
    soma_verts
        Vertex IDs to exclude from shells.  ``None`` or
        empty set means no exclusion.

    Returns
    -------
    List[List[np.ndarray]]
        Shells for this component — same structure as the
        return of :func:`bin_geodesic_shells`.
    """
    v = mesh_vertices
    e_m = mean_edge_len
    exclude = soma_verts or set()
    vert_set = set(int(x) for x in verts)

    # -- geodesic distance from seed to every vertex in component --
    dist_vec = dist_vec_for_component(gsurf, verts, seed_vid)
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
        if not np.isfinite(dist_vec).all():
            # numpy's arange cannot reach an infinite distance
            raise ValueError("arange: cannot compute length")
        shells = geodesic_bins(dist_sub, step)
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
                for part in split_comp_if_elongated(
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
            if is_ring(c, gsurf, vert_set):
                rings.append(c)
            else:
                frags.append(c)
        if frags:
            pending_frags.extend(frags)
        comps = rings

        if comps:
            component_shells.append(comps)

    # -- merge pending fragments via mesh-edge connectivity --
    merge_fragments(gsurf, component_shells, pending_frags)
    return component_shells


def bin_geodesic_shells(
    mesh: Mesh,
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
    :func:`bin_one_component`.

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

        shells = bin_one_component(
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


def recheck_rings(
    gsurf: ig.Graph,
    rebinned: list[list[np.ndarray]],
    verts,
) -> list[list[np.ndarray]]:
    """The first clean-up of the perpendicular re-binning's shells: keep the
    rings (over the neurite's *verts*) and merge the fragments into them."""
    vert_set = set(int(x) for x in verts)
    shells: list[list[np.ndarray]] = []
    pending: list[np.ndarray] = []
    for band in rebinned:
        rings = []
        for c in band:
            if is_ring(c, gsurf, vert_set):
                rings.append(c)
            else:
                pending.append(c)
        if rings:
            shells.append(rings)
    merge_fragments(gsurf, shells, pending)
    return shells


def reunite_bins(
    gsurf: ig.Graph, shells: list[list[np.ndarray]]
) -> list[list[np.ndarray]]:
    """Make every bin one connected piece: its largest stays, and each stray
    piece goes to the bin it touches most, or stays if it touches none.

    Arc-length assignment has no notion of the surface: a vertex whose
    projection lands near a node is claimed by it even when it is a micron
    away across the mesh.  That leaves bins in several disconnected pieces
    and drags the node's centroid into the empty space between them.
    Connectivity is categorical — no threshold.

    Donating a stray can disconnect the bin that receives it, so a single
    pass does not converge — it repeats until every bin is one piece, at
    most eight rounds, refreshing ownership each round.
    """
    owner = owners(shells)
    for _ in range(8):
        changed = False
        for bi, band in enumerate(shells):
            for ci, comp in enumerate(band):
                if len(comp) < 2:
                    continue
                parts = pieces(gsurf, comp)
                if len(parts) < 2:
                    continue
                band[ci] = np.asarray(parts[0], dtype=np.int64)
                for vid in parts[0]:
                    owner[vid] = (bi, ci)
                for stray in parts[1:]:
                    votes: dict[Key, int] = {}
                    for vid in stray:
                        for nb in gsurf.neighbors(vid):
                            key = owner.get(nb)
                            if key is not None and key != (bi, ci):
                                votes[key] = votes.get(key, 0) + 1
                    stray_ids = np.asarray(stray, dtype=np.int64)
                    if not votes:
                        band[ci] = np.concatenate([band[ci], stray_ids])
                        continue
                    tb, tc = max(votes, key=votes.get)
                    shells[tb][tc] = np.concatenate([shells[tb][tc], stray_ids])
                    for vid in stray:
                        owner[vid] = (tb, tc)
                    changed = True
        if not changed:
            break
    return shells


def split_branch_bins(
    gsurf: ig.Graph, shells: list[list[np.ndarray]]
) -> list[list[np.ndarray]]:
    """Split each bin that wraps a branch point into one piece per tube.

    A bin that touches three or more separate neighbourhoods is not a
    cross-section: it is the band wrapping a branch point, holding the
    parent tube and both children at once.  Averaging points on diverging
    tubes puts its node in the notch between them, on or past the surface.
    The number of neighbourhoods is a count, not a threshold.

    One pass, not a loop: the split already sends every vertex to the tube
    nearest it, so there is nothing left to cut.  Repeating it only shaves
    slivers off, because the pieces of a split are bins in their own right
    and each sibling then reads as another neighbourhood.  For the same
    reason every decision is made against the ownership before any bin is
    cut, so the result does not depend on the order bins are visited.
    """
    snapshot = owners(shells)
    planned: list[tuple[int, int, list[np.ndarray]]] = []
    for bi, band in enumerate(shells):
        for ci, comp in enumerate(band):
            if len(comp) < 3:
                continue
            groups = neighbour_groups(comp, gsurf, snapshot, (bi, ci))
            if len(groups) < 3:
                continue
            parts = split_branch_band(comp, groups, gsurf)
            if len(parts) > 1:
                planned.append((bi, ci, parts))

    for bi, ci, parts in planned:
        shells[bi][ci] = parts[0]
        shells[bi].extend(parts[1:])
    return shells


class SurfaceGraph:
    """``skeliner.skeletonize``'s surface graph: one vertex per mesh vertex,
    one edge per unique mesh edge weighted by its length.

    ``graph`` is the igraph graph the methods walk.
    """

    def __init__(self, n_vertices: int, edges, weights):
        edges = np.array(edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be (E, 2)")
        weights = np.array(weights, dtype=np.float64)
        if len(weights) != len(edges):
            raise ValueError("one weight per edge")
        _vertex_ids(edges, n_vertices)
        self.graph = ig.Graph(n=n_vertices, edges=edges.tolist(), directed=False)
        self.graph.es["weight"] = weights.tolist()
        self._edges = edges
        self._lengths = weights

    @staticmethod
    def from_mesh(vertices, faces) -> SurfaceGraph:
        """The surface graph of a mesh, from its faces: ``edges`` and
        ``lengths`` are trimesh's ``edges_unique`` and
        ``edges_unique_length``, the same values in the same order."""
        vertices = rows3("N", vertices)
        faces, n = checked_faces(faces, len(vertices))
        edges = unique_edges(faces, n)
        return SurfaceGraph(n, edges, edge_lengths(vertices, edges))

    @property
    def edges(self) -> np.ndarray:
        """The edges, ``(E, 2)``."""
        return self._edges

    @property
    def lengths(self) -> np.ndarray:
        """The edge weights, ``(E,)``: their lengths."""
        return self._lengths

    def bin_component(
        self,
        verts,
        seeds,
        mean_edge: float,
        step_size: float | None,
        target_shell_count: int,
        min_shell_vertices: int,
        max_shell_width_factor: float,
    ) -> list[list[np.ndarray]]:
        """:func:`bin_one_component` on the preprocessing track (no soma
        exclusion, no elongated-shell split): shells of bins over *verts*
        (ascending) from the nearest of *seeds*."""
        n = self.graph.vcount()
        return bin_one_component(
            self.graph,
            _vertex_ids(verts, n),
            _vertex_ids(seeds, n),
            mesh_vertices=None,
            mean_edge_len=mean_edge,
            step_size=step_size,
            target_shell_count=target_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=max_shell_width_factor,
            split_elongated_shells=False,
        )

    def components(self) -> list[np.ndarray]:
        """Connected components, each ascending, in order of their smallest
        vertex — igraph's ``components()``."""
        return [np.asarray(c, dtype=np.int64) for c in self.graph.components()]

    def bin_mesh(
        self,
        seed: int,
        mean_edge: float,
        step_size: float | None,
        target_shell_count: int,
        min_shell_vertices: int,
        max_shell_width_factor: float,
    ) -> list[list[np.ndarray]]:
        """:func:`bin_geodesic_shells` on the direct track's settings: every
        connected component binned on its own, in :meth:`components` order,
        from *seed* where it lies and otherwise from the component's vertex
        at its index modulo its size; the shells of all components in that
        order."""
        (seed,) = _vertex_ids([seed], self.graph.vcount()).tolist()
        shells: list[list[np.ndarray]] = []
        for cid, verts in enumerate(self.components()):
            start = seed if np.any(verts == seed) else int(verts[cid % len(verts)])
            shells.extend(
                self.bin_component(
                    verts,
                    [start],
                    mean_edge,
                    step_size,
                    target_shell_count,
                    min_shell_vertices,
                    max_shell_width_factor,
                )
            )
        return shells

    def recheck_rings(self, rebinned, verts) -> list[list[np.ndarray]]:
        """Keep the re-binned rings over *verts*, fragments merged into them."""
        n = self.graph.vcount()
        return recheck_rings(self.graph, _shells(rebinned, n), _vertex_ids(verts, n))

    def reunite_bins(self, shells) -> list[list[np.ndarray]]:
        """Make every bin one connected piece."""
        return reunite_bins(self.graph, _shells(shells, self.graph.vcount()))

    def split_branch_bins(self, shells) -> list[list[np.ndarray]]:
        """Split bins that wrap a branch point."""
        return split_branch_bins(self.graph, _shells(shells, self.graph.vcount()))


def skeleton_nodes(shells, vertices, estimators) -> tuple[np.ndarray, list[np.ndarray]]:
    """Each bin's centroid and its radius under each of *estimators*, bins in
    shell order: ``(nodes (N, 3), [radii (N,) per estimator])``."""
    vertices = rows3("V", vertices)
    bins = [b for band in _shells(shells, len(vertices)) for b in band]
    if any(len(b) == 0 for b in bins):
        raise ValueError("a bin has no vertices")
    estimators = list(estimators)
    if bins and not set(estimators) <= RECOMPUTABLE_RADII:
        names = ", ".join(f'"{e}"' for e in estimators)
        raise ValueError(f"Unknown radius estimator in [{names}].")

    nodes = np.empty((len(bins), 3), dtype=np.float64)
    radii = [np.empty(len(bins), dtype=np.float64) for _ in estimators]
    for i, ids in enumerate(bins):
        pts = vertices[ids]
        nodes[i] = pts.mean(axis=0)
        d = np.linalg.norm(pts - nodes[i], axis=1)
        for r, est in zip(radii, estimators):
            r[i] = _estimate_radius(d, method=est, trim_fraction=TRIM_FRACTION)
    return nodes, radii


def merge_nested_nodes(
    nodes,
    radii,
    node2verts,
    inside_frac: float = 0.9,  # 1.0 = 100 % (strict), 0.99 ≈ 99 %, …
    keep_root: bool = True,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Collapse node *j* into node *i* when at least ``inside_frac`` of *j*'s
    radius lies inside *i*'s radius:

        ‖cᵢ – cⱼ‖ + inside_frac · rⱼ  ≤  rᵢ  + tol

    The “keeper” (larger sphere) inherits *j*’s vertex IDs.  Nodes are
    visited biggest first, ties by index, node 0 never when *keep_root*;
    each one's candidates are the nodes within ``rᵢ + max(r)``, by index.

    Returns
    -------
    keep_mask      – Boolean mask to apply to all node-wise arrays.
    node2verts_new – Updated mapping (same order as keep_mask==True).
    old2new        – Vector mapping old → new node IDs (-1 if dropped).
    """
    if not (0.0 < inside_frac <= 1.0):
        raise ValueError("inside_frac must be in (0, 1].")
    nodes = rows3("N", nodes)
    radii = np.asarray(radii, dtype=np.float64)
    if len(radii) != len(nodes) or len(node2verts) != len(nodes):
        raise ValueError("one radius and one vertex list per node")
    node2verts = [_vertex_ids(v, 2**32 - 1) for v in node2verts]

    N = len(nodes)
    keep_mask = np.ones(N, bool)
    old2new = np.arange(N, dtype=np.int64)
    if N == 0:
        return keep_mask, [], old2new

    order = np.argsort(-radii, kind="stable")  # big → small, ties by index
    tree = KDTree(nodes)
    r_max = radii.max()
    for i in order:
        if keep_root and i == 0:
            continue  # never drop soma
        if not keep_mask[i]:
            continue  # already swallowed

        # neighbours that *might* fit: distance ≤ rᵢ + r_max
        cand = np.asarray(
            tree.query_ball_point(nodes[i], radii[i] + r_max, return_sorted=True),
            dtype=np.int64,
        )
        # j is larger or gone: skip it
        cand = cand[(cand != i) & keep_mask[cand] & ~(radii[cand] > radii[i])]
        d = nodes[i] - nodes[cand]
        dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        # modified containment test
        inside = cand[dist + inside_frac * radii[cand] <= radii[i] + tol]
        if len(inside):
            node2verts[i] = np.concatenate(
                [node2verts[i], *(node2verts[j] for j in inside)]
            )
            keep_mask[inside] = False
            old2new[inside] = old2new[i]

    # compact node2verts into surviving order
    node2verts_new = [node2verts[k] for k in np.flatnonzero(keep_mask)]
    return keep_mask, node2verts_new, old2new


def rebin_chains(
    nodes, edges, node2verts, vertices, r_ref, donated
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Re-assign the vertices along each chain of the skeleton (a run of
    degree-2 nodes between two anchors, nodes of any other degree) to the
    chain node nearest by radius-weighted arc length.

    Every chain vertex is projected onto the piecewise-linear path through
    the chain's nodes and goes to the node whose arc position is nearest,
    the distance divided by the node's radius (``r_ref``, floored at 1% of
    its positive median) so thick nodes claim proportionally more.  A
    branch node gives each of its chains only the vertices *donated* to it:
    ``(branch node, neighbour, vertices)``, the vertices facing that
    neighbour.  Anchor vertices no chain claims stay with their node.

    Returns each node's new vertices, then every chain-assigned vertex and
    its distance to the centreline, in node order.
    """
    nodes = rows3("N", nodes)
    vertices = rows3("V", vertices)
    r_ref = np.asarray(r_ref, dtype=np.float64)
    n, nv = len(nodes), len(vertices)
    if len(r_ref) != n or len(node2verts) != n:
        raise ValueError("one radius and one vertex list per node")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be (E, 2)")
    edges = _vertex_ids(edges, n, "node").reshape(-1, 2)
    node2verts = [_vertex_ids(v, nv, "vertex").tolist() for v in node2verts]
    donations: dict[tuple[int, int], list[int]] = {}
    for b, nb, vs in donated:
        vs = _vertex_ids(vs, nv, "vertex").tolist()
        b, nb = _vertex_ids([b, nb], n, "node").tolist()
        donations[(b, nb)] = vs

    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges.tolist():
        adj[a].add(b)
        adj[b].add(a)
    degree = [len(s) for s in adj]
    anchors = [i for i in range(n) if degree[i] != 2]

    # -- degree-2 paths between anchors --
    chains: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()
    for anchor in anchors:
        for nbr in sorted(adj[anchor]):
            edge = (min(anchor, nbr), max(anchor, nbr))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            chain = [anchor, nbr]
            cur, prev = nbr, anchor
            while degree[cur] == 2:
                nexts = [x for x in adj[cur] if x != prev]
                if len(nexts) != 1:
                    break
                prev, cur = cur, nexts[0]
                visited_edges.add((min(prev, cur), max(prev, cur)))
                chain.append(cur)
            chains.append(chain)

    positive = r_ref[r_ref > 0]
    r_floor = np.median(positive) * 0.01 if len(positive) else np.nan

    out: list[list[int]] = [[] for _ in range(n)]
    centerline_dists: list[list[float]] = [[] for _ in range(n)]
    assigned = np.zeros(nv, dtype=bool)
    for chain in chains:
        # -- the chain's vertices, a branch node's donated share only --
        last = len(chain) - 1
        ids_set: set[int] = set()
        for pos, ni in enumerate(chain):
            key = None
            if degree[ni] >= 3 and pos in (0, last):
                key = (ni, chain[1] if pos == 0 else chain[last - 1])
            if key in donations:
                ids_set.update(donations[key])
            else:
                ids_set.update(node2verts[ni])
        ids = [v for v in sorted(ids_set) if not assigned[v]]
        if not ids:
            continue
        pts = vertices[ids]

        # -- project each vertex onto the piecewise-linear path --
        path = nodes[chain]
        seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc_cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        best_dist = np.full(len(ids), np.inf)
        best_t = np.zeros(len(ids))
        for si in range(last):
            p0, s = path[si], seg_lens[si]
            if s < 1e-10:
                continue
            sd = (path[si + 1] - p0) / s
            df = pts - p0
            # summed left to right, as the Rust path does: a BLAS product
            # may fuse or reorder it
            tp = np.clip(df[:, 0] * sd[0] + df[:, 1] * sd[1] + df[:, 2] * sd[2], 0, s)
            d = np.linalg.norm(pts - (p0 + tp[:, None] * sd), axis=1)
            closer = d < best_dist
            best_dist[closer] = d[closer]
            best_t[closer] = (arc_cum[si] + tp)[closer]

        # -- radius-weighted assignment --
        node_radii = np.array([max(r_ref[c], r_floor) for c in chain])
        # with no positive radius to floor them, zero radii divide into
        # infinities and NaNs; argmin takes the first NaN, as Rust does
        with np.errstate(divide="ignore", invalid="ignore"):
            arc_diffs = np.abs(best_t[:, None] - arc_cum[None, :])
            weighted = arc_diffs / node_radii[None, :]
        for k, ci in enumerate(np.argmin(weighted, axis=1).tolist()):
            ni = chain[ci]
            out[ni].append(ids[k])
            centerline_dists[ni].append(float(best_dist[k]))
            assigned[ids[k]] = True

    # -- anchor vertices no chain claimed --
    for ni in anchors:
        for v in node2verts[ni]:
            if not assigned[v]:
                out[ni].append(v)
                assigned[v] = True

    vids = [v for ni in range(n) for v, _ in zip(out[ni], centerline_dists[ni])]
    dists = [d for ni in range(n) for d in centerline_dists[ni]]
    return (
        [np.asarray(o, dtype=np.int64) for o in out],
        np.asarray(vids, dtype=np.int64),
        np.asarray(dists, dtype=np.float64),
    )


# ---------------------------------------------------------------------
# _core's stages
# ---------------------------------------------------------------------
def bfs_parents(edges, n_nodes: int, root: int = 0) -> np.ndarray:
    """Each node's parent in a breadth-first tree from *root* over the
    undirected *edges* ``(E, 2)``, a node's neighbours in the order the edges
    name them; -1 for the root and every node it does not reach."""
    edges = node_pairs(edges, n_nodes)
    if not 0 <= root < n_nodes:
        raise ValueError(f"root {root} is not one of {n_nodes} nodes")
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges.tolist():
        adj[a].append(b)
        adj[b].append(a)
    parent = [-1] * n_nodes
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v != root and parent[v] == -1:
                parent[v] = u
                queue.append(v)
    return np.asarray(parent, dtype=np.int64)


def build_mst(nodes, edges) -> np.ndarray:
    """The minimum spanning forest of the skeleton *edges* ``(E, 2)`` weighted
    by their lengths between *nodes* ``(N, 3)``, igraph's ``spanning_tree``:
    its edges as sorted pairs, in order, ``(K, 2)``."""
    nodes = rows3("N", nodes)
    edges = node_pairs(edges, len(nodes))
    weights = [float(np.linalg.norm(nodes[a] - nodes[b])) for a, b in edges]
    if any(w != w for w in weights):
        raise ValueError("edge lengths must not be NaN")
    g = ig.Graph(n=len(nodes), edges=edges.tolist(), directed=False)
    tree = g.spanning_tree(weights=weights)
    pairs = sorted(tuple(sorted(e)) for e in tree.get_edgelist())
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _arrays_in(nodes, keys, radii, ptr, idx, edges, mesh_vertices):
    """A skeleton's arrays as ``_core``'s stages take them, checked: nodes,
    radii ``(K, N)`` (one row per key), each node's vertices from the CSR
    ``ptr``/``idx`` (ids within *mesh_vertices* when given), edges."""
    nodes = rows3("N", nodes)
    n = len(nodes)
    radii = np.array(radii, dtype=np.float64)
    if radii.ndim != 2 or radii.shape != (len(keys), n):
        raise ValueError("radii must be one row per key, one column per node")
    ptr = np.asarray(ptr, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    if (
        len(ptr) != n + 1
        or ptr[0] != 0
        or np.any(np.diff(ptr) < 0)
        or ptr[-1] != len(idx)
    ):
        raise ValueError("node2verts must be a CSR over the nodes")
    out = idx < 0 if mesh_vertices is None else (idx < 0) | (idx >= len(mesh_vertices))
    if out.any():
        raise ValueError(f"vertex index {idx[out][0]} is out of range")
    node2verts = [idx[ptr[i] : ptr[i + 1]] for i in range(n)]
    return nodes.copy(), radii, node2verts, node_pairs(edges, n)


def _arrays_out(nodes, radii, node2verts, edges):
    """The arrays ``_core``'s stages give back: nodes, radii ``(K, N)``,
    the nodes' vertices as a CSR ``ptr``/``idx``, edges ``(E, 2)``."""
    ptr = np.zeros(len(node2verts) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(v) for v in node2verts])
    idx = (
        np.concatenate(node2verts).astype(np.int64)
        if node2verts
        else np.empty(0, dtype=np.int64)
    )
    return (
        np.asarray(nodes, dtype=np.float64).reshape(-1, 3),
        radii,
        ptr,
        idx,
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def _reestimate(radii, keys, nid: int, d: np.ndarray) -> None:
    """``_radii_from_distances`` for node *nid*: every radius that is a plain
    aggregate of its vertices' distances *d* re-estimated, the rest left."""
    for k, key in enumerate(keys):
        if key in RECOMPUTABLE_RADII:
            radii[k, nid] = _estimate_radius(d, method=key)


def _compact(nodes, radii, node2verts, edges, keep):
    """``compact_state``: the kept nodes, their radii and vertices, and the
    edges between them renumbered; and ``old2new``."""
    old2new = -np.ones(len(keep), dtype=np.int64)
    old2new[keep] = np.arange(int(keep.sum()))
    kept = [node2verts[i] for i in np.flatnonzero(keep)]
    return nodes[keep], radii[:, keep], kept, remap_edges(edges, old2new), old2new


def _merge_single(nodes, keys, radii, node2verts, edges, mesh_vertices, degree):
    """``merge_single_node_branches`` on the unpacked arrays; with the
    indices before of the nodes kept."""
    orig = np.arange(len(nodes))
    while True:
        n = len(nodes)
        deg = np.zeros(n, dtype=np.int64)
        np.add.at(deg, edges.ravel(), 1)
        parent = bfs_parents(edges, n, 0)
        # a leaf the root does not reach has no parent
        singles = [
            leaf
            for leaf in range(1, n)
            if deg[leaf] == 1 and parent[leaf] >= 0 and deg[parent[leaf]] >= degree
        ]
        if not singles:
            return nodes, radii, node2verts, edges, orig
        keep = np.ones(n, dtype=bool)
        for leaf in singles:
            par = parent[leaf]
            node2verts[par] = np.concatenate((node2verts[par], node2verts[leaf]))
            if mesh_vertices is not None and node2verts[par].size:
                d = np.linalg.norm(mesh_vertices[node2verts[par]] - nodes[par], axis=1)
                _reestimate(radii, keys, par, d)
            keep[leaf] = False
        nodes, radii, node2verts, edges, _ = _compact(
            nodes, radii, node2verts, edges, keep
        )
        orig = orig[keep]


def merge_single_node_branches(
    nodes, keys, radii, ptr, idx, edges, mesh_vertices, min_parent_degree
):
    """``_merge_single_node_branches`` on a skeleton's arrays (see
    ``_arrays_in``): every leaf (one edge, not the root) whose parent in a
    breadth-first tree from the root has *min_parent_degree* edges or more
    merged into it — its vertices appended to the parent's, in leaf order,
    and with *mesh_vertices* the parent's radii re-estimated from all its
    vertices — and dropped, again until no leaf qualifies; a leaf the root
    does not reach stays. The arrays after, and each node's index before."""
    if mesh_vertices is not None:
        mesh_vertices = rows3("V", mesh_vertices)
    nodes, radii, node2verts, edges = _arrays_in(
        nodes, keys, radii, ptr, idx, edges, mesh_vertices
    )
    *arrays, kept = _merge_single(
        nodes, keys, radii, node2verts, edges, mesh_vertices, min_parent_degree
    )
    return _arrays_out(*arrays), kept.astype(np.int64)


def prune_neurites(
    nodes,
    keys,
    radii,
    ptr,
    idx,
    edges,
    center,
    r_soma,
    mesh_vertices,
    tip_extent_factor,
    stem_extent_factor,
    drop_single_node_branches,
):
    """``_prune_neurites`` but the soma's refit, on a skeleton's arrays (see
    ``_arrays_in``): each neurite at the root — the tree from one of the
    root's neighbours — whose nodes all lie within ``stem_extent_factor ×
    r_soma`` of the soma's *center* (``tip_extent_factor`` for one whose
    first node is not the root's child in a breadth-first tree) is merged
    into the root: its vertices to the root's, made unique, and with
    *mesh_vertices* the root's radii re-estimated. Then, with
    *drop_single_node_branches*, ``merge_single_node_branches`` (parents of
    three edges or more). The arrays after, ``old2new``, the soma's vertices
    to refit it from (the root's before the single-node merge), and how many
    nodes went into the soma."""
    if mesh_vertices is not None:
        mesh_vertices = rows3("V", mesh_vertices)
    nodes, radii, node2verts, edges = _arrays_in(
        nodes, keys, radii, ptr, idx, edges, mesh_vertices
    )
    n = len(nodes)
    if n == 0:
        raise ValueError("the skeleton has no nodes")
    d2c = np.linalg.norm(nodes - np.asarray(center, dtype=np.float64), axis=1)
    parent = bfs_parents(edges, n, 0)
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges.tolist():
        adj[a].append(b)
        adj[b].append(a)

    merge = np.zeros(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)
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
        factor = stem_extent_factor if parent[child] == 0 else tip_extent_factor
        if d2c[comp].max() <= factor * r_soma:
            merge[comp] = True
    merge[0] = False

    n_merged = int(merge.sum())
    if n_merged:
        node2verts[0] = np.unique(
            np.concatenate(
                [node2verts[0], *(node2verts[i] for i in np.flatnonzero(merge))]
            )
        )
        if mesh_vertices is not None and node2verts[0].size:
            d0 = np.linalg.norm(mesh_vertices[node2verts[0]] - nodes[0], axis=1)
            _reestimate(radii, keys, 0, d0)
    nodes, radii, node2verts, edges, old2new = _compact(
        nodes, radii, node2verts, edges, ~merge
    )
    soma_verts = np.unique(node2verts[0]).astype(np.int64)
    if drop_single_node_branches:
        nodes, radii, node2verts, edges, kept = _merge_single(
            nodes, keys, radii, node2verts, edges, mesh_vertices, 3
        )
        after = -np.ones(len(old2new), dtype=np.int64)
        after[kept] = np.arange(len(kept))
        old2new = np.where(old2new >= 0, after[old2new], -1)
    return (
        _arrays_out(nodes, radii, node2verts, edges),
        old2new,
        soma_verts,
        n_merged,
    )


def merge_near_soma_nodes(
    nodes,
    keys,
    radii,
    ptr,
    idx,
    edges,
    radius_row,
    center,
    axes,
    rotation,
    soma_verts,
    mesh_vertices,
    inside_tol,
    near_factor,
    fat_factor,
):
    """``_merge_near_soma_nodes`` but the soma's refit, on a skeleton's arrays
    (see ``_arrays_in``) and the soma's ellipsoid (*center*, *axes*,
    *rotation*): every node but the root inside the soma
    (``distance_to_surface < -inside_tol``), or near it (``distance_to_center
    < near_factor × r_soma``) and fat (its radius in row *radius_row* above
    ``fat_factor × r_soma``), is merged into the root and dropped; the kept
    nodes it touched are tied to the root. Of the merged nodes' vertices, in
    order, those within ``near_factor × r_soma`` of the surface (all of them
    without *mesh_vertices*) join the root's and the soma's *soma_verts*,
    which end sorted and unique. The arrays after, ``old2new``, the soma's
    vertices (None while it has none) and how many nodes went into it."""
    if mesh_vertices is not None:
        mesh_vertices = rows3("V", mesh_vertices)
    nodes, radii, node2verts, edges = _arrays_in(
        nodes, keys, radii, ptr, idx, edges, mesh_vertices
    )
    if not 0 <= radius_row < len(keys):
        raise ValueError(f"radius row {radius_row} is not one of {len(keys)}")
    if len(nodes) == 0:
        raise ValueError("the skeleton has no nodes")
    soma = Soma(center=center, axes=axes, R=rows3("R", rotation))
    r_soma = soma.spherical_radius
    inside = soma.distance_to_surface(nodes) < -inside_tol
    near = soma.distance_to_center(nodes) < near_factor * r_soma
    fat = radii[radius_row] > fat_factor * r_soma
    keep = ~(inside | (near & fat))
    keep[0] = True
    merged = np.flatnonzero(~keep)
    kept_nodes, kept_radii, kept_verts, kept_edges, old2new = _compact(
        nodes, radii, node2verts, edges, keep
    )

    leaves = set()
    for u, v in edges.tolist():
        if keep[u] and not keep[v]:
            leaves.add(int(old2new[u]))
        elif keep[v] and not keep[u]:
            leaves.add(int(old2new[v]))
    leaves.discard(0)  # the soma took the merged nodes in
    if leaves:
        tie = np.array([[0, nid] for nid in sorted(leaves)], dtype=np.int64)
        kept_edges = np.unique(np.sort(np.vstack([kept_edges, tie]), axis=1), axis=0)

    verts = None if soma_verts is None else np.asarray(soma_verts, dtype=np.int64)
    for i in merged:
        vidx = node2verts[i]
        if mesh_vertices is not None:
            close = soma.distance_to_surface(mesh_vertices[vidx]) < near_factor * r_soma
            contrib = vidx[close]
        else:
            contrib = vidx
        if contrib.size:
            verts = contrib if verts is None else np.concatenate((verts, contrib))
            kept_verts[0] = np.concatenate((kept_verts[0], contrib))
    if verts is not None:
        verts = np.unique(verts).astype(np.int64)
    return (
        _arrays_out(kept_nodes, kept_radii, kept_verts, kept_edges),
        old2new,
        verts,
        int(merged.size),
    )


def find_soma(nodes, radius, pct_large, dist_factor, min_keep):
    """``_find_soma``: the nodes whose *radius* is at least its
    *pct_large*-th percentile and which lie within *dist_factor* times the
    largest radius of the node that has it. A soma with *min_keep* such
    nodes or more is centred on their mean; otherwise on that node.
    ``(soma_nodes, idx_max, center, r_max, has_soma)``, or None when no node
    reaches the percentile."""
    nodes = rows3("N", nodes)
    radius = np.asarray(radius, dtype=np.float64)
    if len(radius) != len(nodes):
        raise ValueError("one radius per node")
    if len(nodes) == 0:
        raise ValueError("empty skeleton")
    if not 0 <= pct_large <= 100:
        raise ValueError("Percentiles must be in the range [0, 100]")
    cand = np.flatnonzero(radius >= np.percentile(radius, pct_large))
    if cand.size == 0:
        return None
    idx_max = int(np.argmax(radius))
    r_max = float(radius[idx_max])
    dists = np.linalg.norm(nodes[cand] - nodes[idx_max], axis=1)
    soma_nodes = cand[dists <= dist_factor * r_max].astype(np.int64)
    has_soma = bool(soma_nodes.size >= min_keep)
    center = (
        nodes[soma_nodes].mean(0) if has_soma and soma_nodes.size else nodes[idx_max]
    )
    return soma_nodes, idx_max, [float(c) for c in center], r_max, has_soma


def soma_vertices(points, ptr, idx, center, r_max) -> np.ndarray:
    """The soma's vertices from its nodes at *points* and their vertex lists
    (a CSR *ptr*, *idx*): those of the nodes closer to *center* than twice
    *r_max* (lists that are not empty), merged, sorted and unique; failing
    those, the first node's list as it is."""
    points = rows3("P", points)
    ptr = np.asarray(ptr, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    k = len(points)
    if (
        len(ptr) != k + 1
        or ptr[0] != 0
        or np.any(np.diff(ptr) < 0)
        or ptr[-1] != len(idx)
        or np.any(idx < 0)
    ):
        raise ValueError("the vertex lists must be a CSR over the points")
    lists = [idx[ptr[i] : ptr[i + 1]] for i in range(k)]
    close = np.linalg.norm(points - np.asarray(center, dtype=np.float64), axis=1) < (
        r_max * 2
    )
    found = [lists[i] for i in np.flatnonzero(close) if lists[i].size]
    if found:
        return np.unique(np.concatenate(found)).astype(np.int64)
    return lists[0] if k else np.empty(0, dtype=np.int64)


def _auto_bridge_max(gaps, edge_mean: float) -> float:
    """The largest bridge, in mean edge lengths, when none is given: the
    55th percentile of the first gaps over the mean edge length, clipped to
    6 to 12."""
    return float(np.clip(np.percentile(gaps, 55.0) / edge_mean, 6.0, 12.0))


def _auto_recalc_after(n_gaps: int) -> int:
    """After how many bridges every gap is measured again, when not given:
    often for few gaps, at most every 32."""
    if n_gaps <= 10:
        return 2
    if n_gaps <= 50:
        return 4
    if n_gaps <= 200:
        return max(4, n_gaps // 20)
    return 32


def bridge_gaps(nodes, edges, bridge_max_factor, bridge_recalc_after) -> np.ndarray:
    """``_bridge_gaps``: every connected component of the skeleton *edges*
    ``(E, 2)`` tied to the component of node 0 by one new edge, nearest
    component first — the closest pair of nodes between it and what is tied
    so far (a component's nodes in ascending order, the first of equally
    close ones; the tied nodes in the order they joined, ties to the
    earliest). A component farther than *bridge_max_factor* mean edge
    lengths (by default from the gaps, ``_auto_bridge_max``) waits, and the
    limit grows by half after twice three times as many waits as there were
    gaps; every gap is measured again after *bridge_recalc_after* bridges
    (by default ``_auto_recalc_after``). The edges with the bridges, unique
    and in order; the edges as they are when there is one component."""
    nodes = rows3("N", nodes)
    edges = node_pairs(edges, len(nodes))
    if len(nodes) == 0:
        raise ValueError("the skeleton has no nodes")
    g = ig.Graph(n=len(nodes), edges=edges.tolist(), directed=False)
    membership = np.asarray(g.components().membership, dtype=np.int64)
    n_comp = int(membership.max()) + 1
    if n_comp == 1:
        return edges
    by_comp = np.argsort(membership, kind="stable")
    comps = np.split(by_comp, np.cumsum(np.bincount(membership))[:-1])
    soma_cid = int(membership[0])
    edge_len_mean = np.linalg.norm(
        nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1
    ).mean()

    island_idx = comps[soma_cid]
    tree = KdTree(nodes[island_idx])

    def closest_pair(cid: int) -> tuple[float, int, int]:
        dists, idx = tree.query(nodes[comps[cid]])
        best = int(np.argmin(dists))
        return float(dists[best]), best, int(idx[best])

    def measured(cids) -> list[tuple[float, int, int, int]]:
        """The gaps of components *cids* as a heap of (gap, component, its
        node, the tied node)."""
        pq = [(g, c, bc, bi) for c in cids for g, bc, bi in (closest_pair(c),)]
        heapq.heapify(pq)
        return pq

    pq = measured([c for c in range(n_comp) if c != soma_cid])
    if bridge_max_factor is None:
        bridge_max_factor = _auto_bridge_max([q[0] for q in pq], edge_len_mean)
    recalc_after = (
        _auto_recalc_after(n_comp - 1)
        if bridge_recalc_after is None
        else bridge_recalc_after
    )

    bridges: list[tuple[int, int]] = []
    since_recalc = stall = 0
    max_stall = 3 * len(pq)
    current_max = bridge_max_factor * edge_len_mean
    while pq:
        _, cid, _, _ = heapq.heappop(pq)
        gap, best_c, best_i = closest_pair(cid)
        if gap > current_max:  # still too far
            heapq.heappush(pq, (gap, cid, best_c, best_i))
            stall += 1
            if stall >= 2 * max_stall:
                pq = measured([q[1] for q in pq])
                current_max *= 1.5
                stall = 0
            continue
        stall = 0
        bridges.append((int(comps[cid][best_c]), int(island_idx[best_i])))
        island_idx = np.concatenate([island_idx, comps[cid]])
        tree = KdTree(nodes[island_idx])
        since_recalc += 1
        if since_recalc >= recalc_after:
            pq = measured([q[1] for q in pq])
            since_recalc = 0
    if not bridges:
        return np.unique(edges, axis=0)
    return np.unique(np.vstack([edges, np.asarray(bridges, dtype=np.int64)]), axis=0)


def edges_between(edges, owner) -> np.ndarray:
    """``_edges_from_mesh``: the node pairs the mesh *edges* ``(E, 2)`` join,
    by each vertex's node *owner* (-1 for none) — each pair of distinct
    nodes once, smaller first, in order, ``(K, 2)``."""
    owner = np.asarray(owner, dtype=np.int64)
    edges = node_pairs(edges, len(owner))
    na, nb = owner[edges[:, 0]], owner[edges[:, 1]]
    keep = (na >= 0) & (nb >= 0) & (na != nb)
    lo = np.minimum(na[keep], nb[keep])
    hi = np.maximum(na[keep], nb[keep])
    # one key per pair sorts as the (lo, hi) rows do
    width = int(hi.max()) + 1 if hi.size else 1
    key = np.unique(lo * width + hi)
    return np.stack([key // width, key % width], axis=1)


def centerline_radii(ptr, idx, vids, dists, n_vertices: int) -> np.ndarray:
    """Each node's ``centerline`` radius: the trimmed mean of its vertices'
    distances to the centreline — *dists[k]* for vertex *vids[k]*, the last
    for a vertex listed twice, 0 for one not listed — and 0 for a node
    without vertices; the nodes' vertices as a CSR (*ptr*, *idx*) into
    *n_vertices* vertices."""
    ptr = np.asarray(ptr, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    if len(ptr) == 0 or ptr[0] != 0 or np.any(np.diff(ptr) < 0) or ptr[-1] != len(idx):
        raise ValueError("node2verts must be a CSR over the nodes")
    vids = np.asarray(vids, dtype=np.int64)
    dists = np.asarray(dists, dtype=np.float64)
    for ids in (idx, vids):
        out = ids[(ids < 0) | (ids >= n_vertices)]
        if out.size:
            raise ValueError(f"vertex index {out[0]} is out of range")
        if ids is idx and len(vids) != len(dists):
            raise ValueError("one distance per vertex")
    cl = dict(zip(vids.tolist(), dists.tolist()))
    radii = np.zeros(len(ptr) - 1)
    for i in range(len(ptr) - 1):
        d = np.array([cl.get(v, 0.0) for v in idx[ptr[i] : ptr[i + 1]].tolist()])
        if len(d):
            radii[i] = _estimate_radius(d, method="trim", trim_fraction=TRIM_FRACTION)
    return radii
