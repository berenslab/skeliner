import heapq
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree

__all__ = [
    "Skeleton",
    "find_soma",
    "skeletonize",
]

# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class Skeleton:
    """Light-weight skeleton graph.

    The skeleton is a forest-shaped graph (acyclic, undirected) whose vertices
    sit on the centre-line of a mesh representation of a neurone. Node 0 is
    reserved for the soma centroid; all other vertices belong to neurites.

    Parameters
    ----------
    nodes
        (N, 3) float32 Cartesian coordinates.
    radii
        (N,) float32 local radii.
    edges
        (E, 2) int64 undirected sorted vertex pairs.
    soma_verts 
        Optional 1D int64 array of mesh-vertex IDs belonging to the soma surface.
        Optional. None when loaded from SWC.
    """
    nodes: np.ndarray  # (N, 3) float32
    radii:  dict[str, np.ndarray]  # (N,)  float32
    edges: np.ndarray  # (E, 2) int64  – undirected, **sorted** pairs
    soma_verts: np.ndarray | None = None
    node2verts: list[np.ndarray] | None = None
    vert2node: dict[int, int] | None = None

    # ---------------------------------------------------------------------
    # sanity checks
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate basic shape constraints."""
        if self.nodes.shape[0] != self.r.shape[0]:
            raise ValueError("nodes and radii length mismatch")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must be (E, 2)")
        if self.soma_verts is not None and self.soma_verts.ndim != 1:
            raise ValueError("soma_verts must be 1-D")

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _igraph(self) -> ig.Graph:
        """Return an :class:`igraph.Graph` view of self (undirected)."""
        return ig.Graph(n=len(self.nodes), edges=[tuple(map(int, e)) for e in self.edges], directed=False)

    # ---------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------
    def to_swc(self, 
               path: str | Path,
               include_header: bool = True, 
               scale: float = 1.0,
               radius_metric: str | None = None
    ) -> None:
        """Write the skeleton to SWC.

        The first node (index 0) is written as type 1 (soma) and acts as the
        root of the morphology tree. Parent IDs are therefore 1‑based to
        comply with the SWC format.

        Parameters
        ----------
        path
            Output filename.
        include_header
            Prepend the canonical SWC header line if *True*.
        scale
            Unit conversion factor applied to *both* coordinates and radii when
            writing; useful e.g. for nm→µm conversion.
        """        
        path = Path(path)
        parent = _bfs_parents(self.edges, len(self.nodes), root=0)
        nodes = self.nodes
        if radius_metric is None:
            radii = self.r
        else:
            if radius_metric not in self.radii:
                raise ValueError(f"Unknown radius estimator '{radius_metric}'")
            radii = self.radii[radius_metric]
        with path.open("w", encoding="utf8") as fh:
            if include_header:
                fh.write("# id type x y z radius parent\n")
            for idx, (p, r, pa) in enumerate(zip(nodes * scale, radii * scale, parent), start=1):
                swc_type = 1 if idx == 1 else 0
                fh.write(f"{idx} {swc_type} {p[0]} {p[1]} {p[2]} {r} {pa + 1 if pa != -1 else -1}\n")

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def check_connectivity(self, *, return_isolated: bool = False):
        """Verify that all nodes are reachable from the soma (node 0).

        Parameters
        ----------
        return_isolated
            If True return the list of orphan node indices instead of a
            boolean flag.

        Returns
        -------
        bool | list[int]
            True when the skeleton is connected, otherwise False or a
            list of isolated nodes if return_isolated=True.
        """
        g = self._igraph()
        order, _, _ = g.bfs(0, mode="ALL")  # returns -1 for unreachable
        reachable: Set[int] = {v for v in order if v != -1}
        if return_isolated:
            return [i for i in range(len(self.nodes)) if i not in reachable]
        return len(reachable) == len(self.nodes)

    def check_acyclic(self, *, return_cycle: bool = False) -> bool | list[tuple[int, int]]:
        """Detect cycles in the edge list.

        The skeleton is expected to be a forest (collection of trees).  This
        method verifies :math:`|E| = |V| - C` where C is the number of
        connected components.

        Parameters
        ----------
        return_cycle
            When True a representative cycle is returned if one exists; an
            empty list means the graph is acyclic.
        """
        g = self._igraph()                        # helper that builds igraph
        num_comp = len(g.components())            # instead of .nc
        acyclic  = g.ecount() == g.vcount() - num_comp
        if acyclic or not return_cycle:
            return acyclic

        # -- find a concrete cycle (slow but rare) -----------------------
        cyc = g.cycle_basis()[0]                  # list of vertex ids
        return [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]

    # ------------------------------------------------------------------
    # radius recommendation
    # ------------------------------------------------------------------
    def recommend_radius(self) -> Tuple[str, str, Dict[str, float]]:
        """Heuristic choice among mean / trim / median with explanation.

        Returns
        -------
        choice : str
            Name of the recommended estimator.
        reason : str
            Short human‑readable explanation.
        stats : dict
            Diagnostic numbers {"p50", "p75", "max"} of mean/median ratio.
        """
        mean = self.radii.get("mean")
        median = self.radii.get("median")
        if mean is None or median is None:
            return "median", "Only one radius column available; using it.", {}

        ratio = mean / median
        p50 = float(np.percentile(ratio, 50))
        p75 = float(np.percentile(ratio, 75))
        pmax = float(ratio.max())

        if p75 < 1.02:
            choice, reason = "mean", "Bias ≤ 2% for 75% of nodes – distribution symmetric."
        elif p50 < 1.05 and "trim" in self.radii:
            choice, reason = "trim", "Moderate tails; 5% trimmed mean is robust and less biased."
        else:
            choice, reason = "median", "Long positive tails detected; median is safest."

        return choice, reason, {"p50": p50, "p75": p75, "max": pmax}


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def r(self) -> np.ndarray:
        """Just the estimator name chosen by :py:meth:`recommend_radius`."""
        choice = self.recommend_radius()[0]
        return self.radii[choice]

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

# -----------------------------------------------------------------------------
#  Soma detection API
# -----------------------------------------------------------------------------
def _likely_has_soma(
    radii: np.ndarray,
    ratio: float = 3.5,       
    min_nodes: int = 5,      
) -> bool:
    """
    Return True if a plausible soma cluster is present.
    """
    return sum(radii > np.mean(radii) * ratio) > min_nodes


def _probe_radius(
    mesh: trimesh.Trimesh,
    *,
    probe_radius: float | None,
    probe_multiplier: float,
) -> float:
    """Return the probe radius in *mesh units*.

    If *probe_radius* is given, return it as‑is.  Otherwise compute
    `probe_multiplier × mean(mesh.edges_unique_length)`.
    """
    if probe_radius is not None:
        return float(probe_radius)

    edge_mean = mesh.edges_unique_length.mean()
    return probe_multiplier * edge_mean


def _find_seed(
    counts: np.ndarray,
    gsurf: ig.Graph,
    *,
    top_seed_frac: float = 0.01,
) -> int:
    """Pick a seed inside the densest *blob* rather than the densest *vertex*.

    Parameters
    ----------
    counts
        Local‑density array (`len == n_vertices`).
    gsurf
        Surface graph for connected‑component look‑ups.
    top_seed_frac
        Fraction of top‑ranked vertices considered as seed candidates.  A value
        between 0.5–2 % is usually sufficient.
    """
    k = max(1, int(len(counts) * top_seed_frac))
    top_idx = np.argpartition(counts, -k)[-k:]      # unsorted top‑k set

    sub = gsurf.induced_subgraph(top_idx)
    comps = sub.components()

    # component score: |comp| × mean(counts)  → favours large & dense blobs
    scores = [len(c) * counts[top_idx[c]].mean() for c in comps]
    best_comp = comps[int(np.argmax(scores))]

    comp_vertices = top_idx[best_comp]
    seed_local = int(np.argmax(counts[comp_vertices]))
    return int(comp_vertices[seed_local])


def _soma_surface_vertices(
    mesh: trimesh.Trimesh,
    *,
    gsurf: ig.Graph | None = None,
    probe_radius: float | None = None,
    probe_multiplier: float = 8.0,
    seed_density_cutoff: float = 0.5,
    dilation_steps: int = 1,
    top_seed_frac: float = 0.01,
    seed_point: np.ndarray | list | tuple | None = None,
) -> Set[int]:
    """Return the set of mesh‑vertex indices that form the detected soma shell."""
    v = mesh.vertices.view(np.ndarray)
    gsurf = gsurf if gsurf is not None else _surface_graph(mesh)

    # 1. Probe radius ----------------------------------------------------------------
    probe_r = _probe_radius(mesh, probe_radius=probe_radius, probe_multiplier=probe_multiplier)

    # 2. Local density (vertex neighbour counts) --------------------------------------
    counts = np.asarray(
        KDTree(v).query_ball_point(v, probe_r, return_length=True, workers=-1)
    )

    # 3. Seed selection ---------------------------------------------------------------
    if seed_point is not None:
        seed_idx = int(np.argmin(np.linalg.norm(v - np.asarray(seed_point), axis=1)))
    else:
        seed_idx = _find_seed(counts, gsurf, top_seed_frac=top_seed_frac)

    # 4. Threshold relative to seed density ------------------------------------------
    density_thr = counts[seed_idx] * (1 - seed_density_cutoff)
    dense_mask = counts >= density_thr
    dense_vids = np.where(dense_mask)[0]

    # 5. Connected component containing the seed -------------------------------------
    sub = gsurf.induced_subgraph(dense_vids)
    comps = sub.components()
    seed_sub_idx = int(np.where(dense_vids == seed_idx)[0][0])
    comp_id = comps.membership[seed_sub_idx]
    soma_set: Set[int] = {int(dense_vids[i]) for i in comps[comp_id]}

    # 6. Optional geodesic dilations --------------------------------------------------
    for _ in range(dilation_steps):
        boundary = {
            nb
            for vv in soma_set
            for nb in gsurf.neighbors(vv)
            if nb not in soma_set
        }
        soma_set.update(boundary)

    return soma_set

def find_soma(
    mesh: trimesh.Trimesh,
    *,
    probe_radius: float | None = None,
    probe_multiplier: float = 8.0,
    seed_density_cutoff: float = 0.30,
    dilation_steps: int = 1,
    top_seed_frac: float = 0.02,
    seed_point: np.ndarray | list | tuple | None = None,
    gsurf: ig.Graph | None = None,
) -> Tuple[np.ndarray, float, Set[int]]:
    """Detect the soma surface and return its centroid, radius, and vertex set.

    Parameters
    ----------
    mesh
        Watertight neuron mesh (coordinates in any consistent physical unit).
    probe_radius
        Absolute probe radius in *mesh units*.  Overrides `probe_multiplier` if
        given.
    probe_multiplier
        Adaptive probe size: `probe_multiplier × ⟨edge length⟩`.
    density_relative_drop
        Keep vertices whose local density is at least `(1‑density_relative_drop)`
        of the seed’s density.
    dilation_steps
        Number of geodesic 1‑ring dilations applied to the detected core.
    top_seed_frac
        Fraction of highest‑density vertices used for seed‑component sampling.
    seed_point
        Optional 3‑D coordinate that forces the seed to the nearest vertex.
    gsurf
        Pre‑computed surface graph (reuse for speed).
    verbose
        If *True*, print timing and parameter info to stdout.

    Returns
    -------
    centre, radius, soma_verts
        • *centre*: centroid of the soma vertices.\
        • *radius*: 90‑th percentile of vertex distances from the centroid.\
        • *soma_verts*: set of vertex indices belonging to the soma shell.
    """
    soma_verts = _soma_surface_vertices(
        mesh,
        gsurf=gsurf,
        probe_radius=probe_radius,
        probe_multiplier=probe_multiplier,
        seed_density_cutoff=seed_density_cutoff,
        dilation_steps=dilation_steps,
        top_seed_frac=top_seed_frac,
        seed_point=seed_point,
    )

    pts = mesh.vertices.view(np.ndarray)[list(soma_verts)]
    centre = pts.mean(axis=0)
    if probe_radius:
        # use the probe radius as a hard limit
        radius = probe_radius
    else:
        radius = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 90))
    return centre, radius, soma_verts

def find_soma_with_seed(
    mesh: trimesh.Trimesh,
    seed_point: np.ndarray | list | tuple | None = None,
    seed_radius: float | None = 5_000.,
    gsurf: ig.Graph | None = None,
):
    """Refine soma location given a seed point.

    A user‑supplied seed point is snapped to the closest mesh vertex and a
    geodesic flood‑fill up to lam × prob_radius returns the soma patch.
    """
    v = mesh.vertices.view(np.ndarray)
    seed_vid = int(np.argmin(np.linalg.norm(v - seed_point, axis=1)))
    cutoff = seed_radius
    if gsurf is None:
        gsurf = _surface_graph(mesh)
    dists = gsurf.shortest_paths_dijkstra(source=seed_vid, weights="weight")[0]
    soma_verts = {vid for vid, d in enumerate(dists) if d <= cutoff}
    pts = v[list(soma_verts)]
    centre = pts.mean(axis=0)
    radius = np.percentile(np.linalg.norm(pts - centre, axis=1), 90)
    return centre, radius, soma_verts

def find_soma_with_radius(
    nodes: np.ndarray,
    radii: np.ndarray,
    *,
    pct_large: float = 99.9,
    radius_quantile: float = 0.90,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Post-skeletonization soma detection, using node radii only.

    Parameters
    ----------
    nodes
        ``(N, 3)`` skeleton coordinates.
    radii
        ``(N,)`` radius column (choose the estimator you like best).
    pct_large
        Nodes whose radius is above this percentile are considered
        “soma candidates”.
    radius_quantile
        Outer envelope of the soma = *radius_quantile*-th percentile of
        *distance + local radius* inside the candidate cluster.

    Returns
    -------
    centre
        Centroid of the candidate cluster.
    radius
        Envelope radius (see above).
    soma_nodes
        1-D array of node IDs classified as belonging to the soma.
    """
    # (a) pick “soma candidates” by very large radius
    is_soma = radii >= np.percentile(radii, pct_large)

    if not np.any(is_soma):
        raise RuntimeError("no soma-sized nodes found – check the percentile")

    centre = nodes[is_soma].mean(axis=0)

    # (b) 90-th percentile of distance+radius ➜ outer envelope
    dist_plus_r = np.linalg.norm(nodes[is_soma] - centre, axis=1) + radii[is_soma]
    radius = float(np.percentile(dist_plus_r, radius_quantile * 100))

    soma_nodes = np.where(is_soma)[0]
    return centre, radius, soma_nodes

# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------
def _dist_vec_for_component(
    gsurf: ig.Graph,
    verts: np.ndarray,        # 1-D int64 array of vertex IDs (one component)
    seed_vid: int,            # mesh-vertex ID, must be in *verts*
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
    vids  = np.fromiter(dist_dict.keys(),   dtype=np.int64)
    dists = np.fromiter(dist_dict.values(), dtype=np.float64)

    # --- construct right-open bin edges --------------------------------
    edges = np.arange(0.0, dists.max() + step, step, dtype=np.float64)
    if edges[-1] <= dists.max():            # ensure last edge is strictly greater
        edges = np.append(edges, edges[-1] + step)

    # --- assign each vertex to a shell ---------------------------------
    idx = np.digitize(dists, edges) - 1     # 0-based indices
    idx[idx == len(edges) - 1] -= 1         # clip the “equal-max” case

    # --- build the bins -------------------------------------------------
    bins = [[] for _ in range(len(edges) - 1)]
    for vid, b in zip(vids, idx):
        bins[b].append(int(vid))

    return bins


def _bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> List[int]:
    """
    Return parent[] array of BFS tree from *root* given undirected edge list.
    """
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


def _estimate_radius(d: np.ndarray, *, method: str = "median", trim_fraction: float = 0.05, q: float = 0.90) -> float:
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
    edges_unique: np.ndarray,        # (E, 2) int64
    v2n: dict[int, int],             # mesh-vertex id -> skeleton node id
    n_mesh_verts: int,
) -> np.ndarray:
    """
    Vectorised remap of mesh edges -> skeleton edges.
    """
    # 1. build an int64 lookup table  mesh_vid -> node_id  (-1 if absent)
    lut = np.full(n_mesh_verts, -1, dtype=np.int64)
    lut[list(v2n.keys())] = list(v2n.values())

    # 2. map both columns in one shot
    a, b = edges_unique.T                   # views, no copy
    na, nb = lut[a], lut[b]                 # vectorised gather

    # 3. keep edges whose *both* endpoints exist and are different
    mask = (na >= 0) & (nb >= 0) & (na != nb)
    na, nb = na[mask], nb[mask]

    edges = np.vstack([na, nb]).T
    edges = np.sort(edges, axis=1)          # canonical order
    edges = np.unique(edges, axis=0)        # drop duplicates
    return edges.astype(np.int64)            # copy to new array


# -----------------------------------------------------------------------
#  Post-processing helpers
# -----------------------------------------------------------------------

def _merge_near_soma_nodes(
    nodes: np.ndarray,
    radii: np.ndarray,
    edges: np.ndarray,                      
    *,
    c_soma: np.ndarray,
    r_soma: float,
    soma_merge_dist_factor: float,
    soma_merge_radius_factor: float,
    node2verts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove centroids that sit (a) well inside the soma volume or
    (b) are *very* fat, and reconnect their neighbours to node 0.

    Parameters
    ----------
    nodes, radii, edges
        Skeleton arrays *before* merging.  `nodes[0]` **must** be the soma
        centre and `radii[0]` its envelope radius.
    c_soma, r_soma
        Refined soma centre / radius from the earlier detection step.
    soma_merge_dist_factor
        Discard a node if its distance to the soma centre is below
        `factor × r_soma`.
    soma_merge_radius_factor
        Discard a node if its own radius is above `factor × r_soma`.

    Returns
    -------
    new_nodes, new_radii, new_edges, keep_mask
        Skeleton arrays *after* merging plus the boolean keep-mask that
        was applied.
    """

    # ── 1. decide which vertices survive ────────────────────────────────
    close = np.linalg.norm(nodes - c_soma, axis=1) < soma_merge_dist_factor * r_soma
    fat   = radii >= soma_merge_radius_factor * r_soma
    keep  = ~(close | fat)
    keep[0] = True                     # never drop the soma itself

    kept_idx = np.where(keep)[0]
    old2new = np.empty(len(keep), dtype=np.int64)
    old2new.fill(-1)
    old2new[kept_idx] = np.arange(len(kept_idx), dtype=np.int64)

    # ── 2. edges whose *both* endpoints survive (fast path) ────────────
    a, b = edges.T
    both_keep = keep[a] & keep[b]
    edges_out = edges[both_keep].copy()
    edges_out[:] = old2new[edges_out]                       

    # ── 3. reconnect neighbours of each removed vertex ─────────────────
    candidates : set[int] = set()                 # survivors next to a removed node
    for u, v in edges:                            # original indices
        ku, kv = keep[u], keep[v]
        if ku ^ kv:                               # exactly one survives
            survivors_idx = u if ku else v
            new_idx = int(old2new[survivors_idx])
            candidates.add(new_idx)

    if candidates:
        extra = np.array([[0, idx] for idx in sorted(candidates)], dtype=np.int64)
        edges_out = np.vstack([edges_out, extra])

    # ── 4. canonicalise & uniq ─────────────────────────────────────────
    edges_out = np.sort(edges_out, axis=1)
    edges_out = np.unique(edges_out, axis=0)

    # ── 5. re-centre node 0  (all centroids now belonging to the soma) ──
    merged_idx = np.where(~keep)[0]                 # the ones we collapsed
    if merged_idx.size:
        # number of mesh vertices represented by each centroid
        weights = np.array([len(node2verts[0]), *[len(node2verts[i]) for i in merged_idx]],
                        dtype=np.float64)

        # centres to average
        soma_centroids = np.vstack((nodes[0], nodes[merged_idx]))

        # weighted average = true centroid of all involved mesh vertices
        nodes_keep = nodes[keep].copy()
        nodes_keep[0] = np.average(soma_centroids, axis=0, weights=weights)
    else:
        nodes_keep = nodes[keep]

    return nodes_keep, radii[keep], edges_out, keep


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
        ``(N, 3)`` float32 array of mesh-vertex coordinates.
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

    def _auto_bridge_max(gaps: list[float],
                        edge_mean: float,
                        *,
                        pct: float = 55.,
                        lo: float = 6.,
                        hi: float = 12.) -> float:
        """
        Choose a bridge_max_factor from the initial gap distribution. Default is the
        55th percentile of the gap distribution, clipped to [6, 20] times the mean edge
        """
        raw = np.percentile(gaps, pct) / edge_mean
        return float(np.clip(raw, lo, hi))

    def _auto_recalc_after(n_gaps: int) -> int:
        """Return a suitable recalc period for the given number of gaps."""
        if n_gaps <= 10:          # tiny: update often
            return 2
        if n_gaps <= 50:          # small: every 3–4 merges
            return 4
        if n_gaps <= 200:         # medium: every ~5 % of gaps
            return max(4, n_gaps // 20)
        # giant meshes: cap to 32 so we never starve
        return 32 


    # -- 0. quick exit if already connected ---------------------------------
    g = ig.Graph(n=len(nodes), edges=[tuple(map(int, e)) for e in edges], directed=False)
    comps = [set(c) for c in g.components()]
    comp_idx = {cid: np.fromiter(verts, dtype=np.int64)
            for cid, verts in enumerate(comps)}
    soma_cid = g.components().membership[0]
    if len(comps) == 1:
        return edges


    # -- 1. build one KD-tree per component ---------------------------------
    edge_len_mean = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1).mean()

    # -- 2. grow the island using a distance-ordered priority queue ---------
    island = set(comps[soma_cid])
    island_idx  = np.fromiter(island, dtype=np.int64)
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
    current_max  = bridge_max_factor * edge_len_mean

    while pq:
        _, cid, _, _ = heapq.heappop(pq)
        # postpone if the component is still too far away
        gap, best_c, best_i = closest_pair(cid)

        if gap > current_max:
            heapq.heappush(pq, (gap, cid, best_c, best_i))   # still too far, re-queue
            stall += 1
            if stall >= 2 * max_stall:
                # after too many futile tries, do a *forced* heap rebuild
                pq = [(g, cid2, bc, bi)
                    for _, cid2, _, _ in pq
                    for g, bc, bi in (closest_pair(cid2),)]
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
        island_idx   = np.concatenate([island_idx, verts_idx])
        island_tree  = KDTree(nodes[island_idx])

        merges_since_recalc += 1
        if merges_since_recalc >= recalc_after:
            # distances of *every* remaining component may have changed
            pq = [(gap, cid, b_comp, b_is)
                    for _, cid, _, _ in pq
                    for gap, b_comp, b_is in (closest_pair(cid),)]
            heapq.heapify(pq)
            merges_since_recalc = 0

    if edges_new:
        edges_aug = np.vstack([edges, np.asarray(edges_new, dtype=np.int64)])
    else:
        edges_aug = edges

    return np.unique(edges_aug, axis=0)

def _build_mst(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Return edge list of the global minimum-spanning tree.
    """
    g = ig.Graph(
        n=len(nodes), edges=[tuple(map(int, e)) for e in edges], directed=False
    )
    g.es["weight"] = [
        float(np.linalg.norm(nodes[a] - nodes[b])) for a, b in edges
    ]
    mst = g.spanning_tree(weights="weight")
    return np.asarray(sorted(tuple(sorted(e)) for e in mst.get_edgelist()), dtype=np.int64)


def _prune_soma_neurites(
    nodes: np.ndarray,
    edges: np.ndarray,
    soma_pos: np.ndarray,
    r_soma: float,
    min_branch_nodes: int = 30,
    min_branch_extent_factor: float = 1.8,
):
    """
    Remove spuriously small neurites that attach directly to the soma.

    Parameters
    ----------
    nodes
        (N, 3) coordinates array after collapse/bridging/MST.
    edges
        (E, 2) int64 undirected edge list (tree).
    soma_pos
        3-vector of the soma centre (node 0).
    r_soma
        Soma radius.
    min_branch_nodes
        Keep a branch if it has at least this many nodes.
    min_branch_extent_factor
        Keep a branch if any node lies further than
        ``min_branch_extent_factor × r_soma`` from the soma.

    Returns
    -------
    keep_mask : np.ndarray[bool]
        True for nodes to keep.
    new_edges : np.ndarray[int64] shape (E′, 2)
        Edge list re-indexed to the surviving nodes.
    """

    # ------------------------------------------------------------------
    # 1. build children lists from the parent[] array
    # ------------------------------------------------------------------
    parent = _bfs_parents(edges, len(nodes), root=0)
    children_of = defaultdict(list)
    for v, p in enumerate(parent):
        if p != -1:
            children_of[p].append(v)

    soma_children = children_of[0]                 # neighbours of the soma
    to_drop: set[int] = set()

    # ------------------------------------------------------------------
    # 2. inspect every branch below each soma child
    # ------------------------------------------------------------------
    for c in soma_children:
        stack = [c]
        desc  = []                                 # vertices in this branch
        while stack:
            u = stack.pop()
            desc.append(u)
            stack.extend(children_of.get(u, []))   # grandchildren …

        desc_arr = np.asarray(desc, dtype=np.int64)
        size     = desc_arr.size
        extent   = np.linalg.norm(nodes[desc_arr] - soma_pos, axis=1).max()

        keep = (size >= min_branch_nodes) or (
            extent >= min_branch_extent_factor * r_soma
        )
        if not keep:
            to_drop.update(desc)

    # nothing to drop?  → fast-path
    if not to_drop:
        return np.ones(len(nodes), bool), edges

    # ------------------------------------------------------------------
    # 3. rebuild node list and edge list
    # ------------------------------------------------------------------
    keep_mask = np.ones(len(nodes), bool)
    keep_mask[list(to_drop)] = False
    keep_mask[0] = True                        # never drop the soma

    # old-index → new-index map
    old2new = {old: new for new, old in enumerate(np.where(keep_mask)[0])}

    new_edges = np.asarray(
        [
            (old2new[a], old2new[b])
            for a, b in edges
            if keep_mask[a] and keep_mask[b]
        ],
        dtype=np.int64,
    )
    return keep_mask, new_edges

def _extreme_vertex(mesh: trimesh.Trimesh,
                    axis: str = "z",
                    mode: str = "min") -> int:
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

# -----------------------------------------------------------------------------
#  Skeletonization Public API
# -----------------------------------------------------------------------------

def skeletonize(
    mesh: trimesh.Trimesh,
    # --- radius estimation ---
    radius_estimators: list[str] = ["median", "mean", "trim"],
    # --- soma detection ---
    detect_soma: str | bool = "post", # options: {"post", "pre", "seed", "off" or False}
    soma_seed_point: np.ndarray | list | tuple | None = None,
    soma_seed_radius: float | None = None,
    soma_seed_radius_multiplier: float = 8.0,
    soma_density_cutoff: float = 0.50,
    soma_dilation_steps: int = 1,
    soma_top_seed_frac: float = 0.02,
    # -- for post-skeletonization soma detection only--
    soma_init_guess_axis: str  = "z",   # "x" | "y" | "z"
    soma_init_guess_mode: str  = "min", # "min" | "max"
    # --- geodesic sampling ---
    target_shell_count: int = 500,
    min_cluster_vertices: int = 6,
    max_shell_width_factor: int = 50,
    # --- bridging disconnected patches ---
    bridge_gaps: bool = True,
    bridge_max_factor: float | None = None,
    bridge_recalc_after: int | None = None,
    # --- post‑processing ---
    # --- collapse soma-like nodes ---
    collapse_soma: bool = True,
    collapse_soma_dist_factor: float = 1.15,
    collapse_soma_radius_factor: float = 0.25,
    # --- prune tiny neurites ---
    prune_tiny_neurites: bool = True,
    prune_min_branch_nodes: int = 50,
    prune_min_branch_extent_factor: float = 1.8,
    # --- misc ---
    verbose: bool = False,
) -> Skeleton:
    """Compute a centre-line skeleton with radii of a neuronal mesh .

    The algorithm proceeds in eight conceptual stages:

      0. optional pre-skeletonization soma detection>
      1. geodesic shell binning of every connected surface patch
      2. cluster each shell ⇒ interior node with local radius
      3. optional post-skeletonization soma detection>
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
        print(f"[skeliner] starting skeletonisation ({len(mesh.vertices)} vertices, "
              f"{len(mesh.faces)} faces)")
        soma_ms = 0.0 # soma detection time
        post_ms = 0.0 # post-processing time

    @contextmanager
    def _timed(label: str, *, verbose: bool = True):      # keep the signature you like
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

        PAD = 47                       # keeps the old alignment
        print(f" {label:<{PAD}} …", end="", flush=True)
        t0 = time.perf_counter()
        _msgs: list[str] = []

        def log(msg: str) -> None:
            _msgs.append(str(msg))

        try:
            yield log                   # the `with`-body gets this function
        finally:
            dt = time.perf_counter() - t0
            print(f" {dt:.2f} s")       # finish first line

            for m in _msgs:             # then all sub-messages, nicely indented
                print(f"      └─ {m}")

    # 0. soma vertices ---------------------------------------------------
    with _timed("↳  build surface graph"):
        gsurf = _surface_graph(mesh)

    if detect_soma == "post" or detect_soma == "off" or detect_soma is False:
        # delay soma detection until after skeletonization
        # but seed point is still needed for binning
        if soma_seed_point is not None:
            seed_vid = int(np.argmin(np.linalg.norm(mesh.vertices.view(np.ndarray) - np.asarray(soma_seed_point), axis=1)))
        else:
            seed_vid = _extreme_vertex(mesh,
                                       axis=soma_init_guess_axis,
                                       mode=soma_init_guess_mode)

        c_soma     = mesh.vertices.view(np.ndarray)[seed_vid]
        r_soma     = float(mesh.edges_unique_length.mean()) * 3   # hard-coded, not ideal
        soma_verts = {seed_vid}
    else:
        # pre skeletonization soma detection
        with _timed("↳  pre-skeletonization soma detection"):
            if detect_soma == "pre":
                c_soma, r_soma, soma_verts = find_soma(
                    mesh,
                    gsurf=gsurf,
                    probe_radius=soma_seed_radius,
                    probe_multiplier=soma_seed_radius_multiplier,
                    seed_density_cutoff=soma_density_cutoff,
                    dilation_steps=soma_dilation_steps,
                    top_seed_frac=soma_top_seed_frac,
                    seed_point=soma_seed_point,
                )
            elif detect_soma == "seed" and soma_seed_point is not None:
                c_soma, r_soma, soma_verts = find_soma_with_seed(
                    mesh,
                    gsurf=gsurf,
                    seed_point=soma_seed_point,
                    seed_radius=soma_seed_radius,
                )

        if verbose:
            soma_ms = time.perf_counter() - _global_start
            

    # 1. binning along geodesic shells ----------------------------------
    with _timed("↳  bin surface vertices by geodesic distance"):
        v = mesh.vertices.view(np.ndarray)
        
        components    = gsurf.components()
        comp_vertices = [np.asarray(c, dtype=np.int64) for c in components]

        soma_vids = np.fromiter(soma_verts, dtype=np.int64)
        all_comps: list[list[np.ndarray]] = []
        edge_m    = float(mesh.edges_unique_length.mean())

        for cid, verts in enumerate(comp_vertices):

            # -------- choose the *one* seed you used before ----------                                   
            if np.intersect1d(verts, soma_vids).size:
                seed_vid = int(
                    soma_vids[
                        np.argmax(np.linalg.norm(v[soma_vids] - c_soma, axis=1))
                    ]
                )
            else:
                seed_vid = int(verts[hash(cid) % len(verts)])

            dist_vec = _dist_vec_for_component(gsurf, verts, seed_vid)
            dist_sub = {int(v): float(d) for v, d in zip(verts, dist_vec)}

            if not dist_sub:
                continue

            arc_len = max(dist_sub.values())
            step    = max(edge_m * 2.0, arc_len / target_shell_count)

            bins: list[list[int]] = []
            while not any(bins) and step < edge_m * max_shell_width_factor:
                bins = _geodesic_bins(dist_sub, step)
                step *= 1.5

            # --- build the component list for every bin ----------------------
            for bin_verts in bins:
                inner  = [vid for vid in bin_verts if vid not in soma_verts]
                sub    = gsurf.induced_subgraph(inner)
                comps  = []
                for comp in sub.components():
                    if len(comp) < min_cluster_vertices:
                        continue
                    comp_idx = np.fromiter((inner[i] for i in comp), dtype=np.int64)
                    comps.append(comp_idx)
                all_comps.append(comps)    

    # 2. create skeleton nodes ------------------------------------------
    with _timed("↳  compute bin centroids and radii"):
        nodes: List[np.ndarray] = [c_soma]
        radii: Dict[str, List[float]] = {
            est: [r_soma] for est in radius_estimators
        }
        node2verts: List[np.ndarray] = [np.fromiter(soma_verts, dtype=np.int64)] # map node ID to mesh-vertex IDs
        vert2node: Dict[int, int] = {v: 0 for v in soma_verts} # map mesh-vertex ID to node ID
        next_id = 1

        for shell_idx, comps in enumerate(all_comps):          # ← iterate over shells
            for comp_idx in comps:                             #    iterate over comps
                pts    = v[comp_idx]
                centre = pts.mean(axis=0)                      # centre of component

                d = np.linalg.norm(pts - centre, axis=1)  # radii sample

                # store radii for every estimator
                for est in radius_estimators:
                    radii.setdefault(est, []).append(
                        _estimate_radius(d, method=est, trim_fraction=0.05)
                    )

                node_id = next_id
                next_id += 1
                nodes.append(centre)
                node2verts.append(comp_idx)
                for vv in comp_idx:
                    vert2node[int(vv)] = node_id

        nodes_arr = np.asarray(nodes, dtype=np.float32)
        radii_dict = {
            k: np.asarray(v, dtype=np.float32) for k, v in radii.items()
        }
        # check if we have soma candidates
        r_primary = radii_dict[radius_estimators[0]]
        if detect_soma == "off" or detect_soma is False:
            has_soma = False
        else:
            has_soma = _likely_has_soma(r_primary)


    if has_soma and detect_soma == "post":
        with _timed("↳  post-skeletonization soma detection"):
            c_soma, r_soma, soma_nodes = find_soma_with_radius(
                nodes_arr, radii_dict[radius_estimators[0]]
            )

            # 1. choose which candidate will become the new root
            if 0 not in soma_nodes:
                # pick the fattest candidate (max radius) for robustness
                new_root = int(
                    soma_nodes[np.argmax(radii_dict[radius_estimators[0]][soma_nodes])]
                )

                # 2. swap node 0  ⇄  new_root  in every structure
                nodes_arr[[0, new_root]] = nodes_arr[[new_root, 0]]
                for k in radii_dict:
                    radii_dict[k][[0, new_root]] = radii_dict[k][[new_root, 0]]
                node2verts[0], node2verts[new_root] = node2verts[new_root], node2verts[0]

                # 3. fix the vertex-to-node lookup used later by _edges_from_mesh()
                for vid, nid in vert2node.items():
                    if nid == 0:
                        vert2node[vid] = new_root
                    elif nid == new_root:
                        vert2node[vid] = 0

                # 4. make the ID set consistent
                soma_nodes = np.where(soma_nodes == new_root, 0, soma_nodes)

            # 5. finally update the canonical soma centre/radius
            c_soma = nodes_arr[0]
            for k in radii_dict:
                radii_dict[k][0] = r_soma

    # 3. edges from mesh connectivity -----------------------------------
    with _timed("↳  map mesh faces to skeleton edges"):
        edges_arr = _edges_from_mesh(
            mesh.edges_unique,
            vert2node,
            n_mesh_verts=len(mesh.vertices),
        )
    

    # 4. collapse soma‑like / fat nodes ---------------------------
    if has_soma and collapse_soma:
        _t0 = time.perf_counter() 

        with _timed("↳  merge redundant near-soma nodes") as log:
            (nodes_arr, radii_arr, edges_arr, keep_mask) = _merge_near_soma_nodes(
                nodes_arr,
                radii_dict[radius_estimators[0]],
                edges_arr,
                c_soma=c_soma,
                r_soma=radii_dict[radius_estimators[0]][0],
                soma_merge_dist_factor=collapse_soma_dist_factor,
                soma_merge_radius_factor=collapse_soma_radius_factor,
                node2verts=node2verts,
            )
            log(f"{(~keep_mask).sum()} nodes merged into soma")

            # --- rebuild the soma list ---
            removed_idx = np.where(~keep_mask)[0]          # nodes that were collapsed
            node2verts_keep = [node2verts[i] for i in np.where(keep_mask)[0]]
            for idx in removed_idx:
                soma_verts.update(node2verts[idx])          # node2verts from stage 2
                node2verts_keep[0] = np.concatenate((node2verts_keep[0], node2verts[idx]))
            node2verts = node2verts_keep
            vert2node = {int(v): int(new_id)
                        for new_id, verts in enumerate(node2verts)
                        for v in verts}
            for k in radii_dict:
                radii_dict[k] = radii_dict[k][keep_mask]
                radii_dict[k][0] = radii_arr[0]

            # update the vertex ID map
            c_soma = nodes_arr[0]

        if verbose:
            post_ms += time.perf_counter() - _t0

    # 5. Connect all components ------------------------------
    if bridge_gaps:
        _t0 = time.perf_counter() 
        with _timed("↳  bridge skeleton gaps")  as log:
            edges_arr = _bridge_gaps(
                nodes_arr, edges_arr, 
                bridge_max_factor = bridge_max_factor,
                bridge_recalc_after= bridge_recalc_after,
            )
        if verbose:
            post_ms += time.perf_counter() - _t0

    # 6. global minimum-spanning tree ------------------------------------
    with _timed("↳  build global minimum-spanning tree"):
        edges_mst = _build_mst(nodes_arr, edges_arr)
        post_ms += time.perf_counter() - _t0

    # 7. prune tiny sub-trees near the soma
    if has_soma and prune_tiny_neurites:
        _t0 = time.perf_counter()
        with _timed("↳  prune tiny soma-attached branches"):
            keep_mask, edges_mst = _prune_soma_neurites(
                nodes_arr,
                edges_mst,
                c_soma,
                radii_dict[radius_estimators[0]][0],
                min_branch_nodes=prune_min_branch_nodes,           
                min_branch_extent_factor=prune_min_branch_extent_factor,
            )
            nodes_arr  = nodes_arr[keep_mask]
            for k in radii_dict:
                radii_dict[k] = radii_dict[k][keep_mask]
            node2verts = [node2verts[i] for i in np.where(keep_mask)[0]]
            vert2node = {int(v): int(new_id)
                        for new_id, verts in enumerate(node2verts)
                        for v in verts}
        if verbose:
            post_ms += time.perf_counter() - _t0

    if verbose:
        total_ms = time.perf_counter() - _global_start
        core_ms  = total_ms - soma_ms - post_ms

        if post_ms > 1e-6:       # at least one optional stage ran
            print(f"{'TOTAL (soma + core + post)':<49}"
                  f"… {total_ms:.1f} s "
                  f"({soma_ms:.1f} + {core_ms:.1f} + {post_ms:.1f})")
            print(f"({len(nodes_arr):,} nodes, {edges_mst.shape[0]:,} edges)")
        else:                    # no post-processing at all
            print(f"{'TOTAL (soma + core)':<49}"
                  f"… {total_ms:.1f} s "
                  f"({soma_ms:.1f} + {core_ms:.1f})")

    return Skeleton(nodes_arr, 
                    radii_dict, 
                    edges_mst, 
                    soma_verts=np.asarray(list(soma_verts), dtype=np.int64),
                    node2verts=node2verts,
                    vert2node=vert2node,
            )