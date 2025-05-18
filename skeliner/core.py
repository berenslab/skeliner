import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import KDTree as _KDTree

__all__ = [
    "Skeleton",
    "find_soma",
    "skeletonize",
    "load_swc",
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
    def soma_mask(self) -> np.ndarray | None:
        """
        Boolean mask over `nodes` that is True for vertices whose mesh
        vertex lay on the detected soma surface (None if not available).
        """
        if self.soma_verts is None:
            return None
        mask = np.zeros(len(self.nodes), dtype=bool)
        mask[list(self.soma_verts)] = True
        return mask

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
#  Soma detection helpers
# -----------------------------------------------------------------------------

def _soma_surface_vertices(
    mesh: trimesh.Trimesh,
    soma_probe_radius_mult: float = 8.0,
    soma_density_threshold: float = 0.5,
    soma_dilation_steps: int = 1,
    gsurf: ig.Graph | None = None,
) -> Set[int]:
    """Heuristic soma‑surface detection.

    A density filter first selects vertices with the maximum neighbourhood
    count inside a spherical probe. The largest connected component of this
    high‑density set is considered soma; optional geodesic dilation helps
    compensate mesh irregularities.

    Notes
    -----
    This is a quick‑and‑dirty heuristic that works well for typical electron
    microscopy segmentations but is not guaranteed to find the anatomical
    soma for arbitrarily shaped meshes.
    """    
    v = mesh.vertices.view(np.ndarray)
    if gsurf is None:
        gsurf = _surface_graph(mesh)

    edge_m = mesh.edges_unique_length.mean()
    probe_r = soma_probe_radius_mult * edge_m

    counts = np.asarray(_KDTree(v).query_ball_point(v, probe_r, return_length=True, workers=-1))

    dense_mask = counts >= counts.max() * soma_density_threshold
    dense_vids = np.where(dense_mask)[0]

    # largest connected component within dense_vids
    sub = gsurf.induced_subgraph(dense_vids)
    soma_comp: list[int] = max(sub.components(), key=len)
    soma_set: Set[int] = {dense_vids[i] for i in soma_comp}

    # geodesic dilation
    for _ in range(soma_dilation_steps):
        boundary = {
            nb
            for vv in soma_set
            for nb in gsurf.neighbors(vv)
            if nb not in soma_set
        }
        soma_set.update(boundary)

    return soma_set

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
    return sub.shortest_paths_dijkstra(
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
    return edges.astype(np.int64, copy=False)


# -----------------------------------------------------------------------
#  Post-processing helpers
# -----------------------------------------------------------------------

def _collapse_soma_nodes(
    nodes: np.ndarray,
    radii: np.ndarray,
    edges: np.ndarray,                      
    *,
    c_soma: np.ndarray,
    r_soma: float,
    soma_merge_dist_factor: float,
    soma_merge_radius_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge near‑soma redundant centroids and reconnect edges.
    """
    # ── 1. decide which vertices survive ────────────────────────────────
    close = np.linalg.norm(nodes - c_soma, axis=1) < soma_merge_dist_factor * r_soma
    fat   = radii >= soma_merge_radius_factor * r_soma
    keep  = ~(close | fat)
    keep[0] = True # keep soma                                        

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
    removed = np.where(~keep)[0]
    if removed.size:
        neigh: Dict[int, List[int]] = {int(r): [] for r in removed}
        for u, v in edges.tolist():                         
            if keep[u] and not keep[v]:
                neigh[v].append(u)
            elif keep[v] and not keep[u]:
                neigh[u].append(v)

        extra: List[Tuple[int, int]] = []
        for vs in neigh.values():
            if len(vs) > 1:
                base = vs[0]
                extra.extend((base, o) for o in vs[1:])

        if extra:
            extra_arr = np.asarray(extra, dtype=np.int64)
            extra_arr[:] = old2new[extra_arr]
            edges_out = np.vstack([edges_out, extra_arr])

    # ── 4. canonicalise & uniq ─────────────────────────────────────────
    edges_out = np.sort(edges_out, axis=1)
    edges_out = np.unique(edges_out, axis=0)

    return nodes[keep], radii[keep], edges_out, keep


def _bridge_components(
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    bridge_k: int,
) -> np.ndarray:
    """
    Use KD-trees to connect all graph components so that everything is
    reachable from the soma.  Adds synthetic edges between k nearest
    pairs (default k=1).
    """
    g_full = ig.Graph(
        n=len(nodes), edges=[tuple(map(int, e)) for e in edges], directed=False
    )
    g_full.es["weight"] = [
        float(np.linalg.norm(nodes[a] - nodes[b])) for a, b in edges
    ]

    vc = g_full.components()
    soma_comp_id = vc.membership[0]
    comps = [set(c) for c in vc]

    if len(comps) == 1:
        return edges  # already connected

    # one KD-tree per component
    kdtrees = {cid: _KDTree(nodes[list(verts)]) for cid, verts in enumerate(comps)}

    main_island = set(comps[soma_comp_id])
    island_tree = kdtrees[soma_comp_id]
    pending = [cid for cid in range(len(comps)) if cid != soma_comp_id]

    edges_aug = edges.copy()
    while pending:
        cid = pending.pop(0)
        verts = comps[cid]
        pts = nodes[list(verts)]

        dists, island_idx = island_tree.query(pts, k=1, workers=-1)
        island_idx = np.asarray(island_idx, dtype=np.intp) # fix type
        order = np.argsort(dists)
        u_candidates = np.fromiter(verts, dtype=np.int64)
        island_list = np.fromiter(main_island, dtype=np.int64)

        for j in order[: bridge_k]:
            u = int(u_candidates[j])
            v = int(island_list[island_idx[j]])
            edges_aug = np.vstack([edges_aug, [u, v]])

        # merge component into island and rebuild KD-tree
        main_island |= verts
        island_tree = _KDTree(nodes[list(main_island)])

    return edges_aug

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

# -----------------------------------------------------------------------------
#  Public API
# -----------------------------------------------------------------------------

def find_soma(
    mesh: trimesh.Trimesh,
    soma_probe_radius_mult: float = 8.0,
    soma_density_threshold: float = 0.30,
    soma_dilation_steps: int = 1,
    gsurf: ig.Graph | None = None,
):
    """Detect the soma surface by local surface-density filtering.

    The algorithm quickly estimates a density map over mesh vertices, keeps the
    densest connected cluster, and optionally performs a few geodesic dilation
    steps to thicken the mask.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Watertight mesh of the neuron.
    soma_probe_radius_mult : float, default ``8.0``
        Probe radius = ``soma_probe_radius_mult × ⟨edge length⟩``.
    soma_density_threshold : float, default ``0.30``
        Keep vertices whose probe counts are at least this fraction of the
        *maximum* count.
    soma_dilation_steps : int, default ``1``
        Number of geodesic dilations (1-ring expansions) of the dense core.
    gsurf : ig.Graph | None, optional
        Pre-computed surface graph to reuse (speed).

    Returns
    -------
    centre : np.ndarray
        Cartesian coordinates of the soma centroid.
    radius : float
        Approximate soma radius (90‑th percentile of point distances).
    soma_verts : set[int]
        Mesh‑vertex indices that form the detected soma surface.
    """
    soma_verts = _soma_surface_vertices(
        mesh,
        gsurf=gsurf,
        soma_probe_radius_mult=soma_probe_radius_mult,
        soma_density_threshold=soma_density_threshold,
        soma_dilation_steps=soma_dilation_steps,
    )
    pts = mesh.vertices.view(np.ndarray)[list(soma_verts)]
    centre = pts.mean(axis=0)
    radius = np.percentile(np.linalg.norm(pts - centre, axis=1), 90)
    return centre, radius, soma_verts


def find_soma_with_seed(
    mesh: trimesh.Trimesh,
    seed: np.ndarray,
    seed_r: float,
    lam: float = 1.15,
    gsurf: ig.Graph | None = None,
):
    """Refine soma location given a seed point.

    A user‑supplied seed point is snapped to the closest mesh vertex and a
    geodesic flood‑fill up to lam × seed_r returns the soma patch.
    """
    v = mesh.vertices.view(np.ndarray)
    seed_vid = int(np.argmin(np.linalg.norm(v - seed, axis=1)))
    cutoff = seed_r * lam
    if gsurf is None:
        gsurf = _surface_graph(mesh)
    dists = gsurf.shortest_paths_dijkstra(source=seed_vid, weights="weight")[0]
    soma_verts = {vid for vid, d in enumerate(dists) if d <= cutoff}
    pts = v[list(soma_verts)]
    centre = pts.mean(axis=0)
    radius = np.percentile(np.linalg.norm(pts - centre, axis=1), 90)
    return centre, radius, soma_verts


def skeletonize(
    mesh: trimesh.Trimesh,
    # --- radius estimation ---
    radius_estimators: list[str] = ["median", "mean", "trim"],
    # --- soma detection ---
    soma_probe_radius_mult: float = 10.0,
    soma_density_threshold: float = 0.30,
    soma_dilation_steps: int = 1,
    soma_seed_point: np.ndarray | None = None,
    soma_seed_radius: float = 7_500.0,
    path_len_relax: float = 1.15,
    # --- geodesic sampling ---
    target_shell_count: int = 500,
    min_cluster_vertices: int = 6,
    max_shell_width_factor: int = 50,
    # --- bridging disconnected patches ---
    bridge_components: bool = True,
    bridge_k: int = 1,
    # --- post‑processing ---
    collapse_soma: bool = True,
    soma_merge_dist_factor: float = 1.0,
    soma_merge_radius_factor: float = 0.25,
    prune_tiny_neurites: bool = True,
    min_branch_nodes: int = 30,
    min_branch_extent_factor: float = 1.8,
    # --- misc ---
    verbose: bool = False,
) -> Skeleton:
    """Compute a centre-line skeleton with radii of a neuronal mesh .

    The algorithm proceeds in eight conceptual stages:

      0. soma localisation (:func:`find_soma` or the seeded variant)
      1. geodesic shell binning of every connected surface patch
      2. cluster each shell ⇒ interior node with local radius
      3. project mesh edges ⇒ graph edges between nodes
      4. optional collapsing of soma-like/fat nodes near the centroid
      5. optional bridging of disconnected components
      6. minimum-spanning tree (global) to remove microscopic cycles
      7. optional pruning of tiny neurites sprouting directly from the soma

    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface mesh of the neuron in *arbitrary* units.
    target_shell_count : int, default ``500``
        Rough number of geodesic shells to produce per component.  The actual
        shell width is adapted to mesh resolution.
    bridge_components : bool, default ``True``
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
        print("[skeliner] starting skeletonisation")

    @contextmanager
    def _timed(label: str):
        """Print *label* before the block and time its execution."""
        if verbose:
            print(f" {label:<47} …", end="", flush=True)
            _t0 = time.perf_counter()
            try:
                yield
            finally:
                dt_ms = (time.perf_counter() - _t0)
                print(f" {dt_ms:8.1f} s")
        else:
            yield

    # 0. soma vertices ---------------------------------------------------
    with _timed("↳  detect soma"):
        gsurf = _surface_graph(mesh)
        if soma_seed_point is None:
            c_soma, r_soma, soma_verts = find_soma(
                mesh,
                gsurf=gsurf,
                soma_probe_radius_mult=soma_probe_radius_mult,
                soma_density_threshold=soma_density_threshold,
                soma_dilation_steps=soma_dilation_steps,
            )
        else:
            c_soma, r_soma, soma_verts = find_soma_with_seed(
                mesh,
                gsurf=gsurf,
                seed=soma_seed_point,
                seed_r=soma_seed_radius,
                lam=path_len_relax,
            )

    # 1. binning along geodesic shells ----------------------------------
    with _timed("↳  partition surface into geodesic shells"):
        v = mesh.vertices.view(np.ndarray)
        
        components    = gsurf.components()
        comp_vertices = [np.asarray(c, dtype=np.int64) for c in components]

        soma_vids = np.fromiter(soma_verts, dtype=np.int64)
        all_bins  : list[list[int]] = []
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

            # -------- adaptive shell binning (unchanged) -------------
            if not dist_sub:
                continue
            arc_len = max(dist_sub.values())
            step    = max(edge_m * 2.0, arc_len / target_shell_count)

            bins: list[list[int]] = []
            while not any(bins) and step < edge_m * max_shell_width_factor:
                bins = _geodesic_bins(dist_sub, step)
                step *= 1.5

            all_bins.extend(bins)
        
    # 2. create skeleton nodes ------------------------------------------
    with _timed("↳  place centroids + local radius"):
        nodes: List[np.ndarray] = [c_soma]
        radii: Dict[str, List[float]] = {
            est: [r_soma] for est in radius_estimators
        }
        v2n: Dict[int, int] = {v: 0 for v in soma_verts}
        next_id = 1

        for verts in all_bins:
            inner = [vid for vid in verts if vid not in soma_verts]
            if len(inner) < min_cluster_vertices:
                continue
            sub = gsurf.induced_subgraph(inner)
            for comp in sub.components():
                if len(comp) < min_cluster_vertices:
                    continue
                comp_idx = np.fromiter((inner[i] for i in comp), dtype=np.int64)
                pts      = v[comp_idx]
                centre   = pts.mean(axis=0)
                d = np.linalg.norm(pts - centre, axis=1)

                # collect radii for every requested estimator
                for est in radius_estimators:
                    val = _estimate_radius(
                        d, method=est, trim_fraction=0.05
                    )
                    radii.setdefault(est, []).append(val)

                node_id = next_id
                next_id += 1
                nodes.append(centre)
                for vv in comp_idx:
                    v2n[int(vv)] = node_id

        nodes_arr = np.asarray(nodes, dtype=np.float32)
        nodes_arr = np.asarray(nodes, dtype=np.float32)
        radii_dict = {
            k: np.asarray(v, dtype=np.float32) for k, v in radii.items()
        }

    # 3. edges from mesh connectivity -----------------------------------
    with _timed("↳  map mesh edges → skeleton edges"):
        edges_arr = _edges_from_mesh(
            mesh.edges_unique,
            v2n,
            n_mesh_verts=len(mesh.vertices),
        )

    # 4. collapse soma‑like / fat nodes ---------------------------
    if collapse_soma:
        with _timed("↳  merge redundant near-soma nodes"):
            (
                nodes_arr,
                _,
                edges_arr,
                keep_mask
            ) = _collapse_soma_nodes(
                nodes_arr,
                radii_dict[radius_estimators[0]],
                edges_arr,
                c_soma=c_soma,
                r_soma=radii_dict[radius_estimators[0]][0],
                soma_merge_dist_factor=soma_merge_dist_factor,
                soma_merge_radius_factor=soma_merge_radius_factor,
            )

            # apply same mask to all radius columns
            for k in radii_dict:
                radii_dict[k] = radii_dict[k][keep_mask]

    # 5. Connect all components ------------------------------
    if bridge_components:
        with _timed("↳  reconnect mesh gaps"):
            edges_arr = _bridge_components(nodes_arr, edges_arr, bridge_k=bridge_k)

    # 6. global minimum-spanning tree ------------------------------------
    with _timed("↳  build global minimum-spanning tree"):
        edges_mst = _build_mst(nodes_arr, edges_arr)

    # 7. prune tiny sub-trees near the soma
    if prune_tiny_neurites:
        with _timed("↳  prune tiny soma-attached branches"):
            keep_mask, edges_mst = _prune_soma_neurites(
                nodes_arr,
                edges_mst,
                c_soma,
                radii_dict[radius_estimators[0]][0],
                min_branch_nodes=min_branch_nodes,           
                min_branch_extent_factor=min_branch_extent_factor,
            )
            nodes_arr  = nodes_arr[keep_mask]
            for k in radii_dict:
                radii_dict[k] = radii_dict[k][keep_mask]

    if verbose:
        total_ms = (time.perf_counter() - _global_start)
        print(f"{'TOTAL':<50} {total_ms:8.1f} s")

    return Skeleton(nodes_arr, 
                    radii_dict, 
                    edges_mst, 
                    soma_verts=np.asarray(list(soma_verts), dtype=np.int64)
            )

# -----------------------------------------------------------------------------
#  SWC loader (unchanged)
# -----------------------------------------------------------------------------

def load_swc(
    path: str | Path,
    scale: float = 1.0,
    keep_types: Iterable[int] | None = None,
) -> Skeleton:
    """
    Read an SWC file into a :class:`Skeleton`.

    Parameters
    ----------
    path : str or Path
        Filename of the SWC file.
    scale : float, default ``1.0``
        Multiply coordinates and radii by this factor on load.
    keep_types : Iterable[int] | None, optional
        Sequence of SWC *type* codes to keep.  When *None* (default) all
        node types are imported.

    Returns
    -------
    Skeleton
        The imported skeleton.  Note that ``soma_verts`` is **None** because
        the SWC format has no concept of surface vertices.

    Raises
    ------
    ValueError
        If the file contains no valid nodes.
    """    
    path = Path(path)
    ids: List[int] = []
    xyz: List[List[float]] = []
    radii: List[float] = []
    parent: List[int] = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            _id, _type = int(parts[0]), int(parts[1])
            if keep_types is not None and _type not in keep_types:
                continue
            ids.append(_id)
            xyz.append([float(parts[2]), float(parts[3]), float(parts[4])])
            radii.append(float(parts[5]))
            parent.append(int(parts[6]))
    if not ids:
        raise ValueError(f"No nodes found in {path}")

    id_map = {old: new for new, old in enumerate(ids)}
    nodes_arr = np.asarray(xyz, dtype=np.float32) * scale
    radii_dict = {"median": np.asarray(radii, dtype=np.float32) * scale}
    edges = [
        (id_map[i], id_map[p])
        for i, p in zip(ids, parent, strict=True)
        if p != -1 and p in id_map
    ]
    return Skeleton(nodes_arr, radii_dict, np.asarray(edges, dtype=np.int64))
