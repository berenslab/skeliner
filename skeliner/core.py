from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import cKDTree as _KDTree

__all__ = [
    "Skeleton",
    "find_soma",
    "skeletonize",
    "load_swc",
]

# -----------------------------------------------------------------------------
#  Core dataclass
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class Skeleton:
    """A minimal, numpy‑backed skeleton graph (igraph backend)."""

    nodes: np.ndarray  # (N, 3) float32
    radii: np.ndarray  # (N,)  float32
    edges: np.ndarray  # (E, 2) int64  – undirected, **sorted** pairs

    # ---------------------------------------------------------------------
    # sanity checks
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.nodes.shape[0] != self.radii.shape[0]:
            raise ValueError("nodes and radii length mismatch")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must be (E, 2)")

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _igraph(self) -> ig.Graph:  # on‑demand igraph view
        return ig.Graph(n=len(self.nodes), edges=[tuple(map(int, e)) for e in self.edges], directed=False)

    # ---------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------
    def to_swc(self, path: str | Path, *, include_header: bool = True, scale: float = 1.0) -> None:
        """Write to *path* in SWC format (id 1 = soma, parents 1‑indexed)."""
        path = Path(path)
        parent = _bfs_parents(self.edges, len(self.nodes), root=0)
        with path.open("w", encoding="utf8") as fh:
            if include_header:
                fh.write("# id type x y z radius parent\n")
            for idx, (p, r, pa) in enumerate(zip(self.nodes * scale, self.radii * scale, parent), start=1):
                swc_type = 1 if idx == 1 else 0
                fh.write(f"{idx} {swc_type} {p[0]} {p[1]} {p[2]} {r} {pa + 1 if pa != -1 else -1}\n")

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def check_connectivity(self, *, return_isolated: bool = False):
        g = self._igraph()
        order, _, _ = g.bfs(0, mode="ALL")  # returns -1 for unreachable
        reachable: Set[int] = {v for v in order if v != -1}
        if return_isolated:
            return [i for i in range(len(self.nodes)) if i not in reachable]
        return len(reachable) == len(self.nodes)

    def check_acyclic(self, *, return_cycle: bool = False) -> bool | list[tuple[int, int]]:
        """
        Verify that **no cycles** are present (i.e. the skeleton is a forest).

        Returns ``True`` when the edge count obeys |E| = |V| − C
        where *C* is the number of connected components.  When
        *return_cycle=True* one representative cycle is returned
        (empty list ⇒ acyclic).
        """
        g = self._igraph()                        # helper that builds igraph
        num_comp = len(g.components())            # instead of .nc
        acyclic  = g.ecount() == g.vcount() - num_comp
        if acyclic or not return_cycle:
            return acyclic

        # -- find a concrete cycle (slow but rare) -----------------------
        cyc = g.cycle_basis()[0]                  # list of vertex ids
        return [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]

# -----------------------------------------------------------------------------
#  Graph helpers (igraph)
# -----------------------------------------------------------------------------

def _surface_graph(mesh: trimesh.Trimesh) -> ig.Graph:
    """Triangle‑adjacency graph of *mesh* (edge weight = Euclidean length)."""
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
    """Return indices of soma‑surface vertices via density filter + dilation."""
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


def find_soma(
    mesh: trimesh.Trimesh,
    soma_probe_radius_mult: float = 8.0,
    soma_density_threshold: float = 0.30,
    soma_dilation_steps: int = 1,
    gsurf: ig.Graph | None = None,
):
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

# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------

def _geodesic_bins(dist_dict: Dict[int, float], step: float) -> List[List[int]]:
    """
    Vectorised distance-to-bins helper.
    Keeps the original semantics: bins are half-open [a, b)
    so a vertex sitting exactly on the global max distance
    joins the *previous* shell.
    """
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
    """Return parent[] array of BFS tree from *root* given undirected edge list."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    parent = [-1] * n_nodes
    # q: List[int] = [root]
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v != root and parent[v] == -1:
                parent[v] = u
                q.append(v)
    return parent


def _edges_from_mesh(
    edges_unique: np.ndarray,        # (E, 2) int64
    v2n: dict[int, int],             # mesh-vertex id -> skeleton node id
    n_mesh_verts: int,
) -> np.ndarray:
    """
    Vectorised remap of mesh edges -> skeleton edges.

    Returns
    -------
    edges_arr : (E′, 2) int64  sorted and unique.
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
    edges: np.ndarray,                      # shape (E, 2)
    *,
    c_soma: np.ndarray,
    r_soma: float,
    soma_merge_dist_factor: float,
    soma_merge_radius_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge close/fat centroids inside soma, then reconnect graph."""
    # ── 1. decide which vertices survive ────────────────────────────────
    close = np.linalg.norm(nodes - c_soma, axis=1) < soma_merge_dist_factor * r_soma
    fat   = radii >= soma_merge_radius_factor * r_soma
    keep  = ~(close | fat)
    keep[0] = True                                         # keep soma

    kept_idx = np.where(keep)[0]
    old2new = np.empty(len(keep), dtype=np.int64)
    old2new.fill(-1)
    old2new[kept_idx] = np.arange(len(kept_idx), dtype=np.int64)

    # ── 2. edges whose *both* endpoints survive (fast path) ────────────
    a, b = edges.T
    both_keep = keep[a] & keep[b]
    edges_out = edges[both_keep].copy()
    edges_out[:] = old2new[edges_out]                       # remap in-place

    # ── 3. reconnect neighbours of each removed vertex ─────────────────
    removed = np.where(~keep)[0]
    if removed.size:
        neigh: Dict[int, List[int]] = {int(r): [] for r in removed}
        for u, v in edges.tolist():                         # now real tuples
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

    return nodes[keep], radii[keep], edges_out


def _bridge_components(
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    bridge_k: int,
) -> np.ndarray:
    """
    Use KD-trees to connect all graph components so that everything is
    reachable from the soma.  Adds synthetic edges between *k* nearest
    pairs.
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
    """Return edge list of the global minimum-spanning tree."""
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
    Remove spuriously small neurites that attach *directly* to the soma.

    Parameters
    ----------
    nodes
        (N, 3) coordinates array **after** collapse/bridging/MST.
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
#  Main algorithm (identical numerics, igraph ops)
# -----------------------------------------------------------------------------

def skeletonize(
    mesh: trimesh.Trimesh,
    # --- soma detection ---
    soma_probe_radius_mult: float = 10.0,
    soma_density_threshold: float = 0.30,
    soma_dilation_steps: int = 1,
    soma_seed_point: np.ndarray | None = None,
    soma_seed_radius: float = 7_500.0,
    path_len_relax: float = 1.15,
    # --- geodesic sampling ---
    target_shell_count: int = 500,
    min_cluster_vertices: int = 1,
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
) -> Skeleton:
    """Extract a centre‑line skeleton **and optionally bridge disconnected mesh patches**.

    Parameters
    ----------
    bridge_components
        If *True* (default) the algorithm will add synthetic edges between
        disconnected node groups so that every component is ultimately
        connected to the soma (node 0).
    bridge_k
        How many *candidate* node pairs (per foreign component) are tested
        when creating a bridge.  ``bridge_k=1`` means "connect via the single
        closest pair" (fast, good for small gaps).  Larger values can create a
        more tortuous but sometimes smoother bridge.
    """

    # 0. soma vertices ---------------------------------------------------
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
    v = mesh.vertices.view(np.ndarray)
    
    components    = gsurf.components()
    comp_vertices = [np.asarray(c, dtype=np.int64) for c in components]

    soma_vids = np.fromiter(soma_verts, dtype=np.int64)
    all_bins  : list[list[int]] = []
    edge_m    = float(mesh.edges_unique_length.mean())

    for cid, verts in enumerate(comp_vertices):

        # -------- choose the *one* seed you used before ----------
        if np.intersect1d(verts, soma_vids).size:
            seeds = [
                int(
                    soma_vids[
                        np.argmax(np.linalg.norm(v[soma_vids] - c_soma, axis=1))
                    ]
                )
            ]
        else:
            seeds = [int(verts[(hash(cid) % len(verts))])]

        # -------- single C call for this component ---------------
        dist_vec = gsurf.shortest_paths_dijkstra(
            source=seeds,
            target=verts,
            weights="weight",
        )[0]                                            # shape = (1, |verts|)

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
    nodes: List[np.ndarray] = [c_soma]
    radii: List[float] = [r_soma]
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
            radius   = float(np.linalg.norm(pts - centre, axis=1).mean())

            node_id = next_id
            next_id += 1
            nodes.append(centre)
            radii.append(radius)
            for vv in comp_idx:
                v2n[int(vv)] = node_id

    nodes_arr = np.asarray(nodes, dtype=np.float32)
    radii_arr = np.asarray(radii, dtype=np.float32)

    # 3. edges from mesh connectivity -----------------------------------
    edges_arr = _edges_from_mesh(
        mesh.edges_unique,
        v2n,
        n_mesh_verts=len(mesh.vertices),
    )

    # 4. collapse soma‑like / fat nodes ---------------------------
    if collapse_soma:
        (
            nodes_arr,
            radii_arr,
            edges_arr,
        ) = _collapse_soma_nodes(
            nodes_arr,
            radii_arr,
            edges_arr,
            c_soma=c_soma,
            r_soma=r_soma,
            soma_merge_dist_factor=soma_merge_dist_factor,
            soma_merge_radius_factor=soma_merge_radius_factor,
        )


    # 5. Connect all components ------------------------------
    if bridge_components:
        edges_arr = _bridge_components(nodes_arr, edges_arr, bridge_k=bridge_k)
    
    # 6. global minimum-spanning tree ------------------------------------
    edges_mst = _build_mst(nodes_arr, edges_arr)

    # 7. prune tiny sub-trees near the soma
    if prune_tiny_neurites:
        keep_mask, edges_mst = _prune_soma_neurites(
            nodes_arr,
            edges_mst,
            c_soma,
            r_soma,
            min_branch_nodes=min_branch_nodes,           
            min_branch_extent_factor=min_branch_extent_factor,
        )
        nodes_arr  = nodes_arr[keep_mask]
        radii_arr  = radii_arr[keep_mask]

    return Skeleton(nodes_arr, radii_arr, edges_mst)

# -----------------------------------------------------------------------------
#  SWC loader (unchanged)
# -----------------------------------------------------------------------------

def load_swc(
    path: str | Path,
    *,
    scale: float = 1.0,
    keep_types: Iterable[int] | None = None,
) -> Skeleton:
    """Read *path* and return a :class:`Skeleton`.  IDs are converted to 0‑based."""
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
    radii_arr = np.asarray(radii, dtype=np.float32) * scale
    edges = [
        (id_map[i], id_map[p])
        for i, p in zip(ids, parent, strict=True)
        if p != -1 and p in id_map
    ]
    return Skeleton(nodes_arr, radii_arr, np.asarray(edges, dtype=np.int64))
