from collections import deque
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
        parent = bfs_parents(self.edges, len(self.nodes), root=0)
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


def _find_cycle_edges(g: ig.Graph) -> List[Tuple[int, int]]:
    """Return one representative cycle as a list of edges (u, v)."""
    parent = [-1] * g.vcount()
    stack: List[int] = [0]
    seen = {0}
    while stack:
        u = stack.pop()
        for v in g.neighbors(u):
            if v == parent[u]:
                continue
            if v in seen:  # back‑edge → reconstruct
                cyc: List[Tuple[int, int]] = []
                x, y = u, v
                while x != y:
                    if x > y:
                        cyc.append((min(x, parent[x]), max(x, parent[x])))
                        x = parent[x]
                    else:
                        cyc.append((min(y, parent[y]), max(y, parent[y])))
                        y = parent[y]
                cyc.append((min(u, v), max(u, v)))
                return cyc
            seen.add(v)
            parent[v] = u
            stack.append(v)
    return []  # should not reach here if a cycle exists


# -----------------------------------------------------------------------------
#  Soma detection helpers (unchanged numerics, igraph graph ops)
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
    soma_comp = max(sub.components(), key=len)
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

def bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> List[int]:
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
    # --- post‑processing ---
    soma_merge_dist_factor: float = 1.0,
    soma_merge_radius_factor: float = 0.25,
    tiny_radius_threshold: float = 0.5,
    # --- bridging disconnected patches ---
    bridge_components: bool = True,
    bridge_k: int = 1,
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
    
    # # split the surface graph into connected components once
    # components = gsurf.components()

    # # We will collect a global list of bins across all components
    # all_bins: List[List[int]] = []
    # soma_vids = np.fromiter(soma_verts, dtype=np.int64)
    # for comp_id, comp_vertices in enumerate(components):
    #     comp_vertices = np.asarray(comp_vertices, dtype=np.int64)

    #     # pick seed: soma furthest-out for soma-component, otherwise a stable pseudo-random one
    #     if np.intersect1d(comp_vertices, soma_vids).size:
    #         # this is the soma component
    #         seed = int(
    #             soma_vids[np.argmax(np.linalg.norm(v[soma_vids] - c_soma, axis=1))]
    #         )
    #     else:
    #         # deterministic but pseudo-random pick for reproducibility
    #         seed = int(comp_vertices[(hash(comp_id) % len(comp_vertices))])

    #     # geodesic distances inside *this* component
    #     dist_vec = gsurf.shortest_paths_dijkstra(
    #         source=seed, target=comp_vertices, weights="weight"
    #     )[0]
    #     dist_sub = {int(vid): float(d) for vid, d in zip(comp_vertices, dist_vec)}

    #     # adaptive shell width as before
    #     if dist_sub:
    #         edge_m  = float(mesh.edges_unique_length.mean())
    #         arc_len = max(dist_sub.values())
    #         step    = max(edge_m * 2.0, arc_len / target_shell_count)

    #         bins: List[List[int]] = []
    #         while not any(bins) and step < edge_m * max_shell_width_factor:
    #             bins = _geodesic_bins(dist_sub, step)
    #             step *= 1.5

    #         all_bins.extend(bins)
    # -----------------------------------------------------------
    # 1-b)  Geodesic distances –   C-batch inside each component
    # -----------------------------------------------------------
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
    edges: Set[Tuple[int, int]] = {
        tuple(sorted((v2n[int(u)], v2n[int(v)])))
        for u, v in mesh.edges_unique
        if u in v2n and v in v2n and v2n[int(u)] != v2n[int(v)]
    }
    edges_arr = np.asarray(sorted(edges), dtype=np.int64)

    # 4. collapse soma‑like / tiny / fat nodes ---------------------------
    if soma_merge_dist_factor > 0.0:

        dist  = np.linalg.norm(nodes_arr - c_soma, axis=1)
        close = dist  < soma_merge_dist_factor * r_soma
        fat   = radii_arr >= soma_merge_radius_factor * r_soma
        tiny  = radii_arr <  tiny_radius_threshold

        keep        = ~(close | fat | tiny)
        keep[0]     = True      # never drop the soma

        old2new = {old: new for new, old in enumerate(np.where(keep)[0])}
        nodes_arr  = nodes_arr[keep]
        radii_arr  = radii_arr[keep]

        # ----------- RECONNECT neighbours of every removed node -----------
        adj: Dict[int, List[int]] = {i: [] for i in range(len(keep))}
        for a, b in edges_arr:               # ← *original* edge list
            adj[a].append(b)
            adj[b].append(a)

        new_edges: Set[Tuple[int, int]] = set()
        for a, b in edges_arr:
            ka, kb = keep[a], keep[b]
            if ka and kb:
                new_edges.add(tuple(sorted((old2new[a], old2new[b]))))
            elif ka and not kb:
                for n in adj[b]:
                    if keep[n] and n != a:
                        new_edges.add(tuple(sorted((old2new[a], old2new[n]))))
            elif kb and not ka:
                for n in adj[a]:
                    if keep[n] and n != b:
                        new_edges.add(tuple(sorted((old2new[b], old2new[n]))))

        edges_arr = np.asarray(sorted(new_edges), dtype=np.int64)

    # --------------------------------------------------------
    # 5. Connect all components ------------------------------
    # --------------------------------------------------------

    g_full = ig.Graph(
        n=len(nodes_arr),
        edges=[tuple(map(int, e)) for e in edges_arr],
        directed=False,
    )
    g_full.es["weight"] = [
        float(np.linalg.norm(nodes_arr[a] - nodes_arr[b])) for a, b in edges_arr
    ]

    if bridge_components:
        # 5-a)  Work out the connected components
        vc           = g_full.components()
        soma_comp_id = vc.membership[0]           # node 0 is the soma
        comps        = [set(c) for c in vc]       # list[set[int]]

        if len(comps) > 1:                        # already fully connected?  → skip
            # 5-b)  Build one KD-tree **per component** so we can find
            #       nearest neighbours between any two components quickly.            
            kdtrees = {
                cid: _KDTree(nodes_arr[list(vids)])
                for cid, vids in enumerate(comps)
            }

            # 5-c)  Greedily graft every *foreign* component onto the
            #       growing “main island” that already reaches the soma.
            main_island = set(comps[soma_comp_id])          # vertices already wired
            island_tree = kdtrees[soma_comp_id]             # KD-tree of it

            pending = [cid for cid in range(len(comps))
                    if cid != soma_comp_id]

            while pending:
                cid   = pending.pop(0)
                vids  = comps[cid]
                pts   = nodes_arr[list(vids)]

                # nearest neighbours *from that component* to the current island
                dists, island_idx = island_tree.query(pts, k=1, workers=-1)
                order = np.argsort(dists)
                u_candidates = np.fromiter(vids, dtype=np.int64)
                island_list  = np.fromiter(main_island, dtype=np.int64)

                for j in order[:bridge_k]:
                    u = int(u_candidates[j])
                    v = int(island_list[island_idx[j]])
                    w = float(np.linalg.norm(nodes_arr[u] - nodes_arr[v]))
                    g_full.add_edge(u, v, weight=w)
                    edges_arr = np.vstack([edges_arr, [u, v]])

                # fuse that component into the island and rebuild the tree
                main_island |= vids
                island_tree  = _KDTree(nodes_arr[list(main_island)])

    # 6. global minimum-spanning tree ------------------------------------
    mst      = g_full.spanning_tree(weights="weight")
    edges_mst = np.asarray(
        sorted(tuple(sorted(e)) for e in mst.get_edgelist()), dtype=np.int64
    )

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
