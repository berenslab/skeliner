# -*- coding: utf-8 -*-
"""Skeliner skeletonization using **python‑igraph** instead of NetworkX.

This is a line‑for‑line port of the original implementation; the public API
(`Skeleton`, `find_soma`, `skeletonize`, `load_swc`) and all numerical outputs
remain identical.  Only the graph backend has changed – every operation now uses
`igraph`, yielding faster runtimes and lower memory usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import igraph as ig
import numpy as np
import trimesh
from scipy.spatial import cKDTree

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

    def check_acyclic(self, *, return_cycle: bool = False):
        g = self._igraph()
        # A forest with *c* components has |E| == |V| − c  ⇔ acyclic
        num_comp = g.components().nc
        acyclic = g.ecount() == g.vcount() - num_comp
        if not return_cycle:
            return acyclic
        if acyclic:
            return []
        return _find_cycle_edges(g)

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
    *,
    soma_probe_radius_mult: float = 8.0,
    soma_density_threshold: float = 0.5,
    soma_dilation_steps: int = 1,
) -> Set[int]:
    """Return indices of soma‑surface vertices via density filter + dilation."""
    v = mesh.vertices.view(np.ndarray)
    gsurf = _surface_graph(mesh)

    edge_m = mesh.edges_unique_length.mean()
    probe_r = soma_probe_radius_mult * edge_m

    counts = np.asarray(cKDTree(v).query_ball_point(v, probe_r, return_length=True))

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
    *,
    soma_probe_radius_mult: float = 8.0,
    soma_density_threshold: float = 0.30,
    soma_dilation_steps: int = 1,
):
    soma_verts = _soma_surface_vertices(
        mesh,
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
    *,
    seed_r: float,
    lam: float = 1.15,
):
    v = mesh.vertices.view(np.ndarray)
    seed_vid = int(np.argmin(np.linalg.norm(v - seed, axis=1)))
    cutoff = seed_r * lam
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
    if not dist_dict:
        return []
    edges = np.arange(0.0, max(dist_dict.values()) + step, step)
    return [[vid for vid, d in dist_dict.items() if a <= d < b] for a, b in zip(edges[:-1], edges[1:])]


def bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> List[int]:
    """Return parent[] array of BFS tree from *root* given undirected edge list."""
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    parent = [-1] * n_nodes
    q: List[int] = [root]
    while q:
        u = q.pop(0)
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
    bridge_k: int = 3,
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
    if soma_seed_point is None:
        c_soma, r_soma, soma_verts = find_soma(
            mesh,
            soma_probe_radius_mult=soma_probe_radius_mult,
            soma_density_threshold=soma_density_threshold,
            soma_dilation_steps=soma_dilation_steps,
        )
    else:
        c_soma, r_soma, soma_verts = find_soma_with_seed(
            mesh,
            seed=soma_seed_point,
            seed_r=soma_seed_radius,
            lam=path_len_relax,
        )

    # 1. binning along geodesic shells ----------------------------------
    v = mesh.vertices.view(np.ndarray)
    gsurf = _surface_graph(mesh)

    soma_vids = np.fromiter(soma_verts, dtype=np.int64)
    seed_vid = soma_vids[np.argmax(np.linalg.norm(v[soma_vids] - c_soma, axis=1))]
    dist_vec = gsurf.shortest_paths_dijkstra(source=seed_vid, weights="weight")[0]
    dist_all = {vid: d for vid, d in enumerate(dist_vec) if np.isfinite(d)}

    arc_len = max(dist_all.values(), default=0.0)
    edge_m = mesh.edges_unique_length.mean()
    step = max(edge_m * 2.0, arc_len / target_shell_count)

    bins: List[List[int]] = []
    while not any(bins) and step < edge_m * max_shell_width_factor:
        bins = _geodesic_bins(dist_all, step)
        step *= 1.5

    # 2. create skeleton nodes ------------------------------------------
    nodes: List[np.ndarray] = [c_soma]
    radii: List[float] = [r_soma]
    v2n: Dict[int, int] = {v: 0 for v in soma_verts}
    next_id = 1

    for verts in bins:
        inner = [vid for vid in verts if vid not in soma_verts]
        if len(inner) < min_cluster_vertices:
            continue
        sub = gsurf.induced_subgraph(inner)
        for comp in sub.components():
            if len(comp) < min_cluster_vertices:
                continue
            pts = v[np.fromiter((inner[i] for i in comp), dtype=np.int64)]
            centre = pts.mean(axis=0)
            radius = float(np.mean(np.linalg.norm(pts - centre, axis=1)))
            node_id = next_id
            next_id += 1
            nodes.append(centre)
            radii.append(radius)
            for local_idx in comp:
                v2n[inner[local_idx]] = node_id

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
        keep = np.ones(len(nodes_arr), bool)
        for i in range(1, len(nodes_arr)):
            close = np.linalg.norm(nodes_arr[i] - c_soma) < soma_merge_dist_factor * r_soma
            fat = radii_arr[i] >= soma_merge_radius_factor * r_soma
            tiny = radii_arr[i] < tiny_radius_threshold
            if close or fat or tiny:
                keep[i] = False
        old2new = {old: new for new, old in enumerate(np.where(keep)[0])}
        nodes_arr = nodes_arr[keep]
        radii_arr = radii_arr[keep]
        remap = lambda a: old2new[int(a)] if keep[int(a)] else None  # noqa: E731
        edges_arr = np.array(
            [tuple(sorted((remap(a), remap(b)))) for a, b in edges_arr
             if remap(a) is not None and remap(b) is not None and remap(a) != remap(b)],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # 5. Connect all components (optional) ------------------------------
    # ------------------------------------------------------------------
    g_full = ig.Graph(n=len(nodes_arr), edges=[tuple(map(int, e)) for e in edges_arr], directed=False)
    g_full.es["weight"] = [float(np.linalg.norm(nodes_arr[a] - nodes_arr[b])) for a, b in edges_arr]

    if bridge_components and g_full.components().nc > 1:
        comp_membership = g_full.components().membership
        soma_comp = comp_membership[0]
        comps = {}
        for idx, comp_id in enumerate(comp_membership):
            comps.setdefault(comp_id, []).append(idx)

        # KD‑tree for the soma component to find nearest neighbors fast
        soma_kdt = cKDTree(nodes_arr[comps[soma_comp]])

        for comp_id, verts in comps.items():
            if comp_id == soma_comp:
                continue
            # find k closest pairs between this component and soma component
            dists, idxs = soma_kdt.query(nodes_arr[verts], k=min(bridge_k, len(comps[soma_comp])))
            dists = np.atleast_2d(dists)
            idxs = np.atleast_2d(idxs)
            # add an edge for the single shortest pair
            flat_idx = np.argmin(dists)
            row, col = np.unravel_index(flat_idx, dists.shape)
            u = verts[row]
            v = comps[soma_comp][idxs[row, col]]
            w = float(np.linalg.norm(nodes_arr[u] - nodes_arr[v]))
            g_full.add_edge(u, v, weight=w)

    # ------------------------------------------------------------------
    # 6. Generate a global minimum‑spanning tree (MST) ------------------
    # ------------------------------------------------------------------
    mst = g_full.spanning_tree(weights="weight")
    tree_edges = np.asarray(mst.get_edgelist(), dtype=np.int64)

    # final sanity (ensure undirected, sorted pairs)
    tree_edges = np.sort(tree_edges, axis=1)

    return Skeleton(nodes_arr, radii_arr, tree_edges)

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
