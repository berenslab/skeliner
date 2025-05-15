
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import trimesh

__all__ = [
    "Skeleton",
    "find_seed",
    "find_soma",
    "skeletonize",
]

@dataclass(slots=True)
class Skeleton:
    """A minimal, numpy‑backed skeleton graph."""

    nodes: np.ndarray  # (N, 3) float32
    radii: np.ndarray  # (N,)  float32
    edges: np.ndarray  # (E, 2) int64

    # convenient accessors -------------------------------------------------
    def __post_init__(self):
        if self.nodes.shape[0] != self.radii.shape[0]:
            raise ValueError("nodes and radii length mismatch")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must be (E, 2)")

    # I/O ------------------------------------------------------------------
    def to_swc(self, path: str | Path, *, include_header: bool = True, scale: float = 1.) -> None:
        """Write the skeleton to *path* in SWC format."""
        path = Path(path)
        parent = bfs_parents(self.edges, len(self.nodes), root=0)
        with path.open("w", encoding="utf8") as f:
            if include_header:
                f.write("# id type x y z radius parent\n")
            for idx, (p, r, pa) in enumerate(zip(self.nodes * scale, self.radii * scale, parent), start=1):
                swc_type = 1 if idx == 1 else 0  # 1 = soma, 0 = undefined/segment
                f.write(
                    f"{idx} {swc_type} {p[0]} {p[1]} {p[2]} {r} "
                    f"{pa + 1 if pa != -1 else -1}\n"
                )

    def check_connectivity(self, *, return_isolated: bool = False) -> bool | list[int]:
        """
        Verify that **all** nodes are reachable from the soma (node 0).

        Parameters
        ----------
        return_isolated
            If *False* (default) return a single bool.
            If *True* return a *list* with the indices of isolated nodes
            (empty list ⇒ fully connected).

        Returns
        -------
        bool | List[int]
            • When *return_isolated=False*: ``True`` if connected, ``False`` otherwise.  
            • When *return_isolated=True*: list of isolated node indices.
        """
        g = nx.Graph()
        g.add_nodes_from(range(len(self.nodes)))
        g.add_edges_from(self.edges)

        reachable = nx.descendants(g, 0) | {0}
        if return_isolated:
            isolated = [i for i in range(len(self.nodes)) if i not in reachable]
            return isolated
        return len(reachable) == len(self.nodes)

    def check_acyclic(self, *, return_cycle: bool = False) -> bool | list[tuple[int, int]]:
        """
        Verify that **no cycles** are present (i.e. the skeleton is a tree / forest).

        Parameters
        ----------
        return_cycle
            If *False* (default) return a single bool.  
            If *True* return one representative list of edges that form
            a cycle (empty list ⇒ no cycles).

        Returns
        -------
        bool | list[tuple[int, int]]
            • When *return_cycle=False*: ``True`` if acyclic, ``False`` otherwise.  
            • When *return_cycle=True*: a list of edge-tuples forming a
              cycle, or ``[]`` when none exist.
        """
        g = nx.Graph()
        g.add_nodes_from(range(len(self.nodes)))
        g.add_edges_from(self.edges)

        try:
            cyc = nx.find_cycle(g)  # raises NetworkXNoCycle if none
        except nx.exception.NetworkXNoCycle:
            return [] if return_cycle else True

        return cyc if return_cycle else False

# --------------------------
#  public helpers
# --------------------------

def find_seed(mesh: "trimesh.Trimesh", seed: np.ndarray, /, *, seed_r: float, lam: float = 1.15) -> tuple[int, float]:
    """Return index of the vertex closest to *seed* and its radius guess.

    The *seed* is a 3‑vector (same units as *mesh* vertices).
    """
    v = mesh.vertices.view(np.ndarray)
    seed_vid = int(np.argmin(np.linalg.norm(v - seed, axis=1)))
    return seed_vid, seed_r * lam


def find_soma(mesh: "trimesh.Trimesh", seed: np.ndarray, *, seed_r: float, lam: float = 1.15) -> tuple[np.ndarray, float, set[int]]:
    """Identify soma‑surface vertices around *seed*.

    Returns the centroid, an estimated radius and the vertex set.
    """
    seed_vid, cutoff = find_seed(mesh, seed, seed_r=seed_r, lam=lam)
    v = mesh.vertices.view(np.ndarray)
    g = _surface_graph(mesh)
    dist = nx.single_source_dijkstra_path_length(g, seed_vid, weight="weight", cutoff=cutoff)
    soma_verts = set(dist)
    pts = v[list(soma_verts)]
    centre = pts.mean(axis=0)
    radius = np.percentile(np.linalg.norm(pts - centre, axis=1), 90)
    return centre, radius, soma_verts

# --------------------------
#  main algorithm
# --------------------------

def skeletonize(
    mesh: "trimesh.Trimesh",
    seed: np.ndarray,
    *,
    seed_r: float = 7_500.0,
    lam: float = 1.15,
    target_bins: int = 500,
    min_bin_sz: int = 1,
    max_bin_fac: int = 50,
    collapse_dist: float = 1.25,
    collapse_rad: float = 0.35,
    radius_eps: float = 1.0,
) -> Skeleton:
    """Extract a centre‑line skeleton from *mesh* starting at *seed*.

    Parameters mirror the original prototype script.
    """
    # 0. soma vertices ----------------------------------------------------
    c_soma, r_soma, soma_verts = find_soma(
        mesh, seed, seed_r=seed_r, lam=lam
    )

    # 1. binning along geodesic shells ----------------------------------
    v = mesh.vertices.view(np.ndarray)
    gsurf = _surface_graph(mesh)
    seed_vid = int(np.argmin(np.linalg.norm(v - seed, axis=1)))
    dist_all = nx.single_source_dijkstra_path_length(gsurf, seed_vid, weight="weight")

    arc_len = max(dist_all.values(), default=0.0)
    edge_m = mesh.edges_unique_length.mean()
    step = max(edge_m * 2.0, arc_len / target_bins)

    bins: list[list[int]] = []
    while not any(bins) and step < edge_m * max_bin_fac:
        bins = _geodesic_bins(dist_all, step)
        step *= 1.5

    nodes: list[np.ndarray] = [c_soma]
    radii: list[float] = [r_soma]
    v2n: dict[int, int] = {v: 0 for v in soma_verts}
    next_id = 1

    for verts in bins:
        inner = [v_ for v_ in verts if v_ not in soma_verts]
        if len(inner) < min_bin_sz:
            continue
        sub = gsurf.subgraph(inner)
        for comp in nx.connected_components(sub):
            comp = list(comp)
            if len(comp) < min_bin_sz:
                continue
            pts = v[comp]
            centre = pts.mean(axis=0)
            radius = np.mean(np.linalg.norm(pts - centre, axis=1))
            node_id = next_id
            next_id += 1
            nodes.append(centre)
            radii.append(radius)
            for vv in comp:
                v2n[vv] = node_id

    nodes_arr = np.asarray(nodes, dtype=np.float32)
    radii_arr = np.asarray(radii, dtype=np.float32)

    # 2. edges from mesh connectivity -------------------------------------
    edges: set[tuple[int, int]] = {
        tuple(sorted((v2n[int(u)], v2n[int(v)])))
        for u, v in mesh.edges_unique
        if v2n.get(int(u)) is not None and v2n.get(int(v)) is not None and v2n[int(u)] != v2n[int(v)]
    }

    # 3. collapse soma‑like / drop tiny nodes -----------------------------
    if collapse_dist > 0.0:
        keep = np.ones(len(nodes_arr), bool)
        for i in range(1, len(nodes_arr)):
            close = np.linalg.norm(nodes_arr[i] - c_soma) < collapse_dist * r_soma
            fat = radii_arr[i] >= collapse_rad * r_soma
            tiny = radii_arr[i] < radius_eps
            if close or fat or tiny:
                keep[i] = False
        old2new = {old: new for new, old in enumerate(np.where(keep)[0])}
        nodes_arr = nodes_arr[keep]
        radii_arr = radii_arr[keep]

        # reconnect neighbours of removed nodes ------------------------------
        adj: dict[int, list[int]] = {i: [] for i in range(len(keep))}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        new_edges: set[tuple[int, int]] = set()
        for a, b in edges:
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

        edges_arr = np.array(sorted(new_edges), dtype=np.int64)
    else:
        edges_arr = np.array(sorted(edges), dtype=np.int64)

    # 3b. keep only the component that contains the soma -----------------
    g = nx.Graph()
    g.add_nodes_from(range(len(nodes_arr)))
    g.add_edges_from(edges_arr)

    main_cc = max(nx.connected_components(g), key=len)
    keep_cc = np.array([i in main_cc for i in range(len(nodes_arr))])
    old2new_cc = {old: new for new, old in enumerate(np.where(keep_cc)[0])}

    nodes_arr = nodes_arr[keep_cc]
    radii_arr = radii_arr[keep_cc]

    cc_edges = {
        (old2new_cc[a], old2new_cc[b])
        for a, b in edges_arr
        if keep_cc[a] and keep_cc[b]
    }
    edges_arr = np.array(sorted(cc_edges), dtype=np.int64)

    # remove cycles → keep only the BFS tree from soma
    # -----------------------------------------------------------------
    parent = bfs_parents(edges_arr, len(nodes_arr), root=0)
    tree_edges = np.array(
        [(min(i, pa), max(i, pa))             # undirected, sorted
         for i, pa in enumerate(parent) if pa != -1],
        dtype=np.int64,
    )

    return Skeleton(nodes_arr, radii_arr, tree_edges)

# --------------------------
#  internal helpers
# --------------------------

def _surface_graph(m: "trimesh.Trimesh") -> "nx.Graph":
    g = nx.Graph()
    g.add_nodes_from(range(len(m.vertices)))
    for (u, v), w in zip(m.edges_unique, m.edges_unique_length):
        g.add_edge(int(u), int(v), weight=float(w))
    return g


def _geodesic_bins(dist_dict: dict[int, float], step: float) -> list[list[int]]:
    if not dist_dict:
        return []
    edges = np.arange(0.0, max(dist_dict.values()) + step, step)
    return [
        [vid for vid, d in dist_dict.items() if a <= d < b]
        for a, b in zip(edges[:-1], edges[1:])
    ]


def bfs_parents(edges: np.ndarray, n_nodes: int, *, root: int = 0) -> list[int]:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    parent = [-1] * n_nodes
    for child in nx.bfs_tree(g, source=root):
        for nb in g.neighbors(child):
            if parent[nb] == -1 and nb != root:
                parent[nb] = child
    return parent

def load_swc(
    path: str | Path,
    *,
    scale: float = 1.0,
    keep_types: Iterable[int] | None = None,
) -> Skeleton:
    """
    Read *path* and return a :class:`Skeleton`.

    Parameters
    ----------
    scale
        Divide the coordinates and radii **after** reading.
        If you saved with ``to_swc(..., scale=1e-3)`` you’ll usually call
        ``load_swc(..., scale=1e-3)`` to get back to the original units.
    keep_types
        Optional whitelist of SWC *type* codes to keep.  By default all nodes
        are imported.

    Notes
    -----
    *Columns*: ``id  type  x  y  z  radius  parent``  
    IDs in the file are 1-based; we convert to zero-based indices.
    """
    path = Path(path)
    ids, xyz, radii, parent = [], [], [], []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 7:
                continue  # malformed row
            _id, _type = int(fields[0]), int(fields[1])
            if keep_types is not None and _type not in keep_types:
                continue
            ids.append(_id)
            xyz.append([float(fields[2]), float(fields[3]), float(fields[4])])
            radii.append(float(fields[5]))
            parent.append(int(fields[6]))

    if not ids:
        raise ValueError(f"No nodes found in {path}")

    # ensure ids are consecutive (SWC files usually are)
    id_map = {old_id: new_id for new_id, old_id in enumerate(ids)}
    nodes     = (np.asarray(xyz, dtype=np.float32) * scale)
    radii_arr = (np.asarray(radii, dtype=np.float32) * scale)

    edges = [
        (id_map[i], id_map[p])
        for i, p in zip(ids, parent, strict=True)
        if p != -1 and p in id_map
    ]

    return Skeleton(nodes, radii_arr, np.asarray(edges, dtype=np.int64))
