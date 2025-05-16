
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import cKDTree

__all__ = [
    "Skeleton",
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


def _soma_surface_vertices(mesh: trimesh.Trimesh,
                           soma_probe_radius_mult: float = 8.0,
                           soma_density_threshold: float = 0.5,
                           soma_dilation_steps: int = 1) -> set[int]:
    """
    Return the *indices* of mesh vertices that belong to the soma surface.

    Parameters
    ----------
    soma_probe_radius_mult     ∝ how large a ball is used for the neighbour count
    soma_density_threshold   keep all vertices whose local neighbour count is
                   at least `soma_density_threshold * max_count`
    soma_dilation_steps     how many geodesic steps to dilate the patch
    """
    v     = mesh.vertices.view(np.ndarray)
    gsurf = _surface_graph(mesh)     

    edge_m  = mesh.edges_unique_length.mean()
    probe_r = soma_probe_radius_mult * edge_m

    kdt     = cKDTree(v)
    counts  = np.asarray(kdt.query_ball_point(v, probe_r,
                                              return_length=True))

    # ---- 1a  keep everything above a relative threshold ----------------
    dense_mask = counts >= counts.max() * soma_density_threshold
    dense_vids = np.where(dense_mask)[0]

    # ---- 1b  pick the largest connected component ----------------------
    sub      = gsurf.subgraph(dense_vids)
    soma_cc  = max(nx.connected_components(sub), key=len)
    soma_set = set(soma_cc)

    # ---- 1c  (optional) geodesic dilation to fill tiny gaps ------------
    for _ in range(soma_dilation_steps):
        boundary = {nb
                    for v_ in soma_set
                    for nb in gsurf.neighbors(v_)
                    if nb not in soma_set}
        soma_set.update(boundary)

    return soma_set

def find_soma(mesh: trimesh.Trimesh,
                   soma_probe_radius_mult: float = 8.0,
                   soma_density_threshold: float = 0.30,
                   soma_dilation_steps: int = 1,
) -> tuple[np.ndarray, float, set[int]]:

    soma_verts = _soma_surface_vertices(mesh,
                                        soma_probe_radius_mult=soma_probe_radius_mult,
                                        soma_density_threshold=soma_density_threshold,
                                        soma_dilation_steps=soma_dilation_steps)

    v       = mesh.vertices.view(np.ndarray)
    pts     = v[list(soma_verts)]

    centre  = pts.mean(axis=0)
    radius  = np.percentile(np.linalg.norm(pts - centre, axis=1), 90)

    return centre, radius, soma_verts


def find_soma_with_seed(mesh: "trimesh.Trimesh", seed: np.ndarray, *, seed_r: float, lam: float = 1.15) -> tuple[np.ndarray, float, set[int]]:
    """Identify soma‑surface vertices around *seed*.

    Returns the centroid, an estimated radius and the vertex set.
    """
    v = mesh.vertices.view(np.ndarray)
    seed_vid = int(np.argmin(np.linalg.norm(v - seed, axis=1)))
    cutoff = seed_r * lam
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
    # --- post-processing ---
    soma_merge_dist_factor: float = 1.,
    soma_merge_radius_factor: float = 0.25,
    tiny_radius_threshold: float = 0.5,
) -> Skeleton:
    """
    Extract a centre-line skeleton from a neuronal surface mesh.

    The algorithm

    1. identifies the soma patch by a local-density test (or uses a
       user-supplied seed);
    2. partitions the remaining surface into geodesic “shells”,
       collapsing each sufficiently large shell component into a graph
       node positioned at its centroid;
    3. connects nodes via the underlying triangle connectivity;
    4. prunes nodes that are soma-like, over-fat, or tiny; and
    5. removes any residual cycles to return a pure tree.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Triangle mesh of the neuron surface (world units).
    # --- soma detection -------------------------------------------------
    soma_probe_radius_mult : float, default 10.0
        Multiplier for the radius of the neighbour-count ball used to
        detect dense soma vertices.  Larger ⇒ a coarser density test and
        a bigger candidate patch.
    soma_density_threshold : float, default 0.30
        Keep vertices whose neighbour count is at least
        ``soma_density_threshold * max(counts)``.
    soma_dilation_steps : int, default 1
        Number of geodesic dilations to close tiny gaps in the initial
        soma patch.
    soma_seed_point : (3,) array_like, optional
        Pre-defined soma centre.  If *None* (default) the soma is
        detected automatically.
    soma_seed_radius : float, default 7500.0
        Radius of the seed sphere when *soma_seed_point* is given.
    path_len_relax : float, default 1.15
        Relaxation factor applied to geodesic path-length cut-offs when
        expanding from the seed.

    # --- geodesic sampling ---------------------------------------------
    target_shell_count : int, default 500
        Desired number of geodesic shells (bins) into which the surface
        is partitioned.
    min_cluster_vertices : int, default 1
        Smallest vertex cluster that may become a skeleton node.
    max_shell_width_factor : int, default 50
        Upper bound on adaptive shell width, expressed as a multiple of
        the mean edge length of *mesh*.

    # --- post-processing ------------------------------------------------
    soma_merge_dist_factor : float, default 1.0
        Collapse any node whose centre lies closer than
        ``soma_merge_dist_factor × soma_radius`` to the soma centre.
    soma_merge_radius_factor : float, default 0.25
        Collapse nodes whose radius exceeds
        ``soma_merge_radius_factor × soma_radius`` (catches fat initial
        dendrite stumps).
    tiny_radius_threshold : float, default 0.5
        Discard nodes with radius below this absolute threshold.

    Returns
    -------
    Skeleton
        Skeleton with radii.  Node 0 is the soma; the
        edge list forms an undirected tree.

    Examples
    --------
    >>> skel = skeletonize(mesh,             # automatic soma detection
    ...                    soma_probe_radius_mult=10.0,
    ...                    soma_density_threshold=0.30,
    ...                    soma_dilation_steps=1,
    ...)           
    >>> skel.to_swc("neuron.swc")          # write in SWC format

    >>> # With a known soma centre (in mesh units):
    >>> skel = skeletonize(mesh,
    ...                    soma_seed_point=[0, 0, 0],
    ...                    soma_seed_radius=5000)
    """
    # 0. soma vertices ----------------------------------------------------
    if soma_seed_point is None:
        c_soma, r_soma, soma_verts = find_soma(mesh, soma_probe_radius_mult=soma_probe_radius_mult,
                                            soma_density_threshold=soma_density_threshold, soma_dilation_steps=soma_dilation_steps)
    else:
        c_soma, r_soma, soma_verts = find_soma_with_seed(mesh, seed=soma_seed_point, seed_r=soma_seed_radius, lam=path_len_relax)

    # 1. binning along geodesic shells ----------------------------------
    v = mesh.vertices.view(np.ndarray)
    gsurf = _surface_graph(mesh)

    soma_vids = np.fromiter(soma_verts, dtype=np.int64)
    seed_vid   = soma_vids[np.argmax(np.linalg.norm(v[soma_vids] - c_soma, axis=1))]
    dist_all  = nx.single_source_dijkstra_path_length(
                    gsurf, seed_vid, weight="weight"
                )
    
    arc_len = max(dist_all.values(), default=0.0)
    edge_m = mesh.edges_unique_length.mean()
    step = max(edge_m * 2.0, arc_len / target_shell_count)

    bins: list[list[int]] = []
    while not any(bins) and step < edge_m * max_shell_width_factor:
        bins = _geodesic_bins(dist_all, step)
        step *= 1.5

    nodes: list[np.ndarray] = [c_soma]
    radii: list[float] = [r_soma]
    v2n: dict[int, int] = {v: 0 for v in soma_verts}
    next_id = 1

    for verts in bins:
        inner = [v_ for v_ in verts if v_ not in soma_verts]
        if len(inner) < min_cluster_vertices:
            continue
        sub = gsurf.subgraph(inner)
        for comp in nx.connected_components(sub):
            comp = list(comp)
            if len(comp) < min_cluster_vertices:
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
        Scale factor for the coordinates and radii.
        If you set scale=1e-3 in `to_swc()`, you should set it to 
        scale=1e3 here.
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
