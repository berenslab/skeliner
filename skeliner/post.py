
from typing import Sequence

import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import cKDTree as _KDTree

from skeliner.core import Skeleton, _surface_graph, skeletonize


def coverage_fraction(
    mesh: trimesh.Trimesh,
    skel: Skeleton | Sequence[Skeleton],
    *,
    cover_mult: float = 1.5,
) -> float:
    """
    Return the fraction of mesh vertices already explained by *skel*.

    A vertex is “covered” when it lies within
    `cover_mult × node_radius` of **any** skeleton node.
    """
    if isinstance(skel, Skeleton):
        skels = [skel]
    else:
        skels = list(skel)

    verts = mesh.vertices.view(np.ndarray)
    kdt   = _KDTree(verts)
    covered = np.zeros(len(verts), bool)

    for s in skels:
        reach = s.radii * cover_mult
        for c, r in zip(s.nodes, reach, strict=True):
            covered[kdt.query_ball_point(c, r)] = True

    return covered.mean()        # ∈ [0, 1]


def _closest_pair(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """
    Return indices (ia, ib) of the closest points in *a* and *b* plus distance.
    """
    if _KDTree is not None:
        ta = _KDTree(a)
        d, ia = ta.query(b)          # distance from each b-point to nearest a
        ib = int(d.argmin())
        ia = int(ia[ib])
        return ia, ib, float(d[ib])

    # ── SciPy not available – brute force (still fine for few-hundred nodes)
    dmat = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ia, ib = np.unravel_index(dmat.argmin(), dmat.shape)
    return int(ia), int(ib), float(dmat[ia, ib])


def merge(
    skels: Sequence[Skeleton],
    *,
    max_link_distance: float | None = None,
) -> Skeleton:
    """
    Hierarchically merge disconnected skeleton fragments.

    *Iteratively* find the two components whose nearest nodes are closest
    together, connect them with one edge, and repeat until a single skeleton
    remains.  The largest skeleton is **not** privileged beyond being part of
    the distance search, so fragments can attach to each other first.

    Parameters
    ----------
    skels : Sequence[Skeleton]
        At least two disconnected skeletons.
    max_link_distance : float, optional
        If set, raise ``ValueError`` when **no** pair of components is closer
        than this threshold (prevents accidental mega-merges).

    Returns
    -------
    Skeleton
        The unified skeleton.
    """
    if len(skels) == 0:
        raise ValueError("merge() received an empty sequence")
    if len(skels) == 1:
        return skels[0]

    # ── find the largest skeleton and use its soma as reference
    main_idx = max(range(len(skels)), key=lambda i: len(skels[i].nodes))
    soma_ref = skels[main_idx].nodes[0].copy()

    # start with *copies* so we never mutate callers’ objects
    components: list[Skeleton] = [
        Skeleton(s.nodes.copy(), s.radii.copy(), s.edges.copy()) for s in skels
    ]

    while len(components) > 1:
        # ── search all unordered pairs for the minimum inter-component distance
        best = None  # (dist, ia, ib, pa, pb)
        for ia, A in enumerate(components[:-1]):
            for ib, B in enumerate(components[ia + 1 :], start=ia + 1):
                pa, pb, dist = _closest_pair(A.nodes, B.nodes)
                if best is None or dist < best[0]:
                    best = (dist, ia, ib, pa, pb)

        assert best is not None  # len>1 ⇒ we found at least one pair
        dist, ia, ib, pa, pb = best

        if max_link_distance is not None and dist > max_link_distance:
            raise ValueError(
                "No component pairs lie within max_link_distance "
                f"({dist:.1f} > {max_link_distance})."
            )

        # ── merge components[ib] INTO components[ia]  (ia < ib)
        A = components[ia]
        B = components[ib]
        offset = len(A.nodes)

        merged_nodes  = np.vstack([A.nodes,  B.nodes]).astype(np.float32, copy=False)
        merged_radii  = np.concatenate([A.radii, B.radii]).astype(np.float32, copy=False)
        merged_edges  = np.vstack(
            [
                A.edges,
                B.edges + offset,
                np.array([[pa, pb + offset]], dtype=np.int64),  # the graft edge
            ]
        ).astype(np.int64, copy=False)

        components[ia] = Skeleton(merged_nodes, merged_radii, merged_edges)
        components.pop(ib)           # remove the consumed component

    final = components[0]
    soma_idx = int(
        np.linalg.norm(final.nodes - soma_ref[None, :], axis=1).argmin()
    )

    if soma_idx != 0:
        order = np.arange(len(final.nodes))
        order[0], order[soma_idx] = order[soma_idx], order[0]

        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        nodes  = final.nodes[order]
        radii  = final.radii[order]
        edges  = inv[final.edges]          # remap every endpoint

        final = Skeleton(nodes, radii, edges)

    return final


# -------------------------------------------------------
#  high-level convenience: trace *all* disconnected parts
# -------------------------------------------------------

def find_path_seeds(
    mesh: trimesh.Trimesh,
    traced: list[Skeleton] | None = None,
    cover_mult: float = 1.5,
    min_component_size: int = 1000,
) -> list[np.ndarray]:
    """
    Return one seed point for every mesh component that is **not**
    already covered by *traced* skeletons.

    Parameters
    ----------
    mesh
        The full neuronal mesh.
    traced
        Skeletons that have been extracted so far (may be an empty list).
    cover_mult
        A vertex is considered “covered” when it lies inside
        `cover_mult * node_radius` of **any** traced skeleton node.
    min_component_size
        Discard very small leftover islands (noise / debris).

    Notes
    -----
    • Coverage is decided purely geometrically with a `cKDTree`,  
      so we do not need to modify `skeletonize` at all.  
    • If *traced* is ``None`` or empty, the routine simply returns
      the centres of *all* connected components except the biggest
      one (which almost always contains the soma).
    """
    verts = mesh.vertices.view(np.ndarray)
    n = len(verts)
    covered = np.zeros(n, bool)

    if traced:
        kdt = _KDTree(verts)
        for sk in traced:
            # grow every node into a sphere and mark the vertices inside
            reach = sk.radii * cover_mult
            for c, r in zip(sk.nodes, reach, strict=True):
                covered[kdt.query_ball_point(c, r)] = True

    # ------------------------------------------------------------------
    # find surface-graph CCs in the *uncovered* part of the mesh
    # ------------------------------------------------------------------
    uncv = np.where(~covered)[0]
    if len(uncv) == 0:                # nothing left to do
        return []

    gsurf = _surface_graph(mesh)
    sub   = gsurf.subgraph(uncv)

    # (optional) keep the soma component → drop it from the seed list
    if not traced:                    # first call, no skeleton yet
        # keep the *largest* component as the “main” one
        main_cc = max(nx.connected_components(sub), key=len)
    else:
        main_cc = set()

    seeds: list[np.ndarray] = []
    for comp in nx.connected_components(sub):
        if comp is main_cc or len(comp) < min_component_size:
            continue

        comp_vids = np.fromiter(comp, dtype=np.int64)
        centre = verts[comp_vids].mean(axis=0)
        seeds.append(centre)

    return seeds


def complete(
    mesh: trimesh.Trimesh,
    skel0: Skeleton,
    cover_mult: float = 1.5,
    min_component_size: int = 800,
    seed_r: float = 200.0,
    lam: float = 1.0,
    collapse_dist: float = 0.0,
    verbose: bool = True,
) -> Skeleton:
    """
    Trace the soma *plus every disconnected component* of **mesh**.

    Parameters
    ----------
    mesh
        Full neuronal surface mesh.
    cover_mult, min_component_size
        Passed straight to :pyfunc:`find_path_seeds`.
    seed_r, lam, collapse_dist
        Defaults fed into :pyfunc:`sk.skeletonize` for the *extra* branches.
    verbose
        ``True`` → print a short progress log.

    Returns
    -------
    merged : Skeleton
        The fully merged skeleton.
    """

    parts: list[Skeleton] = [skel0]

    while True:
        seeds = find_path_seeds(
            mesh, parts,
            cover_mult=cover_mult,
            min_component_size=min_component_size,
        )
        if not seeds:
            break

        if verbose:
            print(f"Found {len(seeds)} extra component(s)")

        for i, seed in enumerate(seeds, start=1):
            if verbose:
                print(f"  ↳ tracing component {i}")

            try:
                skel = skeletonize(
                    mesh,
                    seed=seed,
                    seed_r=seed_r,
                    lam=lam,
                    collapse_dist=collapse_dist,
                )
            except ValueError:
                if verbose:
                    print(f"    ! failed on component {i}")
                continue

            parts.append(skel)

    merged = merge(parts)
    if verbose:
        print(f"  ↳ merged into {len(merged.nodes)} nodes")
    return merged
