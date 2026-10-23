"""skeliner.dx – graph‑theoretic diagnostics for a single Skeleton"""

import warnings
from typing import Any, Dict, List, Sequence, Set, Tuple

import igraph as ig
import numpy as np

from ._backend import kernels

__skeleton__ = [
    "check_connectivity",
    "connectivity",
    "check_acyclicity",
    "acyclicity",
    "cycles",
    "check_bins",
    "check_mesh_pairing",
    "edge_support",
    "face_owner",
    "bin_faces",
    "degree",
    "neighbors",
    "nodes_of_degree",
    "branches_of_length",
    "twigs_of_length",
    "suspicious_tips",
    "suspicious_junctions",
    "distance",
    "node_summary",
    "extract_neurites",
    "neurite_names",
    "neurite_nodes",
    "neurites_out_of_bounds",
    "volume",
    "total_path_length",
]

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _graph(skel) -> ig.Graph:
    """Return the undirected *igraph* view of the skeleton."""
    return skel._igraph()


# -----------------------------------------------------------------------------
# 1. connectivity & cycles
# -----------------------------------------------------------------------------


def check_connectivity(skel, *, return_isolated: bool = False):
    """Verify that **every** node is reachable from the soma (vertex 0).

    Parameters
    ----------
    skel
        A :class:`skeliner.Skeleton` instance.
    return_isolated
        When *True* return a list of orphan node indices instead of a boolean.
    """
    g = _graph(skel)
    order, _, _ = g.bfs(0, mode="ALL")  # order[i] == -1 ⇔ unreachable
    reachable = {v for v in order if v != -1}
    if return_isolated:
        return [i for i in range(g.vcount()) if i not in reachable]
    return len(reachable) == g.vcount()


def connectivity(skel, *, return_isolated: bool = False):
    """Deprecated alias for :func:`check_connectivity`."""
    warnings.warn(
        "dx.connectivity() is deprecated; use dx.check_connectivity() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return check_connectivity(skel, return_isolated=return_isolated)


def check_acyclicity(skel, *, return_cycles: bool = False):
    """Check that the skeleton is a *forest* (|E| = |V| − components).

    If a cycle exists and ``return_cycles`` is *True*, a representative list
    of (u, v) edges forming the cycle is returned.
    """
    g = _graph(skel)
    n_comp = len(g.components())
    acyclic = g.ecount() == g.vcount() - n_comp
    if acyclic or not return_cycles:
        return acyclic
    return _cycle_edges(g, g.minimum_cycle_basis()[0])


def _cycle_edges(g: ig.Graph, eids) -> List[Tuple[int, int]]:
    """Edge ids from a cycle basis → ``(u, v)`` pairs, walked in loop order.

    ``minimum_cycle_basis`` and ``fundamental_cycles`` return **edge** ids,
    where the removed ``cycle_basis`` returned **vertex** ids.  Reading the
    new output as the old — indexing it as a vertex ring — silently yields a
    plausible list of pairs that are not the cycle, which is why this
    conversion is one named function rather than repeated at each call site.
    """
    pairs = [tuple(sorted(g.es[int(e)].tuple)) for e in eids]
    nbr: Dict[int, List[int]] = {}
    for a, b in pairs:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)

    start = min(nbr)
    ring, prev, cur = [start], None, start
    while True:
        nxt = next((w for w in nbr[cur] if w != prev), None)
        if nxt is None or nxt == start:
            break
        ring.append(nxt)
        prev, cur = cur, nxt
    return [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]


def cycles(skel, mesh=None) -> List[Dict[str, Any]]:
    """Every independent loop, and where the tree most likely closed it wrongly.

    A skeleton off the pipeline has no loops — the MST guarantees a tree — so
    a loop only exists because somebody *asserted* a connection
    (:func:`skeliner.post.graft` allows one by default, and restoring a
    surface-supported pair is the usual way).  That assertion is the useful
    part: the loop it opens is a statement that **two of these edges cannot
    both be right**, and one of them is a join the MST made that it should
    not have.

    Which one is not guessed.  With a *mesh*, :func:`edge_support` says which
    tree edges have no surface behind them at all (``T∖G``), and such an edge
    lying **on the loop** is the tree claiming a connection the surface does
    not support while an alternative path exists.  That is the break point,
    on evidence rather than on a threshold.

    Without a mesh the loop is still reported, with no break point named —
    which is the honest answer, not a fallback ranking.

    Returns
    -------
    list of dict
        One entry per independent cycle::

            {"nodes": [...],        # in loop order
             "edges": [(u, v), ...],
             "breaks": [(u, v), ...],   # unsupported by the surface
             "length": float}       # loop cable, skeleton units

        Longest loop first — a two-node loop is a duplicated edge, while a
        long one spans the structure the tree got wrong.
    """
    g = _graph(skel)
    basis = g.minimum_cycle_basis()
    if not basis:
        return []

    unsupported: Set[Tuple[int, int]] = set()
    if mesh is not None:
        rep = edge_support(skel, mesh)
        unsupported = {(int(u), int(v)) for u, v in rep["unsupported"]}

    nodes = np.asarray(skel.nodes, dtype=np.float64)
    out: List[Dict[str, Any]] = []
    for eids in basis:
        edges = _cycle_edges(g, eids)
        ring = [a for a, _ in edges]
        length = float(sum(np.linalg.norm(nodes[a] - nodes[b]) for a, b in edges))
        out.append(
            {
                "nodes": ring,
                "edges": edges,
                "breaks": [
                    (a, b) for a, b in edges if (min(a, b), max(a, b)) in unsupported
                ],
                "length": length,
            }
        )
    out.sort(key=lambda c: -c["length"])
    return out


def acyclicity(skel, *, return_cycles: bool = False):
    """Deprecated alias for :func:`check_acyclicity`."""
    warnings.warn(
        "dx.acyclicity() is deprecated; use dx.check_acyclicity() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return check_acyclicity(skel, return_cycles=return_cycles)


def check_bins(skel, *, mesh=None, return_report: bool = False):
    """Verify that ``node2verts`` and ``vert2node`` still agree.

    A node's position and every radius are computed from the mesh vertices it
    owns, so a wrong entry here is invisible: the skeleton looks fine, exports
    fine, and carries radii belonging to the wrong surface.  This turns that
    into a loud failure.

    What is checked
    ---------------
    * ``node2verts[1:]`` are **pairwise disjoint**.
    * ``vert2node`` is exactly the inverse of ``node2verts``.

    Node 0 is exempt from disjointness.  Its "bin" is ``soma.verts``, assigned
    wholesale by the soma stitch, while neurites are binned over the *face*
    based arbor — and under the ≥2-of-3 rule a face with one soma vertex is an
    arbor face and still contains that vertex.  Boundary vertices therefore
    belong to both by construction, and ``vert2node`` gives them to the arbor
    node.  Coverage is not checked either: soma, organelle, discarded and
    pruned surface are all legitimately unowned.

    Parameters
    ----------
    skel
        A :class:`skeliner.Skeleton` instance.
    mesh
        When given, additionally report which bins are split into more than one
        connected patch of surface.  This is **reported, never failed**: the
        binning's reunite pass is capped at 8 rounds and leaves a handful of
        fragmented bins on real cells.
    return_report
        Return a dict of details instead of a boolean.

    Returns
    -------
    bool or dict
    """
    n2v = skel.node2verts
    v2n = skel.vert2node

    if n2v is None or v2n is None:
        ok = n2v is None and v2n is None
        if not return_report:
            return ok
        return {
            "ok": ok,
            "reason": (
                "no mesh data"
                if ok
                else "one of node2verts / vert2node is None but not the other"
            ),
        }

    if len(n2v) != len(skel.nodes):
        report = {"ok": False, "reason": "node2verts length != node count"}
        return report if return_report else False

    bins = [np.asarray(v, dtype=np.int64).ravel() for v in n2v]
    arbor = bins[1:]

    owned = (
        np.concatenate([b for b in arbor if b.size])
        if any(b.size for b in arbor)
        else np.empty(0, dtype=np.int64)
    )
    uniq, counts = np.unique(owned, return_counts=True)
    duplicated = uniq[counts > 1]

    # `vert2node` must be what rebuild_vert2node() would produce: ascending
    # node order, so an arbor bin overwrites node 0 on the shared boundary.
    keys = np.fromiter(v2n.keys(), np.int64, len(v2n))
    vals = np.fromiter(v2n.values(), np.int64, len(v2n))
    size = 1 + int(
        max(owned.max(initial=-1), bins[0].max(initial=-1), keys.max(initial=-1))
    )

    expect = np.full(size, -1, dtype=np.int64)
    for nid, b in enumerate(bins):
        if b.size:
            expect[b] = nid
    actual = np.full(size, -1, dtype=np.int64)
    actual[keys] = vals
    mismatched = np.flatnonzero(expect != actual)

    ok = duplicated.size == 0 and mismatched.size == 0

    if not return_report:
        return bool(ok)

    soma_overlap = (
        np.intersect1d(bins[0], uniq, assume_unique=False).size if bins[0].size else 0
    )
    report: Dict[str, Any] = {
        "ok": bool(ok),
        "n_nodes": len(bins),
        "n_owned": int(uniq.size),
        "duplicated": duplicated,
        "mismatched": mismatched,
        "empty_bins": [i for i, b in enumerate(bins) if b.size == 0],
        "soma_overlap": int(soma_overlap),
    }
    if mesh is not None:
        report["fragmented"] = _fragmented_bins(bins, mesh)
    return report


def check_mesh_pairing(skel, mesh, *, return_report: bool = False):
    """Is *mesh* the surface ``skel`` was built from?

    A skeleton's bins name vertex *ids* and nothing else, so the ids alone
    cannot say which mesh they belong to.  Paired with a **larger** unrelated
    mesh every id stays in range, bins resolve to real faces, and every answer
    downstream — the surface a node owns, which pairs the surface joins, what
    a repartition would do — is computed against the wrong cell and returned
    without complaint.  A *smaller* one merely raises ``IndexError``, which is
    the safer failure of the two.

    Two independent things are asked, and they answer differently:

    ``in range``
        Every id ``node2verts`` names exists in the mesh.  A **necessary**
        condition, checkable on any skeleton ever written, and the one that
        rules out the smaller-mesh crash.
    ``counts``
        ``meta["mesh"]`` against the loaded mesh, when the skeleton carries
        it.  Exact, and it survives the ``.obj`` / ``.ply`` round trip that
        moves coordinates in the ninth decimal — which is why this is counts
        and not a hash of the vertex array.

    Skeletons written before ``meta["mesh"]`` existed carry no counts, so
    ``ok`` is the most that can be said of them and ``verified`` stays
    *False*.  The two are kept apart deliberately: *nothing contradicts this*
    is a weaker claim than *this is the right mesh*, and collapsing them would
    present a legacy skeleton as confirmed.

    Parameters
    ----------
    skel
        A :class:`skeliner.Skeleton` instance.
    mesh
        The mesh to test it against.
    return_report
        Return a dict of details instead of a boolean.

    Returns
    -------
    bool or dict
        ``ok`` is *False* only when something is positively contradicted.
        The dict also carries ``verified`` and a human-readable ``reason``.
    """
    n_verts = int(len(mesh.vertices))
    n_faces = int(len(mesh.faces))
    recorded = (skel.meta or {}).get("mesh") or {}

    def _out(ok, verified, reason, **extra):
        if not return_report:
            return bool(ok)
        return {
            "ok": bool(ok),
            "verified": bool(verified),
            "reason": reason,
            "meshVertices": n_verts,
            "meshFaces": n_faces,
            **extra,
        }

    n2v = skel.node2verts
    if n2v is None:
        return _out(False, False, "skeleton carries no mesh data")

    highest = -1
    for b in n2v:
        b = np.asarray(b, dtype=np.int64).ravel()
        if b.size:
            highest = max(highest, int(b.max()))
    if highest >= n_verts:
        return _out(
            False,
            False,
            f"the skeleton names vertex {highest:,}, but the mesh has only "
            f"{n_verts:,} vertices — it was not built from this mesh",
            highestVertex=highest,
        )

    want_v, want_f = recorded.get("n_vertices"), recorded.get("n_faces")
    if want_v is None:
        return _out(
            True,
            False,
            "the skeleton records no mesh counts, so the pairing cannot be "
            "confirmed — only that nothing contradicts it",
            highestVertex=highest,
        )
    if int(want_v) != n_verts or (want_f is not None and int(want_f) != n_faces):
        return _out(
            False,
            False,
            f"the skeleton was built from a mesh of {int(want_v):,} vertices / "
            f"{int(want_f):,} faces; this one has {n_verts:,} / {n_faces:,}",
            expectedVertices=int(want_v),
            expectedFaces=int(want_f) if want_f is not None else None,
        )
    return _out(True, True, "counts match the mesh this was built from")


def edge_support(skel, mesh) -> Dict[str, Any]:
    """Which node pairs the mesh surface joins, against which ones the tree has.

    ``skel.edges`` is a spanning tree *T* of the node-adjacency graph *G* the
    mesh implies.  *G* is not stored but is recomputable at any time from the
    mesh and ``vert2node``, and the two differences between them are what an
    edge edit needs to know:

    ``dropped`` (*G∖T*)
        Pairs whose bins share surface that the tree does not carry.  Adding
        one back is a **restore** — the surface really does join them —
        as opposed to a **graft**, which asserts a connection nothing
        supports.  That distinction is the whole reason this exists.
    ``unsupported`` (*T∖G*)
        Tree edges with no surface behind them at all: the soma stems from
        ``_stitch_to_soma`` and the synthetic bridges from ``bridge_gaps``.
        They are why a repartition must never simply re-span *G* — doing so
        would silently delete precisely the edges holding a broken arbor
        together.

    What ``dropped`` is **not** is a defect report.  The surface graph has
    cycles whether or not the arbor does: bins along one tube touch, and so do
    bins on branches that merely pass close.  Measured on 549190673, 147 of the
    156 dropped pairs at least three tree hops apart lie *within one branch*
    with the two bins overlapping in space — a dense axon tuft, not a fusion.
    A wrongly merged mesh and a tightly packed one look identical here, so
    nothing in this list may be presented as something to fix.
    """
    from .skeletonize import _edges_from_mesh

    v2n = skel.vert2node
    if v2n is None:
        raise ValueError("skeleton carries no vert2node — cannot rebuild G")

    g_edges = _edges_from_mesh(np.asarray(mesh.edges_unique), v2n, len(mesh.vertices))
    tree = np.unique(np.sort(np.asarray(skel.edges, dtype=np.int64), axis=1), axis=0)
    tset = {(int(u), int(v)) for u, v in tree}
    gset = {(int(u), int(v)) for u, v in g_edges}

    return {
        "n_nodes": len(skel.nodes),
        "n_tree": len(tset),
        "dropped": sorted(gset - tset),
        "unsupported": sorted(tset - gset),
    }


def _fragmented_bins(bins: List[np.ndarray], mesh) -> Dict[int, int]:
    """Map bin id → number of connected surface patches, for bins with >1.

    One pass over the whole mesh rather than a BFS per bin: keep only the mesh
    edges whose endpoints share an owner, label the connected components of
    what is left, then count distinct labels within each bin.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_verts = len(mesh.vertices)
    owner = np.full(n_verts, -1, dtype=np.int64)
    for nid, b in enumerate(bins):
        if nid and b.size:  # node 0 is soma.verts, not a bin
            owner[b] = nid

    e = np.asarray(mesh.edges_unique)
    keep = (owner[e[:, 0]] == owner[e[:, 1]]) & (owner[e[:, 0]] >= 1)
    e = e[keep]

    adj = coo_matrix(
        (np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])),
        shape=(n_verts, n_verts),
    )
    _, labels = connected_components(adj, directed=False)

    out: Dict[int, int] = {}
    for nid, b in enumerate(bins):
        if nid and b.size > 1:
            n_pieces = np.unique(labels[b]).size
            if n_pieces > 1:
                out[nid] = int(n_pieces)
    return out


def face_owner(skel, mesh) -> np.ndarray:
    """Per-face owning node, ``-1`` where no node owns a majority.

    Bins are sets of *vertices*; almost everything that looks at a mesh —
    highlighting, picking, lassoing — works in *faces*.  The bridge is the same
    ≥2-of-3 majority rule the soma already uses (:func:`skeliner.pre.
    soma_face_mask`): a face belongs to the bin holding at least two of its
    three vertices.

    Measured on two cells, the rule resolves 77–84 % of faces to exactly one
    bin and leaves 0.2–0.3 % straddling three; the rest are soma, organelle or
    discarded surface that no node owns.

    Returns
    -------
    (F,) int64
    """
    if skel.vert2node is None:
        return np.full(len(mesh.faces), -1, dtype=np.int64)

    lut = np.full(len(mesh.vertices), -1, dtype=np.int64)
    keys = np.fromiter(skel.vert2node.keys(), np.int64, len(skel.vert2node))
    vals = np.fromiter(skel.vert2node.values(), np.int64, len(skel.vert2node))
    # An id past the end means this is not the mesh the bins were built over.
    # Left to numpy it is an IndexError naming an array nobody passed in; a
    # mesh one vertex *larger* than the original would not raise at all, which
    # is what :func:`check_mesh_pairing` is for.
    if keys.size and int(keys.max()) >= len(mesh.vertices):
        raise ValueError(
            f"skeleton names vertex {int(keys.max()):,} but the mesh has "
            f"{len(mesh.vertices):,} vertices — it was not built from this "
            "mesh (see dx.check_mesh_pairing)"
        )
    lut[keys] = vals

    per_face = np.sort(lut[np.asarray(mesh.faces)], axis=1)
    return np.where(
        per_face[:, 0] == per_face[:, 1],
        per_face[:, 0],
        np.where(per_face[:, 1] == per_face[:, 2], per_face[:, 1], -1),
    )


def bin_faces(skel, mesh, node: int, *, owner: np.ndarray | None = None) -> np.ndarray:
    """Face ids owned by one node — the surface its radius is measured over.

    Pass a cached :func:`face_owner` array as *owner* to avoid recomputing it
    for every query.
    """
    if owner is None:
        owner = face_owner(skel, mesh)
    return np.flatnonzero(owner == int(node)).astype(np.int64)


# -----------------------------------------------------------------------------
# 2. degree-related helpers
# -----------------------------------------------------------------------------


def degree(skel, node_id: int | Sequence[int]):
    """Return the degree(s) of one node *or* a sequence of nodes."""
    g = _graph(skel)
    if isinstance(node_id, (list, tuple, np.ndarray)):
        return np.asarray(g.degree(node_id))
    return int(g.degree(node_id))


def neighbors(skel, node_id: int) -> List[int]:
    """Neighbour vertex IDs of *node_id* (undirected)."""
    g = _graph(skel)
    return [int(v) for v in g.neighbors(node_id)]


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    """Return Euclidean distance from *point* to the segment [start, end]."""
    vec = end - start
    seg_len2 = float(np.dot(vec, vec))
    if seg_len2 <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, vec) / seg_len2)
    t = min(1.0, max(0.0, t))
    closest = start + t * vec
    return float(np.linalg.norm(point - closest))


def _point_segment_capsule_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    r_start: float,
    r_end: float,
) -> float:
    """
    Return signed distance from *point* to the capsule defined by
    segment [start, end] with radii r_start → r_end.

    Negative values mean the point falls inside the interpolated radius.
    """
    vec = end - start
    seg_len2 = float(np.dot(vec, vec))
    if seg_len2 <= 0.0:
        radius = max(float(r_start), float(r_end))
        return float(np.linalg.norm(point - start)) - radius

    t = float(np.dot(point - start, vec) / seg_len2)
    t = min(1.0, max(0.0, t))
    closest = start + t * vec
    dist = float(np.linalg.norm(point - closest))
    radius = (1.0 - t) * float(r_start) + t * float(r_end)
    return dist - radius


def distance(
    skel,
    point: Sequence[float] | np.ndarray,
    *,
    point_unit: str | None = None,
    k_nearest: int = 4,
    radius_metric: str | None = None,
    mode: str = "surface",
    allowed_nodes: Sequence[int] | None = None,
    allowed_edges: Sequence[tuple[int, int]] | None = None,
) -> float | np.ndarray:
    """
    Distance from an arbitrary point (or collection of points) to the skeleton.

    Parameters
    ----------
    skel
        :class:`skeliner.Skeleton` instance.
    point
        3-vector or array of shape (M, 3) giving query locations.
    point_unit
        Unit of the input coordinates and the returned distance. If ``None`` or
        identical to ``skel.meta['unit']``, no conversion is performed.
    k_nearest
        Number of nearest skeleton nodes considered when refining the distance
        against neighbouring edges (≥ 1). Ignored when a whitelist is provided.
    radius_metric
        Which column of ``skel.radii`` to use. Defaults to the recommended estimator.
        Only consulted when *mode* is ``'surface'``.
    mode
        ``'surface'`` (default) returns the distance to the radius-aware capsule
        envelope (values inside clamp to ``0``). ``'centerline'`` measures distance
        to the centreline alone.
    allowed_nodes
        Optional per-call whitelist of node IDs. When provided, distances are
        restricted to these node centres and to edges incident to any of these nodes.
    allowed_edges
        Optional per-call whitelist of edges (u,v). When provided, distances are
        refined only against these edges (u and v are also considered as centres).
        Edges are treated as undirected; order is ignored.

    Parameters
    ----------
    skel
        :class:`skeliner.Skeleton` instance.
    point
        3-vector or array of shape (M, 3) giving query locations.
    point_unit
        Unit of the input coordinates and the returned distance. If ``None`` or
        identical to ``skel.meta['unit']``, no conversion is performed.
    k_nearest
        Number of nearest skeleton nodes considered when refining the distance
        against neighbouring edges (≥ 1).
    radius_metric
        Which column of ``skel.radii`` to use. Defaults to the recommended estimator.
        Only consulted when *mode* is ``'surface'``.
    mode
        ``'surface'`` (default) returns the distance to the radius-aware capsule
        envelope (values inside clamp to ``0``). ``'centerline'`` measures distance
        to the centreline alone.

    Returns
    -------
    float or ndarray
        Minimum distance(s) in the same unit as *point_unit*. With
        ``mode='surface'`` the envelope distance is returned (0 inside); otherwise
        the pure centreline distance.
    """
    if mode not in {"surface", "centerline"}:
        raise ValueError("mode must be either 'surface' or 'centerline'")
    surface = mode == "surface"

    pts = np.asarray(point, dtype=np.float64)
    if pts.ndim == 1:
        if pts.shape[0] != 3:
            raise ValueError("point must be a 3-vector or an array of shape (M, 3)")
        pts = pts[None, :]
        single_input = True
    elif pts.ndim == 2 and pts.shape[1] == 3:
        single_input = False
    else:
        raise ValueError("point must be a 3-vector or an array of shape (M, 3)")

    if skel.nodes.size == 0:
        raise ValueError("Skeleton has no nodes; cannot compute distances.")
    if k_nearest < 1:
        raise ValueError("k_nearest must be at least 1.")

    tree = skel._ensure_nodes_kdtree()
    neighbours = skel._ensure_node_neighbors()

    skel_unit = skel.meta.get("unit")
    if point_unit is None or skel_unit is None or point_unit == skel_unit:
        scale_in = 1.0
        scale_out = 1.0
    else:
        scale_in = skel._get_unit_conversion_factor(point_unit, skel_unit)
        scale_out = skel._get_unit_conversion_factor(skel_unit, point_unit)

    if surface:
        if radius_metric is None:
            radius_metric = skel.recommend_radius()[0]
        if radius_metric not in skel.radii:
            raise ValueError(
                f"radius_metric '{radius_metric}' not found in skel.radii "
                f"(available keys: {tuple(skel.radii)})"
            )
        radii = np.asarray(skel.radii[radius_metric], dtype=np.float64)
    else:
        radii = None

    distances = np.empty(len(pts), dtype=np.float64)
    max_k = min(int(k_nearest), len(skel.nodes))
    nodes = skel.nodes

    # Prepare whitelist sets if provided
    use_whitelist = (allowed_nodes is not None) or (allowed_edges is not None)
    allowed_nodes_set: Set[int] | None = None
    allowed_edges_set: Set[tuple[int, int]] | None = None
    if allowed_nodes is not None:
        allowed_nodes_set = {int(n) for n in allowed_nodes if 0 <= int(n) < len(nodes)}
    if allowed_edges is not None:
        allowed_edges_set = set()
        for u, v in allowed_edges:
            u2 = int(u)
            v2 = int(v)
            if u2 == v2:
                continue
            if not (0 <= u2 < len(nodes) and 0 <= v2 < len(nodes)):
                continue
            a, b = (u2, v2) if u2 < v2 else (v2, u2)
            allowed_edges_set.add((a, b))

    if (
        use_whitelist
        and not surface
        and allowed_edges_set is None
        and scale_in == 1.0
        and scale_out == 1.0
    ):
        # the same minimum over the allowed centres and their edges, once
        segments = sorted(
            {
                (min(nid, int(nb)), max(nid, int(nb)))
                for nid in allowed_nodes_set
                for nb in neighbours[nid]
            }
        )
        distances = kernels.dx.segment_distances(
            pts,
            np.asarray(nodes, dtype=np.float64),
            sorted(allowed_nodes_set),
            segments,
        )
        return float(distances[0]) if single_input else distances

    for i, p in enumerate(pts):
        p_skel = p * scale_in

        # Initialize best distance from node centres
        if use_whitelist:
            # Collect centres to consider: explicit allowed nodes and endpoints of allowed edges
            centres: Set[int] = set()
            if allowed_nodes_set is not None:
                centres.update(allowed_nodes_set)
            if allowed_edges_set is not None:
                for a, b in allowed_edges_set:
                    centres.add(a)
                    centres.add(b)

            if centres:
                # compute min distance to allowed centres
                centres_list = list(centres)
                diffs = nodes[centres_list] - p_skel
                nn_dist_arr = np.linalg.norm(diffs, axis=1)
                if surface:
                    rad = (
                        np.asarray([radii[c] for c in centres_list], dtype=np.float64)
                        if radii is not None
                        else 0.0
                    )
                    best = float(np.min(nn_dist_arr - rad))
                else:
                    best = float(np.min(nn_dist_arr))
            else:
                best = float("inf")

            # Candidate edges: explicit allowed_edges plus edges incident to allowed_nodes
            candidates: Set[tuple[int, int]] = set()
            if allowed_edges_set is not None:
                candidates.update(allowed_edges_set)
            if allowed_nodes_set is not None:
                for nid in allowed_nodes_set:
                    for nb in neighbours[nid]:
                        a, b = (nid, nb) if nid < nb else (nb, nid)
                        candidates.add((a, b))
        else:
            # default global behaviour via KD-tree + incident edges
            nn_dist, nn_idx = tree.query(p_skel, k=max_k)
            nn_idx_arr = np.atleast_1d(nn_idx).astype(np.int64, copy=False)
            nn_dist_arr = np.atleast_1d(nn_dist)
            if surface:
                best = float(np.min(nn_dist_arr - radii[nn_idx_arr]))
            else:
                best = float(nn_dist_arr.min())

            candidates: set[tuple[int, int]] = set()
            for nid in nn_idx_arr:
                for nb in neighbours[nid]:
                    a, b = (nid, nb) if nid < nb else (nb, nid)
                    candidates.add((a, b))

        if candidates:
            for a_idx, b_idx in candidates:
                if surface:
                    d = _point_segment_capsule_distance(
                        p_skel,
                        nodes[a_idx],
                        nodes[b_idx],
                        radii[a_idx],
                        radii[b_idx],
                    )
                else:
                    d = _point_segment_distance(p_skel, nodes[a_idx], nodes[b_idx])
                if d < best:
                    best = d

        if surface:
            distances[i] = max(best, 0.0) * scale_out
        else:
            distances[i] = best * scale_out

    return float(distances[0]) if single_input else distances


def _node_summary_from_cache(
    skel,
    node_id: int,
    radius_metric: str | None,
    g: ig.Graph,
    deg: np.ndarray,
) -> Dict[str, Any]:
    """Internal helper that reuses a precomputed degree vector + graph."""
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    deg_root = int(deg[node_id])
    r_root = float(skel.radii[radius_metric][node_id])

    summary = {
        "degree": deg_root,
        "radius": r_root,
        "neighbors": [],
    }
    for nb in g.neighbors(node_id):
        summary["neighbors"].append(
            {
                "id": int(nb),
                "degree": int(deg[nb]),
                "radius": float(skel.radii[radius_metric][nb]),
            }
        )
    return summary


def node_summary(
    skel,
    node_id: int,
    *,
    radius_metric: str | None = None,
) -> Dict[str, Any]:
    """Rich information about a single vertex.

    Returned dict structure::

        {
            "degree": int,
            "radius": float,
            "neighbors": [
                {"id": j, "degree": int, "radius": float},
                ...
            ]
        }
    """
    g = _graph(skel)
    deg = np.asarray(g.degree())
    return _node_summary_from_cache(skel, node_id, radius_metric, g, deg)


# -----------------------------------------------------------------------------
# 3. degree distribution with optional detailed map
# -----------------------------------------------------------------------------


def degree_distribution(
    skel,
    *,
    high_deg_percentile: float = 99.5,
    detailed: bool = False,
    radius_metric: str | None = None,
) -> Dict[str, Any]:
    """Histogram + outliers; optionally attach neighbour radii/deg info.

    Parameters
    ----------
    high_deg_percentile
        Percentile threshold that defines *high-degree* nodes.
    detailed
        When *True* each high-degree node is expanded to include its
        neighbours' IDs, degrees and radii.
    radius_metric
        Which radius column to report. Default = the estimator recommended
        by :py:meth:`Skeleton.recommend_radius`.
    """
    g = _graph(skel)
    deg = np.asarray(g.degree())

    hist = np.bincount(deg)
    thresh = np.percentile(deg, high_deg_percentile)
    high = np.where(deg > thresh)[0]

    high_dict: Dict[int, Any] = {}
    for idx in high:
        high_dict[int(idx)] = int(deg[idx])
        if detailed:
            high_dict[int(idx)] = _node_summary_from_cache(
                skel, int(idx), radius_metric, g, deg
            )

    return {
        "degree": np.arange(hist.size)[1:],
        "counts": hist[1:],
        "threshold": float(thresh),
        "high_degree_nodes": high_dict,
    }


def nodes_of_degree(skel, k: int):
    """Return *all* node IDs whose degree == *k* (soma excluded).

    Examples
    --------
    >>> leaves = dx.nodes_of_degree(skel, 1)
    >>> hubs = dx.nodes_of_degree(skel, 4)
    """
    if k < 0:
        raise ValueError("k must be non‑negative")
    g = _graph(skel)
    deg = np.asarray(g.degree())
    idx = np.where(deg == k)[0]
    if k == deg[0]:  # avoid returning the soma
        idx = idx[idx != 0]
    return idx.astype(int)


def branches_of_length(
    skel,
    k: int,
    *,
    include_endpoints: bool = True,
) -> List[List[int]]:
    """Return every *branch* (sequence of degree‑2 vertices) whose length == k.

    Definition of a *branch*
    ------------------------
    A maximal simple path **P** such that:
    * the two endpoints have degree ≠ 2 (soma, bifurcation, or leaf), and
    * every interior vertex (if any) has degree == 2.

    Example – degree pattern ``1‑2‑2‑3``::

        0‑1‑2‑3
        ^   ^   ^
        |   |   +—— endpoint (deg != 2)
        |   +—— interior (deg == 2)
        +—— endpoint (leaf)

    ``branches_of_length(skel, k=3)`` would return ``[[0,1,2]]``.

    Parameters
    ----------
    k
        Desired branch length *in number of nodes* (``len(path)``).
    include_endpoints
        If *True* endpoints are counted as part of the path and therefore
        contribute to *k*.  If *False* only the *interior* degree‑2 vertices
        are counted.
    """
    g = _graph(skel)
    deg = np.asarray(g.degree())

    # Mark endpoints = vertices with degree != 2 OR soma (0) even if deg==2
    endpoints: Set[int] = {i for i, d in enumerate(deg) if d != 2}
    endpoints.add(0)

    visited_edges: Set[Tuple[int, int]] = set()
    branches: List[List[int]] = []

    for ep in endpoints:
        for nb in g.neighbors(ep):
            edge = tuple(sorted((ep, nb)))
            if edge in visited_edges:
                continue

            path = [ep]
            prev, curr = ep, nb
            while True:
                path.append(curr)
                visited_edges.add(
                    (min(int(prev), int(curr)), max(int(prev), int(curr)))
                )
                if curr in endpoints:
                    break
                # internal vertex (deg==2) → continue straight
                nxts = [v for v in g.neighbors(curr) if v != prev]
                if not nxts:
                    break  # should not happen in a well‑formed tree
                prev, curr = curr, nxts[0]

            length = len(path) if include_endpoints else len(path) - 2
            if length == k:
                branches.append([int(v) for v in path])

    return branches


def twigs_of_length(
    skel,
    k: int,
    *,
    include_branching_node: bool = False,
) -> List[List[int]]:
    """
    Return every *terminal twig* whose **chain length** == k.

    *Twig length* counts the leaf (deg==1) and all intermediate deg==2
    vertices **up to but NOT including** the branching point (deg>2 or soma).

    Parameters
    ----------
    k  : int
        Number of vertices in the terminal chain *excluding* the branching
        node.  Example::

            soma-B-1-2-L        # degrees  >2-2-2-1
                 └─┬──────      k = 3   (1-2-L)
                   `- returned path length is 3 or 4
                      depending on include_branching_node
    include_branching_node : bool, default ``False``
        If *True*, the branching node is prepended to each returned path.

    Returns
    -------
    list[list[int]]
        Each sub-list is ordered **proximal ➜ leaf**.
        * Length == k              when include_branching_node=False
        * Length == k + 1          when include_branching_node=True
    """
    g = _graph(skel)
    deg = np.asarray(g.degree())

    twigs: List[List[int]] = []

    # candidates = all leaves (deg==1, exclude soma)
    leaves = [v for v in range(1, len(deg)) if deg[v] == 1]
    parent = g.bfs(0, mode="ALL")[2]

    for leaf in leaves:
        chain = [leaf]
        curr = leaf
        while True:
            par = parent[curr]
            if par == -1:
                break  # should not happen – disconnected
            if deg[par] == 2 and par != 0:
                chain.append(par)
                curr = par
                continue
            # par is branching point (deg!=2 or soma)
            if len(chain) == k:
                if include_branching_node:
                    chain.append(par)
                twigs.append(chain[::-1])  # proximal➜distal order
            break

    return twigs


# -----------------------------------------------------------------------------
# 3. leaf depths (BFS distance in *edges* from soma)
# -----------------------------------------------------------------------------


def leaf_depths(skel) -> np.ndarray:
    """Depth (in *edges*) of every leaf node relative to the soma."""
    g = _graph(skel)
    deg = np.asarray(g.degree())
    leaves = np.where((deg == 1) & (np.arange(len(deg)) != 0))[0]
    if leaves.size == 0:
        return np.empty(0, dtype=int)
    # Only compute distances to the leaf subset to avoid full all-pairs output
    dists = np.asarray(
        g.shortest_paths_dijkstra(source=0, target=leaves.tolist(), weights=None)[0]
    )
    return dists.astype(int)


def suspicious_tips(
    skel,
    *,
    near_factor: float = 1.2,
    path_ratio_thresh: float = 2.0,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[int, Dict[str, float]]]:
    r"""Identify *tip* nodes suspiciously close to the soma.

    A *tip* is a node with graph degree = 1 (i.e. a leaf) and **not** the soma
    itself.  A leaf *i* is flagged when

    1. Its Euclidean distance to the soma center is *small*::

           d\_euclid(i) \le near\_factor × max(soma.axes)

    2. Yet the shortest‑path length along the skeleton is *long*::

           d\_graph(i) / d\_euclid(i) \ge path\_ratio\_thresh

    Parameters
    ----------
    skel
        A fully‑constructed :class:`skeliner.Skeleton` instance.
    near_factor
        Multiplicative factor applied to the largest soma semi‑axis to set the
        *Euclidean* proximity threshold.
    path_ratio_thresh
        Minimum ratio between graph‑path length and straight‑line distance for
        a leaf to be considered suspicious.
    return_stats
        If *True*, a per‑node diagnostic dictionary is returned in addition to
        the sorted list of suspicious node IDs.

    Returns
    -------
    suspicious
        ``list[int]`` – tip node indices, sorted by decreasing *path/straight*
        ratio (most suspicious first).
    stats
        *Optional* ``dict[int, dict]`` where each entry contains::

            {"d_center", "d_surface", "path_len", "ratio"}
    """
    if skel.nodes.size == 0 or skel.edges.size == 0:
        return [] if not return_stats else ([], {})

    soma_c = skel.soma.center.astype(np.float64)
    r_max = float(skel.soma.axes.max())
    near_thr = near_factor * r_max

    # Build an igraph view with edge‑length weights (Euclidean)
    g: ig.Graph = skel._igraph()
    g.es["weight"] = [
        float(np.linalg.norm(skel.nodes[a] - skel.nodes[b])) for a, b in skel.edges
    ]

    # Tip detection – degree = 1, excluding the soma (node 0)
    deg = np.bincount(skel.edges.flatten(), minlength=len(skel.nodes))
    tips = np.where(deg == 1)[0]
    tips = tips[tips != 0]  # exclude soma itself
    if tips.size == 0:
        return [] if not return_stats else ([], {})

    # Shortest path (edge‑weighted) length to soma for every tip
    path_d = np.asarray(
        g.distances(source=list(tips), target=[0], weights="weight"),
        dtype=np.float64,
    ).reshape(-1)

    # Straight‑line metrics
    eucl_d = np.linalg.norm(skel.nodes[tips] - soma_c, axis=1)
    surf_d = skel.soma.distance_to_surface(skel.nodes[tips])

    # Robust guard against division by zero (very unlikely)
    ratio = path_d / np.maximum(eucl_d, 1e-9)

    sus_mask = (eucl_d <= near_thr) & (ratio >= path_ratio_thresh)
    suspicious = tips[sus_mask]

    if not return_stats:
        # sort by descending ratio (most egregious first)
        return sorted(
            map(int, suspicious), key=lambda nid: -ratio[np.where(tips == nid)[0][0]]
        )

    stats: Dict[int, Dict[str, float]] = {
        int(nid): {
            "d_center": float(eucl_d[i]),
            "d_surface": float(surf_d[i]),
            "path_len": float(path_d[i]),
            "ratio": float(ratio[i]),
        }
        for i, nid in enumerate(tips)
        if sus_mask[i]
    }

    suspicious_sorted = sorted(suspicious, key=lambda nid: -stats[int(nid)]["ratio"])
    return suspicious_sorted, stats


def _arm_cables(skel, min_degree: int) -> Dict[int, Dict[str, Any]]:
    """Decompose every junction into the cable each of its arms carries.

    Deleting node *i* splits the tree into one component per neighbour.  The
    component reached through the parent is *proximal* (it holds the soma);
    the rest are *distal*.  This is the measurement
    :func:`suspicious_junctions` thresholds, kept separate because which
    thresholds are right is an open question and the raw numbers are what
    answers it.

    Linear in the number of nodes.  The naive form — delete a node, flood
    each arm, repeat — is O(N²) and visits billions of nodes on a 50k-node
    skeleton.  Rooting the tree at the soma makes the same decomposition a
    single post-order accumulation:

    ``sub[v]``
        cable strictly inside the subtree rooted at *v*.
    distal arm through child *c*
        ``sub[c] + len(i, c)``
    proximal arm
        ``total − sub[i]``

    which sums back to ``total`` exactly, so the arms partition the cable
    rather than merely sampling it.

    A disconnected skeleton is handled by rooting **each** component at its
    lowest node id; for the component holding the soma that is node 0, so the
    usual case is unchanged and the orphan components still get measured
    instead of raising.

    Recursion is avoided throughout — a long neurite is a deep tree, and the
    recursive form overflows the stack on real cells.
    """
    n = len(skel.nodes)
    edges = np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2)
    if n == 0 or edges.size == 0:
        return {}

    nodes = np.asarray(skel.nodes, dtype=np.float64)
    lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)

    # CSR-style adjacency: neighbour ids and the length of the edge used.
    deg = np.bincount(edges.reshape(-1), minlength=n)
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=start[1:])
    fill = start[:-1].copy()
    adj = np.empty(2 * len(edges), dtype=np.int64)
    adj_len = np.empty(2 * len(edges), dtype=np.float64)
    for (a, b), ln in zip(edges, lengths, strict=True):
        adj[fill[a]] = b
        adj_len[fill[a]] = ln
        fill[a] += 1
        adj[fill[b]] = a
        adj_len[fill[b]] = ln
        fill[b] += 1

    parent = np.full(n, -1, dtype=np.int64)
    parent_len = np.zeros(n, dtype=np.float64)
    order = np.empty(n, dtype=np.int64)  # BFS order, roots first
    seen = np.zeros(n, dtype=bool)
    k = 0
    for root in range(n):  # node 0 first, so the soma roots its own component
        if seen[root]:
            continue
        seen[root] = True
        order[k] = root
        k += 1
        head = k - 1
        while head < k:
            v = order[head]
            head += 1
            for j in range(start[v], start[v + 1]):
                w = adj[j]
                if not seen[w]:
                    seen[w] = True
                    parent[w] = v
                    parent_len[w] = adj_len[j]
                    order[k] = w
                    k += 1

    # Post-order accumulation: walk the BFS order backwards, so every child is
    # finished before its parent is read.
    sub = np.zeros(n, dtype=np.float64)
    for idx in range(n - 1, -1, -1):
        v = order[idx]
        p = parent[v]
        if p >= 0:
            sub[p] += sub[v] + parent_len[v]

    # Total cable of the component each node belongs to — the proximal arm is
    # measured against its own component, not the whole file.
    comp_total = np.zeros(n, dtype=np.float64)
    comp_of = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        v = order[idx]
        comp_of[v] = v if parent[v] < 0 else comp_of[parent[v]]
    for v in range(n):
        comp_total[v] = sub[comp_of[v]]

    out: Dict[int, Dict[str, Any]] = {}
    for v in range(n):
        if deg[v] < min_degree or parent[v] < 0:
            # A component root has no proximal arm, so "one arm holds the
            # soma and the others do not" is undefined there.  The soma is
            # legitimately high-degree anyway.
            continue
        distal = [
            float(sub[adj[j]] + adj_len[j])
            for j in range(start[v], start[v + 1])
            if adj[j] != parent[v]
        ]
        distal.sort(reverse=True)
        out[int(v)] = {
            "degree": int(deg[v]),
            "proximal_cable": float(comp_total[v] - sub[v]),
            "distal_cables": distal,
        }
    return out


def suspicious_junctions(
    skel,
    *,
    min_degree: int = 4,
    min_distal_cable: float = 250.0,
    cable_unit: str = "um",
    min_components: int = 2,
    group_regions: bool = True,
    return_stats: bool = False,
):
    """Junctions where two or more substantial arbors meet — merge candidates.

    A segmentation merge fuses two unrelated neurites in the *mesh*, so the
    skeleton grows a junction welding two independent arbors together.  The
    result is still a valid tree — one root, ``E = N − 1``, connected,
    acyclic — so :func:`check_connectivity` and :func:`check_acyclicity`
    cannot see it.  What gives it away is the shape: a genuine bifurcation
    into two long arbors at a single high-degree point is rare, while a merge
    produces exactly that.

    A node is flagged when its degree is at least *min_degree* **and** at
    least *min_components* of its distal arms each carry at least
    *min_distal_cable* of cable.

    This **reports candidates and never clips**.  The rule is a heuristic
    with no validation behind it, and acting on it destroys real cable, so
    the cut is a human decision made against the mesh — the same split
    :mod:`skeliner.pre` draws between its ``find_*`` and ``remove_*``
    functions.

    Parameters
    ----------
    skel
        A :class:`skeliner.Skeleton` instance.  Assumed acyclic — check with
        :func:`check_acyclicity` first, since ``post.graft(...,
        allow_cycle=True)`` can break that assumption.
    min_degree
        Minimum graph degree for a node to be considered at all.
    min_distal_cable
        How much cable a distal arm must carry to count as substantial,
        expressed in *cable_unit*.
    cable_unit
        Unit *min_distal_cable* is given in, converted into the skeleton's
        own unit via ``skel.meta["unit"]``.  Skeletons from
        :func:`~skeliner.skeletonize` are in nanometres by default, so a
        bare ``250`` in skeleton units would be 250 nm — smaller than a
        single node radius, which flags every junction.  Naming the unit
        makes that mistake unrepresentable.
    min_components
        How many distal arms must clear *min_distal_cable*.
    group_regions
        A merge flags a cluster of adjacent nodes rather than one.  When
        *True* each connected cluster of flagged nodes collapses to a single
        representative — the one whose second-largest distal arm is longest
        — turning an N-item review queue into a handful.
    return_stats
        When *True* also return the per-node measurements.  Stats cover
        **every node examined** (degree ≥ *min_degree*), flagged or not, so
        a threshold can be re-chosen from them without recomputing.

    Returns
    -------
    flagged
        ``list[int]`` – node ids, most suspicious first (by the second-largest
        distal arm, which is the arm that made it suspicious).
    stats
        *Optional* ``dict[int, dict]`` with ``{"degree", "proximal_cable",
        "distal_cables"}``; cables are in the skeleton's own unit.
    """
    if min_components < 1:
        raise ValueError("min_components must be at least 1")

    # Resolved before any work: a skeleton that cannot say what its
    # coordinates mean cannot have a cable threshold applied to it, and
    # answering "nothing suspicious" for one would be the silent kind of
    # wrong this whole function exists to avoid.
    skel_unit = (getattr(skel, "meta", {}) or {}).get("unit")
    if skel_unit is None:
        raise ValueError(
            "skeleton has no meta['unit'], so a cable threshold cannot be "
            "converted — set one with skel.set_unit(...)"
        )
    thresh = min_distal_cable * skel._get_unit_conversion_factor(cable_unit, skel_unit)

    stats = _arm_cables(skel, min_degree)
    if not stats:
        return ([], {}) if return_stats else []

    def _rank(nid: int) -> float:
        """Cable of the arm that made it suspicious."""
        d = stats[nid]["distal_cables"]
        return d[min_components - 1] if len(d) >= min_components else -1.0

    flagged = [
        nid
        for nid, s in stats.items()
        if sum(1 for c in s["distal_cables"] if c >= thresh) >= min_components
    ]

    if group_regions and flagged:
        flagged = _collapse_adjacent(skel, flagged, key=_rank)

    flagged.sort(key=lambda nid: -_rank(nid))
    return (flagged, stats) if return_stats else flagged


def _collapse_adjacent(skel, nodes: List[int], *, key) -> List[int]:
    """Reduce each connected cluster of *nodes* to its highest-*key* member."""
    want = set(int(n) for n in nodes)
    edges = np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2)

    nbr: Dict[int, List[int]] = {n: [] for n in want}
    for a, b in edges:
        a, b = int(a), int(b)
        if a in want and b in want:
            nbr[a].append(b)
            nbr[b].append(a)

    reps: List[int] = []
    unseen = set(want)
    while unseen:
        stack = [unseen.pop()]
        cluster = [stack[0]]
        while stack:
            v = stack.pop()
            for w in nbr[v]:
                if w in unseen:
                    unseen.discard(w)
                    cluster.append(w)
                    stack.append(w)
        reps.append(max(cluster, key=key))
    return reps


def extract_neurites(
    skel,
    root: int,
    *,
    include_root: bool = True,
) -> List[int]:
    """Return the full *neurite subtree* emerging distally from ``root``.

    The routine uses *graph distance* to the soma (node 0) to orient edges:
    for every edge ``(u, v)`` the direction is from the **closer** vertex to
    soma → **further** vertex.  All vertices whose shortest path to soma passes
    through ``root`` (including any downstream bifurcations) are returned.

        The skeleton is assumed to be a tree (acyclic).  *Distal* means all
        descendants of ``root`` when the soma (vertex 0) is treated as the
        root.

        Examples
        --------
        >>> skel.dx.extract_neurite(skel, 2)
        [2, 3, 4, 5, ...]   # entire subtree starting at 2
        >>> skel.dx.extract_neurite(skel, 0, include_root=False)
        list(range(1, len(skel.nodes)))  # every non‑soma node

        Parameters
        ----------
        skel
            A :class:`skeliner.Skeleton` instance.
        root
            Index of the *proximal* node that defines the neurite base.
        include_root : bool, default ``True``
            Whether ``root`` itself should be included in the returned list.

        Returns
        -------
        list[int]
            Sorted vertex IDs belonging to the neurite.
    """
    N = len(skel.nodes)
    if root < 0 or root >= N:
        raise ValueError("root is out of range")
    return sorted(_subtree(_distal_children(skel), root, include_root))


def _distal_children(skel) -> List[List[int]]:
    """Each node's children, every edge pointing away from the soma (node 0).

    Edges are oriented by unweighted shortest-path distance to the soma; an
    edge between two nodes at the same distance, which a tree cannot have, is
    left out, and so is every edge of a forest's other trees, which the soma
    does not reach.
    """
    g = skel._igraph()
    # hops from the soma, infinite where it does not reach
    dists = np.asarray(g.distances(source=[0])[0], dtype=np.float64)
    children: List[List[int]] = [[] for _ in range(len(skel.nodes))]
    for a, b in skel.edges:
        da, db = dists[a], dists[b]
        if da == db:
            continue
        parent, child = (a, b) if da < db else (b, a)
        children[parent].append(child)
    return children


def _subtree(children: List[List[int]], root: int, include_root: bool) -> List[int]:
    """*root* and every node below it in *children*, depth first."""
    out: List[int] = []
    stack = [root]
    while stack:
        v = stack.pop()
        if v != root or include_root:
            out.append(v)
        stack.extend(children[v])
    return out


def neurite_names(skel) -> Dict[int, str]:
    """The names the neurites had when this skeleton was built.

    ``{index: label}``, empty when the neurites were unnamed or the
    skeleton did not come from the preprocessing track.  Recorded because
    ``ntype`` cannot carry it — "dendrite 0" and "dendrite 1" are both
    code 3.

    Examples
    --------
    >>> skel.dx.neurite_names()
    {0: 'dendrite 0', 1: 'dendrite 1', 2: 'axon'}
    """
    labels = (skel.meta or {}).get("neurite_labels")
    return {} if labels is None else {i: str(x) for i, x in enumerate(labels)}


def neurite_nodes(skel, which: int | str) -> np.ndarray:
    """The nodes belonging to one neurite, by index or by name.

    ``ntype`` groups neurites by *kind*, which is not the same question:
    two dendrites share code 3.  This resolves a single neurite.

    Parameters
    ----------
    skel
        A :class:`skeliner.Skeleton` from the preprocessing track.
    which : int or str
        The neurite's position, or the name it was given.

    Returns
    -------
    np.ndarray
        Node ids, ascending.  The soma is never included: it belongs to no
        neurite, and under the >=2-of-3 face rule a junction face carries
        soma vertices, so it would otherwise be claimed by whichever
        neurite touches it.

    Examples
    --------
    >>> skel.dx.neurite_nodes("axon")
    array([412, 413, 414, ...])
    """
    owner = (skel.extra or {}).get("node2neurite")
    if owner is None:
        raise KeyError(
            "this skeleton has no per-node neurite map — it was built from "
            "unnamed neurites, or not by the preprocessing track"
        )

    if isinstance(which, str):
        names = neurite_names(skel)
        hits = [i for i, label in names.items() if label == which]
        if not hits:
            raise KeyError(
                f"no neurite called {which!r}; have {sorted(names.values())}"
            )
        if len(hits) > 1:
            raise KeyError(f"{len(hits)} neurites are called {which!r}: {hits}")
        index = hits[0]
    else:
        index = int(which)

    return np.flatnonzero(np.asarray(owner) == index)


def neurites_out_of_bounds(
    skel,
    bounds: tuple[np.ndarray, np.ndarray] | tuple[Sequence[float], Sequence[float]],
    *,
    include_root: bool = True,
) -> list[int]:
    """
    Return all node IDs that belong to a *distal* subtree whose **root is the
    first node that leaves the axis-aligned bounding box** ``bounds``.

    Parameters
    ----------
    bounds
        ``(lo, hi)`` – each a 3-vector.  A node is inside iff
        ``lo <= coord <= hi`` component-wise.
    include_root
        Whether the very first out-of-bounds node should be included in the
        output.  (Default: ``True``.)

    Notes
    -----
    * Works on acyclic skeletons (trees).
    * Uses only igraph helpers; no custom BFS routine.
    """
    lo_hi = _parse_bbox(bounds)
    if lo_hi is None:
        raise ValueError("bounds must be provided")
    lo, hi = lo_hi

    coords = skel.nodes
    outside = np.any((coords < lo) | (coords > hi), axis=1)
    if not outside.any():
        return []

    # one BFS from the soma (vertex 0) and one orientation of the edges, shared
    # by every out-of-bounds node; igraph gives the soma the parent -1 and a
    # node it does not reach -2, so each of those starts a subtree of its own
    parents = skel._igraph().bfs(0, mode="ALL")[2]
    children = _distal_children(skel)

    # ------------------------------------------------------------------
    # Treat each *first* out-of-bounds node as a neurite root
    # ------------------------------------------------------------------
    targets: set[int] = set()
    for nid in np.where(outside)[0]:
        # ensure nid is indeed the *first* outside node on its path
        par = parents[nid]
        if par >= 0 and outside[par]:
            continue  # ancestor already outside – skip
        targets.update(sorted(_subtree(children, int(nid), include_root)))

    return sorted(targets)


# -----------------------------------------------------------------------------
# volume helpers
# -----------------------------------------------------------------------------


def _parse_bbox(bbox) -> tuple[np.ndarray, np.ndarray] | None:
    """Accepts [xmin, xmax, ymin, ymax, zmin, zmax] or ((xlo,ylo,zlo),(xhi,yhi,zhi))."""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 6:
        lo = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float64)
        hi = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float64)
        if not np.all(lo <= hi):
            raise ValueError("bbox must satisfy lo <= hi in each axis")
        return lo, hi
    if isinstance(bbox, (list, tuple)) and len(bbox) == 2:
        lo = np.asarray(bbox[0], dtype=np.float64).reshape(3)
        hi = np.asarray(bbox[1], dtype=np.float64).reshape(3)
        if not np.all(lo <= hi):
            raise ValueError("bbox must satisfy lo <= hi in each axis")
        return lo, hi
    raise ValueError("bbox must be [xmin,xmax,ymin,ymax,zmin,zmax] or (lo, hi)")


def _ellipsoid_aabb(soma) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a rotated ellipsoid."""
    # For x = c + R diag(a) u, u ∈ unit sphere, the extreme along axis i is:
    #   c_i ± sum_j |R_{ij}| * a_j
    R = soma.R
    a = soma.axes
    extents = np.abs(R) @ a
    lo = soma.center - extents
    hi = soma.center + extents
    return lo, hi


def _choose_voxel_size(
    radii: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    target_voxels: float = 1e8,
    user_voxel: float | None = None,
    min_voxels_across_diam: int = 24,
) -> tuple[float, tuple[int, int, int]]:
    if user_voxel is not None and user_voxel <= 0:
        raise ValueError("voxel_size must be positive")

    if user_voxel is not None:
        base = float(user_voxel)
    else:
        pos = radii[radii > 0]
        if pos.size:
            base = float(np.percentile(pos, 25)) / 3.0
            base = max(base, 1e-6)
            r_ref = float(np.percentile(pos, 10))
            if r_ref > 0:
                base = min(base, (2.0 * r_ref) / float(min_voxels_across_diam))
        else:
            span = float(np.max(hi - lo))
            base = max(span / 256.0, 1e-6)

    span = hi - lo
    n_est = np.ceil(span / base).astype(int)
    est_total = float(n_est[0] * n_est[1] * n_est[2])

    if est_total > target_voxels:
        scale = (est_total / target_voxels) ** (1.0 / 3.0)
        base *= scale
        n_est = np.ceil(span / base).astype(int)

    n_est = np.maximum(n_est, 1)
    return float(base), (int(n_est[0]), int(n_est[1]), int(n_est[2]))


def _voxelize_union(
    skel,
    radii: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    voxel_size: float | None,
    include_soma: bool,
):
    """
    Boolean occupancy grid for the union of edge frusta and (optionally) soma,
    inside [lo, hi].  Returns (occ, h, (nx,ny,nz), lo, hi).
    """
    h, (nx, ny, nz) = _choose_voxel_size(radii, lo, hi, user_voxel=voxel_size)
    xs = lo[0] + (np.arange(nx) + 0.5) * h
    ys = lo[1] + (np.arange(ny) + 0.5) * h
    zs = lo[2] + (np.arange(nz) + 0.5) * h

    occ = np.zeros((nx, ny, nz), dtype=bool)
    nodes = skel.nodes.astype(np.float64, copy=False)
    edges = skel.edges.astype(np.int64, copy=False)

    # ---------------- helpers ----------------
    def _range_x(x0, x1):
        i0 = int(max(0, np.floor((x0 - lo[0]) / h)))
        i1 = int(min(nx - 1, np.floor((x1 - lo[0]) / h)))
        return i0, i1

    def _range_y(y0, y1):
        j0 = int(max(0, np.floor((y0 - lo[1]) / h)))
        j1 = int(min(ny - 1, np.floor((y1 - lo[1]) / h)))
        return j0, j1

    def _range_z(z0, z1):
        k0 = int(max(0, np.floor((z0 - lo[2]) / h)))
        k1 = int(min(nz - 1, np.floor((z1 - lo[2]) / h)))
        return k0, k1

    # --------- precompute soma mask once (if there is a soma) ------------
    soma_slice = None
    soma_mask = None
    if getattr(skel, "soma", None) is not None:
        slo, shi = _ellipsoid_aabb(skel.soma)
        # clip to global bbox
        slo = np.maximum(slo, lo)
        shi = np.minimum(shi, hi)
        i0, i1 = _range_x(slo[0], shi[0])
        j0, j1 = _range_y(slo[1], shi[1])
        k0, k1 = _range_z(slo[2], shi[2])

        if (i1 >= i0) and (j1 >= j0) and (k1 >= k0):
            # broadcasted coordinate slabs (no big meshgrid; uses ogrid)
            xi = xs[i0 : i1 + 1][:, None, None]
            yj = ys[j0 : j1 + 1][None, :, None]
            zk = zs[k0 : k1 + 1][None, None, :]

            cx, cy, cz = skel.soma.center
            Rt = skel.soma.R.T  # 3x3
            ax, ay, az = skel.soma.axes
            ax2, ay2, az2 = ax * ax, ay * ay, az * az

            dx = xi - cx
            dy = yj - cy
            dz = zk - cz

            # rotate into soma body-frame: u = R^T (x - c)
            ux = Rt[0, 0] * dx + Rt[0, 1] * dy + Rt[0, 2] * dz
            uy = Rt[1, 0] * dx + Rt[1, 1] * dy + Rt[1, 2] * dz
            uz = Rt[2, 0] * dx + Rt[2, 1] * dy + Rt[2, 2] * dz

            soma_mask = (ux * ux) / ax2 + (uy * uy) / ay2 + (uz * uz) / az2 <= 1.0
            soma_slice = (slice(i0, i1 + 1), slice(j0, j1 + 1), slice(k0, k1 + 1))

    # If soma is to be included, OR it now.
    if include_soma and soma_mask is not None:
        occ[soma_slice] |= soma_mask

    # ---------------- rasterize every edge (broadcasted) -----------------
    for i, j in edges:
        a = nodes[i]
        b = nodes[j]
        r0 = float(radii[i])
        r1 = float(radii[j])

        rmax = max(r0, r1)
        if not np.isfinite(rmax) or rmax < 0.0:
            continue

        # edge AABB padded by rmax, clipped to [lo,hi]
        lo_e = np.maximum(np.minimum(a, b) - rmax, lo)
        hi_e = np.minimum(np.maximum(a, b) + rmax, hi)
        if np.any(lo_e > hi_e):
            continue

        ii0, ii1 = _range_x(lo_e[0], hi_e[0])
        jj0, jj1 = _range_y(lo_e[1], hi_e[1])
        kk0, kk1 = _range_z(lo_e[2], hi_e[2])
        if (ii1 < ii0) or (jj1 < jj0) or (kk1 < kk0):
            continue

        xi = xs[ii0 : ii1 + 1][:, None, None]
        yj = ys[jj0 : jj1 + 1][None, :, None]
        zk = zs[kk0 : kk1 + 1][None, None, :]

        v = b - a
        L2 = float(v @ v)
        if L2 <= 1e-24:
            # degenerate: paint a ball of radius rmax at 'a'
            dx = xi - a[0]
            dy = yj - a[1]
            dz = zk - a[2]
            d2 = dx * dx + dy * dy + dz * dz
            mask = d2 <= (rmax * rmax)
            occ[ii0 : ii1 + 1, jj0 : jj1 + 1, kk0 : kk1 + 1] |= mask
            continue

        vx, vy, vz = v
        # projection parameter s (broadcasted), then clamp to [0,1]
        dx = xi - a[0]
        dy = yj - a[1]
        dz = zk - a[2]
        s = (dx * vx + dy * vy + dz * vz) / L2
        # clip in-place to save a temporary
        np.clip(s, 0.0, 1.0, out=s)

        # distance from voxel center to closest point on segment
        rx = dx - s * vx
        ry = dy - s * vy
        rz = dz - s * vz
        d2 = rx * rx + ry * ry + rz * rz

        # linear radius along the frustum
        r = r0 + s * (r1 - r0)
        mask = d2 <= (r * r)

        occ[ii0 : ii1 + 1, jj0 : jj1 + 1, kk0 : kk1 + 1] |= mask

    # If soma is to be excluded, carve it out once (reuses precomputed mask).
    if (not include_soma) and (soma_mask is not None):
        occ[soma_slice] &= ~soma_mask

    return occ, float(h), (int(nx), int(ny), int(nz)), lo, hi


def volume(
    skel,
    bbox: list[float] | tuple[Sequence[float], Sequence[float]] | None = None,
    *,
    radius_metric: str | None = None,
    voxel_size: float | None = None,
    include_soma: bool = True,
    return_details: bool = False,
):
    """
    Estimate the morphology volume, optionally restricted to an axis-aligned bbox.

    Robust union via voxelization inside the bbox: fills voxels that lie
    inside any edge frustum or the soma ellipsoid. Correctly handles
    branch overlaps and bbox clipping. Accuracy controlled by `voxel_size`
    (defaults to ~1/3 of a thin-branch radius, auto-scaled to keep the grid
    under ~60M voxels unless you pass voxel_size explicitly).

    Parameters
    ----------
    bbox
        None (whole neuron) or [xmin, xmax, ymin, ymax, zmin, zmax] or (lo, hi).
    radius_metric
        Which `skel.radii[metric]` column to use; defaults to the
        choice from `skel.recommend_radius()`.
    voxel_size
        Edge length of voxels. If None, a size is chosen
        from radii and capped so the grid stays reasonably small.
    include_soma
        Whether to include the soma ellipsoid in the volume.
    return_details
        If True, returns (V, details_dict) with diagnostic info for debugging.

    Returns
    -------
    float or (float, dict)
        Estimated volume (in the cube of your skeleton units). If
        `return_details=True`, also returns a small diagnostics dict.

    Notes
    -----
    * 'frustum' is blazing-fast and good for whole-cell summaries. It trims
      soma overlap on edges but **does not** de-overlap at branch junctions.
    * 'voxel' is the accurate union (no double counting) and is recommended
      whenever `bbox` is used or precise union is important.
    """
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_metric], dtype=np.float64).reshape(-1)
    if radii.shape[0] != skel.nodes.shape[0]:
        raise ValueError("radius_metric array length must match number of nodes")

    # dispatch
    lo_hi = _parse_bbox(bbox)

    if lo_hi is None:
        # Auto bbox that tightly encloses neuron + radii + soma extents
        lo_nodes = (skel.nodes - radii[:, None]).min(axis=0)
        hi_nodes = (skel.nodes + radii[:, None]).max(axis=0)
        if include_soma and skel.soma is not None:
            slo, shi = _ellipsoid_aabb(skel.soma)
            lo = np.minimum(lo_nodes, slo)
            hi = np.maximum(hi_nodes, shi)
        else:
            lo, hi = lo_nodes, hi_nodes
    else:
        lo, hi = lo_hi

    occ, h, (nx, ny, nz), lo, hi = _voxelize_union(
        skel, radii, lo, hi, voxel_size=voxel_size, include_soma=include_soma
    )
    count = int(occ.sum())
    V = float(count) * (h**3)
    if not return_details:
        return V
    return V, {
        "voxel_size": h,
        "grid_shape": (nx, ny, nz),
        "bbox_lo": lo,
        "bbox_hi": hi,
        "filled_voxels": count,
    }


# -----------------------------------------------------------------------------
# path length (sum of edge lengths) with optional bbox clipping
# -----------------------------------------------------------------------------


def _segment_aabb_clip_length(
    a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """
    Return the length of the line segment [a,b] that lies inside the
    axis-aligned box [lo, hi].  Zero if there is no intersection.
    """
    v = b - a
    L = float(np.linalg.norm(v))
    if L <= 0.0:
        # Degenerate edge → contributes nothing (even if "inside").
        return 0.0

    # Liang–Barsky style parametric clipping in 3D.
    t0, t1 = 0.0, 1.0
    for d in range(3):
        vd = float(v[d])
        ad = float(a[d])
        eps = 1e-12 * max(1.0, abs(v[0]), abs(v[1]), abs(v[2]))
        if abs(vd) < eps:
            # Segment is parallel to this slab; reject if outside.
            if ad < lo[d] or ad > hi[d]:
                return 0.0
            continue

        t_enter = (lo[d] - ad) / vd
        t_exit = (hi[d] - ad) / vd
        if t_enter > t_exit:
            t_enter, t_exit = t_exit, t_enter

        t0 = max(t0, t_enter)
        t1 = min(t1, t_exit)
        if t0 > t1:
            return 0.0

    return L * max(0.0, t1 - t0)


def total_path_length(
    skel,
    bbox: list[float] | tuple[Sequence[float], Sequence[float]] | None = None,
    *,
    return_details: bool = False,
):
    """
    Sum of Euclidean edge lengths, optionally **clipped** to an axis-aligned bbox.

    Parameters
    ----------
    bbox
        None → full skeleton length (no clipping); or
        [xmin, xmax, ymin, ymax, zmin, zmax]; or ((xlo,ylo,zlo), (xhi,yhi,zhi)).
    return_details
        When True, also returns a small diagnostics dict.

    Returns
    -------
    float or (float, dict)
        Total path length in the same units as your coordinates.

    Notes
    -----
    * This is purely geometric path length over the graph edges. It does **not**
      subtract portions running inside the soma ellipsoid.
    * Complexity O(|E|). Numerically robust to nearly-parallel edges.
    """
    nodes = skel.nodes.astype(np.float64)
    edges = np.asarray(skel.edges, dtype=int)

    if bbox is None:
        # Fast vectorized sum with no clipping.
        if edges.size == 0:
            return (
                (
                    0.0,
                    {
                        "bbox_lo": None,
                        "bbox_hi": None,
                        "edges_total": 0,
                        "edges_intersected": 0,
                        "clipped": False,
                    },
                )
                if return_details
                else 0.0
            )
        seg = nodes[edges[:, 1]] - nodes[edges[:, 0]]
        L = float(np.linalg.norm(seg, axis=1).sum())
        if not return_details:
            return L
        return L, {
            "bbox_lo": None,
            "bbox_hi": None,
            "edges_total": int(edges.shape[0]),
            "edges_intersected": int(edges.shape[0]),
            "clipped": False,
        }

    lo, hi = _parse_bbox(bbox)

    total = 0.0
    n_hit = 0
    for u, v in edges:
        ell = _segment_aabb_clip_length(nodes[u], nodes[v], lo, hi)
        if ell > 0.0:
            total += ell
            n_hit += 1

    if not return_details:
        return float(total)

    return float(total), {
        "bbox_lo": lo,
        "bbox_hi": hi,
        "edges_total": int(edges.shape[0]),
        "edges_intersected": n_hit,
        "clipped": True,
    }
