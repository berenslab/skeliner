import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from importlib import metadata as _metadata
from typing import Any

import numpy as np

from ._backend import backend, kernels
from ._core import (
    _bridge_gaps,
    _build_mst,
    _detect_soma,
    _merge_near_soma_nodes,
    _prune_neurites,
)
from ._state import rebuild_vert2node, vertex_lut
from ._threads import threads
from .dataclass import Mesh, MeshComponents, Skeleton, Soma, mesh_args

_SKELINER_VERSION = _metadata.version("skeliner")


__all__ = [
    "skeletonize",
]

# the Python path's igraph binning, under the names `dev/` and the tests
# use; imported on first use, as it imports igraph
_PYTHON_BINNING = {
    "_surface_graph": "surface_graph",
    "_dist_vec_for_component": "dist_vec_for_component",
    "_neighbour_groups": "neighbour_groups",
    "_split_branch_band": "split_branch_band",
    "_bin_one_component": "bin_one_component",
    "_bin_geodesic_shells": "bin_geodesic_shells",
}


def __getattr__(name: str):
    if name in _PYTHON_BINNING:
        from .python import skeletonize as python_skeletonize

        return getattr(python_skeletonize, _PYTHON_BINNING[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _jsonable(v):
    """Coerce a parameter value to something ``json.dumps`` accepts.

    ``to_swc`` writes ``meta`` as a single ``# meta {...}`` line with a plain
    ``json.dumps`` and no fallback encoder, so a numpy scalar or array left in
    here would raise on *export* — long after the run that recorded it.
    ``soma_seed_point`` is the reachable case.
    """
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _skel_meta(track: str, unit: str, id, params: dict, mesh=None) -> dict:
    """Provenance for one skeletonize run.

    *track* and *params* are recorded by the assembler that ran, not by the
    dispatcher, so what lands is what actually took effect.  The preprocessing
    track ignores most of :func:`skeletonize`'s arguments — soma detection,
    gap bridging and pruning all happen upstream — and copying them in would
    describe a run that did not happen.

    Without this a saved skeleton cannot say which track produced it, and the
    two tracks disagree about the same cell by construction.  It is written
    into every export, so it cannot be added retroactively.

    ``mesh`` records the surface ``node2verts`` indexes into.  A skeleton is
    only meaningful beside that mesh — the bins name vertex ids, and every
    radius was measured over them — but nothing in the vertex ids themselves
    says which mesh they belong to.  Pairing a skeleton with a *larger*
    unrelated mesh keeps every index in range, so bins resolve to real faces
    and read as a correct answer.  Counts settle it: they are exact, they
    survive the ``.obj``/``.ply`` round trip that perturbs coordinates in the
    ninth decimal, and they cost two integers.
    """
    meta = {
        "skeliner_version": _SKELINER_VERSION,
        "skeletonized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "unit": unit,
        "id": _jsonable(id),
        "track": track,
        "params": {k: _jsonable(v) for k, v in params.items()},
    }
    if mesh is not None:
        meta["mesh"] = {
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
        }
    return meta


# -----------------------------------------------------------------------------
#  Verbose timing helper
# -----------------------------------------------------------------------------


@contextmanager
def _timed(label: str, *, verbose: bool):
    """Context manager printing  ``↳  label … N.NN s``.

    Use the yielded ``log()`` callback for sub-messages::

        with _timed("stage", verbose=True) as log:
            log("detail 1")
    """
    if not verbose:
        yield lambda *_: None
        return

    PAD = 47
    print(f" {label:<{PAD}} …", end="", flush=True)
    t0 = time.perf_counter()
    _msgs: list[str] = []

    def log(msg: str) -> None:
        _msgs.append(str(msg))

    try:
        yield log
    finally:
        dt = time.perf_counter() - t0
        print(f" {dt:.2f} s")
        for m in _msgs:
            print(f"      └─ {m}")


def _no_stage(_label: str):
    """Report nothing — the default when a caller wants no progress."""
    return nullcontext()


def _recorded(tag: str, into: list[tuple[str, float]]):
    """A stage reporter that keeps each stage's label and time in *into*, to
    print once the stage's work is all done (:func:`_print_recorded`)."""

    @contextmanager
    def stage(label: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            into.append((f"↳  {tag}{label}", time.perf_counter() - t0))

    return stage


def _print_recorded(stages: list[tuple[str, float]]) -> None:
    """Print stages as :func:`_timed` prints them."""
    for label, dt in stages:
        print(f" {label:<47} … {dt:.2f} s")


# -----------------------------------------------------------------------------
#  Graph helpers
# -----------------------------------------------------------------------------


def _surface_graph_of(mesh: Mesh):
    """The active path's ``SurfaceGraph`` of *mesh*, built from the faces.
    Its ``edges`` and ``lengths`` are the mesh's ``edges_unique`` and
    ``edges_unique_length``, value for value, without trimesh computing
    them."""
    return kernels.skeletonize.SurfaceGraph.from_mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _edges_from_mesh(
    edges_unique: np.ndarray,  # (E, 2) int64
    v2n: dict[int, int],  # mesh-vertex id -> skeleton node id
    n_mesh_verts: int,
) -> np.ndarray:
    """
    Vectorised remap of mesh edges -> skeleton edges.
    """
    lut = vertex_lut(v2n, n_mesh_verts)
    return kernels.skeletonize.edges_between(
        np.asarray(edges_unique, dtype=np.int64).reshape(-1, 2), lut
    )


def _edges_from_bins(
    edges_unique: np.ndarray, node2verts: list[np.ndarray], n_mesh_verts: int
) -> np.ndarray:
    """:func:`_edges_from_mesh` from the nodes' vertex lists, which share no
    vertex, without a ``vert2node`` dict."""
    owner = np.full(n_mesh_verts, -1, dtype=np.int64)
    if node2verts:
        lens = [len(v) for v in node2verts]
        owner[np.concatenate(node2verts).astype(np.int64)] = np.repeat(
            np.arange(len(node2verts), dtype=np.int64), lens
        )
    return kernels.skeletonize.edges_between(
        np.asarray(edges_unique, dtype=np.int64).reshape(-1, 2), owner
    )


def _extreme_vertex(mesh: Mesh, axis: str = "z", mode: str = "min") -> int:
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


def _make_nodes(
    all_shells: list[list[np.ndarray]],
    vertices: np.ndarray,
    *,
    radius_estimators: list[str],
    merge_nested: bool = True,
    merge_kwargs: dict | None = None,
    with_vert2node: bool = True,
) -> tuple[
    np.ndarray,  # nodes_arr
    dict[str, np.ndarray],  # radii_dict
    list[np.ndarray],  # node2verts
    dict[int, int] | None,  # vert2node, None without with_vert2node
]:
    """
    Convert geodesic bins into skeleton nodes **and** run the optional
    nested-node merge, both on the active path.

    Parameters
    ----------
    all_shells
        Shells of bins, as the binning returns them.
    vertices
        `mesh.vertices` as `(N,3) float64`.
    radius_estimators
        Names understood by `_estimate_radius()`; the first one weighs the
        merge.
    merge_nested
        Whether to collapse nodes nested inside bigger ones afterwards.
    merge_kwargs
        Passed straight through to ``merge_nested_nodes`` (``inside_frac``,
        ``keep_root``, ``tol``).
    with_vert2node
        Whether to build ``vert2node``; None in its place otherwise.

    Returns
    -------
    nodes_arr, radii_dict, node2verts, vert2node
    """
    bins = [[np.asarray(b, dtype=np.int64) for b in band] for band in all_shells]
    node2verts = [b for band in bins for b in band]
    if not node2verts:
        empty = {k: np.asarray(np.array([])) for k in radius_estimators}
        return (
            np.asarray([], dtype=np.float64),
            empty,
            [],
            {} if with_vert2node else None,
        )
    nodes_arr, radii = kernels.skeletonize.skeleton_nodes(
        [[b.tolist() for b in band] for band in bins],
        vertices,
        list(radius_estimators),
    )
    radii_dict = dict(zip(radius_estimators, radii))
    if merge_nested:
        keep, node2verts, _ = kernels.skeletonize.merge_nested_nodes(
            nodes_arr,
            radii_dict[radius_estimators[0]],
            node2verts,
            **(merge_kwargs or {}),
        )
        nodes_arr = nodes_arr[keep]
        radii_dict = {k: np.asanyarray(v[keep]) for k, v in radii_dict.items()}
    if not with_vert2node:
        return nodes_arr, radii_dict, node2verts, None
    vert2node = rebuild_vert2node(node2verts) or {}
    return nodes_arr, radii_dict, node2verts, vert2node


def _branch_donations(
    nodes_arr: np.ndarray,
    adj: dict[int, set[int]],
    degree: np.ndarray,
    node2verts: list[np.ndarray],
    mesh_vertices: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """For each branch node's bin, which vertices face which neighbour: each
    vertex goes to the neighbour its offset projects onto most.  numpy's on
    both paths, as the projection goes through BLAS."""
    donated: dict[tuple[int, int], list[int]] = {}
    for b in range(len(nodes_arr)):
        if degree[b] < 3:
            continue
        bv = np.asarray(node2verts[b], dtype=np.int64)
        if len(bv) < 3:
            continue
        nbrs = sorted(adj[b])
        dirs = nodes_arr[nbrs] - nodes_arr[b]
        ln = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = np.divide(dirs, ln, out=np.zeros_like(dirs), where=ln > 1e-6)
        proj = (mesh_vertices[bv] - nodes_arr[b]) @ dirs.T
        best = np.argmax(proj, axis=1)
        for k, nb in enumerate(nbrs):
            donated[(b, nb)] = bv[best == k].tolist()
    return donated


def _perpendicular_rebin(
    nodes_arr: np.ndarray,
    edges_mst: np.ndarray,
    node2verts: list[np.ndarray],
    mesh_vertices: np.ndarray,
    radius_estimators: list[str],
    radii_dict: dict[str, np.ndarray],
) -> tuple[list[list[np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """Re-assign vertices along skeleton chains for perpendicular
    bins.

    For every degree-2 path (chain) between branch points / tips,
    project all chain vertices onto the piecewise-linear skeleton
    path and assign each vertex to the nearest node by arc-length
    distance (radius-weighted so thick nodes claim proportionally
    more territory): ``rebin_chains`` on the active path, after the
    branch nodes' donations.

    Returns new shells (one band per node, one bin per band) and the
    re-assigned vertices with their distances to the centreline, as
    ``(vids, dists)`` (:func:`_as_map`).
    """
    n_skel = len(nodes_arr)
    if n_skel < 2 or edges_mst.size == 0:
        return [[n2v] for n2v in node2verts], _no_centreline()
    adj: dict[int, set[int]] = {i: set() for i in range(n_skel)}
    for a, b in edges_mst:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    degree = np.array([len(adj[i]) for i in range(n_skel)])
    donated = _branch_donations(nodes_arr, adj, degree, node2verts, mesh_vertices)
    out, vids, dists = kernels.skeletonize.rebin_chains(
        np.asarray(nodes_arr, dtype=np.float64),
        np.asarray(edges_mst, dtype=np.int64).reshape(-1, 2),
        [np.asarray(v, dtype=np.int64).tolist() for v in node2verts],
        mesh_vertices,
        np.asarray(radii_dict[radius_estimators[0]], dtype=np.float64),
        [(b, nb, v) for (b, nb), v in donated.items()],
    )
    shells = [[v] for v in out if len(v)]
    return shells, _as_map(vids, dists)


def _no_centreline() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)


def _as_map(vids: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``vids`` → ``vals`` as ``dict(zip(vids, vals))`` would hold them, in
    two arrays: each id once, where it first appears, with its last value."""
    vids = np.asarray(vids, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    if vids.size == 0 or np.bincount(vids).max() <= 1:
        return vids, vals
    m = dict(zip(vids.tolist(), vals.tolist()))
    return np.fromiter(m, np.int64, len(m)), np.fromiter(m.values(), np.float64, len(m))


def _skeletonize_component(
    mesh: Mesh,
    sgraph,
    vert_ids: np.ndarray,
    *,
    seed_vid: int | np.ndarray,
    radius_estimators: list[str],
    step_size: float | None = None,
    target_shell_count: int = 500,
    min_shell_vertices: int = 6,
    max_shell_width_factor: float = 50.0,
    second_pass: bool = True,
    stage=_no_stage,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
]:
    """Bin, build nodes, and extract edges for one component.

    The preprocessing track's work on one connected vertex set: geodesic
    shells (``sgraph.bin_component``), their nodes (:func:`_make_nodes`,
    no nested merge) and edges (:func:`_edges_from_mesh`); then the second
    pass — re-binning perpendicular to the spanning tree's centreline
    (:func:`_perpendicular_rebin`), the shells cleaned up on ``sgraph`` —
    and the nodes, centreline radii and edges again.

    Parameters
    ----------
    mesh
        The neuron mesh (for ``vertices``).
    sgraph
        The active path's surface graph of the mesh
        (:func:`_surface_graph_of`).
    vert_ids
        1-D int64 array — vertex IDs for this component, ascending.
    seed_vid
        Where to start the geodesic.  A single ``int`` or a
        1-D array for multi-source seeding.
    radius_estimators
        Passed to :func:`_make_nodes`.

    Returns
    -------
    nodes_arr, radii_dict, node2verts, edges_arr, vid_cl_dist

    ``vid_cl_dist`` is ``(vids, dists)``: each mesh vertex the second pass
    re-assigned and its perpendicular distance to the centreline, empty
    when the second pass does not run.  It is
    returned rather than consumed because it is the only input to the
    ``centerline`` radius that cannot be recovered from the vertices alone;
    keeping it lets that radius be recomputed after a bin is edited.
    """
    mesh_vertices = mesh.vertices.view(np.ndarray)
    e_m = float(sgraph.lengths.mean())
    vid_cl_dist = _no_centreline()
    verts = np.asarray(vert_ids, dtype=np.int64)

    with stage("bin vertices by geodesic distance"):
        all_shells = sgraph.bin_component(
            verts,
            np.atleast_1d(np.asarray(seed_vid, dtype=np.int64)),
            e_m,
            step_size,
            target_shell_count,
            min_shell_vertices,
            max_shell_width_factor,
        )

    with stage("compute bin centroids and radii"):
        nodes_arr, radii_dict, node2verts, _ = _make_nodes(
            all_shells,
            mesh_vertices,
            radius_estimators=radius_estimators,
            merge_nested=False,
            with_vert2node=False,
        )

    with stage("derive edges from mesh edges"):
        edges_arr = _edges_from_bins(sgraph.edges, node2verts, len(mesh.vertices))

    if second_pass and len(nodes_arr) > 1 and edges_arr.size:
        with stage("re-bin perpendicular to the centreline"):
            edges_mst = _build_mst(nodes_arr, edges_arr)
            rebin_shells, vid_cl_dist = _perpendicular_rebin(
                nodes_arr,
                edges_mst,
                node2verts,
                mesh_vertices,
                radius_estimators,
                radii_dict,
            )

        with stage("re-check rings, merge non-rings"):
            final_shells = sgraph.recheck_rings(rebin_shells, verts)

        with stage("reunite bins split across the surface"):
            final_shells = sgraph.reunite_bins(final_shells)

        with stage("split bins that wrap a branch point"):
            final_shells = sgraph.split_branch_bins(final_shells)

        with stage("recompute centroids and centerline radii"):
            nodes_arr, radii_dict, node2verts, _ = _make_nodes(
                final_shells,
                mesh_vertices,
                radius_estimators=radius_estimators,
                merge_nested=False,
                with_vert2node=False,
            )
            ptr = np.zeros(len(node2verts) + 1, dtype=np.int64)
            ptr[1:] = np.cumsum([len(v) for v in node2verts])
            radii_dict["centerline"] = kernels.skeletonize.centerline_radii(
                ptr,
                np.concatenate(node2verts).astype(np.int64)
                if node2verts
                else np.empty(0, dtype=np.int64),
                *vid_cl_dist,
                len(mesh_vertices),
            )

        with stage("re-derive edges from mesh edges"):
            edges_arr = _edges_from_bins(sgraph.edges, node2verts, len(mesh.vertices))

    return nodes_arr, radii_dict, node2verts, edges_arr, vid_cl_dist


# -----------------------------------------------------------------------------
#  Preprocessing track helpers
# -----------------------------------------------------------------------------


def _concat_radii(
    parts: list[dict[str, np.ndarray]],
    radius_estimators: list[str],
    *,
    lead_value: float | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate per-neurite radii, keeping keys beyond the estimators.

    The second binning pass writes a ``centerline`` entry that is not in
    ``radius_estimators``; collecting only the estimators silently drops it.

    A neurite too small for the second pass has no ``centerline`` of its
    own, so its nodes fall back to the first estimator — that keeps every
    radius array the same length as ``nodes`` and finite, which exporters
    and :meth:`Skeleton.recommend_radius` both rely on.

    Parameters
    ----------
    parts : list of dict
        One radii dict per sub-skeleton, in node order.
    radius_estimators : list of str
        Keys guaranteed to be present on every sub-skeleton.
    lead_value : float or None
        Radius for a single node prepended before the parts (the soma).
    """
    keys = list(radius_estimators)
    for radii in parts:
        for k in radii:
            if k not in keys:
                keys.append(k)
    fallback = radius_estimators[0]

    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    if lead_value is not None:
        for k in keys:
            out[k].append(np.array([lead_value], dtype=np.float64))
    for radii in parts:
        for k in keys:
            v = radii.get(k, radii[fallback])
            out[k].append(np.asarray(v, dtype=np.float64))
    return {
        k: (np.concatenate(v) if v else np.empty(0, dtype=np.float64))
        for k, v in out.items()
    }


def _stitch_to_soma(
    sub_skeletons: list[
        tuple[
            np.ndarray,
            dict[str, np.ndarray],
            list[np.ndarray],
            np.ndarray,
            tuple[np.ndarray, np.ndarray],
        ]
    ],
    soma: Soma,
    radius_estimators: list[str],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    np.ndarray,
]:
    """Combine per-neurite sub-skeletons with soma node 0.

    Each sub-skeleton's node IDs are offset so they sit after
    the soma and all earlier sub-skeletons.  One synthetic
    stem edge ``(0, root_i)`` is added per neurite, where
    ``root_i`` is the node closest to the soma centre.

    Returns
    -------
    nodes, radii, node2verts, vert2node, edges
    """
    r_soma = soma.spherical_radius

    # -- node 0 = soma --
    soma_verts_arr = soma.verts if soma.verts is not None else np.empty(0, np.int64)
    all_nodes = [soma.center.reshape(1, 3)]
    radii_parts: list[dict[str, np.ndarray]] = []
    all_n2v: list[np.ndarray] = [soma_verts_arr]
    all_edges: list[np.ndarray] = []

    offset = 1  # node 0 is soma
    for nodes, radii, n2v, edges, _cl in sub_skeletons:
        if len(nodes) == 0:
            continue

        all_nodes.append(nodes)
        radii_parts.append(radii)
        all_n2v.extend(n2v)

        # offset edge endpoints
        all_edges.append(edges + offset)

        # stem edge: soma → closest node
        dists = np.linalg.norm(nodes - soma.center, axis=1)
        root = int(np.argmin(dists)) + offset
        all_edges.append(np.array([[0, root]], dtype=np.int64))

        offset += len(nodes)

    nodes_arr = np.vstack(all_nodes) if all_nodes else np.empty((0, 3), np.float64)
    radii_dict = _concat_radii(radii_parts, radius_estimators, lead_value=r_soma)
    edges_arr = np.vstack(all_edges) if all_edges else np.empty((0, 2), np.int64)
    edges_arr = np.sort(edges_arr, axis=1)
    edges_arr = np.unique(edges_arr, axis=0)

    vert2node = rebuild_vert2node(all_n2v) or {}

    return (
        nodes_arr,
        radii_dict,
        all_n2v,
        vert2node,
        edges_arr,
    )


def _pick_neurite_seed(
    neurite_verts: np.ndarray,
    mesh_vertices: np.ndarray,
    soma: Soma | None,
) -> int | np.ndarray:
    """Choose a geodesic seed for one neurite.

    With soma: return the full boundary ring (all neurite
    vertices shared with the soma surface).  Multi-source
    seeding from the ring produces perpendicular shells.
    Without soma: return the single vertex closest to the
    neurite centroid.
    """
    if soma is not None and soma.verts is not None:
        boundary = np.intersect1d(neurite_verts, soma.verts)
        if boundary.size > 1:
            return boundary
        if boundary.size == 1:
            return int(boundary[0])
        # no shared verts — fall back to nearest
        dists = np.linalg.norm(
            mesh_vertices[neurite_verts] - soma.center,
            axis=1,
        )
        return int(neurite_verts[np.argmin(dists)])

    # no soma — seed at centroid of the neurite
    centroid = mesh_vertices[neurite_verts].mean(axis=0)
    dists = np.linalg.norm(mesh_vertices[neurite_verts] - centroid, axis=1)
    return int(neurite_verts[np.argmin(dists)])


def _concatenate_sub_skeletons(
    sub_skeletons: list[
        tuple[
            np.ndarray,
            dict[str, np.ndarray],
            list[np.ndarray],
            np.ndarray,
            tuple[np.ndarray, np.ndarray],
        ]
    ],
    radius_estimators: list[str],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[np.ndarray],
    dict[int, int],
    np.ndarray,
]:
    """Join per-neurite sub-skeletons without a soma node.

    Unlike :func:`_stitch_to_soma`, no node 0 is prepended
    and no stem edges are added.  The result is a forest
    (one tree per neurite).
    """
    all_nodes: list[np.ndarray] = []
    radii_parts: list[dict[str, np.ndarray]] = []
    all_n2v: list[np.ndarray] = []
    all_edges: list[np.ndarray] = []

    offset = 0
    for nodes, radii, n2v, edges, _cl in sub_skeletons:
        if len(nodes) == 0:
            continue
        all_nodes.append(nodes)
        radii_parts.append(radii)
        all_n2v.extend(n2v)
        if edges.size:
            all_edges.append(edges + offset)
        offset += len(nodes)

    nodes_arr = np.vstack(all_nodes) if all_nodes else np.empty((0, 3), np.float64)
    radii_dict = _concat_radii(radii_parts, radius_estimators)
    edges_arr = np.vstack(all_edges) if all_edges else np.empty((0, 2), np.int64)
    if edges_arr.size:
        edges_arr = np.sort(edges_arr, axis=1)
        edges_arr = np.unique(edges_arr, axis=0)

    vert2node = rebuild_vert2node(all_n2v) or {}

    return (
        nodes_arr,
        radii_dict,
        all_n2v,
        vert2node,
        edges_arr,
    )


def _skeletonize_preproc(
    mesh: Mesh,
    components: MeshComponents,
    *,
    radius_estimators: list[str],
    geodesic_step_size: float | None,
    geodesic_shell_count: int,
    min_shell_vertices: int,
    max_shell_width_factor: float,
    soma_init_guess_axis: str,
    soma_init_guess_mode: str,
    unit: str,
    id: str | int | None,
    verbose: bool,
    second_pass: bool = True,
) -> Skeleton:
    """Preprocessing track: per-neurite skeletonization.

    Each neurite in *components* is skeletonized via
    :func:`_skeletonize_component`, then all sub-skeletons
    are grafted onto the precomputed soma.

    Skips: soma detection, gap bridging, neurite pruning,
    near-soma collapse, nested-node merge, elongated-shell
    splitting (all unnecessary on clean preprocessed
    neurites).
    """
    # Snapshot the arguments before anything else is bound, so the recorded
    # parameters cannot drift from the signature the way a hand-kept list
    # would.  A stale provenance record is worse than none.
    _params = {
        k: v
        for k, v in locals().items()
        if k not in ("mesh", "components", "unit", "id", "verbose")
    }

    soma = components.soma
    has_soma = soma is not None

    if verbose:
        _global_start = time.perf_counter()
        soma_tag = "with soma" if has_soma else "no soma"
        print(
            f"[skeliner] skeletonize from preprocessed mesh components "
            f"({len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} faces, "
            f"{len(components.neurites)} neurites, "
            f"{soma_tag})"
        )

    with _timed("↳  build surface graph", verbose=verbose):
        sgraph = _surface_graph_of(mesh)

    mesh_vertices = mesh.vertices.view(np.ndarray)

    # Each stage of the per-neurite work reports itself as a top-level
    # step, the way `_skeletonize_direct` does.  Wrapping the loop in one
    # `_timed` instead would report a single number for the slowest part
    # of the run, and `_timed` holds its sub-messages until the block
    # ends, so nothing would appear while it was actually working.
    n_neurites = len(components.neurites)

    def one(i: int, stage):
        face_idx = components.neurites[i]
        neurite_verts = np.unique(mesh.faces[face_idx].ravel()).astype(np.int64)
        seed_vid = _pick_neurite_seed(
            neurite_verts,
            mesh_vertices,
            soma,
        )
        sub = _skeletonize_component(
            mesh,
            sgraph,
            neurite_verts,
            seed_vid=seed_vid,
            radius_estimators=radius_estimators,
            step_size=geodesic_step_size,
            target_shell_count=geodesic_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=(max_shell_width_factor),
            second_pass=second_pass,
            stage=stage,
        )
        return len(neurite_verts), sub

    def made(i: int, n_verts: int, sub) -> None:
        if verbose:
            print(
                f"      └─ neurite {i}: {n_verts:,} verts → {sub[0].shape[0]:,} nodes"
            )

    tags = [f"neurite {i}: " if n_neurites > 1 else "" for i in range(n_neurites)]
    # The neurites are independent and the Rust path's kernels let go of the
    # GIL, so there they run at once, each on a thread; a neurite's stages
    # then print when all are done, in order.
    workers = min(n_neurites, threads.get()) if backend.get() == "rust" else 1
    sub_skeletons = []
    if workers > 1:
        reports: list[list[tuple[str, float]]] = [[] for _ in range(n_neurites)]
        with ThreadPoolExecutor(workers) as pool:
            done = list(
                pool.map(
                    lambda i: one(i, _recorded(tags[i], reports[i])), range(n_neurites)
                )
            )
        for i, (n_verts, sub) in enumerate(done):
            if verbose:
                _print_recorded(reports[i])
            made(i, n_verts, sub)
            sub_skeletons.append(sub)
    else:
        for i in range(n_neurites):

            def stage(label, _tag=tags[i]):
                return _timed(f"↳  {_tag}{label}", verbose=verbose)

            n_verts, sub = one(i, stage)
            made(i, n_verts, sub)
            sub_skeletons.append(sub)

    # -- assemble skeleton -----------------------------------------
    if has_soma:
        with _timed(
            "↳  stitch neurites to soma",
            verbose=verbose,
        ):
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_arr,
            ) = _stitch_to_soma(sub_skeletons, soma, radius_estimators)
    else:
        with _timed(
            "↳  concatenate neurites",
            verbose=verbose,
        ):
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_arr,
            ) = _concatenate_sub_skeletons(sub_skeletons, radius_estimators)

    # -- keep the per-vertex centreline distances --
    # The `centerline` radius is an aggregate of these, so keeping them is
    # what lets it be recomputed when a bin changes; every other radius is
    # already recoverable from the vertices alone.  Vertex ids are global,
    # so the per-neurite maps merge without any offsetting.
    cl_vids = np.concatenate(
        [np.empty(0, np.int64)] + [cl[0] for *_rest, cl in sub_skeletons]
    )
    cl_dists = np.concatenate(
        [np.empty(0, np.float64)] + [cl[1] for *_rest, cl in sub_skeletons]
    )

    # -- global MST --
    with _timed(
        "↳  build global minimum-spanning tree",
        verbose=verbose,
    ):
        if len(nodes_arr) > 1 and edges_arr.size:
            edges_mst = _build_mst(nodes_arr, edges_arr)
        else:
            edges_mst = edges_arr

    if verbose:
        total = time.perf_counter() - _global_start
        print(f"{'TOTAL':<49}… {total:.2f} s")
        print(f"({len(nodes_arr):,} nodes, {edges_mst.shape[0]:,} edges)")

    ntype = np.zeros(len(nodes_arr), np.int8)
    if has_soma:
        ntype[0] = 1
    elif len(nodes_arr):
        ntype[0] = -1
        # pick a deterministic root like the direct track
        root_vid = _extreme_vertex(
            mesh,
            axis=soma_init_guess_axis,
            mode=soma_init_guess_mode,
        )
        if root_vid in vert2node:
            root_nid = vert2node[root_vid]
        else:
            # extreme vertex not in any neurite — pick
            # the closest skeleton node instead
            dists = np.linalg.norm(
                nodes_arr - mesh_vertices[root_vid],
                axis=1,
            )
            root_nid = int(np.argmin(dists))
        if root_nid != 0:
            # swap node 0 and root_nid
            nodes_arr[[0, root_nid]] = nodes_arr[[root_nid, 0]]
            for k in radii_dict:
                radii_dict[k][[0, root_nid]] = radii_dict[k][[root_nid, 0]]
            node2verts[0], node2verts[root_nid] = (
                node2verts[root_nid],
                node2verts[0],
            )
            m0 = edges_mst == 0
            mi = edges_mst == root_nid
            edges_mst[m0] = root_nid
            edges_mst[mi] = 0
            edges_mst = np.sort(edges_mst, axis=1)
            vert2node = rebuild_vert2node(node2verts)
        # placeholder soma at the root
        r0 = float(list(radii_dict.values())[0][0])
        soma = Soma.from_sphere(nodes_arr[0], r0, verts=None)

    # -- carry the neurites' hand-given identity onto the skeleton --
    # Read through `vert2node` rather than by re-deriving the node ranges:
    # the assembly lays sub-skeletons down in order, but skips any that came
    # out empty, so an index into `components.neurites` is not an index into
    # the node blocks.  Ownership is exact and needs no such assumption.
    #
    # Node 0 is excluded throughout: under the >=2-of-3 face rule a face at
    # the soma junction carries soma vertices, so a neurite's vertex set
    # reaches node 0, and stamping it would retype the soma as a dendrite.
    #
    # Three things are recorded, because `ntype` alone loses information:
    # it collapses "dendrite 0" and "dendrite 1" into the same code 3, and
    # SWC has no field for a name.  The labels go in `meta`, which both the
    # npz and the SWC header carry, and the per-node owner goes in `extra`,
    # which is what lets a name be resolved back to nodes.
    skel_meta = _skel_meta("preproc", unit, id, _params, mesh)
    skel_extra: dict[str, Any] = (
        {
            "cl_dist_vids": np.asarray(cl_vids, dtype=np.int64),
            "cl_dist_vals": np.asarray(cl_dists, dtype=np.float64),
        }
        if cl_vids.size
        else {}
    )

    if components.neurites.named:
        node2neurite = np.full(len(nodes_arr), -1, dtype=np.int32)
        owner = vertex_lut(vert2node, len(mesh.vertices))
        for ni, (face_idx, code) in enumerate(
            zip(components.neurites.components, components.neurites.swc_types)
        ):
            owned = np.unique(owner[np.asarray(mesh.faces)[face_idx].ravel()])
            owned = owned[owned > 0]
            if not owned.size:
                continue
            node2neurite[owned] = ni
            if code:
                ntype[owned] = code
        skel_meta["neurite_labels"] = list(components.neurites.labels)
        skel_meta["neurite_swc_types"] = [int(c) for c in components.neurites.swc_types]
        skel_extra["node2neurite"] = node2neurite

    return Skeleton(
        nodes=nodes_arr,
        radii=radii_dict,
        edges=edges_mst,
        ntype=ntype,
        soma=soma,
        node2verts=node2verts,
        vert2node=vert2node,
        meta=skel_meta,
        extra=skel_extra,
    )


# -----------------------------------------------------------------------------
#  Skeletonization Public API
# -----------------------------------------------------------------------------


def _skeletonize_direct(
    mesh: Mesh,
    # --- radius estimation ---
    radius_estimators: list[str] = ["median", "mean", "trim"],
    # --- soma detection ---
    detect_soma: bool = True,
    soma_seed_point: np.ndarray | list | tuple | None = None,
    soma_radius_percentile_threshold: float = 99.9,
    soma_radius_distance_factor: float = 4,
    soma_min_nodes: int = 3,
    # -- soma seed heuristic for the geodesic origin --
    soma_init_guess: str | None = None,  # "nucleus" | "<axis>-<mode>"
    soma_init_guess_axis: str = "z",  # legacy, when soma_init_guess is None
    soma_init_guess_mode: str = "min",  # legacy, when soma_init_guess is None
    # --- geodesic binning ---
    geodesic_step_size: float | None = None,
    geodesic_shell_count: int = 1000,  # higher = more bins, smaller bin size
    min_shell_vertices: int = 6,
    max_shell_width_factor: int = 50,
    split_elongated_shells: bool = False,
    split_aspect_thr: float = 3.0,  # λ1 / λ2
    split_min_shell_vertices: int = 15,
    split_max_vertices_per_slice: int | None = None,
    merge_nodes_overlap_fraction: float = 0.8,  # merge nested nodes if inside_frac ≥ this
    # --- bridging disconnected patches ---
    bridge_gaps: bool = True,
    bridge_max_factor: float | None = None,
    bridge_recalc_after: int | None = None,
    # -- post‑processing --
    # --- collapse soma-like nodes ---
    collapse_soma: bool = True,
    collapse_soma_dist_factor: float = 1.2,
    collapse_soma_radius_factor: float = 0.2,
    # --- prune tiny neurites ---
    prune_tiny_neurites: bool = True,
    prune_tip_extent_factor: float = 1.2,  # tip twigs (<–× r_soma)
    prune_stem_extent_factor: float = 3.0,  # stems touching soma
    prune_drop_single_node_branches: bool = True,
    # --- misc ---
    unit: str = "nm",
    id: str | int | None = None,
    verbose: bool = False,
    postprocess: bool = True,
) -> Skeleton:
    """Direct track: full pipeline from raw mesh.

    Stages: surface graph, geodesic binning, node creation,
    soma detection, edge mapping, near-soma collapse, gap
    bridging, MST, neurite pruning.  The graph, the binning and the nodes
    run on the active path (``sk.backend``), but for the legacy
    ``split_elongated_shells``, which bins on the Python path's igraph graph
    whichever path is active.
    """
    # See _skeletonize_preproc: snapshot the arguments before anything else
    # is bound, so the record cannot drift from the signature.
    _params = {
        k: v for k, v in locals().items() if k not in ("mesh", "unit", "id", "verbose")
    }

    # ------------------------------------------------------------------
    #  Direct track: helpers for verbose timing
    # ------------------------------------------------------------------
    if verbose:
        _global_start = time.perf_counter()
        # Named for what it consumes, and matching how the other track names
        # itself: which of the two ran decides which parameters were
        # consulted, so a header saying only "starting" cannot answer it.
        print(
            f"[skeliner] skeletonize from raw mesh "
            f"({len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} faces)"
        )
        soma_ms = 0.0  # soma detection time
        post_ms = 0.0  # post-processing time

    run_mst = True
    if not postprocess:
        detect_soma = False
        collapse_soma = False
        bridge_gaps = False
        prune_tiny_neurites = False
        run_mst = False

    legacy_split = split_elongated_shells
    if legacy_split:
        from .python import skeletonize as python_skeletonize

    # 0. build surface graph -----------------------------------------------
    with _timed("↳  build surface graph", verbose=verbose):
        gsurf = (
            python_skeletonize.surface_graph(mesh)
            if legacy_split
            else _surface_graph_of(mesh)
        )

    # 1. binning surface vertices by geodesic distance ----------------------------------
    with _timed("↳  bin surface vertices by geodesic distance", verbose=verbose):
        mesh_vertices = mesh.vertices.view(np.ndarray)

        # -- resolve soma_init_guess → soma_seed_point ----
        if soma_seed_point is None and soma_init_guess is not None:
            if soma_init_guess == "nucleus":
                from .pre import find_nucleus_center

                nuc = find_nucleus_center(mesh, verbose=verbose)
                if nuc is not None:
                    soma_seed_point = nuc["center"]
            elif "-" in soma_init_guess:
                ax, md = soma_init_guess.split("-", 1)
                soma_init_guess_axis = ax
                soma_init_guess_mode = md

        # -- pick seed vertex -----------------------------
        if soma_seed_point is not None:
            seed_vid = int(
                np.argmin(
                    np.linalg.norm(
                        mesh_vertices - np.asarray(soma_seed_point),
                        axis=1,
                    )
                )
            )
        else:
            seed_vid = _extreme_vertex(
                mesh,
                axis=soma_init_guess_axis,
                mode=soma_init_guess_mode,
            )

        if legacy_split:
            all_shells = python_skeletonize.bin_geodesic_shells(
                mesh,
                gsurf,
                seed_vid=seed_vid,
                step_size=geodesic_step_size,
                target_shell_count=geodesic_shell_count,
                min_shell_vertices=min_shell_vertices,
                max_shell_width_factor=max_shell_width_factor,
                split_elongated_shells=split_elongated_shells,
                split_aspect_thr=split_aspect_thr,
                split_min_shell_vertices=split_min_shell_vertices,
                split_max_vertices_per_slice=split_max_vertices_per_slice,
            )
        else:
            all_shells = gsurf.bin_mesh(
                seed_vid,
                float(gsurf.lengths.mean()),
                geodesic_step_size,
                geodesic_shell_count,
                min_shell_vertices,
                max_shell_width_factor,
            )

    # 2. create skeleton nodes ------------------------------------------
    with _timed("↳  compute bin centroids and radii", verbose=verbose):
        (nodes_arr, radii_dict, node2verts, vert2node) = _make_nodes(
            all_shells,
            mesh_vertices,
            radius_estimators=radius_estimators,
            merge_nested=True,
            merge_kwargs={
                "inside_frac": merge_nodes_overlap_fraction
            },  # tune `inside_frac`/`keep_root` here if needed
        )

    # 3. soma detection (optional) -----------------------------------
    _t0 = time.perf_counter()
    with _timed("↳  post-skeletonization soma detection", verbose=verbose) as log:
        (
            nodes_arr,
            radii_dict,
            node2verts,
            vert2node,
            soma,
            has_soma,
            _,
        ) = _detect_soma(
            nodes_arr,
            radii_dict,
            node2verts,
            vert2node,
            soma_radius_percentile_threshold=soma_radius_percentile_threshold,
            soma_radius_distance_factor=soma_radius_distance_factor,
            soma_min_nodes=soma_min_nodes,
            detect_soma=detect_soma,
            mesh_vertices=mesh_vertices,
            radius_key=radius_estimators[0],
            log=log,
        )
        soma_ms = time.perf_counter() - _t0

    # 4. edges from mesh connectivity -----------------------------------
    with _timed("↳  map mesh faces to skeleton edges", verbose=verbose):
        edges_arr = _edges_from_bins(
            mesh.edges_unique if legacy_split else gsurf.edges,
            node2verts,
            len(mesh.vertices),
        )

    # 5. collapse soma‑like / fat nodes ---------------------------
    if has_soma and collapse_soma:
        _t0 = time.perf_counter()

        with _timed("↳  merge redundant near-soma nodes", verbose=verbose) as log:
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_arr,
                soma,
                _,
            ) = _merge_near_soma_nodes(
                nodes_arr,
                radii_dict,
                edges_arr,
                node2verts,
                soma=soma,
                radius_key=radius_estimators[0],
                mesh_vertices=mesh_vertices,
                fat_factor=collapse_soma_radius_factor,
                near_factor=collapse_soma_dist_factor,
                log=log,
                # pruning rebuilds it from the final bins
                with_vert2node=not prune_tiny_neurites,
            )

        if verbose:
            post_ms += time.perf_counter() - _t0

    # 6. Connect all components ------------------------------
    if bridge_gaps:
        _t0 = time.perf_counter()
        with _timed("↳  bridge skeleton gaps", verbose=verbose) as log:
            edges_arr = _bridge_gaps(
                nodes_arr,
                edges_arr,
                bridge_max_factor=bridge_max_factor,
                bridge_recalc_after=bridge_recalc_after,
            )
        if verbose:
            post_ms += time.perf_counter() - _t0

    # 7. global minimum-spanning tree ------------------------------------
    if run_mst:
        _t0 = time.perf_counter()
        with _timed("↳  build global minimum-spanning tree", verbose=verbose):
            edges_mst = _build_mst(nodes_arr, edges_arr)
        if verbose:
            post_ms += time.perf_counter() - _t0
    else:
        edges_mst = edges_arr

    # 8. prune tiny sub-trees near the soma
    if has_soma and prune_tiny_neurites:
        _t0 = time.perf_counter()
        with _timed("↳  prune tiny neurites", verbose=verbose) as log:
            (
                nodes_arr,
                radii_dict,
                node2verts,
                vert2node,
                edges_mst,
                soma,
                _,
            ) = _prune_neurites(
                nodes_arr,
                radii_dict,
                node2verts,
                edges_mst,
                soma=soma,
                mesh_vertices=mesh_vertices,
                tip_extent_factor=prune_tip_extent_factor,
                stem_extent_factor=prune_stem_extent_factor,
                drop_single_node_branches=prune_drop_single_node_branches,
                log=log,
            )
        if verbose:
            post_ms += time.perf_counter() - _t0

    if verbose:
        total_ms = time.perf_counter() - _global_start
        core_ms = total_ms - soma_ms - post_ms

        if post_ms > 1e-6:  # at least one optional stage ran
            print(
                f"{'TOTAL (soma + core + post)':<49}"
                f"… {total_ms:.2f} s "
                f"({soma_ms:.2f} + {core_ms:.2f} + {post_ms:.2f})"
            )
            print(f"({len(nodes_arr):,} nodes, {edges_mst.shape[0]:,} edges)")
        else:  # no post-processing at all
            print(
                f"{'TOTAL (soma + core)':<49}"
                f"… {total_ms:.2f} s "
                f"({soma_ms:.2f} + {core_ms:.2f})"
            )

    ntype = np.zeros(len(nodes_arr), np.int8)
    ntype[0] = 1 if has_soma else -1

    return Skeleton(
        nodes=nodes_arr,
        radii=radii_dict,
        edges=edges_mst,
        ntype=ntype,
        soma=soma,
        node2verts=node2verts,
        vert2node=vert2node,
        meta=_skel_meta("direct", unit, id, _params, mesh),
    )


@mesh_args()
def skeletonize(
    mesh: Mesh,
    # --- preprocessing track ---
    components: MeshComponents | None = None,
    # --- radius estimation ---
    radius_estimators: list[str] = ["median", "mean", "trim"],
    # --- soma detection ---
    detect_soma: bool = True,
    soma_seed_point: np.ndarray | list | tuple | None = None,
    soma_radius_percentile_threshold: float = 99.9,
    soma_radius_distance_factor: float = 4,
    soma_min_nodes: int = 3,
    # -- soma seed heuristic for the geodesic origin --
    soma_init_guess: str | None = None,  # "nucleus" | "<axis>-<mode>"
    soma_init_guess_axis: str = "z",  # legacy, when soma_init_guess is None
    soma_init_guess_mode: str = "min",  # legacy, when soma_init_guess is None
    # --- geodesic binning ---
    geodesic_step_size: float | None = None,
    geodesic_shell_count: int = 1000,  # higher = more bins, smaller bin size
    min_shell_vertices: int = 6,
    max_shell_width_factor: int = 50,
    split_elongated_shells: bool = False,
    split_aspect_thr: float = 3.0,  # λ1 / λ2
    split_min_shell_vertices: int = 15,
    split_max_vertices_per_slice: int | None = None,
    merge_nodes_overlap_fraction: float = 0.8,  # merge nested nodes if inside_frac ≥ this
    # --- bridging disconnected patches ---
    bridge_gaps: bool = True,
    bridge_max_factor: float | None = None,
    bridge_recalc_after: int | None = None,
    # -- post‑processing --
    # --- collapse soma-like nodes ---
    collapse_soma: bool = True,
    collapse_soma_dist_factor: float = 1.2,
    collapse_soma_radius_factor: float = 0.2,
    # --- prune tiny neurites ---
    prune_tiny_neurites: bool = True,
    prune_tip_extent_factor: float = 1.2,  # tip twigs (<–× r_soma)
    prune_stem_extent_factor: float = 3.0,  # stems touching soma
    prune_drop_single_node_branches: bool = True,
    # --- misc ---
    unit: str = "nm",
    id: str | int | None = None,
    verbose: bool = False,
    postprocess: bool = True,
    second_pass: bool = True,
) -> Skeleton:
    """Compute a center-line skeleton with radii from a neuronal mesh.

    Two tracks are available, named for what they read and selected by
    the *components* parameter:

    **From a raw mesh** (``components=None``, the default):
      Full pipeline — geodesic binning, post-skel soma detection,
      gap bridging, MST, neurite pruning.  Works on any raw mesh.

    **From preprocessed mesh components**
    (``components=MeshComponents(...)``):
      Per-neurite skeletonization using the soma and neurite
      partition from :func:`~skeliner.pre.preprocess` or
      :func:`~skeliner.pre.break_up_mesh`.  Skips soma detection,
      gap bridging, and pruning (all handled upstream).  Supports
      meshes with or without a soma.

    Which one ran is the first line of the verbose log, because it
    decides which of the parameters below were consulted at all.

    Parameters
    ----------
    mesh : Mesh
        Surface mesh of the neuron.
    components : MeshComponents or None
        If provided, skeletonize from preprocessed mesh components.
        The soma and neurite partition are taken from *components*;
        most other parameters (bridging, pruning, soma detection)
        are accepted and then ignored, so setting one against the
        wrong track is silent rather than refused.
    soma_init_guess : str or None
        Soma-seed heuristic when skeletonizing from a raw mesh.
        ``"nucleus"``
        runs :func:`~skeliner.pre.find_nucleus_center` to locate
        the soma before binning.  ``"<axis>-<mode>"`` (e.g.
        ``"z-min"``) selects an extreme vertex.  ``None`` falls
        back to *soma_init_guess_axis* / *soma_init_guess_mode*.
    verbose : bool, default ``False``
        Print per-stage timing.
    second_pass : bool, default ``True``
        Run the perpendicular re-binning pass: the first-pass bins
        build a rough skeleton, then every degree-2 chain is re-binned
        by projecting its vertices onto that path, and a ``centerline``
        radius is recorded.  Roughly halves median bin tilt.
        **Read only when skeletonizing from preprocessed mesh
        components** — ignored when ``components`` is not given.

    Returns
    -------
    Skeleton
        Acyclic skeleton.  Node 0 is the soma centroid (when
        detected) or a deterministic root vertex.
    """
    # ------------------------------------------------------------------
    #  Dispatch to the appropriate track
    # ------------------------------------------------------------------
    if components is not None:
        return _skeletonize_preproc(
            mesh,
            components,
            radius_estimators=radius_estimators,
            geodesic_step_size=geodesic_step_size,
            geodesic_shell_count=geodesic_shell_count,
            min_shell_vertices=min_shell_vertices,
            max_shell_width_factor=max_shell_width_factor,
            soma_init_guess_axis=soma_init_guess_axis,
            soma_init_guess_mode=soma_init_guess_mode,
            unit=unit,
            id=id,
            verbose=verbose,
            second_pass=second_pass,
        )

    return _skeletonize_direct(
        mesh,
        radius_estimators=radius_estimators,
        detect_soma=detect_soma,
        soma_seed_point=soma_seed_point,
        soma_radius_percentile_threshold=soma_radius_percentile_threshold,
        soma_radius_distance_factor=soma_radius_distance_factor,
        soma_min_nodes=soma_min_nodes,
        soma_init_guess=soma_init_guess,
        soma_init_guess_axis=soma_init_guess_axis,
        soma_init_guess_mode=soma_init_guess_mode,
        geodesic_step_size=geodesic_step_size,
        geodesic_shell_count=geodesic_shell_count,
        min_shell_vertices=min_shell_vertices,
        max_shell_width_factor=max_shell_width_factor,
        split_elongated_shells=split_elongated_shells,
        split_aspect_thr=split_aspect_thr,
        split_min_shell_vertices=split_min_shell_vertices,
        split_max_vertices_per_slice=split_max_vertices_per_slice,
        merge_nodes_overlap_fraction=merge_nodes_overlap_fraction,
        bridge_gaps=bridge_gaps,
        bridge_max_factor=bridge_max_factor,
        bridge_recalc_after=bridge_recalc_after,
        collapse_soma=collapse_soma,
        collapse_soma_dist_factor=collapse_soma_dist_factor,
        collapse_soma_radius_factor=collapse_soma_radius_factor,
        prune_tiny_neurites=prune_tiny_neurites,
        prune_tip_extent_factor=prune_tip_extent_factor,
        prune_stem_extent_factor=prune_stem_extent_factor,
        prune_drop_single_node_branches=prune_drop_single_node_branches,
        unit=unit,
        id=id,
        verbose=verbose,
        postprocess=postprocess,
    )
