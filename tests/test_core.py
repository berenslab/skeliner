"""
Core pipeline smoke-test.

Runs `skeletonize()` on the reference mesh and checks a handful of
topological / numerical invariants so that regressions blow up early.
"""

import re
from pathlib import Path

import igraph as ig
import numpy as np
import pytest
import trimesh

import skeliner as sk
from skeliner import _rs, skeletonize
from skeliner._backend import PATHS
from skeliner._core import _build_mst
from skeliner.dataclass import (
    Discarded,
    MeshComponents,
    Neurites,
    Organelles,
    Soma,
)
from skeliner.io import load_mesh
from skeliner.python import skeletonize as python_skel
from skeliner.skeletonize import (
    _bin_geodesic_shells,
    _edges_from_mesh,
    _extreme_vertex,
    _make_nodes,
    _neighbour_groups,
    _perpendicular_rebin,
    _pick_neurite_seed,
    _split_branch_band,
    _surface_graph,
    _surface_graph_of,
)

from ._parity import assert_parity, both
from .test_rust_core import _dirty_mesh


def _assert_skeleton_valid(skel, *, expect_soma: bool = False):
    """A few cheap invariants that *must* hold for every result."""
    # ----- basic shapes -------------------------------------------------
    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.nodes.shape[0] > 0, "no nodes produced"
    assert (skel.r > 0).all(), "non-positive radii"

    # ----- edges are sorted & acyclic (forest) --------------------------
    assert (skel.edges[:, 0] < skel.edges[:, 1]).all(), "edges not sorted"

    g = skel._igraph()
    n_components = len(g.components())
    # for every forest: |E| = |V| − #components
    assert skel.edges.shape[0] == skel.nodes.shape[0] - n_components

    # ----- node 0 type --------------------------------------------------
    if expect_soma:
        assert skel.ntype[0] == 1, "node 0 should be soma"
    else:
        assert skel.ntype[0] == -1, "node 0 should be root"


@pytest.fixture(scope="session")
def reference_mesh():
    data_dir = Path(__file__).parent / "data"
    mesh_path = data_dir / "60427.obj"
    return load_mesh(mesh_path)


# ----- direct track -------------------------------------------------------


def test_skeletonize_smoke(reference_mesh):
    skel = skeletonize(reference_mesh, verbose=False)
    _assert_skeleton_valid(skel)


def test_skeletonize_soma_init_guess_axis_mode(reference_mesh):
    """soma_init_guess='z-min' should behave like the default."""
    skel = skeletonize(
        reference_mesh,
        soma_init_guess="z-min",
        verbose=False,
    )
    _assert_skeleton_valid(skel)


def _shells_equal(a, b) -> bool:
    return len(a) == len(b) and all(
        len(x) == len(y) and all(np.array_equal(u, v) for u, v in zip(x, y))
        for x, y in zip(a, b)
    )


def _skeleton_fields(skel) -> dict:
    """What a skeleton holds, but its provenance record, for comparing two."""
    soma = skel.soma
    return {
        "nodes": skel.nodes,
        "edges": skel.edges,
        "ntype": skel.ntype,
        "radii": dict(skel.radii),
        "node2verts": list(skel.node2verts),
        "vert2node": dict(skel.vert2node),
        "soma": None if soma is None else [soma.center, soma.axes, soma.R],
        "extra": dict(skel.extra),
    }


def _direct_bins(mesh, seed, step=None):
    """The direct track's shells three ways: the legacy igraph binning with
    its split off, and ``bin_mesh`` on the Rust path and on the Python path."""
    legacy = _bin_geodesic_shells(
        mesh,
        _surface_graph(mesh),
        seed_vid=seed,
        step_size=step,
        target_shell_count=1000,
        min_shell_vertices=6,
        max_shell_width_factor=50,
        split_elongated_shells=False,
    )
    e_m = float(mesh.edges_unique_length.mean())
    rust, python = both(
        lambda: _surface_graph_of(mesh).bin_mesh(seed, e_m, step, 1000, 6, 50)
    )
    return legacy, rust, python


def test_direct_components_come_in_igraphs_order(reference_mesh):
    legacy = [list(c) for c in _surface_graph(reference_mesh).components()]
    rust, python = both(lambda: _surface_graph_of(reference_mesh).components())
    assert len(rust) == 7 and [c.tolist() for c in rust] == legacy
    assert_parity(rust, python)


@pytest.mark.parametrize("path", PATHS)
def test_surface_graph_has_trimeshs_unique_edges_and_lengths(reference_mesh, path):
    """Built from the faces, value for value what trimesh computes, so the
    mean edge and the edge mapping see the same numbers either way."""
    with sk.backend(path):
        for mesh in [reference_mesh, _dirty_mesh()]:
            g = _surface_graph_of(mesh)
            assert np.array_equal(g.edges, mesh.edges_unique)
            assert np.array_equal(g.lengths, mesh.edges_unique_length)
            assert g.lengths.mean() == mesh.edges_unique_length.mean()
    # the dirty mesh's degenerate faces keep their [v, v] edges
    assert (g.edges[:, 0] == g.edges[:, 1]).any()


def test_edges_from_mesh_are_the_distinct_node_pairs_in_row_order():
    rng = np.random.default_rng(0)
    edges = np.sort(rng.integers(0, 300, (2000, 2)), axis=1)
    owned = rng.permutation(300)[:250]
    v2n = {int(v): int(n) for v, n in zip(owned, rng.integers(0, 40, 250))}
    out = _edges_from_mesh(edges, v2n, 300)

    lut = np.full(300, -1)
    lut[list(v2n)] = list(v2n.values())
    pairs = lut[edges]
    pairs = pairs[(pairs >= 0).all(axis=1) & (pairs[:, 0] != pairs[:, 1])]
    assert len(out) > 100
    assert np.array_equal(out, np.unique(np.sort(pairs, axis=1), axis=0))
    assert out.dtype == np.int64 and out.flags.c_contiguous
    assert _edges_from_mesh(edges, {}, 300).shape == (0, 2)


@pytest.mark.parametrize("step_edges", [None, 3.0])
def test_direct_binning_is_the_same_on_both_paths(reference_mesh, step_edges):
    """Auto shell width, and a fixed one of three mean edges; the seed in
    the main component, and in a small one so the main falls back.  The
    legacy igraph binning, its split off, gives the same shells."""
    mesh = reference_mesh
    step = None
    if step_edges is not None:
        step = step_edges * float(mesh.edges_unique_length.mean())
    small = min(_surface_graph(mesh).components(), key=len)
    for seed in [_extreme_vertex(mesh), _extreme_vertex(mesh, "x", "max"), small[0]]:
        legacy, rust, python = _direct_bins(mesh, int(seed), step)
        assert sum(len(band) for band in rust) > 20
        assert_parity(rust, python, f"seed {seed}")
        assert _shells_equal(legacy, rust), seed


def test_direct_binning_is_the_same_on_both_paths_on_a_dirty_mesh():
    mesh = _dirty_mesh()
    legacy, rust, python = _direct_bins(mesh, _extreme_vertex(mesh))
    assert_parity(rust, python)
    assert _shells_equal(legacy, rust)


def test_direct_nodes_and_nested_merge_are_the_same_on_both_paths(reference_mesh):
    mesh = reference_mesh
    e_m = float(mesh.edges_unique_length.mean())
    shells = _surface_graph_of(mesh).bin_mesh(
        _extreme_vertex(mesh), e_m, None, 1000, 6, 50
    )
    rust, python = both(
        _make_nodes,
        shells,
        mesh.vertices.view(np.ndarray),
        radius_estimators=["median", "mean", "trim"],
        merge_nested=True,
        merge_kwargs={"inside_frac": 0.8},
    )
    assert len(rust[0]) < sum(len(band) for band in shells), "nothing merged"
    assert_parity(rust, python)


def test_nested_merge_is_the_same_on_both_paths_on_crowded_nodes():
    rng = np.random.default_rng(3)
    nodes = rng.random((400, 3)) * 50
    radii = rng.random(400) * 8
    radii[[5, 9, 40]] = radii[3]  # ties, visited by index
    n2v = [np.arange(3 * k, 3 * k + 3, dtype=np.int64) for k in range(400)]
    rust = _rs.skeletonize.merge_nested_nodes(nodes, radii, n2v, 0.8)
    python = python_skel.merge_nested_nodes(nodes, radii, n2v, 0.8)
    assert 50 < rust[0].sum() < 350
    assert_parity(rust, python)
    # the default fraction, with the root free to be absorbed and to absorb
    rust = _rs.skeletonize.merge_nested_nodes(nodes, radii, n2v, keep_root=False)
    python = python_skel.merge_nested_nodes(nodes, radii, n2v, keep_root=False)
    assert not rust[0][0] or rust[2][0] == 0
    assert_parity(rust, python)


def test_nested_merge_refuses_the_same_input_on_both_paths():
    nodes, radii = np.zeros((2, 3)), np.ones(2)
    n2v = [np.arange(1), np.arange(1, 2)]
    for merge in (_rs.skeletonize.merge_nested_nodes, python_skel.merge_nested_nodes):
        with pytest.raises(ValueError, match=r"inside_frac must be in \(0, 1\]"):
            merge(nodes, radii, n2v, 1.5)
        with pytest.raises(ValueError, match="one radius and one vertex list"):
            merge(nodes, radii[:1], n2v)
        with pytest.raises(ValueError, match="vertex index -1 is out of range"):
            merge(nodes, radii, [np.arange(1), np.array([-1])])
    assert_parity(
        _rs.skeletonize.merge_nested_nodes(np.empty((0, 3)), np.empty(0), []),
        python_skel.merge_nested_nodes(np.empty((0, 3)), np.empty(0), []),
    )


def test_direct_track_is_the_same_on_both_paths(reference_mesh):
    rust, python = both(skeletonize, reference_mesh)
    _assert_skeleton_valid(rust)
    assert len(rust.nodes) > 20
    assert_parity(_skeleton_fields(rust), _skeleton_fields(python))


def test_the_legacy_split_is_the_same_on_both_paths(reference_mesh):
    """The elongated-shell split bins on the Python path's igraph graph
    whichever path is active; the nodes it makes come from the active one."""
    rust, python = both(skeletonize, reference_mesh, split_elongated_shells=True)
    plain = skeletonize(reference_mesh)
    assert not np.array_equal(rust.nodes, plain.nodes), "the split did nothing"
    assert_parity(_skeleton_fields(rust), _skeleton_fields(python))


# ----- preprocessing track ------------------------------------------------


def _make_components(mesh, *, with_soma: bool):
    """Build synthetic MeshComponents from a mesh.

    Uses only the main connected component's faces (like
    real ``break_up_mesh`` would), since the preproc track
    expects each neurite to be surface-connected.
    """
    nF = len(mesh.faces)
    # find the largest connected vertex component
    gsurf = _surface_graph(mesh)
    comps = list(gsurf.components())
    main = set(max(comps, key=len))
    main_faces = np.array(
        [fi for fi in range(nF) if all(int(v) in main for v in mesh.faces[fi])],
        dtype=np.int64,
    )

    org = Organelles(
        pocket=np.zeros(nF, bool),
        isolated=np.zeros(nF, bool),
        expanded=np.zeros(nF, bool),
    )
    neurites = Neurites([main_faces])
    discarded = Discarded([])

    if with_soma:
        center = mesh.vertices.mean(axis=0)
        r = float(mesh.edges_unique_length.mean()) * 5
        verts = np.where(np.linalg.norm(mesh.vertices - center, axis=1) < r)[0].astype(
            np.int64
        )
        soma = Soma.from_sphere(center, r, verts=verts)
    else:
        soma = None

    return MeshComponents(
        soma=soma,
        organelles=org,
        neurites=neurites,
        discarded=discarded,
    )


def test_preproc_track_no_soma(reference_mesh):
    """Preprocessing track without soma → valid forest."""
    comp = _make_components(reference_mesh, with_soma=False)
    skel = skeletonize(reference_mesh, components=comp, verbose=False)
    _assert_skeleton_valid(skel, expect_soma=False)


def test_preproc_track_with_soma(reference_mesh):
    """Preprocessing track with soma → valid tree rooted at soma."""
    comp = _make_components(reference_mesh, with_soma=True)
    skel = skeletonize(reference_mesh, components=comp, verbose=False)
    _assert_skeleton_valid(skel, expect_soma=True)
    # soma center should be close to the mesh centroid
    mesh_center = reference_mesh.vertices.mean(axis=0)
    assert np.linalg.norm(skel.soma.center - mesh_center) < 1e3


def test_preproc_track_is_the_same_on_both_paths(reference_mesh):
    comp = _make_components(reference_mesh, with_soma=True)
    rust, python = both(skeletonize, reference_mesh, components=comp)
    assert "centerline" in rust.radii
    assert_parity(_skeleton_fields(rust), _skeleton_fields(python))


def _lists(shells):
    return [[np.asarray(b).tolist() for b in band] for band in shells]


def _preproc_steps(mesh, comp):
    """The preprocessing track's steps on its first neurite, on the active
    path: the shells, their nodes, the re-binned shells with their
    centreline distances, and the three clean-ups in turn."""
    sgraph = _surface_graph_of(mesh)
    v = mesh.vertices.view(np.ndarray)
    verts = np.unique(mesh.faces[comp.neurites[0]].ravel()).astype(np.int64)
    seeds = np.atleast_1d(_pick_neurite_seed(verts, v, comp.soma))
    est = ["median", "mean", "trim"]
    e_m = float(sgraph.lengths.mean())
    shells = sgraph.bin_component(verts, seeds, e_m, None, 500, 6, 50.0)
    nodes, radii, node2verts, vert2node = _make_nodes(
        shells, v, radius_estimators=est, merge_nested=False
    )
    mst = _build_mst(nodes, _edges_from_mesh(sgraph.edges, vert2node, len(v)))
    rebinned, cl_dist = _perpendicular_rebin(nodes, mst, node2verts, v, est, radii)
    rings = sgraph.recheck_rings(_lists(rebinned), verts)
    united = sgraph.reunite_bins(_lists(rings))
    return {
        "shells": shells,
        "nodes": (nodes, radii, node2verts, vert2node),
        "branch nodes": int((np.bincount(mst.ravel()) >= 3).sum()),
        "rebinned": (rebinned, cl_dist),
        "rings": rings,
        "united": united,
        "split": sgraph.split_branch_bins(_lists(united)),
    }


def test_preproc_steps_are_the_same_on_both_paths(reference_mesh):
    comp = _make_components(reference_mesh, with_soma=True)
    rust, python = both(_preproc_steps, reference_mesh, comp)
    assert_parity(rust, python)
    # a branch node donates, and the ring check and the reunion have work
    assert rust["branch nodes"] > 0
    assert not _shells_equal(rust["rebinned"][0], rust["rings"])
    assert not _shells_equal(rust["rings"], rust["united"])


def test_binning_from_several_seeds_is_the_same_on_both_paths(reference_mesh):
    mesh = reference_mesh
    verts = np.asarray(max(_surface_graph(mesh).components(), key=len))
    e_m = float(mesh.edges_unique_length.mean())

    def bins(seeds):
        return _surface_graph_of(mesh).bin_component(
            verts, seeds, e_m, None, 500, 6, 50.0
        )

    rust, python = both(bins, verts[::97])
    assert not _shells_equal(rust, bins(verts[:1])), "the seeds changed nothing"
    assert_parity(rust, python)


def _y_skeleton():
    """A branch node with three arms, each node's vertices scattered around
    it: ``(nodes, edges, node2verts, vertices)``."""
    rng = np.random.default_rng(7)
    unit = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1]]
    nodes = np.asarray(unit, dtype=np.float64) * 10
    edges = np.array([[0, 1], [1, 2], [0, 3], [3, 4], [0, 5]])
    verts = np.repeat(nodes, 12, axis=0) + rng.normal(scale=2.0, size=(72, 3))
    node2verts = [np.arange(12 * k, 12 * k + 12) for k in range(6)]
    return nodes, edges, node2verts, verts


@pytest.mark.parametrize("radius", [1.0, 0.0])
def test_chains_rebin_the_same_on_both_paths(radius):
    """The branch node's donations, and radii with no positive one to take
    the floor from."""
    nodes, edges, node2verts, verts = _y_skeleton()
    radii = {"median": np.full(len(nodes), radius)}
    rust, python = both(
        _perpendicular_rebin, nodes, edges, node2verts, verts, ["median"], radii
    )
    assert sorted(v for band in rust[0] for v in band[0]) == list(range(len(verts)))
    assert_parity(rust, python)


@pytest.mark.parametrize(
    "rebin_chains",
    [_rs.skeletonize.rebin_chains, python_skel.rebin_chains],
    ids=PATHS,
)
def test_chains_refuse_the_same_input_on_both_paths(rebin_chains):
    nodes, _, node2verts, verts = _y_skeleton()
    node2verts = [v.tolist() for v in node2verts]
    radii = np.ones(len(nodes))
    edge = np.array([[0, 1]])
    with pytest.raises(ValueError, match="node 9 is out of range"):
        rebin_chains(nodes, np.array([[0, 9]]), node2verts, verts, radii, [])
    with pytest.raises(ValueError, match="vertex 99 is out of range"):
        rebin_chains(nodes, edge, [[99], *node2verts[1:]], verts, radii, [])
    with pytest.raises(ValueError, match="vertex 99 is out of range"):
        rebin_chains(nodes, edge, node2verts, verts, radii, [(0, 1, [99])])
    with pytest.raises(ValueError, match="one radius and one vertex list"):
        rebin_chains(nodes, edge, node2verts, verts, radii[:2], [])
    with pytest.raises(ValueError, match="edges must be"):
        rebin_chains(nodes, np.zeros((1, 3), np.int64), node2verts, verts, radii, [])


def _two_boxes():
    """Two unit cubes apart: vertices and faces of a two-component mesh."""
    a, b = trimesh.creation.box(), trimesh.creation.box()
    b.apply_translation([5.0, 0.0, 0.0])
    faces = np.vstack([a.faces, b.faces + len(a.vertices)])
    return np.vstack([a.vertices, b.vertices]), faces


@pytest.mark.parametrize(
    "SurfaceGraph",
    [_rs.skeletonize.SurfaceGraph, python_skel.SurfaceGraph],
    ids=PATHS,
)
def test_the_surface_graph_refuses_the_same_input_on_both_paths(SurfaceGraph):
    vertices, faces = _two_boxes()
    with pytest.raises(ValueError, match="edges must be"):
        SurfaceGraph(3, np.zeros((2, 3), np.int64), np.zeros(2))
    with pytest.raises(ValueError, match="one weight per edge"):
        SurfaceGraph(3, np.zeros((2, 2), np.int64), np.zeros(1))
    with pytest.raises(ValueError, match="vertex index 5 is out of range"):
        SurfaceGraph(3, np.array([[0, 5]]), np.ones(1))
    with pytest.raises(ValueError, match="names vertex 99"):
        SurfaceGraph.from_mesh(vertices, np.array([[0, 1, 99]]))

    g = SurfaceGraph.from_mesh(vertices, faces)
    first, second = g.components()
    with pytest.raises(ValueError, match="no seed vertex lies in this component"):
        g.bin_component(first, second[:1], 1.0, None, 500, 6, 50.0)
    # a fixed shell width, and vertices no seed reaches
    with pytest.raises(ValueError, match="arange: cannot compute length"):
        g.bin_component(np.arange(len(vertices)), first[:1], 1.0, 0.5, 500, 6, 50.0)
    with pytest.raises(ValueError, match="vertex index -1 is out of range"):
        g.bin_mesh(-1, 1.0, None, 500, 6, 50.0)
    with pytest.raises(ValueError, match="vertex index 99 is out of range"):
        g.reunite_bins([[[0, 99]]])


@pytest.mark.parametrize(
    "skeleton_nodes",
    [_rs.skeletonize.skeleton_nodes, python_skel.skeleton_nodes],
    ids=PATHS,
)
def test_skeleton_nodes_refuse_the_same_input_on_both_paths(skeleton_nodes):
    v = np.random.default_rng(0).random((10, 3))
    with pytest.raises(ValueError, match="a bin has no vertices"):
        skeleton_nodes([[[0, 1], []]], v, ["median"])
    unknown = re.escape('Unknown radius estimator in ["median", "nope"].')
    with pytest.raises(ValueError, match=unknown):
        skeleton_nodes([[[0, 1]]], v, ["median", "nope"])
    # no bin, so no estimator to know
    nodes, radii = skeleton_nodes([], v, ["nope"])
    assert nodes.shape == (0, 3) and len(radii) == 1 and radii[0].shape == (0,)


def test_every_radius_estimator_is_the_same_on_both_paths(reference_mesh):
    from skeliner._core import RECOMPUTABLE_RADII

    mesh = reference_mesh
    e_m = float(mesh.edges_unique_length.mean())
    shells = _lists(
        _surface_graph_of(mesh).bin_mesh(_extreme_vertex(mesh), e_m, None, 1000, 6, 50)
    )
    v = mesh.vertices.view(np.ndarray)
    est = sorted(RECOMPUTABLE_RADII)
    assert_parity(
        _rs.skeletonize.skeleton_nodes(shells, v, est),
        python_skel.skeleton_nodes(shells, v, est),
    )


def test_branch_bins_split_the_same_on_both_paths():
    g, _, owner = _pants()
    edges = np.sort(np.asarray(g.get_edgelist(), dtype=np.int64), axis=1)
    shells: list[list[list[int]]] = [[[]] for _ in range(4)]
    for v, (bi, _) in sorted(owner.items()):
        shells[bi][0].append(v)
    split = [
        SurfaceGraph(g.vcount(), edges, np.ones(len(edges))).split_branch_bins(shells)
        for SurfaceGraph in (_rs.skeletonize.SurfaceGraph, python_skel.SurfaceGraph)
    ]
    assert [len(band) for band in split[0]] == [1, 3, 1, 1]
    assert_parity(*split)


def _barbell_surface():
    """Two wide tubes joined by a neck too thin to make a legal shell.

    Built as a surface graph rather than a mesh because the failure needs a
    neck whose geodesic bands hold fewer than ``min_shell_vertices``
    vertices, and the reference mesh is too coarse to have one — raising the
    threshold on it collapses the whole skeleton to a node or two instead of
    severing it.

    Returns ``(graph, vertices, neck_ids)``.
    """
    pts: list[list[float]] = []
    edges: list[tuple[int, int]] = []

    def ring(x, n, r):
        base = len(pts)
        for k in range(n):
            a = 2 * np.pi * k / n
            pts.append([x, r * np.cos(a), r * np.sin(a)])
        for k in range(n):
            edges.append((base + k, base + (k + 1) % n))
        return base, n

    prev = None
    for i in range(6):  # wide tube A
        cur = ring(i * 1.0, 12, 3.0)
        if prev:
            for k in range(12):
                edges.append((prev[0] + k, cur[0] + k))
        prev = cur

    neck_start = len(pts)
    for i in range(3):  # the neck: 3 vertices per ring, below the cut
        cur = ring(6.0 + i * 1.0, 3, 0.4)
        for k in range(3):
            edges.append((prev[0] + (k * 4 if prev[1] == 12 else k), cur[0] + k))
        prev = cur
    neck = list(range(neck_start, len(pts)))

    for i in range(6):  # wide tube B
        cur = ring(9.0 + i * 1.0, 12, 3.0)
        for k in range(3):
            edges.append((prev[0] + k, cur[0] + k))
        prev = cur

    verts = np.asarray(pts, dtype=np.float64)
    elist = sorted(set(tuple(sorted(e)) for e in edges))
    g = ig.Graph(n=len(verts), edges=elist, directed=False)
    g.es["weight"] = [float(np.linalg.norm(verts[a] - verts[b])) for a, b in elist]
    return g, verts, neck


def test_a_shell_too_small_to_bin_still_keeps_its_vertices():
    """A dropped band severs the skeleton, however continuous the surface.

    `_edges_from_mesh` joins two nodes only when a mesh edge has **both**
    endpoints binned.  So a shell sub-component discarded for being smaller
    than ``min_shell_vertices`` does not merely cost a node — it leaves no
    edge across the gap, and at a narrow neck the sub-components are small
    for several consecutive shells, which cuts the arbor in two.  Being too
    thin to be its own bin is not the same as belonging nowhere.
    """
    from skeliner.skeletonize import _bin_one_component

    g, verts, neck = _barbell_surface()
    assert len(g.components()) == 1, "the fixture must be one connected surface"

    shells = _bin_one_component(
        g,
        np.arange(len(verts), dtype=np.int64),
        0,
        mesh_vertices=verts,
        mean_edge_len=1.0,
        min_shell_vertices=6,
    )

    binned: set[int] = set()
    for band in shells:
        for comp in band:
            binned.update(int(x) for x in comp)

    missing_neck = sorted(v for v in neck if v not in binned)
    assert not missing_neck, f"neck vertices dropped: {missing_neck}"
    assert binned == set(range(len(verts))), (
        f"{len(verts) - len(binned)} vertices left unbinned"
    )


# ----- centreline distances survive so `centerline` can be recomputed ------
#
# Every other radius is an aggregate over the vertices a node owns, so it
# falls out of a re-partition for free.  `centerline` aggregates a *per
# vertex* perpendicular distance that only the second binning pass knows,
# so it is kept on the skeleton — otherwise editing a bin would leave a
# stale `centerline` beside freshly recomputed `mean` / `median` / `trim`.


def test_centerline_distances_are_kept(reference_mesh):
    comp = _make_components(reference_mesh, with_soma=True)
    skel = skeletonize(reference_mesh, components=comp, verbose=False)
    if "centerline" not in skel.radii:
        pytest.skip("second pass did not run on this mesh")

    vids = skel.extra["cl_dist_vids"]
    vals = skel.extra["cl_dist_vals"]
    assert vids.size == vals.size > 0
    assert np.unique(vids).size == vids.size, "one distance per vertex"


def test_centerline_is_reproducible_from_what_was_kept(reference_mesh):
    """The stored map must reproduce `radii['centerline']` exactly, or
    keeping it buys nothing."""
    from skeliner._core import _estimate_radius

    comp = _make_components(reference_mesh, with_soma=True)
    skel = skeletonize(reference_mesh, components=comp, verbose=False)
    if "centerline" not in skel.radii:
        pytest.skip("second pass did not run on this mesh")

    lut = np.full(len(reference_mesh.vertices), 0.0)
    lut[skel.extra["cl_dist_vids"]] = skel.extra["cl_dist_vals"]

    stored = np.asarray(skel.radii["centerline"])
    for ni in range(1, len(skel.nodes)):  # node 0 is the soma, not a bin
        d = lut[np.asarray(skel.node2verts[ni], np.int64)]
        want = _estimate_radius(d, method="trim", trim_fraction=0.05) if d.size else 0.0
        assert want == pytest.approx(stored[ni], abs=1e-9), f"node {ni}"


# ----- verbose timing breakdown -------------------------------------------


def test_preproc_track_reports_each_stage_as_it_finishes(reference_mesh, capsys):
    """One number for the whole neurite loop says nothing about which
    stage is slow, and `_timed` holds sub-messages until its block ends,
    so the stages are top-level steps that print when they complete."""
    comp = _make_components(reference_mesh, with_soma=False)
    skeletonize(reference_mesh, components=comp, verbose=True)
    out = capsys.readouterr().out

    stages = [ln for ln in out.splitlines() if ln.lstrip().startswith("↳")]
    named = {ln.split("…")[0].split("↳")[1].strip() for ln in stages}
    assert "bin vertices by geodesic distance" in named, named
    assert "split bins that wrap a branch point" in named, named
    assert "skeletonize neurites" not in named, "the opaque wrapper is gone"
    for ln in stages:
        assert ln.rstrip().endswith(" s"), ln


def test_no_breakdown_when_quiet(reference_mesh, capsys):
    comp = _make_components(reference_mesh, with_soma=False)
    skeletonize(reference_mesh, components=comp, verbose=False)
    assert capsys.readouterr().out == ""


# ----- branch-band split --------------------------------------------------
#
# A geodesic shell that lands on a branch point wraps the parent tube and
# both children in one connected band.  It passes the ring test, so it
# becomes a bin, and its node is the mean of points on two diverging tubes
# — which lands between them instead of inside one.  `_neighbour_groups`
# counts how many separate neighbourhoods a bin touches (two for a
# cross-section, three at a branch) and `_split_branch_band` cuts on that.


def _ring(start, n):
    """Vertex ids of a cycle plus its edges, as (ids, edges)."""
    ids = list(range(start, start + n))
    edges = [(ids[i], ids[(i + 1) % n]) for i in range(n)]
    return ids, edges


def _pants():
    """Three rings around a fourth: the parent, the band, and two children.

    ``band`` touches all three of the others, so it is the bin a branch
    point produces.  Returns the graph, the band's vertex ids, and the
    ownership map.
    """
    n = 8
    parent, e_parent = _ring(0, n)
    band, e_band = _ring(n, 2 * n)  # wide enough to feed two children
    childA, e_a = _ring(3 * n, n)
    childB, e_b = _ring(4 * n, n)

    edges = e_parent + e_band + e_a + e_b
    # parent sits under the first half of the band, the children over the
    # second half, one on each quarter
    for i in range(n):
        edges.append((parent[i], band[i]))
    for i in range(n):
        edges.append((childA[i], band[n + (i % (n // 2))]))
        edges.append((childB[i], band[n + n // 2 + (i % (n // 2))]))

    owner = {}
    for bin_id, verts in enumerate((parent, band, childA, childB)):
        for v in verts:
            owner[v] = (bin_id, 0)
    g = ig.Graph(n=5 * n, edges=edges)
    return g, np.asarray(band, dtype=np.int64), owner


def test_branch_band_sees_three_neighbourhoods():
    g, band, owner = _pants()
    assert len(_neighbour_groups(band, g, owner, (1, 0))) == 3


def test_two_patches_of_one_bin_are_one_neighbourhood():
    """Dipping into the same neighbour twice is one tube, not two.

    Counting vertex patches instead of bins made a plain band look like a
    branch point and cut it into slivers.
    """
    g, band, owner = _pants()
    # merge both children into a single bin: the band now touches two bins
    # (parent, child) but along three separate patches
    for v in list(owner):
        if owner[v] == (3, 0):
            owner[v] = (2, 0)
    assert len(_neighbour_groups(band, g, owner, (1, 0))) == 2


def test_end_ring_sees_one_neighbourhood():
    """The parent ring touches only the band: nothing to split."""
    g, _, owner = _pants()
    parent = np.arange(0, 8, dtype=np.int64)
    assert len(_neighbour_groups(parent, g, owner, (0, 0))) == 1


def test_split_yields_one_connected_piece_per_tube():
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    parts = _split_branch_band(band, groups, g)
    assert len(parts) == 3

    allv = np.concatenate(parts).tolist()
    assert sorted(allv) == sorted(band.tolist())
    assert len(set(allv)) == len(allv), "a vertex landed in two pieces"
    for part in parts:
        # a disconnected piece leaves its node with no place to sit
        sub = g.induced_subgraph([int(v) for v in part])
        assert len(part) > 0 and len(sub.components()) == 1


def test_split_is_deterministic():
    g, band, owner = _pants()
    groups = _neighbour_groups(band, g, owner, (1, 0))
    first = [p.tolist() for p in _split_branch_band(band, groups, g)]
    second = [p.tolist() for p in _split_branch_band(band, groups, g)]
    assert first == second


# ----- one radius estimator, one meaning -----------------------------------
#
# `post` used to carry a near-copy of `_estimate_radius` differing only in
# its defaults, so `trim` and `percentile` quietly meant different things
# depending on which module you reached from — and the merge paths inside
# `_prune_neurites` recomputed radii with a different trim than
# `_make_nodes` had built them with.


def test_every_estimator_is_reachable_from_one_place():
    from skeliner._core import RECOMPUTABLE_RADII, _estimate_radius

    d = np.linspace(1.0, 100.0, 200)
    for method in RECOMPUTABLE_RADII:
        assert np.isfinite(_estimate_radius(d, method=method))
    with pytest.raises(ValueError, match="Unknown radius estimator"):
        _estimate_radius(d, method="nonsense")


def test_recompute_skips_keys_that_are_not_distance_aggregates():
    """`centerline` aggregates perpendicular distances and `calibrated` is
    ray-cast, so neither can be rebuilt from a node's own distances.  Passing
    them to the estimator is what used to raise."""
    from skeliner._core import _radii_from_distances

    radii = {
        "trim": np.zeros(1),
        "centerline": np.full(1, 7.0),
        "calibrated": np.full(1, 9.0),
    }
    stale = _radii_from_distances(radii, 0, np.linspace(1.0, 10.0, 50))

    assert stale == ["centerline", "calibrated"]
    assert radii["trim"][0] > 0, "the aggregate keys are rebuilt"
    assert radii["centerline"][0] == 7.0, "left alone, not corrupted"
    assert radii["calibrated"][0] == 9.0


def test_centerline_is_rebuilt_when_its_distances_are_supplied():
    from skeliner._core import _radii_from_distances

    radii = {"centerline": np.zeros(1)}
    stale = _radii_from_distances(
        radii, 0, np.linspace(1.0, 10.0, 50), cl_d=np.full(50, 12.0)
    )
    assert stale == []
    assert radii["centerline"][0] == pytest.approx(12.0)


def test_trim_means_the_same_thing_everywhere():
    """A merged node's `trim` radius must be comparable with every other
    node's, which means one trim fraction across the whole array."""
    from skeliner._core import TRIM_FRACTION, _estimate_radius, _radii_from_distances

    d = np.concatenate([np.full(300, 80.0), np.full(120, 400.0)])
    radii = {"trim": np.zeros(1)}
    _radii_from_distances(radii, 0, d)
    assert radii["trim"][0] == pytest.approx(
        _estimate_radius(d, method="trim", trim_fraction=TRIM_FRACTION)
    )


# ── naming neurites, and the SWC type it carries ──────────────────────
#
# Names live at a *position* in Neurites.components, and a re-derive
# re-sorts by size, splits and merges, so a name that survived one would
# land on different surface.  Naming is therefore terminal: everything
# break_up_mesh produces is unnamed, and re-breaking drops the names.


def test_break_up_mesh_produces_unnamed_neurites():
    from skeliner import pre
    from skeliner.dataclass import Organelles

    mesh = trimesh.creation.cylinder(radius=50.0, height=1000.0, sections=16)
    z = np.zeros(len(mesh.faces), bool)
    org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
    c = pre.break_up_mesh(mesh, None, org)
    assert c.neurites.named is False
    assert c.neurites.labels is None and c.neurites.swc_types is None


@pytest.mark.parametrize(
    "label, code",
    [
        ("axon", 2),
        ("dendrite 1", 3),
        ("Apical tuft", 4),
        ("basal 2", 3),
        ("something else", 0),
        ("", 0),
    ],
)
def test_swc_type_is_read_from_the_leading_word(label, code):
    from skeliner.dataclass import swc_type_for

    assert swc_type_for(label) == code


def test_naming_one_neurite_names_them_all():
    """No holes: the skeleton always has a code to stamp."""
    n = Neurites([np.arange(4), np.arange(4, 8), np.arange(8, 12)])
    n.name(1, "axon")
    assert n.labels == ["neurite 0", "axon", "neurite 2"]
    assert n.swc_types == [0, 2, 0]


def test_an_explicit_code_overrides_the_name():
    n = Neurites([np.arange(4)])
    n.name(0, "the weird one", swc_type=2)
    assert n.swc_types == [2]


def test_clear_names_goes_back_to_unnamed():
    n = Neurites([np.arange(4)])
    n.name(0, "axon")
    n.clear_names()
    assert n.named is False


def test_labels_and_types_must_agree_with_the_components():
    with pytest.raises(ValueError, match="one entry per component"):
        Neurites([np.arange(4)], labels=["a", "b"], swc_types=[2, 3])
    with pytest.raises(ValueError, match="together"):
        Neurites([np.arange(4)], labels=["a"])


def test_names_survive_the_npz_round_trip(tmp_path):
    from skeliner import io

    n = Neurites([np.arange(4), np.arange(4, 8)])
    n.name(0, "axon")
    n.name(1, "dendrite 1")
    io.save_neurites_npz(n, tmp_path / "n.npz")
    back = io.load_neurites_npz(tmp_path / "n.npz")
    assert back.labels == ["axon", "dendrite 1"]
    assert back.swc_types == [2, 3]


def test_an_unnamed_npz_round_trips_as_unnamed(tmp_path):
    from skeliner import io

    io.save_neurites_npz(Neurites([np.arange(4)]), tmp_path / "n.npz")
    assert io.load_neurites_npz(tmp_path / "n.npz").named is False


def _ball_with_two_processes():
    """A soma with one process each way, long enough to bin into nodes."""
    ball = trimesh.creation.icosphere(subdivisions=4, radius=100.0)
    parts = [ball]
    for sign in (1, -1):
        tube = trimesh.creation.cylinder(radius=15.0, height=900.0, sections=24)
        tube = tube.subdivide().subdivide()
        tube.apply_translation([0.0, 0.0, sign * 520.0])
        parts.append(tube)
    verts, faces, n = [], [], 0
    for p in parts:
        verts.append(p.vertices)
        faces.append(p.faces + n)
        n += len(p.vertices)
    mesh = trimesh.Trimesh(
        vertices=np.vstack(verts), faces=np.vstack(faces), process=False
    )

    from skeliner import pre
    from skeliner.dataclass import Organelles

    z = np.zeros(len(mesh.faces), bool)
    org = Organelles(pocket=z.copy(), isolated=z.copy(), expanded=z.copy())
    sv = np.flatnonzero(np.linalg.norm(mesh.vertices, axis=1) <= 105.0)
    soma = Soma.fit(mesh.vertices[sv], verts=sv)
    pieces = pre._face_components_fast(
        np.asarray(mesh.faces),
        np.flatnonzero(
            pre._usable_face_mask(mesh)
            & ~pre.soma_face_mask(np.asarray(mesh.faces), sv)
            & ~org.mask
        ),
    )
    assert len(pieces) == 2, "fixture must give one component per process"
    claim = np.concatenate([p[:1] for p in pieces])
    return mesh, pre.break_up_mesh(mesh, soma, org, rescued=claim)


def test_unnamed_neurites_skeletonize_to_undefined():
    mesh, comp = _ball_with_two_processes()
    skel = skeletonize(mesh, components=comp, verbose=False)
    assert skel.ntype[0] == 1, "node 0 is the soma"
    assert set(skel.ntype[1:].tolist()) == {0}


def test_a_named_neurite_stamps_its_code_on_its_nodes():
    mesh, comp = _ball_with_two_processes()
    comp.neurites.name(0, "axon")
    comp.neurites.name(1, "dendrite 1")
    skel = skeletonize(mesh, components=comp, verbose=False)

    assert skel.ntype[0] == 1, "the soma must not be retyped by the fringe"
    assert set(skel.ntype[1:].tolist()) == {2, 3}
    assert (skel.ntype == 2).sum() > 0 and (skel.ntype == 3).sum() > 0


def test_a_zero_code_leaves_its_nodes_alone():
    """A name with no SWC meaning is still a name — it just does not type."""
    mesh, comp = _ball_with_two_processes()
    comp.neurites.name(0, "the weird one")
    comp.neurites.name(1, "axon")
    skel = skeletonize(mesh, components=comp, verbose=False)
    assert set(skel.ntype[1:].tolist()) == {0, 2}


# ── the name and its SWC code reach every persisted form ──────────────
#
# `ntype` alone loses the name: "dendrite 0" and "dendrite 1" are both
# code 3, and SWC has no field for a name.  So the labels ride in `meta`,
# which both the npz and the SWC header carry, and the per-node owner in
# `extra`, which is what resolves a name back to nodes.


def _named_skeleton():
    mesh, comp = _ball_with_two_processes()
    comp.neurites.name(0, "axon")
    comp.neurites.name(1, "dendrite 1")
    return skeletonize(mesh, components=comp, verbose=False)


def test_the_skeleton_records_the_names_and_the_owner_of_each_node():
    skel = _named_skeleton()
    assert skel.meta["neurite_labels"] == ["axon", "dendrite 1"]
    assert skel.meta["neurite_swc_types"] == [2, 3]
    owner = skel.extra["node2neurite"]
    assert len(owner) == len(skel.nodes)
    assert owner[0] == -1, "the soma belongs to no neurite"
    assert set(owner.tolist()) == {-1, 0, 1}


def test_an_unnamed_skeleton_records_neither():
    mesh, comp = _ball_with_two_processes()
    skel = skeletonize(mesh, components=comp, verbose=False)
    assert "neurite_labels" not in skel.meta
    assert "node2neurite" not in skel.extra


def test_names_and_ntype_survive_the_skeleton_npz(tmp_path):
    from skeliner import io

    skel = _named_skeleton()
    io.save_skeleton_npz(skel, tmp_path / "s.npz")
    back = io.load_skeleton_npz(tmp_path / "s.npz")

    assert set(back.ntype[1:].tolist()) == {2, 3}
    assert back.meta["neurite_labels"] == ["axon", "dendrite 1"]
    assert np.array_equal(back.extra["node2neurite"], skel.extra["node2neurite"])


def test_names_and_ntype_survive_the_swc(tmp_path):
    """SWC carries the type in its own column and the names in the
    `# meta` header, which the loader parses back."""
    from skeliner import io

    skel = _named_skeleton()
    io.save_skeleton_swc(skel, tmp_path / "s.swc")
    back = io.load_skeleton_swc(tmp_path / "s.swc")

    assert set(back.ntype[1:].tolist()) == {2, 3}
    assert back.meta["neurite_labels"] == ["axon", "dendrite 1"]


def test_the_swc_type_column_is_what_was_named(tmp_path):
    """Read the file as text — the column is the whole point of naming."""
    from skeliner import io

    skel = _named_skeleton()
    io.save_skeleton_swc(skel, tmp_path / "s.swc")
    types = [
        int(line.split()[1])
        for line in (tmp_path / "s.swc").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert types[0] == 1, "the soma"
    assert set(types[1:]) == {2, 3}


# ── the scripting API for naming ──────────────────────────────────────


def test_rename_takes_a_sequence():
    n = Neurites([np.arange(4), np.arange(4, 8), np.arange(8, 12)])
    n.rename(["dendrite 0", "dendrite 1", "axon"])
    assert n.labels == ["dendrite 0", "dendrite 1", "axon"]
    assert n.swc_types == [3, 3, 2]


def test_rename_takes_a_mapping_and_leaves_the_rest_alone():
    n = Neurites([np.arange(4), np.arange(4, 8), np.arange(8, 12)])
    n.rename({2: "axon"})
    assert n.labels == ["neurite 0", "neurite 1", "axon"]
    assert n.swc_types == [0, 0, 2]


def test_rename_returns_self_so_it_chains():
    n = Neurites([np.arange(4)])
    assert n.rename(["axon"]) is n


def test_rename_takes_explicit_codes():
    n = Neurites([np.arange(4), np.arange(4, 8)])
    n.rename(["odd one", "axon"], swc_types=[5, 2])
    assert n.swc_types == [5, 2]


def test_a_sequence_of_the_wrong_length_is_an_error():
    n = Neurites([np.arange(4), np.arange(4, 8)])
    with pytest.raises(ValueError, match="one name per component"):
        n.rename(["axon"])
    with pytest.raises(ValueError, match="one code per component"):
        n.rename(["a", "b"], swc_types=[2])


def test_index_of_finds_a_neurite_by_name():
    n = Neurites([np.arange(4), np.arange(4, 8)])
    n.rename(["dendrite 0", "axon"])
    assert n.index_of("axon") == 1


def test_index_of_refuses_an_ambiguous_or_missing_name():
    """Names are free text and nothing enforces uniqueness, so a silent
    first match would be the wrong answer half the time."""
    n = Neurites([np.arange(4), np.arange(4, 8)])
    n.rename(["axon", "axon"])
    with pytest.raises(KeyError, match="2 neurites are called"):
        n.index_of("axon")
    with pytest.raises(KeyError, match="no neurite called"):
        n.index_of("dendrite 9")
    with pytest.raises(KeyError, match="no names"):
        Neurites([np.arange(4)]).index_of("axon")


def test_summary_shows_the_names_and_codes():
    n = Neurites([np.arange(4), np.arange(4, 8)])
    assert "2 neurites" in n.summary()
    assert "axon" not in n.summary()
    n.rename({1: "axon"})
    assert "axon (SWC 2)" in n.summary()


def test_dx_resolves_a_neurite_by_name_and_by_index():
    mesh, comp = _ball_with_two_processes()
    comp.neurites.rename(["axon", "dendrite 1"])
    skel = skeletonize(mesh, components=comp, verbose=False)

    assert skel.dx.neurite_names() == {0: "axon", 1: "dendrite 1"}
    by_name = skel.dx.neurite_nodes("axon")
    by_index = skel.dx.neurite_nodes(0)
    assert np.array_equal(by_name, by_index)
    assert 0 not in by_name.tolist(), "the soma belongs to no neurite"
    assert set(skel.ntype[by_name].tolist()) == {2}


def test_dx_neurite_helpers_on_an_unnamed_skeleton():
    mesh, comp = _ball_with_two_processes()
    skel = skeletonize(mesh, components=comp, verbose=False)
    assert skel.dx.neurite_names() == {}
    with pytest.raises(KeyError, match="no per-node neurite map"):
        skel.dx.neurite_nodes(0)


def test_dx_neurite_nodes_refuses_an_unknown_name():
    mesh, comp = _ball_with_two_processes()
    comp.neurites.rename(["axon", "dendrite 1"])
    skel = skeletonize(mesh, components=comp, verbose=False)
    with pytest.raises(KeyError, match="no neurite called"):
        skel.dx.neurite_nodes("apical")


# ---------------------------------------------------------------------
# _core's stages on both paths
# ---------------------------------------------------------------------
def _stable_kruskal(nodes, edges) -> list[tuple[int, int]]:
    """Kruskal's forest with ties in input order: not igraph's."""
    w = [float(np.linalg.norm(nodes[a] - nodes[b])) for a, b in edges]
    comp = list(range(len(nodes)))

    def root(x):
        while comp[x] != x:
            x = comp[x]
        return x

    tree = []
    for i in np.argsort(w, kind="stable"):
        a, b = (root(int(v)) for v in edges[i])
        if a != b:
            comp[a] = b
            tree.append(tuple(sorted(map(int, edges[i]))))
    return sorted(tree)


def test_bfs_parents_and_spanning_forest_on_both_paths():
    rng = np.random.default_rng(3)
    ties_matter = 0
    for _ in range(300):
        n = int(rng.integers(2, 60))
        edges = rng.integers(0, n, size=(int(rng.integers(0, 4 * n)), 2))
        # nodes on a small grid, so many edge lengths tie exactly
        nodes = rng.integers(0, 3, size=(n, 3)).astype(np.float64)
        root = int(rng.integers(0, n))
        assert_parity(
            _rs.skeletonize.bfs_parents(edges, n, root),
            python_skel.bfs_parents(edges, n, root),
        )
        mst = _rs.skeletonize.build_mst(nodes, edges)
        assert_parity(mst, python_skel.build_mst(nodes, edges))
        assert mst.shape[1] == 2
        ties_matter += [tuple(e) for e in mst.tolist()] != _stable_kruskal(nodes, edges)
    # the forests above differ from a stable sort's: igraph's tie order is
    # what they test
    assert ties_matter > 10
    # no edges: an empty (0, 2) forest, every node its own root
    assert _rs.skeletonize.build_mst(np.zeros((3, 3)), np.empty((0, 2), int)).shape == (
        0,
        2,
    )
    assert _rs.skeletonize.bfs_parents(np.empty((0, 2), int), 3, 0).tolist() == [-1] * 3


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda m: m.build_mst(np.full((2, 3), np.nan), np.array([[0, 1]])),
            "must not be NaN",
        ),
        (
            lambda m: m.build_mst(np.zeros((2, 3)), np.array([[0, 2]])),
            "edge names node 2, not",
        ),
        (
            lambda m: m.build_mst(np.zeros((2, 3)), np.array([[0, 1, 1]])),
            r"got shape \[1, 3\]",
        ),
        (lambda m: m.bfs_parents(np.empty((0, 2), int), 3, 3), "root 3 is not one"),
        (lambda m: m.bfs_parents(np.array([[0, -1]]), 3, 0), "edge names node -1, not"),
    ],
)
def test_core_stage_errors_alike_on_both_paths(call, message):
    for module in (_rs.skeletonize, python_skel):
        with pytest.raises(ValueError, match=message):
            call(module)


def test_single_node_branches_apart_from_the_root_stay():
    from skeliner._core import _merge_single_node_branches

    # 0-6, 1-6, 4-6: node 6, the last, has three edges; 2-3 and 2-5 hang
    # apart from the root, where a BFS from it gives 3 and 5 no parent
    nodes = np.arange(21, dtype=float).reshape(7, 3)
    edges = np.array([[0, 6], [1, 6], [4, 6], [2, 3], [2, 5]])
    n2v = [np.array([i]) for i in range(7)]
    out = _merge_single_node_branches(
        nodes,
        {"median": np.ones(7)},
        n2v,
        edges,
        mesh_vertices=None,
        return_mapping=True,
    )
    assert out[5].tolist() == [0, 2, 3, 5, 6]
    assert [v.tolist() for v in out[2]] == [[0], [2], [3], [5], [6, 1, 4]]


def _random_skeleton(rng, n: int, n_mesh: int = 400):
    """A random tree of *n* nodes with a few extra edges, radii under keys
    the stages re-estimate and keys they leave, and vertex lists (some
    empty) into a random mesh: the arrays ``_core``'s stages take."""
    edges = [[int(rng.integers(0, i)), i] for i in range(1, n)]
    edges += rng.integers(0, n, size=(int(rng.integers(0, 3)), 2)).tolist()
    keys = ["median", "mean", "trim", "centerline", "calibrated"]
    lens = rng.integers(0, 6, size=n)
    ptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
    idx = rng.integers(0, n_mesh, size=int(ptr[-1])).astype(np.int64)
    return dict(
        nodes=rng.normal(size=(n, 3)) * 10,
        keys=keys,
        radii=rng.random((len(keys), n)),
        ptr=ptr,
        idx=idx,
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        mesh_vertices=rng.normal(size=(n_mesh, 3)) * 10,
    )


def test_single_node_merge_and_pruning_on_both_paths():
    rng = np.random.default_rng(11)
    merged = pruned = 0
    for trial in range(150):
        s = _random_skeleton(rng, int(rng.integers(2, 50)))
        mesh = s["mesh_vertices"] if trial % 3 else None
        args = [s[k] for k in ("nodes", "keys", "radii", "ptr", "idx", "edges")]
        for degree in (1, 2, 3):
            rust = _rs.skeletonize.merge_single_node_branches(*args, mesh, degree)
            assert_parity(
                rust, python_skel.merge_single_node_branches(*args, mesh, degree)
            )
            merged += len(rust[1]) < len(s["nodes"])
        center = s["nodes"][0] + rng.normal(size=3)
        r_soma = float(rng.uniform(1, 15))
        for drop in (True, False):
            rest = (center, r_soma, mesh, 1.2, float(rng.uniform(1, 3)), drop)
            rust = _rs.skeletonize.prune_neurites(*args, *rest)
            assert_parity(rust, python_skel.prune_neurites(*args, *rest))
            pruned += rust[3] > 0
    assert merged > 50 and pruned > 20
    s = _random_skeleton(rng, 5)
    args = [s[k] for k in ("nodes", "keys", "radii", "ptr", "idx", "edges")]
    out_of_mesh = np.full_like(s["idx"], 400)
    for module in (_rs.skeletonize, python_skel):
        if out_of_mesh.size:
            with pytest.raises(ValueError, match="vertex index 400 is out of range"):
                module.merge_single_node_branches(
                    *args[:4], out_of_mesh, args[5], s["mesh_vertices"], 3
                )
        with pytest.raises(ValueError, match="CSR over the nodes"):
            module.merge_single_node_branches(
                *args[:3], args[3][:-1], *args[4:], None, 3
            )


def test_pruning_ends_alike_on_both_paths(reference_mesh):
    skel = sk.skeletonize(reference_mesh, verbose=False)
    for kwargs in (
        dict(stem_extent_factor=3.0),
        dict(stem_extent_factor=6.0, drop_single_node_branches=False),
    ):
        rust, python = both(
            sk.post.prune_neurites,
            skel,
            mesh_vertices=reference_mesh.vertices,
            verbose=False,
            **kwargs,
        )
        assert_parity(_skeleton_fields(rust), _skeleton_fields(python))


def test_soma_merge_on_both_paths():
    rng = np.random.default_rng(13)
    merged = with_verts = 0
    for trial in range(150):
        s = _random_skeleton(rng, int(rng.integers(2, 50)))
        mesh = s["mesh_vertices"] if trial % 3 else None
        # a soma around node 0, big enough to take some nodes in
        soma = Soma.fit(s["nodes"][0] + rng.normal(size=(8, 3)) * rng.uniform(2, 12))
        verts = None if trial % 4 == 0 else rng.integers(0, 400, size=5)
        args = [s[k] for k in ("nodes", "keys", "radii", "ptr", "idx", "edges")]
        rest = (
            int(rng.integers(0, 2)),
            soma.center,
            soma.axes,
            soma.R,
            verts,
            mesh,
            float(rng.choice([0.0, 1.0])),
            float(rng.uniform(0.5, 3)),
            float(rng.uniform(0.0, 0.5)),
        )
        rust = _rs.skeletonize.merge_near_soma_nodes(*args, *rest)
        assert_parity(rust, python_skel.merge_near_soma_nodes(*args, *rest))
        merged += rust[3] > 0
        with_verts += rust[2] is not None and len(rust[2]) > (0 if verts is None else 5)
        assert not (rust[0][4][:, 0] == rust[0][4][:, 1]).any()
    assert merged > 50 and with_verts > 20


def test_soma_distances_follow_numpy():
    """The kernel's distances to the soma reach the merge masks through
    every point; check the ellipsoid on its own too, through the merge's
    `inside` test with a zero tolerance and no fat test."""
    rng = np.random.default_rng(17)
    for _ in range(40):
        n = int(rng.integers(2, 200))
        nodes = rng.normal(size=(n, 3)) * rng.uniform(1, 1e3)
        soma = Soma.fit(rng.normal(size=(20, 3)) * rng.uniform(1, 1e3))
        d = soma.distance_to_surface(nodes)
        args = (
            nodes,
            ["median"],
            np.ones((1, n)),
            np.zeros(n + 1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            0,
            soma.center,
            soma.axes,
            soma.R,
            None,
            None,
        )
        for tol in np.unique(np.abs(d))[:3]:
            # nodes exactly at -tol stay: the masks need the exact distances
            rust = _rs.skeletonize.merge_near_soma_nodes(*args, float(tol), 0.0, 1e9)
            want = d < -tol
            want[0] = False
            assert rust[1].tolist() == np.where(want, -1, np.cumsum(~want) - 1).tolist()


def test_soma_merge_ends_alike_on_both_paths(reference_mesh):
    skel = sk.skeletonize(reference_mesh, verbose=False)
    for kwargs in (
        dict(near_factor=3.0, fat_factor=0.05),
        dict(near_factor=1.2, fat_factor=0.2, inside_tol=0.5),
    ):
        rust, python = both(
            sk.post.merge_near_soma_nodes,
            skel,
            mesh_vertices=reference_mesh.vertices,
            verbose=False,
            **kwargs,
        )
        assert_parity(_skeleton_fields(rust), _skeleton_fields(python))


def test_soma_detection_collects_the_soma_nodes_after_moving_one_to_the_root():
    from skeliner._core import _detect_soma

    # node 0 lies apart; nodes 2, 3, 5 make the soma, 3 the biggest, so it
    # moves to node 0 — and its vertices must count, the old root's not
    rng = np.random.default_rng(2)
    nodes = np.array(
        [[100.0, 0, 0], [50, 0, 0], [0, 0, 1], [0, 1, 0], [60, 0, 0], [1, 0, 0]]
    )
    radii = {"median": np.array([1.0, 10, 12, 30, 11, 13])}
    node2verts = [np.arange(4 * i, 4 * i + 4) for i in range(6)]
    mesh_vertices = np.repeat(nodes, 4, axis=0) + rng.normal(size=(24, 3))
    *_, soma, has_soma, old2new = _detect_soma(
        nodes.copy(),
        {k: v.copy() for k, v in radii.items()},
        [v.copy() for v in node2verts],
        {},
        soma_radius_percentile_threshold=50,
        soma_radius_distance_factor=3,
        soma_min_nodes=2,
        detect_soma=True,
        radius_key="median",
        mesh_vertices=mesh_vertices,
    )
    assert has_soma and old2new[3] == 0 and old2new[0] == 3
    assert set(soma.verts) == set(node2verts[2]) | set(node2verts[3]) | set(
        node2verts[5]
    )


def test_soma_finding_on_both_paths():
    rng = np.random.default_rng(19)
    found = 0
    for trial in range(200):
        n = int(rng.integers(1, 60))
        nodes = rng.normal(size=(n, 3)) * 10
        radius = rng.gamma(2.0, size=n)
        if trial % 5 == 0:
            radius = np.round(radius)  # ties at the percentile and the max
        pct = float(rng.choice([50.0, 90.0, 95.0, 99.9, 100.0, 0.0]))
        args = (nodes, radius, pct, float(rng.uniform(0.5, 6)), int(rng.integers(1, 4)))
        rust = _rs.skeletonize.find_soma(*args)
        assert_parity(rust, python_skel.find_soma(*args))
        found += rust[4]
        soma_nodes = rust[0]
        lists = [rng.integers(0, 500, size=int(rng.integers(0, 4))) for _ in soma_nodes]
        ptr = np.r_[0, np.cumsum([len(v) for v in lists])].astype(np.int64)
        idx = np.concatenate(lists).astype(np.int64) if lists else np.empty(0, int)
        vargs = (nodes[soma_nodes], ptr, idx, np.asarray(rust[2]), rust[3])
        assert_parity(
            _rs.skeletonize.soma_vertices(*vargs), python_skel.soma_vertices(*vargs)
        )
    assert found > 50
    with_nan = np.array([1.0, np.nan, 2.0])
    for module in (_rs.skeletonize, python_skel):
        assert module.find_soma(np.zeros((3, 3)), with_nan, 50.0, 3.0, 2) is None
        with pytest.raises(ValueError, match="empty skeleton"):
            module.find_soma(np.zeros((0, 3)), np.zeros(0), 50.0, 3.0, 2)
        with pytest.raises(ValueError, match="in the range"):
            module.find_soma(np.zeros((2, 3)), np.ones(2), 101.0, 3.0, 2)


def test_soma_detection_ends_alike_on_both_paths(reference_mesh):
    skel = sk.skeletonize(reference_mesh, verbose=False)
    rust, python = both(
        sk.post.detect_soma,
        skel,
        mesh_vertices=reference_mesh.vertices,
        soma_radius_percentile_threshold=90,
        soma_radius_distance_factor=4,
        verbose=False,
    )
    assert_parity(_skeleton_fields(rust), _skeleton_fields(python))


def test_gap_bridging_on_both_paths():
    rng = np.random.default_rng(23)
    bridged = relaxed = 0
    for trial in range(120):
        n = int(rng.integers(2, 80))
        # a forest: each node joins an earlier one, or starts a new tree
        edges = [
            [int(rng.integers(0, i)), i] for i in range(1, n) if rng.random() > 0.15
        ]
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        nodes = rng.normal(size=(n, 3)) * 10
        if trial % 3 == 0:
            nodes = np.round(nodes)  # on a grid: equal gaps
        for factor, recalc in ((None, None), (0.5, 1), (2.0, 3)):
            args = (nodes, edges, factor, recalc)
            rust = _rs.skeletonize.bridge_gaps(*args)
            assert_parity(rust, python_skel.bridge_gaps(*args))
            bridged += len(rust) > len(edges)
            relaxed += factor == 0.5 and len(rust) > len(edges)
    assert bridged > 100 and relaxed > 20
    for module in (_rs.skeletonize, python_skel):
        one = np.array([[0, 1], [1, 2]])
        assert module.bridge_gaps(np.zeros((3, 3)), one, None, None).tolist() == [
            [0, 1],
            [1, 2],
        ]
        with pytest.raises(ValueError, match="no nodes"):
            module.bridge_gaps(np.zeros((0, 3)), np.empty((0, 2), int), None, None)


def test_mesh_edges_and_centreline_radii_on_both_paths():
    rng = np.random.default_rng(29)
    for _ in range(100):
        n_vertices = int(rng.integers(1, 300))
        n_nodes = int(rng.integers(1, 40))
        owner = rng.integers(-1, n_nodes, size=n_vertices)
        edges = rng.integers(0, n_vertices, size=(int(rng.integers(0, 600)), 2))
        assert_parity(
            _rs.skeletonize.edges_between(edges, owner),
            python_skel.edges_between(edges, owner),
        )
        lens = rng.integers(0, 6, size=n_nodes)
        ptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
        idx = rng.integers(0, n_vertices, size=int(ptr[-1])).astype(np.int64)
        # some vertices measured twice (the last counts), some not at all
        vids = rng.integers(0, n_vertices, size=int(rng.integers(0, n_vertices + 1)))
        dists = rng.gamma(2.0, size=len(vids))
        assert_parity(
            _rs.skeletonize.centerline_radii(ptr, idx, vids, dists, n_vertices),
            python_skel.centerline_radii(ptr, idx, vids, dists, n_vertices),
        )
    for module in (_rs.skeletonize, python_skel):
        with pytest.raises(ValueError, match="vertex index 5 is out of range"):
            module.centerline_radii(
                np.array([0, 1]), np.array([5]), np.empty(0, int), np.empty(0), 5
            )
        assert module.edges_between(np.empty((0, 2), int), np.zeros(3, int)).shape == (
            0,
            2,
        )
