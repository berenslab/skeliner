"""
Core pipeline smoke-test.

Runs `skeletonize()` on the reference mesh and checks a handful of
topological / numerical invariants so that regressions blow up early.
"""

import importlib
from pathlib import Path

import igraph as ig
import numpy as np
import pytest
import trimesh

from skeliner import _rs, skeletonize
from skeliner.dataclass import (
    Discarded,
    MeshComponents,
    Neurites,
    Organelles,
    Soma,
)
from skeliner.io import load_mesh
from skeliner.skeletonize import (
    _bin_geodesic_shells,
    _direct_on_rust,
    _edges_from_mesh,
    _extreme_vertex,
    _make_nodes,
    _make_nodes_direct_rs,
    _merge_nested_nodes,
    _neighbour_groups,
    _split_branch_band,
    _surface_graph,
    _surface_graph_rs,
)

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


def _direct_bins_both_ways(mesh, seed, step=None):
    py = _bin_geodesic_shells(
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
    rs = _surface_graph_rs(mesh).bin_mesh(seed, e_m, step, 1000, 6, 50)
    return py, rs


def test_direct_components_come_in_igraphs_order(reference_mesh):
    py = [list(c) for c in _surface_graph(reference_mesh).components()]
    rs = [c.tolist() for c in _surface_graph_rs(reference_mesh).components()]
    assert len(rs) == 7 and rs == py


def test_surface_graph_has_trimeshs_unique_edges_and_lengths(reference_mesh):
    """Built from the faces, value for value what trimesh computes, so the
    mean edge and the edge mapping see the same numbers either way."""
    for mesh in [reference_mesh, _dirty_mesh()]:
        g = _surface_graph_rs(mesh)
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
def test_direct_binning_on_the_rust_core_matches_the_python(reference_mesh, step_edges):
    """Auto shell width, and a fixed one of three mean edges; the seed in
    the main component, and in a small one so the main falls back."""
    mesh = reference_mesh
    step = None
    if step_edges is not None:
        step = step_edges * float(mesh.edges_unique_length.mean())
    small = min(_surface_graph_rs(mesh).components(), key=len)
    for seed in [_extreme_vertex(mesh), _extreme_vertex(mesh, "x", "max"), small[0]]:
        py, rs = _direct_bins_both_ways(mesh, int(seed), step)
        assert sum(len(band) for band in rs) > 20
        assert _shells_equal(py, rs), seed


def test_direct_binning_matches_the_python_on_a_dirty_mesh():
    mesh = _dirty_mesh()
    py, rs = _direct_bins_both_ways(mesh, _extreme_vertex(mesh))
    assert _shells_equal(py, rs)


def test_direct_nodes_and_nested_merge_match_the_python(reference_mesh):
    mesh = reference_mesh
    e_m = float(mesh.edges_unique_length.mean())
    shells = _surface_graph_rs(mesh).bin_mesh(
        _extreme_vertex(mesh), e_m, None, 1000, 6, 50
    )
    v = mesh.vertices.view(np.ndarray)
    est = ["median", "mean", "trim"]
    py = _make_nodes(
        [list(band) for band in shells],
        v,
        radius_estimators=est,
        merge_nested=True,
        merge_kwargs={"inside_frac": 0.8},
    )
    rs = _make_nodes_direct_rs(shells, v, est, 0.8)
    assert len(rs[0]) < sum(len(band) for band in shells), "nothing merged"
    assert np.array_equal(py[0], rs[0])
    assert all(np.array_equal(py[1][k], rs[1][k]) for k in est)
    assert len(py[2]) == len(rs[2])
    assert all(np.array_equal(a, b) for a, b in zip(py[2], rs[2]))
    assert py[3] == rs[3]


def test_nested_merge_on_the_rust_core_matches_on_crowded_nodes():
    rng = np.random.default_rng(3)
    nodes = rng.random((400, 3)) * 50
    radii = rng.random(400) * 8
    radii[[5, 9, 40]] = radii[3]  # ties, visited by index
    n2v = [np.arange(3 * k, 3 * k + 3, dtype=np.int64) for k in range(400)]
    keep_py, verts_py, old2new_py = _merge_nested_nodes(
        nodes, radii, list(n2v), inside_frac=0.8
    )
    keep_rs, verts_rs, old2new_rs = _rs.merge_nested_nodes(nodes, radii, n2v, 0.8)
    assert 50 < keep_rs.sum() < 350
    assert np.array_equal(keep_py, keep_rs)
    assert np.array_equal(old2new_py, old2new_rs)
    assert all(np.array_equal(a, b) for a, b in zip(verts_py, verts_rs))


def test_direct_track_is_the_same_on_rust_and_in_python(reference_mesh, monkeypatch):
    rs = skeletonize(reference_mesh)
    module = importlib.import_module("skeliner.skeletonize")
    monkeypatch.setattr(module, "_direct_on_rust", lambda _split: False)
    py = skeletonize(reference_mesh)
    _assert_skeleton_valid(rs)
    assert len(rs.nodes) > 20
    assert np.array_equal(rs.nodes, py.nodes)
    assert np.array_equal(rs.edges, py.edges)
    assert np.array_equal(rs.ntype, py.ntype)
    assert rs.radii.keys() == py.radii.keys()
    assert all(np.array_equal(rs.radii[k], py.radii[k]) for k in rs.radii)
    assert len(rs.node2verts) == len(py.node2verts)
    assert all(np.array_equal(a, b) for a, b in zip(rs.node2verts, py.node2verts))
    for attr in ("center", "axes", "R"):
        assert np.array_equal(getattr(rs.soma, attr), getattr(py.soma, attr))


def test_only_the_legacy_split_keeps_the_direct_track_in_python():
    assert _direct_on_rust(False) and not _direct_on_rust(True)


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
