"""
Smoke-check every public diagnostic helper in `skeliner.dx`.

No hard-coded biology – we just compare against ground-truth values
computed on-the-fly with igraph so the test is independent of the mesh
content.
"""

import copy
import itertools
from pathlib import Path

import numpy as np
import pytest

import skeliner as sk
from skeliner import dx, post, skeletonize
from skeliner.dataclass import Skeleton, Soma
from skeliner.io import load_mesh


# ---------------------------------------------------------------------
# shared fixture: skeleton of the reference mesh
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def mesh():
    return load_mesh(Path(__file__).parent / "data" / "60427.obj")


@pytest.fixture(scope="session")
def skel(mesh):
    return skeletonize(mesh, verbose=False)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _igraph(skel):
    return skel._igraph()


# ---------------------------------------------------------------------
# individual tests
# ---------------------------------------------------------------------
def test_check_connectivity(skel):
    assert dx.check_connectivity(skel)


def test_connectivity_deprecated_alias_warns(skel):
    with pytest.warns(DeprecationWarning, match="check_connectivity"):
        assert dx.connectivity(skel)


def test_check_acyclicity(skel):
    assert dx.check_acyclicity(skel) is True
    # check_acyclicity(..., return_cycles=True) must return a boolean when acyclic
    assert dx.check_acyclicity(skel, return_cycles=True) is True


# ---------------------------------------------------------------------
# cycles
#
# A pipeline skeleton has none — the MST returns a tree — so a loop exists
# only because a connection was asserted.  That makes the loop a statement
# that two of its edges cannot both be right, and `edge_support` says which.
# ---------------------------------------------------------------------
def test_a_tree_has_no_cycles(skel, mesh):
    assert dx.check_acyclicity(skel) is True
    assert dx.cycles(skel, mesh) == []


def test_a_graft_opens_a_loop_that_names_its_own_break(skel, mesh):
    """The grafted edge is the one the surface will not vouch for.

    On a tube, two nodes several bins apart share no surface, so asserting
    an edge between them is exactly the claim `edge_support` refuses — and
    the loop that appears is where a real merge would be adjudicated.
    """
    looped = copy.deepcopy(skel)
    u, v = 2, 6
    post.graft(looped, u, v)

    assert dx.check_acyclicity(looped) is False
    found = dx.cycles(looped, mesh)
    assert len(found) == 1

    loop = found[0]
    assert set(loop["nodes"]) == {2, 3, 4, 5, 6}
    assert len(loop["edges"]) == len(loop["nodes"])
    assert loop["length"] > 0
    assert loop["breaks"] == [(v, u)] or loop["breaks"] == [(u, v)]


def test_a_cycle_walks_in_loop_order(skel, mesh):
    """Consecutive entries must be joined, or "trace round the loop" is a
    lie and the drawn segments jump across the cell."""
    looped = copy.deepcopy(skel)
    post.graft(looped, 2, 6)
    loop = dx.cycles(looped, mesh)[0]

    present = {tuple(sorted(map(int, e))) for e in np.asarray(looped.edges)}
    for a, b in loop["edges"]:
        assert tuple(sorted((a, b))) in present, f"({a},{b}) is not an edge"


def test_the_cycle_basis_is_read_as_edges_not_vertices(skel):
    """igraph's `minimum_cycle_basis` returns **edge** ids; the removed
    `cycle_basis` returned **vertex** ids.

    Reading one as the other yields a list of pairs that looks entirely
    reasonable and is not the cycle, so this pins the conversion against the
    skeleton's own edge list rather than against a shape.
    """
    looped = copy.deepcopy(skel)
    post.graft(looped, 2, 6)

    reported = dx.check_acyclicity(looped, return_cycles=True)
    assert reported is not True
    present = {tuple(sorted(map(int, e))) for e in np.asarray(looped.edges)}
    for a, b in reported:
        assert tuple(sorted((a, b))) in present, (
            f"({a},{b}) is not an edge — basis ids read in the wrong space"
        )


def test_cycles_without_a_mesh_report_the_loop_and_name_no_break(skel):
    """Which edge is wrong is a question about the surface.  With no mesh
    the honest answer is 'here is the loop', not a fallback ranking."""
    looped = copy.deepcopy(skel)
    post.graft(looped, 2, 6)

    loop = dx.cycles(looped)[0]
    assert set(loop["nodes"]) == {2, 3, 4, 5, 6}
    assert loop["breaks"] == []


def test_acyclicity_deprecated_alias_warns(skel):
    with pytest.warns(DeprecationWarning, match="check_acyclicity"):
        assert dx.acyclicity(skel, return_cycles=True) is True


# ---------------------------------------------------------------------
# check_bins
#
# A wrong node2verts / vert2node is the one corruption that is invisible:
# the skeleton looks fine, exports fine, and carries radii belonging to
# the wrong surface.  These tests are what make it loud.
# ---------------------------------------------------------------------
class TestCheckBins:
    def test_a_fresh_skeleton_passes(self, skel):
        assert dx.check_bins(skel) is True

    def test_report_shape(self, skel):
        r = dx.check_bins(skel, return_report=True)
        assert r["ok"] is True
        assert r["n_nodes"] == len(skel.nodes)
        assert r["duplicated"].size == 0
        assert r["mismatched"].size == 0

    def test_a_wrong_vert2node_entry_fails(self, skel):
        bad = copy.deepcopy(skel)
        vid = int(bad.node2verts[5][0])
        bad.vert2node[vid] = 999_999
        assert dx.check_bins(bad) is False
        assert dx.check_bins(bad, return_report=True)["mismatched"].tolist() == [vid]

    def test_a_vertex_missing_from_vert2node_fails(self, skel):
        bad = copy.deepcopy(skel)
        del bad.vert2node[int(bad.node2verts[5][0])]
        assert dx.check_bins(bad) is False

    def test_two_arbor_bins_sharing_a_vertex_fails(self, skel):
        bad = copy.deepcopy(skel)
        stolen = bad.node2verts[7][:3]
        bad.node2verts[6] = np.concatenate([bad.node2verts[6], stolen])
        assert dx.check_bins(bad) is False
        assert dx.check_bins(bad, return_report=True)["duplicated"].size == 3

    def test_node_0_may_overlap_an_arbor_bin(self, skel):
        """Structural, not a defect.

        ``node2verts[0]`` is ``soma.verts`` wholesale while neurites are
        binned over the face-based arbor, so under the >=2-of-3 rule a
        boundary vertex belongs to both.  A check that failed on this
        would fire on every skeleton the pipeline produces.
        """
        ok = copy.deepcopy(skel)
        shared = ok.node2verts[9][:5]
        ok.node2verts[0] = np.concatenate([ok.node2verts[0], shared])
        assert dx.check_bins(ok) is True
        assert dx.check_bins(ok, return_report=True)["soma_overlap"] >= 5

    def test_disowned_surface_is_legal(self, skel):
        """Coverage is not an invariant: ``prune`` drops a twig's vertices
        on the floor, and soma / organelle / discarded surface is never
        owned in the first place."""
        pruned = copy.deepcopy(skel)
        post.prune(pruned, kind="twigs", num_nodes=3)
        assert dx.check_bins(pruned) is True

    def test_length_mismatch_is_caught(self, skel):
        bad = copy.deepcopy(skel)
        bad.node2verts = bad.node2verts[:-1]
        assert dx.check_bins(bad) is False
        assert "length" in dx.check_bins(bad, return_report=True)["reason"]

    def test_no_mesh_data_is_vacuously_ok(self, skel):
        bare = copy.deepcopy(skel)
        bare.node2verts = None
        bare.vert2node = None
        assert dx.check_bins(bare) is True

    def test_half_the_mesh_data_is_not(self, skel):
        bare = copy.deepcopy(skel)
        bare.vert2node = None
        assert dx.check_bins(bare) is False

    def test_fragmented_bins_are_reported_not_failed(self, skel, mesh):
        """The binning's reunite pass is capped at 8 rounds, so real cells
        keep a few multi-piece bins.  Reporting them is useful; failing on
        them would fire on the pipeline's own output."""
        r = dx.check_bins(skel, mesh=mesh, return_report=True)
        assert r["ok"] is True
        assert isinstance(r["fragmented"], dict)
        assert all(v > 1 for v in r["fragmented"].values())

    def test_face_owner_uses_the_same_rule_as_the_soma(self, skel, mesh):
        """Bins own vertices; everything that looks at a mesh works in
        faces.  The bridge is the >=2-of-3 majority rule, the same one
        ``pre.soma_face_mask`` uses."""
        owner = dx.face_owner(skel, mesh)
        assert owner.shape == (len(mesh.faces),)

        faces = np.asarray(mesh.faces)
        for fi in range(0, len(faces), max(1, len(faces) // 200)):
            owners = [skel.vert2node.get(int(v), -1) for v in faces[fi]]
            counts = {o: owners.count(o) for o in set(owners)}
            want = next((o for o, c in counts.items() if c >= 2), -1)
            assert owner[fi] == want, f"face {fi}: {owners}"

    def test_bin_faces_partition_the_owned_surface(self, skel, mesh):
        owner = dx.face_owner(skel, mesh)
        seen = np.zeros(len(mesh.faces), dtype=bool)
        for node in range(len(skel.nodes)):
            f = dx.bin_faces(skel, mesh, node, owner=owner)
            assert not seen[f].any(), f"node {node} claims a face already claimed"
            seen[f] = True
        assert seen.sum() == int((owner >= 0).sum())

    def test_bin_faces_accepts_a_cached_owner(self, skel, mesh):
        owner = dx.face_owner(skel, mesh)
        assert np.array_equal(
            dx.bin_faces(skel, mesh, 2), dx.bin_faces(skel, mesh, 2, owner=owner)
        )

    def test_face_owner_without_mesh_data(self, skel, mesh):
        bare = copy.deepcopy(skel)
        bare.vert2node = None
        assert (dx.face_owner(bare, mesh) == -1).all()

    def test_a_bin_split_across_the_surface_is_detected(self, skel, mesh):
        bad = copy.deepcopy(skel)
        # hand node 3 a vertex from the far end of the arbor: same bin,
        # two patches, no shared surface between them
        far = int(bad.node2verts[len(bad.node2verts) - 1][0])
        bad.node2verts[3] = np.append(bad.node2verts[3], far)
        bad.node2verts[len(bad.node2verts) - 1] = bad.node2verts[
            len(bad.node2verts) - 1
        ][1:]
        bad.vert2node[far] = 3
        r = dx.check_bins(bad, mesh=mesh, return_report=True)
        assert r["ok"] is True, "still a valid partition — just an ugly one"
        assert r["fragmented"].get(3, 1) > 1


# ---------------------------------------------------------------------
# edge_support
#
# Which node pairs the mesh joins, against which ones the tree carries.
# This is a classifier for the edge-editing verbs — restore (the surface
# supports it) versus graft (it does not) — and nothing more.  It is
# deliberately not a defect report: see the docstring.
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def torus():
    """A mesh whose surface graph closes a cycle the tree has to cut.

    A tube gives a path, so its ``G`` and ``T`` are identical and there is
    nothing to say.  A ring is the smallest thing with a dropped edge.
    """
    import trimesh

    return trimesh.creation.torus(
        major_radius=300.0, minor_radius=60.0, major_sections=64, minor_sections=16
    )


@pytest.fixture(scope="session")
def torus_skel(torus):
    return skeletonize(torus, verbose=False)


@pytest.fixture
def tangled(skel):
    """A bin owning surface in two places, so several pairs are dropped.

    Half of one bin is handed to a distant node, which is what a densely
    packed arbor looks like from the graph's side: bins that share surface
    while sitting far apart in the tree.
    """
    bad = copy.deepcopy(skel)
    far = len(bad.nodes) - 1
    take = np.asarray(bad.node2verts[3])[: len(bad.node2verts[3]) // 2]
    bad.node2verts[3] = np.setdiff1d(np.asarray(bad.node2verts[3]), take)
    bad.node2verts[far] = np.union1d(np.asarray(bad.node2verts[far]), take)
    for v in take:
        bad.vert2node[int(v)] = far
    return bad


class TestEdgeSupport:
    def test_a_tree_shaped_arbor_drops_nothing(self, skel, mesh):
        """The common case: every adjacency the surface has is in the tree,
        so no pair is a restore and every graft is honestly a graft."""
        rep = dx.edge_support(skel, mesh)
        assert rep["dropped"] == []
        assert rep["n_tree"] == len({tuple(sorted(map(int, e))) for e in skel.edges})

    def test_a_ring_drops_the_one_edge_that_closes_it(self, torus_skel, torus):
        rep = dx.edge_support(torus_skel, torus)
        assert len(rep["dropped"]) == 1

    def test_a_dropped_pair_is_real_surface_adjacency(self, torus_skel, torus):
        """Not an invention: the two bins really do share mesh edges, which
        is the whole difference between a restore and a graft."""
        u, v = dx.edge_support(torus_skel, torus)["dropped"][0]
        v2n = torus_skel.vert2node
        touching = [
            (a, b)
            for a, b in np.asarray(torus.edges_unique)
            if {v2n.get(int(a), -1), v2n.get(int(b), -1)} == {u, v}
        ]
        assert touching, "a dropped pair names bins that do not touch"

    def test_dropped_and_the_tree_are_disjoint_and_canonical(self, tangled, mesh):
        rep = dx.edge_support(tangled, mesh)
        tree = {tuple(sorted(map(int, e))) for e in tangled.edges}
        assert rep["dropped"]
        for u, v in rep["dropped"]:
            assert (u, v) not in tree
            assert u < v

    def test_unsupported_are_tree_edges_with_no_surface(self, skel, mesh):
        """The soma stems and the ``bridge_gaps`` bridges — the edges a
        re-span of ``G`` would silently delete."""
        rep = dx.edge_support(skel, mesh)
        tree = {tuple(sorted(map(int, e))) for e in skel.edges}
        assert rep["unsupported"], "reference skeleton has soma stems"
        for u, v in rep["unsupported"]:
            assert (u, v) in tree
        assert not set(rep["unsupported"]) & set(rep["dropped"])

    def test_clipping_makes_the_cut_edge_a_dropped_pair(self, torus_skel, torus):
        """The two halves of the classification meet: cut a supported tree
        edge and it moves from the tree into the surface-supported set, so
        putting it back is offered as a restore rather than as a graft."""
        cut = copy.deepcopy(torus_skel)
        u, v = (int(x) for x in cut.edges[len(cut.edges) // 2])
        post.clip(cut, u, v)
        rep = dx.edge_support(cut, torus)
        assert (min(u, v), max(u, v)) in rep["dropped"]

    def test_a_skeleton_without_bins_is_refused(self, skel, mesh):
        bare = copy.deepcopy(skel)
        bare.vert2node = None
        with pytest.raises(ValueError, match="vert2node"):
            dx.edge_support(bare, mesh)


def test_degree_and_neighbors_match_igraph(skel):
    g = _igraph(skel)
    degrees_ref = g.degree()
    # vector query
    assert np.array_equal(
        [dx.degree(skel, node_id=n) for n in range(len(skel.nodes))], degrees_ref
    )
    # scalar query + neighbors
    nid = 0  # arbitrary but deterministic
    assert dx.degree(skel, nid) == degrees_ref[nid]
    assert set(dx.neighbors(skel, nid)) == set(g.neighbors(nid))


def test_nodes_of_degree(skel):
    g = _igraph(skel)
    deg = np.asarray(g.degree())
    for k in (0, 1, 2, 3):  # 0 included on purpose – should be empty
        expected = {int(i) for i in np.where(deg == k)[0] if i != 0}
        got = set(dx.nodes_of_degree(skel, k))
        assert got == expected


def test_branches_and_twigs_lengths(skel):
    # We do not assume k actually exists – just assert path lengths.
    for k in (1, 2, 3):
        for path in dx.branches_of_length(skel, k):
            assert len(path) == k
        for twig in dx.twigs_of_length(skel, k):
            assert len(twig) == k


def test_suspicious_tips_are_leaves(skel):
    tips = dx.suspicious_tips(skel)  # may be empty
    if not tips:
        return
    g = _igraph(skel)
    deg = np.asarray(g.degree())
    assert all(deg[t] == 1 and t != 0 for t in tips)


# ---------------------------------------------------------------------
# suspicious_junctions — the arm decomposition, and the threshold's unit
# ---------------------------------------------------------------------
def _junction_skeleton(arm_lengths, *, unit="nm"):
    """A soma, a stalk, and one junction carrying arms of exact length.

    Node 1 is the junction: one proximal arm (the stalk back to the soma)
    and ``len(arm_lengths)`` distal arms, each a straight chain of the given
    total length laid along its own axis so no two arms overlap.
    """

    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # soma, junction
    edges = [[0, 1]]
    for k, length in enumerate(arm_lengths):
        angle = 2.0 * np.pi * k / len(arm_lengths)
        step = np.asarray([0.0, np.cos(angle), np.sin(angle)]) * (length / 2.0)
        prev = 1
        for hop in (1, 2):  # two hops, so each arm has an interior node
            nodes.append((np.asarray([1.0, 0.0, 0.0]) + step * hop).tolist())
            edges.append([prev, len(nodes) - 1])
            prev = len(nodes) - 1

    nodes = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray([sorted(e) for e in edges], dtype=np.int64)
    return Skeleton(
        soma=Soma.from_sphere(center=np.zeros(3), radius=0.5, verts=None),
        nodes=nodes,
        radii={"median": np.full(len(nodes), 0.1)},
        edges=edges,
        ntype=None,
        meta={"unit": unit},
    )


def _fork_skeleton():
    """Soma 0; arm 0-1-2-3 along +x with a twig 2-4; arm 0-5-6 along -x."""

    nodes = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0], [-1, 0, 0], [-2, 0, 0]],
        dtype=np.float64,
    )
    edges = np.asarray([[0, 1], [1, 2], [2, 3], [2, 4], [0, 5], [5, 6]], dtype=np.int64)
    return Skeleton(
        soma=Soma.from_sphere(center=np.zeros(3), radius=0.5, verts=None),
        nodes=nodes,
        radii={"median": np.full(len(nodes), 0.1)},
        edges=edges,
        ntype=None,
    )


def test_extract_neurites_is_the_distal_subtree():
    s = _fork_skeleton()
    assert dx.extract_neurites(s, 1) == [1, 2, 3, 4]
    assert dx.extract_neurites(s, 2, include_root=False) == [3, 4]
    assert dx.extract_neurites(s, 0, include_root=False) == [1, 2, 3, 4, 5, 6]


def test_neurites_out_of_bounds_cut_at_the_first_node_outside():
    s = _fork_skeleton()
    box = ([-1.5, -1.0, -1.0], [1.5, 2.0, 1.0])  # nodes 2, 3, 4 and 6 outside
    assert dx.neurites_out_of_bounds(s, box) == [2, 3, 4, 6]
    assert dx.neurites_out_of_bounds(s, box, include_root=False) == [3, 4]
    assert dx.neurites_out_of_bounds(s, ([-5.0] * 3, [5.0] * 3)) == []


def test_neurite_walks_on_a_forest_keep_to_the_somas_tree():
    """A tree the soma does not reach has no distal direction: its nodes have
    no subtree, and each one outside the box starts its own.  The hop count
    to them is infinite, which used to fail as an integer."""

    s = _fork_skeleton()
    s = Skeleton(
        soma=s.soma,
        nodes=np.vstack([s.nodes, [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]]),
        radii={"median": np.full(len(s.nodes) + 2, 0.1)},
        edges=np.vstack([s.edges, [[7, 8]]]),
        ntype=None,
    )
    assert dx.extract_neurites(s, 1) == [1, 2, 3, 4]
    assert dx.extract_neurites(s, 7) == [7]
    assert dx.extract_neurites(s, 0, include_root=False) == [1, 2, 3, 4, 5, 6]
    box = ([-1.5, -1.0, -1.0], [1.5, 2.0, 1.0])  # nodes 2, 3, 4, 6, 7, 8 outside
    assert dx.neurites_out_of_bounds(s, box) == [2, 3, 4, 6, 7, 8]
    assert dx.neurites_out_of_bounds(s, box, include_root=False) == [3, 4]


def test_the_arms_of_a_junction_partition_all_the_cable(skel):
    """Every measurement is one arm's share of a total that stays whole.

    An arm decomposition that double-counts a subtree, or drops the edge it
    was reached through, still looks plausible per-arm — it only shows up
    against the total.
    """
    stats = dx.suspicious_junctions(skel, min_degree=3, return_stats=True)[1]
    assert stats, "fixture has no junction of degree >= 3 to decompose"

    edges = np.asarray(skel.edges).reshape(-1, 2)
    total = float(
        np.linalg.norm(skel.nodes[edges[:, 0]] - skel.nodes[edges[:, 1]], axis=1).sum()
    )
    for nid, s in stats.items():
        got = s["proximal_cable"] + sum(s["distal_cables"])
        assert got == pytest.approx(total, rel=1e-9), f"node {nid} loses cable"
        assert len(s["distal_cables"]) == s["degree"] - 1


def test_the_linear_pass_agrees_with_deleting_the_node(skel):
    """The O(N) subtree form against the O(N^2) form it replaces.

    The naive version — delete the node, flood each remaining component —
    is the definition; the subtree accumulation is an optimisation of it,
    and only this pins them together.
    """
    import collections

    stats = dx.suspicious_junctions(skel, min_degree=3, return_stats=True)[1]
    assert stats, "fixture has no junction of degree >= 3 to decompose"

    edges = np.asarray(skel.edges).reshape(-1, 2)
    length = {
        tuple(sorted((int(a), int(b)))): float(
            np.linalg.norm(skel.nodes[a] - skel.nodes[b])
        )
        for a, b in edges
    }
    adj = collections.defaultdict(list)
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    for nid, s in stats.items():
        seen, arms = {nid}, []
        for seed in adj[nid]:
            if seed in seen:
                continue
            seen.add(seed)
            stack, comp = [seed], {seed}
            while stack:
                v = stack.pop()
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w)
                        comp.add(w)
                        stack.append(w)
            cable = sum(v for (a, b), v in length.items() if a in comp and b in comp)
            cable += length[tuple(sorted((nid, seed)))]
            arms.append((comp, cable))

        proximal = [c for c in arms if 0 in c[0]]
        distal = sorted((c[1] for c in arms if 0 not in c[0]), reverse=True)
        assert proximal[0][1] == pytest.approx(s["proximal_cable"], rel=1e-9)
        assert distal == pytest.approx(s["distal_cables"], rel=1e-9)


def test_the_threshold_is_read_in_the_unit_it_declares():
    """250 um on a nanometre skeleton is 250_000, not 250.

    Skeletons from `skeletonize` are in nm by default, so a bare number
    ported from a micrometre-based source is off by 1000x — and in the
    direction that flags everything, which reads as a working detector.
    """
    # arms of 300 um, 200 um and 0.1 um, on a skeleton measured in nm
    skel = _junction_skeleton([300_000.0, 200_000.0, 100.0], unit="nm")

    # 250 um: only the 300 um arm clears, so one substantial arm, not two.
    assert dx.suspicious_junctions(skel, min_distal_cable=250.0, cable_unit="um") == []
    # 150 um: the 300 and 200 um arms both clear.
    flagged = dx.suspicious_junctions(skel, min_distal_cable=150.0, cable_unit="um")
    assert flagged == [1], "two arms clear 150 um; the junction should flag"

    # The identical number read as nanometres is 1000x smaller, so the very
    # same call that found nothing now flags — this is the porting mistake.
    assert dx.suspicious_junctions(skel, min_distal_cable=250.0, cable_unit="nm") == [1]


def test_a_skeleton_without_a_unit_refuses_rather_than_assuming_one():
    skel = _junction_skeleton([300_000.0, 200_000.0], unit="nm")
    skel.meta.pop("unit")
    with pytest.raises(ValueError, match="unit"):
        dx.suspicious_junctions(skel)


def test_a_junction_needs_enough_substantial_arms():
    """One long arm is a bend, not a merge."""
    one_long = _junction_skeleton([300_000.0, 100.0, 100.0], unit="nm")
    assert dx.suspicious_junctions(one_long, min_distal_cable=100.0) == []

    two_long = _junction_skeleton([300_000.0, 300_000.0, 100.0], unit="nm")
    assert dx.suspicious_junctions(two_long, min_distal_cable=100.0) == [1]


def test_adjacent_flagged_nodes_collapse_to_one_row():
    """A merge flags a cluster; a review queue should show it once."""
    skel = _junction_skeleton([300_000.0, 300_000.0, 300_000.0], unit="nm")
    # Split the junction in two: node 1 keeps two arms, a new neighbour
    # takes the third, so both are degree-4 and adjacent.
    grouped = dx.suspicious_junctions(
        skel, min_degree=3, min_distal_cable=100.0, group_regions=True
    )
    ungrouped = dx.suspicious_junctions(
        skel, min_degree=3, min_distal_cable=100.0, group_regions=False
    )
    assert set(grouped) <= set(ungrouped)
    assert len(grouped) <= len(ungrouped)
    assert grouped == [1]


def test_detection_does_not_touch_the_skeleton(skel):
    """It reports candidates and never clips."""
    before_nodes = skel.nodes.copy()
    before_edges = np.asarray(skel.edges).copy()
    dx.suspicious_junctions(skel, min_degree=3, min_distal_cable=0.001)
    assert np.array_equal(skel.nodes, before_nodes)
    assert np.array_equal(np.asarray(skel.edges), before_edges)


def test_distance_point_queries(skel):
    unit = skel.meta.get("unit", "nm")
    soma = skel.nodes[0]
    # distance to a node should be zero irrespective of units
    assert dx.distance(skel, soma, point_unit=unit) == pytest.approx(0.0, abs=1e-9)

    # take edge midpoint and move it away along a perpendicular direction
    u, v = map(int, skel.edges[0])
    edge_vec = skel.nodes[v] - skel.nodes[u]
    mid = 0.5 * (skel.nodes[u] + skel.nodes[v])

    # robust perpendicular
    perp = None
    for axis in np.eye(3):
        candidate = np.cross(edge_vec, axis)
        norm = np.linalg.norm(candidate)
        if norm > 1e-9:
            perp = candidate / norm
            break
    if perp is None:  # degenerate edge, fall back to arbitrary axis
        perp = np.array([1.0, 0.0, 0.0])

    offset_nm = perp * 500.0  # 500 nm away from the edge
    point_nm = mid + offset_nm

    radius_key = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_key], dtype=float)

    def brute_centerline(point_nm_space: np.ndarray) -> float:
        """Distance to the centreline (no radii)."""
        d_nodes = np.linalg.norm(skel.nodes - point_nm_space, axis=1).min()
        d_edges = np.inf
        for a, b in skel.edges:
            d_edges = min(
                d_edges,
                dx._point_segment_distance(
                    point_nm_space, skel.nodes[a], skel.nodes[b]
                ),
            )
        return min(d_nodes, d_edges)

    def brute_surface(point_nm_space: np.ndarray) -> float:
        """Distance to the capsule envelope (clamped to zero inside)."""
        d_nodes = float(
            np.min(np.linalg.norm(skel.nodes - point_nm_space, axis=1) - radii)
        )
        d_edges = np.inf
        for a, b in skel.edges:
            d_edges = min(
                d_edges,
                dx._point_segment_capsule_distance(
                    point_nm_space,
                    skel.nodes[a],
                    skel.nodes[b],
                    radii[a],
                    radii[b],
                ),
            )
        return max(min(d_nodes, d_edges), 0.0)

    expected_center_nm = brute_centerline(point_nm)
    expected_surface_nm = brute_surface(point_nm)

    # --- centreline mode -------------------------------------------------
    d_center_nm = dx.distance(skel, point_nm, point_unit="nm", mode="centerline")
    assert d_center_nm == pytest.approx(expected_center_nm, rel=1e-6)

    point_um = point_nm * 1e-3
    d_center_um = dx.distance(skel, point_um, point_unit="um", mode="centerline")
    assert d_center_um == pytest.approx(expected_center_nm * 1e-3, rel=1e-6)

    # --- surface mode ----------------------------------------------------
    d_surface_nm = dx.distance(skel, point_nm, point_unit="nm", mode="surface")
    assert d_surface_nm == pytest.approx(expected_surface_nm, rel=1e-6)

    point_um_surface = dx.distance(skel, point_um, point_unit="um", mode="surface")
    assert point_um_surface == pytest.approx(expected_surface_nm * 1e-3, rel=1e-6)

    # vectorised query mixes modes
    arr_nm = np.vstack([point_nm, mid])
    distances_surface = dx.distance(skel, arr_nm, point_unit="nm", mode="surface")
    assert distances_surface.shape == (2,)
    assert distances_surface[0] == pytest.approx(expected_surface_nm, rel=1e-6)
    assert distances_surface[1] == pytest.approx(0.0, abs=1e-9)

    distances_center = dx.distance(skel, arr_nm, point_unit="nm", mode="centerline")
    assert distances_center.shape == (2,)
    assert distances_center[0] == pytest.approx(expected_center_nm, rel=1e-6)
    assert distances_center[1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------
# check_mesh_pairing — is this the mesh the bins were built over?
# ---------------------------------------------------------------------
# A skeleton's bins name vertex *ids*, which say nothing about which mesh
# they index.  Paired with a larger unrelated mesh every id stays in range,
# so bins resolve to real faces and every answer downstream is confidently
# wrong.  Nothing else in `dx` notices: `check_bins` compares node2verts
# against vert2node, and both are internally consistent with each other no
# matter which mesh is standing next to them.


@pytest.fixture(scope="session")
def bigger_mesh(mesh):
    """An unrelated mesh with strictly more vertices than the reference."""
    import trimesh

    other = trimesh.creation.icosphere(subdivisions=5, radius=1000.0)
    assert len(other.vertices) > len(mesh.vertices)
    return other


@pytest.fixture(scope="session")
def smaller_mesh():
    import trimesh

    return trimesh.creation.icosphere(subdivisions=1, radius=1000.0)


def test_skeletonize_records_the_mesh_it_measured(skel, mesh):
    """Written at skeletonize time or not at all — a skeleton cannot be told
    afterwards which mesh it came from."""
    assert skel.meta["mesh"] == {
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
    }


def test_the_right_mesh_is_verified(skel, mesh):
    rep = dx.check_mesh_pairing(skel, mesh, return_report=True)
    assert rep["ok"] and rep["verified"]


def test_a_larger_unrelated_mesh_is_refused(skel, bigger_mesh):
    """The dangerous direction: every vertex id is in range, so nothing
    raises and every bin resolves to real faces of the wrong cell."""
    assert dx.check_mesh_pairing(skel, bigger_mesh) is False
    rep = dx.check_mesh_pairing(skel, bigger_mesh, return_report=True)
    assert str(len(bigger_mesh.vertices)) in rep["reason"].replace(",", "")


def test_a_smaller_unrelated_mesh_is_refused_without_the_counts(skel, smaller_mesh):
    """The bounds half stands alone, which is what covers skeletons written
    before the counts existed."""
    legacy = copy.deepcopy(skel)
    legacy.meta.pop("mesh")
    assert dx.check_mesh_pairing(legacy, smaller_mesh) is False


def test_a_legacy_skeleton_is_not_contradicted_but_not_confirmed(
    skel, mesh, bigger_mesh
):
    """`ok` and `verified` say different things, and collapsing them would
    present every pre-existing npz as confirmed."""
    legacy = copy.deepcopy(skel)
    legacy.meta.pop("mesh")

    right = dx.check_mesh_pairing(legacy, mesh, return_report=True)
    assert right["ok"] and not right["verified"]

    # Bounds alone cannot see this one; saying so is the point.
    wrong = dx.check_mesh_pairing(legacy, bigger_mesh, return_report=True)
    assert wrong["ok"] and not wrong["verified"]


def test_face_owner_names_the_mismatch_instead_of_indexerror(skel, smaller_mesh):
    """Left to numpy this is an IndexError naming an array the caller never
    passed in."""
    with pytest.raises(ValueError, match="not built from this mesh"):
        dx.face_owner(skel, smaller_mesh)


def test_check_bins_cannot_see_a_wrong_mesh(skel, bigger_mesh):
    """Why this check has to exist separately: node2verts and vert2node are
    consistent with each other whatever mesh is loaded."""
    assert dx.check_bins(skel) is True
    assert dx.check_bins(skel, mesh=bigger_mesh) is True


def test_segment_distances_on_both_paths_agree():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    from ._parity import assert_parity

    rng = np.random.default_rng(1)
    nodes = rng.normal(size=(50, 3)) * 100
    points = rng.normal(size=(500, 3)) * 100
    centres, segments = [3, 7, 11], [(0, 1), (1, 2), (5, 9), (20, 21), (4, 4)]
    assert_parity(
        _rs.dx.segment_distances(points, nodes, centres, segments),
        python_dx.segment_distances(points, nodes, centres, segments),
    )


def _index_case(rng, n):
    """A random tree of *n* nodes and its neighbours as a CSR."""
    nodes = rng.normal(size=(n, 3)) * 10
    edges = np.array([[int(rng.integers(0, i)), i] for i in range(1, n)]).reshape(-1, 2)
    nbrs = [[] for _ in range(n)]
    for a, b in edges.tolist():
        nbrs[a].append(b)
        nbrs[b].append(a)
    ptr = np.r_[0, np.cumsum([len(x) for x in nbrs])].astype(np.int64)
    idx = np.array([v for x in nbrs for v in x], dtype=np.int64)
    return nodes, edges, ptr, idx


def test_distances_on_both_paths():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    from ._parity import assert_parity

    rng = np.random.default_rng(31)
    for trial in range(60):
        n = int(rng.integers(1, 80))
        nodes, edges, ptr, idx = _index_case(rng, n)
        if trial % 3 == 0:
            nodes = np.round(nodes / 3)  # a grid: equally near nodes
        points = rng.normal(size=(40, 3)) * 12
        radii = rng.gamma(2.0, size=n)
        indices = [
            _rs.dx.SkeletonIndex(nodes, ptr, idx),
            python_dx.SkeletonIndex(nodes, ptr, idx),
        ]
        some = rng.choice(n, size=min(n, 5), replace=False).tolist()
        cases = [
            (r, k, None, None, s)
            for r in (radii, None)
            for k in (1, 4, 9)
            for s in (1.0, 0.5)
        ] + [
            (radii, 4, sorted(some), [tuple(e) for e in edges[:3].tolist()], 2.0),
            (None, 4, [], [], 1.0),
            (radii, 4, sorted(some), [], 1.0),
        ]
        for r, k, centres, segments, s in cases:
            got = [
                ix.distance(points, r, k, centres, segments, s, 1.0 / s)
                for ix in indices
            ]
            assert_parity(got[0], got[1])
    for module in (_rs.dx, python_dx):
        with pytest.raises(ValueError, match="no nodes"):
            module.SkeletonIndex(np.zeros((0, 3)), np.zeros(1, int), np.empty(0, int))


@pytest.mark.parametrize("mode", ["surface", "centerline"])
def test_distance_ends_alike_on_both_paths(skel, mode):
    from ._parity import both

    rng = np.random.default_rng(37)
    lo, hi = skel.nodes.min(0), skel.nodes.max(0)
    points = lo + rng.random((200, 3)) * (hi - lo)
    allowed = list(range(0, len(skel.nodes), 3))
    for kwargs in ({}, {"allowed_nodes": allowed}, {"allowed_edges": skel.edges[:5]}):
        rust, python = both(dx.distance, skel, points, mode=mode, **kwargs)
        np.testing.assert_array_equal(rust, python)
    # the cached index survives a deep copy (and so pickling)
    copied = copy.deepcopy(skel)
    np.testing.assert_array_equal(
        dx.distance(copied, points, mode=mode), dx.distance(skel, points, mode=mode)
    )


def test_voxel_union_on_both_paths():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    rng = np.random.default_rng(41)
    filled = 0
    for trial in range(40):
        n = int(rng.integers(2, 25))
        nodes = rng.normal(size=(n, 3)) * 8
        nodes[1] = nodes[0]  # a degenerate edge, a ball
        edges = np.array([[int(rng.integers(0, i)), i] for i in range(1, n)])
        radii = rng.gamma(2.0, size=n)
        if trial % 7 == 0:
            radii[2 % n] = np.nan  # skipped, as numpy skips it
        soma = Soma.fit(nodes[0] + rng.normal(size=(12, 3)) * 3)
        lo, hi = (
            soma.center - np.abs(soma.R) @ soma.axes,
            soma.center + np.abs(soma.R) @ soma.axes,
        )
        soma_in = (
            soma.center.tolist(),
            soma.axes.tolist(),
            soma.R.T.tolist(),
            lo.tolist(),
            hi.tolist(),
        )
        box_lo = nodes.min(0) - 2
        box_hi = nodes.max(0) + 2
        if trial % 3 == 0:  # a box cutting through the cell
            box_hi = (box_lo + box_hi) / 2
        h = float(rng.uniform(0.4, 1.5))
        shape = np.maximum(np.ceil((box_hi - box_lo) / h).astype(int), 1).tolist()
        for s, include in ((soma_in, True), (soma_in, False), (None, True)):
            args = (
                nodes,
                radii,
                edges,
                box_lo.tolist(),
                box_hi.tolist(),
                h,
                shape,
                s,
                include,
            )
            rust = _rs.dx.voxel_union_count(*args)
            assert rust == python_dx.voxel_union_count(*args)
            filled += rust > 0
    assert filled > 80


def test_volume_ends_alike_on_both_paths(skel):
    from ._parity import assert_parity, both

    # a box around a node half way along the cell, which cuts through it
    c = skel.nodes[len(skel.nodes) // 2]
    half = (skel.nodes.max(0) - skel.nodes.min(0)) / 8
    for kwargs in ({}, {"bbox": (c - half, c + half)}, {"include_soma": False}):
        rust, python = both(dx.volume, skel, return_details=True, **kwargs)
        assert_parity(rust, python)
        assert rust[1]["filled_voxels"] > 0


def test_path_length_in_a_box_on_both_paths(skel):
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    from ._parity import assert_parity, both

    rng = np.random.default_rng(43)
    hit = 0
    for _ in range(80):
        n = int(rng.integers(2, 40))
        nodes = rng.normal(size=(n, 3)) * 10
        if n > 3:
            nodes[3] = nodes[2] + [5.0, 0.0, 0.0]  # parallel to two slabs
        edges = rng.integers(0, n, size=(int(rng.integers(1, 3 * n)), 2))
        c, half = rng.normal(size=3) * 4, rng.uniform(1, 15, size=3)
        args = (nodes, edges, (c - half).tolist(), (c + half).tolist())
        rust = _rs.dx.path_length_in_box(*args)
        assert_parity(rust, python_dx.path_length_in_box(*args))
        hit += rust[1] > 0
    assert hit > 40
    c = skel.nodes[len(skel.nodes) // 2]
    half = (skel.nodes.max(0) - skel.nodes.min(0)) / 8
    rust, python = both(
        dx.total_path_length, skel, bbox=(c - half, c + half), return_details=True
    )
    assert_parity(rust, python)
    assert rust[0] > 0


def test_arm_cables_on_both_paths(skel):
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    from ._parity import assert_parity, both

    rng = np.random.default_rng(47)
    junctions = 0
    for _ in range(60):
        n = int(rng.integers(2, 60))
        nodes = rng.normal(size=(n, 3)) * 10
        # a forest with branch points, a self-loop and a doubled edge
        edges = [
            [int(rng.integers(0, max(1, i // 2))), i]
            for i in range(1, n)
            if rng.random() > 0.1
        ]
        edges += [[0, 0], edges[0]] if edges else []
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        for min_degree in (2, 3, 4):
            rust = _rs.dx.arm_cables(nodes, edges, min_degree)
            assert_parity(rust, python_dx.arm_cables(nodes, edges, min_degree))
            junctions += len(rust[0])
    assert junctions > 200
    rust, python = both(dx.suspicious_junctions, skel, min_degree=3, return_stats=True)
    assert_parity(rust, python)


def _random_multigraph(rng, n, *, cycles):
    """A forest on *n* nodes with branch points; with *cycles*, extra edges, a
    doubled edge and a self-loop now and then. Edges in either orientation."""
    edges = [
        [int(rng.integers(0, max(1, i // 2))), i]
        for i in range(1, n)
        if rng.random() > 0.1
    ]
    if cycles:
        edges += rng.integers(0, n, size=(int(rng.integers(0, 5)), 2)).tolist()
        if edges and rng.random() < 0.5:
            edges.append(list(edges[int(rng.integers(0, len(edges)))]))
        if rng.random() < 0.3:
            v = int(rng.integers(0, n))
            edges.append([v, v])
    edges = [e[::-1] if rng.random() < 0.5 else e for e in edges]
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def test_graph_walks_on_both_paths():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    from ._parity import assert_parity

    rng = np.random.default_rng(53)
    found = {"cycles": 0, "branches": 0, "twigs": 0, "neurite": 0, "outside": 0}
    for trial in range(400):
        n = int(rng.integers(1, 40))
        edges = _random_multigraph(rng, n, cycles=trial % 3 != 0)
        nodes = rng.normal(size=(n, 3)) * 5
        lo = rng.normal(size=3) * 3 - 4
        hi = lo + rng.uniform(2, 12, size=3)
        calls = [
            ("components", (edges, n)),
            ("hops", (edges, n, int(rng.integers(0, n)))),
            ("cycle_basis", (edges, n)),
            ("neurites_out_of_bounds", (nodes, edges, lo, hi, trial % 2 == 0)),
        ]
        for k in range(0, 6):
            calls.append(("branches", (edges, n, k, k % 2 == 0)))
            calls.append(("twigs", (edges, n, k, k % 3 == 0)))
        for root in rng.integers(0, n, size=3).tolist():
            calls.append(("neurite", (edges, n, root, root % 2 == 0)))
        for name, args in calls:
            rust = getattr(_rs.dx, name)(*args)
            assert_parity(rust, getattr(python_dx, name)(*args), name)
            if name in ("cycle_basis", "branches", "twigs"):
                found[name if name != "cycle_basis" else "cycles"] += len(rust)
            elif name == "neurite":
                found["neurite"] += len(rust) > 1
            elif name == "neurites_out_of_bounds":
                found["outside"] += len(rust) > 0
    assert min(found.values()) > 100, found


def _tied_cycle_graphs():
    """Graphs with many minimum cycle bases, so which one igraph returns
    turns on the edge ids: complete graphs, the cube, the Petersen graph, a
    bundle of parallel edges, dense random multigraphs."""
    for k in (4, 5, 6):
        yield f"K{k}", k, np.array(list(itertools.combinations(range(k), 2)))
    cube = [(a, a ^ bit) for a in range(8) for bit in (1, 2, 4) if a < a ^ bit]
    yield "cube", 8, np.array(cube)
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    star = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    yield "petersen", 10, np.array(outer + spokes + star)
    yield "bundle", 3, np.array([[0, 1]] * 5 + [[1, 2]] * 3 + [[0, 2]])
    rng = np.random.default_rng(3)
    for t in range(3):
        yield f"dense{t}", 9, rng.integers(0, 9, size=(30, 2))


def test_the_cycle_basis_picks_igraphs_among_tied_cycles():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    rng = np.random.default_rng(0)
    for name, n, edges in _tied_cycle_graphs():
        picked = set()
        for _ in range(12):
            perm = rng.permutation(len(edges))  # edge ids decide among ties
            e = edges[perm]
            flip = rng.random(len(e)) < 0.5
            e[flip] = e[flip, ::-1]
            rust = [c.tolist() for c in _rs.dx.cycle_basis(e, n)]
            assert rust == [c.tolist() for c in python_dx.cycle_basis(e, n)], name
            picked.add(frozenset(frozenset(perm[c].tolist()) for c in rust))
        assert len(picked) > 1, name  # the ties did decide


def test_path_lengths_on_both_paths():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    from ._parity import assert_parity

    rng = np.random.default_rng(59)
    reached = 0
    for trial in range(300):
        n = int(rng.integers(2, 40))
        edges = _random_multigraph(rng, n, cycles=trial % 2 == 0)
        # integer lengths tie paths; some zero, some infinite (never taken)
        weights = rng.integers(0, 4, size=len(edges)).astype(np.float64)
        weights += rng.random(len(edges)) * (trial % 3 == 0)
        weights[rng.random(len(edges)) < 0.05] = np.inf
        sources = rng.integers(0, n, size=int(rng.integers(0, 6)))
        target = int(rng.integers(0, n))
        args = (edges, n, weights, sources, target)
        rust = _rs.dx.path_lengths(*args)
        assert_parity(rust, python_dx.path_lengths(*args))
        reached += int(np.isfinite(rust).sum())
    assert reached > 300


def test_graph_kernels_refuse_alike():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    w = np.ones(2)
    cases = [
        ("hops", (edges, 3, 3), "source 3 is not one of 3 nodes"),
        ("hops", (edges, 3, -1), "source -1 is not one of 3 nodes"),
        ("neurite", (edges, 3, 5, True), "root 5 is not one of 3 nodes"),
        ("components", (edges, 2), "edge names node 2, not one of 2"),
        ("path_lengths", (edges, 3, np.ones(3), [1], 0), "one weight per edge"),
        ("path_lengths", (edges, 3, [1.0, np.nan], [1], 0), "must not be NaN"),
        ("path_lengths", (edges, 3, [1.0, -1.0], [1], 0), "must not be negative"),
        ("path_lengths", (edges, 3, w, [3], 0), "source 3 is not one of 3 nodes"),
        ("path_lengths", (edges, 3, w, [1], 4), "target 4 is not one of 3 nodes"),
    ]
    for name, args, message in cases:
        args = tuple(np.asarray(a) if isinstance(a, list) else a for a in args)
        for path in (_rs.dx, python_dx):
            with pytest.raises(ValueError, match=message):
                getattr(path, name)(*args)


def test_bin_checks_on_both_paths():
    from skeliner import _rs
    from skeliner.python import dx as python_dx

    from ._parity import assert_parity

    rng = np.random.default_rng(61)
    seen = {"duplicated": 0, "mismatched": 0, "fragmented": 0, "owned": 0}
    for _ in range(200):
        m = int(rng.integers(1, 60))  # mesh vertices
        n_bins = int(rng.integers(1, 12))
        sizes = rng.integers(0, 8, size=n_bins)
        idx = rng.integers(0, m, size=int(sizes.sum())).astype(np.int64)
        ptr = np.zeros(n_bins + 1, dtype=np.int64)
        np.cumsum(sizes, out=ptr[1:])
        # vert2node: the inverse, with a few entries wrong or missing
        lut = np.full(m, -1, dtype=np.int64)
        for nid in range(n_bins):
            lut[idx[ptr[nid] : ptr[nid + 1]]] = nid
        keys = np.flatnonzero(lut >= 0)
        vals = lut[keys].copy()
        flip = rng.random(len(keys)) < 0.1
        vals[flip] = rng.integers(0, n_bins, size=int(flip.sum()))
        keep = rng.random(len(keys)) > 0.05
        keys, vals = keys[keep], vals[keep]
        rust = _rs.dx.check_bins(ptr, idx, keys, vals)
        assert_parity(rust, python_dx.check_bins(ptr, idx, keys, vals))
        seen["duplicated"] += rust[1].size > 0
        seen["mismatched"] += rust[2].size > 0

        mesh_edges = rng.integers(0, m, size=(int(rng.integers(0, 3 * m)), 2))
        rust = _rs.dx.fragmented_bins(ptr, idx, mesh_edges, m)
        assert_parity(rust, python_dx.fragmented_bins(ptr, idx, mesh_edges, m))
        seen["fragmented"] += rust[0].size > 0

        faces = rng.integers(0, m, size=(int(rng.integers(0, 30)), 3))
        owner = rng.integers(-1, 4, size=m)
        rust = _rs.dx.face_owner(faces, owner)
        assert_parity(rust, python_dx.face_owner(faces, owner))
        seen["owned"] += int((rust >= 0).sum())
    assert min(seen.values()) > 20, seen

    ptr, idx = np.array([0, 2, 3]), np.array([0, 4, 1])
    cases = [
        ("check_bins", (np.array([0, 2]), idx, idx, idx), "must be a CSR"),
        ("check_bins", (ptr, idx, np.array([-1]), np.array([0])), "not be negative"),
        ("fragmented_bins", (ptr, idx, np.zeros((0, 2), np.int64), 4), "vertex 4"),
        ("face_owner", (np.array([[0, 1, 5]]), np.zeros(3, np.int64)), "vertex 5"),
    ]
    for name, args, message in cases:
        for path in (_rs.dx, python_dx):
            with pytest.raises(ValueError, match=message):
                getattr(path, name)(*args)


def test_the_graph_diagnostics_end_alike_on_both_paths(skel, mesh):
    from ._parity import assert_parity, both

    looped = copy.deepcopy(skel)
    leaves = dx.nodes_of_degree(looped, 1)
    post.graft(looped, int(leaves[0]), int(leaves[-1]))
    forest = copy.deepcopy(skel)
    post.clip(forest, *map(int, forest.edges[len(forest.edges) // 2]))
    lo, hi = skel.nodes.min(0), skel.nodes.max(0)
    box = (lo + (hi - lo) * 0.2, hi - (hi - lo) * 0.2)
    for s in (skel, looped, forest):
        calls = [
            (dx.check_connectivity, (s,), {"return_isolated": True}),
            (dx.check_acyclicity, (s,), {"return_cycles": True}),
            (dx.cycles, (s, mesh), {}),
            (dx.check_bins, (s,), {"mesh": mesh, "return_report": True}),
            (dx.edge_support, (s, mesh), {}),
            (dx.face_owner, (s, mesh), {}),
            (dx.leaf_depths, (s,), {}),
            (dx.suspicious_tips, (s,), {"return_stats": True}),
            (dx.extract_neurites, (s, int(leaves[len(leaves) // 2]) // 2), {}),
            (dx.neurites_out_of_bounds, (s, box), {}),
            (dx.degree_distribution, (s,), {"detailed": True}),
        ]
        calls += [(dx.branches_of_length, (s, k), {}) for k in (2, 3, 5)]
        calls += [(dx.twigs_of_length, (s, k), {}) for k in (1, 2, 3)]
        for fn, args, kwargs in calls:
            rust, python = both(fn, *args, **kwargs)
            assert_parity(rust, python, fn.__name__)


def test_edge_support_is_the_two_set_differences(skel, mesh):
    from skeliner.skeletonize import _edges_from_mesh

    looped = copy.deepcopy(skel)
    leaves = dx.nodes_of_degree(looped, 1)
    post.graft(looped, int(leaves[0]), int(leaves[-1]))  # no surface behind it
    rep = dx.edge_support(looped, mesh)
    g = _edges_from_mesh(
        np.asarray(mesh.edges_unique), looped.vert2node, len(mesh.vertices)
    )
    gset = {(int(u), int(v)) for u, v in g}
    tset = {tuple(sorted(map(int, e))) for e in looped.edges}
    assert rep["dropped"] == sorted(gset - tset)
    assert rep["unsupported"] == sorted(tset - gset)
    assert rep["n_tree"] == len(tset)
    assert (min(leaves[0], leaves[-1]), max(leaves[0], leaves[-1])) in rep[
        "unsupported"
    ]


def _forest_skeleton():
    """0-1-2-3 with a twig 1-4, and apart from it 5-6-7: two trees."""
    nodes = np.arange(24, dtype=np.float64).reshape(8, 3)
    edges = np.array([[0, 1], [1, 2], [2, 3], [1, 4], [5, 6], [6, 7]])
    return Skeleton(
        soma=Soma.from_sphere(nodes[0], 1.0, verts=None),
        nodes=nodes,
        radii={"median": np.ones(8)},
        edges=edges,
        ntype=None,
    )


def test_leaves_of_another_tree_have_no_twig_and_no_depth():
    # Before: igraph's parent -2 for a node node 0 does not reach indexed the
    # second-last node, so leaves 5 and 7 grew twigs through unrelated
    # nodes, and their depth was an infinity cast to an integer.
    skel = _forest_skeleton()
    for path in ("rust", "python"):
        with sk.backend(path):
            assert dx.twigs_of_length(skel, 1) == [[4]]
            assert dx.twigs_of_length(skel, 2, include_branching_node=True) == [
                [1, 2, 3]
            ]
            assert dx.leaf_depths(skel).tolist() == [3, 2, -1, -1]
            assert dx.check_connectivity(skel, return_isolated=True) == [5, 6, 7]
            assert dx.branches_of_length(skel, 3) == [[1, 2, 3], [5, 6, 7]]
