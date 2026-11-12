"""sk.qc: a cell's report, many cells' table, and the triage."""

import csv
import json

import numpy as np
import pytest
import trimesh

from skeliner import pre, qc
from skeliner.dataclass import (
    Discarded,
    MeshComponents,
    Neurites,
    Organelles,
    Skeleton,
    Soma,
)

from .test_pre import (
    NO_CAP,
    _arbor_pieces,
    _ball_with_stub,
    _combine,
    _subdivided_tube,
    _two_cylinders,
)


def _ball():
    return trimesh.creation.icosphere(subdivisions=3, radius=3000.0)


def _sites(rep, kind):
    return [s for s in rep["sites"] if s["kind"] == kind]


# ── one cell ──────────────────────────────────────────────────────────


def test_a_report_says_what_the_run_did(monkeypatch):
    mesh = _two_cylinders(separation=1500.0)
    real = pre.find_gaps
    monkeypatch.setattr(pre, "find_gaps", lambda m, **kw: real(m, **kw, **NO_CAP))
    clean, comps = pre.preprocess(mesh, organelles=False, soma=False)
    rep = qc.report(clean, comps, meta={"cell": "c1", "type": "t"})

    s = rep["summary"]
    assert rep["qc"] == qc.SCHEMA
    assert rep["meta"] == {"cell": "c1", "type": "t"}
    assert s["record"] is True and s["soma_asked"] == "off"
    assert s["faces"] == len(clean.faces)
    assert s["input_faces"] == len(mesh.faces)
    (b,) = comps.record["bridges"]
    (site,) = _sites(rep, "bridge")
    assert s["bridges"] == 1
    assert s["bridge_longest_edges"] == pytest.approx(
        b["length"] / comps.record["median_edge"]
    )
    assert s["bridge_most_relative"] == pytest.approx(b["length"] / b["diag"])
    np.testing.assert_allclose(
        site["position"], (np.asarray(b["a"]) + np.asarray(b["b"])) / 2
    )
    # told it has none, the cell misses no soma
    assert rep["flags"] == []
    assert json.loads(json.dumps(rep)) == rep


def test_a_missing_soma_is_a_fact_unless_the_cell_has_none():
    ball = _ball()
    _, found = pre.preprocess(ball, organelles=False)
    assert qc.report(ball, found)["flags"] == []
    assert _sites(qc.report(ball, found), "soma")

    far = [30000.0, 0.0, 0.0]
    _, none = pre.preprocess(ball, organelles=False, soma=far)
    rep = qc.report(ball, none)
    assert rep["flags"] == ["no_soma_at_point"]
    assert [s["position"] for s in _sites(rep, "soma_point")] == [far]

    # components no run made: nothing says the cell has none
    bare = MeshComponents(
        soma=None,
        organelles=Organelles.empty(len(ball.faces)),
        neurites=Neurites([np.arange(len(ball.faces))]),
        discarded=Discarded([]),
    )
    rep = qc.report(ball, bare)
    assert rep["summary"]["record"] is False
    assert rep["flags"] == ["no_soma"]


def test_a_discarded_piece_is_placed_by_what_cut_it_off():
    """Three discarded pieces, one of each kind: one beside an organelle
    (cut off by it), a loose speck within the reach find_gaps gives a piece
    of its size, and a loose speck far off.  The first two are sites, the
    third is counted only."""
    tube = _subdivided_tube()
    me = pre._median_edge(tube)
    near_speck = trimesh.creation.box(extents=[8.0, 8.0, 8.0])
    # two median edges off the arbor's surface (the skeleton's radius, 50)
    near_speck.apply_translation([50.0 + 2 * me + 4.0, 0.0, -300.0])
    far_speck = trimesh.creation.box(extents=[8.0, 8.0, 8.0])
    far_speck.apply_translation([5000.0, 0.0, 0.0])
    mesh = _combine([tube, near_speck, far_speck])
    c = mesh.triangles_center
    n_t, n_s = len(tube.faces), len(near_speck.faces)
    on_tube = np.arange(len(mesh.faces)) < n_t
    # the side's row from z 0 to 250: its faces pointing down (centres a
    # third of the way up) the band, those pointing up — beside them — the piece
    row = on_tube & (c[:, 0] > 0)
    band = row & (c[:, 2] > 0) & (c[:, 2] < 125)
    piece = np.flatnonzero(row & (c[:, 2] > 125) & (c[:, 2] < 250))
    assert band.any() and len(piece), "fixture: a band and a piece beside it"
    rest = np.flatnonzero(on_tube & ~band & ~np.isin(np.arange(len(c)), piece))
    near = np.arange(n_t, n_t + n_s)
    far = np.arange(n_t + n_s, len(mesh.faces))
    z = np.zeros(len(mesh.faces), dtype=bool)
    comps = MeshComponents(
        soma=None,
        organelles=Organelles(pocket=band, isolated=z.copy(), expanded=z.copy()),
        neurites=Neurites([rest]),
        discarded=Discarded([piece, near, far]),
    )
    zs = np.arange(-500.0, 501.0, 100.0)
    nodes = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=1)
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 50.0, verts=None),
        nodes=nodes,
        radii={"median": np.full(len(zs), 50.0)},
        edges=np.array([[i, i + 1] for i in range(len(zs) - 1)]),
        ntype=None,
    )

    n_f = len(mesh.faces)
    s = qc.report(mesh, comps)["summary"]
    assert s["discarded"] == 3
    assert (s["discarded_at_soma"], s["discarded_at_organelle"]) == (0, 1)
    assert s["discarded_soma_largest"] is None
    assert s["discarded_organelle_largest"] == len(piece)
    assert s["discarded_organelle_share"] == pytest.approx(len(piece) / n_f)
    assert s["discarded_within_reach"] is None, "no skeleton, no arbor to be near"

    rep = qc.report(mesh, comps, skel)
    s = rep["summary"]
    assert s["discarded_within_reach"] == 1, "the near speck; the piece is not loose"
    assert s["discarded_reach_largest"] == n_s
    assert s["discarded_reach_share"] == pytest.approx(n_s / n_f)
    cut, loose = _sites(rep, "discarded")
    assert cut["index"] == 0 and cut["faces"] == len(piece)
    assert cut["at_organelle"] and not cut["at_soma"]
    np.testing.assert_allclose(
        cut["position"],
        np.asarray(mesh.vertices)[np.asarray(mesh.faces)[piece]].reshape(-1, 3).mean(0),
    )
    assert loose["index"] == 1 and not (loose["at_organelle"] or loose["at_soma"])
    # its nearest corners stand 4 off the x axis: to the skeleton's surface
    corner = np.hypot(50.0 + 2 * me, 4.0) - 50.0
    assert loose["arbor_distance"] == pytest.approx(corner, rel=1e-9)
    assert loose["arbor_distance"] <= loose["reach"]


def test_a_neurite_discarded_at_the_soma_is_a_fact():
    """A process standing out of the soma, discarded (the 95% rule drops a
    real stub where one neurite holds the rest): a fact, by break_up_mesh's
    own test for a neurite against a patch of the soma — and triage lists
    it with no top.  A stump that does not stand out is no such fact."""
    for height, want in ((160.0, True), (40.0, False)):
        mesh, soma, org = _ball_with_stub(height=height)
        (stub,) = _arbor_pieces(mesh, soma, org)
        comps = MeshComponents(
            soma=soma,
            organelles=org,
            neurites=Neurites([]),
            discarded=Discarded([stub]),
        )
        rep = qc.report(mesh, comps, meta={"cell": f"h{height:g}"})
        s = rep["summary"]
        (site,) = _sites(rep, "discarded")
        assert site["at_soma"] and site["neurite"] is want, height
        assert s["discarded_neurites"] == int(want)
        assert ("discarded_neurite" in rep["flags"]) is want
        if want:
            assert s["discarded_neurite_largest"] == len(stub)
            assert site["out_of_soma"] > 3 * site["junction"]
            (cell,) = qc.triage(qc.collect([rep]))
            (reason,) = cell["reasons"]
            assert reason["reason"] == (
                f"{qc.FLAGS['discarded_neurite']} (the largest {len(stub):,} faces)"
            )
            assert reason["value"] == len(stub) and reason["rank"] is None
        else:
            assert s["discarded_neurite_largest"] is None
            assert qc.triage(qc.collect([rep])) == []


def _stub_components(height):
    mesh, soma, org = _ball_with_stub(height=height)
    (stub,) = _arbor_pieces(mesh, soma, org)
    comps = MeshComponents(
        soma=soma, organelles=org, neurites=Neurites([]), discarded=Discarded([stub])
    )
    return mesh, comps


def test_a_plot_draws_the_cell_and_zooms_on_each_discarded_neurite(tmp_path):
    """The whole cell as an orthographic sheet (xz, xy, zy and the key), then
    a row of three views for each discarded neurite; saved where asked.  A
    stump no fact names gets no row of its own."""
    mesh, comps = _stub_components(160.0)
    rep = qc.report(mesh, comps, meta={"cell": "c9"})
    fig = qc.plot(mesh, comps, report=rep, path=tmp_path / "c9.png")
    assert len(fig.axes) == 4 + 3
    key = fig.axes[3].texts[0].get_text()
    assert key.startswith("c9\n" + qc.FLAGS["discarded_neurite"])
    assert f"piece 0: {len(comps.discarded[0]):,} faces" in key
    assert (tmp_path / "c9.png").stat().st_size > 0
    assert fig.subfigs[1]._suptitle.get_text().startswith("discarded piece 0:")

    mesh, comps = _stub_components(40.0)
    fig = qc.plot(mesh, comps)
    assert len(fig.axes) == 4
    assert "no fact" in fig.axes[3].texts[0].get_text()


def test_a_report_or_plot_refuses_components_of_another_mesh():
    """Components name faces and vertices of the mesh they were made on;
    read against another — the raw mesh beside the cleaned one — they say
    nothing true, so both refuse, saying which mesh to pass."""
    mesh, comps = _stub_components(160.0)
    other = trimesh.creation.icosphere(subdivisions=3, radius=100.0)
    for fn in (qc.report, qc.plot):
        with pytest.raises(ValueError, match="made on a mesh of"):
            fn(other, comps)
    # as many faces, but the soma's vertices are not where its fit put them
    # (another frame, another cell of the same size)
    moved = trimesh.Trimesh(
        np.asarray(mesh.vertices) + [10000.0, 0.0, 0.0], mesh.faces, process=False
    )
    with pytest.raises(ValueError, match="vertex ids"):
        qc.report(moved, comps)


def test_the_skeleton_says_its_trees_loops_and_stems():
    ball = _ball()
    _, comps = pre.preprocess(ball, organelles=False)
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],  # on the soma's surface
            [6000.0, 0.0, 0.0],  # 5,000 off it
            [3000.0, 1000.0, 0.0],
            [0.0, 9000.0, 0.0],  # a tree of its own
            [0.0, 9500.0, 0.0],
        ]
    )
    skel = Skeleton(
        soma=Soma.from_sphere(nodes[0], 1000.0, verts=None),
        nodes=nodes,
        radii={"median": np.full(len(nodes), 100.0)},
        edges=np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5]]),
        ntype=None,
    )
    rep = qc.report(ball, comps, skel)
    s = rep["summary"]
    assert (s["skel_nodes"], s["skel_trees"], s["skel_loops"]) == (6, 2, 1)
    assert rep["flags"] == ["skeleton_forest", "skeleton_loops"]
    assert [t["node"] for t in _sites(rep, "tree")] == [4]
    assert len(_sites(rep, "loop")) == 1
    stems = {t["node"]: t["gap"] for t in _sites(rep, "stem")}
    assert stems == pytest.approx({1: 0.0, 2: 5000.0})
    assert s["stem_longest"] == pytest.approx(5000.0)


def test_the_report_says_where_the_cell_meets_the_volume_s_edge():
    ball = _ball()
    _, comps = pre.preprocess(ball, organelles=False)
    rep = qc.report(ball, comps, bounds=((-2500.0, -5e3, -5e3), (5e3, 5e3, 5e3)))
    assert rep["summary"]["touches_bounds"] is True
    (side,) = _sites(rep, "bounds")
    assert (side["axis"], side["side"]) == (0, "lo")
    assert side["vertices"] == rep["summary"]["bounds_vertices"]
    me = rep["summary"]["median_edge"]
    assert side["vertices"] == int(
        (np.asarray(ball.vertices)[:, 0] <= -2500 + me).sum()
    )

    rep = qc.report(ball, comps, bounds=((-1e4,) * 3, (1e4,) * 3))
    assert rep["summary"]["touches_bounds"] is False
    assert not _sites(rep, "bounds")


def test_a_report_reads_back_as_written(tmp_path):
    ball = _ball()
    _, comps = pre.preprocess(ball, organelles=False)
    rep = qc.report(ball, comps, meta={"cell": "x"})
    qc.save_report(rep, tmp_path / "qc.json")
    assert qc.load_report(tmp_path / "qc.json") == rep

    (tmp_path / "old.json").write_text(json.dumps({**rep, "qc": 0}))
    with pytest.raises(ValueError, match="layout 0"):
        qc.load_report(tmp_path / "old.json")


# ── many cells ────────────────────────────────────────────────────────


def _rep(cell, flags=(), meta=None, **summary):
    return {
        "qc": qc.SCHEMA,
        "skeliner": "test",
        "meta": {"cell": cell, **(meta or {})},
        "summary": summary,
        "flags": list(flags),
        "sites": [],
    }


def test_collect_makes_a_row_a_cell_named_by_its_meta_or_its_folder(tmp_path):
    (tmp_path / "c7").mkdir()
    rep = _rep("ignored", bridges=2)
    del rep["meta"]["cell"]
    qc.save_report(rep, tmp_path / "c7" / "qc.json")
    rows = qc.collect(
        [_rep("a", meta={"type": "t1"}, bridges=1), tmp_path / "c7" / "qc.json"]
    )
    assert rows == [
        {"cell": "a", "type": "t1", "bridges": 1, "flags": []},
        {"cell": "c7", "bridges": 2, "flags": []},
    ]


def test_collect_takes_a_folder_of_reports(tmp_path):
    """A run's folder, as handed to it: every report under it in path order
    (a cell's folder, or a file of its own name), what is no report passed
    over; one file or a string path as well — never a path read as the
    characters of its name."""
    run = tmp_path / "run"
    for cell, n in (("c2", 2), ("c1", 1)):
        (run / cell).mkdir(parents=True)
        rep = _rep(cell, bridges=n)
        del rep["meta"]["cell"]
        qc.save_report(rep, run / cell / "qc.json")
    flat = _rep("x", bridges=3)
    del flat["meta"]["cell"]
    qc.save_report(flat, run / "c3.json")
    (run / "meta.json").write_text(json.dumps({"started": "now"}))
    (run / "qc.csv").write_text("not, a, report\n")

    want = [
        {"cell": "c1", "bridges": 1, "flags": []},
        {"cell": "c2", "bridges": 2, "flags": []},
        {"cell": "c3", "bridges": 3, "flags": []},
    ]
    assert qc.collect(run) == want
    assert qc.collect(str(run)) == want
    assert qc.collect(str(run / "c1" / "qc.json")) == want[:1]
    assert qc.collect([run / "c3.json", str(run / "c2")]) == [want[2], want[1]]

    old = _rep("o")
    old["qc"] = 0
    (run / "old.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="layout 0"):
        qc.collect(run)


def test_triage_orders_the_facts_by_kind_then_by_size():
    """No soma inside the volume, then at its edge; then the discarded
    neurites, the largest first — whatever order the rows came in."""
    rows = qc.collect(
        [
            _rep("n1", ["discarded_neurite"], discarded_neurite_largest=500),
            _rep("edge", ["no_soma"], touches_bounds=True),
            _rep("n2", ["discarded_neurite"], discarded_neurite_largest=7000),
            _rep("inside", ["no_soma"], touches_bounds=False),
            _rep("ok", bridge_longest_edges=3.0),
        ]
    )
    got = qc.triage(rows)
    assert [t["cell"] for t in got] == ["inside", "edge", "n2", "n1"]
    assert [r["value"] for t in got[2:] for r in t["reasons"]] == [7000, 500]


def test_triage_puts_facts_first_then_the_most_extreme(tmp_path):
    rows = qc.collect(
        [
            _rep(
                "a", meta={"type": "x"}, bridge_longest_edges=10.0, touches_bounds=False
            ),
            _rep(
                "b", meta={"type": "x"}, bridge_longest_edges=30.0, touches_bounds=False
            ),
            _rep(
                "c",
                ["no_soma"],
                {"type": "y"},
                bridge_longest_edges=None,
                touches_bounds=True,
            ),
            _rep(
                "d", meta={"type": "y"}, bridge_longest_edges=5.0, touches_bounds=False
            ),
        ]
    )
    facts = qc.triage(rows)
    assert [(t["cell"], [r["reason"] for r in t["reasons"]]) for t in facts] == [
        ("c", ["no soma found (the cell meets the volume's edge)"])
    ]

    got = qc.triage(rows, top=1, measures=["bridge_longest_edges"])
    assert [t["cell"] for t in got] == ["c", "b"]
    assert got[1]["reasons"] == [
        {
            "reason": qc.MEASURES["bridge_longest_edges"],
            "value": 30.0,
            "rank": 1,
            "of": 3,
        }
    ]
    # within its kind: d tops the y cells
    got = qc.triage(rows, top=1, by="type", measures=["bridge_longest_edges"])
    assert [(t["cell"], t["type"]) for t in got] == [("c", "y"), ("b", "x"), ("d", "y")]

    qc.save_table(got, tmp_path / "triage.csv")
    with open(tmp_path / "triage.csv", newline="") as fh:
        read = list(csv.DictReader(fh))
    assert read[1] == {
        "cell": "b",
        "type": "x",
        "reasons": f"{qc.MEASURES['bridge_longest_edges']} (30, 1/2)",
    }


def test_a_table_round_trips_through_csv_and_triages_the_same(tmp_path):
    rows = qc.collect(
        [
            _rep("a", bridge_longest_edges=10.0, record=True),
            _rep("b", ["skeleton_forest"], bridge_longest_edges=None, record=True),
        ]
    )
    qc.save_table(rows, tmp_path / "qc.csv")
    with open(tmp_path / "qc.csv", newline="") as fh:
        read = list(csv.DictReader(fh))
    assert read == [
        {"cell": "a", "bridge_longest_edges": "10.0", "record": "True", "flags": ""},
        {
            "cell": "b",
            "bridge_longest_edges": "",
            "record": "True",
            "flags": "skeleton_forest",
        },
    ]
    read[1]["flags"] = read[1]["flags"].split("; ")
    read[0]["flags"] = []
    assert [t["cell"] for t in qc.triage(read, top=1)] == [
        t["cell"] for t in qc.triage(rows, top=1)
    ]
