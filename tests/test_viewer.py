"""Tests for skeliner.plot.viewer — the interactive server.

The viewer mirrors an operation's terminal output into the browser over
the WebSocket, which is the only sign a long-running step is moving.
"""

import asyncio
import contextlib
import copy
import io
import threading

import numpy as np
import pytest
import trimesh

from skeliner.dataclass import Discarded, Neurites, Organelles, Soma
from skeliner.plot.viewer import _LogTee, _create_app
from skeliner.skeletonize import _timed


def _tee_output(emit):
    """Run *emit* through a ``_LogTee``.

    Returns what it broadcast and what reached the wrapped stream.
    """
    sent = []
    original = io.StringIO()

    async def broadcast(msg):
        sent.append(msg["text"])

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        tee = _LogTee(original, loop, broadcast)
        with contextlib.redirect_stdout(tee):
            emit()
        tee.finish()
        # queued in order, so this one completing means all of them did
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
    return sent, original.getvalue()


# ── _LogTee ───────────────────────────────────────────────────────────
#
# Driven through the real ``_timed`` helper rather than hand-written
# strings, so these cannot drift from what the pipeline actually prints.


def test_timed_stage_reaches_the_browser_as_one_line():
    """A stage is a label printed before the work and an elapsed time
    printed after, on the same line.  The browser must get the label
    immediately — it is the only sign a long stage is running — and then
    the finished line, never a bare "1.99 s" on its own."""

    def emit():
        with _timed("↳  build surface graph", verbose=True):
            pass

    sent, _ = _tee_output(emit)
    assert len(sent) == 2, sent
    assert sent[0].strip().startswith("↳  build surface graph")
    assert sent[0].endswith("…"), "the label must arrive before the timing"
    assert sent[1].startswith(sent[0]), "the finished line completes the label"
    assert sent[1].endswith(" s")


def test_timed_sub_messages_are_their_own_lines():
    def emit():
        with _timed("↳  skeletonize neurites", verbose=True) as log:
            log("neurite 0: 12 verts → 3 nodes")

    sent, _ = _tee_output(emit)
    assert len(sent) == 3, sent
    assert sent[2].strip() == "└─ neurite 0: 12 verts → 3 nodes"


def test_whole_lines_are_sent_once_each_and_reach_the_terminal():
    def emit():
        print("alpha")
        print("beta")

    sent, raw = _tee_output(emit)
    assert sent == ["alpha", "beta"]
    assert raw == "alpha\nbeta\n", "the terminal must still get everything"


# ── Reassign routes ───────────────────────────────────────────────────
#
# Driven through the real ASGI app, so the guards and the preview/apply
# handshake are exercised the way the page uses them.


def _tube_with_components():
    """A tube split by a soma band, with components already derived."""
    from skeliner import pre

    mesh = trimesh.creation.cylinder(radius=50.0, height=1000.0, sections=16)
    mesh = mesh.subdivide().subdivide()
    z = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    band = np.flatnonzero(np.abs(z) <= 100.0)
    soma_verts = np.unique(np.asarray(mesh.faces)[band])
    soma = Soma.fit(mesh.vertices[soma_verts], verts=soma_verts)
    empty = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=empty.copy(), isolated=empty.copy(), expanded=empty.copy())
    arbor = np.flatnonzero(~pre.soma_face_mask(mesh.faces, soma_verts))
    neurites = Neurites(pre._face_components_fast(mesh.faces, arbor))
    return mesh, soma, org, neurites


@pytest.fixture
def reassign_client(tmp_path, monkeypatch):
    """A viewer serving a tube whose components are already derived."""
    from starlette.testclient import TestClient

    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    mesh, soma, org, neurites = _tube_with_components()
    app = _create_app(preload_mesh=mesh, port=8912)

    # Reach the app's private state the way no caller normally would; the
    # alternative is re-running detection, which these tests are not about.
    state = None
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if isinstance(val, dict) and "pending_reassignment" in val:
                state = val
        if state is not None:
            break
    assert state is not None, "could not reach mesh_state"

    with TestClient(app) as client:
        state["mesh"] = mesh
        state["soma"] = soma
        state["organelles"] = org
        state["neurites"] = neurites
        state["discarded"] = Discarded([])
        yield client, state, neurites


def test_reassign_refuses_before_components_exist(reassign_client):
    client, state, _ = reassign_client
    state["neurites"] = Neurites([])
    r = client.post("/reassign_preview", json={"faces": [0], "to": "soma"})
    assert r.status_code == 400
    assert "break_up_mesh" in r.json()["error"]


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"faces": [], "to": "soma"}, "No faces selected"),
        ({"faces": [0], "to": "bogus"}, "to must be"),
        ({"faces": [10**9], "to": "soma"}, "face ids must lie"),
    ],
)
def test_reassign_preview_rejects_bad_input(reassign_client, body, expected):
    client, _, _ = reassign_client
    r = client.post("/reassign_preview", json=body)
    assert r.status_code == 400
    assert expected in r.json()["error"]


def test_apply_without_a_preview_is_refused(reassign_client):
    client, _, _ = reassign_client
    r = client.post("/reassign_apply")
    assert r.status_code == 400
    assert "preview first" in r.json()["error"]


def test_preview_reports_the_fringe_and_leaves_state_alone(reassign_client):
    client, state, neurites = reassign_client
    sel = neurites[0][:64].tolist()
    before = len(state["soma"].verts)

    r = client.post("/reassign_preview", json={"faces": sel, "to": "soma"})
    assert r.status_code == 200
    body = r.json()
    # the ≥2-of-3 rule drags neighbours along, so more leaves than was picked
    assert set(sel) <= set(body["leaving"])
    assert len(body["leaving"]) > len(sel)
    assert body["entering"] == []
    assert len(state["soma"].verts) == before, "preview must not commit"


def test_apply_commits_exactly_what_the_preview_promised(reassign_client):
    client, state, neurites = reassign_client
    sel = neurites[0][:64].tolist()

    preview = client.post("/reassign_preview", json={"faces": sel, "to": "soma"}).json()
    applied = client.post("/reassign_apply").json()

    assert applied["ok"]
    assert applied["nNeurites"] == preview["nNeurites"]
    assert applied["nDiscarded"] == preview["nDiscarded"]
    assert set(np.unique(np.asarray(state["mesh"].faces)[sel]).tolist()) <= set(
        state["soma"].verts.tolist()
    )


# ── releasing soma surface back to the arbor ──────────────────────────
#
# Releasing is only half the operation.  break_up_mesh absorbs any
# component whose boundary is mostly soma, so a stub released from the
# soma is re-absorbed by the very re-derive the reassignment runs — the
# release happens and is undone in the same call.  Recording the claim is
# what makes it stick.


def _soma_faces_of(state):
    from skeliner import pre

    faces = np.asarray(state["mesh"].faces)
    return np.flatnonzero(pre.soma_face_mask(faces, state["soma"].verts))


def test_releasing_to_the_arbor_records_the_claim(reassign_client):
    client, state, _ = reassign_client
    sel = [int(f) for f in _soma_faces_of(state)[:64]]
    assert len(state["released"]) == 0

    client.post("/reassign_preview", json={"faces": sel, "to": "remainder"})
    client.post("/reassign_apply")

    assert set(sel) <= set(state["released"].tolist())
    assert len(state["rescued"]) == 0, (
        "a lasso claim is not a component-level rescue — it is floored"
    )


def test_assigning_to_the_soma_withdraws_the_claim(reassign_client):
    """The opposite statement, so leaving the claim standing would have
    the override fight the assignment just made."""
    client, state, _ = reassign_client
    sel = [int(f) for f in _soma_faces_of(state)[:64]]

    client.post("/reassign_preview", json={"faces": sel, "to": "remainder"})
    client.post("/reassign_apply")
    assert set(sel) <= set(state["released"].tolist())

    client.post("/reassign_preview", json={"faces": sel, "to": "soma"})
    client.post("/reassign_apply")
    assert not (set(sel) & set(state["released"].tolist()))


def test_assigning_to_an_organelle_withdraws_the_claim(reassign_client):
    client, state, _ = reassign_client
    sel = [int(f) for f in _soma_faces_of(state)[:64]]
    client.post("/reassign_preview", json={"faces": sel, "to": "remainder"})
    client.post("/reassign_apply")

    client.post("/reassign_preview", json={"faces": sel, "to": "organelle"})
    client.post("/reassign_apply")
    assert not (set(sel) & set(state["released"].tolist()))


def test_the_preview_forecasts_the_release_with_the_claim_in_force(reassign_client):
    """A preview run without the claim forecasts a re-absorption the
    commit will not perform — the two must use the same set."""
    client, state, _ = reassign_client
    sel = [int(f) for f in _soma_faces_of(state)[:64]]

    preview = client.post(
        "/reassign_preview", json={"faces": sel, "to": "remainder"}
    ).json()
    applied = client.post("/reassign_apply").json()

    assert applied["nNeurites"] == preview["nNeurites"]
    assert applied["nDiscarded"] == preview["nDiscarded"]
    assert applied["somaVerts"] == len(state["soma"].verts)


def test_a_claim_survives_the_next_re_derive(reassign_client):
    """The point of recording it: a later break must not take it back."""
    client, state, _ = reassign_client
    sel = [int(f) for f in _soma_faces_of(state)[:64]]
    client.post("/reassign_preview", json={"faces": sel, "to": "remainder"})
    n_neurites = client.post("/reassign_apply").json()["nNeurites"]

    again = client.post("/break_up_mesh")
    assert again.status_code == 200
    assert again.json()["nNeurites"] == n_neurites


def test_cancel_retires_the_preview(reassign_client):
    client, _, neurites = reassign_client
    client.post(
        "/reassign_preview", json={"faces": neurites[0][:64].tolist(), "to": "soma"}
    )
    assert client.post("/reassign_cancel").json()["ok"]
    assert client.post("/reassign_apply").status_code == 400


def test_compacting_the_mesh_retires_the_preview(reassign_client):
    """Face ids do not survive compaction, so applying afterwards would
    reassign whatever now sits at those indices."""
    client, _, neurites = reassign_client
    client.post(
        "/reassign_preview", json={"faces": neurites[0][:64].tolist(), "to": "soma"}
    )
    assert client.post("/compact_mesh").status_code == 200
    assert client.post("/reassign_apply").status_code == 400


def test_organelle_target_writes_the_manual_mask(reassign_client):
    client, state, neurites = reassign_client
    sel = neurites[0][:32].tolist()
    client.post("/reassign_preview", json={"faces": sel, "to": "organelle"})
    client.post("/reassign_apply")
    assert state["organelles"].manual[sel].all()
    assert not state["organelles"].pocket.any()


# ── /bin ──────────────────────────────────────────────────────────────
#
# A node *is* the set of mesh vertices it owns, so Edit ▸ Skeleton renders a
# selected node as that surface.  A bin is small enough to fetch per click,
# which is why the partition is never shipped in bulk.


@pytest.fixture
def bin_client(tmp_path, monkeypatch):
    """A viewer with a mesh and a skeleton layer built from it.

    Subdivided three times so the tube is long enough in mesh edges to bin
    into several nodes; two subdivisions give four, which is too few for the
    neighbour and non-overlap checks to say anything.
    """
    from starlette.testclient import TestClient

    from skeliner import skeletonize
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    mesh = trimesh.creation.cylinder(radius=40.0, height=1200.0, sections=24)
    mesh = mesh.subdivide().subdivide().subdivide()
    skel = skeletonize(mesh, verbose=False)
    assert len(skel.nodes) >= 6, "fixture must produce enough bins to test with"

    app = _create_app(preload_mesh=mesh, port=8913)
    with TestClient(app) as client:
        path = tmp_path / "skeleton.npz"
        skel.to_npz(path)
        client.post("/upload", files={"file": ("skeleton.npz", path.read_bytes())})
        name = next(iter(client.get("/skeletons").json()))
        yield client, name, mesh, skel


def test_a_bin_is_the_surface_its_node_owns(bin_client):
    client, name, mesh, skel = bin_client
    r = client.post("/bin", json={"name": name, "node": 3})
    assert r.status_code == 200
    body = r.json()

    assert body["node"] == 3
    assert body["nVerts"] == len(skel.node2verts[3])
    assert body["radius"] == pytest.approx(float(skel.r[3]))
    assert len(body["faces"]) > 0

    # every returned face really is a >=2-of-3 majority of that bin
    owned = set(np.asarray(skel.node2verts[3]).tolist())
    for f in body["faces"]:
        assert sum(v in owned for v in mesh.faces[f]) >= 2


def test_clicking_a_face_finds_the_bin_that_owns_it(bin_client):
    """The inverse lookup: what is in front of you is a strip of surface,
    not a node id."""
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 4}).json()["faces"]
    back = client.post("/bin", json={"name": name, "face": faces[0]}).json()
    assert back["node"] == 4


def test_bins_do_not_overlap(bin_client):
    client, name, _, skel = bin_client
    seen = set()
    for node in range(1, min(8, len(skel.nodes))):
        faces = client.post("/bin", json={"name": name, "node": node}).json()["faces"]
        assert not (seen & set(faces)), f"node {node} shares faces with an earlier bin"
        seen |= set(faces)


def test_node_0_is_reported_as_not_editable(bin_client):
    """Node 0's "bin" is ``soma.verts``, assigned wholesale by the soma
    stitch.  It belongs to Edit ▸ Mesh Comp., not Edit ▸ Skeleton."""
    client, name, _, _ = bin_client
    assert (
        client.post("/bin", json={"name": name, "node": 0}).json()["editable"] is False
    )
    assert (
        client.post("/bin", json={"name": name, "node": 1}).json()["editable"] is True
    )


def test_a_face_owned_by_no_bin_is_refused(bin_client, tmp_path):
    """Soma, organelle and discarded surface is display-only here — saying
    so beats silently returning an empty selection.

    A bare tube has no unowned surface, so one is made: emptying a bin
    leaves its faces owned by nobody, which is exactly the shape of the
    real case.
    """
    from skeliner import dx

    client, _, mesh, skel = bin_client
    holed = copy.deepcopy(skel)
    orphaned = np.asarray(holed.node2verts[2], dtype=np.int64)
    holed.node2verts[2] = np.empty(0, dtype=np.int64)
    for v in orphaned:
        holed.vert2node.pop(int(v), None)

    path = tmp_path / "holed.npz"
    holed.to_npz(path)
    client.post("/upload", files={"file": ("holed.npz", path.read_bytes())})

    owner = dx.face_owner(holed, mesh)
    orphans = np.flatnonzero(owner < 0)
    assert orphans.size > 0, "emptying a bin must leave surface unowned"

    r = client.post("/bin", json={"name": "holed.npz", "face": int(orphans[0])})
    assert r.status_code == 400
    assert "no bin" in r.json()["error"]


@pytest.mark.parametrize(
    "body, expected",
    [
        ({}, "No such skeleton"),
        ({"name": "nope", "node": 0}, "No such skeleton"),
        ({"name": "skeleton.npz"}, "Pass either node or face"),
        ({"name": "skeleton.npz", "node": 10**9}, "node id out of range"),
        ({"name": "skeleton.npz", "face": 10**9}, "face id out of range"),
    ],
)
def test_bin_rejects_bad_input(bin_client, body, expected):
    client, name, _, _ = bin_client
    if body.get("name") == "skeleton.npz":
        body = {**body, "name": name}
    r = client.post("/bin", json=body)
    assert r.status_code == 400
    assert expected in r.json()["error"]


# ── skeleton invalidation ─────────────────────────────────────────────
#
# `node2verts` / `vert2node` index *the mesh the skeleton was built from*.
# After a face removal or a compaction they name different vertices, with
# no shape mismatch to catch it — so a skeleton kept across a mesh edit is
# not stale, it is wrong.  Staleness is derived by comparing the bound mesh
# by identity rather than set by a flag, which is what these pin down.


def _mesh_state_of(app):
    """Reach the app's private mesh_state, as the reassign fixture does."""
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if isinstance(val, dict) and "pending_reassignment" in val:
                return val
    raise AssertionError("could not reach mesh_state")


def test_a_skeleton_built_from_another_mesh_is_refused(bin_client):
    client, name, mesh, _ = bin_client
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 200

    _mesh_state_of(client.app)["mesh"] = mesh.copy()  # same geometry, new object

    r = client.post("/bin", json={"name": name, "node": 3})
    assert r.status_code == 409
    assert "different mesh" in r.json()["error"]


def test_a_stale_skeleton_cannot_be_exported(bin_client):
    """The one irreversible step: exported radii would belong to surface
    that is no longer there, and nothing downstream could tell."""
    client, name, mesh, _ = bin_client
    assert client.get(f"/export_skeleton?name={name}").status_code == 200

    _mesh_state_of(client.app)["mesh"] = mesh.copy()

    r = client.get(f"/export_skeleton?name={name}")
    assert r.status_code == 409
    assert "Re-skeletonize" in r.json()["error"]


@pytest.mark.parametrize("route, payload", [("/remove_selected", "faces")])
def test_a_real_preprocessing_action_invalidates_the_skeleton(
    bin_client, route, payload
):
    """The synthetic swaps above prove the rule; this proves it is wired to
    the routes a user actually clicks, which is where a missed site shows."""
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]

    r = client.post(route, json={payload: faces[:5]})
    assert r.status_code == 200, r.text

    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 409
    assert client.get(f"/export_skeleton?name={name}").status_code == 409


def test_undo_after_a_real_edit_restores_currency(bin_client):
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]
    client.post("/remove_selected", json={"faces": faces[:5]})
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 409

    assert client.post("/undo").status_code == 200
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 200


def test_returning_to_the_original_mesh_makes_it_current_again(bin_client):
    """Comparing by identity gets undo right for free: restoring the very
    mesh a skeleton was built from restores the same object."""
    client, name, mesh, _ = bin_client
    state = _mesh_state_of(client.app)
    original = state["mesh"]

    state["mesh"] = mesh.copy()
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 409

    state["mesh"] = original
    assert client.post("/bin", json={"name": name, "node": 3}).status_code == 200


def test_the_owner_cache_follows_the_skeleton(bin_client):
    """The cache is keyed on the mesh *and* the skeleton it was built from,
    so swapping the skeleton under an unchanged mesh recomputes.

    Driven by an actual swap, because a cache that is only ever read proves
    nothing.
    """
    client, name, mesh, skel = bin_client
    before = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]
    assert before, "nothing to invalidate"

    # a different skeleton object: node 3 now owns what node 4 owned
    swapped = copy.deepcopy(skel)
    swapped.node2verts[3], swapped.node2verts[4] = (
        swapped.node2verts[4],
        swapped.node2verts[3],
    )
    swapped.vert2node = {}
    for nid, vs in enumerate(swapped.node2verts):
        for v in np.asarray(vs):
            swapped.vert2node[int(v)] = nid

    for route in client.app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if isinstance(val, dict) and name in val:
                val[name]["skeleton"] = swapped

    after = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]
    assert after != before, "the cache served an answer for the old skeleton"
    assert after == sorted(
        int(f) for f in np.flatnonzero(_dx().face_owner(swapped, mesh) == 3)
    )


def _dx():
    from skeliner import dx

    return dx


# ── /rescue_as_neurite ────────────────────────────────────────────────
#
# The neurite/discarded split is an automatic size threshold.  Rescuing
# overrides it, and because the split is re-derived on every break and
# every reassignment the override has to be replayed or it vanishes.


@pytest.fixture
def rescue_client(tmp_path, monkeypatch):
    """A viewer serving a tube plus a speck the threshold discards."""
    from starlette.testclient import TestClient

    from skeliner import pre
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)

    tube = trimesh.creation.cylinder(radius=50.0, height=1000.0, sections=64)
    tube = tube.subdivide()
    speck = trimesh.creation.box(extents=[8.0, 8.0, 8.0])
    speck.apply_translation([500.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([tube, speck])

    empty = np.zeros(len(mesh.faces), dtype=bool)
    org = Organelles(pocket=empty.copy(), isolated=empty.copy(), expanded=empty.copy())
    comp = pre.break_up_mesh(mesh, None, org)
    assert len(comp.discarded) == 1, "fixture must produce something to rescue"

    app = _create_app(preload_mesh=mesh, port=8914)
    with TestClient(app) as client:
        state = _mesh_state_of(app)
        state["mesh"] = mesh
        state["soma"] = None
        state["organelles"] = org
        state["neurites"] = comp.neurites
        state["discarded"] = comp.discarded
        yield client, state, comp.discarded[0]


def test_rescuing_promotes_the_whole_fragment(rescue_client):
    client, state, speck = rescue_client
    n_before = len(state["neurites"])

    r = client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})
    assert r.status_code == 200
    body = r.json()
    assert body["nRescued"] == 1
    assert body["facesRescued"] == len(speck)
    assert body["nDiscarded"] == 0
    assert len(state["neurites"]) == n_before + 1
    assert len(state["discarded"]) == 0


def test_rescuing_relabels_and_does_not_recompute(rescue_client):
    """A rescue changes which list a component sits in, nothing else.

    Driven by a partition ``break_up_mesh`` would *not* reproduce: if the
    route re-derived, these hand-split neurites would be replaced by
    freshly computed ones and the split would vanish.
    """
    client, state, speck = rescue_client
    whole = state["neurites"][0]
    split = [whole[: len(whole) // 2], whole[len(whole) // 2 :]]
    state["neurites"] = Neurites([a.copy() for a in split])

    r = client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})
    assert r.status_code == 200

    survived = [n for n in state["neurites"] if set(n.tolist()) != set(speck.tolist())]
    assert len(survived) == 2, "the hand-made split was recomputed away"
    assert all(
        np.array_equal(np.sort(a), np.sort(b))
        for a, b in zip(sorted(survived, key=len), sorted(split, key=len))
    )


def test_rescuing_leaves_the_stored_inputs_alone(rescue_client):
    client, state, speck = rescue_client
    mesh, soma, org = state["mesh"], state["soma"], state["organelles"]

    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})

    assert state["mesh"] is mesh
    assert state["soma"] is soma
    assert state["organelles"] is org


def test_the_rescued_fragment_is_an_ordinary_neurite(rescue_client):
    client, state, speck = rescue_client
    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})

    matches = [n for n in state["neurites"] if set(n.tolist()) == set(speck.tolist())]
    assert len(matches) == 1, "the fragment should be in the neurites, once"
    assert not hasattr(state["neurites"], "rescued"), (
        "nothing on the result may record that it was ever discarded"
    )


def test_a_later_reassignment_does_not_undo_the_rescue(rescue_client):
    """The bug this exists to prevent.

    A reassignment re-derives the components, and a rescue that lived
    only in the neurites list was silently dropped by that re-derive.
    """
    client, state, speck = rescue_client
    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})

    sel = [int(f) for f in state["neurites"][0][:32]]
    assert (
        client.post("/reassign_preview", json={"faces": sel, "to": "organelle"}).json()[
            "nDiscarded"
        ]
        == 0
    ), "the preview already re-derives"

    applied = client.post("/reassign_apply").json()
    assert applied["nDiscarded"] == 0
    assert any(set(n.tolist()) == set(speck.tolist()) for n in state["neurites"])


def test_re_breaking_does_not_undo_the_rescue(rescue_client):
    client, state, speck = rescue_client
    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})

    r = client.post("/break_up_mesh")
    assert r.status_code == 200
    assert r.json()["nDiscarded"] == 0


# ── /preprocess ───────────────────────────────────────────────────────
#
# The whole pipeline in one call.  It ends in break_up_mesh like the
# routes above, so it inherits their hazard: a re-derive that does not
# replay the override undoes it.


def test_preprocess_runs_the_whole_pipeline(rescue_client):
    client, state, speck = rescue_client
    state["neurites"] = None
    state["discarded"] = None

    r = client.post("/preprocess")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["nNeurites"] >= 1
    assert state["neurites"] is not None
    assert len(state["neurites"]) == body["nNeurites"]


def test_preprocess_does_not_undo_the_rescue(rescue_client):
    """The one-click run is a from-scratch re-derive, which is exactly
    when a rescue that is not fed back in disappears."""
    client, state, speck = rescue_client
    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})
    assert len(state["rescued"]) > 0

    r = client.post("/preprocess")
    assert r.status_code == 200
    assert r.json()["nDiscarded"] == 0
    assert any(set(n.tolist()) == set(speck.tolist()) for n in state["neurites"])


def test_preprocess_drops_the_caches_keyed_to_the_replaced_mesh(rescue_client):
    """Every cache names faces of the mesh the run replaced, and the run
    has already consumed and removed what they point at."""
    client, state, speck = rescue_client
    state["mesh_stats"] = object()
    state["gap_clusters"] = [object()]
    state["fusion_clusters"] = [object()]

    assert client.post("/preprocess").status_code == 200

    assert state["mesh_stats"] is None
    assert state["gap_clusters"] is None
    assert state["fusion_clusters"] is None


def test_preprocess_leaves_the_mesh_uncompacted(rescue_client):
    """Compaction reindexes faces, and the rescue list, the annotations
    and any loaded skeleton are all stated in face ids.  /compact_mesh
    remaps them; this route does not compact so it need not."""
    client, state, speck = rescue_client
    n_before = len(state["mesh"].faces)

    assert client.post("/preprocess").status_code == 200

    assert len(state["mesh"].faces) == n_before


def test_preprocess_refuses_without_a_mesh(rescue_client):
    client, state, _ = rescue_client
    state["mesh"] = None
    r = client.post("/preprocess")
    assert r.status_code == 400
    assert "No mesh loaded" in r.json()["error"]


def test_the_override_is_recorded_as_face_ids(rescue_client):
    client, state, speck = rescue_client
    assert len(state["rescued"]) == 0
    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})
    assert set(state["rescued"].tolist()) == set(speck.tolist())


def test_the_labels_left_behind_name_the_right_components(rescue_client):
    """A rescue republishes every component, so no label can go stale."""
    client, state, speck = rescue_client
    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})

    ann = client.get("/annotations").json()
    labels = [h["label"] for h in ann["highlights"]]
    assert not any(lbl.startswith("discarded ") for lbl in labels)
    n_neurites = sum(lbl.startswith("neurite ") for lbl in labels)
    assert n_neurites == len(state["neurites"])


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"faces": []}, "No faces selected"),
        ({"faces": [0]}, "No discarded component matches"),
    ],
)
def test_rescue_rejects_bad_input(rescue_client, body, expected):
    client, state, speck = rescue_client
    # face 0 belongs to the tube, not to the speck
    r = client.post("/rescue_as_neurite", json=body)
    assert r.status_code == 400
    assert expected in r.json()["error"]


def test_rescue_is_refused_when_nothing_was_discarded(rescue_client):
    client, state, speck = rescue_client
    state["discarded"] = Discarded([])
    r = client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})
    assert r.status_code == 400
    assert "No discarded components" in r.json()["error"]


def test_compacting_carries_the_override_through_the_reindex(rescue_client):
    """Compaction renumbers every face, so the stored ids must move too.

    Driven through a real removal first: on an already-clean mesh
    compaction drops nothing, the face map is the identity, and a test
    that only checks the ids survived would pass without them ever being
    remapped.
    """
    client, state, speck = rescue_client
    client.post("/rescue_as_neurite", json={"faces": [int(speck[0])]})

    mesh_before = state["mesh"]
    want = mesh_before.vertices[mesh_before.faces[speck]].mean(axis=1)
    ids_before = set(state["rescued"].tolist())

    # Degenerate some tube faces, which sit *before* the speck, so
    # compacting them away shifts every id the override holds.
    doomed = [int(f) for f in state["neurites"][0][:40]]
    assert client.post("/remove_selected", json={"faces": doomed}).status_code == 200
    assert client.post("/compact_mesh").status_code == 200

    assert len(state["mesh"].faces) < len(mesh_before.faces), "nothing was compacted"
    ids_after = set(state["rescued"].tolist())
    assert ids_after and ids_after != ids_before, "the ids were never remapped"

    # Same triangles, whatever they are now called.
    mesh_after = state["mesh"]
    got = mesh_after.vertices[mesh_after.faces[sorted(ids_after)]].mean(axis=1)
    assert np.allclose(np.sort(got, axis=0), np.sort(want, axis=0))

    # And the override still does its job on the renumbered mesh.
    assert client.post("/break_up_mesh").json()["nDiscarded"] == 0


# ── /bin_reassign_* ───────────────────────────────────────────────────
#
# Moving surface from one bin into another.  The preview runs the real
# primitive on a copy and keeps it, so these check both that the summary
# is right and that applying installs the very thing it described.


def _skel_state_of(app, name):
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if isinstance(val, dict) and name in val and "skeleton" in val[name]:
                return val[name]
    raise AssertionError("could not reach the skeleton state")


def _a_donor_face(client, name, node):
    """A face owned by one of *node*'s graph neighbours."""
    bin_ = client.post("/bin", json={"name": name, "node": node}).json()
    assert bin_["neighbors"], "fixture node has no neighbour to take from"
    nbr = bin_["neighbors"][0]
    nbr_bin = client.post("/bin", json={"name": name, "node": nbr}).json()
    return nbr, nbr_bin["faces"]


def test_the_scope_is_the_bin_and_its_neighbours(bin_client):
    client, name, mesh, skel = bin_client
    body = client.post("/bin", json={"name": name, "node": 3}).json()

    allowed = set(body["neighbors"]) | {3}
    owner = _dx().face_owner(skel, mesh)
    for f in body["scopeFaces"]:
        assert int(owner[f]) in allowed
    # and it is not merely the bin itself
    assert set(body["scopeFaces"]) > set(body["faces"])


def test_a_preview_moves_nothing_until_applied(bin_client):
    client, name, mesh, skel = bin_client
    nbr, faces = _a_donor_face(client, name, 3)
    before = len(skel.node2verts[3])

    r = client.post(
        "/bin_reassign_preview",
        json={"name": name, "faces": faces[:2], "to": 3},
    )
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["moved"] > 0
    assert body["donors"] == [nbr]
    assert len(skel.node2verts[3]) == before, "preview must not commit"


def test_applying_installs_exactly_what_was_previewed(bin_client):
    client, name, mesh, _ = bin_client
    nbr, faces = _a_donor_face(client, name, 3)
    n_before = client.post("/bin", json={"name": name, "node": 3}).json()["nVerts"]

    preview = client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces[:2], "to": 3}
    ).json()
    applied = client.post("/bin_reassign_apply", json={"name": name}).json()
    assert applied["ok"]
    assert applied["moved"] == preview["moved"]
    assert applied["radiusAfter"] == preview["radiusAfter"]

    live = _skel_state_of(client.app, name)["skeleton"]
    assert float(live.r[3]) == pytest.approx(preview["radiusAfter"])
    after = client.post("/bin", json={"name": name, "node": 3}).json()
    assert after["nVerts"] == n_before + preview["moved"]


def test_a_bin_may_not_take_from_a_distant_bin(bin_client):
    """The scope is a correctness constraint, so the server enforces it too."""
    client, name, mesh, skel = bin_client
    body = client.post("/bin", json={"name": name, "node": 1}).json()
    far = [
        n for n in range(2, len(skel.nodes)) if n not in body["neighbors"] and n != 1
    ]
    assert far, "fixture has no non-neighbour to test with"
    far_faces = client.post("/bin", json={"name": name, "node": far[-1]}).json()[
        "faces"
    ]

    r = client.post(
        "/bin_reassign_preview", json={"name": name, "faces": far_faces, "to": 1}
    )
    assert r.status_code == 400
    assert "do not touch" in r.json()["error"]


def test_node_0_is_refused_as_a_destination(bin_client):
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 2}).json()["faces"]
    r = client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces, "to": 0}
    )
    assert r.status_code == 400
    assert "soma" in r.json()["error"]


@pytest.mark.parametrize(
    "body, code, expected",
    [
        ({"name": "nope", "faces": [0], "to": 1}, 400, "No such skeleton"),
        ({"faces": [], "to": 1}, 400, "No faces selected"),
        ({"faces": [0]}, 400, "No destination bin"),
        ({"faces": [10**9], "to": 1}, 400, "face id out of range"),
        ({"faces": [0], "to": 10**6}, 400, "node id out of range"),
    ],
)
def test_bin_reassign_rejects_bad_input(bin_client, body, code, expected):
    client, name, _, _ = bin_client
    body = {"name": name, **body} if "name" not in body else body
    r = client.post("/bin_reassign_preview", json=body)
    assert r.status_code == code
    assert expected in r.json()["error"]


def test_applying_without_a_preview_is_refused(bin_client):
    client, name, _, _ = bin_client
    r = client.post("/bin_reassign_apply", json={"name": name})
    assert r.status_code == 400
    assert "preview first" in r.json()["error"]


def test_cancel_retires_the_bin_preview(bin_client):
    client, name, _, _ = bin_client
    _, faces = _a_donor_face(client, name, 3)
    client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces[:2], "to": 3}
    )
    assert client.post("/bin_reassign_cancel", json={"name": name}).json()["ok"]
    r = client.post("/bin_reassign_apply", json={"name": name})
    assert r.status_code == 400


def test_a_preview_does_not_survive_the_mesh_changing(bin_client):
    """It names vertices of one mesh; against another it would move others."""
    client, name, mesh, _ = bin_client
    _, faces = _a_donor_face(client, name, 3)
    client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces[:2], "to": 3}
    )

    _mesh_state_of(client.app)["mesh"] = mesh.copy()

    r = client.post("/bin_reassign_apply", json={"name": name})
    assert r.status_code == 409
    assert "changed since the preview" in r.json()["error"]


def test_the_owner_cache_follows_an_applied_edit(bin_client):
    """The cache compares by identity, so the edit must install a new object."""
    client, name, _, _ = bin_client
    nbr, faces = _a_donor_face(client, name, 3)
    before = client.post("/bin", json={"name": name, "node": nbr}).json()["faces"]

    client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces[:2], "to": 3}
    )
    client.post("/bin_reassign_apply", json={"name": name})

    after = client.post("/bin", json={"name": name, "node": nbr}).json()["faces"]
    assert after != before, "the cache served the pre-edit partition"


def test_moving_a_whole_bin_drops_it_and_renumbers(bin_client):
    """The merge verb: a bin that gives up everything has no position or
    radius left, so it is removed and the nodes after it shift down."""
    client, name, _, _ = bin_client
    nbr, faces = _a_donor_face(client, name, 3)
    n_nodes = len(_skel_state_of(client.app, name)["skeleton"].nodes)

    preview = client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces, "to": 3}
    ).json()
    assert preview["dropped"] == [nbr], preview

    client.post("/bin_reassign_apply", json={"name": name})
    live = _skel_state_of(client.app, name)["skeleton"]
    assert len(live.nodes) == n_nodes - 1


def test_the_preview_reports_what_it_could_not_recompute(bin_client):
    client, name, _, _ = bin_client
    _, faces = _a_donor_face(client, name, 3)
    body = client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces[:2], "to": 3}
    ).json()
    # Shapes the panel renders unconditionally; all four must always be there.
    assert isinstance(body["staleRadii"], list)
    assert isinstance(body["fragmented"], list)
    assert isinstance(body["movedFaces"], list) and body["movedFaces"]
    assert body["ignoredUnowned"] >= 0 and body["ignoredFar"] >= 0


def test_merging_bins_needs_no_lasso(bin_client):
    """Clicking the bins *is* the selection; the merge takes them whole."""
    client, name, _, _ = bin_client
    nbr, _ = _a_donor_face(client, name, 3)
    donor_verts = client.post("/bin", json={"name": name, "node": nbr}).json()["nVerts"]

    r = client.post(
        "/bin_reassign_preview", json={"name": name, "fromNodes": [nbr], "to": 3}
    )
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["moved"] == donor_verts, "a merge takes the whole bin"
    assert body["dropped"] == [nbr]


def test_a_merge_keeps_the_skeleton_in_one_piece(bin_client):
    """The absorbed node's edges are contracted onto the survivor, so no
    neighbour it was holding gets stranded."""
    client, name, _, _ = bin_client
    live = _skel_state_of(client.app, name)["skeleton"]
    e = np.asarray(live.edges)

    interior = None
    for n in range(1, len(live.nodes)):
        nbrs = np.unique(e[(e[:, 0] == n) | (e[:, 1] == n)])
        nbrs = nbrs[nbrs != n]
        if nbrs.size >= 2 and 0 not in nbrs.tolist():
            interior, nbrs = int(n), [int(x) for x in nbrs]
            break
    assert interior is not None, "fixture has no interior node to absorb"

    client.post(
        "/bin_reassign_preview",
        json={"name": name, "fromNodes": [interior], "to": nbrs[0]},
    )
    assert client.post("/bin_reassign_apply", json={"name": name}).json()["ok"]

    after = _skel_state_of(client.app, name)["skeleton"]
    assert _n_components_of(after) == 1, "the merge broke the skeleton apart"


def _n_components_of(skel):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(skel.nodes)
    e = np.asarray(skel.edges, dtype=np.int64)
    adj = sp.coo_matrix(
        (np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n)
    )
    return connected_components(adj, directed=False)[0]


def test_bins_that_do_not_touch_cannot_be_merged(bin_client):
    client, name, _, skel = bin_client
    body = client.post("/bin", json={"name": name, "node": 1}).json()
    far = [n for n in range(2, len(skel.nodes)) if n not in body["neighbors"]]
    assert far, "fixture has no non-neighbour to test with"

    r = client.post(
        "/bin_reassign_preview", json={"name": name, "fromNodes": [far[-1]], "to": 1}
    )
    assert r.status_code == 400
    assert "connected piece" in r.json()["error"]


def test_a_chain_of_bins_merges_even_though_the_ends_do_not_touch(bin_client):
    """Connectivity is checked *through the selection*, so a-b-c works."""
    client, name, _, skel = bin_client
    e = np.asarray(skel.edges)

    chain = None
    for b in range(1, len(skel.nodes)):
        nbrs = np.unique(e[(e[:, 0] == b) | (e[:, 1] == b)])
        nbrs = [int(x) for x in nbrs if x != b and x != 0]
        if len(nbrs) >= 2:
            a, c = nbrs[0], nbrs[1]
            if (
                c
                not in client.post("/bin", json={"name": name, "node": a}).json()[
                    "neighbors"
                ]
            ):
                chain = (a, int(b), c)
                break
    if chain is None:
        pytest.skip("fixture has no a-b-c chain with a and c apart")

    a, b, c = chain
    r = client.post(
        "/bin_reassign_preview", json={"name": name, "fromNodes": [b, c], "to": a}
    )
    assert r.status_code == 200, r.json()
    assert sorted(r.json()["dropped"]) == sorted([b, c])


def test_merging_with_the_soma_is_refused(bin_client):
    client, name, _, _ = bin_client
    r = client.post(
        "/bin_reassign_preview", json={"name": name, "fromNodes": [0], "to": 1}
    )
    assert r.status_code == 400
    assert "soma" in r.json()["error"]


# ── /edge_support and /edge_edit_* ────────────────────────────────────
#
# `edges` is the one thing on a Skeleton nothing else derives from, so it
# is the only part that can honestly be edited directly.  Which of the
# three verbs a node pair names is the server's to decide, from the tree
# and from the surface — never from the gesture.


@pytest.fixture
def loop_client(tmp_path, monkeypatch):
    """A viewer whose surface graph drops an edge the tree could restore.

    A tube gives a path, so ``G`` and ``T`` are the same and there is no
    restore to test.  A ring is the smallest thing that drops one.
    """
    from starlette.testclient import TestClient

    from skeliner import skeletonize
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    mesh = trimesh.creation.torus(
        major_radius=300.0, minor_radius=60.0, major_sections=64, minor_sections=16
    )
    skel = skeletonize(mesh, verbose=False)

    app = _create_app(preload_mesh=mesh, port=8915)
    with TestClient(app) as client:
        path = tmp_path / "skeleton.npz"
        skel.to_npz(path)
        client.post("/upload", files={"file": ("skeleton.npz", path.read_bytes())})
        name = next(iter(client.get("/skeletons").json()))
        yield client, name, mesh, skel


def _a_dropped_pair(client, name):
    body = client.post("/edge_support", json={"name": name}).json()
    assert body["ok"], body
    assert body["dropped"], "fixture was supposed to drop an adjacency"
    return body["dropped"][0]


def test_edge_support_reports_what_the_tree_does_not_carry(loop_client):
    client, name, _, skel = loop_client
    body = client.post("/edge_support", json={"name": name}).json()

    assert body["nTree"] == len(skel.edges)
    assert len(body["dropped"]) == 1
    u, v = body["dropped"][0]
    assert (u, v) not in {tuple(sorted(map(int, e))) for e in skel.edges}


def test_a_stale_skeleton_is_neither_checked_nor_edited(bin_client):
    """vert2node indexes the mesh it was built from; against another one
    the answer would be plausible and wrong."""
    client, name, _, _ = bin_client
    faces = client.post("/bin", json={"name": name, "node": 3}).json()["faces"]
    assert client.post("/edge_support", json={"name": name}).status_code == 200

    assert client.post("/remove_selected", json={"faces": faces[:5]}).status_code == 200

    assert client.post("/edge_support", json={"name": name}).status_code == 409
    r = client.post("/edge_edit_preview", json={"name": name, "u": 1, "v": 2})
    assert r.status_code == 409
    assert "Re-skeletonize" in r.json()["error"]


def test_restoring_an_edge_the_surface_supports(loop_client):
    client, name, _, skel = loop_client
    u, v = _a_dropped_pair(client, name)

    r = client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["verb"] == "restore"
    assert body["supported"] is True
    assert body["hops"] >= len(skel.nodes) // 2
    assert body["cyclesAfter"] == body["cyclesBefore"] + 1
    assert body["componentsAfter"] == body["componentsBefore"]
    assert body["orphans"] == []


def test_grafting_is_not_dressed_up_as_a_restore(bin_client):
    """A pair the surface does not join is a leap across a gap — a
    different claim, and it has to be labelled as one."""
    client, name, _, skel = bin_client
    far = len(skel.nodes) - 1
    assert far >= 3
    r = client.post("/edge_edit_preview", json={"name": name, "u": 1, "v": far})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["verb"] == "graft"
    assert body["supported"] is False


def test_clipping_says_what_it_orphans(bin_client):
    """Cutting is asymmetric: on a tree edge it strands a whole subtree, and
    the size of that is the thing to know before committing."""
    client, name, _, skel = bin_client
    u, v = (int(x) for x in skel.edges[len(skel.edges) // 2])

    body = client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v}).json()
    assert body["verb"] == "clip"
    assert body["orphans"], "cutting a tree edge must strand something"
    assert body["componentsAfter"] == body["componentsBefore"] + 1
    assert 0 not in body["orphans"], "the soma side is never the orphan"


def test_clipping_a_cycle_edge_orphans_nothing(loop_client):
    """The other half of the asymmetry — and the reason the preview says
    which case this is rather than warning either way."""
    client, name, _, _ = loop_client
    u, v = _a_dropped_pair(client, name)
    client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})
    assert client.post("/edge_edit_apply", json={"name": name}).json()["ok"]

    live = _skel_state_of(client.app, name)["skeleton"]
    a, b = (int(x) for x in live.edges[len(live.edges) // 2])
    body = client.post("/edge_edit_preview", json={"name": name, "u": a, "v": b}).json()
    assert body["verb"] == "clip"
    assert body["orphans"] == []
    assert body["componentsAfter"] == body["componentsBefore"]
    assert body["cyclesAfter"] == body["cyclesBefore"] - 1


def test_a_preview_changes_nothing_until_applied(loop_client):
    client, name, _, _ = loop_client
    u, v = _a_dropped_pair(client, name)
    before = len(_skel_state_of(client.app, name)["skeleton"].edges)

    client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})
    assert len(_skel_state_of(client.app, name)["skeleton"].edges) == before

    assert client.post("/edge_edit_apply", json={"name": name}).json()["ok"]
    after = _skel_state_of(client.app, name)["skeleton"]
    assert len(after.edges) == before + 1
    assert (min(u, v), max(u, v)) in {tuple(sorted(map(int, e))) for e in after.edges}


def test_applying_installs_a_new_object_so_caches_invalidate(loop_client):
    """`clip` and `graft` mutate in place, which the identity-compared
    face-owner cache cannot see — so the edit is made on a copy."""
    client, name, _, _ = loop_client
    was = _skel_state_of(client.app, name)["skeleton"]
    u, v = _a_dropped_pair(client, name)
    client.post("/bin", json={"name": name, "node": 2})  # warm the cache
    client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})
    client.post("/edge_edit_apply", json={"name": name})

    sstate = _skel_state_of(client.app, name)
    assert sstate["skeleton"] is not was
    assert len(was.edges) + 1 == len(sstate["skeleton"].edges), "the base was mutated"


def test_a_restored_edge_is_thereafter_a_tree_edge(loop_client):
    client, name, _, _ = loop_client
    u, v = _a_dropped_pair(client, name)
    client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})
    client.post("/edge_edit_apply", json={"name": name})

    again = client.post("/edge_support", json={"name": name}).json()
    assert again["dropped"] == [], "the tree now carries it, so nothing is dropped"


def test_a_grafted_edge_is_thereafter_called_out_as_unsupported(bin_client):
    """Soma stems and gap bridges have no surface behind them, which is
    exactly why re-spanning ``G`` would delete them.  A graft makes one, so
    the round trip both produces the case and checks it is reported."""
    client, name, _, skel = bin_client
    far = len(skel.nodes) - 1
    assert (
        client.post("/edge_edit_preview", json={"name": name, "u": 1, "v": far}).json()[
            "verb"
        ]
        == "graft"
    )
    assert client.post("/edge_edit_apply", json={"name": name}).json()["ok"]

    scan = client.post("/edge_support", json={"name": name}).json()
    assert [1, far] in scan["unsupported"]

    body = client.post(
        "/edge_edit_preview", json={"name": name, "u": 1, "v": far}
    ).json()
    assert body["verb"] == "clip", "it is a tree edge now"
    assert body["unsupportedTree"] is True
    assert body["orphans"] == [], "and it is on a cycle, so nothing is stranded"


@pytest.mark.parametrize(
    "body, code, expected",
    [
        ({"name": "nope", "u": 1, "v": 2}, 400, "No such skeleton"),
        ({"u": 1}, 400, "two nodes"),
        ({"u": 2, "v": 2}, 400, "two different nodes"),
        ({"u": 1, "v": 10**6}, 400, "node id out of range"),
        ({"u": -1, "v": 2}, 400, "node id out of range"),
    ],
)
def test_edge_edit_rejects_bad_input(bin_client, body, code, expected):
    client, name, _, _ = bin_client
    body = {"name": name, **body} if "name" not in body else body
    r = client.post("/edge_edit_preview", json=body)
    assert r.status_code == code
    assert expected in r.json()["error"]


def test_applying_an_edge_edit_without_a_preview_is_refused(loop_client):
    client, name, _, _ = loop_client
    r = client.post("/edge_edit_apply", json={"name": name})
    assert r.status_code == 400
    assert "preview first" in r.json()["error"]


def test_cancel_retires_the_edge_preview(loop_client):
    client, name, _, _ = loop_client
    u, v = _a_dropped_pair(client, name)
    client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})
    assert client.post("/edge_edit_cancel", json={"name": name}).json()["ok"]
    assert client.post("/edge_edit_apply", json={"name": name}).status_code == 400


def test_a_bin_edit_landing_first_retires_the_edge_preview(bin_client):
    """Both replace the skeleton, and node ids are positions in it — so the
    second one to land is naming a different graph than it previewed."""
    client, name, _, skel = bin_client
    u, v = (int(x) for x in skel.edges[len(skel.edges) // 2])
    client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})

    nbr, faces = _a_donor_face(client, name, 3)
    client.post(
        "/bin_reassign_preview", json={"name": name, "faces": faces[:2], "to": 3}
    )
    assert client.post("/bin_reassign_apply", json={"name": name}).json()["ok"]

    r = client.post("/edge_edit_apply", json={"name": name})
    assert r.status_code == 409
    assert "skeleton changed" in r.json()["error"]


# ── /bin_split_preview ────────────────────────────────────────────────
#
# The third bin verb.  It shares the pending slot and the apply route with
# the other two, and it deliberately makes only one edge.


def _own_faces(client, name, node):
    return client.post("/bin", json={"name": name, "node": node}).json()["faces"]


def test_a_split_promotes_part_of_a_bin_to_its_own_node(bin_client):
    client, name, _, skel = bin_client
    n_before = len(skel.nodes)
    faces = _own_faces(client, name, 3)
    assert len(faces) >= 4, "fixture bin is too small to split"

    r = client.post(
        "/bin_split_preview",
        json={"name": name, "splitFrom": 3, "faces": faces[: len(faces) // 2]},
    )
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["split"] is True
    assert body["parent"] == 3
    assert body["to"] == n_before, "the new node is appended"
    assert body["moved"] > 0
    assert body["dropped"] == [], "a split empties nothing"
    assert len(_skel_state_of(client.app, name)["skeleton"].nodes) == n_before


def test_applying_a_split_installs_it_and_joins_only_the_parent(bin_client):
    """The whole design rests on this: the other edges are the user's, made
    with Edge, where the surface says which joins it backs."""
    client, name, _, skel = bin_client
    faces = _own_faces(client, name, 3)
    preview = client.post(
        "/bin_split_preview",
        json={"name": name, "splitFrom": 3, "faces": faces[: len(faces) // 2]},
    ).json()
    assert client.post("/bin_reassign_apply", json={"name": name}).json()["ok"]

    live = _skel_state_of(client.app, name)["skeleton"]
    new = preview["to"]
    assert len(live.nodes) == len(skel.nodes) + 1
    e = np.asarray(live.edges)
    touching = set(np.unique(e[(e[:, 0] == new) | (e[:, 1] == new)]).tolist()) - {new}
    assert touching == {3}
    assert _n_components_of(live) == _n_components_of(skel), "a piece went adrift"


def test_after_a_split_the_surface_offers_the_re_route(bin_client):
    """A split leaves the new node a leaf; putting it in the chain is a
    restore, and `edge_support` is what tells the user so."""
    client, name, mesh, skel = bin_client
    node, nbrs = None, None
    e = np.asarray(skel.edges)
    for n in range(1, len(skel.nodes)):
        nb = np.unique(e[(e[:, 0] == n) | (e[:, 1] == n)])
        nb = [int(x) for x in nb if x != n and x != 0]
        if len(nb) >= 2 and len(skel.node2verts[n]) >= 4:
            node, nbrs = n, nb
            break
    if node is None:
        pytest.skip("fixture has no splittable interior bin")

    # the half of the bin nearest one neighbour, so the new node lands there
    owned = np.asarray(skel.node2verts[node])
    d = np.linalg.norm(np.asarray(mesh.vertices)[owned] - skel.nodes[nbrs[0]], axis=1)
    near = set(owned[np.argsort(d)[: len(owned) // 2]].tolist())
    faces = [
        f
        for f in _own_faces(client, name, node)
        if sum(v in near for v in mesh.faces[f]) >= 2
    ]
    if not faces or len(faces) >= len(_own_faces(client, name, node)):
        pytest.skip("cannot isolate one side of the fixture bin")

    new = client.post(
        "/bin_split_preview",
        json={"name": name, "splitFrom": node, "faces": faces},
    ).json()["to"]
    client.post("/bin_reassign_apply", json={"name": name})

    dropped = {
        tuple(p)
        for p in client.post("/edge_support", json={"name": name}).json()["dropped"]
    }
    assert any(new in p for p in dropped), (
        "the split left no surface-backed join to re-route with"
    )


def test_a_split_and_an_edge_edit_share_one_pending_slot(bin_client):
    """One skeleton, one pending edit — whichever landed second would be
    refused by the identity check anyway."""
    client, name, _, skel = bin_client
    u, v = (int(x) for x in skel.edges[len(skel.edges) // 2])
    client.post("/edge_edit_preview", json={"name": name, "u": u, "v": v})

    faces = _own_faces(client, name, 3)
    client.post(
        "/bin_split_preview",
        json={"name": name, "splitFrom": 3, "faces": faces[:2]},
    )
    assert client.post("/bin_reassign_apply", json={"name": name}).json()["ok"]
    assert client.post("/edge_edit_apply", json={"name": name}).status_code == 409


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"faces": [0]}, "No bin to split"),
        ({"splitFrom": 3, "faces": []}, "No faces selected"),
        ({"splitFrom": 3, "faces": [10**9]}, "face id out of range"),
        ({"splitFrom": 10**6, "faces": [0]}, "node id out of range"),
        ({"splitFrom": 0, "faces": [0]}, "node 0 is the soma"),
    ],
)
def test_bin_split_rejects_bad_input(bin_client, body, expected):
    client, name, _, _ = bin_client
    r = client.post("/bin_split_preview", json={"name": name, **body})
    assert r.status_code == 400
    assert expected in r.json()["error"]


def test_splitting_off_a_whole_bin_is_refused_by_the_route(bin_client):
    client, name, _, _ = bin_client
    faces = _own_faces(client, name, 3)
    r = client.post(
        "/bin_split_preview", json={"name": name, "splitFrom": 3, "faces": faces}
    )
    assert r.status_code == 400
    assert "leave something behind" in r.json()["error"]


# ── naming neurites ───────────────────────────────────────────────────
#
# The last step of the workflow: a name sits at a position in `neurites`,
# and every re-derive rebuilds that list, so break_up_mesh returns
# neurites unnamed and a re-break drops the names on purpose.


def test_naming_a_neurite_reaches_the_component(reassign_client):
    client, state, neurites = reassign_client
    assert state["neurites"].named is False

    r = client.post("/name_neurite", json={"index": 0, "label": "axon"})
    assert r.status_code == 200
    body = r.json()
    assert body["labels"][0] == "axon"
    assert body["swcTypes"][0] == 2, "the code is read from the leading word"
    assert state["neurites"].labels[0] == "axon"


def test_an_explicit_code_beats_the_name(reassign_client):
    client, state, _ = reassign_client
    r = client.post(
        "/name_neurite", json={"index": 0, "label": "the odd one", "swcType": 2}
    )
    assert r.json()["swcTypes"][0] == 2


def test_the_name_reaches_the_annotation_label(reassign_client):
    client, state, _ = reassign_client
    client.post("/name_neurite", json={"index": 0, "label": "dendrite 1"})

    ann = client.get("/annotations").json()
    named = [h for h in ann["highlights"] if h.get("neurite") == 0]
    assert len(named) == 1
    assert named[0]["label"].startswith("dendrite 1 (")


def test_every_neurite_highlight_carries_its_index(reassign_client):
    """The page names a neurite back by index; once renamed the label no
    longer carries one, and position in `highlights` depends on the soma."""
    client, state, _ = reassign_client
    client.post("/name_neurite", json={"index": 0, "label": "axon"})

    ann = client.get("/annotations").json()
    idx = [h["neurite"] for h in ann["highlights"] if "neurite" in h]
    assert idx == list(range(len(state["neurites"])))


def test_re_breaking_drops_the_names(reassign_client):
    """Not a limitation to route around: a re-derive re-sorts, splits and
    merges, so a surviving name would sit on different surface."""
    client, state, _ = reassign_client
    client.post("/name_neurite", json={"index": 0, "label": "axon"})
    assert state["neurites"].named is True

    assert client.post("/break_up_mesh").status_code == 200
    assert state["neurites"].named is False


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"index": 0, "label": "   "}, "cannot be empty"),
        ({"index": 10**6, "label": "axon"}, "out of range"),
        ({"label": "axon"}, "index must be an integer"),
    ],
)
def test_name_neurite_rejects_bad_input(reassign_client, body, expected):
    client, _, _ = reassign_client
    r = client.post("/name_neurite", json=body)
    assert r.status_code == 400
    assert expected in r.json()["error"]


def test_naming_before_a_break_is_refused(reassign_client):
    client, state, _ = reassign_client
    state["neurites"] = Neurites([])
    r = client.post("/name_neurite", json={"index": 0, "label": "axon"})
    assert r.status_code == 400
    assert "break_up_mesh" in r.json()["error"]


def _skeleton_of(client):
    """The Skeleton the app is holding, reached the way the fixture does."""
    for route in client.app.routes:
        fn = getattr(route, "endpoint", None)
        for cell in getattr(fn, "__closure__", None) or ():
            val = cell.cell_contents
            if (
                isinstance(val, dict)
                and val
                and all(isinstance(v, dict) and "skeleton" in v for v in val.values())
            ):
                return next(iter(val.values()))["skeleton"]
    raise AssertionError("could not reach skeleton_states")


@pytest.fixture
def named_client(tmp_path, monkeypatch):
    """A viewer serving a soma with one process each way.

    The reassign fixture's tube is too small to bin into any node at all —
    its skeleton is the soma alone — so a test about node types needs its
    own mesh.  Asserted below, since a skeleton with no arbor nodes would
    pass an ntype test vacuously.
    """
    from starlette.testclient import TestClient

    from skeliner import pre
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)

    parts = [trimesh.creation.icosphere(subdivisions=4, radius=100.0)]
    for sign in (1, -1):
        tube = trimesh.creation.cylinder(radius=15.0, height=900.0, sections=24)
        tube = tube.subdivide().subdivide()
        tube.apply_translation([0.0, 0.0, sign * 520.0])
        parts.append(tube)
    verts, faces, n = [], [], 0
    for part in parts:
        verts.append(part.vertices)
        faces.append(part.faces + n)
        n += len(part.vertices)
    mesh = trimesh.Trimesh(
        vertices=np.vstack(verts), faces=np.vstack(faces), process=False
    )

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
    comp = pre.break_up_mesh(
        mesh, soma, org, rescued=np.concatenate([p[:1] for p in pieces])
    )
    assert len(comp.neurites) == 2

    app = _create_app(preload_mesh=mesh, port=8916)
    state = _mesh_state_of(app)
    with TestClient(app) as client:
        state["mesh"] = mesh
        state["soma"] = comp.soma
        state["organelles"] = comp.organelles
        state["neurites"] = comp.neurites
        state["discarded"] = comp.discarded
        yield client, state


def test_the_fixture_bins_into_arbor_nodes(named_client):
    """Guards every ntype assertion below from passing vacuously."""
    client, _ = named_client
    assert client.post("/skeletonize", json={}).status_code == 200
    assert len(_skeleton_of(client).nodes) > 1


def test_the_name_reaches_the_skeleton_through_the_route(named_client):
    """The whole point of naming, driven the way the page drives it.

    /skeletonize rebuilds MeshComponents from mesh_state, and a version
    that rebuilt a bare Neurites dropped the names on the way — every
    neurite exported as SWC type 0 however it had been named, with
    nothing anywhere saying so.
    """
    client, state = named_client
    assert (
        client.post("/name_neurite", json={"index": 0, "label": "axon"}).status_code
        == 200
    )
    assert (
        client.post(
            "/name_neurite", json={"index": 1, "label": "dendrite 1"}
        ).status_code
        == 200
    )

    assert client.post("/skeletonize", json={}).status_code == 200

    ntype = _skeleton_of(client).ntype
    assert ntype[0] == 1, "the soma must not be retyped by the junction fringe"
    assert set(ntype[1:].tolist()) == {2, 3}


def test_an_unnamed_neurite_skeletonizes_to_undefined(named_client):
    client, state = named_client
    assert client.post("/skeletonize", json={}).status_code == 200
    ntype = _skeleton_of(client).ntype
    assert ntype[0] == 1
    assert set(ntype[1:].tolist()) == {0}


def test_compacting_keeps_the_components_and_their_names(named_client):
    """Compaction reindexes faces; it does not re-derive anything.

    The route used to hand `compact_mesh` empty Neurites/Discarded and
    write the result straight back, so a compaction erased every neurite
    from mesh_state — and any name on them — without saying a word.
    """
    client, state = named_client
    client.post("/name_neurite", json={"index": 0, "label": "dendrite 0"})
    n_before = len(state["neurites"])
    assert n_before == 2

    assert client.post("/compact_mesh").status_code == 200

    assert len(state["neurites"]) == n_before, "compaction must not drop components"
    assert state["neurites"].labels[0] == "dendrite 0"


def test_the_exported_components_carry_the_names(named_client, tmp_path):
    from skeliner import io as skio

    client, state = named_client
    client.post("/name_neurite", json={"index": 0, "label": "dendrite 0"})

    resp = client.get("/export_components?prefix=")
    assert resp.status_code == 200
    out = tmp_path / "components.npz"
    out.write_bytes(resp.content)

    back = skio.load_components_npz(out)
    assert back.neurites.labels == ["dendrite 0", "neurite 1"]
    assert back.neurites.swc_types == [3, 0]


@pytest.mark.parametrize("fmt", ["npz", "swc"])
def test_the_exported_skeleton_carries_the_names_and_types(named_client, tmp_path, fmt):
    from skeliner import io as skio

    client, _ = named_client
    client.post("/name_neurite", json={"index": 0, "label": "axon"})
    client.post("/name_neurite", json={"index": 1, "label": "dendrite 1"})
    assert client.post("/skeletonize", json={}).status_code == 200

    resp = client.get(f"/export_skeleton?name=skeleton&format={fmt}")
    assert resp.status_code == 200, resp.text
    out = tmp_path / f"s.{fmt}"
    out.write_bytes(resp.content)

    loader = skio.load_skeleton_npz if fmt == "npz" else skio.load_skeleton_swc
    back = loader(out)
    assert set(back.ntype[1:].tolist()) == {2, 3}
    assert back.meta["neurite_labels"] == ["axon", "dendrite 1"]


def test_a_dropped_components_file_shows_the_names_it_carries(named_client, tmp_path):
    """Round trip through the file, the way the user does it.

    The upload path built its own highlight list instead of going through
    _publish_components, so a file whose neurites had been named came back
    labelled "neurite 0" — the names were in the archive and on the
    components, and only the drawing had forgotten them.
    """
    from skeliner import io as skio

    client, state = named_client
    client.post("/name_neurite", json={"index": 0, "label": "axon"})
    client.post("/name_neurite", json={"index": 1, "label": "dendrite 1"})

    out = tmp_path / "components.npz"
    out.write_bytes(client.get("/export_components?prefix=").content)
    assert skio.load_components_npz(out).neurites.labels == ["axon", "dendrite 1"]

    # Forget them, then drop the file back in.
    state["neurites"].clear_names()
    r = client.post(
        "/upload",
        files={
            "file": ("components.npz", out.read_bytes(), "application/octet-stream")
        },
    )
    assert r.status_code == 200, r.text

    assert state["neurites"].labels == ["axon", "dendrite 1"]
    ann = client.get("/annotations").json()
    named = {h["neurite"]: h["label"] for h in ann["highlights"] if "neurite" in h}
    assert named[0].startswith("axon (")
    assert named[1].startswith("dendrite 1 (")


def test_a_dropped_neurites_file_shows_the_names_it_carries(named_client, tmp_path):
    from skeliner import io as skio

    client, state = named_client
    client.post("/name_neurite", json={"index": 0, "label": "axon"})

    out = tmp_path / "neurites.npz"
    skio.save_neurites_npz(state["neurites"], out)

    r = client.post(
        "/upload",
        files={"file": ("neurites.npz", out.read_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 200, r.text

    ann = client.get("/annotations").json()
    named = {h["neurite"]: h["label"] for h in ann["highlights"] if "neurite" in h}
    assert named[0].startswith("axon (")


# ── uploading a drop as one gesture ───────────────────────────────────
#
# A drag-drop is one gesture, but the browser hands the files over in the
# OS's order.  Applied independently, a mesh arriving after its companions
# clears them as though they had belonged to a previous session — and the
# sweep unlinks them from the port directory too, so nothing is recoverable.


@pytest.fixture
def drop_files(tmp_path):
    """A mesh, the components naming its faces, and a skeleton of it."""
    from skeliner import io as skio
    from skeliner.dataclass import MeshComponents
    from skeliner.skeletonize import skeletonize

    mesh, soma, org, neurites = _tube_with_components()
    comp = MeshComponents(
        soma=soma, organelles=org, neurites=neurites, discarded=Discarded([])
    )

    mesh_path = tmp_path / "mesh_cleaned.obj"
    comp_path = tmp_path / "components.npz"
    skel_path = tmp_path / "skeleton.npz"
    mesh.export(mesh_path)
    skio.save_components_npz(comp, comp_path)
    skeletonize(mesh, verbose=False).to_npz(skel_path)

    # The order a Finder selection arrives in, and the one that used to lose
    # the components: "components.npz" sorts ahead of "mesh_cleaned.obj".
    assert sorted(p.name for p in (mesh_path, comp_path)) == [
        "components.npz",
        "mesh_cleaned.obj",
    ]
    return mesh_path, comp_path, skel_path


@pytest.fixture
def drop_client(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path / "state", raising=False)
    with TestClient(_create_app(port=8913)) as client:
        yield client


def _as_batch(*paths):
    return [
        ("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in paths
    ]


def _neurite_labels(client):
    ann = client.get("/annotations").json()
    return [h.get("label", "") for h in ann.get("highlights", []) if "neurite" in h]


def test_a_drop_applies_the_mesh_before_the_files_that_index_it(
    drop_client, drop_files
):
    """Ordered by what the files are, not by when they arrived.  Sorting on
    the client would fix the drag-drop path and leave every other caller
    with the old behaviour."""
    mesh_path, comp_path, _ = drop_files
    r = drop_client.post("/upload_batch", files=_as_batch(comp_path, mesh_path))

    assert r.status_code == 200, r.text
    assert [x["name"] for x in r.json()["results"]] == [
        "mesh_cleaned.obj",
        "components.npz",
    ]


def test_components_survive_a_mesh_that_arrived_after_them(drop_client, drop_files):
    mesh_path, comp_path, _ = drop_files
    drop_client.post("/upload_batch", files=_as_batch(comp_path, mesh_path))

    assert drop_client.get("/loaded").json()["mesh"] is not None
    assert _neurite_labels(drop_client), "the mesh cleared the components it came with"


def test_a_drop_still_clears_what_belonged_to_the_previous_mesh(
    drop_client, drop_files
):
    """Batching spares the gesture's own files, not the session before it."""
    mesh_path, comp_path, _ = drop_files
    drop_client.post("/upload_batch", files=_as_batch(comp_path, mesh_path))
    assert _neurite_labels(drop_client)

    drop_client.post("/upload_batch", files=_as_batch(mesh_path))
    assert not _neurite_labels(drop_client)


def test_a_skeleton_in_a_drop_binds_to_the_mesh_it_came_with(drop_client, drop_files):
    """It is applied after the mesh, so ``_skel_is_stale`` sees the object
    the drop carried rather than whatever happened to be loaded."""
    mesh_path, _, skel_path = drop_files
    drop_client.post("/upload_batch", files=_as_batch(skel_path, mesh_path))

    r = drop_client.post(
        "/edge_edit_preview", json={"name": "skeleton.npz", "u": 1, "v": 2}
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"], r.json()


def test_a_drop_naming_two_meshes_is_refused(drop_client, drop_files, tmp_path):
    """Which mesh the companions belong to is unanswerable, and guessing
    silently discards one set."""
    mesh_path, comp_path, _ = drop_files
    other = tmp_path / "other.obj"
    trimesh.creation.box(extents=(4, 4, 4)).export(other)

    r = drop_client.post("/upload_batch", files=_as_batch(mesh_path, other, comp_path))
    assert r.status_code == 400
    assert "more than one mesh" in r.json()["error"]


def test_a_new_mesh_clears_the_components_that_chose_the_track(drop_client, drop_files):
    """``/skeletonize`` reads ``neurites`` to pick the preprocessing track.
    Left behind, a components file from the mesh being replaced sends the
    next run down that track naming faces that no longer exist."""
    mesh_path, comp_path, _ = drop_files
    drop_client.post("/upload_batch", files=_as_batch(comp_path, mesh_path))
    assert _neurite_labels(drop_client)

    drop_client.post(
        "/upload",
        files={"file": (mesh_path.name, mesh_path.read_bytes(), "application/octet")},
    )
    assert not _neurite_labels(drop_client)


# ── a mesh edit invalidates the components ────────────────────────────
#
# `break_up_mesh` runs once the mesh is settled, so a components split and a
# mesh edit should not coexist.  The split names faces of the surface as it
# was, and `/skeletonize` reads it to choose the preprocessing track — left
# behind, it sends the next run down that track naming faces of a mesh that
# has moved on.


def test_a_mesh_edit_drops_the_components(reassign_client):
    """The routes a user actually clicks, which is where a missed site
    shows — the same reason the skeleton rule is tested through one."""
    client, state, neurites = reassign_client
    assert state["neurites"] is not None

    r = client.post(
        "/remove_selected", json={"faces": [int(f) for f in neurites[0][:5]]}
    )
    assert r.status_code == 200, r.text

    assert state["neurites"] is None
    assert state["discarded"] is None


def test_dropping_the_components_is_announced(reassign_client, capsys):
    """An invalidation that happens quietly is read as a bug.  Checked on
    stdout because ``_log`` prints and broadcasts the same line."""
    client, _, neurites = reassign_client

    client.post("/remove_selected", json={"faces": [int(f) for f in neurites[0][:5]]})

    said = capsys.readouterr().out
    assert "components dropped" in said, said


def test_preprocess_does_not_announce_a_drop_it_repairs(reassign_client, capsys):
    """It ends in break_up_mesh, so the components come straight back; a
    note telling the user to rebuild them would be wrong."""
    client, state, _ = reassign_client

    r = client.post("/preprocess", json={})
    assert r.status_code == 200, r.text

    said = capsys.readouterr().out
    assert "components dropped" not in said, said
    assert state["neurites"] is not None, "preprocess must republish"


# ── which track a run takes, and which parameters it reads ────────────
#
# `skeletonize` accepts every parameter on either track and forwards only
# the applicable ones, so a setting made against the track that is not
# running is ignored in silence.  The viewer picks the track from state the
# user cannot see, so it has to say which one it picked and which of the
# parameters that one reads.

TRACK_HEADERS = {
    "direct": "skeletonize from raw mesh",
    "preprocessing": "skeletonize from preprocessed mesh components",
}


def _skel_defaults_tracks():
    """Parse the dialog's per-parameter track tags out of viewer.html."""
    import re
    from pathlib import Path

    from skeliner.plot import viewer as viewer_mod

    html = Path(viewer_mod.__file__).with_name("viewer.html").read_text()
    block = html.split("const SKEL_DEFAULTS = {")[1].split("\n            };")[0]

    tags = {}
    for m in re.finditer(r"(\w+):\s*\{", block):
        depth, j = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(block[j], 0)
            j += 1
        body = block[m.end() : j - 1]
        raw = re.search(r"tracks:\s*(BOTH|\[[^\]]*\])", body)
        assert raw, f"{m.group(1)} carries no tracks tag"
        tags[m.group(1)] = (
            ["direct", "preprocessing"]
            if raw.group(1) == "BOTH"
            else re.findall(r'"(\w+)"', raw.group(1))
        )
    return tags


def test_the_dialog_greys_exactly_what_the_track_ignores():
    """The tags decide which fields are disabled.  Wrong ones would either
    grey out a live parameter or leave a dead one editable, and both read as
    the dialog working."""
    import inspect
    import re
    import sys
    from pathlib import Path

    src = Path(sys.modules["skeliner.skeletonize"].__file__).read_text()
    forwarded = src.split("return _skeletonize_preproc(")[1].split("\n        )")[0]
    prepro = set(re.findall(r"(\w+)=", forwarded))
    direct = set(
        inspect.signature(
            sys.modules["skeliner.skeletonize"]._skeletonize_direct
        ).parameters
    )

    tags = _skel_defaults_tracks()
    assert len(tags) > 10, "parser found almost nothing — it has drifted"

    for key, declared in tags.items():
        actual = sorted(
            (["direct"] if key in direct else [])
            + (["preprocessing"] if key in prepro else [])
        )
        assert sorted(declared) == actual, (
            f"{key}: dialog says {sorted(declared)}, tracks forward {actual}"
        )


@pytest.mark.parametrize("track", ["direct", "preprocessing"])
def test_the_run_reports_and_logs_the_track_it_took(reassign_client, capsys, track):
    """Reported by the server rather than inferred by the page, so the label
    and the branch cannot disagree."""
    client, state, _ = reassign_client
    if track == "direct":
        state["neurites"] = None

    assert client.get("/loaded").json()["track"] == track

    r = client.post("/skeletonize", json={"params": {}})
    assert r.status_code == 200, r.text
    assert r.json()["ranTrack"] == track
    assert TRACK_HEADERS[track] in capsys.readouterr().out


def _skel_defaults_values():
    """Parse the dialog's declared default for each parameter."""
    import re
    from pathlib import Path

    from skeliner.plot import viewer as viewer_mod

    html = Path(viewer_mod.__file__).with_name("viewer.html").read_text()
    block = html.split("const SKEL_DEFAULTS = {")[1].split("\n            };")[0]

    literals = {"true": True, "false": False}
    out = {}
    for m in re.finditer(r"(\w+):\s*\{", block):
        depth, j = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(block[j], 0)
            j += 1
        raw = re.search(r"value:\s*([^,]+),", block[m.end() : j - 1]).group(1).strip()
        out[m.group(1)] = (
            literals[raw]
            if raw in literals
            else (float(raw) if "." in raw else int(raw))
        )
    return out


def test_the_dialog_agrees_with_skeletonize_about_the_defaults():
    """The dialog decides what counts as "unchanged", and `getSkelParams`
    sends only what differs.  A dialog default drifting from the signature
    would either send a value the user never chose, or withhold one they
    did — and the panel would look right either way."""
    import inspect
    import sys

    sig = inspect.signature(sys.modules["skeliner.skeletonize"].skeletonize).parameters
    declared = _skel_defaults_values()
    assert len(declared) > 10, "parser found almost nothing — it has drifted"

    for key, val in declared.items():
        assert key in sig, f"{key} is offered by the dialog but is not a parameter"
        assert sig[key].default == val, (
            f"{key}: dialog offers {val!r}, skeletonize defaults to "
            f"{sig[key].default!r}"
        )


# ── the mode selector ─────────────────────────────────────────────────
#
# Three modes, named after the three objects in the derivation chain: the
# mesh, the components broken out of it, the skeleton built from those.
# Each is wired through three parallel tables — the buttons, the row
# containers they reveal, and the click handlers — so renaming one means
# editing all three.  A site missed leaves a button that lights up over an
# empty panel, which reads as the mode simply having nothing in it.


def _mode_wiring():
    """Parse the mode keys, their element ids and the click wiring."""
    import re
    from pathlib import Path

    from skeliner.plot import viewer as viewer_mod

    html = Path(viewer_mod.__file__).with_name("viewer.html").read_text()

    def table(name):
        block = html.split(f"const {name} = {{")[1].split("};")[0]
        return dict(re.findall(r'(\w+):\s*"([^"]+)"', block))

    wired = dict(
        re.findall(
            r'getElementById\("(\w+)"\)\s*\.addEventListener\('
            r'"click",\s*\(\)\s*=>\s*setMode\("(\w+)"\)',
            html,
        )
    )
    default = re.search(r'let panelMode = "(\w+)"', html).group(1)
    return html, table("MODE_ROWS"), table("MODE_BTNS"), wired, default


def test_every_mode_has_a_button_a_panel_and_a_handler():
    """`setMode` shows one row container and lights one button, both looked
    up by id.  An id that names nothing throws inside the loop, leaving the
    panel on whichever mode it was already displaying."""
    html, rows, btns, wired, default = _mode_wiring()

    assert set(rows) == set(btns), (
        f"MODE_ROWS has {sorted(rows)}, MODE_BTNS has {sorted(btns)}"
    )
    assert len(rows) == 3, f"expected three modes, parsed {sorted(rows)}"
    assert default in rows, f"panelMode starts at {default!r}, not a mode"

    for mode, row_id in rows.items():
        assert f'id="{row_id}"' in html, f"{mode}: no element id={row_id!r}"
    for mode, btn_id in btns.items():
        assert f'id="{btn_id}"' in html, f"{mode}: no element id={btn_id!r}"
        assert wired.get(btn_id) == mode, (
            f"{btn_id} calls setMode({wired.get(btn_id)!r}), expected {mode!r}"
        )


def test_the_mode_buttons_are_exactly_the_ones_styled_as_modes():
    """The active-mode highlight is CSS on `button.mode-btn[data-active]`.
    A mode button without the class switches the panel while still looking
    inactive; anything else carrying it looks like a fourth mode."""
    import re

    html, _, btns, _, _ = _mode_wiring()

    styled = set(re.findall(r'<button\s+id="(\w+)"\s+class="mode-btn"', html))
    assert styled == set(btns.values()), (
        f"styled as modes: {sorted(styled)}, declared: {sorted(btns.values())}"
    )


# ── a skeleton must belong to the mesh it is loaded beside ────────────
#
# Binding is not checking.  A skeleton binds to whatever mesh is loaded when
# it arrives, which is the *claim* that the two go together; nothing tested
# the claim.  `_skel_is_stale` cannot: it compares mesh identity, so it sees
# a mesh swapped mid-session and is blind to a pair that never matched.
#
# The failure was asymmetric and the quiet half was the dangerous one — a
# smaller wrong mesh raised IndexError inside dx.face_owner (a 500), while a
# larger one kept every vertex id in range and returned bins, surface support
# and edit previews computed against the wrong cell, all with 200.


@pytest.fixture
def mismatch_client(tmp_path, monkeypatch):
    """A skeleton of a tube, and two unrelated meshes to mispair it with."""
    from starlette.testclient import TestClient

    from skeliner import skeletonize
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    own = trimesh.creation.cylinder(radius=40.0, height=1200.0, sections=24)
    own = own.subdivide().subdivide()
    skel = skeletonize(own, verbose=False)
    (tmp_path / "skeleton.npz").write_bytes(b"")
    skel.to_npz(tmp_path / "skeleton.npz")

    bigger = trimesh.creation.icosphere(subdivisions=4, radius=300.0)
    smaller = trimesh.creation.icosphere(subdivisions=1, radius=300.0)
    assert len(bigger.vertices) > len(own.vertices) > len(smaller.vertices)
    for m, n in ((own, "own.obj"), (bigger, "bigger.obj"), (smaller, "smaller.obj")):
        m.export(tmp_path / n)

    def load(mesh_file):
        client = TestClient(viewer_mod._create_app(port=8914))
        client.__enter__()
        client.post(
            "/upload_batch",
            files=[
                ("files", (mesh_file, (tmp_path / mesh_file).read_bytes())),
                ("files", ("skeleton.npz", (tmp_path / "skeleton.npz").read_bytes())),
            ],
        )
        name = next(iter(client.get("/skeletons").json()))
        return client, name

    yield load


@pytest.mark.parametrize("mesh_file", ["bigger.obj", "smaller.obj"])
def test_a_mispaired_skeleton_is_refused_by_every_route_that_reads_the_mesh(
    mismatch_client, mesh_file
):
    client, name = mismatch_client(mesh_file)
    for url, payload in (
        ("/bin", {"name": name, "node": 2}),
        ("/edge_support", {"name": name}),
        ("/bin_reassign_preview", {"name": name, "to": 2, "faces": [0, 1, 2]}),
        ("/bin_split_preview", {"name": name, "node": 2, "faces": [0, 1]}),
        ("/edge_edit_preview", {"name": name, "u": 1, "v": 2}),
    ):
        r = client.post(url, json=payload)
        assert r.status_code == 409, f"{url} returned {r.status_code}"
        assert "does not belong to the loaded mesh" in r.json()["error"], url


def test_the_matching_mesh_is_still_allowed(mismatch_client):
    """The guard has to let the real pair through, or it has just disabled
    the feature it protects."""
    client, name = mismatch_client("own.obj")
    assert client.post("/bin", json={"name": name, "node": 2}).status_code == 200
    assert client.post("/edge_support", json={"name": name}).status_code == 200


def test_export_survives_a_mispairing(mismatch_client):
    """The skeleton is not damaged by a wrong mesh standing next to it, and
    edits are already refused, so there is nothing left to protect against."""
    client, name = mismatch_client("bigger.obj")
    assert client.get(f"/export_skeleton?name={name}").status_code == 200


def test_the_pairing_verdict_rides_along_with_the_layer(mismatch_client):
    """The client cannot ask a question it does not know to ask, so the
    verdict travels with the skeleton rather than waiting for a click."""
    client, name = mismatch_client("bigger.obj")
    layer = client.get("/skeletons").json()[name]
    assert layer["pairing"]["ok"] is False
    assert layer["pairing"]["verified"] is False

    good, gname = mismatch_client("own.obj")
    assert good.get("/skeletons").json()[gname]["pairing"] == {
        "ok": True,
        "verified": True,
        "reason": "counts match the mesh this was built from",
        "meshVertices": 770,
        "meshFaces": 1536,
    }


# ---------------------------------------------------------------------
# diagnostics — listing candidates to look at
#
# The panel's whole purpose is that a threshold here is unvalidated, so a
# candidate has to be reachable by eye.  These check the route produces
# something the marker channel can carry, and that it stays a *lister*.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind",
    ["junctions", "cycles", "twigs", "degree", "tips", "orphans"],
)
def test_every_offered_diagnostic_runs(bin_client, kind):
    """A panel entry that 500s is worse than no panel entry.

    The dropdown is built from `DIAGNOSTICS`, so each key has to survive a
    real skeleton — several of these reach for `soma`, `radii` or a rooted
    traversal, and a fixture without one would take the route down.
    """
    client, name, _mesh, _skel = bin_client
    r = client.post("/diagnose", json={"name": name, "kind": kind})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["kind"] == kind
    assert body["n"] >= 0


def test_the_dropdown_offers_exactly_the_diagnostics_that_exist(bin_client):
    """A named option with no implementation is a dead menu entry, and an
    implementation with no option is unreachable."""
    import re
    from pathlib import Path

    from skeliner.plot import viewer as viewer_mod

    client, name, _mesh, _skel = bin_client
    html = Path(viewer_mod.__file__).with_name("viewer.html").read_text()
    sel = re.search(r'id="dxKindSel".*?</select>', html, re.S | re.I)
    assert sel, "diagnostics dropdown not found"
    offered = set(re.findall(r'<option value="([a-z_]+)"', sel.group(0)))

    for kind in offered:
        assert (
            client.post("/diagnose", json={"name": name, "kind": kind}).status_code
            == 200
        ), f"dropdown offers {kind!r} but the route refuses it"

    declared = set(
        re.findall(
            r"^\s*(\w+):\s*\[",
            re.search(r"const DX_PARAMS = \{(.*?)\n            \};", html, re.S).group(
                1
            ),
            re.M,
        )
    )
    assert declared == offered, (
        f"DX_PARAMS and the dropdown disagree: {declared ^ offered}"
    )


def test_the_diagnostics_are_listed_in_the_order_dx_lists_them():
    """Three places name this list, and a menu with no rule gets appended to.

    `dx.__skeleton__` already runs invariant checks, then enumerations, then
    heuristics, so the panel follows it rather than inventing an order —
    which also makes the dropdown read from "this is broken" down to "this
    might repay a look".
    """
    import re
    from pathlib import Path

    from skeliner import dx
    from skeliner.plot import viewer as viewer_mod

    html = Path(viewer_mod.__file__).with_name("viewer.html").read_text()
    sel = re.search(r'id="dxKindSel".*?</select>', html, re.S | re.I)
    dropdown = re.findall(r'<option value="([a-z_]+)"', sel.group(0))
    params = re.findall(
        r"^\s*(\w+):\s*\[",
        re.search(r"const DX_PARAMS = \{(.*?)\n            \};", html, re.S).group(1),
        re.M,
    )
    source = Path(viewer_mod.__file__).read_text()
    registry = re.findall(
        r'^\s*"([a-z_]+)": \(',
        re.search(r"DIAGNOSTICS = \{(.*?)\n    \}", source, re.S).group(1),
        re.M,
    )

    assert dropdown == params, f"dropdown {dropdown} vs DX_PARAMS {params}"
    assert dropdown == registry, f"dropdown {dropdown} vs DIAGNOSTICS {registry}"

    # and that shared order is dx's own
    backing = {
        "orphans": "check_connectivity",
        "cycles": "cycles",
        "degree": "nodes_of_degree",
        "twigs": "twigs_of_length",
        "tips": "suspicious_tips",
        "junctions": "suspicious_junctions",
    }
    assert set(backing) == set(dropdown), "a diagnostic has no declared dx backing"
    ranks = [dx.__skeleton__.index(backing[k]) for k in dropdown]
    assert ranks == sorted(ranks), (
        f"panel order {dropdown} does not follow dx.__skeleton__ (ranks {ranks})"
    )


def test_a_diagnostic_marks_nodes_without_touching_the_skeleton(bin_client):
    """It reports candidates and never edits — the same split `pre` draws
    between its `find_*` and `remove_*` functions."""
    client, name, _mesh, skel = bin_client
    before_nodes = skel.nodes.copy()
    before_edges = np.asarray(skel.edges).copy()

    # The fixture is a tube, so its skeleton is a chain with no node above
    # degree 2 — asking for degree>=3 here would pass on an empty list and
    # assert nothing.  Degree 1 is what this fixture actually has.
    r = client.post(
        "/diagnose",
        json={"name": name, "kind": "degree", "params": {"minDegree": 1}},
    )
    assert r.status_code == 200
    assert r.json()["n"] > 0, "fixture produced no candidates to check with"

    after = client.get("/skeletons").json()[name]
    assert after["nNodes"] == len(before_nodes)
    assert np.array_equal(np.asarray(skel.edges), before_edges)

    markers = client.get("/annotations").json().get("markers", [])
    assert markers, "a diagnostic must leave something to look at"
    for m in markers:
        assert m["skelName"] == name
        assert isinstance(m["node"], int)
        assert m["dx"] == "degree"
        assert len(m["position"]) == 3
        assert m["radius"] > 0


def test_re_running_a_diagnostic_replaces_its_own_markers_only(bin_client):
    """Turning a threshold and re-running must answer *what does this cut do*,
    not stack three answers on top of each other — while leaving the other
    diagnostics', and the user's own, markers alone."""
    client, name, _mesh, _skel = bin_client

    client.post(
        "/diagnose",
        json={"name": name, "kind": "degree", "params": {"minDegree": 1}},
    )
    wide = client.get("/annotations").json()["markers"]
    assert len(wide) > 0, "fixture produced no candidates to check with"

    client.post("/diagnose", json={"name": name, "kind": "tips"})
    both = client.get("/annotations").json()["markers"]
    assert len([m for m in both if m["dx"] == "degree"]) == len(wide)

    client.post(
        "/diagnose",
        json={"name": name, "kind": "degree", "params": {"minDegree": 99}},
    )
    after = client.get("/annotations").json()["markers"]
    assert [m for m in after if m["dx"] == "degree"] == []
    assert len([m for m in after if m["dx"] == "tips"]) == len(
        [m for m in both if m["dx"] == "tips"]
    )


def test_a_diagnostic_refuses_a_skeleton_that_is_not_the_loaded_mesh(
    mismatch_client,
):
    """Deciding a junction is wrong needs the surface it came from.  Computing
    the list mesh-free is possible but pointless: the answer exists only to be
    checked against a mesh, so an unpaired one is refused rather than shown."""
    client, name = mismatch_client("bigger.obj")
    r = client.post("/diagnose", json={"name": name, "kind": "degree"})
    assert r.status_code == 409
    assert "does not belong to the loaded mesh" in r.json()["error"]


def test_a_marker_lands_exactly_on_the_node_it_marks(tmp_path, monkeypatch):
    """Everything the client draws is centroid-relative.

    `_mesh_to_buffers` subtracts the centroid from the vertices and
    `_skeleton_to_buffers` subtracts the same one from the nodes, so the two
    overlay.  A marker built from raw skeleton coordinates is off by a whole
    centroid — on a CAVE cell that is hundreds of microns, far outside the
    mesh, and it reads as "the diagnostic found nothing there" rather than
    as a coordinate bug.

    The mesh is deliberately built **away from the origin**.  Every other
    fixture here is a trimesh primitive centred on it, where the centroid is
    zero and subtracting it is a no-op — this test passes on the broken code
    against any of them.
    """
    from starlette.testclient import TestClient

    from skeliner import skeletonize
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    mesh = trimesh.creation.cylinder(radius=40.0, height=1200.0, sections=24)
    mesh = mesh.subdivide().subdivide()
    mesh.apply_translation([480_000.0, -310_000.0, 205_000.0])
    assert np.linalg.norm(mesh.vertices.mean(axis=0)) > 1e5, (
        "the fixture must sit away from the origin or a missing centroid "
        "subtraction is invisible"
    )
    skel = skeletonize(mesh, verbose=False)

    app = _create_app(preload_mesh=mesh, port=8917)
    with TestClient(app) as client:
        path = tmp_path / "skeleton.npz"
        skel.to_npz(path)
        client.post("/upload", files={"file": ("skeleton.npz", path.read_bytes())})
        name = next(iter(client.get("/skeletons").json()))

        r = client.post(
            "/diagnose",
            json={"name": name, "kind": "degree", "params": {"minDegree": 1}},
        )
        assert r.json()["n"] > 0, "fixture produced no candidates to check with"

        drawn = client.get("/skeletons").json()[name]["nodes"]
        markers = client.get("/annotations").json()["markers"]
        assert markers

        for m in markers:
            nid = m["node"]
            expected = drawn[3 * nid : 3 * nid + 3]
            assert m["position"] == pytest.approx(expected, abs=1.0), (
                f"marker for node {nid} is not where that node is drawn"
            )


def test_a_loop_is_drawn_so_it_can_be_traced(tmp_path, monkeypatch):
    """A loop is the one candidate whose *edges* are the answer.

    It is a single connected thing you follow round to find where the tree
    closed it wrongly, so drawing it is the whole point — unlike ninety
    scattered twigs, which pooled into one annotation can be neither focused
    nor acted on together and were dropped for that reason.

    Also guards the coordinate frame: segments are drawn centroid-relative,
    and a missed subtraction puts the loop a whole centroid from the cell.
    """
    from pathlib import Path

    from starlette.testclient import TestClient

    from skeliner import post, skeletonize
    from skeliner.io import load_mesh
    from skeliner.plot import viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_STATE_DIR", tmp_path, raising=False)
    mesh = load_mesh(Path(__file__).parent / "data" / "60427.obj")
    skel = skeletonize(mesh, verbose=False)
    # Assert a connection the MST did not make — the only way a loop exists.
    post.graft(skel, 2, 6)

    app = _create_app(preload_mesh=mesh, port=8919)
    with TestClient(app) as client:
        path = tmp_path / "skeleton.npz"
        skel.to_npz(path)
        client.post("/upload", files={"file": ("skeleton.npz", path.read_bytes())})
        name = next(iter(client.get("/skeletons").json()))

        r = client.post("/diagnose", json={"name": name, "kind": "cycles"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n"] == 1, "the graft should have opened exactly one loop"

        ann = client.get("/annotations").json()
        groups = [g for g in ann.get("edge_groups", []) if g.get("dx") == "cycles"]
        assert groups, "the loop never reached the geometry"
        assert any("the loop" in g["label"] for g in groups)
        assert all(g.get("overlay") for g in groups), (
            "a highlight tracing edges the skeleton already draws must sit on "
            "top of it, or it is only visible once the skeleton is hidden"
        )

        drawn = np.asarray(client.get("/skeletons").json()[name]["nodes"]).reshape(
            -1, 3
        )
        lo, hi = drawn.min(axis=0) - 1.0, drawn.max(axis=0) + 1.0
        for g in groups:
            assert g["segments"], "an empty group draws nothing"
            for seg in g["segments"]:
                for point in seg:
                    p = np.asarray(point)
                    assert ((p >= lo) & (p <= hi)).all(), (
                        "segment drawn outside the skeleton — centroid not applied"
                    )


def test_an_unknown_diagnostic_is_named_rather_than_crashing(bin_client):
    client, name, _mesh, _skel = bin_client
    r = client.post("/diagnose", json={"name": name, "kind": "nonesuch"})
    assert r.status_code == 400
    assert "nonesuch" in r.json()["error"]


# ── every annotation channel you can list, you can fly to ─────────────
#
# `markers` had a list row, a colour swatch, a visibility toggle and a
# remove button — and no way to centre the camera on it, while `highlights`
# and `edge_groups` had both a focus button and a double-click.  Nothing
# failed; the channel was simply unreachable, which reads as "the list is
# for reading".  It is also the channel a node-level diagnostic emits into,
# so the omission was on the one that most needed it.


def _focus_wiring():
    """Which channels have a focus helper, and how it is reachable."""
    import re
    from pathlib import Path

    from skeliner.plot import viewer as viewer_mod

    html = Path(viewer_mod.__file__).with_name("viewer.html").read_text()
    defined = set(re.findall(r"function (focusOn\w+)\(", html))
    dblclick = set(re.findall(r'"dblclick",\s*\(\)\s*=>\s*(focusOn\w+)\(', html))
    buttons = set(re.findall(r'"click",\s*\(\)\s*=>\s*(focusOn\w+)\(', html))
    return html, defined, dblclick, buttons


def test_every_listed_annotation_channel_can_be_centred_on():
    html, defined, dblclick, buttons = _focus_wiring()

    # The channels applyAnnotations renders, that also get a list row.
    channels = {
        "highlights": "focusOnAnnotation",
        "edge_groups": "focusOnEdgeGroup",
        "markers": "focusOnMarker",
    }
    for channel, fn in channels.items():
        assert f"annotations.{channel}" in html, f"{channel} is no longer rendered"
        assert fn in defined, f"{channel} has a list but no {fn}()"
        assert fn in dblclick, f"{fn} is not reachable by double-clicking the row"
        assert fn in buttons, f"{fn} has no focus button"


def test_a_marker_can_name_the_node_it_marks():
    """What makes a marker a diagnostic result rather than a pin: carrying
    the node id lets focusing one select its bin, which is what arms the
    edit verbs the row exists to lead to."""
    html, _, _, _ = _focus_wiring()
    body = html.split("function focusOnMarker(")[1].split("\n            }")[0]

    assert "mk.node" in body, "focusOnMarker ignores the node a marker names"
    assert "selectBin(mk.node)" in body, "focusing a node marker does not select it"
    # Toggling would make focusing the same row twice deselect it.
    assert "clearBinSelection()" in body, "selection is toggled rather than set"
    assert 'panelMode !== "skeleton"' in body, (
        "bin selection is attempted outside the mode that has a bin panel"
    )


# ── contact sites ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def two_kinds():
    """A tube touched by one tube crossing it and one lying beside it."""
    from skeliner import pair

    from .test_pair import _tube

    a = _tube([0, 0, 1], [0.0, 0.0, 0.0])
    b = trimesh.util.concatenate(
        [_tube([1, 0, 0], [0.0, 405.0, 0.0]), _tube([0, 0, 1], [405.0, 0.0, 0.0])]
    )
    return a, b, pair.scan_contacts(a, b, tol=20.0)


def _overlay(a, b, sites, **kw):
    from skeliner.plot.viewer import _contact_overlay

    args = dict(sides="both", show_aabb=True, aabb_mode="union", color_by="kind")
    return _contact_overlay(a, b, np.zeros(3), sites, **{**args, **kw})


def test_contact_sites_are_coloured_and_labelled_by_kind(two_kinds):
    from skeliner.plot.viewer import _KIND_COLORS

    a, b, sites = two_kinds
    assert sorted(sites.kind.tolist()) == ["crossing", "other"]
    ann, state = _overlay(a, b, sites)
    for h in ann["highlights"]:
        kind = next(k for k in _KIND_COLORS if f" · {k} " in h["label"])
        assert h["color"] == _KIND_COLORS[kind]
    assert [k["kind"] for k in state["kinds"]] == ["crossing", "other"]
    assert [k["count"] for k in state["kinds"]] == [1, 1]
    assert all(" · " in g["label"] for g in ann["edge_groups"])


def test_colouring_by_site_keeps_the_kinds_in_the_labels(two_kinds):
    from skeliner.plot.viewer import _SITE_COLORS

    a, b, sites = two_kinds
    ann, state = _overlay(a, b, sites, color_by="site")
    colours = {h["label"].split(" ")[1][:-1]: h["color"] for h in ann["highlights"]}
    assert colours == {"0": _SITE_COLORS[0], "1": _SITE_COLORS[1]}
    assert all(" · " in h["label"] for h in ann["highlights"])
    assert "kinds" not in state


def test_sites_without_kinds_look_as_before(two_kinds):
    from skeliner import pair
    from skeliner.plot.viewer import _SITE_COLORS

    a, b, _ = two_kinds
    plain = pair.scan_contacts(a, b, tol=20.0, label=False)
    ann, state = _overlay(a, b, plain)
    assert all(" · " not in h["label"] for h in ann["highlights"])
    assert {tuple(h["color"]) for h in ann["highlights"]} <= {
        tuple(c) for c in _SITE_COLORS
    }
    assert "kinds" not in state and state["nSites"] == 2
