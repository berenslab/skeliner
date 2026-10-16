"""
Interactive mesh & skeleton viewer for skeliner.

Usage:
    skeliner view                        # empty viewer, drag & drop files
    skeliner view path/to/mesh.obj       # pre-load a mesh

Launches a local web server with a Three.js viewer. Supports:
  - Mesh files: .obj, .ply, .stl (always in nm)
  - Skeleton files: .swc (always in µm → converted to nm),
                    .npz (unit from metadata)

State files are written to /tmp/skeliner_view/<port>/ for
communication with external tools (e.g. Claude).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

_STATE_DIR = Path(os.environ.get("SKELINER_VIEW_STATE_DIR", "/tmp/skeliner_view"))


def _ensure_state_dir():
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


# ── Mesh helpers ──────────────────────────────────────────────────────


def _mesh_to_buffers(
    mesh: trimesh.Trimesh, *, centroid: np.ndarray | None = None
) -> dict[str, Any]:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if centroid is None:
        centroid = verts.mean(axis=0)
    verts_centered = verts - centroid
    return {
        "vertices": verts_centered.ravel().tolist(),
        "faces": faces.ravel().tolist(),
        "nVertices": len(verts),
        "nFaces": len(faces),
        "centroid": centroid.tolist(),
    }


# ── Skeleton helpers ──────────────────────────────────────────────────


def _skeleton_to_buffers(skel, centroid: np.ndarray) -> dict[str, Any]:
    """Convert a Skeleton to line-segment buffers for Three.js.

    Coordinates are shifted by the same centroid as the mesh so they
    overlay correctly.
    """
    nodes = np.asarray(skel.nodes, dtype=np.float32) - centroid
    edges = np.asarray(skel.edges, dtype=np.int32)
    radii = np.asarray(skel.r, dtype=np.float32)

    # Line segments: pairs of (x,y,z) for each edge
    positions = np.empty((len(edges) * 2, 3), dtype=np.float32)
    positions[0::2] = nodes[edges[:, 0]]
    positions[1::2] = nodes[edges[:, 1]]

    # Per-edge radius (average of endpoints)
    edge_radii = (radii[edges[:, 0]] + radii[edges[:, 1]]) / 2.0

    return {
        "nodes": nodes.ravel().tolist(),
        "edges": edges.ravel().tolist(),
        "positions": positions.ravel().tolist(),
        "radii": radii.tolist(),
        "edgeRadii": edge_radii.tolist(),
        "nNodes": len(nodes),
        "nEdges": len(edges),
    }


def _load_skeleton_as_nm(path: Path) -> Any:
    """Load a skeleton file and convert to nm."""
    from skeliner.io import load_skeleton_npz, load_skeleton_swc

    suffix = path.suffix.lower()
    if suffix == ".swc":
        skel = load_skeleton_swc(path)
        # Check if unit is in meta; if not, infer from coordinate magnitude
        unit = skel.meta.get("unit", None)
        if unit is None:
            # Heuristic: if max coordinate > 10000, likely nm already
            if skel.nodes.max() > 10000:
                unit = "nm"
            else:
                unit = "um"
        if unit in ("um", "µm", "μm", "micron", "micrometer"):
            skel.nodes *= 1000.0
            for k in skel.radii:
                skel.radii[k] *= 1000.0
            skel.soma.center *= 1000.0
            skel.soma.axes *= 1000.0
        return skel
    elif suffix == ".npz":
        skel = load_skeleton_npz(path)
        unit = skel.meta.get("unit", "nm")
        if unit in ("um", "µm", "μm", "micron", "micrometer"):
            skel.nodes *= 1000.0
            for k in skel.radii:
                skel.radii[k] *= 1000.0
            skel.soma.center *= 1000.0
            skel.soma.axes *= 1000.0
        return skel
    else:
        raise ValueError(f"Unsupported skeleton format: {suffix}")


def _is_organelle_data(path: Path) -> bool:
    """Check if an npz file contains organelle masks."""
    try:
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            return "pocket" in files or "isolated" in files
    except Exception:
        return False


def _is_mesh_stats_data(path: Path) -> bool:
    """Check if an npz file contains standalone MeshStats data."""
    try:
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            return (
                "outward_dots" in files
                and "face_comp" in files
                and "main_ci" in files
                and "pocket" not in files
                and "isolated" not in files
            )
    except Exception:
        return False


def _is_soma_data(path: Path) -> bool:
    """Check if an npz file contains standalone soma data."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return "center" in data and "axes" in data and "R" in data
    except Exception:
        return False


def _is_components_data(path: Path) -> bool:
    """Check if an npz file contains MeshComponents data."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return "n_neurites" in data and "n_discarded" in data
    except Exception:
        return False


def _is_neurites_data(path: Path) -> bool:
    """Check if an npz file contains Neurites data."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return "n" in data and "c0" in data and "n_neurites" not in data
    except Exception:
        return False


def _is_l2_graph(path: Path) -> bool:
    """Check if an npz file is an L2 graph (not a skeliner skeleton)."""
    with np.load(path, allow_pickle=False) as data:
        return "graph_nodes" in data and "graph_edges" in data


#: Order in which the files of one drop are applied.  A mesh has to land
#: before anything that indexes it — components and annotations name its
#: faces, and a skeleton binds to whichever mesh is loaded when it arrives
#: (see ``_skel_is_stale``).  The browser hands over a multi-file drop in the
#: OS's order, usually alphabetical, so without this a ``components.npz`` that
#: sorts ahead of ``mesh.obj`` was applied first and then wiped by the very
#: mesh it belongs to.
_TIER_MESH = 0
_TIER_MESH_DERIVED = 1
_TIER_SKELETON = 2
_TIER_ANNOTATION = 3


def _upload_tier(path: Path) -> int:
    """Which tier *path* belongs to; lower tiers are applied first."""
    suffix = path.suffix.lower()
    if suffix in (".obj", ".ply", ".stl"):
        return _TIER_MESH
    if suffix == ".swc":
        return _TIER_SKELETON
    if suffix == ".json":
        return _TIER_ANNOTATION
    if suffix == ".npz":
        # Sniffed in the same order as _apply_upload dispatches, so a file
        # lands in the tier belonging to the branch that will handle it.
        if (
            _is_organelle_data(path)
            or _is_mesh_stats_data(path)
            or _is_soma_data(path)
            or _is_components_data(path)
            or _is_neurites_data(path)
        ):
            return _TIER_MESH_DERIVED
        return _TIER_SKELETON  # L2 graph or skeleton npz
    return _TIER_ANNOTATION


def _l2_graph_to_buffers(path: Path, centroid: np.ndarray) -> dict[str, Any]:
    """Load an L2 supervoxel graph and convert to skeleton-like buffers."""
    with np.load(path, allow_pickle=False) as data:
        nodes = np.asarray(data["graph_nodes"], dtype=np.float32) - centroid
        edges = np.asarray(data["graph_edges"], dtype=np.int32)
        radii = np.zeros(len(nodes), dtype=np.float32)

        if "max_dt_nm" in data:
            n_l2 = len(data["max_dt_nm"])
            l2_radii = np.asarray(data["max_dt_nm"], dtype=np.float32)
            orig_edges = np.asarray(data["edges"], dtype=np.int32)
            n_edges = len(orig_edges)
            radii[:n_l2] = l2_radii
            # boundary_src [N..N+M-1] inherits from src L2 node
            radii[n_l2 : n_l2 + n_edges] = l2_radii[orig_edges[:, 0]]
            # boundary_dst [N+M..N+2M-1] inherits from dst L2 node
            radii[n_l2 + n_edges :] = l2_radii[orig_edges[:, 1]]

    positions = np.empty((len(edges) * 2, 3), dtype=np.float32)
    positions[0::2] = nodes[edges[:, 0]]
    positions[1::2] = nodes[edges[:, 1]]
    edge_radii = (radii[edges[:, 0]] + radii[edges[:, 1]]) / 2.0

    return {
        "nodes": nodes.ravel().tolist(),
        "edges": edges.ravel().tolist(),
        "positions": positions.ravel().tolist(),
        "radii": radii.tolist(),
        "edgeRadii": edge_radii.tolist(),
        "nNodes": len(nodes),
        "nEdges": len(edges),
    }


class _LogTee(io.TextIOBase):
    """Wraps stdout: writes to the original AND broadcasts lines via WS.

    Both timing helpers (``pre._timed``, ``skeletonize._timed``) print a
    stage label with ``end=""`` and only close the line with the elapsed
    time once the stage returns, so a stage that runs for 30 s is an
    unterminated line for all of it.  Broadcasting only on newline would
    leave the browser silent for exactly the stages worth reporting, so
    an unterminated line is sent as soon as it is written — that is the
    progress indication — and the newline that later terminates it is
    not re-sent.
    """

    def __init__(self, original, loop, broadcast_fn):
        self._original = original
        self._loop = loop
        self._broadcast = broadcast_fn
        self._partial = ""  # written, no newline yet
        self._sent = ""  # last text broadcast, so it is not repeated

    def _emit(self, text):
        text = text.rstrip()
        if not text or text == self._sent:
            return
        self._sent = text
        asyncio.run_coroutine_threadsafe(
            self._broadcast({"type": "log", "text": text}),
            self._loop,
        )

    def write(self, s):
        self._original.write(s)
        self._original.flush()
        *complete, self._partial = (self._partial + s).split("\n")
        for line in complete:
            self._emit(line)
        if self._partial:
            self._emit(self._partial)
        return len(s)

    def finish(self):
        """Close out a line the writer never terminated."""
        self._emit(self._partial)
        self._partial = ""

    def flush(self):
        self._original.flush()


def _get_viewer_html() -> str:
    html_path = Path(__file__).parent / "viewer.html"
    return html_path.read_text(encoding="utf-8")


#: How much of a claimed component there must be for it to be a neurite.
#: ``break_up_mesh`` honours every claim by default, which is right when a
#: caller named a component it had in hand.  Here the claims are drawn with
#: a lasso, which grazes slivers it did not mean and strands isolated
#: triangles at the boundary it moves, so the viewer sets a floor.  Under it
#: a claimed piece still leaves the soma — it lands in Discarded, visible,
#: rather than becoming a two-face neurite.
_CLAIM_MIN_FACES = 16


# ── Server ────────────────────────────────────────────────────────────


def _create_app(
    mesh_path: str | Path | None = None,
    port: int = 8777,
    *,
    preload_mesh: trimesh.Trimesh | None = None,
    preload_centroid: np.ndarray | None = None,
    extra_meshes: dict[str, dict[str, Any]] | None = None,
    contact_state: dict[str, Any] | None = None,
    mesh_color: list[float] | None = None,
    preload_skeletons: list[tuple[str, Any]] | None = None,
):
    """Create the Starlette app."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket

    # ── Shared mutable state ──────────────────────────────────────────
    # These are modified by upload/remove endpoints
    mesh_state: dict[str, Any] = {
        "mesh": None,  # trimesh object
        "buffers": None,  # JSON-serialisable mesh data
        "path": None,  # source file path
        "centroid": np.zeros(3, dtype=np.float32),
        "offsets": None,  # cached from detect_offsets
        "mesh_stats": None,  # cached from compute_mesh_stats
        "organelles": None,  # dict(pocket, isolated, expanded) bool masks, or None
        "fusion_clusters": None,  # cached from detect_fusions
        "soma": None,  # cached from detect_soma
        "disconnected": None,  # cached from detect_disconnected
        "gap_clusters": None,  # cached from detect_gaps
        "hole_loops": None,  # cached from detect_holes
        "pending_reassignment": None,  # previewed, not yet committed
        # Faces the user rescued from the discard threshold.  Input
        # state, like soma and organelles: the neurite/discarded split
        # is re-derived on every break and reassignment, so an override
        # that lived on the result would be undone by the next one.
        "rescued": np.empty(0, dtype=np.int64),
        # The same claim drawn with a lasso.  Kept apart from `rescued`
        # because the two differ in how well the user could aim: a
        # double-clicked component is exact, a lasso grazes slivers, so
        # only the second wants a size floor.
        "released": np.empty(0, dtype=np.int64),
        # Input state for the same reason, and held here rather than
        # taken per request so that a re-derive nobody passed it to still
        # uses the floor the current components were produced with.
        "claim_min_faces": _CLAIM_MIN_FACES,
    }

    def _claim_floor(body) -> int:
        """The claim size floor, updated if the request carried one."""
        raw = (body or {}).get("claimMinFaces")
        if raw is not None:
            try:
                mesh_state["claim_min_faces"] = max(0, int(raw))
            except (TypeError, ValueError):
                pass
        return mesh_state["claim_min_faces"]

    async def _body_of(request) -> dict:
        """Parse a JSON body, tolerating routes that are posted without one."""
        try:
            return await request.json() or {}
        except Exception:
            return {}

    def _organelle_mask(org) -> np.ndarray | None:
        """Combined bool mask from Organelles object."""
        if org is None:
            return None
        return org.mask

    # Multiple skeletons, keyed by filename
    skeleton_states: dict[str, dict[str, Any]] = {}
    SKEL_COLORS = [
        [1.0, 0.4, 0.1],  # orange
        [0.2, 0.6, 1.0],  # blue
        [0.1, 0.9, 0.4],  # green
        [0.9, 0.2, 0.8],  # magenta
        [1.0, 0.9, 0.1],  # yellow
    ]

    # Extra meshes (keyed by name), each: {mesh, buffers, color, opacity}
    extra_mesh_states: dict[str, dict[str, Any]] = {}
    if extra_meshes is not None:
        extra_mesh_states.update(extra_meshes)

    # Contact-site overlay data (set by view_contacts)
    _contact_state: dict[str, Any] | None = contact_state

    # Pre-load mesh from path or object
    if preload_mesh is not None:
        mesh = preload_mesh
        print(
            f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces"
        )
        buffers = _mesh_to_buffers(mesh, centroid=preload_centroid)
        mesh_state["mesh"] = mesh
        mesh_state["buffers"] = buffers
        mesh_state["path"] = "(preloaded)"
        mesh_state["centroid"] = np.asarray(buffers["centroid"], dtype=np.float32)
    elif mesh_path is not None:
        mesh_path = Path(mesh_path)
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        print(
            f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces"
        )

        buffers = _mesh_to_buffers(mesh)

        mesh_state["mesh"] = mesh
        mesh_state["buffers"] = buffers
        mesh_state["path"] = str(mesh_path.resolve())
        mesh_state["centroid"] = np.asarray(buffers["centroid"], dtype=np.float32)

    # Pre-load skeletons if given
    if preload_skeletons:
        centroid = mesh_state["centroid"]
        for name, skel in preload_skeletons:
            color = SKEL_COLORS[len(skeleton_states) % len(SKEL_COLORS)]
            buffers = _skeleton_to_buffers(skel, centroid)
            buffers["color"] = color
            skeleton_states[name] = {
                "skeleton": skel,
                "buffers": buffers,
                "path": name,
                "color": color,
                "l2_graph": False,
                # the mesh this skeleton's vertex maps index; see _skel_is_stale
                "mesh": mesh_state["mesh"],
            }
            print(
                f"Loaded skeleton: {name} ({len(skel.nodes):,} nodes, {len(skel.edges):,} edges)"
            )

    # ── State files ───────────────────────────────────────────────────
    _ensure_state_dir()
    port_dir = _STATE_DIR / str(port)
    port_dir.mkdir(parents=True, exist_ok=True)
    state_path = port_dir / "state.json"
    annotations_path = port_dir / "annotations.json"
    camera_cmd_path = port_dir / "camera.json"

    # Wipe leftover state from previous sessions
    for f in port_dir.iterdir():
        if f.is_file():
            f.unlink()

    if not annotations_path.exists():
        annotations_path.write_text("{}", encoding="utf-8")
    if not camera_cmd_path.exists():
        camera_cmd_path.write_text("{}", encoding="utf-8")

    # Write initial state
    initial = {}
    if mesh_state["mesh"] is not None:
        initial["mesh"] = {
            "path": mesh_state["path"],
            "nVertices": len(mesh_state["mesh"].vertices),
            "nFaces": len(mesh_state["mesh"].faces),
        }
    state_path.write_text(json.dumps(initial, indent=2), encoding="utf-8")

    connected_clients: list[WebSocket] = []
    _undo_stack: list[trimesh.Trimesh] = []
    _UNDO_LIMIT = 10

    # ── Broadcast helper ──────────────────────────────────────────────
    def _current_track() -> str:
        """Which track ``/skeletonize`` would take if pressed right now.

        Asked in one place so that the label the user reads and the branch
        that actually runs cannot disagree — ``run_skeletonize`` reads this
        too rather than repeating the test.
        """
        return "preprocessing" if mesh_state.get("neurites") is not None else "direct"

    async def broadcast(msg: dict):
        # Stamped on the way out rather than announced by whoever changed the
        # components: six places publish them and three drop them, and the
        # track is derived from state anyway.  A message that carries it
        # cannot be the one site that forgot to send it.
        data = json.dumps({**msg, "track": _current_track()})
        for ws in connected_clients:
            try:
                await ws.send_text(data)
            except Exception:
                pass

    # ── Log-capturing executor helper ─────────────────────────────────

    async def _log(text: str):
        """Print to terminal AND broadcast to browser."""
        print(text)
        await broadcast({"type": "log", "text": text, "partial": False})

    async def _run_with_log(func, *args, **kwargs):
        """Run *func* in executor, streaming its stdout to WS clients."""
        loop = asyncio.get_event_loop()

        def _wrapper():
            old = sys.stdout
            tee = _LogTee(old, loop, broadcast)
            sys.stdout = tee
            try:
                return func(*args, **kwargs)
            finally:
                sys.stdout = old
                tee.finish()

        result = await loop.run_in_executor(None, _wrapper)
        await broadcast({"type": "log_end"})
        return result

    # ── Routes ────────────────────────────────────────────────────────

    async def index(_request):
        return HTMLResponse(_get_viewer_html())

    async def get_mesh(_request):
        if mesh_state["buffers"] is None:
            return JSONResponse(None)
        buf = mesh_state["buffers"]
        if mesh_color is not None:
            buf = {**buf, "color": mesh_color}
        return JSONResponse(buf)

    async def get_skeletons(_request):
        """Return all loaded skeletons."""
        result = {}
        for name, state in skeleton_states.items():
            result[name] = state["buffers"]
        return JSONResponse(result if result else None)

    async def get_extra_meshes(_request):
        """Return all extra meshes (for multi-mesh / contact mode)."""
        if not extra_mesh_states:
            return JSONResponse(None)
        result = {}
        for name, state in extra_mesh_states.items():
            result[name] = {
                **state["buffers"],
                "color": state.get("color", [0.55, 0.55, 0.6]),
                "opacity": state.get("opacity", 1.0),
            }
        return JSONResponse(result)

    async def get_contact_sites(_request):
        """Return contact-site overlay data."""
        if _contact_state is None:
            return JSONResponse(None)
        return JSONResponse(_contact_state)

    async def get_state(_request):
        if state_path.exists():
            return JSONResponse(json.loads(state_path.read_text(encoding="utf-8")))
        return JSONResponse({})

    async def get_annotations(_request):
        if annotations_path.exists():
            return JSONResponse(
                json.loads(annotations_path.read_text(encoding="utf-8"))
            )
        return JSONResponse({})

    async def get_save_availability(_request):
        """Return which data is available for saving."""
        return JSONResponse(
            {
                "mesh": mesh_state["mesh"] is not None,
                "skeleton": len(skeleton_states) > 0,
                "soma": mesh_state.get("soma") is not None,
                "organelles": mesh_state.get("organelles") is not None,
                "mesh_stats": mesh_state.get("mesh_stats") is not None,
                "neurites": mesh_state.get("neurites") is not None
                and len(mesh_state["neurites"]) > 0,
                "discarded": mesh_state.get("discarded") is not None
                and len(mesh_state["discarded"]) > 0,
                "components": mesh_state.get("neurites") is not None,
                "annotations": annotations_path.exists(),
            }
        )

    async def get_loaded(_request):
        """Return what's currently loaded."""
        # The page asks once on load; after that the track rides on every
        # broadcast, so this is the starting value rather than the channel.
        result = {"mesh": None, "skeletons": {}, "track": _current_track()}
        if mesh_state["path"]:
            result["mesh"] = {
                "path": mesh_state["path"],
                "nVertices": len(mesh_state["mesh"].vertices),
                "nFaces": len(mesh_state["mesh"].faces),
            }
        for name, state in skeleton_states.items():
            result["skeletons"][name] = {
                "path": state["path"],
                "nNodes": state["buffers"]["nNodes"],
                "nEdges": state["buffers"]["nEdges"],
                "color": state["color"],
            }
        return JSONResponse(result)

    async def _reset_for_new_mesh(*, keep: set[Path]) -> None:
        """Clear every piece of state keyed to the mesh being replaced.

        A different mesh invalidates anything naming its faces or vertices,
        so the session is cleared rather than left describing a mesh that is
        gone.  ``keep`` names the files that arrived in the *same gesture* as
        the new mesh; without it the sweep below deletes the very companions
        the user dropped alongside it — see :func:`upload_batch`.
        """
        for old_file in port_dir.iterdir():
            if old_file in keep:
                continue
            if old_file.suffix in (".obj", ".ply", ".stl", ".npz", ".swc"):
                old_file.unlink(missing_ok=True)
        skeleton_states.clear()
        mesh_state["soma"] = None
        mesh_state["organelles"] = None
        mesh_state["mesh_stats"] = None
        # Set by _publish_components and read by /skeletonize to choose the
        # track.  Left behind, components belonging to the mesh being
        # replaced silently send the next skeletonization down the
        # preprocessing path naming faces that no longer exist.
        mesh_state["neurites"] = None
        mesh_state["discarded"] = None
        mesh_state["rescued"] = np.empty(0, dtype=np.int64)
        mesh_state["released"] = np.empty(0, dtype=np.int64)
        annotations_path.write_text("{}", encoding="utf-8")
        await broadcast({"type": "all_skeletons_removed"})

    async def _apply_upload(tmp_path: Path, *, reset: bool = True):
        """Apply one upload that has already been written to ``port_dir``.

        Shared by ``/upload`` and ``/upload_batch``.  ``reset`` is False for
        a batch member: a batch clears the session once, up front, so that a
        mesh landing partway through does not wipe the files that arrived
        with it.
        """
        filename = tmp_path.name
        suffix = tmp_path.suffix.lower()

        try:
            if suffix in (".obj", ".ply", ".stl"):
                mesh = trimesh.load_mesh(str(tmp_path), process=False)
                print(
                    f"Uploaded mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces"
                )

                if reset:
                    await _reset_for_new_mesh(keep={tmp_path})

                buffers = _mesh_to_buffers(mesh)

                mesh_state["mesh"] = mesh
                mesh_state["buffers"] = buffers
                mesh_state["path"] = str(tmp_path.resolve())
                mesh_state["centroid"] = np.asarray(
                    buffers["centroid"], dtype=np.float32
                )

                # Update state file
                current = {}
                if state_path.exists():
                    current = json.loads(state_path.read_text(encoding="utf-8"))
                current["mesh"] = {
                    "path": mesh_state["path"],
                    "nVertices": len(mesh.vertices),
                    "nFaces": len(mesh.faces),
                }
                state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

                await broadcast({"type": "mesh_loaded", "payload": buffers})
                # Re-send all skeletons (centroid changed)
                await _rebroadcast_skeletons()

                return JSONResponse({"ok": True, "type": "mesh", "name": filename})

            elif suffix == ".npz" and _is_organelle_data(tmp_path):
                from skeliner.io import load_organelles_npz

                org = load_organelles_npz(tmp_path)
                mesh_state["organelles"] = org
                loaded = []

                loaded.append(f"pocket={int(org.pocket.sum()):,}")
                loaded.append(f"isolated={int(org.isolated.sum()):,}")
                if org.expanded.any():
                    loaded.append(f"expanded={int(org.expanded.sum()):,}")

                # Visualize
                ann = {}
                if annotations_path.exists():
                    ann = json.loads(annotations_path.read_text(encoding="utf-8"))
                if "highlights" not in ann:
                    ann["highlights"] = []
                if org.pocket.any():
                    ann["highlights"].append(
                        {
                            "faces": np.where(org.pocket)[0].tolist(),
                            "color": [1, 0.15, 0.15],
                            "label": "organelle:pocket",
                        }
                    )
                if org.isolated.any():
                    ann["highlights"].append(
                        {
                            "faces": np.where(org.isolated)[0].tolist(),
                            "color": [0.15, 0.8, 0.15],
                            "label": "organelle:isolated",
                        }
                    )
                if org.expanded.any():
                    ann["highlights"].append(
                        {
                            "faces": np.where(org.expanded)[0].tolist(),
                            "color": [1, 0.6, 0.15],
                            "label": "organelle:expanded",
                        }
                    )

                annotations_path.write_text(json.dumps(ann), encoding="utf-8")
                print(f"Loaded organelle npz: {', '.join(loaded)}")
                await broadcast({"type": "annotations_updated"})
                return JSONResponse(
                    {
                        "ok": True,
                        "type": "organelles",
                        "name": filename,
                        "loaded": loaded,
                    }
                )

            elif suffix == ".npz" and _is_mesh_stats_data(tmp_path):
                from skeliner.io import load_mesh_stats_npz

                cur_mesh = mesh_state.get("mesh")
                try:
                    ms = load_mesh_stats_npz(tmp_path, mesh=cur_mesh)
                except ValueError as exc:
                    return JSONResponse(
                        {"ok": False, "error": str(exc)}, status_code=400
                    )
                mesh_state["mesh_stats"] = ms
                n_faces = len(ms.outward_dots) if ms.outward_dots is not None else 0
                print(f"Loaded mesh_stats: {n_faces:,} faces")
                return JSONResponse(
                    {
                        "ok": True,
                        "type": "mesh_stats",
                        "name": filename,
                        "nFaces": n_faces,
                    }
                )

            elif suffix == ".npz" and _is_soma_data(tmp_path):
                from skeliner.io import load_soma_npz

                soma = load_soma_npz(tmp_path)
                mesh_state["soma"] = soma
                n_verts = len(soma.verts) if soma.verts is not None else 0
                print(
                    f"Loaded soma: center=[{soma.center[0]:.0f}, "
                    f"{soma.center[1]:.0f}, {soma.center[2]:.0f}], "
                    f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
                    f"{soma.axes[2]:.0f}], {n_verts:,} verts"
                )

                # Annotate soma (faces + ellipsoid wireframe)
                mesh = mesh_state.get("mesh")
                if mesh is not None and soma.verts is not None:
                    soma_vset = set(int(v) for v in soma.verts)
                    soma_faces = [
                        int(fi)
                        for fi in range(len(mesh.faces))
                        if sum(1 for v in mesh.faces[fi] if int(v) in soma_vset) >= 2
                    ]
                    ann = {}
                    if annotations_path.exists():
                        ann = json.loads(annotations_path.read_text(encoding="utf-8"))
                    if "highlights" not in ann:
                        ann["highlights"] = []
                    ann["highlights"].append(
                        {
                            "faces": soma_faces,
                            "color": [0.9, 0.5, 0.9],
                            "label": "soma",
                        }
                    )
                    centroid = mesh_state["centroid"]
                    if "ellipsoids" not in ann:
                        ann["ellipsoids"] = []
                    ann["ellipsoids"].append(
                        {
                            "center": (soma.center - centroid).tolist(),
                            "axes": soma.axes.tolist(),
                            "R": soma.R.tolist(),
                            "color": [0.9, 0.5, 0.9],
                        }
                    )
                    annotations_path.write_text(json.dumps(ann), encoding="utf-8")
                    await broadcast({"type": "annotations_updated"})

                return JSONResponse(
                    {
                        "ok": True,
                        "type": "soma",
                        "name": filename,
                    }
                )

            elif suffix == ".npz" and _is_components_data(tmp_path):
                from skeliner.dataclass import MeshComponents
                from skeliner.io import load_components_npz

                comp = load_components_npz(tmp_path)

                # The file's split is already whatever its author decided;
                # an override left over from another mesh would name
                # unrelated faces.
                mesh_state["rescued"] = np.empty(0, dtype=np.int64)
                mesh_state["released"] = np.empty(0, dtype=np.int64)

                # A components file with no soma does not clear one that is
                # already loaded — it says nothing about the soma.
                if comp.soma is None and mesh_state.get("soma") is not None:
                    comp = MeshComponents(
                        soma=mesh_state["soma"],
                        organelles=comp.organelles,
                        neurites=comp.neurites,
                        discarded=comp.discarded,
                    )

                org_mask = comp.organelles.mask
                if mesh_state.get("mesh") is None:
                    # Nothing to draw them against; keep the state so a mesh
                    # dropped afterwards has components to publish.
                    mesh_state["soma"] = comp.soma
                    mesh_state["organelles"] = comp.organelles
                    mesh_state["neurites"] = comp.neurites
                    mesh_state["discarded"] = comp.discarded
                else:
                    # One implementation of "draw these components", shared
                    # with break_up_mesh and every reassignment — a second
                    # copy here is how a loaded file came back showing
                    # "neurite 0" over components that had been named.
                    _publish_components(comp)

                loaded = []
                if comp.soma is not None:
                    loaded.append("soma")
                loaded.append(f"organelles={int(org_mask.sum()):,}")
                loaded.append(f"{len(comp.neurites)} neurites")
                loaded.append(f"{len(comp.discarded)} discarded")
                print(f"Loaded components: {', '.join(loaded)}")
                await broadcast({"type": "annotations_updated"})
                return JSONResponse(
                    {
                        "ok": True,
                        "type": "components",
                        "name": filename,
                        "loaded": loaded,
                    }
                )

            elif suffix == ".npz" and _is_neurites_data(tmp_path):
                from skeliner.io import load_neurites_npz

                neurites = load_neurites_npz(tmp_path)

                ann = {}
                if annotations_path.exists():
                    ann = json.loads(annotations_path.read_text(encoding="utf-8"))
                if "highlights" not in ann:
                    ann["highlights"] = []

                neurite_colors = [
                    [0.2, 0.6, 1.0],
                    [0.3, 1.0, 0.3],
                    [1.0, 0.4, 0.1],
                    [0.0, 0.9, 0.9],
                    [1.0, 0.2, 0.6],
                ]
                # A bare neurites file is not a full set of components, so
                # this cannot go through _publish_components — but it does
                # carry names, and drawing them as "neurite {i}" would throw
                # away what the file says.
                labels = neurites.labels
                for i, nf in enumerate(neurites):
                    c = neurite_colors[i % len(neurite_colors)]
                    name = labels[i] if labels is not None else f"neurite {i}"
                    ann["highlights"].append(
                        {
                            "faces": nf.tolist(),
                            "color": c,
                            "label": f"{name} ({len(nf):,}f)",
                            "neurite": i,
                        }
                    )

                annotations_path.write_text(json.dumps(ann), encoding="utf-8")
                print(f"Loaded neurites: {len(neurites)} components")
                await broadcast({"type": "annotations_updated"})
                return JSONResponse({"ok": True, "type": "neurites", "name": filename})

            elif suffix == ".npz" and _is_l2_graph(tmp_path):
                buffers = _l2_graph_to_buffers(tmp_path, mesh_state["centroid"])
                print(
                    f"Uploaded L2 graph: {buffers['nNodes']:,} nodes, {buffers['nEdges']:,} edges"
                )

                color = SKEL_COLORS[len(skeleton_states) % len(SKEL_COLORS)]
                buffers["color"] = color

                skeleton_states[filename] = {
                    "skeleton": None,
                    "path": str(tmp_path.resolve()),
                    "buffers": buffers,
                    "color": color,
                    "l2_graph": True,
                }

                await broadcast(
                    {
                        "type": "skeleton_loaded",
                        "payload": {"name": filename, **buffers},
                    }
                )
                return JSONResponse({"ok": True, "type": "skeleton", "name": filename})

            elif suffix in (".swc", ".npz"):
                skel = _load_skeleton_as_nm(tmp_path)
                print(
                    f"Uploaded skeleton: {len(skel.nodes):,} nodes, {len(skel.edges):,} edges"
                )
                # SWC carries no vertex map, so `node2verts` is None and every
                # edit is refused by _skel_edit_target with an error naming a
                # cause but not a cure.  Say it here, where the fix — export
                # npz instead — still applies.
                if skel.node2verts is None:
                    print(
                        f"  {filename} has no mesh data; it can be viewed but "
                        "not edited. Save as .npz to keep the vertex maps."
                    )

                color = SKEL_COLORS[len(skeleton_states) % len(SKEL_COLORS)]
                buffers = _skeleton_to_buffers(skel, mesh_state["centroid"])
                buffers["color"] = color

                skeleton_states[filename] = {
                    "skeleton": skel,
                    "path": str(tmp_path.resolve()),
                    "buffers": buffers,
                    "color": color,
                    # An uploaded skeleton was built from some mesh offline;
                    # loading it alongside this one is the claim that they go
                    # together, so bind it to whatever is loaded now — then
                    # test the claim, because binding does not establish it.
                    "mesh": mesh_state["mesh"],
                }

                # Said now rather than on the first click.  Every edit and
                # every diagnostic reads the mesh through this pairing, so a
                # wrong one is not a failed operation but a whole session's
                # worth of confident wrong answers.
                pairing = _skel_pairing(skeleton_states[filename])
                buffers["pairing"] = pairing
                if not pairing["ok"]:
                    print(
                        f"  {filename} does NOT match the loaded mesh: {pairing['reason']}."
                    )
                    print(
                        "  Editing and bin queries are refused until the right mesh is loaded."
                    )
                elif not pairing["verified"] and skel.node2verts is not None:
                    print(
                        f"  {filename} carries no mesh counts (written before "
                        "skeliner recorded them), so the pairing cannot be confirmed."
                    )

                await broadcast(
                    {
                        "type": "skeleton_loaded",
                        "payload": {"name": filename, **buffers},
                    }
                )
                return JSONResponse({"ok": True, "type": "skeleton", "name": filename})

            elif suffix == ".json":
                # Annotation JSON — merge with deduplication
                incoming = json.loads(tmp_path.read_text(encoding="utf-8"))
                current = {}
                if annotations_path.exists():
                    current = json.loads(annotations_path.read_text(encoding="utf-8"))
                if "highlights" not in current:
                    current["highlights"] = []

                # Deduplicate: skip incoming highlights whose label
                # already exists with the same face count
                existing = {
                    (h.get("label", ""), len(h.get("faces", []))): True
                    for h in current["highlights"]
                }
                added = 0
                for h in incoming.get("highlights", []):
                    key = (h.get("label", ""), len(h.get("faces", [])))
                    if key not in existing:
                        current["highlights"].append(h)
                        existing[key] = True
                        added += 1

                # Copy non-highlight keys (skip duplicates)
                for k, v in incoming.items():
                    if k != "highlights" and k not in current:
                        current[k] = v

                annotations_path.write_text(json.dumps(current), encoding="utf-8")
                print(
                    f"Loaded annotations: {added} new highlights "
                    f"(total {len(current['highlights'])})"
                )
                await broadcast({"type": "annotations_updated"})
                return JSONResponse(
                    {
                        "ok": True,
                        "type": "annotations",
                        "name": filename,
                        "added": added,
                        "total": len(current["highlights"]),
                    }
                )

            else:
                return JSONResponse(
                    {"ok": False, "error": f"Unsupported format: {suffix}"},
                    status_code=400,
                )

        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    async def upload_file(request):
        """Handle a single file upload."""
        try:
            form = await request.form()
        except Exception:
            # Client disconnected during upload (e.g. drag-drop retry)
            return JSONResponse(
                {"ok": False, "error": "Upload interrupted"}, status_code=499
            )
        # Starlette never closes a parsed form; its uploads are spooled temp
        # files that would otherwise stay open until garbage collection.
        try:
            upload = form["file"]
            tmp_path = port_dir / upload.filename
            tmp_path.write_bytes(await upload.read())
        finally:
            await form.close()
        return await _apply_upload(tmp_path)

    async def upload_batch(request):
        """Apply a whole drag-drop as one gesture.

        The files of one drop are related — a mesh with its components, its
        skeleton, its annotations — but arrive in whatever order the OS
        listed them.  Applied independently, a mesh landing after its
        companions clears them as though they belonged to a previous
        session.  So the batch is ordered here, by type rather than by
        arrival, and the session is cleared once before any of it lands.

        Ordering on the server rather than in the drop handler is deliberate:
        it is a property of what the files *are*, and a second caller that
        forgot to sort would silently get the old behaviour back.
        """
        try:
            form = await request.form()
        except Exception:
            return JSONResponse(
                {"ok": False, "error": "Upload interrupted"}, status_code=499
            )

        # Written before anything is applied because the tier of an .npz is
        # decided by sniffing its contents, which needs it on disk.
        paths: list[Path] = []
        try:
            for upload in form.getlist("files"):
                tmp_path = port_dir / upload.filename
                tmp_path.write_bytes(await upload.read())
                paths.append(tmp_path)
        finally:
            await form.close()
        if not paths:
            return JSONResponse(
                {"ok": False, "error": "No files in batch"}, status_code=400
            )

        meshes = [p for p in paths if _upload_tier(p) == _TIER_MESH]
        if len(meshes) > 1:
            names = ", ".join(p.name for p in meshes)
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"Drop names more than one mesh ({names}). Which "
                    "one the companions belong to is ambiguous — drop one "
                    "mesh with its files.",
                },
                status_code=400,
            )

        # Once, up front, and told to spare this batch's own files.  Doing it
        # here rather than in the mesh branch is what makes the drop one
        # gesture: nothing a batch carries can be cleared by a sibling.
        if meshes:
            await _reset_for_new_mesh(keep=set(paths))

        paths.sort(key=lambda p: (_upload_tier(p), p.name))

        results = []
        for tmp_path in paths:
            resp = await _apply_upload(tmp_path, reset=False)
            results.append(
                {
                    "name": tmp_path.name,
                    "ok": 200 <= resp.status_code < 300,
                    **json.loads(bytes(resp.body).decode("utf-8")),
                }
            )

        return JSONResponse({"ok": True, "results": results})

    async def remove_item(request):
        """Remove mesh or skeleton."""
        body = await request.json()
        item_type = body.get("type")

        if item_type == "mesh":
            mesh_state["mesh"] = None
            mesh_state["buffers"] = None
            mesh_state["path"] = None
            mesh_state["centroid"] = np.zeros(3, dtype=np.float32)
            await _rebroadcast_skeletons()
            await broadcast({"type": "mesh_removed"})
            return JSONResponse({"ok": True})

        elif item_type == "skeleton":
            name = body.get("name")
            if name and name in skeleton_states:
                del skeleton_states[name]
                await broadcast({"type": "skeleton_removed", "payload": {"name": name}})
            else:
                # Remove all skeletons
                skeleton_states.clear()
                await broadcast({"type": "all_skeletons_removed"})
            return JSONResponse({"ok": True})

        return JSONResponse({"ok": False, "error": "Unknown type"}, status_code=400)

    async def update_annotations(request):
        body = await request.json()
        annotations_path.write_text(json.dumps(body), encoding="utf-8")
        return JSONResponse({"ok": True})

    async def post_state(request):
        body = await request.json()
        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current.update(body)
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return JSONResponse({"ok": True})

    async def post_selection(request):
        body = await request.json()
        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current["selection"] = body
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return JSONResponse({"ok": True})

    async def detect_offsets(_request):
        """Run offset detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_offsets

        mesh = mesh_state["mesh"]
        await _log("[skeliner.pre] Detecting offsets...")
        offsets = await _run_with_log(find_offsets, mesh, verbose=True)
        mesh_state["offsets"] = offsets

        if not offsets:
            await _log("[skeliner.pre] No offsets found")
            return JSONResponse({"ok": True, "nOffsets": 0})

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        if "edge_groups" not in ann:
            ann["edge_groups"] = []

        # Clear previous offset annotations
        ann["highlights"] = [
            h for h in ann["highlights"] if not h.get("label", "").startswith("offset ")
        ]
        ann["edge_groups"] = [
            eg
            for eg in ann["edge_groups"]
            if not eg.get("label", "").startswith("offset ")
        ]

        colors = [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3],
            [1.0, 0.3, 1.0],
            [0.3, 1.0, 1.0],
        ]

        for i, o in enumerate(offsets):
            dx, dy = o["offset"]
            color = colors[i % len(colors)]
            ann["highlights"].append(
                {
                    "faces": o["cap_faces"].tolist(),
                    "color": color,
                    "label": (
                        f"offset {i}: z={o['z_floor']:.0f}→{o['z_ceil']:.0f} "
                        f"d=({dx:.0f},{dy:.0f}) "
                        f"err={o['match_error']:.0f} "
                        f"{len(o['shifted_verts']):,}v"
                    ),
                }
            )

            # Add a direction vector: red (floor) → green (ceil)
            # Coordinates must be centroid-relative for the viewer
            centroid = mesh_state["centroid"]
            start = (o["floor_center"] - centroid).tolist()
            end = (o["ceil_center"] - centroid).tolist()
            midpt = [(s + e) / 2 for s, e in zip(start, end)]
            ann["edge_groups"].append(
                {
                    "segments": [[start, midpt]],
                    "color": [1.0, 0.2, 0.2],
                    "label": f"offset {i} from",
                }
            )
            ann["edge_groups"].append(
                {
                    "segments": [[midpt, end]],
                    "color": [0.2, 1.0, 0.2],
                    "label": f"offset {i} to",
                }
            )

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nOffsets": len(offsets)})

    async def do_remove_offsets(_request):
        """Remove detected offsets from the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import remove_offsets

        mesh = mesh_state["mesh"]
        cached = mesh_state.get("offsets")
        ms = mesh_state.get("mesh_stats")

        def _do_remove():
            return remove_offsets(mesh, offsets=cached, verbose=True, mesh_stats=ms)

        new_mesh = await _run_with_log(_do_remove)
        await _apply_new_mesh(new_mesh)

        # Clear cached offsets; mesh_stats invalidated in-place by remove_offsets
        mesh_state["offsets"] = None

        return JSONResponse({"ok": True})

    async def save_offsets(_request):
        """Save detected offsets to a file."""
        cached = mesh_state.get("offsets")
        if not cached:
            return JSONResponse(
                {"ok": False, "error": "No offsets detected"}, status_code=400
            )
        import pickle

        save_path = port_dir / "offsets.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(cached, f)
        return JSONResponse({"ok": True, "path": str(save_path), "n": len(cached)})

    async def load_offsets(_request):
        """Load previously saved offsets and apply annotations."""
        import pickle

        save_path = port_dir / "offsets.pkl"
        if not save_path.exists():
            return JSONResponse(
                {"ok": False, "error": "No saved offsets found"}, status_code=404
            )
        with open(save_path, "rb") as f:
            offsets = pickle.load(f)
        mesh_state["offsets"] = offsets

        # Rebuild annotations
        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        if "edge_groups" not in ann:
            ann["edge_groups"] = []
        ann["highlights"] = [
            h for h in ann["highlights"] if not h.get("label", "").startswith("offset ")
        ]
        ann["edge_groups"] = [
            eg
            for eg in ann["edge_groups"]
            if not eg.get("label", "").startswith("offset ")
        ]

        colors = [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3],
            [1.0, 0.3, 1.0],
            [0.3, 1.0, 1.0],
        ]
        centroid = mesh_state["centroid"]
        for i, o in enumerate(offsets):
            dx, dy = o["offset"]
            color = colors[i % len(colors)]
            ann["highlights"].append(
                {
                    "faces": o["cap_faces"].tolist(),
                    "color": color,
                    "label": (
                        f"offset {i}: z={o['z_floor']:.0f}→{o['z_ceil']:.0f} "
                        f"d=({dx:.0f},{dy:.0f}) "
                        f"err={o['match_error']:.0f} "
                        f"{len(o['shifted_verts']):,}v"
                    ),
                }
            )
            start = (o["floor_center"] - centroid).tolist()
            end = (o["ceil_center"] - centroid).tolist()
            midpt = [(s + e) / 2 for s, e in zip(start, end)]
            ann["edge_groups"].append(
                {
                    "segments": [[start, midpt]],
                    "color": [1.0, 0.2, 0.2],
                    "label": f"offset {i} from",
                }
            )
            ann["edge_groups"].append(
                {
                    "segments": [[midpt, end]],
                    "color": [0.2, 1.0, 0.2],
                    "label": f"offset {i} to",
                }
            )

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nOffsets": len(offsets)})

    async def detect_organelles(request):
        """Run organelle detection."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        det_type = request.query_params.get("type", "all")

        highlights = []
        nF = len(mesh.faces)

        from skeliner.dataclass import Organelles
        from skeliner.pre import (
            compute_mesh_stats,
            find_isolated_organelles,
            find_organelles,
            find_pocket_organelles,
        )

        cached = mesh_state.get("organelles")

        # Ensure mesh_stats are available
        ms = mesh_state.get("mesh_stats")
        if ms is None or len(ms.outward_dots) != nF:
            ms = await _run_with_log(compute_mesh_stats, mesh, None, 5.0, True)
            mesh_state["mesh_stats"] = ms

        if det_type in ("pocket", "surface"):
            mask = await _run_with_log(
                find_pocket_organelles,
                mesh,
                verbose=True,
                mesh_stats=ms,
            )
            pocket = mask if cached is None else cached.pocket | mask
            isolated = np.zeros(nF, dtype=bool) if cached is None else cached.isolated
            expanded = np.zeros(nF, dtype=bool) if cached is None else cached.expanded
            faces = [int(fi) for fi in np.where(mask)[0]]
            if faces:
                highlights.append(
                    {
                        "faces": faces,
                        "color": [1, 0.15, 0.15],
                        "label": "organelle:pocket",
                    }
                )
        elif det_type == "isolated":
            mask = await _run_with_log(
                find_isolated_organelles,
                mesh,
                verbose=True,
                mesh_stats=ms,
            )
            pocket = np.zeros(nF, dtype=bool) if cached is None else cached.pocket
            isolated = mask if cached is None else cached.isolated | mask
            expanded = np.zeros(nF, dtype=bool) if cached is None else cached.expanded
            faces = [int(fi) for fi in np.where(mask)[0]]
            if faces:
                highlights.append(
                    {
                        "faces": faces,
                        "color": [0.15, 0.8, 0.15],
                        "label": f"organelle:isolated ({len(faces):,})",
                    }
                )
        else:
            result = await _run_with_log(
                find_organelles,
                mesh,
                verbose=True,
                mesh_stats=ms,
            )
            pocket = result.pocket
            isolated = result.isolated
            expanded = np.zeros(nF, dtype=bool)
            sf = [int(fi) for fi in np.where(pocket)[0]]
            iso = [int(fi) for fi in np.where(isolated)[0]]
            if sf:
                highlights.append(
                    {
                        "faces": sf,
                        "color": [1, 0.15, 0.15],
                        "label": "organelle:pocket",
                    }
                )
            if iso:
                highlights.append(
                    {
                        "faces": iso,
                        "color": [0.15, 0.8, 0.15],
                        "label": "organelle:isolated",
                    }
                )

        mesh_state["organelles"] = Organelles(
            pocket=pocket,
            isolated=isolated,
            expanded=expanded,
            # Detection re-runs against the same mesh, so face ids — and
            # therefore any hand assignment made against them — still hold.
            manual=None if cached is None else cached.manual,
        )

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        ann["highlights"].extend(highlights)
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        n_faces = sum(len(h["faces"]) for h in highlights)
        return JSONResponse({"ok": True, "nFaces": n_faces})

    async def detect_fragments(_request):
        """Run fragment detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_fragments

        mesh = mesh_state["mesh"]
        mask = await _run_with_log(find_fragments, mesh, verbose=True)
        mesh_state["fragment_mask"] = mask

        faces = [int(fi) for fi in np.where(mask)[0]]
        if faces:
            ann = {}
            if annotations_path.exists():
                ann = json.loads(annotations_path.read_text(encoding="utf-8"))
            if "highlights" not in ann:
                ann["highlights"] = []
            ann["highlights"].append(
                {
                    "faces": faces,
                    "color": [0.2, 0.8, 0.8],
                    "label": "fragments",
                }
            )
            annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        return JSONResponse({"ok": True, "nFaces": len(faces)})

    async def detect_disconnected(_request):
        """Run disconnected-component detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_disconnected, find_soma_via_ring_cutoff

        mesh = mesh_state["mesh"]
        soma = mesh_state.get("soma")
        if soma is None:
            await _log("[skeliner.pre] Detecting soma first...")
            soma = await _run_with_log(
                find_soma_via_ring_cutoff,
                mesh,
                organelles=mesh_state.get("organelles"),
                mesh_stats=mesh_state.get("mesh_stats"),
                verbose=True,
            )
            mesh_state["soma"] = soma
        await _log("[skeliner.pre] Detecting disconnected components...")
        cached_fusions = mesh_state.get("fusion_clusters")
        if cached_fusions:
            await _log(
                f"[skeliner.pre]   using {len(cached_fusions)} cached "
                f"fusion clusters as walls"
            )
        components = await _run_with_log(
            find_disconnected,
            mesh,
            verbose=True,
            soma=soma,
            organelles=mesh_state.get("organelles"),
            mesh_stats=mesh_state.get("mesh_stats"),
            fusions=cached_fusions,
        )
        mesh_state["disconnected"] = components

        if not components:
            await _log("[skeliner.pre] No disconnected components found")
            return JSONResponse({"ok": False, "nComponents": 0})

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []

        colors = [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3],
            [1.0, 0.3, 1.0],
            [0.3, 1.0, 1.0],
        ]
        for i, faces in enumerate(components):
            ann["highlights"].append(
                {
                    "faces": faces,
                    "color": colors[i % len(colors)],
                    "label": f"disconnected {i}",
                }
            )

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nComponents": len(components)})

    async def detect_soma(request):
        """Run soma detection and write result to annotations.

        Accepts optional JSON body ``{"method": "z_contour"}`` or
        ``{"method": "ring_cutoff"}``.  Defaults to z_contour.
        """
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        # Parse method choice from request body (POST with JSON)
        method = "z_contour"
        try:
            body = await request.json()
            method = body.get("method", "z_contour")
        except Exception:
            pass

        if method == "ring_cutoff":
            from skeliner.pre import find_soma_via_ring_cutoff as _find

            label_prefix = "soma"
        else:
            from skeliner.pre import find_soma_via_z_contour as _find

            label_prefix = "soma (z_contour)"
        color = [0.9, 0.5, 0.9]

        mesh = mesh_state["mesh"]
        if method == "z_contour":
            soma = await _run_with_log(
                _find,
                mesh,
                verbose=True,
            )
        else:
            soma = await _run_with_log(
                _find,
                mesh,
                organelles=mesh_state.get("organelles"),
                mesh_stats=mesh_state.get("mesh_stats"),
                verbose=True,
            )
        if soma is None:
            return JSONResponse({"ok": False, "error": "No soma found"})

        mesh_state["soma"] = soma

        # Highlight soma surface vertices via their faces
        # Include faces where at least 2 of 3 vertices are soma verts
        soma_vset = set(int(v) for v in soma.verts)
        soma_faces = [
            int(fi)
            for fi in range(len(mesh.faces))
            if sum(1 for v in mesh.faces[fi] if int(v) in soma_vset) >= 2
        ]

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []
        ann["highlights"].append(
            {
                "faces": soma_faces,
                "color": color,
                "label": label_prefix,
            }
        )

        # Wireframe ellipsoid (coordinates shifted by mesh centroid)
        centroid = mesh_state["centroid"]
        if "ellipsoids" not in ann:
            ann["ellipsoids"] = []
        ann["ellipsoids"].append(
            {
                "center": (soma.center - centroid).tolist(),
                "axes": soma.axes.tolist(),
                "R": soma.R.tolist(),
                "color": color,
            }
        )

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        return JSONResponse(
            {
                "ok": True,
                "method": method,
                "nFaces": len(soma_faces),
                "nVerts": len(soma.verts),
            }
        )

    async def detect_gaps(_request):
        """Run gap detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_gaps, find_soma_via_ring_cutoff

        mesh = mesh_state["mesh"]

        # Use cached soma or compute it
        soma = mesh_state.get("soma")
        if soma is None:
            await _log("[skeliner.pre] Detecting soma first...")
            soma = await _run_with_log(
                find_soma_via_ring_cutoff,
                mesh,
                organelles=mesh_state.get("organelles"),
                mesh_stats=mesh_state.get("mesh_stats"),
                verbose=True,
            )
            mesh_state["soma"] = soma

        cached_disc = mesh_state.get("disconnected")
        cached_fusions = mesh_state.get("fusion_clusters")
        if cached_fusions:
            await _log(
                f"[skeliner.pre]   using {len(cached_fusions)} cached "
                f"fusion clusters as walls"
            )
        await _log("[skeliner.pre] Detecting gaps...")
        gaps = await _run_with_log(
            find_gaps,
            mesh,
            verbose=True,
            soma=soma,
            disconnected=cached_disc,
            mesh_stats=mesh_state.get("mesh_stats"),
            fusions=cached_fusions,
        )
        mesh_state["gap_clusters"] = gaps

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []

        colors = [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
            [1.0, 1.0, 0.3],
            [1.0, 0.3, 1.0],
            [0.3, 1.0, 1.0],
        ]
        for i, (faces_a, faces_b, dist, da, db) in enumerate(gaps):
            label_a = "main" if da == -1 else f"disc {da}"
            label_b = "main" if db == -1 else f"disc {db}"
            color = colors[i % len(colors)]
            ann["highlights"].append(
                {
                    "faces": faces_a + faces_b,
                    "color": color,
                    "label": f"gap {i}: {label_a} ↔ {label_b}, {dist:.0f}nm",
                }
            )

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse(
            {
                "ok": True,
                "nGaps": len(gaps),
            }
        )

    async def do_remove_gaps(_request):
        """Bridge all detected gaps."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import remove_gaps

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)

        soma = mesh_state.get("soma")
        cached_gaps = mesh_state.get("gap_clusters")
        # Capture the pre-bridge fusion clusters before _apply_new_mesh
        # nulls the cache below.  remove_gaps preserves face/vertex indices
        # (degenerate-in-place + append), so this list stays valid on the
        # bridged mesh and is re-stored afterward — see the restore below.
        cached_fusions = mesh_state.get("fusion_clusters")

        # Pass mesh_stats so remove_gaps can invalidate/pad it
        ms = mesh_state.get("mesh_stats")
        org = mesh_state.get("organelles")

        new_mesh = await _run_with_log(
            remove_gaps,
            mesh,
            verbose=True,
            soma=soma,
            gaps=cached_gaps,
            mesh_stats=ms,
        )
        n_after = len(new_mesh.faces)
        n_added = n_after - n_before

        # Pad organelle masks for appended stitch faces
        if n_added > 0 and org is not None:
            from skeliner.dataclass import Organelles

            pad = np.zeros(n_added, dtype=bool)
            mesh_state["organelles"] = Organelles(
                pocket=np.concatenate([org.pocket, pad]),
                isolated=np.concatenate([org.isolated, pad]),
                expanded=np.concatenate([org.expanded, pad]),
                manual=np.concatenate([org.manual, pad]),
            )

        # ms was mutated in-place by remove_gaps (topology invalidated,
        # outward_dots padded) — update the reference in mesh_state
        if ms is not None:
            mesh_state["mesh_stats"] = ms

        _clear_annotations("gap ", "disconnected ")
        await _apply_new_mesh(new_mesh)
        mesh_state["gap_clusters"] = None
        # Restore the pre-bridge fusion list (nulled by _apply_new_mesh) so a
        # subsequent remove_fusions uses the same clusters preprocess()
        # threads through, instead of re-detecting on the bridged mesh.
        # Re-detection would flag the bridges' own non-manifold junctions and
        # sever them — diverging from the pipeline.
        mesh_state["fusion_clusters"] = cached_fusions
        await _log(f"Remove gaps: {n_after:,} faces after bridging")

        return JSONResponse(
            {
                "ok": True,
                "nFaces": n_after,
            }
        )

    async def chunk_grid(_request):
        """Compute chunk boundary grid and return as line segments."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        centroid = mesh_state["centroid"]

        def _run():
            from skeliner.pre import find_chunk_boundaries

            boundaries = find_chunk_boundaries(mesh)
            verts = mesh.vertices

            bx = np.sort(boundaries.get(0, np.array([])))
            by = np.sort(boundaries.get(1, np.array([])))
            bz = np.sort(boundaries.get(2, np.array([])))

            # Compute spacings. X/Y share chunk size, Z is different.
            all_spacings = {}
            for ax, bv in [(0, bx), (1, by), (2, bz)]:
                if len(bv) >= 2:
                    all_spacings[ax] = float(np.median(np.diff(bv)))

            def _get_spacing(ax):
                if ax in all_spacings:
                    return all_spacings[ax]
                # X/Y share spacing — borrow from the other
                if ax in (0, 1):
                    partner = 1 - ax
                    if partner in all_spacings:
                        return all_spacings[partner]
                # Fall back to any available spacing
                if all_spacings:
                    return next(iter(all_spacings.values()))
                return None

            def _extend_grid(bvals, vmin, vmax, spacing):
                """Extend boundary values to cover mesh extent."""
                if len(bvals) == 0 or spacing is None:
                    return bvals
                while bvals[0] - spacing > vmin - spacing:
                    bvals = np.concatenate([[bvals[0] - spacing], bvals])
                while bvals[-1] + spacing < vmax + spacing:
                    bvals = np.concatenate([bvals, [bvals[-1] + spacing]])
                return bvals

            x_vals = (
                _extend_grid(bx, verts[:, 0].min(), verts[:, 0].max(), _get_spacing(0))
                - centroid[0]
            )
            y_vals = (
                _extend_grid(by, verts[:, 1].min(), verts[:, 1].max(), _get_spacing(1))
                - centroid[1]
            )
            z_vals = (
                _extend_grid(bz, verts[:, 2].min(), verts[:, 2].max(), _get_spacing(2))
                - centroid[2]
            )

            segs = []
            for y in y_vals:
                for z in z_vals:
                    segs.append(
                        [
                            [float(x_vals[0]), float(y), float(z)],
                            [float(x_vals[-1]), float(y), float(z)],
                        ]
                    )
            for x in x_vals:
                for z in z_vals:
                    segs.append(
                        [
                            [float(x), float(y_vals[0]), float(z)],
                            [float(x), float(y_vals[-1]), float(z)],
                        ]
                    )
            for x in x_vals:
                for y in y_vals:
                    segs.append(
                        [
                            [float(x), float(y), float(z_vals[0])],
                            [float(x), float(y), float(z_vals[-1])],
                        ]
                    )

            n_boundaries = len(bx) + len(by) + len(bz)
            return segs, n_boundaries

        segs, n_boundaries = await asyncio.get_event_loop().run_in_executor(None, _run)
        return JSONResponse(
            {"ok": True, "segments": segs, "n_boundaries": n_boundaries}
        )

    async def detect_parallel_patches(_request):
        """Detect parallel-patch merge artifacts at chunk boundaries."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]

        await _log("[skeliner.pre] Detecting parallel patches...")
        from skeliner.pre import find_parallel_patches

        results = await _run_with_log(find_parallel_patches, mesh, verbose=True)

        if not results:
            await _log("[skeliner.pre] No parallel patches found")
            return JSONResponse({"ok": True, "nPatches": 0})

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []

        # Remove previous patch highlights
        ann["highlights"] = [
            h for h in ann["highlights"] if not h.get("label", "").startswith("patch:")
        ]

        axis_colors = {
            0: {"up": [1.0, 0.3, 0.3], "down": [0.6, 0, 0]},
            1: {"up": [0.3, 1.0, 0.3], "down": [0, 0.6, 0]},
            2: {"up": [0.3, 0.3, 1.0], "down": [0, 0, 0.6]},
        }
        axis_names = ["X", "Y", "Z"]

        total_faces = 0
        for r in results:
            ax = r["axis"]
            colors = axis_colors[ax]
            name = axis_names[ax]
            bval = r["bval"]
            if r["up_faces"]:
                ann["highlights"].append(
                    {
                        "faces": r["up_faces"],
                        "color": colors["up"],
                        "label": f"patch: {name}={bval} + ({len(r['up_faces'])}f)",
                    }
                )
            if r["down_faces"]:
                ann["highlights"].append(
                    {
                        "faces": r["down_faces"],
                        "color": colors["down"],
                        "label": f"patch: {name}={bval} - ({len(r['down_faces'])}f)",
                    }
                )
            total_faces += len(r["faces"])

        mesh_state["parallel_patches"] = results
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        await broadcast({"type": "annotations_updated"})
        await _log(
            f"[skeliner.pre] {len(results)} parallel patches, {total_faces} faces"
        )
        return JSONResponse(
            {
                "ok": True,
                "nPatches": len(results),
                "totalFaces": total_faces,
            }
        )

    async def do_remove_parallel(_request):
        """Remove parallel-patch artifacts."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        from skeliner.pre import remove_parallel_patches

        mesh = mesh_state["mesh"]
        ms = mesh_state.get("mesh_stats")
        patches = mesh_state.get("parallel_patches")
        n_before = len(mesh.faces)
        new_mesh = await _run_with_log(
            remove_parallel_patches,
            mesh,
            patches=patches,
            verbose=True,
            mesh_stats=ms,
        )

        # Clear only annotations whose faces were actually removed
        removed = set(int(i) for i in np.where(np.all(new_mesh.faces == 0, axis=1))[0])
        if annotations_path.exists():
            try:
                ann = json.loads(annotations_path.read_text(encoding="utf-8"))
                if "highlights" in ann:
                    ann["highlights"] = [
                        h
                        for h in ann["highlights"]
                        if not (
                            h.get("label", "").startswith("patch:")
                            and all(f in removed for f in h.get("faces", []))
                        )
                    ]
                    annotations_path.write_text(
                        json.dumps(ann, separators=(",", ":")), encoding="utf-8"
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        await _apply_new_mesh(new_mesh)
        n_degen = len(removed)
        await _log(f"Remove parallel: {n_degen:,} faces degenerated")
        return JSONResponse(
            {
                "ok": True,
                "facesBefore": n_before,
                "facesRemoved": n_degen,
            }
        )

    async def detect_fusions(_request):
        """Run fusion detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_fusions

        mesh = mesh_state["mesh"]
        clusters = await _run_with_log(
            find_fusions,
            mesh,
            verbose=True,
            mesh_stats=mesh_state.get("mesh_stats"),
        )
        mesh_state["fusion_clusters"] = clusters

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "highlights" not in ann:
            ann["highlights"] = []

        colors = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0],
        ]
        for i, cluster in enumerate(clusters):
            ann["highlights"].append(
                {
                    "faces": cluster,
                    "color": colors[i % len(colors)],
                    "label": f"fusion {i}",
                }
            )

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse(
            {
                "ok": True,
                "nClusters": len(clusters),
            }
        )

    async def detect_rims(_request):
        """Run rim detection and write results as edge annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import compute_mesh_stats, find_rims

        mesh = mesh_state["mesh"]
        centroid = mesh_state["centroid"]

        # Reuse cached precomputed data
        ms = mesh_state.get("mesh_stats")
        if ms is None or len(ms.outward_dots) != len(mesh.faces):
            ms = compute_mesh_stats(mesh, None, 5.0, True)
            mesh_state["mesh_stats"] = ms

        def _run():
            rims = find_rims(mesh, verbose=True, mesh_stats=ms)
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            colors = [
                [0.2, 1.0, 0.6],
                [0.1, 0.8, 0.9],
                [0.9, 1.0, 0.2],
                [1.0, 0.5, 0.8],
                [0.5, 1.0, 0.4],
                [0.3, 0.7, 1.0],
            ]
            edge_groups = []
            for i, rim_edges in enumerate(rims):
                color = colors[i % len(colors)]
                segments = []
                for e in rim_edges:
                    a = (verts[e[0]] - centroid).tolist()
                    b = (verts[e[1]] - centroid).tolist()
                    segments.append([a, b])
                edge_groups.append(
                    {
                        "segments": segments,
                        "color": color,
                        "label": f"rim {i}",
                    }
                )
            return edge_groups, len(rims)

        edge_groups, n_rims = await _run_with_log(_run)

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "edge_groups" not in ann:
            ann["edge_groups"] = []
        ann["edge_groups"].extend(edge_groups)
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nFaces": n_rims})

    def _clear_annotations(*prefixes: str) -> None:
        """Remove highlight annotations whose label starts with any prefix."""
        if not annotations_path.exists():
            return
        try:
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return
        if "highlights" not in ann:
            return
        ann["highlights"] = [
            h
            for h in ann["highlights"]
            if not any(h.get("label", "").startswith(p) for p in prefixes)
        ]
        annotations_path.write_text(
            json.dumps(ann, separators=(",", ":")), encoding="utf-8"
        )

    def _skel_is_stale(sstate) -> bool:
        """Was this skeleton built from a mesh that is no longer loaded?

        A ``Skeleton`` holds ``node2verts`` / ``vert2node``, which index *the
        mesh it was built from*.  After a face removal or a ``compact_mesh``
        reindex those maps point at different vertices — silently, with no
        shape mismatch to catch it.  Moving vertices with the transform gizmo
        keeps the indices but invalidates every position and radius, which is
        no better.  So any mesh change invalidates a skeleton, and the honest
        response is to re-skeletonize.

        Derived by identity rather than set by a flag, for the same reason the
        face-owner cache is: there are four places that swap the mesh and a
        fifth will be added.  Identity also gets undo right for free — undoing
        back to the very mesh a skeleton was built from restores the *same
        object*, so the skeleton is correctly current again.

        It errs conservative: ``compact_mesh`` builds a new mesh even when it
        removes nothing, so a no-op compaction still reads as stale.  That is
        the right direction to be wrong in — a spurious stale costs one
        re-skeletonize, a spurious current costs radii measured against
        surface that is no longer there, with nothing downstream able to tell.

        A skeleton uploaded while no mesh was loaded binds to ``None``, which
        is not "current" but "never checked against anything", so it reads as
        stale the moment a mesh exists.  Nothing reaches that state today —
        every path back to a loaded mesh goes through the upload branch,
        whose reset clears ``skeleton_states`` first — but the two are
        independent mechanisms, and only this one is about whether the vertex
        maps mean anything.  A drop applies its own mesh before its skeletons
        (see :func:`upload_batch`), so a skeleton arriving in one gesture
        still binds to the mesh it came with.

        What identity cannot see is a skeleton that *never* belonged to the
        mesh it bound to.  Binding happens on arrival and asks nothing about
        the pair, so dropping a cell's skeleton beside another cell's mesh
        produces a "current" layer whose bins name the wrong surface.  That is
        :func:`_skel_pairing` — a different question, asked separately.
        """
        origin = sstate.get("mesh")
        if origin is None:
            return mesh_state["mesh"] is not None
        return origin is not mesh_state["mesh"]

    def _skel_pairing(sstate) -> dict:
        """Does this skeleton belong to the loaded mesh at all?

        Distinct from :func:`_skel_is_stale`, which asks whether the mesh has
        *changed since* the skeleton was built.  Both can be false while the
        pair is nonsense: a skeleton binds to whatever mesh is loaded when it
        arrives, and nothing on arrival checks that the two go together.

        Cached against the pair it was computed for, as the face-owner map is,
        so a click does not re-walk every bin.
        """
        from skeliner import dx

        mesh, skel = mesh_state["mesh"], sstate.get("skeleton")
        if mesh is None or skel is None or skel.node2verts is None:
            return {"ok": True, "verified": False, "reason": "nothing to check"}
        cached = sstate.get("_pairing")
        if cached is not None and cached[0] is mesh and cached[1] is skel:
            return cached[2]
        rep = dx.check_mesh_pairing(skel, mesh, return_report=True)
        sstate["_pairing"] = (mesh, skel, rep)
        return rep

    async def _rebroadcast_skeletons():
        """Re-send every skeleton layer, e.g. after the centroid moved."""
        for sname, sstate in skeleton_states.items():
            if sstate.get("l2_graph"):
                sstate["buffers"] = _l2_graph_to_buffers(
                    Path(sstate["path"]), mesh_state["centroid"]
                )
            else:
                sstate["buffers"] = _skeleton_to_buffers(
                    sstate["skeleton"], mesh_state["centroid"]
                )
            sstate["buffers"]["color"] = sstate["color"]
            stale = _skel_is_stale(sstate)
            sstate["buffers"]["stale"] = stale
            pairing = _skel_pairing(sstate)
            sstate["buffers"]["pairing"] = pairing
            if not pairing["ok"] and not sstate.get("_pairing_announced"):
                sstate["_pairing_announced"] = True
                await _log(
                    f"[skeliner] '{sname}' does not belong to the loaded mesh: "
                    f"{pairing['reason']}."
                )
            if stale and not sstate.get("_stale_announced"):
                sstate["_stale_announced"] = True
                await _log(
                    f"[skeliner] '{sname}' was built from a different mesh — "
                    "its vertex maps no longer match. Re-skeletonize before "
                    "editing or exporting it."
                )
            await broadcast(
                {
                    "type": "skeleton_loaded",
                    "payload": {"name": sname, **sstate["buffers"]},
                }
            )

    async def _apply_new_mesh(new_mesh, *, rederived: bool = False):
        """Replace the current mesh with a modified one and broadcast.

        ``rederived`` says the caller publishes fresh components straight
        after — it only silences the note below, since the drop itself is
        unconditional and a republish overwrites it either way.
        """
        # Save current mesh for undo
        old = mesh_state["mesh"]
        if old is not None:
            _undo_stack.append(old)
            if len(_undo_stack) > _UNDO_LIMIT:
                _undo_stack.pop(0)

        mesh_state["mesh"] = new_mesh
        # Organelle mask/precompute and soma survive — _rebuild_mesh
        # preserves all face/vertex indices (degenerate faces only).
        mesh_state["fusion_clusters"] = None
        mesh_state["disconnected"] = None
        mesh_state["hole_loops"] = None
        # break_up_mesh runs once the mesh is settled, so a components split
        # and a mesh edit should not coexist: the split describes the surface
        # as it was, and /skeletonize reads it to choose the preprocessing
        # track.  Dropped rather than remapped — re-deriving is the honest
        # response, and it is what the one caller that continues past here
        # does anyway.
        had_components = mesh_state.get("neurites") is not None
        mesh_state["neurites"] = None
        mesh_state["discarded"] = None
        if had_components and not rederived:
            await _log(
                "[skeliner] Mesh changed — components dropped. Press Break "
                "again once the mesh is settled."
            )
        # A preview names faces of the mesh it was computed against.  Face
        # ids do not survive a mesh change, so applying it afterwards would
        # reassign whatever now sits at those indices.
        mesh_state["pending_reassignment"] = None
        # Keep the original centroid so the camera doesn't shift
        original_centroid = mesh_state["centroid"]
        buffers = _mesh_to_buffers(new_mesh, centroid=original_centroid)
        mesh_state["buffers"] = buffers

        # Both vertex and face indices are stable — all annotations
        # survive mesh changes without remapping.

        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})

        # Re-send skeletons (centroid may have changed)
        await _rebroadcast_skeletons()

        # Update state file
        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current["mesh"] = {
            "path": mesh_state["path"],
            "nVertices": len(new_mesh.vertices),
            "nFaces": len(new_mesh.faces),
        }
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    async def do_remove_organelles(_request):
        """Remove organelles from the mesh.

        Reuses the cached mask from detect_organelles if available,
        otherwise runs full detection.
        """
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        cached_mask = _organelle_mask(mesh_state.get("organelles"))

        from skeliner.pre import _rebuild_mesh

        def _do_remove():
            if (
                cached_mask is not None
                and len(cached_mask) == len(mesh.faces)
                and cached_mask.any()
            ):
                print(
                    f"[skeliner.pre] Using cached organelle mask ({int(cached_mask.sum()):,} faces)"
                )
                return _rebuild_mesh(mesh, ~cached_mask)
            else:
                reason = (
                    "no cached mask"
                    if cached_mask is None
                    else f"length mismatch ({len(cached_mask)} vs {len(mesh.faces)})"
                    if len(cached_mask) != len(mesh.faces)
                    else "mask is empty"
                )
                print(
                    f"[skeliner.pre] No cached organelle mask ({reason}), running detection"
                )
                from skeliner.pre import remove_organelles as _remove_organelles

                return _remove_organelles(mesh, verbose=True)

        new_mesh = await _run_with_log(_do_remove)

        _clear_annotations("organelle:")
        await _apply_new_mesh(new_mesh)
        n_degen = int(np.all(new_mesh.faces == 0, axis=1).sum())
        await _log(f"Remove organelles: {n_before:,} faces, {n_degen:,} degenerated")
        return JSONResponse(
            {
                "ok": True,
                "facesBefore": n_before,
                "facesRemoved": n_degen,
            }
        )

    async def do_remove_fusions(_request):
        """Remove fusions from the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        from skeliner.pre import remove_fusions as _remove_fusions

        mesh = mesh_state["mesh"]
        ms = mesh_state.get("mesh_stats")
        n_before = len(mesh.faces)
        cached = mesh_state.get("fusion_clusters")
        new_mesh = await _run_with_log(
            _remove_fusions,
            mesh,
            fusions=cached,
            verbose=True,
            mesh_stats=ms,
        )
        n_after = len(new_mesh.faces)

        _clear_annotations("fusion ")
        await _apply_new_mesh(new_mesh)
        n_degen = int(np.all(new_mesh.faces == 0, axis=1).sum())
        await _log(f"Remove fusions: {n_degen:,} faces degenerated")
        return JSONResponse(
            {
                "ok": True,
                "facesBefore": n_before,
                "facesAfter": n_after,
                "facesRemoved": n_before - n_after,
            }
        )

    async def do_remove_fragments(_request):
        """Remove fragments (islands and fins) from the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        from skeliner.pre import remove_fragments as _remove_fragments

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        cached = mesh_state.get("fragment_mask")

        def _do_remove():
            if cached is not None and len(cached) == len(mesh.faces) and cached.any():
                print(
                    f"[skeliner.pre] Using cached fragment mask ({int(cached.sum()):,} faces)"
                )
                return _remove_fragments(mesh, fragments=cached, verbose=True)
            else:
                return _remove_fragments(mesh, verbose=True)

        new_mesh = await _run_with_log(_do_remove)

        _clear_annotations("fragments ")
        await _apply_new_mesh(new_mesh)
        n_degen = int(np.all(new_mesh.faces == 0, axis=1).sum())
        await _log(f"Remove fragments: {n_degen:,} faces degenerated")
        return JSONResponse(
            {
                "ok": True,
                "facesBefore": n_before,
                "facesRemoved": n_degen,
            }
        )

    async def do_fill_holes(request):
        """Fill holes in the mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        from skeliner.pre import fill_holes

        method = request.query_params.get("method", "advancing_front")
        dome_factor = float(request.query_params.get("dome_factor", "0.5"))

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        cached_holes = mesh_state.get("hole_loops")
        new_mesh = await _run_with_log(
            fill_holes,
            mesh,
            holes=cached_holes,
            method=method,
            dome_factor=dome_factor,
            verbose=True,
        )
        n_after = len(new_mesh.faces)

        _clear_annotations("hole ")
        await _apply_new_mesh(new_mesh)
        await _log(
            f"Fill holes: {n_before:,} → {n_after:,} faces ({n_after - n_before:,} added)"
        )
        return JSONResponse(
            {
                "ok": True,
                "facesBefore": n_before,
                "facesAfter": n_after,
            }
        )

    async def do_merge_selected(request):
        """Merge two components by stitching boundary loops of selected faces."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        body = await request.json()
        face_indices = body.get("faces", [])
        if not face_indices:
            return JSONResponse(
                {"ok": False, "error": "No faces selected"}, status_code=400
            )

        from skeliner.pre import merge_selected_faces

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        new_mesh = await _run_with_log(
            merge_selected_faces, mesh, face_indices, verbose=True
        )
        n_after = len(new_mesh.faces)

        await _apply_new_mesh(new_mesh)
        return JSONResponse(
            {
                "ok": True,
                "facesBefore": n_before,
                "facesAfter": n_after,
                "facesRemoved": len(face_indices),
                "facesStitched": n_after - n_before + len(face_indices),
            }
        )

    async def do_remove_selected(request):
        """Remove selected faces from the mesh, leaving open holes."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        body = await request.json()
        face_indices = body.get("faces", [])
        if not face_indices:
            return JSONResponse(
                {"ok": False, "error": "No faces selected"}, status_code=400
            )

        from skeliner.pre import remove_selected_faces

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        new_mesh = await _run_with_log(
            remove_selected_faces, mesh, face_indices, verbose=True
        )

        await _apply_new_mesh(new_mesh)
        await _log(f"Remove selected: {len(face_indices):,} faces removed")
        return JSONResponse(
            {
                "ok": True,
                "facesBefore": n_before,
                "facesRemoved": len(face_indices),
            }
        )

    async def do_rescue_as_neurite(request):
        """Promote the discarded fragments the selection touches to neurites.

        A relabel, not a re-derive.  The components in hand were already
        computed from this soma and these organelle masks, and a rescue
        changes neither — only which list a component sits in.  Running
        ``break_up_mesh`` again would recompute every component to reach
        the same face sets (235 ms on a 110k-face cell), and would quietly
        replace the components of a loaded ``components.npz`` with
        freshly derived ones.

        The override is recorded so the *next* re-derive — a reassignment
        or a re-break — reapplies it, which is the part a plain list move
        cannot do on its own.
        """
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        body = await request.json()
        sel_faces = set(body.get("faces", []))
        if not sel_faces:
            return JSONResponse(
                {"ok": False, "error": "No faces selected"},
                status_code=400,
            )

        discarded = mesh_state.get("discarded")
        if discarded is None or len(discarded) == 0:
            return JSONResponse(
                {"ok": False, "error": "No discarded components"},
                status_code=400,
            )

        # Touching a fragment anywhere claims all of it: the threshold
        # discards whole components, so a partial rescue has no meaning.
        hit = [i for i, d in enumerate(discarded) if sel_faces.intersection(d.tolist())]
        if not hit:
            return JSONResponse(
                {"ok": False, "error": "No discarded component matches the selection"},
                status_code=400,
            )

        components = _current_components()
        faces = np.concatenate([components.discarded[i] for i in hit])
        components.rescue_discarded(hit)

        mesh_state["rescued"] = np.union1d(
            np.asarray(mesh_state["rescued"], dtype=np.int64),
            faces.astype(np.int64, copy=False),
        )
        # Component ids shift when one leaves the discarded list, so a
        # preview named against the old numbering no longer means anything.
        mesh_state["pending_reassignment"] = None
        out = _publish_components(components)
        await broadcast({"type": "annotations_updated"})

        await _log(f"Rescued {len(hit)} discarded → neurites ({len(faces):,} faces)")
        return JSONResponse(
            {"ok": True, "nRescued": len(hit), "facesRescued": len(faces), **out}
        )

    async def edit_vertices(request):
        """Apply vertex position edits from the transform gizmo."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        body = await request.json()
        face_edits = body.get("faceEdits", [])
        if not face_edits:
            return JSONResponse(
                {"ok": False, "error": "No edits provided"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        verts = np.array(mesh.vertices, dtype=np.float64)  # copy
        faces = np.asarray(mesh.faces, dtype=np.int32)
        centroid = mesh_state["centroid"]

        # Push undo before modifying
        old = mesh_state["mesh"]
        if old is not None:
            _undo_stack.append(old)
            if len(_undo_stack) > _UNDO_LIMIT:
                _undo_stack.pop(0)

        # Apply edits: each faceEdit has {face: int, verts: [[x,y,z]*3]}
        # The frontend positions are centroid-subtracted, so add centroid back
        for edit in face_edits:
            fi = edit["face"]
            new_verts = edit["verts"]
            for vi in range(3):
                vert_idx = faces[fi, vi]
                verts[vert_idx] = [
                    new_verts[vi][0] + centroid[0],
                    new_verts[vi][1] + centroid[1],
                    new_verts[vi][2] + centroid[2],
                ]

        new_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_state["mesh"] = new_mesh

        # Keep the original centroid
        original_centroid = mesh_state["centroid"]
        buffers = _mesh_to_buffers(new_mesh, centroid=original_centroid)
        mesh_state["buffers"] = buffers
        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})

        n_faces_edited = len(face_edits)
        await _log(f"Edit vertices: {n_faces_edited} faces modified")

        return JSONResponse({"ok": True, "facesEdited": n_faces_edited})

    async def undo_mesh(_request):
        """Revert to the previous mesh state."""
        if not _undo_stack:
            return JSONResponse(
                {"ok": False, "error": "Nothing to undo"}, status_code=400
            )
        prev_mesh = _undo_stack.pop()
        # Apply without pushing to undo stack
        mesh_state["mesh"] = prev_mesh
        mesh_state["pending_reassignment"] = None
        buffers = _mesh_to_buffers(prev_mesh)
        mesh_state["buffers"] = buffers
        mesh_state["centroid"] = np.asarray(buffers["centroid"], dtype=np.float32)
        annotations_path.write_text("{}", encoding="utf-8")
        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})

        await _rebroadcast_skeletons()

        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text(encoding="utf-8"))
        current["mesh"] = {
            "path": mesh_state["path"],
            "nVertices": len(prev_mesh.vertices),
            "nFaces": len(prev_mesh.faces),
        }
        state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

        await _log(
            f"Undo: restored mesh with {len(prev_mesh.faces):,} faces "
            f"({len(_undo_stack)} steps remaining)"
        )
        return JSONResponse(
            {
                "ok": True,
                "nFaces": len(prev_mesh.faces),
                "undoRemaining": len(_undo_stack),
            }
        )

    def _publish_components(result):
        """Install a MeshComponents into mesh_state and redraw its annotations.

        Shared by ``break_up_mesh`` and by committing a reassignment, so the
        two cannot drift into showing the same components differently.
        """
        from skeliner.pre import soma_face_mask

        mesh_state["soma"] = result.soma
        mesh_state["organelles"] = result.organelles
        mesh_state["neurites"] = result.neurites
        mesh_state["discarded"] = result.discarded

        centroid = mesh_state["centroid"]
        faces = mesh_state["mesh"].faces
        new_soma = result.soma
        new_org = result.organelles.mask

        soma_mask = soma_face_mask(
            faces, new_soma.verts if new_soma is not None else None
        )

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))

        # Replace all highlights and ellipsoids with the component results
        highlights = []
        if new_soma is not None:
            highlights.append(
                {
                    "faces": np.where(soma_mask)[0].tolist(),
                    "color": [0.9, 0.5, 0.9],
                    "label": "soma",
                }
            )

        org_only = new_org & ~soma_mask
        highlights.append(
            {
                "faces": np.where(org_only)[0].tolist(),
                "color": [1.0, 0.8, 0.0],
                "label": f"organelles ({int(org_only.sum()):,}f)",
            }
        )

        neurite_colors = [
            [0.2, 0.6, 1.0],
            [0.3, 1.0, 0.3],
            [1.0, 0.4, 0.1],
            [0.0, 0.9, 0.9],
            [1.0, 0.2, 0.6],
        ]
        labels = result.neurites.labels
        for i, nf in enumerate(result.neurites):
            c = neurite_colors[i % len(neurite_colors)]
            name = labels[i] if labels is not None else f"neurite {i}"
            highlights.append(
                {
                    "faces": nf.tolist(),
                    "color": c,
                    "label": f"{name} ({len(nf):,}f)",
                    # Which neurite this is, for the page to name it back.
                    # Once renamed the label no longer carries the index,
                    # and position in `highlights` depends on whether there
                    # is a soma — neither is something to parse.
                    "neurite": i,
                }
            )

        for i, df in enumerate(result.discarded):
            highlights.append(
                {
                    "faces": df.tolist(),
                    "color": [0.5, 0.5, 0.5],
                    "label": f"discarded {i} ({len(df):,}f)",
                }
            )

        ann["highlights"] = highlights
        if new_soma is not None:
            ann["ellipsoids"] = [
                {
                    "center": (new_soma.center - centroid).tolist(),
                    "axes": new_soma.axes.tolist(),
                    "R": new_soma.R.tolist(),
                    "color": [0.9, 0.5, 0.9],
                }
            ]
        else:
            ann["ellipsoids"] = []

        annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        return {
            "nNeurites": len(result.neurites),
            "nDiscarded": len(result.discarded),
            "somaVerts": (len(new_soma.verts) if new_soma is not None else 0),
            "orgFaces": int(new_org.sum()),
        }

    async def do_break_up_mesh(request):
        """Break mesh at soma: classify components, expand soma + organelles."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        soma = mesh_state.get("soma")
        org = mesh_state.get("organelles")
        if org is None:
            return JSONResponse(
                {"ok": False, "error": "Run organelle detection first"},
                status_code=400,
            )

        from skeliner.pre import break_up_mesh

        mesh = mesh_state["mesh"]
        floor = _claim_floor(await _body_of(request))

        def _run():
            return break_up_mesh(
                mesh,
                soma,
                org,
                rescued=mesh_state["rescued"],
                released=mesh_state["released"],
                claim_min_faces=floor,
                verbose=True,
            )

        result = await _run_with_log(_run)
        # Breaking up redefines the components a pending reassignment was
        # previewed against, so that preview no longer means anything.
        mesh_state["pending_reassignment"] = None
        return JSONResponse({"ok": True, **_publish_components(result)})

    async def name_neurite(request):
        """Give one neurite a name, and the SWC code it exports as.

        The last step of the workflow, not a part of it: a name is pinned
        to a position in ``neurites``, and every re-derive rebuilds that
        list — re-sorted by size, with pieces split and merged — so
        ``break_up_mesh`` returns neurites unnamed and any break, release
        or Auto run drops the names.  That is the intended behaviour, not
        a limitation to work around: a name carried across a re-derive
        would sit on different surface without saying so.
        """
        neurites = mesh_state.get("neurites")
        if not neurites:
            return JSONResponse(
                {"ok": False, "error": "Run break_up_mesh first"}, status_code=400
            )

        body = await _body_of(request)
        try:
            index = int(body["index"])
        except (KeyError, TypeError, ValueError):
            return JSONResponse(
                {"ok": False, "error": "index must be an integer"}, status_code=400
            )
        label = str(body.get("label", "")).strip()
        if not label:
            return JSONResponse(
                {"ok": False, "error": "A name cannot be empty"}, status_code=400
            )
        if not -len(neurites) <= index < len(neurites):
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"neurite {index} out of range for {len(neurites)}",
                },
                status_code=400,
            )

        raw_type = body.get("swcType")
        try:
            neurites.name(
                index, label, swc_type=None if raw_type is None else int(raw_type)
            )
        except (TypeError, ValueError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

        out = _publish_components(_current_components())
        await _log(
            f"neurite {index} is '{label}' (SWC type {neurites.swc_types[index]})"
        )
        return JSONResponse(
            {
                "ok": True,
                "labels": list(neurites.labels),
                "swcTypes": list(neurites.swc_types),
                **out,
            }
        )

    async def do_preprocess(request):
        """Run the whole pipeline in one call.

        This *is* ``pre.preprocess`` rather than the panel's buttons pressed
        in order by the server, so the viewer and the library cannot drift
        into two pipelines wearing one name.  It differs from pressing the
        buttons in one way worth knowing: the soma comes from
        ``find_soma_via_ring_cutoff``, which is the pipeline's choice, while
        the Soma button defaults to ``z_contour``.

        Left uncompacted.  ``compact_mesh`` reindexes faces, and annotations,
        the rescue list and every loaded skeleton are all stated in face ids;
        ``/compact_mesh`` already remaps them, so compaction stays the
        separate button it is instead of being duplicated here.
        """
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import preprocess

        mesh = mesh_state["mesh"]
        n_before = len(mesh.faces)
        floor = _claim_floor(await _body_of(request))

        # An override the user has already made is input to this run, not a
        # casualty of it — break_up_mesh re-derives the neurite/discarded
        # split from scratch and would otherwise re-discard what they rescued.
        new_mesh, components = await _run_with_log(
            preprocess,
            mesh,
            rescued=mesh_state["rescued"],
            released=mesh_state["released"],
            claim_min_faces=floor,
            verbose=True,
        )

        # rederived: the run ends in break_up_mesh, so the components dropped
        # below are replaced a few lines down rather than left for the user
        # to rebuild.
        await _apply_new_mesh(new_mesh, rederived=True)
        # _apply_new_mesh drops the caches keyed to the mesh it replaced;
        # these two are keyed to it as well, and the run has already consumed
        # and removed what they name.
        mesh_state["mesh_stats"] = None
        mesh_state["gap_clusters"] = None
        out = _publish_components(components)
        await _log(
            f"Preprocess: {n_before:,} → {len(new_mesh.faces):,} faces, "
            f"{out['nNeurites']} neurites, {out['nDiscarded']} discarded"
        )
        return JSONResponse({"ok": True, **out})

    def _current_components():
        """A MeshComponents view of what the viewer currently holds."""
        from skeliner.dataclass import Discarded, MeshComponents, Neurites

        return MeshComponents(
            soma=mesh_state.get("soma"),
            organelles=mesh_state["organelles"],
            neurites=mesh_state.get("neurites") or Neurites([]),
            discarded=mesh_state.get("discarded") or Discarded([]),
        )

    def _released_after(target, sel) -> np.ndarray:
        """The lasso claims a reassignment to *target* leaves behind.

        Sending faces to the arbor is a claim that they are arbor, and
        without recording it the re-derive that the reassignment runs takes
        them straight back: a released stub is still surrounded by soma, so
        the absorption rule re-absorbs it, and a released speck is still
        small, so the threshold re-discards it.  The release only looks like
        it failed — it happened, and was undone in the same call.

        The two other targets are the same statement negated, so they
        withdraw the claim; leaving it in place would have the override
        fight the assignment that was just made.

        Derived here rather than remembered so the preview and the commit
        cannot use different sets — a preview that forecasts one outcome and
        applies another is worse than no preview.
        """
        held = np.asarray(mesh_state["released"], dtype=np.int64)
        sel = np.asarray(sel, dtype=np.int64)
        if target == "remainder":
            return np.union1d(held, sel)
        return np.setdiff1d(held, sel)

    async def reassign_preview(request):
        """Preview handing the selected faces to soma / organelle / neurites."""
        from skeliner.pre import preview_reassignment

        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        if mesh_state.get("organelles") is None:
            return JSONResponse(
                {"ok": False, "error": "Run organelle detection first"},
                status_code=400,
            )
        if not mesh_state.get("neurites"):
            return JSONResponse(
                {"ok": False, "error": "Run break_up_mesh first"},
                status_code=400,
            )

        body = await request.json()
        sel = body.get("faces") or []
        target = body.get("to")
        if not sel:
            return JSONResponse(
                {"ok": False, "error": "No faces selected"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        components = _current_components()
        floor = _claim_floor(body)

        def _run():
            return preview_reassignment(
                mesh,
                components,
                sel,
                to=target,
                rescued=mesh_state["rescued"],
                released=_released_after(target, sel),
                claim_min_faces=floor,
                verbose=True,
            )

        try:
            r = await _run_with_log(_run)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

        mesh_state["pending_reassignment"] = r
        await _log(f"Preview: {r.summary}")
        return JSONResponse(
            {
                "ok": True,
                "target": r.target,
                "summary": r.summary,
                "effects": [list(e) for e in r.effects],
                "nSelected": len(r.selected),
                "leaving": r.leaving.tolist(),
                "entering": r.entering.tolist(),
                "nNeurites": len(r.components.neurites),
                "nDiscarded": len(r.components.discarded),
            }
        )

    async def reassign_apply(_request):
        """Commit the pending reassignment preview."""
        from skeliner.pre import apply_reassignment

        r = mesh_state.get("pending_reassignment")
        if r is None:
            return JSONResponse(
                {"ok": False, "error": "Nothing to apply — preview first"},
                status_code=400,
            )

        components = _current_components()
        apply_reassignment(components, r)
        # The same set the preview ran with, so what was forecast is what
        # is now in force — and what the next re-derive will honour.
        mesh_state["released"] = _released_after(r.target, r.selected)
        if r.target != "remainder":
            # Assigning faces to the soma or an organelle is the opposite
            # statement, so it withdraws a deliberate rescue too.
            mesh_state["rescued"] = np.setdiff1d(
                np.asarray(mesh_state["rescued"], dtype=np.int64),
                np.asarray(r.selected, dtype=np.int64),
            )
        mesh_state["pending_reassignment"] = None
        out = _publish_components(components)
        await _log(f"Applied: {r.summary}")
        return JSONResponse({"ok": True, **out})

    async def reassign_cancel(_request):
        """Drop the pending reassignment preview."""
        mesh_state["pending_reassignment"] = None
        return JSONResponse({"ok": True})

    async def do_compact_mesh(_request):
        """Compact mesh: remove degenerate faces, reindex vertices, remap annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.dataclass import (
            Discarded,
            MeshComponents,
            Neurites,
            Organelles,
        )
        from skeliner.pre import compact_mesh

        mesh = mesh_state["mesh"]
        soma = mesh_state.get("soma")
        org = mesh_state.get("organelles")
        n_faces_before = len(mesh.faces)
        n_verts_before = len(mesh.vertices)
        nF = len(mesh.faces)

        # Build a MeshComponents from current viewer state.  The neurites
        # and discarded must be the real ones: `compact_mesh` remaps every
        # face array it is handed, and the result is written straight back
        # into mesh_state below — so passing empty lists here did not mean
        # "leave them alone", it meant the components were *erased* by a
        # compaction, silently, along with anything they had been named.
        if org is None:
            org = Organelles(
                pocket=np.zeros(nF, dtype=bool),
                isolated=np.zeros(nF, dtype=bool),
                expanded=np.zeros(nF, dtype=bool),
            )
        components = MeshComponents(
            soma=soma,
            organelles=org,
            neurites=mesh_state.get("neurites") or Neurites([]),
            discarded=mesh_state.get("discarded") or Discarded([]),
        )

        def _run():
            return compact_mesh(mesh, components, return_maps=True, verbose=True)

        clean, components, vert_map, face_map = await _run_with_log(_run)

        # Compute new centroid and the delta from old
        old_centroid = mesh_state["centroid"].copy()
        buffers = _mesh_to_buffers(clean)
        new_centroid = np.asarray(buffers["centroid"], dtype=np.float32)
        delta = (old_centroid - new_centroid).tolist()

        # Remap annotations
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))

            # Remap face-indexed highlights
            if "highlights" in ann:
                new_highlights = []
                for h in ann["highlights"]:
                    old_faces = np.array(h.get("faces", []), dtype=np.int64)
                    valid = old_faces < n_faces_before
                    mapped = face_map[old_faces[valid]]
                    new_faces = mapped[mapped >= 0].tolist()
                    if new_faces:
                        h = dict(h)
                        h["faces"] = new_faces
                        new_highlights.append(h)
                ann["highlights"] = new_highlights

            # Shift ellipsoid centers by centroid delta
            for ell in ann.get("ellipsoids", []):
                c = ell["center"]
                ell["center"] = [c[0] + delta[0], c[1] + delta[1], c[2] + delta[2]]

            # Shift edge group segments by centroid delta
            for eg in ann.get("edge_groups", []):
                for seg in eg.get("segments", []):
                    for pt in seg:
                        pt[0] += delta[0]
                        pt[1] += delta[1]
                        pt[2] += delta[2]

            annotations_path.write_text(json.dumps(ann), encoding="utf-8")

        # Update state — compact is destructive, clear all cached face-indexed data
        mesh_state["mesh"] = clean
        mesh_state["soma"] = components.soma
        mesh_state["organelles"] = components.organelles
        mesh_state["neurites"] = components.neurites
        mesh_state["discarded"] = components.discarded
        # Remapped rather than cleared with the derived state below: a
        # claim is something the user decided and cannot be recovered
        # by re-running anything, so it is kept like soma and organelles.
        for key in ("rescued", "released"):
            old = np.asarray(mesh_state[key], dtype=np.int64)
            mapped = face_map[old[old < n_faces_before]]
            mesh_state[key] = mapped[mapped >= 0]
        mesh_state["mesh_stats"] = None
        mesh_state["fusion_clusters"] = None
        mesh_state["disconnected"] = None
        mesh_state["gap_clusters"] = None
        mesh_state["hole_loops"] = None
        # Compacting reindexes every face, so a preview's face ids now name
        # different triangles.  Applying it would reassign the wrong ones.
        mesh_state["pending_reassignment"] = None

        mesh_state["buffers"] = buffers
        mesh_state["centroid"] = new_centroid
        buffers["keepCamera"] = True
        await broadcast({"type": "mesh_loaded", "payload": buffers})
        await broadcast({"type": "annotations_updated"})

        # Re-send skeletons
        await _rebroadcast_skeletons()

        await _log(
            f"Compact: {n_verts_before:,} → {len(clean.vertices):,} verts, "
            f"{n_faces_before:,} → {len(clean.faces):,} faces"
        )
        return JSONResponse(
            {
                "ok": True,
                "vertsBefore": n_verts_before,
                "vertsAfter": len(clean.vertices),
                "facesBefore": n_faces_before,
                "facesAfter": len(clean.faces),
            }
        )

    async def export_mesh(request):
        """Export the current mesh as a downloadable file."""
        import tempfile

        from starlette.responses import Response

        from skeliner.io import save_mesh

        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        mesh = mesh_state["mesh"]
        fmt = request.query_params.get("format", "obj")
        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=f".{fmt}"))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: save_mesh(mesh, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}mesh_cleaned.{fmt}"'
            },
        )

    async def export_organelles(request):
        """Save organelle masks (pocket, isolated, expanded)."""
        import tempfile

        from starlette.responses import Response

        from skeliner.io import save_organelles_npz

        org = mesh_state.get("organelles")
        if org is None:
            return JSONResponse(
                {"ok": False, "error": "No organelle masks"}, status_code=400
            )

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        save_organelles_npz(org, tmp)
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}organelles.npz"'
            },
        )

    async def export_mesh_stats(request):
        """Save the cached MeshStats as a standalone NPZ."""
        import tempfile

        from starlette.responses import Response

        from skeliner.io import save_mesh_stats_npz

        ms = mesh_state.get("mesh_stats")
        if ms is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh_stats cached"}, status_code=400
            )

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        save_mesh_stats_npz(ms, tmp)
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}mesh_stats.npz"'
            },
        )

    async def export_soma(request):
        """Export the cached soma as a downloadable NPZ file."""
        import tempfile

        from starlette.responses import Response

        soma = mesh_state.get("soma")
        if soma is None:
            return JSONResponse(
                {"ok": False, "error": "No soma detected"}, status_code=400
            )

        from skeliner.io import save_soma_npz

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: save_soma_npz(soma, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{prefix}soma.npz"'},
        )

    async def export_components(request):
        """Export MeshComponents (soma + organelles + neurites + discarded)."""
        import tempfile

        from starlette.responses import Response

        from skeliner.dataclass import Discarded, MeshComponents, Neurites
        from skeliner.io import save_components_npz

        soma = mesh_state.get("soma")
        org = mesh_state.get("organelles")
        if org is None:
            return JSONResponse(
                {"ok": False, "error": "No organelle data"}, status_code=400
            )

        # Build MeshComponents from current state
        # Neurites/discarded may not exist if break_up_mesh hasn't run
        components = MeshComponents(
            soma=soma,
            organelles=org,
            neurites=mesh_state.get("neurites", Neurites([])),
            discarded=mesh_state.get("discarded", Discarded([])),
        )

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        save_components_npz(components, tmp)
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}components.npz"'
            },
        )

    async def export_neurites(request):
        """Export neurite face arrays as NPZ."""
        import tempfile

        from starlette.responses import Response

        from skeliner.io import save_neurites_npz

        neurites = mesh_state.get("neurites")
        if neurites is None or len(neurites) == 0:
            return JSONResponse(
                {"ok": False, "error": "No neurites (run Break first)"}, status_code=400
            )

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        save_neurites_npz(neurites, tmp)
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}neurites.npz"'
            },
        )

    async def export_discarded(request):
        """Export discarded fragment face arrays as NPZ."""
        import tempfile

        from starlette.responses import Response

        from skeliner.io import save_discarded_npz

        discarded = mesh_state.get("discarded")
        if discarded is None or len(discarded) == 0:
            return JSONResponse(
                {"ok": False, "error": "No discarded fragments (run Break first)"},
                status_code=400,
            )

        prefix = request.query_params.get("prefix", "")
        tmp = Path(tempfile.mktemp(suffix=".npz"))
        save_discarded_npz(discarded, tmp)
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}discarded.npz"'
            },
        )

    async def export_annotations(request):
        """Export annotations as a downloadable JSON file."""
        from starlette.responses import Response

        if not annotations_path.exists():
            return JSONResponse(
                {"ok": False, "error": "No annotations"}, status_code=400
            )
        content = annotations_path.read_bytes()
        prefix = request.query_params.get("prefix", "")
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}annotations.json"'
            },
        )

    async def export_annotation_submesh(request):
        """Export a single highlight annotation as an OBJ submesh."""
        import tempfile

        from starlette.responses import Response

        from skeliner.io import save_mesh

        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        if not annotations_path.exists():
            return JSONResponse(
                {"ok": False, "error": "No annotations"}, status_code=400
            )
        try:
            idx = int(request.query_params.get("index", ""))
        except ValueError:
            return JSONResponse(
                {"ok": False, "error": "Invalid index"}, status_code=400
            )
        ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        highlights = ann.get("highlights") or []
        if idx < 0 or idx >= len(highlights):
            return JSONResponse(
                {"ok": False, "error": "Index out of range"}, status_code=400
            )
        entry = highlights[idx]
        faces = entry.get("faces") or []
        if not faces:
            return JSONResponse(
                {"ok": False, "error": "Annotation has no faces"},
                status_code=400,
            )

        mesh = mesh_state["mesh"]
        sub = mesh.submesh([faces], append=True)

        fmt = request.query_params.get("format", "obj")
        prefix = request.query_params.get("prefix", "")
        label = entry.get("label") or f"highlight_{idx}"
        safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)

        tmp = Path(tempfile.mktemp(suffix=f".{fmt}"))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: save_mesh(sub, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{prefix}{safe_label}.{fmt}"'
                )
            },
        )

    async def export_skeleton(request):
        """Export a skeleton as a downloadable SWC or NPZ file."""
        import tempfile

        from starlette.responses import Response

        name = request.query_params.get("name")
        fmt = request.query_params.get("format", "swc")
        if not name or name not in skeleton_states:
            return JSONResponse(
                {"ok": False, "error": "No skeleton found"}, status_code=400
            )
        sstate = skeleton_states[name]
        skel = sstate.get("skeleton")
        if skel is None:
            return JSONResponse(
                {"ok": False, "error": "Skeleton has no exportable data"},
                status_code=400,
            )
        if _skel_is_stale(sstate):
            # Exporting is the one irreversible thing here: a skeleton whose
            # node2verts index a mesh nobody has any more carries radii
            # belonging to the wrong surface, and nothing downstream can tell.
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"'{name}' was built from a different mesh — its "
                    "radii belong to surface that is no longer there. "
                    "Re-skeletonize before exporting.",
                },
                status_code=409,
            )

        from skeliner.io import save_skeleton_npz, save_skeleton_swc

        loop = asyncio.get_event_loop()
        tmp = Path(tempfile.mktemp(suffix=f".{fmt}"))
        if fmt == "npz":
            await loop.run_in_executor(None, lambda: save_skeleton_npz(skel, tmp))
        else:
            await loop.run_in_executor(None, lambda: save_skeleton_swc(skel, tmp))
        content = tmp.read_bytes()
        tmp.unlink(missing_ok=True)

        prefix = request.query_params.get("prefix", "")
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{prefix}skeleton.{fmt}"'
            },
        )

    async def detect_holes(_request):
        """Run hole detection and write results to annotations."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.pre import find_holes

        mesh = mesh_state["mesh"]
        centroid = mesh_state["centroid"]

        def _run():
            loops = find_holes(mesh, verbose=True)
            mesh_state["hole_loops"] = loops
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            colors = [
                [0.2, 0.6, 1.0],
                [0.1, 0.9, 0.4],
                [0.9, 0.2, 0.8],
                [1.0, 0.9, 0.1],
                [1.0, 0.4, 0.1],
                [0.4, 0.9, 0.9],
            ]
            edge_groups = []
            for i, loop in enumerate(loops):
                color = colors[i % len(colors)]
                segments = []
                for j in range(len(loop)):
                    a = (verts[loop[j]] - centroid).tolist()
                    b = (verts[loop[(j + 1) % len(loop)]] - centroid).tolist()
                    segments.append([a, b])
                edge_groups.append(
                    {
                        "segments": segments,
                        "color": color,
                        "label": f"hole {i}",
                    }
                )
            return edge_groups, len(loops)

        edge_groups, n_holes = await _run_with_log(_run)

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        if "edge_groups" not in ann:
            ann["edge_groups"] = []
        ann["edge_groups"].extend(edge_groups)
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        return JSONResponse({"ok": True, "nHoles": n_holes})

    async def run_skeletonize(request):
        """Run skeletonization on the current mesh."""
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )

        from skeliner.skeletonize import skeletonize

        body = await request.json()
        params = body.get("params", {})
        mesh = mesh_state["mesh"]

        # If break_up_mesh has run, use the preprocessing track.  Asked of
        # _current_track so the panel cannot label one track and run the other.
        track = _current_track()
        if track == "preprocessing":
            from skeliner.dataclass import (
                Discarded,
                MeshComponents,
                Neurites,
            )

            # Copy the component lists, but carry the names across with
            # them: they are what the skeleton reads to set `ntype`, and
            # rebuilding a bare Neurites here silently exported every
            # neurite as type 0 no matter what it had been called.
            src = mesh_state["neurites"]
            labels = getattr(src, "labels", None)
            swc_types = getattr(src, "swc_types", None)
            params["components"] = MeshComponents(
                soma=mesh_state.get("soma"),
                organelles=mesh_state["organelles"],
                neurites=Neurites(
                    list(src),
                    labels=None if labels is None else list(labels),
                    swc_types=None if swc_types is None else list(swc_types),
                ),
                discarded=Discarded(list(mesh_state.get("discarded") or [])),
            )

        await _log("[skeliner] Skeletonizing...")
        skel = await _run_with_log(skeletonize, mesh, verbose=True, **params)
        await _log(
            f"Skeletonized: {len(skel.nodes):,} nodes, {len(skel.edges):,} edges"
        )

        # Add as a skeleton layer
        skel_name = "skeleton"
        color = SKEL_COLORS[len(skeleton_states) % len(SKEL_COLORS)]
        buffers = _skeleton_to_buffers(skel, mesh_state["centroid"])
        buffers["color"] = color

        skeleton_states[skel_name] = {
            "skeleton": skel,
            "path": "",
            "buffers": buffers,
            "color": color,
            "mesh": mesh,
        }

        await broadcast(
            {"type": "skeleton_loaded", "payload": {"name": skel_name, **buffers}}
        )
        return JSONResponse(
            {
                "ok": True,
                "nNodes": len(skel.nodes),
                "nEdges": len(skel.edges),
                # Which track produced this one.  Reported rather than
                # inferred by the client, so a run and its label agree even
                # if the components changed while it was working.
                "ranTrack": track,
            }
        )

    def _face_owner_for(name: str):
        """Cached per-face bin owner for one skeleton layer.

        Recomputing costs an O(faces) pass — cheap, but not free at a million
        faces and one query per click.

        The cache holds the mesh and skeleton it was built from and compares
        them by identity, so replacing either invalidates it automatically.
        The alternative, clearing a flag at every site that swaps a mesh or a
        skeleton, is five call sites today and a silent staleness bug the first
        time someone adds a sixth.  Keeping the references alive is also what
        makes ``is`` sound here: neither object can be freed and have its id
        reused while the cache still points at it.

        Identity cannot see a skeleton **mutated in place**, and the ``post``
        primitives all mutate in place (``prune``, ``graft``, ``clip``).  Any
        route that edits a skeleton must therefore drop ``_face_owner`` itself
        — one line, in the function doing the mutating.
        """
        sstate = skeleton_states[name]
        mesh, skel = mesh_state["mesh"], sstate["skeleton"]
        cached = sstate.get("_face_owner")
        if cached is None or cached[0] is not mesh or cached[1] is not skel:
            from skeliner import dx

            sstate["_face_owner"] = (mesh, skel, dx.face_owner(skel, mesh))
        return sstate["_face_owner"][2]

    def _bin_neighbors(skel, node: int) -> np.ndarray:
        """Nodes sharing an edge with *node*."""
        edges = np.asarray(skel.edges)
        touching = edges[(edges[:, 0] == node) | (edges[:, 1] == node)]
        nbrs = np.unique(touching)
        return nbrs[nbrs != node]

    def _skel_edit_target(body):
        """Resolve and check the skeleton an edit names.

        Shared by the bin edits and the edge edits: both need a skeleton that
        exists, carries mesh data, and was built from the mesh now loaded.

        Returns ``(name, sstate, skel, error_response)``; only one of the
        last two is ever meaningful.
        """
        name = body.get("name")
        if not name or name not in skeleton_states:
            return (
                None,
                None,
                None,
                JSONResponse(
                    {"ok": False, "error": "No such skeleton"}, status_code=400
                ),
            )
        sstate = skeleton_states[name]
        skel = sstate.get("skeleton")
        if skel is None or skel.node2verts is None:
            return (
                None,
                None,
                None,
                JSONResponse(
                    {"ok": False, "error": "Skeleton carries no mesh data"},
                    status_code=400,
                ),
            )
        if mesh_state["mesh"] is None:
            return (
                None,
                None,
                None,
                JSONResponse({"ok": False, "error": "No mesh loaded"}, status_code=400),
            )
        if _skel_is_stale(sstate):
            return (
                None,
                None,
                None,
                JSONResponse(
                    {
                        "ok": False,
                        "error": f"'{name}' was built from a different mesh — its "
                        "vertex maps no longer match. Re-skeletonize first.",
                    },
                    status_code=409,
                ),
            )
        pairing = _skel_pairing(sstate)
        if not pairing["ok"]:
            return (
                None,
                None,
                None,
                JSONResponse(
                    {
                        "ok": False,
                        "error": f"'{name}' does not belong to the loaded mesh: "
                        f"{pairing['reason']}.",
                    },
                    status_code=409,
                ),
            )
        return name, sstate, skel, None

    async def bin_reassign_preview(request):
        """Work out what moving the selected surface into a bin would do.

        Runs the real :func:`skeliner.post.reassign_verts` on a copy and keeps
        that copy, so committing installs the very skeleton this described
        rather than a second computation of it.  The copy costs ~100 ms on the
        largest cell measured, which is the price of the two never disagreeing.

        The surface to move is named one of two ways, because the two
        gestures that produce it are different in kind:

        ``faces``
            What a lasso picked.  The lasso picks *faces*; a node owns
            *vertices*, so a selected face contributes its three, and the
            donors are restricted to the destination's graph neighbours —
            see :meth:`get_bin`.
        ``fromNodes``
            Whole bins, picked by clicking them.  This is the merge, and it
            needs no lasso: the bins are already the selection.  Merging is
            direction-free — the union of two bins has the same centroid and
            the same radii whichever id survives — so *to* only decides which
            node id remains.
        """
        import copy as _copy

        from skeliner import dx, post

        body = await request.json()
        name, sstate, skel, err = _skel_edit_target(body)
        if err is not None:
            return err
        mesh = mesh_state["mesh"]

        faces = np.asarray(body.get("faces") or [], dtype=np.int64)
        from_nodes = np.asarray(body.get("fromNodes") or [], dtype=np.int64)
        if faces.size == 0 and from_nodes.size == 0:
            return JSONResponse(
                {"ok": False, "error": "No faces selected"}, status_code=400
            )
        if faces.size and (faces.min() < 0 or faces.max() >= len(mesh.faces)):
            return JSONResponse(
                {"ok": False, "error": "face id out of range"}, status_code=400
            )
        if from_nodes.size and (
            from_nodes.min() < 0 or from_nodes.max() >= len(skel.nodes)
        ):
            return JSONResponse(
                {"ok": False, "error": "node id out of range"}, status_code=400
            )

        to = body.get("to")
        if to is None:
            return JSONResponse(
                {"ok": False, "error": "No destination bin"}, status_code=400
            )
        to = int(to)
        if not 0 <= to < len(skel.nodes):
            return JSONResponse(
                {"ok": False, "error": "node id out of range"}, status_code=400
            )
        if to == 0:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "node 0 is the soma, not a bin — its vertices "
                    "belong to the components. Edit ▸ Mesh Comp. owns those.",
                },
                status_code=400,
            )

        owner = np.full(len(mesh.vertices), -1, dtype=np.int64)
        for nid, owned in enumerate(skel.node2verts):
            owned = np.asarray(owned, dtype=np.int64)
            if nid and owned.size:
                owner[owned] = nid

        n_unowned = n_far = 0
        if from_nodes.size:
            # Merging bins that do not touch would make a "bin" that is not a
            # cross-section of anything, with a centroid between the places it
            # draws from — the same reason the lasso is bounded below.
            merged = set(int(n) for n in from_nodes) | {to}
            if 0 in merged:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "node 0 is the soma, not a bin — it cannot "
                        "take part in a merge.",
                    },
                    status_code=400,
                )
            seen, stack = {to}, [to]
            while stack:
                for nb in _bin_neighbors(skel, stack.pop()):
                    nb = int(nb)
                    if nb in merged and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            stranded = sorted(merged - seen)
            if stranded:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"node(s) {', '.join(map(str, stranded))} do not "
                        f"connect to node {to} through the selection — the bins "
                        "in a merge must form one connected piece.",
                    },
                    status_code=400,
                )
            moving = np.unique(
                np.concatenate(
                    [
                        np.asarray(skel.node2verts[int(n)], dtype=np.int64)
                        for n in from_nodes
                        if int(n) != to
                    ]
                    or [np.empty(0, dtype=np.int64)]
                )
            )
        else:
            picked = np.unique(np.asarray(mesh.faces)[faces])
            allowed = np.append(_bin_neighbors(skel, to), to)
            own = owner[picked]
            moving = picked[np.isin(own, allowed) & (own != to)]
            n_unowned = int((own < 0).sum())
            n_far = int((~np.isin(own, allowed) & (own >= 0)).sum())

        if moving.size == 0:
            why = "already owned by that bin"
            if n_far:
                why = (
                    f"owned by bins that do not touch node {to} — a bin may "
                    "only take surface from its neighbours"
                )
            elif n_unowned:
                why = (
                    "soma, organelle or discarded surface, which Edit ▸ Mesh Comp. owns"
                )
            return JSONResponse(
                {"ok": False, "error": f"Nothing to move: the selection is {why}"},
                status_code=400,
            )

        before = dx._fragmented_bins(skel.node2verts, mesh)
        after_skel = _copy.deepcopy(skel)
        result = post.reassign_verts(after_skel, moving, to, mesh=mesh)
        after = dx._fragmented_bins(after_skel.node2verts, mesh)

        # Report only bins this edit *made* worse: 20 of 582 bins on a real
        # cell are already fragmented, and the skeletonizer produced them.
        old2new = result.get("old2new")
        touched = [to, *result["donors"]]
        broke = []
        for nid in touched:
            new_id = nid if old2new is None else int(old2new[nid])
            if new_id < 0:
                continue
            now = after.get(new_id, 1)
            if now > before.get(nid, 1):
                broke.append({"node": new_id, "pieces": now})

        moved_faces = np.flatnonzero(
            np.isin(np.asarray(mesh.faces), moving).sum(axis=1) >= 2
        )
        summary = {
            "to": to,
            "moved": int(result["moved"]),
            "donors": [int(d) for d in result["donors"]],
            "dropped": [int(d) for d in result["dropped"]],
            "staleRadii": list(result["stale_radii"]),
            "ignoredUnowned": n_unowned,
            "ignoredFar": n_far,
            "fragmented": broke,
            "radiusBefore": float(skel.r[to]),
            "radiusAfter": float(
                after_skel.r[to if old2new is None else int(old2new[to])]
            ),
            "movedFaces": moved_faces.tolist(),
        }
        sstate["pending_bin_edit"] = {
            "base": skel,
            "mesh": mesh,
            "skeleton": after_skel,
            "summary": summary,
        }
        return JSONResponse({"ok": True, **summary})

    async def bin_reassign_apply(request):
        """Install the previewed skeleton."""
        body = await request.json()
        name = body.get("name")
        sstate = skeleton_states.get(name or "")
        pending = sstate.get("pending_bin_edit") if sstate else None
        if pending is None:
            return JSONResponse(
                {"ok": False, "error": "Nothing to apply — preview first"},
                status_code=400,
            )
        # The preview names vertices of one mesh and one partition.  If
        # either has been replaced since, applying it would edit something
        # else — the same identity check staleness uses.
        if pending["mesh"] is not mesh_state["mesh"] or pending[
            "base"
        ] is not sstate.get("skeleton"):
            sstate["pending_bin_edit"] = None
            return JSONResponse(
                {
                    "ok": False,
                    "error": "The mesh or the skeleton changed since the "
                    "preview — draw the selection again.",
                },
                status_code=409,
            )

        summary = pending["summary"]
        # A new object, so the cached face-owner map invalidates by identity.
        sstate["skeleton"] = pending["skeleton"]
        sstate["pending_bin_edit"] = None
        await _rebroadcast_skeletons()
        if summary.get("split"):
            await _log(
                f"Split node {summary['parent']}: {summary['moved']:,} verts "
                f"→ new node {summary['to']}"
            )
        else:
            note = ""
            if summary["dropped"]:
                note = f", {len(summary['dropped'])} node(s) dropped and renumbered"
            await _log(
                f"Bin edit: {summary['moved']:,} verts → node {summary['to']}{note}"
            )
        return JSONResponse({"ok": True, **summary})

    async def bin_split_preview(request):
        """Work out what promoting part of a bin to its own node would do.

        The third bin verb, and the one the move gesture cannot express,
        because its destination does not exist yet.  Shares the pending slot
        and the apply route with the other two — there is one skeleton, so
        there is one pending edit.

        The new node is joined to its parent and to nothing else; see
        :func:`skeliner.post.split_node` for why its other edges are not
        inferred.  ``supported`` reports whether even that one edge has
        surface behind it, which it does not when the split cuts a bin that
        was already in two patches.
        """
        import copy as _copy

        from skeliner import dx, post

        body = await request.json()
        name, sstate, skel, err = _skel_edit_target(body)
        if err is not None:
            return err
        mesh = mesh_state["mesh"]

        node = body.get("splitFrom")
        if node is None:
            return JSONResponse(
                {"ok": False, "error": "No bin to split"}, status_code=400
            )
        node = int(node)
        if not 0 <= node < len(skel.nodes):
            return JSONResponse(
                {"ok": False, "error": "node id out of range"}, status_code=400
            )

        faces = np.asarray(body.get("faces") or [], dtype=np.int64)
        if faces.size == 0:
            return JSONResponse(
                {"ok": False, "error": "No faces selected"}, status_code=400
            )
        if faces.min() < 0 or faces.max() >= len(mesh.faces):
            return JSONResponse(
                {"ok": False, "error": "face id out of range"}, status_code=400
            )
        picked = np.unique(np.asarray(mesh.faces)[faces])

        before = dx._fragmented_bins(skel.node2verts, mesh)
        after_skel = _copy.deepcopy(skel)
        try:
            result = post.split_node(after_skel, node, picked, mesh=mesh)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        after = dx._fragmented_bins(after_skel.node2verts, mesh)

        new_id = result["node"]
        broke = [
            {"node": nid, "pieces": after.get(nid, 1)}
            for nid in (node, new_id)
            if after.get(nid, 1) > before.get(nid, 1)
        ]
        moved_faces = np.flatnonzero(
            np.isin(np.asarray(mesh.faces), after_skel.node2verts[new_id]).sum(axis=1)
            >= 2
        )
        summary = {
            "split": True,
            "to": new_id,
            "parent": node,
            "moved": result["moved"],
            "ignoredNotOwned": result["ignored"],
            "supported": result["supported"],
            # Nothing is emptied and nothing is renumbered — the new node is
            # appended — so these stay empty and the client's shared summary
            # renderer says nothing about them.
            "donors": [node],
            "dropped": [],
            "staleRadii": list(result["stale_radii"]),
            "fragmented": broke,
            "radiusBefore": float(skel.r[node]),
            "radiusAfter": float(after_skel.r[node]),
            "radiusNew": float(after_skel.r[new_id]),
            "movedFaces": moved_faces.tolist(),
        }
        sstate["pending_bin_edit"] = {
            "base": skel,
            "mesh": mesh,
            "skeleton": after_skel,
            "summary": summary,
        }
        return JSONResponse({"ok": True, **summary})

    async def bin_reassign_cancel(request):
        """Drop the pending bin edit."""
        body = await request.json()
        sstate = skeleton_states.get(body.get("name") or "")
        if sstate is not None:
            sstate["pending_bin_edit"] = None
        return JSONResponse({"ok": True})

    async def get_bin(request):
        """The surface one skeleton node owns.

        A node's position and every radius are computed from the mesh vertices
        it owns, so this is what the node actually *is*.  Small enough to fetch
        per click — a bin is ~0.3–0.9 KB of face ids on real cells — which is
        why the partition is never shipped in bulk.

        Accepts either ``node`` (which bin?) or ``face`` (which bin owns this
        piece of surface?), the same lookup in both directions.
        """
        body = await request.json()
        name = body.get("name")
        if not name or name not in skeleton_states:
            return JSONResponse(
                {"ok": False, "error": "No such skeleton"}, status_code=400
            )
        skel = skeleton_states[name].get("skeleton")
        if skel is None or skel.node2verts is None:
            return JSONResponse(
                {"ok": False, "error": "Skeleton carries no mesh data"},
                status_code=400,
            )
        if mesh_state["mesh"] is None:
            return JSONResponse(
                {"ok": False, "error": "No mesh loaded"}, status_code=400
            )
        if _skel_is_stale(skeleton_states[name]):
            # vert2node indexes the mesh this was built from; against the
            # current one it would name a plausible but wrong patch.
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"'{name}' was built from a different mesh — its "
                    "vertex maps no longer match. Re-skeletonize first.",
                },
                status_code=409,
            )
        pairing = _skel_pairing(skeleton_states[name])
        if not pairing["ok"]:
            # Same consequence as staleness, different cause: this skeleton was
            # never this mesh's.  Against a *larger* wrong mesh every id is in
            # range, so the patch returned would look like an answer.
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"'{name}' does not belong to the loaded mesh: "
                    f"{pairing['reason']}.",
                },
                status_code=409,
            )

        owner = _face_owner_for(name)

        node = body.get("node")
        if node is None:
            face = body.get("face")
            if face is None:
                return JSONResponse(
                    {"ok": False, "error": "Pass either node or face"},
                    status_code=400,
                )
            if not (0 <= int(face) < len(owner)):
                return JSONResponse(
                    {"ok": False, "error": "face id out of range"}, status_code=400
                )
            node = int(owner[int(face)])
            if node < 0:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "That face belongs to no bin — soma, organelle "
                        "and discarded surface are not part of the neurites.",
                    },
                    status_code=400,
                )
        node = int(node)
        if not (0 <= node < len(skel.nodes)):
            return JSONResponse(
                {"ok": False, "error": "node id out of range"}, status_code=400
            )

        faces = np.flatnonzero(owner == node)
        nbrs = _bin_neighbors(skel, node)
        # Surface this bin may legitimately take from.  Taking vertices from a
        # distant bin would build a "bin" that is not a cross-section of
        # anything, with a centroid between the two places it draws from, so
        # the donors are the graph neighbours and nothing else.
        scope = np.flatnonzero(np.isin(owner, np.append(nbrs, node)))
        return JSONResponse(
            {
                "ok": True,
                "node": node,
                "faces": faces.tolist(),
                "nVerts": int(len(skel.node2verts[node])),
                "radius": float(skel.r[node]),
                "neighbors": nbrs.tolist(),
                "scopeFaces": scope.tolist(),
                # node 0's "bin" is soma.verts, not a bin: it is assigned
                # wholesale by the soma stitch and is Edit ▸ Mesh Comp.'s to
                # change.
                "editable": node != 0,
            }
        )

    # ── Edge editing (B2) ────────────────────────────────────────────────
    # `edges` is the one thing on a Skeleton that nothing else is derived
    # from, which is why it can be edited directly — and why such an edit
    # does not survive a re-skeletonize.

    def _graph_shape(skel):
        """Components and independent cycles — what an edge edit moves.

        An editing UI that can produce an orphan or a cycle should say so
        when it happens, not leave it for an exporter to trip on.
        """
        g = skel._igraph()
        n_comp = len(g.components())
        return {
            "components": int(n_comp),
            "cycles": int(g.ecount() - g.vcount() + n_comp),
        }

    async def edge_support(request):
        """Which node pairs the surface joins, so the verb can be named.

        The client needs this to tell a **restore** (the surface really does
        join these bins) from a **graft** (it does not), and to say which tree
        edges have no surface behind them at all.

        Deliberately not a list of things to look at.  An earlier version
        grouped the dropped pairs into "loops" and ranked them by how far the
        tree detours, on the theory that a big detour meant a wrongly merged
        mesh.  Measured on 549190673, 147 of the 156 pairs at least three hops
        apart lie within a single branch with the bins overlapping in space —
        a dense axon tuft, which is indistinguishable here from a fusion.  A
        signal that cannot separate the two has no business being a menu.
        """
        from skeliner import dx

        body = await request.json()
        name, sstate, skel, err = _skel_edit_target(body)
        if err is not None:
            return err

        rep = dx.edge_support(skel, mesh_state["mesh"])
        return JSONResponse(
            {
                "ok": True,
                "name": name,
                "nTree": rep["n_tree"],
                "dropped": [[int(u), int(v)] for u, v in rep["dropped"]],
                "unsupported": [[int(u), int(v)] for u, v in rep["unsupported"]],
            }
        )

    #: Diagnostics offered in the skeleton panel.  Each returns a list of
    #: ``{"node", "detail"}``, optionally with ``"cut"`` / ``"keep"`` edge
    #: lists which the route draws.  They are *candidate listers* — none of
    #: them edits, and none of them ranks by a threshold the corpus has not
    #: justified, so every cut is a parameter the user sets and can see the
    #: effect of.
    def _dx_junctions(skel, p):
        from skeliner import dx

        flagged, stats = dx.suspicious_junctions(
            skel,
            min_degree=int(p.get("minDegree", 4)),
            min_distal_cable=float(p.get("minCable", 5.0)),
            cable_unit=p.get("cableUnit", "um"),
            min_components=int(p.get("minComponents", 2)),
            return_stats=True,
        )
        # Lists junctions and what each arm carries.  It proposes no cut:
        # `arm_pairing` used to ride along here and was removed, because its
        # score is purely directional and cannot tell a 0.8 um binning stub
        # from a 261 um dendrite — so it paired stubs with major arms and
        # ranked its most destructive proposals highest.  See
        # `.claude/labbook/2026-08-11-arm-pairing-retracted.md`.
        out = []
        for nid in flagged:
            d = sorted(stats[nid]["distal_cables"], reverse=True)
            arms = ", ".join(f"{c / 1000:.1f}" for c in d[:4])
            out.append(
                {
                    "node": nid,
                    "detail": f"deg {stats[nid]['degree']} · arms {arms} µm",
                }
            )
        return out

    def _dx_short_twigs(skel, p):
        """Terminal twigs, measured against the branch node they hang off.

        Deliberately *not* called phantom-anything.  Whether a short twig is
        a binning artefact or a real branch is exactly what cannot be decided
        from the numbers — it is why ``postprocess=False`` leaves pruning off
        — so this lists them for a human and names the ratio it sorted by.
        """
        import numpy as _np

        edges = _np.asarray(skel.edges, dtype=_np.int64).reshape(-1, 2)
        n = len(skel.nodes)
        if not len(edges):
            return []
        adj = [[] for _ in range(n)]
        for a, b in edges:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        deg = _np.zeros(n, dtype=_np.int64)
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        par = _np.full(n, -1, dtype=_np.int64)
        seen = _np.zeros(n, dtype=bool)
        seen[0] = True
        order = [0]
        for v in order:
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    par[w] = v
                    order.append(w)

        def _len(u, w):
            return float(_np.linalg.norm(skel.nodes[u] - skel.nodes[w]))

        radii = skel.r
        max_ratio = float(p.get("maxRatio", 3.0))
        max_nodes = int(p.get("maxNodes", 3))
        out = []
        for leaf in range(1, n):
            if deg[leaf] != 1:
                continue
            chain = [leaf]
            u = int(par[leaf])
            while u > 0 and deg[u] == 2:
                chain.append(u)
                u = int(par[u])
            if u < 0 or deg[u] < 3 or len(chain) > max_nodes:
                continue
            host_r = float(radii[u])
            if host_r <= 0:
                continue
            cable = sum(_len(c, int(par[c])) for c in chain)
            ratio = cable / host_r
            if ratio > max_ratio:
                continue
            out.append(
                {
                    "node": leaf,
                    "detail": (
                        f"{len(chain)}-node twig · {cable / 1000:.2f} µm "
                        f"= {ratio:.1f}× host radius (node {u})"
                    ),
                    "ratio": ratio,
                    # No edges drawn.  Ninety scattered twigs pooled into one
                    # annotation cannot be focused (they are all over the
                    # cell) and cannot be acted on together (most of them are
                    # real branches), so the group is noise in the list.
                    # Edges are worth drawing when they form one thing you
                    # trace — a loop — not N unrelated fragments; a twig is
                    # already at the marker you focused.
                }
            )
        out.sort(key=lambda r: r["ratio"])
        return out

    def _dx_degree(skel, p):
        from skeliner import dx

        k = int(p.get("minDegree", 4))
        out = []
        for d in range(k, 12):
            for nid in dx.nodes_of_degree(skel, d):
                out.append({"node": int(nid), "detail": f"degree {d}"})
        return out

    def _dx_orphans(skel, _p):
        from skeliner import dx

        iso = dx.check_connectivity(skel, return_isolated=True)
        if iso is True:
            return []
        return [{"node": int(nid), "detail": "unreachable from soma"} for nid in iso]

    def _dx_tips(skel, p):
        from skeliner import dx

        tips = dx.suspicious_tips(
            skel,
            near_factor=float(p.get("nearFactor", 1.2)),
            path_ratio_thresh=float(p.get("pathRatio", 2.0)),
        )
        return [
            {"node": int(nid), "detail": "tip near soma, long path"} for nid in tips
        ]

    def _dx_cycles(skel, _p):
        """Loops, and the edge on each that the surface will not vouch for.

        The only diagnostic here whose candidate rests on evidence outside
        the skeleton.  A loop cannot occur by accident — the MST returns a
        tree — so one exists only because a connection was asserted, and the
        loop then says *two of these edges cannot both be right*.
        `edge_support` settles which: a tree edge with no surface joining its
        bins, sitting on a loop that offers another route, is the join the
        MST should not have made.
        """
        from skeliner import dx

        out = []
        for i, cyc in enumerate(dx.cycles(skel, mesh_state["mesh"])):
            n_break = len(cyc["breaks"])
            detail = (
                f"loop of {len(cyc['nodes'])} nodes, "
                f"{cyc['length'] / 1000:.1f} µm · "
                + (
                    f"{n_break} edge(s) the surface does not support"
                    if n_break
                    else "every edge is surface-supported — the loop is real"
                )
            )
            # Where to put the marker.  `nodes[0]` is merely the lowest id
            # on the ring, which is usually node 0 — the soma, the least
            # informative point on any loop and the one place a marker
            # scaled by node radius swallows the structure.  Prefer an end
            # of the break edge, since that is what the row is *for*;
            # failing that, the point on the loop furthest from the soma.
            if cyc["breaks"]:
                at = int(cyc["breaks"][0][0])
            else:
                ring = [n for n in cyc["nodes"] if n != 0] or cyc["nodes"]
                soma = skel.nodes[0]
                at = max(
                    ring, key=lambda n: float(np.linalg.norm(skel.nodes[n] - soma))
                )
            out.append(
                {
                    "node": at,
                    "detail": detail,
                    # The whole loop, so it can be followed round, and the
                    # break candidates separately so they stand out on it.
                    "keep": cyc["edges"],
                    "cut": cyc["breaks"],
                }
            )
        return out

    #: Named after the `dx` function each one runs, and ordered the way
    #: `dx.__skeleton__` orders them — the one invariant check, then the
    #: plain enumerations, then the heuristics.  Both so that what the panel
    #: calls a thing and what the library calls it cannot drift apart, and
    #: so the list reads from "this is broken" to "this might repay a look".
    DIAGNOSTICS = {
        "orphans": ("isolated nodes", [0.9, 0.3, 0.9], _dx_orphans, None),
        "cycles": (
            "cycles",
            [1.0, 0.5, 0.0],
            _dx_cycles,
            ("no surface behind it", "the loop"),
        ),
        "degree": ("nodes of degree", [0.95, 0.75, 0.2], _dx_degree, None),
        "twigs": ("short twigs", [0.35, 0.75, 0.95], _dx_short_twigs, None),
        "tips": ("suspicious tips", [0.4, 0.9, 0.5], _dx_tips, None),
        "junctions": (
            "suspicious junctions",
            [0.95, 0.35, 0.25],
            _dx_junctions,
            None,
        ),
    }

    async def diagnose(request):
        """List candidate defects as markers so they can be looked at.

        The whole point is the looking.  Every threshold below is a heuristic
        with no validation behind it, and the corpus can rank candidates but
        cannot say which are real — that needs the mesh beside the skeleton,
        which is why this refuses to run without a paired one.  Nothing here
        edits; the verbs stay where they already are.

        Markers replace the previous run of the *same* diagnostic rather than
        accumulating, so re-running with a different cut answers "what does
        this threshold do" instead of piling three answers on top of
        each other.
        """
        body = await request.json()
        kind = body.get("kind")
        if kind not in DIAGNOSTICS:
            return JSONResponse(
                {"ok": False, "error": f"Unknown diagnostic {kind!r}"},
                status_code=400,
            )
        name, sstate, skel, err = _skel_edit_target(body)
        if err is not None:
            return err

        label, color, fn, edge_labels = DIAGNOSTICS[kind]
        try:
            found = fn(skel, body.get("params") or {})
        except Exception as exc:  # noqa: BLE001 - report, do not 500 the panel
            return JSONResponse(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                status_code=400,
            )

        # The client renders everything centroid-relative — `_mesh_to_buffers`
        # subtracts it from the vertices and `_skeleton_to_buffers` subtracts
        # the same one from the nodes, so the two overlay.  A marker built
        # from raw node coordinates lands a whole centroid away, which on CAVE
        # cells is hundreds of microns off screen.
        centroid = np.asarray(mesh_state["centroid"], dtype=np.float64)

        def _seg(u, v):
            return [
                [float(x) for x in (skel.nodes[int(u)] - centroid)],
                [float(x) for x in (skel.nodes[int(v)] - centroid)],
            ]

        radii = skel.r
        markers, cut_segs, keep_segs = [], [], []
        for row in found:
            nid = int(row["node"])
            r = float(radii[nid]) if nid < len(radii) else 0.0
            markers.append(
                {
                    "position": [float(x) for x in (skel.nodes[nid] - centroid)],
                    "color": color,
                    # The marker is the node's own size, near enough — a bead
                    # on it, not a balloon around it.  A multiple of the
                    # radius overshadows exactly the surface being judged,
                    # and a flat floor inflates the thinnest nodes most,
                    # which is where twigs live.  1.15x is just enough to
                    # break the tube surface and be seen through it.
                    # The ceiling is only for the soma, whose true radius
                    # would swallow itself.  Candidates are found from the
                    # list, not by being spotted across the cell, so there is
                    # nothing to buy by drawing them larger than they are.
                    "radius": min(max(r * 1.15, 50.0), 2000.0),
                    "label": f"{label}: {row['detail']}",
                    "skelName": name,
                    "node": nid,
                    "dx": kind,
                }
            )
            cut_segs.extend(_seg(u, v) for u, v in row.get("cut", ()))
            keep_segs.extend(_seg(u, v) for u, v in row.get("keep", ()))

        # Drawn on the geometry, because that is what the claim is about.
        # `overlay` keeps them visible: these segments are *the same edges*
        # the skeleton already draws, so without it they z-fight with it and
        # the only way to see the highlight is to hide the skeleton — which
        # removes the context the highlight exists to sit in.
        cut_name, keep_name = edge_labels or ("proposed cut", "cable that continues")
        groups = []
        if cut_segs and cut_name:
            groups.append(
                {
                    "segments": cut_segs,
                    "color": [1.0, 0.25, 0.25],
                    "label": f"{label}: {cut_name} ({len(cut_segs)} edges)",
                    "overlay": True,
                    "dx": kind,
                }
            )
        if keep_segs and keep_name:
            groups.append(
                {
                    "segments": keep_segs,
                    "color": [0.3, 0.9, 0.4],
                    "label": f"{label}: {keep_name} ({len(keep_segs)} edges)",
                    "overlay": True,
                    "dx": kind,
                }
            )

        ann = {}
        if annotations_path.exists():
            ann = json.loads(annotations_path.read_text(encoding="utf-8"))
        ann["markers"] = [
            m for m in ann.get("markers", []) if m.get("dx") != kind
        ] + markers
        ann["edge_groups"] = [
            g for g in ann.get("edge_groups", []) if g.get("dx") != kind
        ] + groups
        annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        await broadcast({"type": "annotations_updated"})

        await _log(f"[skeliner.dx] {label}: {len(markers)} candidate(s) on '{name}'")
        return JSONResponse(
            {
                "ok": True,
                "kind": kind,
                "n": len(markers),
                "nCut": len(cut_segs),
            }
        )

    async def edge_edit_preview(request):
        """Work out what clipping or grafting one edge would do.

        Which of the three verbs a node pair names is not the client's to
        decide, so it is derived here from the two graphs:

        =============  ==========================================
        in ``T``       **clip** — and cutting is asymmetric: on a
                       tree edge it orphans a whole subtree, on a
                       cycle edge it disconnects nothing.
        in ``G∖T``     **restore** — the surface really is joined
                       there; the MST cut this cycle elsewhere.
        in neither     **graft** — a leap across a genuine gap,
                       which is a different claim and is labelled
                       as one.
        =============  ==========================================

        Runs the real primitive on a copy and keeps it, as the bin edits do,
        so applying installs the very skeleton this described.  The copy also
        means the installed object is a *new* one, which is what invalidates
        the cached face-owner map — ``clip`` and ``graft`` mutate in place and
        would otherwise leave it stale.
        """
        import copy as _copy

        from skeliner import dx, post
        from skeliner.skeletonize import _edges_from_mesh

        body = await request.json()
        name, sstate, skel, err = _skel_edit_target(body)
        if err is not None:
            return err
        mesh = mesh_state["mesh"]

        u, v = body.get("u"), body.get("v")
        if u is None or v is None:
            return JSONResponse(
                {"ok": False, "error": "Pass two nodes, u and v"}, status_code=400
            )
        u, v = int(u), int(v)
        if u == v:
            return JSONResponse(
                {"ok": False, "error": "An edge needs two different nodes"},
                status_code=400,
            )
        n = len(skel.nodes)
        if not (0 <= u < n and 0 <= v < n):
            return JSONResponse(
                {"ok": False, "error": "node id out of range"}, status_code=400
            )

        pair = (min(u, v), max(u, v))
        tree = np.unique(
            np.sort(np.asarray(skel.edges, dtype=np.int64), axis=1), axis=0
        )
        in_tree = bool(((tree[:, 0] == pair[0]) & (tree[:, 1] == pair[1])).any())
        surface = _edges_from_mesh(
            np.asarray(mesh.edges_unique), skel.vert2node, len(mesh.vertices)
        )
        supported = bool(
            ((surface[:, 0] == pair[0]) & (surface[:, 1] == pair[1])).any()
        )
        verb = "clip" if in_tree else ("restore" if supported else "graft")

        # How far apart the two already are along the tree, which is how big
        # a cycle a restore or a graft would close.  ``-1`` when the tree does
        # not join them at all — then the edit reconnects the skeleton instead
        # of closing anything.  Asked of the components first, because
        # ``get_shortest_path`` warns when there is no path to find.
        g = skel._igraph()
        comp = g.components().membership
        hops = len(g.get_shortest_path(u, v)) - 1 if comp[u] == comp[v] else -1

        before = _graph_shape(skel)
        before_iso = set(dx.check_connectivity(skel, return_isolated=True))
        after_skel = _copy.deepcopy(skel)
        if verb == "clip":
            post.clip(after_skel, u, v)
        else:
            post.graft(after_skel, u, v)
        after = _graph_shape(after_skel)
        orphans = sorted(
            set(dx.check_connectivity(after_skel, return_isolated=True)) - before_iso
        )

        summary = {
            "verb": verb,
            "u": u,
            "v": v,
            "supported": supported,
            "hops": hops,
            "orphans": [int(o) for o in orphans],
            "unsupportedTree": in_tree and not supported,
            "componentsBefore": before["components"],
            "componentsAfter": after["components"],
            "cyclesBefore": before["cycles"],
            "cyclesAfter": after["cycles"],
        }
        sstate["pending_edge_edit"] = {
            "base": skel,
            "skeleton": after_skel,
            "summary": summary,
        }
        return JSONResponse({"ok": True, **summary})

    async def edge_edit_apply(request):
        """Install the previewed skeleton."""
        body = await request.json()
        name = body.get("name")
        sstate = skeleton_states.get(name or "")
        pending = sstate.get("pending_edge_edit") if sstate else None
        if pending is None:
            return JSONResponse(
                {"ok": False, "error": "Nothing to apply — preview first"},
                status_code=400,
            )
        # Node ids are positions in *this* skeleton, so the check is on the
        # skeleton alone — unlike a bin edit, an edge edit names no vertices
        # and so does not care which mesh is loaded.  A bin edit applied in
        # the meantime replaces the object and lands here.
        if pending["base"] is not sstate.get("skeleton"):
            sstate["pending_edge_edit"] = None
            return JSONResponse(
                {
                    "ok": False,
                    "error": "The skeleton changed since the preview — its "
                    "node ids may name different nodes now. Pick the edge again.",
                },
                status_code=409,
            )

        summary = pending["summary"]
        sstate["skeleton"] = pending["skeleton"]
        sstate["pending_edge_edit"] = None
        await _rebroadcast_skeletons()
        note = ""
        if summary["orphans"]:
            note = f", {len(summary['orphans']):,} node(s) now cut off from the soma"
        await _log(
            f"{summary['verb'].capitalize()} edge {summary['u']}—{summary['v']}{note}"
        )
        return JSONResponse({"ok": True, **summary})

    async def edge_edit_cancel(request):
        """Drop the pending edge edit."""
        body = await request.json()
        sstate = skeleton_states.get(body.get("name") or "")
        if sstate is not None:
            sstate["pending_edge_edit"] = None
        return JSONResponse({"ok": True})

    async def shortest_path_endpoint(request):
        """Compute shortest path between two faces or two skeleton nodes."""
        import heapq

        body = await request.json()
        path_type = body.get("type")

        if path_type == "mesh":
            if mesh_state["mesh"] is None:
                return JSONResponse(
                    {"ok": False, "error": "No mesh loaded"}, status_code=400
                )
            face1 = body.get("face1")
            face2 = body.get("face2")
            mode = body.get("mode", "edge")  # "edge" or "vertex"
            if face1 is None or face2 is None:
                return JSONResponse(
                    {"ok": False, "error": "Need face1 and face2"}, status_code=400
                )

            mesh = mesh_state["mesh"]

            def _run_mesh_path():
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import dijkstra

                n_faces = len(mesh.faces)
                if face1 < 0 or face1 >= n_faces or face2 < 0 or face2 >= n_faces:
                    return None, 0
                if face1 == face2:
                    return [face1], 0.0

                print(
                    f"[skeliner.path] Building {mode} adjacency for {n_faces:,} faces..."
                )
                centroids = mesh.triangles_center

                if mode == "vertex":
                    # Sparse face-vertex incidence matrix M, then
                    # M @ M.T gives vertex-adjacency (crosses fusions).
                    n_verts = len(mesh.vertices)
                    rows = np.repeat(np.arange(n_faces), 3)
                    cols = mesh.faces.ravel().astype(np.int32)
                    M = csr_matrix(
                        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                        shape=(n_faces, n_verts),
                    )
                    adj = M @ M.T
                    adj.setdiag(0)
                    adj.eliminate_zeros()
                    fi, fj = adj.nonzero()
                else:
                    # Edge-based: faces sharing an edge.
                    adj_pairs = mesh.face_adjacency
                    fi = np.concatenate([adj_pairs[:, 0], adj_pairs[:, 1]])
                    fj = np.concatenate([adj_pairs[:, 1], adj_pairs[:, 0]])

                dists = np.linalg.norm(centroids[fi] - centroids[fj], axis=1).astype(
                    np.float64
                )
                graph = csr_matrix((dists, (fi, fj)), shape=(n_faces, n_faces))

                print(f"[skeliner.path] Running Dijkstra face {face1} → {face2}...")
                dist_arr, predecessors = dijkstra(
                    graph,
                    directed=False,
                    indices=face1,
                    return_predecessors=True,
                )

                if np.isinf(dist_arr[face2]):
                    return None, 0

                path = []
                cur = face2
                while cur != face1 and cur >= 0:
                    path.append(int(cur))
                    cur = int(predecessors[cur])
                if cur < 0:
                    return None, 0
                path.append(int(face1))
                path.reverse()
                print(
                    f"[skeliner.path] Found path: {len(path)} faces, length={dist_arr[face2]:.1f}"
                )
                return path, float(dist_arr[face2])

            path, length = await _run_with_log(_run_mesh_path)

            if path is None:
                return JSONResponse(
                    {"ok": False, "error": "No path found (disconnected components?)"}
                )

            return JSONResponse(
                {
                    "ok": True,
                    "type": "mesh",
                    "path": path,
                    "length": length,
                    "nFaces": len(path),
                }
            )

        elif path_type == "skeleton":
            skel_name = body.get("skelName")
            node1 = body.get("node1")
            node2 = body.get("node2")
            if skel_name not in skeleton_states:
                return JSONResponse(
                    {"ok": False, "error": "Skeleton not found"}, status_code=400
                )

            buffers = skeleton_states[skel_name]["buffers"]

            def _run_skel_path():
                nodes = np.array(buffers["nodes"], dtype=np.float32).reshape(-1, 3)
                edges = np.array(buffers["edges"], dtype=np.int32).reshape(-1, 2)
                n_nodes = len(nodes)

                if node1 < 0 or node1 >= n_nodes or node2 < 0 or node2 >= n_nodes:
                    return None, None, 0
                if node1 == node2:
                    return [node1], [], 0.0

                print(
                    f"[skeliner.path] Building skeleton adjacency for {n_nodes:,} nodes..."
                )
                adj = [[] for _ in range(n_nodes)]
                for ei, (a, b) in enumerate(edges):
                    d = float(np.linalg.norm(nodes[a] - nodes[b]))
                    adj[int(a)].append((int(b), d, ei))
                    adj[int(b)].append((int(a), d, ei))

                print(f"[skeliner.path] Running Dijkstra node {node1} → {node2}...")
                dist_arr = [float("inf")] * n_nodes
                dist_arr[node1] = 0
                prev = [(-1, -1)] * n_nodes
                pq = [(0.0, node1)]
                visited = [False] * n_nodes

                while pq:
                    d, u = heapq.heappop(pq)
                    if visited[u]:
                        continue
                    visited[u] = True
                    if u == node2:
                        break
                    for v, w, ei in adj[u]:
                        if not visited[v]:
                            nd = d + w
                            if nd < dist_arr[v]:
                                dist_arr[v] = nd
                                prev[v] = (u, ei)
                                heapq.heappush(pq, (nd, v))

                if dist_arr[node2] == float("inf"):
                    return None, None, 0

                path_nodes = []
                path_edges = []
                cur = node2
                while cur != node1:
                    path_nodes.append(int(cur))
                    pn, pe = prev[cur]
                    path_edges.append(int(pe))
                    cur = pn
                path_nodes.append(int(node1))
                path_nodes.reverse()
                path_edges.reverse()

                segments = []
                for ei in path_edges:
                    a, b = int(edges[ei][0]), int(edges[ei][1])
                    segments.append([nodes[a].tolist(), nodes[b].tolist()])

                print(
                    f"[skeliner.path] Found path: {len(path_nodes)} nodes, length={dist_arr[node2]:.1f}"
                )
                return path_nodes, segments, float(dist_arr[node2])

            path_nodes, segments, length = await _run_with_log(_run_skel_path)

            if path_nodes is None:
                return JSONResponse({"ok": False, "error": "No path found"})

            return JSONResponse(
                {
                    "ok": True,
                    "type": "skeleton",
                    "pathNodes": path_nodes,
                    "segments": segments,
                    "length": length,
                    "nNodes": len(path_nodes),
                }
            )

        return JSONResponse(
            {"ok": False, "error": "Unknown path type"}, status_code=400
        )

    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        connected_clients.append(ws)
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "state_update":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current.update(msg["payload"])
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "selection":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["selection"] = msg["payload"]
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "node_selection":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["nodeSelection"] = msg["payload"]
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "manual_highlight":
                    current = {}
                    if state_path.exists():
                        current = json.loads(state_path.read_text(encoding="utf-8"))
                    current["manualHighlight"] = msg["payload"].get(
                        "manualHighlight", []
                    )
                    state_path.write_text(
                        json.dumps(current, indent=2), encoding="utf-8"
                    )
                elif msg.get("type") == "save_highlight":
                    # Append user-selected faces as a named annotation
                    ann = {}
                    if annotations_path.exists():
                        ann = json.loads(annotations_path.read_text(encoding="utf-8"))
                    if "highlights" not in ann:
                        ann["highlights"] = []
                    p = msg["payload"]
                    ann["highlights"].append(
                        {
                            "faces": p.get("faces", []),
                            "color": p.get("color", [1, 0.8, 0]),
                            "label": p.get("label", "selection"),
                        }
                    )
                    annotations_path.write_text(json.dumps(ann), encoding="utf-8")
        except Exception:
            pass
        finally:
            connected_clients.remove(ws)

    # ── File watcher ──────────────────────────────────────────────────

    html_path = Path(__file__).parent / "viewer.html"

    async def file_watcher():
        last_ann_mtime = 0.0
        last_cam_mtime = 0.0
        last_html_mtime = html_path.stat().st_mtime if html_path.exists() else 0.0
        while True:
            await asyncio.sleep(0.5)
            try:
                mtime = annotations_path.stat().st_mtime
                if mtime > last_ann_mtime:
                    last_ann_mtime = mtime
                    content = annotations_path.read_text(encoding="utf-8")
                    await broadcast(
                        {
                            "type": "annotations",
                            "payload": json.loads(content),
                        }
                    )
            except FileNotFoundError:
                pass
            try:
                mtime = camera_cmd_path.stat().st_mtime
                if mtime > last_cam_mtime:
                    last_cam_mtime = mtime
                    content = camera_cmd_path.read_text(encoding="utf-8")
                    parsed = json.loads(content)
                    if parsed:
                        await broadcast(
                            {
                                "type": "camera_command",
                                "payload": parsed,
                            }
                        )
            except FileNotFoundError:
                pass
            try:
                mtime = html_path.stat().st_mtime
                if mtime > last_html_mtime:
                    last_html_mtime = mtime
                    await broadcast({"type": "reload"})
            except FileNotFoundError:
                pass

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        asyncio.create_task(file_watcher())
        yield

    app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/", index),
            Route("/mesh", get_mesh),
            Route("/skeletons", get_skeletons),
            Route("/extra_meshes", get_extra_meshes),
            Route("/contact_sites", get_contact_sites),
            Route("/loaded", get_loaded),
            Route("/state", get_state, methods=["GET"]),
            Route("/save_availability", get_save_availability, methods=["GET"]),
            Route("/annotations", get_annotations, methods=["GET"]),
            Route("/update_annotations", update_annotations, methods=["POST"]),
            Route("/upload", upload_file, methods=["POST"]),
            Route("/upload_batch", upload_batch, methods=["POST"]),
            Route("/remove", remove_item, methods=["POST"]),
            Route("/update_state", post_state, methods=["POST"]),
            Route("/update_selection", post_selection, methods=["POST"]),
            Route("/detect_offsets", detect_offsets, methods=["POST"]),
            Route("/save_offsets", save_offsets, methods=["POST"]),
            Route("/load_offsets", load_offsets, methods=["POST"]),
            Route("/remove_offsets", do_remove_offsets, methods=["POST"]),
            Route("/detect_organelles", detect_organelles, methods=["POST"]),
            Route("/chunk_grid", chunk_grid, methods=["POST"]),
            Route(
                "/detect_parallel_patches", detect_parallel_patches, methods=["POST"]
            ),
            Route("/remove_parallel_patches", do_remove_parallel, methods=["POST"]),
            Route("/detect_fusions", detect_fusions, methods=["POST"]),
            Route("/detect_rims", detect_rims, methods=["POST"]),
            Route("/detect_holes", detect_holes, methods=["POST"]),
            Route("/remove_organelles", do_remove_organelles, methods=["POST"]),
            Route("/remove_fusions", do_remove_fusions, methods=["POST"]),
            Route("/detect_fragments", detect_fragments, methods=["POST"]),
            Route("/detect_disconnected", detect_disconnected, methods=["POST"]),
            Route("/detect_soma", detect_soma, methods=["POST"]),
            Route("/detect_gaps", detect_gaps, methods=["POST"]),
            Route("/remove_gaps", do_remove_gaps, methods=["POST"]),
            Route("/remove_fragments", do_remove_fragments, methods=["POST"]),
            Route("/fill_holes", do_fill_holes, methods=["POST"]),
            Route("/merge_selected", do_merge_selected, methods=["POST"]),
            Route("/remove_selected", do_remove_selected, methods=["POST"]),
            Route("/rescue_as_neurite", do_rescue_as_neurite, methods=["POST"]),
            Route("/edit_vertices", edit_vertices, methods=["POST"]),
            Route("/undo", undo_mesh, methods=["POST"]),
            Route("/break_up_mesh", do_break_up_mesh, methods=["POST"]),
            Route("/preprocess", do_preprocess, methods=["POST"]),
            Route("/name_neurite", name_neurite, methods=["POST"]),
            Route("/reassign_preview", reassign_preview, methods=["POST"]),
            Route("/reassign_apply", reassign_apply, methods=["POST"]),
            Route("/reassign_cancel", reassign_cancel, methods=["POST"]),
            Route("/compact_mesh", do_compact_mesh, methods=["POST"]),
            Route("/export_mesh", export_mesh, methods=["GET"]),
            Route("/export_skeleton", export_skeleton, methods=["GET"]),
            Route("/export_organelles", export_organelles, methods=["GET"]),
            Route("/export_mesh_stats", export_mesh_stats, methods=["GET"]),
            Route("/export_soma", export_soma, methods=["GET"]),
            Route("/export_components", export_components, methods=["GET"]),
            Route("/export_neurites", export_neurites, methods=["GET"]),
            Route("/export_discarded", export_discarded, methods=["GET"]),
            Route("/export_annotations", export_annotations, methods=["GET"]),
            Route(
                "/export_annotation_submesh",
                export_annotation_submesh,
                methods=["GET"],
            ),
            Route("/skeletonize", run_skeletonize, methods=["POST"]),
            Route("/bin", get_bin, methods=["POST"]),
            Route("/bin_reassign_preview", bin_reassign_preview, methods=["POST"]),
            Route("/bin_reassign_apply", bin_reassign_apply, methods=["POST"]),
            Route("/bin_split_preview", bin_split_preview, methods=["POST"]),
            Route("/bin_reassign_cancel", bin_reassign_cancel, methods=["POST"]),
            Route("/edge_support", edge_support, methods=["POST"]),
            Route("/diagnose", diagnose, methods=["POST"]),
            Route("/edge_edit_preview", edge_edit_preview, methods=["POST"]),
            Route("/edge_edit_apply", edge_edit_apply, methods=["POST"]),
            Route("/edge_edit_cancel", edge_edit_cancel, methods=["POST"]),
            Route("/shortest_path", shortest_path_endpoint, methods=["POST"]),
            WebSocketRoute("/ws", ws_endpoint),
        ],
    )

    return app


def _has_running_loop() -> bool:
    """Check if we're inside an already-running asyncio event loop (e.g. Jupyter)."""
    import asyncio

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


# Track background server so it can be stopped / replaced
_active_server = None


def stop_viewer():
    """Stop the background viewer server (Jupyter only)."""
    global _active_server
    if _active_server is not None:
        _active_server.should_exit = True
        _active_server = None


def _launch_app(app, *, host: str, port: int, no_browser: bool):
    """Shared launcher: print info, open browser, run uvicorn."""
    global _active_server

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "The viewer requires uvicorn. Install with:\n"
            "  pip install uvicorn[standard]"
        )

    # Stop any previous background server on this port
    if _active_server is not None:
        _active_server.should_exit = True
        _active_server = None

    port_dir = _STATE_DIR / str(port)
    url = f"http://{host}:{port}"
    print("\nSkeliner Viewer")
    print(f"  URL:          {url}")
    print(f"  State file:   {port_dir / 'state.json'}")
    print(f"  Annotations:  {port_dir / 'annotations.json'}")
    print(f"  Camera cmd:   {port_dir / 'camera.json'}")
    print()

    if not no_browser:

        def _open():
            import webbrowser

            webbrowser.open(url)

        threading.Timer(1.5, _open).start()

    if _has_running_loop():
        # Inside Jupyter / IPython — run uvicorn in a daemon thread
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        _active_server = server
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        print("  Viewer running in background. Call sk.plot.stop_viewer() to stop.")
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")


def view(
    mesh_path: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8777,
    no_browser: bool = False,
):
    """Launch the interactive viewer."""
    app = _create_app(mesh_path, port=port)
    _launch_app(app, host=host, port=port, no_browser=no_browser)


def _bbox_from_faces(
    mesh: trimesh.Trimesh,
    faces_idx: np.ndarray,
    centroid: np.ndarray,
) -> list[list[float]] | None:
    """AABB from a face subset, in centroid-shifted coordinates. Returns [[lo], [hi]]."""
    faces_idx = np.asarray(faces_idx, np.int64)
    if faces_idx.size == 0:
        return None
    vidx = np.unique(mesh.faces[faces_idx].ravel())
    if vidx.size == 0:
        return None
    V = mesh.vertices[vidx].astype(np.float32) - centroid
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    return [lo.tolist(), hi.tolist()]


def _bbox_union(
    a: list[list[float]] | None, b: list[list[float]] | None
) -> list[list[float]] | None:
    if a is None:
        return b
    if b is None:
        return a
    lo = np.minimum(a[0], b[0]).tolist()
    hi = np.maximum(a[1], b[1]).tolist()
    return [lo, hi]


def _bbox_segments(box: list[list[float]]) -> list[list[list[float]]]:
    """Convert [[lo_x,lo_y,lo_z],[hi_x,hi_y,hi_z]] to 12 line segments."""
    lo, hi = box
    c = [
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], hi[1], lo[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ]
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (7, 6),
        (7, 5),
        (7, 4),
        (6, 2),
        (6, 3),
        (3, 5),
        (5, 1),
        (2, 4),
        (4, 1),
    ]
    return [[c[a], c[b]] for a, b in edges]


def _as_iter(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _resolve_mesh_colors(mesh_color, n: int) -> list[list[float]]:
    """Resolve mesh_color argument into a list of [r, g, b] per mesh."""
    DEFAULT = [
        [0.55, 0.55, 0.6],
        [0.6, 0.5, 0.5],
        [0.5, 0.6, 0.5],
        [0.5, 0.5, 0.6],
        [0.6, 0.55, 0.5],
    ]
    if mesh_color == "same":
        return [[0.55, 0.55, 0.6]] * n
    if mesh_color is None:
        return [DEFAULT[i % len(DEFAULT)] for i in range(n)]
    # Array-like of RGB tuples (e.g. cm.tab20.colors)
    colors = np.asarray(mesh_color, dtype=np.float64)
    if colors.ndim == 1 and colors.size == 3:
        return [colors.tolist()] * n
    return [colors[i % len(colors)].tolist() for i in range(n)]


def view3d(
    skels=None,
    meshes=None,
    *,
    scale: float | tuple[float, float] = 1.0,
    mesh_color=None,
    mesh_opacity: float = 0.2,
    host: str = "127.0.0.1",
    port: int = 8777,
    no_browser: bool = False,
):
    """Launch viewer with pre-loaded skeletons and/or meshes."""
    skels = _as_iter(skels)
    meshes = _as_iter(meshes)

    # Parse scale
    if isinstance(scale, (list, tuple)):
        skel_scale, mesh_scale = float(scale[0]), float(scale[1])
    else:
        skel_scale = mesh_scale = float(scale)

    # Scale meshes
    scaled_meshes = []
    for m in meshes:
        if mesh_scale != 1.0:
            mc = m.copy()
            mc.vertices = m.vertices * mesh_scale
            scaled_meshes.append(mc)
        else:
            scaled_meshes.append(m)

    # Scale skeletons
    scaled_skels = []
    for s in skels:
        if skel_scale != 1.0:
            import copy

            sc = copy.copy(s)
            sc.nodes = s.nodes * skel_scale
            sc.radii = {k: v * skel_scale for k, v in s.radii.items()}
            scaled_skels.append(sc)
        else:
            scaled_skels.append(s)

    # Resolve colors
    colors = _resolve_mesh_colors(mesh_color, len(scaled_meshes))

    # Shared centroid from all meshes
    all_verts = [m.vertices for m in scaled_meshes if m.vertices.size]
    centroid = (
        np.concatenate(all_verts).mean(axis=0).astype(np.float32)
        if all_verts
        else np.zeros(3, dtype=np.float32)
    )

    primary = scaled_meshes[0] if scaled_meshes else None
    primary_color = colors[0] if colors else None
    extra = {}

    for i, m in enumerate(scaled_meshes[1:], 1):
        buffers = _mesh_to_buffers(m, centroid=centroid)
        extra[f"mesh_{i}"] = {
            "mesh": m,
            "buffers": buffers,
            "color": colors[i],
            "opacity": mesh_opacity,
        }

    skel_pairs = [(f"skel_{i}", s) for i, s in enumerate(scaled_skels)]

    app = _create_app(
        port=port,
        preload_mesh=primary,
        preload_centroid=centroid if len(scaled_meshes) > 1 else None,
        extra_meshes=extra if extra else None,
        preload_skeletons=skel_pairs if skel_pairs else None,
        mesh_color=primary_color,
    )

    _launch_app(app, host=host, port=port, no_browser=no_browser)


def view_contacts(
    A: trimesh.Trimesh,
    B: trimesh.Trimesh,
    contacts,  # ContactSites
    *,
    scale: float = 1.0,
    color_A: tuple[float, float, float] = (0.82, 0.86, 1.00),
    color_B: tuple[float, float, float] = (1.00, 0.85, 0.85),
    sides: str = "A",
    show_aabb: bool = True,
    aabb_mode: str = "union",
    host: str = "127.0.0.1",
    port: int = 8777,
    no_browser: bool = False,
):
    """Visualize two meshes with contact-site overlays in the web viewer."""
    # Shared centroid from both meshes
    all_verts = np.concatenate(
        [A.vertices.astype(np.float32), B.vertices.astype(np.float32)]
    )
    centroid = all_verts.mean(axis=0)

    # Scale meshes
    A_scaled = A.copy()
    B_scaled = B.copy()
    if scale != 1.0:
        A_scaled.vertices = A.vertices * float(scale)
        B_scaled.vertices = B.vertices * float(scale)
        centroid = centroid * float(scale)

    # Buffers for B (extra); A renders through the primary mesh_state
    buf_B = _mesh_to_buffers(B_scaled, centroid=centroid)

    extra_meshes = {
        "mesh_B": {
            "mesh": B_scaled,
            "buffers": buf_B,
            "color": list(color_B),
            "opacity": 1.0,
        },
    }

    # Build contact sites as annotations (highlights + edge_groups)
    SITE_COLORS = [
        [1.0, 0.4, 0.1],  # orange
        [0.2, 0.6, 1.0],  # blue
        [0.1, 0.9, 0.4],  # green
        [0.9, 0.2, 0.8],  # magenta
        [1.0, 0.9, 0.1],  # yellow
        [0.0, 0.85, 0.7],  # teal
        [0.95, 0.5, 0.5],  # salmon
        [0.6, 0.4, 1.0],  # violet
    ]

    s = sides.lower()
    doA = s in ("a", "both")
    doB = s in ("b", "both")
    n_sites = max(
        len(contacts.faces_A) if doA else 0,
        len(contacts.faces_B) if doB else 0,
    )

    highlights = []
    edge_groups = []

    for i in range(n_sites):
        col = SITE_COLORS[i % len(SITE_COLORS)]
        bbox_a, bbox_b = None, None

        if doA and i < len(contacts.faces_A):
            fa = np.asarray(contacts.faces_A[i], np.int64)
            if fa.size > 0:
                bbox_a = _bbox_from_faces(A_scaled, fa, centroid)
                side_label = "A" if (doA and doB) else ""
                highlights.append(
                    {
                        "faces": fa.tolist(),
                        "color": col,
                        "label": f"Site {i}{side_label} ({fa.size} faces)",
                        "meshKey": "primary",
                    }
                )

        if doB and i < len(contacts.faces_B):
            fb = np.asarray(contacts.faces_B[i], np.int64)
            if fb.size > 0:
                bbox_b = _bbox_from_faces(B_scaled, fb, centroid)
                side_label = "B" if (doA and doB) else ""
                highlights.append(
                    {
                        "faces": fb.tolist(),
                        "color": col,
                        "label": f"Site {i}{side_label} ({fb.size} faces)",
                        "meshKey": "mesh_B",
                    }
                )

        # AABB wireframe as edge_group
        if show_aabb:
            box = None
            if aabb_mode == "union":
                box = _bbox_union(bbox_a, bbox_b)
            elif aabb_mode == "split":
                for bb in (bbox_a, bbox_b):
                    if bb is not None:
                        edge_groups.append(
                            {
                                "segments": _bbox_segments(bb),
                                "color": col,
                                "label": f"AABB {i}",
                            }
                        )
                continue  # skip union path
            if box is not None:
                edge_groups.append(
                    {
                        "segments": _bbox_segments(box),
                        "color": col,
                        "label": f"AABB {i}",
                    }
                )

    annotations = {}
    if highlights:
        annotations["highlights"] = highlights
    if edge_groups:
        annotations["edge_groups"] = edge_groups

    contact_state = {
        "sides": sides,
        "nSites": n_sites,
    }

    app = _create_app(
        port=port,
        preload_mesh=A_scaled,
        preload_centroid=centroid,
        extra_meshes=extra_meshes,
        contact_state=contact_state,
        mesh_color=list(color_A),
    )

    # Write contact annotations (after _create_app clears state dir)
    if annotations:
        ann_path = _STATE_DIR / str(port) / "annotations.json"
        ann_path.write_text(json.dumps(annotations), encoding="utf-8")

    _launch_app(app, host=host, port=port, no_browser=no_browser)
