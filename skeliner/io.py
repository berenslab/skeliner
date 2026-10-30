import json
import re
import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np

from ._backend import kernels
from ._core import _bfs_parents
from ._state import rebuild_vert2node
from .dataclass import (
    coincident_merged,
    Discarded,
    Mesh,
    MeshComponents,
    MeshStats,
    Neurites,
    Organelles,
    Skeleton,
    Soma,
)

__all__ = [
    "load_mesh",
    "save_mesh",
    "load_skeleton",
    "save_skeleton",
    "load_skeleton_swc",
    "save_skeleton_swc",
    "load_skeleton_npz",
    "save_skeleton_npz",
    "load_soma",
    "save_soma",
    "load_soma_npz",
    "save_soma_npz",
    "load_organelles",
    "save_organelles",
    "load_organelles_npz",
    "save_organelles_npz",
    "load_mesh_stats",
    "save_mesh_stats",
    "load_mesh_stats_npz",
    "save_mesh_stats_npz",
    "load_neurites",
    "save_neurites",
    "load_discarded",
    "save_discarded",
    "load_components",
    "save_components",
    "load_contact_sites",
    "save_contact_sites",
]

_META_KV = re.compile(r"#\s*([^:]+)\s*:\s*(.+)")  #  key: value
_META_JSON = re.compile(r"#\s*meta\s+(\{.*\})")  #  single-line JSON


# ------------
# --- Mesh ---
# ------------


def _trimesh():
    """trimesh, which reads and writes the formats skeliner does not."""
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "formats other than OBJ (and EyeWire 1's CTM) go through trimesh: "
            "pip install 'skeliner[trimesh]'"
        ) from None
    return trimesh


def load_mesh(filepath: str | Path) -> Mesh:
    """Read a mesh file as a :class:`skeliner.Mesh` of the arrays
    ``trimesh.load_mesh(filepath, process=False)`` reads, with the copies of
    a vertex along a seam merged: a mesh stored in chunks (CloudVolume's
    multi-resolution meshes, for one) or as separate triangles (STL) copies
    the vertices along every seam, which cuts it there. Every face names the
    first vertex at each position that a seam joins it to, the copies are
    dropped as the readers drop any vertex no face names, and a face the
    merge made a copy of another collapses to ``[0, 0, 0]``. Vertices that
    only touch — a seam is a chain of edges both sides of it repeat — stay
    apart: :func:`skeliner.pre.remove_fusions` leaves a cut so, and a mesh it
    cleaned reads back as written. Faces keep their places, so face ids into
    the file stay valid.

    OBJ files of plain vertices and triangles, the form EyeWire 2, MICrONS
    and FlyWire meshes come in and skeliner's own pipeline writes, are read
    by skeliner's own reader on the active path (:class:`skeliner.backend`),
    to the same arrays. An OBJ with anything more (normals, texture
    coordinates, groups, polygons, ...) and every other format are read by
    trimesh, which must then be installed (``skeliner[trimesh]``).

    ``.ctm`` (OpenCTM) has a reader of its own here, but it is only for
    reading EyeWire 1's meshes
    (https://drive.google.com/file/d/1-qmE1sEL3AIGpNfi0w50MEOgm2WIF4_1/view),
    which trimesh cannot read without ``python-openctm`` (x86-64 and Python
    3.11 at most): OpenCTM v5 with RAW, MG1 or MG2 compression, vertices and
    triangles only, no normals, UV maps or attribute maps. Those files store
    each vertex as (z, x, y), the retina's depth first; it is read as
    (x, y, z), the depth in z as in the other data.
    """
    filepath = Path(filepath)

    # OBJ and EyeWire 1's CTM on the active path; the readers decline (None)
    # whatever else such a file holds, and trimesh reads it
    reader = {".obj": "read_obj", ".ctm": "read_ctm"}.get(filepath.suffix.lower())
    if reader is not None:
        found = getattr(kernels.io, reader)(str(filepath))
        if found is not None:
            vertices, faces = found
            if reader == "read_ctm":
                # (z, x, y) to (x, y, z): a rotation, so the faces keep facing out
                vertices = np.ascontiguousarray(vertices[:, [1, 2, 0]])
            return _without_copies(vertices, faces)
    read = _trimesh().load_mesh(filepath, process=False)
    read = Mesh(read.vertices, read.faces)
    return _without_copies(read.vertices, read.faces)


def _without_copies(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """The mesh of a file's arrays with the copies of a vertex along a seam
    merged and dropped (the vertices the merge left no face naming), the
    faces renumbered."""
    merged = coincident_merged(vertices, faces)
    if merged is faces:
        return Mesh._adopt(vertices, faces)
    named = np.zeros(len(vertices), dtype=bool)
    named[faces.ravel()] = True
    still = np.zeros(len(vertices), dtype=bool)
    still[merged.ravel()] = True
    keep = ~named | still
    renumbered = np.cumsum(keep) - 1
    return Mesh._adopt(np.ascontiguousarray(vertices[keep]), renumbered[merged])


def save_mesh(mesh, path: str | Path) -> None:
    """Write a mesh to disk, the format by the file's suffix.

    A :class:`skeliner.Mesh` (or anything with ``vertices`` and ``faces``) goes
    to OBJ through skeliner's own writer on the active path, as trimesh
    writes a plain mesh, and to every other format (ply, stl, glb, ...)
    through trimesh, which must then be installed. A mesh with an
    ``export`` of its own — a ``trimesh.Trimesh`` — writes itself.

    An OBJ keeps every vertex, but :func:`load_mesh` reads back only the
    vertices a face names, as trimesh reads OBJ, and numbers the rest on: a
    mesh the ``remove_*`` stages left vertices no face names (``preprocess``
    without ``compact=True``) reads back with other vertex ids, and vertex
    ids saved beside it — a soma's ``verts`` — would name other vertices.
    Writing such a mesh to OBJ warns; :func:`skeliner.pre.compact_mesh` it
    first, with its components, when vertex ids are saved beside it.
    """
    path = Path(path)
    if not isinstance(mesh, Mesh) and hasattr(mesh, "export"):
        mesh.export(str(path))
        return
    mesh = Mesh.of(mesh)
    if path.suffix.lower() == ".obj":
        named = np.zeros(len(mesh.vertices), dtype=bool)
        named[mesh.faces.ravel()] = True
        if not named.all():
            warnings.warn(
                f"{path}: {int((~named).sum()):,} vertices no face names are"
                " written but not read back, so the vertex ids from"
                f" {int(np.argmin(named)):,} on change when the file is read;"
                " compact the mesh first when vertex ids are saved beside it",
                stacklevel=2,
            )
        kernels.io.write_obj(str(path), mesh.vertices, mesh.faces)
    else:
        mesh.to_trimesh().export(str(path))


# -----------
# --- SWC ---
# -----------


def load_skeleton_swc(
    path: str | Path,
    *,
    scale: float = 1.0,
    keep_types: Iterable[int] | None = None,
) -> Skeleton:
    """
    Load an SWC file into a :class:`Skeleton`.

    Because SWC stores just a point-list, the soma is reconstructed *ad hoc*
    as a **sphere** centred on node 0 with radius equal to that node’s radius.

    Parameters
    ----------
    path
        SWC file path.
    scale
        Uniform scale factor applied to coordinates *and* radii.
    keep_types
        Optional set/sequence of SWC type codes to keep (e.g. ``{1, 2, 3}``).
        ``None`` ⇒ keep everything.

    Returns
    -------
    Skeleton
        Fully initialised skeleton.  ``soma.verts`` is ``None`` because the
        SWC format has no surface-vertex concept.

    Raises
    ------
    ValueError
        If the file contains no nodes after filtering.
    """
    path = Path(path)

    ids: List[int] = []
    xyz: List[List[float]] = []
    radii: List[float] = []
    parent: List[int] = []
    ntype: List[int] = []

    meta = {}

    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            # ------- 1) try single-line JSON -------------------------
            j = _META_JSON.match(line)
            if j:
                try:
                    meta.update(json.loads(j.group(1)))
                except json.JSONDecodeError:
                    pass
                continue

            # ------- 2) try simple "key: value" ----------------------
            m = _META_KV.match(line)
            if m:
                key, val = m.groups()
                meta[key.strip()] = val.strip()
                continue

            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            _id, _type = int(float(parts[0])), int(float(parts[1]))
            if keep_types is not None and _type not in keep_types:
                continue
            ids.append(_id)
            xyz.append([float(parts[2]), float(parts[3]), float(parts[4])])
            radii.append(float(parts[5]))
            parent.append(int(float(parts[6])))
            ntype.append(_type)

    if not ids:
        raise ValueError(f"No usable nodes found in {path}")

    # --- core arrays ----------------------------------------------------
    nodes_arr = np.asarray(xyz, dtype=np.float64) * scale
    radii_arr = np.asarray(radii, dtype=np.float64) * scale
    radii_dict = {
        "median": radii_arr,
        "mean": radii_arr.copy(),
        "trim": radii_arr.copy(),
    }
    ntype_arr = np.asarray(ntype, dtype=np.int8)
    # --- edges (parent IDs → 0-based indices) ---------------------------
    id_map = {old: new for new, old in enumerate(ids)}
    edges = [
        (id_map[i], id_map[p])
        for i, p in zip(ids, parent, strict=True)
        if p != -1 and p in id_map
    ]
    edges_arr = np.asarray(edges, dtype=np.int64)
    if edges_arr.size:
        edges_arr = np.sort(edges_arr, axis=1)
        edges_arr = np.unique(edges_arr, axis=0)

    # --- minimal spherical soma around node 0 --------------------------
    soma_centre = nodes_arr[0].copy()
    soma_radius = radii_arr[0]
    soma = Soma.from_sphere(soma_centre, soma_radius, verts=None)

    # --- build and return Skeleton -------------------------------------
    return Skeleton(
        nodes=nodes_arr,
        radii=radii_dict,
        edges=edges_arr,
        ntype=ntype_arr,
        soma=soma,
        node2verts=None,
        vert2node=None,
        meta=meta,
    )


def save_skeleton_swc(
    skeleton,
    path: str | Path,
    include_header: bool = True,
    include_meta: bool = True,
    scale: float = 1.0,
    radius_metric: str | None = None,
    axis_order: tuple[int, int, int] | str = (0, 1, 2),
) -> None:
    """Write the skeleton to SWC.

    The first node (index 0) is written as type -1 if ntype is None and acts as the
    root of the morphology tree. Parent IDs are therefore 1‑based to
    comply with the SWC format.

    Parameters
    ----------
    path
        Output filename.
    include_header
        Prepend the canonical SWC header line if *True*.
    scale
        Unit conversion factor applied to *both* coordinates and radii when
        writing; useful e.g. for nm→µm conversion.
    """

    # --- normalise axis_order ------------------------------------------
    if isinstance(axis_order, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        try:
            axis_order = tuple(axis_map[c.lower()] for c in axis_order)
        except KeyError:
            raise ValueError("axis_order string must be a permutation of 'xyz'")
    axis_order = tuple(map(int, axis_order))
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError("axis_order must be a permutation of (0,1,2)")

    # --- check suffix and convert path -------------------------------
    path = Path(path)

    # add .swc to the path if not present
    if not path.suffix:
        path = path.with_suffix(".swc")

    # --- prepare arrays -----------------------------------------------
    parent = _bfs_parents(skeleton.edges, len(skeleton.nodes), root=0)
    nodes = skeleton.nodes
    if radius_metric is None:
        radii = skeleton.r
    else:
        if radius_metric not in skeleton.radii:
            raise ValueError(f"Unknown radius estimator '{radius_metric}'")
        radii = skeleton.radii[radius_metric]

    # --- Node types (guarantee soma = 1, others = 3 as default if not set)
    if skeleton.ntype is not None:
        ntype = skeleton.ntype.astype(int, copy=False)
    else:
        ntype = np.full(len(nodes), 0, dtype=int)  # default to "unknown"
        if len(ntype):
            ntype[0] = -1  # default to "root"

    # --- write SWC file -----------------------------------------------
    with path.open("w", encoding="utf8") as fh:
        if include_meta and skeleton.meta:
            blob = json.dumps(skeleton.meta, separators=(",", ":"), ensure_ascii=False)
            fh.write(f"# meta {blob}\n")

        if include_header:
            fh.write("# id type x y z radius parent\n")

        for idx, (coord, r, pa, t) in enumerate(
            zip(nodes[:, axis_order] * scale, radii * scale, parent, ntype), start=1
        ):
            # ensure soma has type -1 or +1
            fh.write(
                f"{idx} {int(-1 if ((idx == 1) and t not in [-1, 1]) else t)} "
                f"{coord[0]} {coord[1]} {coord[2]} {r} "
                f"{(pa + 1) if pa != -1 else -1}\n"
            )


# -----------
# --- npz ---
# -----------


def load_skeleton_npz(path: str | Path) -> Skeleton:
    """
    Load a Skeleton that was written with `Skeleton.to_npz` / `save_skeleton_npz`.
    """
    path = Path(path)

    with np.load(path, allow_pickle=True) as z:
        nodes = z["nodes"].astype(np.float64)
        edges = z["edges"].astype(np.int64)

        # radii dict  (keys start with 'r_')
        radii = {k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("r_")}

        # node types (optional in older archives)
        if "ntype" in z:
            ntype = z["ntype"].astype(np.int8)
        else:
            ntype = np.full(len(nodes), 0, dtype=np.int8)  # default to "unknown"
            if len(ntype):
                ntype[0] = -1  # default to "root"

        # reconstruct ragged node2verts
        idx = z["node2verts_idx"].astype(np.int64)
        off = z["node2verts_off"].astype(np.int64)
        node2verts = [idx[off[i] : off[i + 1]] for i in range(len(off) - 1)]

        vert2node = rebuild_vert2node(node2verts) or {}

        soma = Soma(
            center=z["soma_centre"],
            axes=z["soma_axes"],
            R=z["soma_R"],
            verts=(z["soma_verts"].astype(np.int64) if "soma_verts" in z else None),
        )

        # ----------- NEW: arbitrary, user-defined metadata ---------------
        extra = {}
        if "extra" in z.files:
            # stored as length-1 object array; .item() unwraps the dict
            extra = z["extra"].item()

        meta = {}
        if "meta" in z.files:
            # stored as length-1 object array; .item() unwraps the dict
            meta = z["meta"].item()

        # older archives also hold a pickled scipy KD-tree ("kdtree_nodes"),
        # left unread
        adjacency = None
        if "neighbors_idx" in z.files and "neighbors_off" in z.files:
            adjacency = (
                z["neighbors_off"].astype(np.int64),
                z["neighbors_idx"].astype(np.int64),
            )

    skel = Skeleton(
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
        soma=soma,
        node2verts=node2verts,
        vert2node=vert2node,
        extra=extra,
        meta=meta,
    )
    if adjacency is not None:
        skel._adjacency = adjacency
    return skel


def save_skeleton_npz(
    skeleton: Skeleton,
    path: str | Path,
    *,
    compress: bool = True,
    cache_kdtree: bool = True,
) -> None:
    """
    Write the skeleton to a compressed `.npz` archive.

    Parameters
    ----------
    compress
        When *True* use :func:`numpy.savez_compressed`.
    cache_kdtree
        When *True* (default) ensure the adjacency cache (each node's
        neighbours) is built and embedded in the archive for fast reloads.
        A KD-tree is no longer stored: :func:`skeliner.dx.distance` builds
        its own index in milliseconds.
    """
    path = Path(path)

    # add .npz to the path if not present
    if not path.suffix:
        path = path.with_suffix(".npz")

    if cache_kdtree:
        skeleton._ensure_adjacency()

    # radii_<name>  : one array per estimator
    radii_flat = {f"r_{k}": v for k, v in skeleton.radii.items()}

    # ragged node2verts  → index + offset
    if skeleton.node2verts is not None:
        n2v_idx = np.concatenate(skeleton.node2verts)
        n2v_off = np.cumsum([0, *map(len, skeleton.node2verts)])

        # Determine smallest dtype for n2v_idx
        if len(n2v_idx) > 0:
            idx_min, idx_max = n2v_idx.min(), n2v_idx.max()
            if idx_min >= 0 and idx_max <= np.iinfo(np.uint16).max:
                n2v_idx = n2v_idx.astype(np.uint16)
            elif (
                idx_min < 0
                and idx_min >= np.iinfo(np.int16).min
                and idx_max <= np.iinfo(np.int16).max
            ):
                n2v_idx = n2v_idx.astype(np.int16)
            elif idx_min >= 0 and idx_max <= np.iinfo(np.uint32).max:
                n2v_idx = n2v_idx.astype(np.uint32)
            elif (
                idx_min < 0
                and idx_min >= np.iinfo(np.int32).min
                and idx_max <= np.iinfo(np.int32).max
            ):
                n2v_idx = n2v_idx.astype(np.int32)
            else:
                n2v_idx = n2v_idx.astype(np.int64)
        else:
            n2v_idx = n2v_idx.astype(np.uint16)

        # Determine smallest dtype for n2v_off
        off_min, off_max = n2v_off.min(), n2v_off.max()
        if off_min >= 0 and off_max <= np.iinfo(np.uint16).max:
            n2v_off = n2v_off.astype(np.uint16)
        elif (
            off_min < 0
            and off_min >= np.iinfo(np.int16).min
            and off_max <= np.iinfo(np.int16).max
        ):
            n2v_off = n2v_off.astype(np.int16)
        elif off_min >= 0 and off_max <= np.iinfo(np.uint32).max:
            n2v_off = n2v_off.astype(np.uint32)
        elif (
            off_min < 0
            and off_min >= np.iinfo(np.int32).min
            and off_max <= np.iinfo(np.int32).max
        ):
            n2v_off = n2v_off.astype(np.int32)
        else:
            n2v_off = n2v_off.astype(np.int64)
    else:
        n2v_idx = np.array([], dtype=np.uint16)
        n2v_off = np.array([0], dtype=np.uint16)
    # ----------- NEW: persist the metadata dict -------------------------
    # We wrap it in a 0-D object array because np.savez can only store
    # ndarrays — this keeps the archive a single *.npz* with no sidecars.
    extra = {"extra": np.array(skeleton.extra, dtype=object)}
    meta = {"meta": np.array(skeleton.meta, dtype=object)} if skeleton.meta else {}
    tree_payload = {}
    if skeleton._adjacency is not None:
        offsets, data = skeleton._adjacency
        tree_payload["neighbors_idx"] = data.astype(np.int64, copy=False)
        tree_payload["neighbors_off"] = offsets.astype(np.int64, copy=False)

    save_fun = np.savez_compressed if compress else np.savez

    save_fun(
        path,
        nodes=skeleton.nodes,
        edges=skeleton.edges,
        ntype=skeleton.ntype
        if skeleton.ntype is not None
        else np.array([], dtype=np.int8),
        soma_centre=skeleton.nodes[0],
        soma_axes=skeleton.soma.axes,
        soma_R=skeleton.soma.R,
        soma_verts=skeleton.soma.verts
        if skeleton.soma.verts is not None
        else np.array([], dtype=np.int64),
        node2verts_idx=n2v_idx,
        node2verts_off=n2v_off,
        **radii_flat,
        **extra,
        **meta,
        **tree_payload,
    )


# ------------
# --- Soma ---
# ------------


def save_soma_npz(soma: Soma, path: str | Path, *, compress: bool = True) -> None:
    """Write a standalone :class:`Soma` to a compressed ``.npz`` archive."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")

    payload = dict(
        center=soma.center,
        axes=soma.axes,
        R=soma.R,
    )
    if soma.verts is not None:
        payload["verts"] = soma.verts.astype(np.int64, copy=False)
    if soma.nucleus is not None:
        nuc = soma.nucleus
        payload["nucleus_center"] = np.asarray(nuc["center"], dtype=np.float64)
        payload["nucleus_peak_r"] = np.array(nuc["peak_r"], dtype=np.float64)
        payload["nucleus_z_range"] = np.array(nuc["z_range"], dtype=np.float64)
        payload["nucleus_slices"] = np.asarray(nuc["slices"], dtype=np.float64)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **payload)


def load_soma_npz(path: str | Path) -> Soma:
    """Load a :class:`Soma` written by :func:`save_soma_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        nucleus = None
        if "nucleus_center" in z:
            nucleus = {
                "center": z["nucleus_center"].astype(np.float64),
                "peak_r": float(z["nucleus_peak_r"]) if "nucleus_peak_r" in z else 0.0,
                "z_range": tuple(z["nucleus_z_range"].astype(np.float64))
                if "nucleus_z_range" in z
                else (0.0, 0.0),
                "slices": z["nucleus_slices"].astype(np.float64)
                if "nucleus_slices" in z
                else np.empty((0, 4)),
            }
        return Soma(
            center=z["center"].astype(np.float64),
            axes=z["axes"].astype(np.float64),
            R=z["R"].astype(np.float64),
            verts=z["verts"].astype(np.int64) if "verts" in z else None,
            nucleus=nucleus,
        )


# ---------------------------
# --- Organelles NPZ I/O ---
# ---------------------------


def save_organelles_npz(
    organelles: Organelles,
    path: str | Path,
    *,
    compress: bool = True,
) -> None:
    """Write an :class:`Organelles` to a compressed ``.npz`` archive.

    Only stores the organelle masks.  See :func:`save_mesh_stats_npz`
    for persisting the associated :class:`MeshStats` separately.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")

    payload: dict[str, np.ndarray] = {
        "pocket": np.asarray(organelles.pocket, dtype=bool),
        "isolated": np.asarray(organelles.isolated, dtype=bool),
        "expanded": np.asarray(organelles.expanded, dtype=bool),
        "manual": np.asarray(organelles.manual, dtype=bool),
    }

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **payload)


def load_organelles_npz(path: str | Path) -> Organelles:
    """Load an :class:`Organelles` written by :func:`save_organelles_npz`.

    Older files that bundled ``outward_dots``/``face_comp``/``main_ci``
    are still readable — those fields are silently ignored.  Use
    :func:`load_mesh_stats_npz` if you need the cached mesh statistics.
    Files written before ``expanded``/``manual`` existed load with those
    masks all False.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        pocket = z["pocket"].astype(bool)
        return Organelles(
            pocket=pocket,
            isolated=z["isolated"].astype(bool),
            expanded=z["expanded"].astype(bool)
            if "expanded" in z
            else np.zeros_like(pocket),
            manual=z["manual"].astype(bool) if "manual" in z else np.zeros_like(pocket),
        )


def save_mesh_stats_npz(
    mesh_stats: MeshStats,
    path: str | Path,
    *,
    compress: bool = True,
) -> None:
    """Write a :class:`MeshStats` to a compressed ``.npz`` archive.

    Persists ``outward_dots``, ``face_comp``, and ``main_ci`` so that
    expensive recomputation can be skipped on reload.  The lifecycle
    is tied to the mesh — if you mutate the mesh after saving, the
    stored stats become stale.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")

    payload: dict[str, np.ndarray] = {
        "outward_dots": np.asarray(mesh_stats.outward_dots),
        "face_comp": np.asarray(mesh_stats.face_comp),
        "main_ci": np.array(mesh_stats.main_ci),
    }

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **payload)


def load_mesh_stats_npz(
    path: str | Path,
    mesh=None,
) -> MeshStats:
    """Load a :class:`MeshStats` written by :func:`save_mesh_stats_npz`.

    Parameters
    ----------
    path : str or Path
        Source archive.
    mesh : Mesh or None, optional
        If provided, the loaded face count is validated against
        ``len(mesh.faces)``.  A mismatch raises :class:`ValueError`
        because the cached stats no longer match the mesh.

    Notes
    -----
    Face count is necessary but not sufficient — same face count can
    still mean different topology after edits.  Treat the validation
    as a sanity check, not a guarantee.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        outward_dots = z["outward_dots"]
        face_comp = z["face_comp"]
        main_ci = int(z["main_ci"])

    if mesh is not None:
        n_faces = len(mesh.faces)
        if len(outward_dots) != n_faces or len(face_comp) != n_faces:
            raise ValueError(
                f"Loaded mesh_stats face count "
                f"({len(outward_dots)}) does not match mesh "
                f"({n_faces}); the cached stats are stale."
            )

    return MeshStats(
        outward_dots=outward_dots,
        face_comp=face_comp,
        main_ci=main_ci,
    )


# --------------------------------
# --- Neurites / Discarded NPZ ---
# --------------------------------


def save_neurites_npz(
    neurites: Neurites,
    path: str | Path,
    *,
    compress: bool = True,
) -> None:
    """Write a :class:`Neurites` to a compressed ``.npz`` archive."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    payload: dict[str, np.ndarray] = {
        "n": np.array(len(neurites.components)),
    }
    for i, comp in enumerate(neurites.components):
        payload[f"c{i}"] = np.asarray(comp, dtype=np.int64)
    if neurites.labels is not None:
        payload["labels"] = np.asarray(neurites.labels, dtype=np.str_)
        payload["swc_types"] = np.asarray(neurites.swc_types, dtype=np.int16)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **payload)


def load_neurites_npz(path: str | Path) -> Neurites:
    """Load a :class:`Neurites` written by :func:`save_neurites_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        components = [z[f"c{i}"].astype(np.int64) for i in range(n)]
        labels = [str(x) for x in z["labels"]] if "labels" in z else None
        swc_types = [int(x) for x in z["swc_types"]] if "swc_types" in z else None
    return Neurites(components, labels=labels, swc_types=swc_types)


def save_discarded_npz(
    discarded: Discarded,
    path: str | Path,
    *,
    compress: bool = True,
) -> None:
    """Write a :class:`Discarded` to a compressed ``.npz`` archive."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    payload: dict[str, np.ndarray] = {
        "n": np.array(len(discarded.components)),
    }
    for i, comp in enumerate(discarded.components):
        payload[f"c{i}"] = np.asarray(comp, dtype=np.int64)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **payload)


def load_discarded_npz(path: str | Path) -> Discarded:
    """Load a :class:`Discarded` written by :func:`save_discarded_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        components = [z[f"c{i}"].astype(np.int64) for i in range(n)]
    return Discarded(components)


# ----------------------------------
# --- MeshComponents NPZ I/O ---
# ----------------------------------


def save_components_npz(
    components: MeshComponents,
    path: str | Path,
    *,
    compress: bool = True,
) -> None:
    """Write a :class:`MeshComponents` to a single ``.npz`` archive.

    Packs soma, organelles, neurites, and discarded into one file.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")

    payload: dict[str, np.ndarray] = {}

    # Soma
    if components.soma is not None:
        s = components.soma
        payload["soma_center"] = s.center
        payload["soma_axes"] = s.axes
        payload["soma_R"] = s.R
        if s.verts is not None:
            payload["soma_verts"] = s.verts.astype(np.int64, copy=False)
        if s.nucleus is not None:
            nuc = s.nucleus
            payload["soma_nucleus_center"] = np.asarray(nuc["center"], dtype=np.float64)
            payload["soma_nucleus_peak_r"] = np.array(nuc["peak_r"], dtype=np.float64)
            payload["soma_nucleus_z_range"] = np.array(nuc["z_range"], dtype=np.float64)
            payload["soma_nucleus_slices"] = np.asarray(nuc["slices"], dtype=np.float64)

    # Organelles
    organelles = components.organelles
    payload["organelles_pocket"] = np.asarray(organelles.pocket, dtype=bool)
    payload["organelles_isolated"] = np.asarray(organelles.isolated, dtype=bool)
    payload["organelles_expanded"] = np.asarray(organelles.expanded, dtype=bool)
    payload["organelles_manual"] = np.asarray(organelles.manual, dtype=bool)

    # Neurites
    payload["n_neurites"] = np.array(len(components.neurites.components))
    for i, comp in enumerate(components.neurites.components):
        payload[f"neurite_{i}"] = np.asarray(comp, dtype=np.int64)
    if components.neurites.labels is not None:
        payload["neurite_labels"] = np.asarray(
            components.neurites.labels, dtype=np.str_
        )
        payload["neurite_swc_types"] = np.asarray(
            components.neurites.swc_types, dtype=np.int16
        )

    # Discarded
    payload["n_discarded"] = np.array(len(components.discarded.components))
    for i, comp in enumerate(components.discarded.components):
        payload[f"discarded_{i}"] = np.asarray(comp, dtype=np.int64)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **payload)


def load_components_npz(path: str | Path) -> MeshComponents:
    """Load a :class:`MeshComponents` written by :func:`save_components_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        # Soma
        if "soma_center" in z:
            nucleus = None
            if "soma_nucleus_center" in z:
                nucleus = {
                    "center": z["soma_nucleus_center"].astype(np.float64),
                    "peak_r": float(z["soma_nucleus_peak_r"]),
                    "z_range": tuple(z["soma_nucleus_z_range"].astype(np.float64)),
                    "slices": z["soma_nucleus_slices"].astype(np.float64),
                }
            soma = Soma(
                center=z["soma_center"].astype(np.float64),
                axes=z["soma_axes"].astype(np.float64),
                R=z["soma_R"].astype(np.float64),
                verts=z["soma_verts"].astype(np.int64) if "soma_verts" in z else None,
                nucleus=nucleus,
            )
        else:
            soma = None

        # Organelles
        organelles = Organelles(
            pocket=z["organelles_pocket"].astype(bool),
            isolated=z["organelles_isolated"].astype(bool),
            expanded=z["organelles_expanded"].astype(bool),
            manual=z["organelles_manual"].astype(bool)
            if "organelles_manual" in z
            else None,
        )

        # Neurites
        n_n = int(z["n_neurites"])
        neurites = Neurites(
            [z[f"neurite_{i}"].astype(np.int64) for i in range(n_n)],
            labels=(
                [str(x) for x in z["neurite_labels"]] if "neurite_labels" in z else None
            ),
            swc_types=(
                [int(x) for x in z["neurite_swc_types"]]
                if "neurite_swc_types" in z
                else None
            ),
        )

        # Discarded
        n_d = int(z["n_discarded"])
        discarded = Discarded(
            [z[f"discarded_{i}"].astype(np.int64) for i in range(n_d)]
        )

    return MeshComponents(
        soma=soma, organelles=organelles, neurites=neurites, discarded=discarded
    )


# --------------------------
# --- Mesh Contact Sites ---
# --------------------------

# --- shared helpers (feel free to move to serde.py if you prefer) -----


def _json_encode_meta(meta: dict) -> np.ndarray:
    import json

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return np.array(json.dumps(meta, default=default, sort_keys=True), dtype="U")


def _json_decode_meta(meta_json: np.ndarray) -> dict:
    import json

    s = meta_json.item() if isinstance(meta_json, np.ndarray) else str(meta_json)
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": s}


def _pack_ragged_1d(
    arrs: list[np.ndarray], dtype=np.int64
) -> tuple[np.ndarray, np.ndarray]:
    if not arrs:
        return np.empty(0, dtype), np.zeros(1, np.int64)
    lengths = np.fromiter((len(a) for a in arrs), np.int64, count=len(arrs))
    ptr = np.concatenate(([0], np.cumsum(lengths)))
    data = np.concatenate(arrs) if ptr[-1] else np.empty(0, dtype)
    return data.astype(dtype, copy=False), ptr


def _unpack_ragged_1d(data: np.ndarray, ptr: np.ndarray) -> list[np.ndarray]:
    return [data[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]


def _pack_ragged_2d_fixed(
    arrs: list[np.ndarray], width: int, dtype=np.float64
) -> tuple[np.ndarray, np.ndarray]:
    if not arrs:
        return np.empty((0, width), dtype), np.zeros(1, np.int64)
    rows = np.fromiter((a.shape[0] for a in arrs), np.int64, count=len(arrs))
    ptr = np.concatenate(([0], np.cumsum(rows)))
    stacked = np.vstack(arrs) if ptr[-1] else np.empty((0, width), dtype)
    return stacked.astype(dtype, copy=False), ptr


def _unpack_ragged_2d_fixed(stacked: np.ndarray, ptr: np.ndarray) -> list[np.ndarray]:
    return [stacked[ptr[i] : ptr[i + 1], :] for i in range(len(ptr) - 1)]


# --- public API for ContactSitesResult ---------------------------------


def _save_stats_to_payload(stats: dict | None, prefix: str, payload: dict) -> None:
    """Pack dict of 1D arrays into NPZ payload using '<prefix>_keys' +
    '<prefix>__{key}'."""
    if not stats:
        return
    keys = list(stats.keys())
    payload[f"{prefix}_keys"] = np.array(keys, dtype="U")
    for k in keys:
        payload[f"{prefix}__{k}"] = np.asarray(stats[k], np.float64)


def _load_stats_from_npz(z, prefix: str) -> dict | None:
    """Inverse of _save_stats_to_payload; returns dict or None if not found."""
    keys_name = f"{prefix}_keys"
    if keys_name not in z:
        return None
    keys = list(z[keys_name].astype("U"))
    out = {}
    for k in keys:
        arr_name = f"{prefix}__{k}"
        if arr_name in z:
            out[k] = z[arr_name].astype(np.float64, copy=False)
    return out if out else None


def save_contact_sites_npz(res, path: str | Path, *, compress: bool = True) -> None:
    """
    Serialize ContactSitesResult to a pickle-free NPZ:
      - faces_* and pairs_AB are ragged → packed as (data, ptr)
      - meta is JSON string in 'meta_json'
      - schema tag in '_schema'
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")

    # Ragged lists
    fa_data, fa_ptr = _pack_ragged_1d(res.faces_A, dtype=np.int64)
    fb_data, fb_ptr = _pack_ragged_1d(res.faces_B, dtype=np.int64)
    if res.pairs_AB is None:
        pairs_flag = np.array(0, np.uint8)
        pab_data = np.empty((0, 3), np.float64)
        pab_ptr = np.zeros(1, np.int64)
    else:
        pairs_flag = np.array(1, np.uint8)
        pab_data, pab_ptr = _pack_ragged_2d_fixed(
            res.pairs_AB, width=3, dtype=np.float64
        )

    payload = dict(
        faces_A_data=fa_data,
        faces_A_ptr=fa_ptr,
        faces_B_data=fb_data,
        faces_B_ptr=fb_ptr,
        pairs_flag=pairs_flag,
        pairs_AB_data=pab_data,
        pairs_AB_ptr=pab_ptr,
        area_A=np.asarray(res.area_A, np.float64),
        area_B=np.asarray(res.area_B, np.float64),
        area_mean=np.asarray(res.area_mean, np.float64),
        seeds_A=np.asarray(res.seeds_A, np.float64),
        seeds_B=np.asarray(res.seeds_B, np.float64),
        bbox_A=np.asarray(res.bbox_A, np.float64),
        bbox_B=np.asarray(res.bbox_B, np.float64),
        bbox=np.asarray(res.bbox, np.float64),
        meta_json=_json_encode_meta(res.meta),
        _schema=np.array("ContactSitesResult@3", dtype="U"),
    )
    if getattr(res, "kind", None) is not None:
        payload["kind"] = np.asarray(res.kind, dtype="U")
    # Optional stats: flattened into many arrays with explicit key list per group
    _save_stats_to_payload(getattr(res, "stats_A", None), "stats_A", payload)
    _save_stats_to_payload(getattr(res, "stats_B", None), "stats_B", payload)
    _save_stats_to_payload(getattr(res, "stats_pair", None), "stats_pair", payload)
    _save_stats_to_payload(getattr(res, "geometry", None), "geometry", payload)
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def load_contact_sites_npz(path: str | Path):
    """
    Read a ContactSitesResult written by save_contact_sites_npz.
    Returns a ContactSitesResult instance.
    """
    from .pair import ContactSites  # local import avoids cycles

    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        faces_A = _unpack_ragged_1d(z["faces_A_data"], z["faces_A_ptr"])
        faces_B = _unpack_ragged_1d(z["faces_B_data"], z["faces_B_ptr"])
        if "pairs_flag" in z and int(z["pairs_flag"]) == 1:
            pairs_AB = _unpack_ragged_2d_fixed(z["pairs_AB_data"], z["pairs_AB_ptr"])
        else:
            pairs_AB = None

        meta = _json_decode_meta(z["meta_json"]) if "meta_json" in z else {}

        # Load optional stats (v2); else try meta['stats_*']
        stats_A = _load_stats_from_npz(z, "stats_A")
        stats_B = _load_stats_from_npz(z, "stats_B")
        stats_pair = _load_stats_from_npz(z, "stats_pair")
        geometry = _load_stats_from_npz(z, "geometry")
        kind = z["kind"].astype(str) if "kind" in z else None

        if stats_A is None and isinstance(meta.get("stats_A"), dict):
            stats_A = {k: np.asarray(v, np.float64) for k, v in meta["stats_A"].items()}
        if stats_B is None and isinstance(meta.get("stats_B"), dict):
            stats_B = {k: np.asarray(v, np.float64) for k, v in meta["stats_B"].items()}
        if stats_pair is None and isinstance(meta.get("stats_pair"), dict):
            stats_pair = {
                k: np.asarray(v, np.float64) for k, v in meta["stats_pair"].items()
            }

        return ContactSites(
            faces_A=faces_A,
            faces_B=faces_B,
            area_A=z["area_A"].astype(np.float64, copy=False),
            area_B=z["area_B"].astype(np.float64, copy=False),
            area_mean=z["area_mean"].astype(np.float64, copy=False),
            seeds_A=z["seeds_A"].astype(np.float64, copy=False),
            seeds_B=z["seeds_B"].astype(np.float64, copy=False),
            pairs_AB=pairs_AB,
            bbox_A=z["bbox_A"].astype(np.float64, copy=False)
            if "bbox_A" in z  # backward compatibility
            else np.array([]),
            bbox_B=z["bbox_B"].astype(np.float64, copy=False)
            if "bbox_B" in z
            else np.array([]),
            bbox=z["bbox"].astype(np.float64, copy=False)
            if "bbox" in z
            else np.array([]),
            meta=meta,
            stats_A=stats_A,
            stats_B=stats_B,
            stats_pair=stats_pair,
            kind=kind,
            geometry=geometry,
        )


# Backward-compat wrappers — will be removed in a future release.


def _deprecated_alias(old_name: str, new_func):
    import functools

    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name}() is deprecated, use {new_func.__name__}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_func(*args, **kwargs)

    return wrapper


def save_skeleton(skeleton, path: str | Path, **kwargs) -> None:
    """Save a skeleton to ``.swc`` or ``.npz``.

    The format is inferred from the file extension.
    Extra *kwargs* are forwarded to the format-specific
    saver (e.g. ``scale=`` for SWC).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".swc":
        return save_skeleton_swc(skeleton, path, **kwargs)
    if ext == ".npz":
        return save_skeleton_npz(skeleton, path, **kwargs)
    raise ValueError(
        f"Cannot infer skeleton format from extension '{ext}'. Use .swc or .npz."
    )


def load_skeleton(path: str | Path, **kwargs) -> "Skeleton":
    """Load a skeleton from ``.swc`` or ``.npz``.

    The format is inferred from the file extension.
    Extra *kwargs* are forwarded to the format-specific
    loader (e.g. ``scale=`` for SWC).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".swc":
        return load_skeleton_swc(path, **kwargs)
    if ext == ".npz":
        return load_skeleton_npz(path, **kwargs)
    raise ValueError(
        f"Cannot infer skeleton format from extension '{ext}'. Use .swc or .npz."
    )


# -- short aliases (drop the _npz / _swc suffix) ----------

load_soma = load_soma_npz
load_organelles = load_organelles_npz
load_mesh_stats = load_mesh_stats_npz
load_neurites = load_neurites_npz
load_discarded = load_discarded_npz
load_components = load_components_npz
load_contact_sites = load_contact_sites_npz

save_soma = save_soma_npz
save_organelles = save_organelles_npz
save_mesh_stats = save_mesh_stats_npz
save_neurites = save_neurites_npz
save_discarded = save_discarded_npz
save_components = save_components_npz
save_contact_sites = save_contact_sites_npz


# -- deprecated aliases ------------------------------------

load_swc = _deprecated_alias("load_swc", load_skeleton_swc)
to_swc = _deprecated_alias("to_swc", save_skeleton_swc)
load_npz = _deprecated_alias("load_npz", load_skeleton_npz)
to_npz = _deprecated_alias("to_npz", save_skeleton_npz)
