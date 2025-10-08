import json
import pickle
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import trimesh

from .core import Skeleton, Soma, _bfs_parents

__all__ = [
    "load_mesh",
    "load_swc",
    "to_swc",
    "load_npz",
    "to_npz",
]

_META_KV = re.compile(r"#\s*([^:]+)\s*:\s*(.+)")  #  key: value
_META_JSON = re.compile(r"#\s*meta\s+(\{.*\})")  #  single-line JSON


# ------------
# --- Mesh ---
# ------------


def load_mesh(filepath: str | Path) -> trimesh.Trimesh:
    filepath = Path(filepath)
    if filepath.suffix.lower() == ".ctm":
        print(
            "CTM file detected.  skeliner no longer bundles explicit OpenCTM "
            "support.  Loading will fall back to trimesh’s limited reader.\n"
            "Full read/write support is still possible on compatible setups:\n"
            "  • Python ≤ 3.11, x86-64  →  pip install python-openctm\n"
            "Then load manually:\n"
            "    import openctm, trimesh\n"
            "    mesh = openctm.import_mesh(filepath)\n"
            "    mesh = trimesh.Trimesh(vertices=mesh.vertices,\n"
            "                            faces=mesh.faces,\n"
            "                            process=False)\n"
        )

    mesh = trimesh.load_mesh(filepath, process=False)

    return mesh


# -----------
# --- SWC ---
# -----------


def load_swc(
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
    radii_dict = {"median": radii_arr, "mean": radii_arr, "trim": radii_arr}
    ntype_arr = np.asarray(ntype, dtype=np.int8)
    # --- edges (parent IDs → 0-based indices) ---------------------------
    id_map = {old: new for new, old in enumerate(ids)}
    edges = [
        (id_map[i], id_map[p])
        for i, p in zip(ids, parent, strict=True)
        if p != -1 and p in id_map
    ]
    edges_arr = np.asarray(edges, dtype=np.int64)

    # --- minimal spherical soma around node 0 --------------------------
    soma_centre = nodes_arr[0]
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


def to_swc(
    skeleton,
    path: str | Path,
    include_header: bool = True,
    include_meta: bool = True,
    scale: float = 1.0,
    radius_metric: str | None = None,
    axis_order: tuple[int, int, int] | str = (0, 1, 2),
) -> None:
    """Write the skeleton to SWC.

    The first node (index 0) is written as type 1 (soma) and acts as the
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
        ntype = np.full(len(nodes), 3, dtype=int)
        if len(ntype):
            ntype[0] = 1

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
            fh.write(
                f"{idx} {int(t if idx != 1 else 1)} "  # ensure soma has type 1
                f"{coord[0]} {coord[1]} {coord[2]} {r} "
                f"{(pa + 1) if pa != -1 else -1}\n"
            )


# -----------
# --- npz ---
# -----------


def load_npz(path: str | Path) -> Skeleton:
    """
    Load a Skeleton that was written with `Skeleton.to_npz`.
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
            ntype = np.full(len(nodes), 3, dtype=np.int8)
            if len(ntype):
                ntype[0] = 1

        # reconstruct ragged node2verts
        idx = z["node2verts_idx"].astype(np.int64)
        off = z["node2verts_off"].astype(np.int64)
        node2verts = [idx[off[i] : off[i + 1]] for i in range(len(off) - 1)]

        vert2node = {int(v): i for i, vs in enumerate(node2verts) for v in vs}

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

        node_kdtree = None
        if "kdtree_nodes" in z.files:
            blob = z["kdtree_nodes"]
            if blob.size:
                node_kdtree = pickle.loads(blob.item())

        node_neighbors = None
        if "neighbors_idx" in z.files and "neighbors_off" in z.files:
            idx = z["neighbors_idx"].astype(np.int64)
            off = z["neighbors_off"].astype(np.int64)
            node_neighbors = tuple(idx[off[i] : off[i + 1]] for i in range(len(off) - 1))

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
    if node_kdtree is not None:
        skel._nodes_kdtree = node_kdtree
    if node_neighbors is not None:
        skel._node_neighbors = node_neighbors
    return skel



def to_npz(
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
        When *True* (default) ensure the node KD-tree and adjacency caches are
        built and embedded in the archive for fast reloads.
    """
    path = Path(path)

    # add .npz to the path if not present
    if not path.suffix:
        path = path.with_suffix(".npz")

    c = {} if not compress else {"compress": True}

    if cache_kdtree:
        skeleton._ensure_nodes_kdtree()
        skeleton._ensure_node_neighbors()

    # radii_<name>  : one array per estimator
    radii_flat = {f"r_{k}": v for k, v in skeleton.radii.items()}

    # ragged node2verts  → index + offset
    if skeleton.node2verts is not None:
        n2v_idx = np.concatenate(skeleton.node2verts)
        n2v_off = np.cumsum([0, *map(len, skeleton.node2verts)]).astype(np.int64)
    else:
        n2v_idx = np.array([], dtype=np.int64)
        n2v_off = np.array([0], dtype=np.int64)

    # ----------- NEW: persist the metadata dict -------------------------
    # We wrap it in a 0-D object array because np.savez can only store
    # ndarrays — this keeps the archive a single *.npz* with no sidecars.
    extra = {"extra": np.array(skeleton.extra, dtype=object)}
    meta = {"meta": np.array(skeleton.meta, dtype=object)} if skeleton.meta else {}
    tree_payload = {}
    if skeleton._nodes_kdtree is not None:
        tree_payload["kdtree_nodes"] = np.array(
            pickle.dumps(skeleton._nodes_kdtree), dtype=object
        )
    if skeleton._node_neighbors is not None:
        lengths = np.fromiter(
            (len(nbrs) for nbrs in skeleton._node_neighbors),
            dtype=np.int64,
            count=len(skeleton._node_neighbors),
        )
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        data = (
            np.concatenate(skeleton._node_neighbors)
            if offsets[-1]
            else np.empty(0, dtype=np.int64)
        )
        tree_payload["neighbors_idx"] = data.astype(np.int64, copy=False)
        tree_payload["neighbors_off"] = offsets.astype(np.int64, copy=False)

    np.savez(
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
        **c,
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
    """Pack dict of 1D arrays into NPZ payload using '<prefix>_keys' + '<prefix>__{key}'."""
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
        _schema=np.array("ContactSitesResult@2", dtype="U"),
    )
    # Optional stats: flattened into many arrays with explicit key list per group
    _save_stats_to_payload(getattr(res, "stats_A", None), "stats_A", payload)
    _save_stats_to_payload(getattr(res, "stats_B", None), "stats_B", payload)
    _save_stats_to_payload(getattr(res, "stats_pair", None), "stats_pair", payload)
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
        )
