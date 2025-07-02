import json
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

_META_KV   = re.compile(r"#\s*([^:]+)\s*:\s*(.+)")         #  key: value
_META_JSON = re.compile(r"#\s*meta\s+(\{.*\})")   #  single-line JSON

 
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

    ids: List[int]     = []
    xyz: List[List[float]] = []
    radii: List[float] = []
    parent: List[int]  = []
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
    nodes_arr  = np.asarray(xyz, dtype=np.float64) * scale
    radii_arr  = np.asarray(radii, dtype=np.float64) * scale
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

def to_swc(skeleton, 
            path: str | Path,
            include_header: bool = True, 
            include_meta: bool = True,          
            scale: float = 1.0,
            radius_metric: str | None = None,
            axis_order: tuple[int, int, int] | str = (0, 1, 2)
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
        nodes  = z["nodes"].astype(np.float64)
        edges  = z["edges"].astype(np.int64)

        # radii dict  (keys start with 'r_')
        radii = {
            k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("r_")
        }

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
        node2verts = [idx[off[i]:off[i+1]] for i in range(len(off)-1)]

        vert2node = {int(v): i for i, vs in enumerate(node2verts) for v in vs}

        soma = Soma(
            center=z["soma_centre"],
            axes=z["soma_axes"],
            R=z["soma_R"],
            verts=(
                z["soma_verts"].astype(np.int64) if "soma_verts" in z else None
            ),
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

    return Skeleton(nodes=nodes, radii=radii, edges=edges, ntype=ntype, soma=soma,
                    node2verts=node2verts, vert2node=vert2node, extra=extra, meta=meta)

def to_npz(skeleton: Skeleton, path: str | Path, *, compress: bool = True) -> None:
    """
    Write the skeleton to a compressed `.npz` archive.
    """
    path = Path(path)

    # add .npz to the path if not present
    if not path.suffix:
        path = path.with_suffix(".npz")

    c = {} if not compress else {"compress": True}

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

    np.savez(
        path, 
        nodes=skeleton.nodes,
        edges=skeleton.edges,
        ntype=skeleton.ntype if skeleton.ntype is not None else np.array([], dtype=np.int8),
        soma_centre=skeleton.nodes[0],
        soma_axes=skeleton.soma.axes,
        soma_R=skeleton.soma.R,
        soma_verts=skeleton.soma.verts if skeleton.soma.verts is not None else np.array([], dtype=np.int64),
        node2verts_idx=n2v_idx,
        node2verts_off=n2v_off,
        **radii_flat,
        **extra,
        **meta,
        **c,
    )