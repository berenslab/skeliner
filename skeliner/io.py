from pathlib import Path
from typing import Iterable, List

import numpy as np
import openctm
import trimesh

from .core import Skeleton, Soma, _bfs_parents

__all__ = ["load_mesh", "load_swc"]

# ------------
# --- Mesh ---
# ------------

def load_mesh(filepath: str | Path) -> trimesh.Trimesh:

    filepath = Path(filepath)

    if filepath.suffix == ".ctm":
        mesh = openctm.import_mesh(filepath)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    else:
        mesh = trimesh.load_mesh(filepath, process=False)

    return mesh

def to_ctm(mesh: trimesh.Trimesh, path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() != ".ctm":
        mesh.export(path, file_type=path.suffix.lstrip("."))
        raise ValueError(
            f"Expected a .ctm file, got {path.suffix}. "
        )

    verts  = mesh.vertices.astype(np.float32, copy=False)
    faces  = mesh.faces.astype(np.uint32,  copy=False)

    ctm_mesh = openctm.CTM(verts, faces, None)
    openctm.export_mesh(ctm_mesh, path)

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

    with path.open("r", encoding="utf8") as fh:
        for line in fh:
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

    if not ids:
        raise ValueError(f"No usable nodes found in {path}")

    # --- core arrays ----------------------------------------------------
    nodes_arr  = np.asarray(xyz, dtype=np.float32) * scale
    radii_arr  = np.asarray(radii, dtype=np.float32) * scale
    radii_dict = {"median": radii_arr}          # only one estimator available

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
        soma=soma,
        node2verts=None,
        vert2node=None,
    )

def to_swc(skeleton, 
            path: str | Path,
            include_header: bool = True, 
            scale: float = 1.0,
            radius_metric: str | None = None
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
    
    path = Path(path)

    # add .swc to the path if not present
    if not path.suffix:
        path = path.with_suffix(".swc")

    parent = _bfs_parents(skeleton.edges, len(skeleton.nodes), root=0)
    nodes = skeleton.nodes
    if radius_metric is None:
        radii = skeleton.r
    else:
        if radius_metric not in skeleton.radii:
            raise ValueError(f"Unknown radius estimator '{radius_metric}'")
        radii = skeleton.radii[radius_metric]
    with path.open("w", encoding="utf8") as fh:
        if include_header:
            fh.write("# id type x y z radius parent\n")
        for idx, (p, r, pa) in enumerate(zip(nodes * scale, radii * scale, parent), start=1):
            swc_type = 1 if idx == 1 else 0
            fh.write(f"{idx} {swc_type} {p[0]} {p[1]} {p[2]} {r} {pa + 1 if pa != -1 else -1}\n")

# -----------
# --- npz ---
# -----------

def load_npz(path: str | Path) -> Skeleton:
    """
    Load a Skeleton that was written with `Skeleton.to_npz`.
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as z:
        nodes  = z["nodes"].astype(np.float32)
        edges  = z["edges"].astype(np.int64)

        # radii dict  (keys start with 'r_')
        radii = {
            k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("r_")
        }

        # reconstruct ragged node2verts
        idx = z["node2verts_idx"].astype(np.int64)
        off = z["node2verts_off"].astype(np.int64)
        node2verts = [idx[off[i]:off[i+1]] for i in range(len(off)-1)]

        vert2node = {int(v): i for i, vs in enumerate(node2verts) for v in vs}

        soma = Soma(
            centre=z["soma_centre"],
            axes=z["soma_axes"],
            R=z["soma_R"],
            verts=(
                z["soma_verts"].astype(np.int64) if "soma_verts" in z else None
            ),
        )

    return Skeleton(nodes, radii, edges, soma=soma,
                    node2verts=node2verts, vert2node=vert2node)

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

    np.savez(
        path, 
        nodes=skeleton.nodes,
        edges=skeleton.edges,
        soma_centre=skeleton.soma.centre,
        soma_axes=skeleton.soma.axes,
        soma_R=skeleton.soma.R,
        soma_verts=skeleton.soma.verts if skeleton.soma.verts is not None else np.array([], dtype=np.int64),
        node2verts_idx=n2v_idx,
        node2verts_off=n2v_off,
        **radii_flat,
        **c
    )