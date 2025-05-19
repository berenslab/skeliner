from pathlib import Path
from typing import Iterable, List

import numpy as np
import openctm
import trimesh

from .core import Skeleton

__all__ = ["load_mesh", "load_swc"]

def load_mesh(filepath: str | Path) -> trimesh.Trimesh:

    filepath = Path(filepath)

    if filepath.suffix == ".ctm":
        mesh = openctm.import_mesh(filepath)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    else:
        mesh = trimesh.load_mesh(filepath, process=False)

    return mesh


def load_swc(
    path: str | Path,
    scale: float = 1.0,
    keep_types: Iterable[int] | None = None,
) -> Skeleton:
    """
    Read an SWC file into a :class:`Skeleton`.

    Parameters
    ----------
    path : str or Path
        Filename of the SWC file.
    scale : float, default ``1.0``
        Multiply coordinates and radii by this factor on load.
    keep_types : Iterable[int] | None, optional
        Sequence of SWC *type* codes to keep.  When *None* (default) all
        node types are imported.

    Returns
    -------
    Skeleton
        The imported skeleton.  Note that ``soma_verts`` is **None** because
        the SWC format has no concept of surface vertices.

    Raises
    ------
    ValueError
        If the file contains no valid nodes.
    """    
    path = Path(path)
    ids: List[int] = []
    xyz: List[List[float]] = []
    radii: List[float] = []
    parent: List[int] = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            _id, _type = int(parts[0]), int(parts[1])
            if keep_types is not None and _type not in keep_types:
                continue
            ids.append(_id)
            xyz.append([float(parts[2]), float(parts[3]), float(parts[4])])
            radii.append(float(parts[5]))
            parent.append(int(parts[6]))
    if not ids:
        raise ValueError(f"No nodes found in {path}")

    id_map = {old: new for new, old in enumerate(ids)}
    nodes_arr = np.asarray(xyz, dtype=np.float32) * scale
    radii_dict = {"median": np.asarray(radii, dtype=np.float32) * scale}
    edges = [
        (id_map[i], id_map[p])
        for i, p in zip(ids, parent, strict=True)
        if p != -1 and p in id_map
    ]
    return Skeleton(nodes_arr, radii_dict, np.asarray(edges, dtype=np.int64))
