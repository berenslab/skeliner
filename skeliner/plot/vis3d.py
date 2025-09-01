from typing import Sequence

import numpy as np
import trimesh

from ..core import Skeleton

#### 3D Visualization

def skeliner_to_osteoid(
    skel,
    *,
    segid: int | None = None,
    include_soma: bool = True,   # ← switch it on/off here
    scale: float = 1.          # nm → µm by default; set to 1.0 if you work in µm already
):
    """
    Convert a Skeliner Skeleton → osteoid.Skeleton.

    Parameters
    ----------
    include_soma
        If *False*, drop node 0 and every edge that references it,
        then compact the indices so the new vertex array starts at 0.
    scale
        Multiplicative factor applied to coordinates & radii
        (e.g. 1 nm → 0.001 µm).

    Returns
    -------
    osteoid.Skeleton  ready for microviewer.
    """
    try:
        import osteoid
    except ImportError:
        raise ImportError(
            "`osteoid` is not installed. "
            "Please install it with `pip install --upgrade skeliner[3d]`."
        )

    # --- copy & scale raw data --------------------------------------------
    verts  = (skel.nodes  * scale).astype(np.float32)
    radii  = (skel.r      * scale).astype(np.float32)
    edges  = skel.edges.astype(np.int64)          

    if not include_soma and len(verts):
        keep = (edges[:, 0] != 0) & (edges[:, 1] != 0)
        edges = edges[keep]

        edges -= 1
        verts  = verts[1:]
        radii  = radii[1:]

    edges = edges.astype(np.uint32)

    return osteoid.Skeleton(
            verts, edges, 
            radii=radii,
            segid=segid if segid is not None else 0,  
        )

def trimesh_to_zmesh(tm, segid: int | None = None, scale: float = 1.0):
    """
    Convert a trimesh.Trimesh → zmesh.Mesh and
    bake the desired opacity into the RGBA colour.
    """
    try:
        import zmesh
    except ImportError:
        raise ImportError(
            "`zmesh` is not installed. "
            "Please install it with `pip install --upgrade skeliner[3d]`."
        )

    zm = zmesh.Mesh(
        vertices = tm.vertices.astype(np.float32) * scale, 
        faces    = tm.faces.astype(np.uint32),
        normals  = None,
        id       = segid if segid is not None else 0,
    )
    return zm


def view3d(skels: list[Skeleton] | Skeleton, meshes: list[trimesh.Trimesh] | trimesh.Trimesh,
           include_soma:bool=False, 
           scale: float | tuple | list = 1.0,
           box: "Bbox | list[float] | None" = None,  # noqa: F821 # type: ignore
           mesh_color: str | Sequence = "diff",  # "diff" or "same"
):
    """
    Visualise a list of skeletons and meshes in 3D using microviewer>=1.16.0.

    Parameters
    ----------
    skels : list[Skeleton] | Skeleton
        A list of Skeleton objects or a single Skeleton object.
    meshes : list[trimesh.Trimesh] | trimesh.Trimesh
        A list of trimesh.Trimesh objects or a single trimesh.Trimesh object
    include_soma : bool, optional
        If True, include the skeleton soma in the visualization. Default is False.
    scale : float | tuple[float, float] | list[float], optional
        The scale factor(s) to apply to the skeletons and meshes. Default is 1.0.
    box : Bbox | list[float] | None, optional
        The bounding box ( [x0, y0, z0, x1, y1, z1])to use for the visualization. If None, it will be computed from the
        vertices of the meshes. Default is None.
    mesh_color : str, optional
        The color scheme for the meshes. If "diff", each mesh will have a different color
    Examples
    --------
    import skeliner as sk
    import matplotlib.cm as cm

    name = [720575940550605504, 720575940573924400]
    meshes = [sk.io.load_mesh(f"../temp_io/{n}.ctm") for n in name]
    skels = [sk.io.load_npz(f"../temp_io/{n}.npz") for n in name]

    sk.view3d(skels, meshes, scale=1e-3, mesh_color=cm.tab20.colors)
    """

    try:
        from microviewer import objects
    except ImportError:
        raise ImportError(
            "microviewer is not installed. "
            "Please install all dependencies with `pip install --upgrade skeliner[3d]`."
        )
    try:
        from osteoid.lib import Bbox
    except ImportError:
        raise ImportError(
            "osteoid is not installed. "
            "Please install all dependencies with `pip install --upgrade skeliner[3d]`."
        )

    if isinstance(skels, Skeleton):
        skels = [skels]
    if isinstance(meshes, trimesh.Trimesh):
        meshes = [meshes]

    if isinstance(scale, (int, float)):
        skel_scale = float(scale)
        mesh_scale = float(scale)
    else:
        if len(scale) != 2:
            raise ValueError("scale must be a scalar or a pair/list of two scalars")
        skel_scale, mesh_scale = map(float, scale)

    ost_skels = [skeliner_to_osteoid(skel, segid=segid, include_soma=include_soma, scale=skel_scale) for (segid, skel) in enumerate(skels)]
    zm_meshes = [trimesh_to_zmesh(mesh, segid=segid, scale=mesh_scale) for (segid, mesh) in enumerate(meshes)]

    if box is None:
        # use the min and max of all vertices in the meshes
        box = Bbox(
            np.min([mesh.vertices.min(axis=0) for mesh in zm_meshes], axis=0),
            np.max([mesh.vertices.max(axis=0) for mesh in zm_meshes], axis=0)
        )
    else:
        if len(box) == 6:
            box = Bbox(box[:3], box[3:])
        elif len(box) == 3:
            box = Bbox([0, 0, 0], box)
        else:
            raise ValueError("Invalid bounding box format. Expected [x0, y0, z0, x1, y1, z1].")

    objects([box, *zm_meshes, *ost_skels], mesh_color=mesh_color)