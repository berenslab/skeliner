from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse
from scipy.stats import binned_statistic_2d

from ..core import Skeleton

__all__ = ["projection", "threeviews", "details", "node_details"]


Number = int | float

_PLANE_AXES = {
    "xy": (0, 1), "yx": (1, 0),
    "xz": (0, 2), "zx": (2, 0),
    "yz": (1, 2), "zy": (2, 1),  
}

_PLANE_NORMAL = {
    "xy": np.array([0, 0, 1.0]),
    "yx": np.array([0, 0, 1.0]),
    "xz": np.array([0, 1.0, 0]),
    "zx": np.array([0, 1.0, 0]),
    "yz": np.array([1.0, 0, 0]),
    "zy": np.array([1.0, 0, 0]),
}


_GOLDEN_RATIO = 0.618033988749895          # for visually distinct colours

def _project(arr: np.ndarray, ix: int, iy: int, /) -> np.ndarray:
    """Return 2-column slice (arr[:, (ix, iy)])."""
    return arr[:, (ix, iy)].copy()

def _component_labels(n_verts: int,
                      node2verts: list[np.ndarray]) -> np.ndarray:
    """
    LAB[mesh_vid] = *component id* (“cluster id”)  – or –1 if vertex does not
    belong to any skeleton node (shouldn’t normally happen).

    One node ↔ one component, therefore a simple linear scan is sufficient.
    """
    lab = np.full(n_verts, -1, dtype=np.int64)
    for cid, verts in enumerate(node2verts):
        lab[verts] = cid
    return lab

def _radii_to_sizes(rr: np.ndarray, ax: Axes) -> tuple[np.ndarray, float]:
    """
    Convert radii (data units) → *scatter* sizes (points²) so that the same
    physical radius is rendered identically in every subplot.
    """
    fig = ax.figure
    dpi = fig.dpi

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox   = ax.get_window_extent()

    ppd_x = bbox.width  / abs(x1 - x0)
    ppd_y = bbox.height / abs(y1 - y0)
    ppd   = min(ppd_x, ppd_y)

    r_px = rr * ppd
    r_pt = r_px * 72.0 / dpi
    return np.pi * r_pt**2, ppd


def _trapezoid_3d(
    p0_3d: np.ndarray, p1_3d: np.ndarray,
    r0: float, r1: float,
    plane: str,
) -> np.ndarray | None:
    """
    Return 4×2 vertices of the projected trapezoid that joins the two circles
    centred at *p0_3d* and *p1_3d* with radii *r0*, *r1*.

    The bases are perpendicular to the true 3-D axis, so their projected
    length equals the node diameters.  Returns **None** for zero-length edges.
    """
    v = p1_3d - p0_3d
    if not np.any(v):                 # degenerate edge
        return None

    n3 = np.cross(v, _PLANE_NORMAL[plane])
    L  = np.linalg.norm(n3)
    if L == 0:                        # edge is parallel to the projection normal
        return None
    n3 /= L                           # now unit length in 3-D

    # project to 2-D
    ix, iy = _PLANE_AXES[plane]
    n2 = n3[[ix, iy]]

    p0 = p0_3d[[ix, iy]]
    p1 = p1_3d[[ix, iy]]

    return np.array([
        p0 + n2 * r0,
        p0 - n2 * r0,
        p1 - n2 * r1,
        p1 + n2 * r1,
    ], dtype=float)


def _soma_ellipse2d(soma, plane: str, *, scale: float = 1.0) -> Ellipse:
    """
    Exact orthographic projection of a 3-D ellipsoid onto *plane*.

    The ellipse is given by   (x-c)ᵀ Q (x-c) = 1
    with       Q = B_pp − B_pq B_qp / B_qq
    where      B = R diag(1/a²) Rᵀ   is the quadric matrix in world coords
    and the indices p,q denote the kept/dropped coordinate.
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {_PLANE_AXES.keys()}")

    ix, iy = _PLANE_AXES[plane]
    k      = 3 - ix - iy                      # the coordinate we project away

    # inverse shape matrix of the ellipsoid
    B = soma.R @ np.diag(1.0 / soma.axes**2) @ soma.R.T

    B_pp = B[[ix, iy]][:, [ix, iy]]           # 2×2
    B_pq = B[[ix, iy], k].reshape(2, 1)       # 2×1
    B_qq = B[k, k]

    Q = B_pp - (B_pq @ B_pq.T) / B_qq         # 2×2 positive-definite

    # eigen-decomposition → half-axes in the projection plane
    eigval, eigvec = np.linalg.eigh(Q)        # λ₁, λ₂ > 0
    half_axes      = 1.0 / np.sqrt(eigval)    # r₁, r₂
    order          = np.argsort(-half_axes)   # big → small

    width, height  = 2 * half_axes[order] * scale
    angle_deg      = np.degrees(np.arctan2(eigvec[1, order[0]],
                                           eigvec[0, order[0]]))
    centre_xy      = soma.center[[ix, iy]] * scale

    return Ellipse(centre_xy, width, height, angle=angle_deg,
                   linewidth=.8, linestyle="--",
                   facecolor="none", edgecolor="k", alpha=.9)

def _make_lut(name: str, n: int) -> np.ndarray:
    """
    Return an ``(n, 4)`` RGBA array from *name* colormap, shuffled so that
    neighbouring IDs get well-separated colours.

    Works on Matplotlib ≥ 3.5 and stays silent on ≥ 3.7.
    """
    # Matplotlib ≥ 3.5 – the recommended public API
    cmap = mpl.colormaps.get_cmap(name).resampled(max(n, 1))

    # golden-ratio shift ensures adjacent IDs differ strongly
    idx = (np.arange(max(n, 1)) * _GOLDEN_RATIO) % 1.0
    return cmap(idx)


def projection(
    skel: Skeleton,
    mesh: trimesh.Trimesh | None= None,
    *,
    plane: str = "xy",
    radius_metric: str | None = None,
    bins: int | tuple[int, int] = 800,
    scale: float | Sequence[Number] = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    draw_skel: bool = True,
    draw_edges: bool = True,
    draw_cylinders: bool = False,
    ax: Axes | None = None,
    mesh_cmap: str = "Blues", # mesh color map
    skel_cmap: str = "Pastel2",  # skeleton color map
    vmax_fraction: float = 0.10,
    edge_lw: float = 0.5,
    circle_alpha: float = 0.25,
    cylinder_alpha: float = 0.5,
    highlight_nodes: int | Sequence[int] | None = None,
    highlight_face_alpha: float = 0.5,
    unit: str | None = None,
    # soma --------------------------------------------------------------- #
    draw_soma_mask: bool = True,
    # colors
    color_by: str = "fixed",  # "ntype" or "fixed"
) -> tuple[Figure, Axes]:
    """Orthographic 2‑D overview of a skeleton with an **optional** mesh‑density
    background.

    Parameters
    ----------
    skel : Skeleton
        The centre‑line skeleton to visualise.
    mesh : trimesh.Trimesh | None, default *None*
        Surface mesh used to draw a vertex‑density heat‑map and (optionally) the
        soma surface.  Pass *None* to **omit** the background histogram and any
        mesh‑based overlays.
    plane : {"xy", "xz", "yz", "yx", "zx", "zy"}
        Projection plane.
    bins : int | (int,int), default *800*
        Resolution of the background histogram.  Ignored when *mesh* is *None*.
    scale : float | (float, float), default *1*
        Multiplicative scale(s).  Either a scalar applied to both skeleton and
        mesh or a pair ``(s_skel, s_mesh)``.
    xlim, ylim : (min, max) or *None*
        Spatial extent **before** plotting.  If not given, limits are inferred
        from the histogram (when *mesh* is available) or the skeleton.
    draw_skel, draw_edges, draw_cylinders : bool
        Toggles for skeleton glyphs.
    ax : matplotlib.axes.Axes | None
        Existing *Axes* to draw into.  When *None*, a new figure is created.
    mesh_cmap, vmax_fraction : appearance of the histogram – see original docs.
    circle_alpha, cylinder_alpha : transparencies of skeleton glyphs.
    highlight_nodes : node IDs to highlight.
    unit : str | None
        Axis‑label unit.
    draw_soma_mask : bool, default *True*
        Draw the soma shell when both *mesh* **and** soma vertices are
        available.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # ───────────────────────────────── validation & setup ──────────────────
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")

    ix, iy = _PLANE_AXES[plane]

    # normalise *scale* → [skel_scale, mesh_scale]
    if not isinstance(scale, Sequence):
        scale = [scale, scale]
    if len(scale) != 2:
        raise ValueError("scale must be a scalar or a pair of two scalars")
    scl_skel, scl_mesh = map(float, scale)

    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    if unit is None: # try to grab from metadata
        unit = skel.meta.get("unit", None)

    highlight_set = (set(map(int, np.atleast_1d(highlight_nodes)))
                     if highlight_nodes is not None else set())

    # ─────────────────────────────colormap ─────────────
    # grab the full Dark2 cmap (8 colours)
    cmap = mpl.colormaps.get_cmap(skel_cmap).resampled(8)
    # convenience: lambda i -> RGBA row
    dc = lambda i: cmap(i)               # noqa: E731

    # build SWC-indexed palette (row 0..N).  Feel free to extend >6 if you need.
    swc_colors = np.vstack([
        dc(7),      # 0  undefined   – grey
        dc(0),      # 1  soma        – green-teal
        dc(1),      # 2  axon        – orange
        dc(2),      # 3  dendrite    – olive-green
        dc(3),      # 4  apical      – purple
        dc(4),      # 5  fork point  – yellow-orange
        dc(5),      # 6  end point   – brown
    ])

    # ───────────────────────────── project (and optionally crop) ───────────
    xy_skel = _project(skel.nodes, ix, iy) * scl_skel
    rr      = skel.radii[radius_metric] * scl_skel

    if mesh is not None:
        xy_mesh = _project(mesh.vertices, ix, iy) * scl_mesh
    else:  # empty placeholder for unified code‑path
        xy_mesh = np.empty((0, 2), dtype=float)

    # helper – applies *xlim/ylim* cropping on a 2‑column array
    def _crop_window(xy: np.ndarray) -> np.ndarray:
        keep = np.ones(len(xy), dtype=bool)
        if xlim is not None:
            keep &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            keep &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        return keep

    # crop *before* heavy lifting
    keep_skel = _crop_window(xy_skel)
    keep_mask = keep_skel                # ← keep the original name for edges
    idx_keep  = np.flatnonzero(keep_mask)   # 1-D array of kept node IDs
    xy_skel   = xy_skel[keep_mask]          # already done
    rr        = rr[keep_mask]               # already done

    # colour array for the *kept* nodes
    if color_by == "ntype" and skel.ntype is not None:
        col_nodes = swc_colors[skel.ntype[idx_keep]]
    else:
        col_nodes = "red"

    if mesh is not None and xy_mesh.size:
        keep_mesh = _crop_window(xy_mesh)
        xy_mesh   = xy_mesh[keep_mesh]

    # ─────────────────────────────── histogram (mesh may be None) ──────────
    if mesh is not None and xy_mesh.size:
        # ensure bins argument correct
        if isinstance(bins, int):
            bins_arg: int | tuple[int, int] = bins
        elif (isinstance(bins, tuple) and len(bins) == 2 and
              all(isinstance(b, int) for b in bins)):
            bins_arg = (int(bins[0]), int(bins[1]))
        else:
            raise ValueError("bins must be an int or a tuple of two ints")

        hist, xedges, yedges, _ = binned_statistic_2d(
            xy_mesh[:, 0], xy_mesh[:, 1], None,
            statistic="count", bins=bins_arg,
        )
        hist = hist.T  # imshow expects (rows = y)
    else:
        hist = None

    # ───────────────────────────── figure / axes boilerplate ───────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # background image – only when we do have a histogram
    if hist is not None:
        ax.imshow(
            hist,
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            origin="lower",
            cmap=mesh_cmap,
            vmax=hist.max() * vmax_fraction,
            alpha=1.0,
        )

    # ──────────────────────── draw skeleton circles (always) ───────────────
    if draw_skel and xy_skel.size:
        # limits need to be defined before converting radii → scatter sizes
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        # elif hist is None:  # fallback to skeleton extents
        else:
            ax.set_xlim((xy_skel[:, 0].min(), xy_skel[:, 0].max()))
            ax.set_ylim((xy_skel[:, 1].min(), xy_skel[:, 1].max()))

        ax.set_aspect("equal", adjustable="box")
        sizes, _ppd = _radii_to_sizes(rr, ax)

        ax.scatter(
            xy_skel[:, 0][1:],
            xy_skel[:, 1][1:],
            s=sizes[1:],
            facecolors="none",
            edgecolors=col_nodes[1:] if isinstance(col_nodes, np.ndarray) else col_nodes,
            linewidths=1.0,
            alpha=circle_alpha,
            zorder=2,
        )

        # highlighted nodes – filled circles
        if highlight_set:
            orig_ids = np.flatnonzero(keep_skel)
            hilite_mask = np.isin(orig_ids, list(highlight_set))
            if hilite_mask.any():
                ax.scatter(
                    xy_skel[hilite_mask, 0],
                    xy_skel[hilite_mask, 1],
                    s=sizes[hilite_mask],
                    facecolors="green",
                    edgecolors="green",
                    linewidths=0.9,
                    alpha=highlight_face_alpha,
                    zorder=3.5,
                )

    # ───────────────────────── soma shell & center (if possible) ───────────
    c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
    centre_col = swc_colors[1] if color_by == "ntype" else "k"
    ax.scatter(*c_xy, color="black", s=15, zorder=3)

    if (draw_soma_mask and mesh is not None and skel.soma is not None and
            skel.soma.verts is not None):
        xy_soma = _project(mesh.vertices[np.asarray(skel.soma.verts, int)], ix, iy)
        xy_soma = xy_soma * scl_mesh
        xy_soma = xy_soma[_crop_window(xy_soma)]  # respect crop

        col_soma = swc_colors[1] if color_by == "ntype" else "pink"

        ax.scatter(
            xy_soma[:, 0], xy_soma[:, 1],
            s=1,
            c=[col_soma],
            alpha=0.5, linewidths=0,
            label="soma surface"
        )
        # soma center (node 0)
        # c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
        # ax.scatter(*c_xy, c="k", s=15, zorder=3)

        # dashed ellipse outline
        ell = _soma_ellipse2d(skel.soma, plane, scale=scl_skel)
        ell.set_edgecolor("k")
        ell.set_facecolor("none")
        ell.set_linestyle("--")
        ell.set_linewidth(0.8)
        ell.set_alpha(0.9)
        ax.add_patch(ell)
    else:
        soma_circle = Circle(
            c_xy, skel.soma.equiv_radius * scl_skel,
            facecolor="none", edgecolor=centre_col, linewidth=0.8,
            linestyle="--", alpha=0.9,
        )
        ax.add_patch(soma_circle)


    # ─────────────────────── draw edges & cylinders (unchanged) ────────────
    if draw_skel and skel.edges.size:
        keep = keep_skel  # alias
        if draw_edges:
            ekeep = keep[skel.edges[:, 0]] & keep[skel.edges[:, 1]]
            edges_kept = skel.edges[ekeep]
            if edges_kept.size:
                # original → compressed index map
                idx_map = -np.ones(len(keep), int)
                idx_map[np.flatnonzero(keep)] = np.arange(keep.sum())

                seg_start = xy_skel[idx_map[edges_kept[:, 0]]]
                seg_end   = xy_skel[idx_map[edges_kept[:, 1]]]
                segments  = np.stack((seg_start, seg_end), axis=1)

                lc = LineCollection(segments.tolist(), colors="black",
                                    linewidths=edge_lw, alpha=cylinder_alpha)
                ax.add_collection(lc)

        if draw_cylinders:
            ekeep = keep_skel[skel.edges[:, 0]] & keep_skel[skel.edges[:, 1]]
            edges_kept = skel.edges[ekeep]
            if edges_kept.size:
                idx_map = -np.ones(len(keep_skel), int)
                idx_map[np.flatnonzero(keep_skel)] = np.arange(keep_skel.sum())

                quads = []
                for n0, n1 in edges_kept:
                    i0, i1 = idx_map[[n0, n1]]
                    quad = _trapezoid_3d(
                        skel.nodes[n0] * scl_skel,
                        skel.nodes[n1] * scl_skel,
                        rr[i0], rr[i1],
                        plane,
                    )
                    if quad is not None:
                        quads.append(quad)

                if quads:
                    # make sure axes limits are already set before adding
                    if xlim is not None: 
                        ax.set_xlim(xlim)
                    if ylim is not None: 
                        ax.set_ylim(ylim)

                    pc = PolyCollection(quads, facecolors="red",
                                         edgecolors="red", alpha=cylinder_alpha,
                                         zorder=10)
                    ax.add_collection(pc)

    # ────────────────────────────── final cosmetics ────────────────────────
    # ax.set_aspect("equal")

    if unit is None:
        unit_str = "" if scl_skel == 1.0 else f"(×{scl_skel:g})"
    else:
        unit_str = f"({unit})"

    ax.set_xlabel(f"{plane[0]} {unit_str}")
    ax.set_ylabel(f"{plane[1]} {unit_str}")

    # guarantee limits if user requested specific window
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    return fig, ax


def details(
    skel: Skeleton,
    mesh: trimesh.Trimesh | None = None,
    *,
    plane: str = "xy",
    # background histogram ------------------------------------------------- #
    bins: int | tuple[int, int] = 800,
    hist_cmap: str = "Blues",
    vmax_fraction: float = 0.10,
    # overlays ------------------------------------------------------------- #
    draw_nodes: bool = True,
    draw_edges: bool = False,
    draw_cylinders: bool = False,
    draw_soma_mask: bool = True,
    show_node_ids: bool = False,
    radius_metric: str | None = None,
    # appearance ----------------------------------------------------------- #
    cluster_cmap: str = "tab20",
    circle_alpha: float = 0.9,
    cylinder_alpha: float = 0.25,
    edge_color: str = "0.25",
    edge_lw: float = 0.8,
    id_fontsize: int = 6,
    id_color: str = "black",
    id_offset: tuple[float, float] = (0.0, 0.0),
    # geometry ------------------------------------------------------------- #
    scale: Number | tuple[Number, Number] | Sequence[Number] = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    # title --------------------------------------------------------------- #
    title: str | None = None,
    unit: str | None = None,
    # highlight ------------------------------------------------------------ #
    highlight_nodes: int | Sequence[int] | None = None,
    highlight_face_alpha: float = 0.5,
    highlight_id_color: str = "red",
    # axes ----------------------------------------------------------------- #
    ax: Axes | None = None,
):
    """Mesh‑optional cluster overview.

    Pass ``mesh=None`` to skip the vertex cloud & soma shell.  All skeleton
    overlays still work.  Other parameters keep their legacy meaning; see the
    original docs.
    """

    # ────────────── housekeeping / defaults ───────────────────────────────
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")
    ix, iy = _PLANE_AXES[plane]

    if isinstance(scale, (int, float)):
        scale = (float(scale),) * 2
    if len(scale) != 2:
        raise ValueError("scale must be a scalar or a pair of two scalars")
    scl_skel, scl_mesh = map(float, scale)

    if unit is None: # try to grab from metadata
        unit = skel.meta.get("unit", None)

    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    highlight_set = (set(map(int, np.atleast_1d(highlight_nodes)))
                     if highlight_nodes is not None else set())

    # ────────────── project skeleton (mandatory) ──────────────────────────
    xy_skel = _project(skel.nodes, ix, iy) * scl_skel
    rr      = skel.radii[radius_metric] * scl_skel

    # ────────────── optional mesh handling ────────────────────────────────
    have_mesh = mesh is not None and mesh.vertices.size
    if have_mesh:
        xy_mesh_all = _project(mesh.vertices.view(np.ndarray), ix, iy) * scl_mesh
    else:
        xy_mesh_all = np.empty((0, 2), dtype=float)

    # crop helper
    def _mask_window(xy: np.ndarray) -> np.ndarray:
        keep = np.ones(len(xy), bool)
        if xlim is not None:
            keep &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            keep &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        return keep

    keep_skel = _mask_window(xy_skel)
    xy_skel   = xy_skel[keep_skel]
    rr        = rr[keep_skel]

    if have_mesh:
        keep_mesh     = _mask_window(xy_mesh_all)
        xy_mesh_crop  = xy_mesh_all[keep_mesh]
    else:
        xy_mesh_crop = np.empty((0, 2), dtype=float)

    # ────────────── density histogram (mesh may be absent) ────────────────
    if have_mesh and xy_mesh_crop.size:
        if isinstance(bins, int):
            bins_arg: int | tuple[int, int] = bins
        else:
            if (not isinstance(bins, tuple)) or len(bins) != 2:
                raise ValueError("bins must be int or (int, int)")
            bins_arg = tuple(map(int, bins))

        from scipy.stats import binned_statistic_2d  # heavy import lazily
        hist, xedges, yedges, _ = binned_statistic_2d(
            xy_mesh_crop[:, 0], xy_mesh_crop[:, 1], None,
            statistic="count", bins=bins_arg,
        )
        hist = hist.T  # imshow rows=y
    else:
        hist = None

    # ────────────── component / cluster labels (optional) ─────────────────
    if have_mesh and skel.node2verts is not None and len(xy_mesh_crop):
        lab_full   = _component_labels(len(mesh.vertices), skel.node2verts)
        in_cluster = lab_full >= 0
        mask_mesh  = keep_mesh & in_cluster
        xy_mesh_scatter = xy_mesh_all[mask_mesh]
        lab_mesh        = lab_full[mask_mesh]
        n_comp          = int(lab_full.max() + 1)
    else:
        xy_mesh_scatter = xy_mesh_crop  # possibly empty
        lab_mesh        = None
        n_comp          = 0

    # ────────────── figure / axes ─────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # background density image
    if hist is not None:
        ax.imshow(
            hist,
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            origin="lower",
            cmap=hist_cmap,
            vmax=hist.max() * vmax_fraction,
            alpha=1.0,
            zorder=0,
        )

    # ────────────── mesh vertex cloud overlay (if any) ────────────────────
    if xy_mesh_scatter.size:
        if n_comp:
            lut = _make_lut(cluster_cmap, n_comp)
            colours = lut[lab_mesh]
        else:
            colours = "0.6"

        ax.scatter(
            xy_mesh_scatter[:, 0], xy_mesh_scatter[:, 1],
            s=1.0, c=colours, alpha=0.75, linewidths=0, zorder=9,
        )

    # ────────────── axis limits before scatter‑size calc ──────────────────
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    elif xy_mesh_scatter.size:
        ax.set_xlim((xy_mesh_scatter[:, 0].min(), xy_mesh_scatter[:, 0].max()))
        ax.set_ylim((xy_mesh_scatter[:, 1].min(), xy_mesh_scatter[:, 1].max()))
    else:  # fallback to skeleton extents
        ax.set_xlim((xy_skel[:, 0].min(), xy_skel[:, 0].max()))
        ax.set_ylim((xy_skel[:, 1].min(), xy_skel[:, 1].max()))

    sizes, _ppd = _radii_to_sizes(rr, ax)

    # ────────────── skeleton circles & centres ────────────────────────────
    if draw_nodes and xy_skel.size:
        if n_comp:
            # per‑node colour maps to cluster of its first vertex
            node_comp = np.array([
                lab_full[skel.node2verts[i][0]] if len(skel.node2verts[i]) else -1
                for i in range(len(skel.nodes))
            ])
            node_comp   = node_comp[keep_skel]
            node_colors = lut[node_comp]
        else:
            node_colors = "red"  # simple fallback

        slicing = 0 if skel.soma.verts is None else 1
        ax.scatter(
            xy_skel[:, 0][slicing:], xy_skel[:, 1][slicing:],
            s=sizes[slicing:], facecolors="none", edgecolors=node_colors[slicing:],
            linewidths=0.9, alpha=circle_alpha, zorder=3,
        )
        ax.scatter(
            xy_skel[:, 0], xy_skel[:, 1],
            s=10, c=node_colors, alpha=circle_alpha, zorder=4, linewidths=0,
        )

        # optional node‑ID labels
        if show_node_ids:
            orig_ids = np.flatnonzero(keep_skel)
            dx, dy   = id_offset
            for i_compressed, nid in enumerate(orig_ids):
                color = highlight_id_color if nid in highlight_set else id_color
                x, y = xy_skel[i_compressed]
                ax.text(
                    x + dx, y + dy, str(nid),
                    fontsize=id_fontsize, color=color,
                    ha="center", va="center", zorder=5,
                )

        # fill highlighted nodes
        if highlight_set:
            orig_ids = np.flatnonzero(keep_skel)
            hilite_mask = np.isin(orig_ids, list(highlight_set))
            if hilite_mask.any():
                ax.scatter(
                    xy_skel[hilite_mask, 0], xy_skel[hilite_mask, 1],
                    s=sizes[hilite_mask], facecolors=node_colors[hilite_mask] if n_comp else "green",
                    edgecolors=node_colors[hilite_mask] if n_comp else "green",
                    linewidths=0.9, alpha=highlight_face_alpha, zorder=4.5,
                )

    # ────────────── edges & cylinders (use skeleton only) ─────────────────
    if draw_edges and skel.edges.size:
        keep_flags = keep_skel
        ekeep      = keep_flags[skel.edges[:, 0]] & keep_flags[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]
        if edges_kept.size:
            idx_map = -np.ones(len(keep_flags), int)
            idx_map[np.flatnonzero(keep_flags)] = np.arange(keep_flags.sum())
            seg_start = xy_skel[idx_map[edges_kept[:, 0]]]
            seg_end   = xy_skel[idx_map[edges_kept[:, 1]]]
            segs      = np.stack((seg_start, seg_end), axis=1)
            lc = LineCollection(segs, colors=edge_color, linewidths=edge_lw,
                                alpha=circle_alpha * 0.9, zorder=2)
            ax.add_collection(lc)

    if draw_nodes and draw_cylinders and skel.edges.size:
        ekeep      = keep_skel[skel.edges[:, 0]] & keep_skel[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]
        if edges_kept.size:
            idx_map = -np.ones(len(keep_skel), int)
            idx_map[np.flatnonzero(keep_skel)] = np.arange(keep_skel.sum())
            quads, facecols = [], []
            for n0, n1 in edges_kept:
                i0, i1 = idx_map[[n0, n1]]
                quad = _trapezoid_3d(
                    skel.nodes[n0] * scl_skel,
                    skel.nodes[n1] * scl_skel,
                    rr[i0], rr[i1], plane,
                )
                if quad is None:
                    continue
                quads.append(quad)
                facecols.append(node_colors[i0] if draw_nodes and n_comp else "0.25")

            if quads:
                pc = PolyCollection(quads, facecolors=facecols, edgecolors="none",
                                     alpha=cylinder_alpha, zorder=2.8)
                ax.add_collection(pc)

    # ────────────── optional soma shell (need mesh) ───────────────────────
    if (draw_soma_mask and have_mesh and skel.soma is not None and
            skel.soma.verts is not None):
        xy_soma = _project(mesh.vertices[np.asarray(skel.soma.verts, int)], ix, iy)
        xy_soma = xy_soma * scl_mesh
        soma_keep = _mask_window(xy_soma)
        xy_soma   = xy_soma[soma_keep]
        ax.scatter(xy_soma[:, 0], xy_soma[:, 1], s=1.0, c="C0", alpha=0.45,
                   linewidths=0, zorder=9)
        c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
        ax.scatter(*c_xy, c="k", s=16, zorder=9)
        ell = _soma_ellipse2d(skel.soma, plane, scale=scl_skel)
        ell.set_edgecolor("k"); ell.set_facecolor("none"); ell.set_linestyle("--"); ell.set_linewidth(0.8)
        ax.add_patch(ell)

    # ────────────── cosmetics & labels ────────────────────────────────────
    ax.set_aspect("equal")
    if unit is None:
        ax.set_xlabel(f"{plane[0]}")
        ax.set_ylabel(f"{plane[1]}")
    else:
        ax.set_xlabel(f"{plane[0]} ({unit})")
        ax.set_ylabel(f"{plane[1]} ({unit})")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


## Plot Three Views

def _axis_extents(v: np.ndarray):
    """Return min/max tuples and ranges along x, y, z of *v* (μm)."""
    gx = (v[:, 0].min(), v[:, 0].max())   # x-limits
    gy = (v[:, 1].min(), v[:, 1].max())   # y-limits
    gz = (v[:, 2].min(), v[:, 2].max())   # z-limits
    dx, dy, dz = np.ptp(v, axis=0)        # ranges
    return dict(x=gx, y=gy, z=gz), dict(x=dx, y=dy, z=dz)

def _plane_axes(plane: str) -> tuple[str, str]:
    """Return (horizontal_axis, vertical_axis) for a 2-letter plane code."""
    if len(plane) != 2 or any(c not in "xyz" for c in plane.lower()):
        raise ValueError(f"invalid plane spec '{plane}'")
    return plane[0].lower(), plane[1].lower()

def threeviews(
    skel: Skeleton,
    mesh: trimesh.Trimesh|None = None,
    *,
    planes: tuple[str, str, str] | list[str] = ["xy", "xz", "zy"],
    scale: float | tuple[float, float] = 1.,                 # nm → µm by default
    title: str | None = None,
    figsize: tuple[int, int] = (8, 8),
    draw_edges: bool = True,
    draw_cylinders: bool = False,
    draw_soma_mask: bool = True,
    **plot_kwargs,
):
    """
    2 × 2 mosaic of orthogonal projections (A, B, C panels).

    Layout::

        B .
        A C

    By default this shows **A = yx**, **B = yz**, **C = zx**, matching the
    classic neuroanatomy view (sagittal, coronal, axial).

    Parameters
    ----------
    skel, mesh
        Skeleton and surface mesh to visualise.  ``skel`` can be *None* if
        you only want coloured surface clusters.
    planes
        Three distinct plane codes (any of ``"xy" "yx" "xz" "zx" "yz" "zy"``)
        that map, in order, to panels **A**, **B**, **C**.
    scale
        Coordinate conversion factor applied *once* to the mesh for limits
        (and forwarded to the projection helper).
    title
        Optional super-title.
    figsize
        Size of the whole mosaic figure in inches.
    draw_edges, draw_soma_mask
        Passed straight to :pyfunc:`plot_components_projection`.
    **plot_kwargs
        Any additional keyword arguments accepted by
        :pyfunc:`plot_components_projection` (e.g. ``show_node_ids``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict[str, matplotlib.axes.Axes]
        Figure plus the mapping ``{"A": axA, "B": axB, "C": axC}``.
    """
    planes = list(planes)
    if len(planes) != 3:
        raise ValueError("planes must be a sequence of exactly three plane strings")

    if not isinstance(scale, Sequence):
        scale = [scale, scale]
    if len(scale) != 2:
        raise ValueError("scale must be a scalar or a pair of two scalars")
    scl_skel, scl_mesh = map(float, scale)

    if title is None:
        title = skel.meta.get("id", None)

    # ── 0. global bounding box (already scaled) ────────────────────────────
    if mesh is not None and mesh.vertices.size:
        v_mesh = mesh.vertices.view(np.ndarray) * scl_mesh
        v_all  = np.vstack((v_mesh, skel.nodes * scl_skel))
    else:
        v_all = skel.nodes * scl_skel

    lims, spans = _axis_extents(v_all)

    # helper – pick window limits for a given plane string
    def _limits(p: str):
        h, v = _plane_axes(p)
        return lims[h], lims[v]  # returns (xlim, ylim)

    # ── 1. gridspec ratios derived from the chosen planes ──────────────────
    A, B, C = planes                           # unpack for readability
    _, vA = _plane_axes(A)
    _, vB = _plane_axes(B)
    hA, _ = _plane_axes(A)
    hC, _ = _plane_axes(C)

    height_ratios = [spans[vB], spans[vA]]     # row0, row1
    width_ratios  = [spans[hA], spans[hC]]     # col0, col1

    mosaic = """
    B.
    AC
    """

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=figsize,
        gridspec_kw={
            "height_ratios": height_ratios,
            "width_ratios":  width_ratios,
        },
    )

    # ── 2. render every occupied panel ─────────────────────────────────────
    for label, plane in zip(("A", "B", "C"), planes):
        xlim, ylim = _limits(plane)
        projection(
            skel,
            mesh,
            plane=plane,
            scale=scale,
            ax=axd[label],
            xlim=xlim,
            ylim=ylim,
            draw_edges=draw_edges,
            draw_cylinders=draw_cylinders,
            draw_soma_mask=draw_soma_mask,
            **plot_kwargs,
        )
        axd[label].set_aspect("equal")

    # ── 3. cosmetic tweaks ────────────────────────────────────────────────
    axd["B"].set_xlabel("")
    axd["B"].set_xticklabels([])
    axd["C"].set_ylabel("")
    axd["C"].set_yticklabels([])

    if title is not None:
        fig.suptitle(title, y=0.98)

    fig.tight_layout()
    return fig, axd

## Zoomed-in details on a single node

def _window_for_node(
    skel: Skeleton,
    node_id: int,
    plane: str = "xy",
    *,
    multiplier: float = 1.0,
    scl_skel: float = 1.0,          # ← same scale you pass to `details`
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return (xlim, ylim) that encloses *multiplier × soma_radius* around
    *node_id* in the requested 2-D plane, **after** applying `scl_skel`.
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")

    ix, iy = _PLANE_AXES[plane]          # which coordinates are horizontal / vertical?
    r = skel.soma.spherical_radius * multiplier * scl_skel
    node = skel.nodes[node_id] * scl_skel

    xlim = (node[ix] - r, node[ix] + r)
    ylim = (node[iy] - r, node[iy] + r)
    return xlim, ylim

def node_details(
    skel: Skeleton,
    mesh: trimesh.Trimesh | None = None,
    node_id: int = 0,
    *,
    plane: str = "xy",
    multiplier: float = 0.25,
    scale: Number | tuple[Number, Number] | Sequence[Number] = 1.0,
    highlight_alpha: float = 0.5,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Zoomed‑in view of a specific skeleton node (mesh optional).

    A surface `mesh` can now be omitted.  The zoom window is derived purely
    from the node coordinates and soma radius inside the skeleton; if a mesh is
    provided, coloured clusters and the soma shell will be drawn as in the
    full‑overview plot.
    """

    # normalise *scale* into (s_skel, s_mesh)
    if isinstance(scale, (int, float)):
        scl_skel, _ = float(scale), float(scale)
    else:
        if len(scale) != 2:
            raise ValueError("scale must be a scalar or a pair/list of two scalars")
        scl_skel = float(scale[0])

    xlim, ylim = _window_for_node(
        skel, node_id,
        plane=plane,
        multiplier=multiplier,
        scl_skel=scl_skel,
    )

    fig, ax = details(
        skel,
        mesh,  # may be None – handled by refactored `details()`
        plane=plane,
        xlim=xlim,
        ylim=ylim,
        scale=scale,
        highlight_nodes=node_id,
        highlight_face_alpha=highlight_alpha,
        **kwargs,
    )
    return fig, ax



