"""Figures of skeletons and meshes, drawn with matplotlib.

The interactive web viewer is :mod:`skeliner.viewer`.
"""

import importlib
import sys
import warnings
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse

from .dataclass import Mesh, MeshComponents, Organelles, Skeleton

__all__ = [
    "projection",
    "threeviews",
    "details",
    "node_details",
    "z_section",
    "z_section_grid",
    "diagnose_soma",
    "diagnose_components",
    "diagnose_discarded",
]


Number = int | float

_PLANE_AXES = {
    "xy": (0, 1),
    "yx": (1, 0),
    "xz": (0, 2),
    "zx": (2, 0),
    "yz": (1, 2),
    "zy": (2, 1),
}

_PLANE_NORMAL = {
    "xy": np.array([0, 0, 1.0]),
    "yx": np.array([0, 0, 1.0]),
    "xz": np.array([0, 1.0, 0]),
    "zx": np.array([0, 1.0, 0]),
    "yz": np.array([1.0, 0, 0]),
    "zy": np.array([1.0, 0, 0]),
}


_GOLDEN_RATIO = 0.618033988749895  # for visually distinct colours


def _project(arr: np.ndarray, ix: int, iy: int, /) -> np.ndarray:
    """Return 2-column slice (arr[:, (ix, iy)])."""
    return arr[:, (ix, iy)].copy()


def _component_labels(n_verts: int, node2verts: list[np.ndarray]) -> np.ndarray:
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
    bbox = ax.get_window_extent()

    ppd_x = bbox.width / abs(x1 - x0)
    ppd_y = bbox.height / abs(y1 - y0)
    ppd = min(ppd_x, ppd_y)

    r_px = rr * ppd
    r_pt = r_px * 72.0 / dpi
    return np.pi * r_pt**2, ppd


def _trapezoid_3d(
    p0_3d: np.ndarray,
    p1_3d: np.ndarray,
    r0: float,
    r1: float,
    plane: str,
) -> np.ndarray | None:
    """
    Return 4×2 vertices of the projected trapezoid that joins the two circles
    centred at *p0_3d* and *p1_3d* with radii *r0*, *r1*.

    The bases are perpendicular to the true 3-D axis, so their projected
    length equals the node diameters.  Returns **None** for zero-length edges.
    """
    v = p1_3d - p0_3d
    if not np.any(v):  # degenerate edge
        return None

    n3 = np.cross(v, _PLANE_NORMAL[plane])
    L = np.linalg.norm(n3)
    if L == 0:  # edge is parallel to the projection normal
        return None
    n3 /= L  # now unit length in 3-D

    # project to 2-D
    ix, iy = _PLANE_AXES[plane]
    n2 = n3[[ix, iy]]

    p0 = p0_3d[[ix, iy]]
    p1 = p1_3d[[ix, iy]]

    return np.array(
        [
            p0 + n2 * r0,
            p0 - n2 * r0,
            p1 - n2 * r1,
            p1 + n2 * r1,
        ],
        dtype=float,
    )


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
    k = 3 - ix - iy  # the coordinate we project away

    # inverse shape matrix of the ellipsoid
    B = soma.R @ np.diag(1.0 / soma.axes**2) @ soma.R.T

    B_pp = B[[ix, iy]][:, [ix, iy]]  # 2×2
    B_pq = B[[ix, iy], k].reshape(2, 1)  # 2×1
    B_qq = B[k, k]

    Q = B_pp - (B_pq @ B_pq.T) / B_qq  # 2×2 positive-definite

    # eigen-decomposition → half-axes in the projection plane
    eigval, eigvec = np.linalg.eigh(Q)  # λ₁, λ₂ > 0
    half_axes = 1.0 / np.sqrt(eigval)  # r₁, r₂
    order = np.argsort(-half_axes)  # big → small

    width, height = 2 * half_axes[order] * scale
    angle_deg = np.degrees(np.arctan2(eigvec[1, order[0]], eigvec[0, order[0]]))
    centre_xy = soma.center[[ix, iy]] * scale

    return Ellipse(
        centre_xy,
        width,
        height,
        angle=angle_deg,
        linewidth=0.8,
        linestyle="--",
        facecolor="none",
        edgecolor="k",
        alpha=0.9,
    )


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


def _as_cmap(cmap_like) -> mcolors.Colormap:
    """
    Normalize a matplotlib colormap-like input to a Colormap.
    Accepts: str name, Colormap instance, or a sequence of color specs.
    """
    if isinstance(cmap_like, mcolors.Colormap):
        return cmap_like
    if isinstance(cmap_like, (list, tuple, np.ndarray)):
        return mcolors.ListedColormap(cmap_like)
    # name-like
    try:
        return mpl.colormaps.get_cmap(str(cmap_like))
    except Exception:
        # fallback for older Matplotlib
        return mpl.colormaps.get_cmap(str(cmap_like))


def _resample_n(cmap: mcolors.Colormap, n: int) -> mcolors.Colormap:
    """
    Return a version of *cmap* with *n* discrete colors. Uses Colormap.resampled
    when available; otherwise builds a ListedColormap by sampling.
    """
    try:
        return cmap.resampled(n)  # Matplotlib ≥ 3.5
    except Exception:
        return mcolors.ListedColormap(cmap(np.linspace(0.0, 1.0, n)))


_SWC_ALIASES = {
    "root": -1,
    "undefined": 0,
    "undef": 0,
    "unknown": 0,
    "soma": 1,
    "axon": 2,
    "dendrite": 3,
    "basal": 3,
    "apical": 4,
    "fork": 5,
    "branchpoint": 5,
    "bifurcation": 5,
    "end": 6,
    "terminal": 6,
    "tip": 6,
}

# Keep stable aesthetic order (your previous mapping of SWC→index)
_DEFAULT_IDX_BY_TYPE = {-1: 7, 0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def _key_to_swc_typecode(key: int | str) -> int:
    if isinstance(key, int):
        if -1 <= key <= 6:
            return key
        raise ValueError("SWC type code must be in -1..6")
    k = key.strip().lower()
    if k not in _SWC_ALIASES:
        valid = ", ".join(sorted(set(_SWC_ALIASES)))
        raise ValueError(f"Unknown SWC type name '{key}'. Valid: {valid}")
    return _SWC_ALIASES[k]


def _palette_from_base(base_cmap_like) -> np.ndarray:
    """Return (7,4) RGBA palette ordered by SWC -1..6 from a base colormap."""
    base = _resample_n(_as_cmap(base_cmap_like), 8)
    rows = (
        np.asarray(base.colors)
        if hasattr(base, "colors")
        else base(np.linspace(0, 1, 8))
    )
    swc = np.empty((8, 4), float)
    for t in range(-1, 7):
        swc[t] = rows[_DEFAULT_IDX_BY_TYPE[t]]
    return swc


def _resolve_swc_palette_from_skel_cmap(skel_cmap) -> np.ndarray:
    """
    Accepts:
      - str / Colormap / sequence → derive 8-color palette
      - dict {name|code: color, ...} with optional '__base__'
    Returns (8,4) RGBA array indexed by SWC code -1..6.
    """
    if isinstance(skel_cmap, Mapping):
        base_arg = skel_cmap.get("__base__", "Pastel2")  # default fallback
        palette = _palette_from_base(base_arg)
        for k, v in skel_cmap.items():
            if k == "__base__":
                continue
            t = _key_to_swc_typecode(k)
            palette[t] = mcolors.to_rgba(v)
        return palette

    # not a dict → treat as cmap-like and derive palette
    return _palette_from_base(skel_cmap)


def projection(
    skel: Skeleton,
    mesh: Mesh | None = None,
    *,
    plane: str = "xy",
    radius_metric: str | None = None,
    bins: int | tuple[int, int] = 800,
    scale: float | Sequence[Number] = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    draw_skel: bool = True,
    draw_mesh: bool = True,
    draw_edges: bool = True,
    draw_cylinders: bool = False,
    rasterized: bool = True,
    ax: Axes | None = None,
    mesh_cmap: str
    | mcolors.Colormap
    | Sequence[str]
    | Mapping[str, str] = "Blues",  # mesh color map
    skel_cmap: str
    | mcolors.Colormap
    | Sequence[str]
    | Mapping[str, str] = "Pastel2",  # skeleton color map
    vmax_fraction: float = 0.10,
    edge_lw: float = 0.5,
    circle_alpha: float = 0.25,
    cylinder_alpha: float = 0.5,
    highlight_nodes: int | Sequence[int] | None = None,
    highlight_face_alpha: float = 0.5,
    unit: str | None = None,
    # soma --------------------------------------------------------------- #
    draw_soma_mask: bool = True,
    soma_style: str = "dashed",  # "dashed" | "filled"
    # colors
    color_by: str = "fixed",  # "ntype" or "fixed"
) -> tuple[Figure, Axes]:
    """Orthographic 2‑D overview of a skeleton with an **optional** mesh‑density
    background.

    Parameters
    ----------
    skel : Skeleton
        The centre‑line skeleton to visualise.
    mesh : Mesh | None, default *None*
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
    draw_skel, draw_mesh, draw_edges, draw_cylinders: bool
        Toggles for skeleton glyphs.
    soma_style : str
        How to plot the soma, currently supported styles are:
        - "dashed" : dashed ellipse outline (default)
        - "filled" : filled circle with soma colour
    rasterized : bool | int
        Rasterize skeleton. If int and > 1, not only skeleton will be rasterized.
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

    if unit is None:  # try to grab from metadata
        unit = skel.meta.get("unit", None)

    highlight_set = (
        set(map(int, np.atleast_1d(highlight_nodes)))
        if highlight_nodes is not None
        else set()
    )

    # ─────────────────────────────colormap ─────────────
    swc_colors = _resolve_swc_palette_from_skel_cmap(skel_cmap)

    # ───────────────────────────── project (and optionally crop) ───────────
    xy_skel = _project(skel.nodes, ix, iy) * scl_skel
    rr = skel.radii[radius_metric] * scl_skel

    if mesh is not None and draw_mesh:
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
    keep_mask = keep_skel  # ← keep the original name for edges
    idx_keep = np.flatnonzero(keep_mask)  # 1-D array of kept node IDs
    xy_skel = xy_skel[keep_mask]  # already done
    rr = rr[keep_mask]  # already done

    # colour array for the *kept* nodes
    if color_by == "ntype" and skel.ntype is not None:
        col_nodes = swc_colors[skel.ntype[idx_keep]]
    else:
        col_nodes = "red"

    if mesh is not None and xy_mesh.size and draw_mesh:
        keep_mesh = _crop_window(xy_mesh)
        xy_mesh = xy_mesh[keep_mesh]

    # ─────────────────────────────── histogram (mesh may be None) ──────────
    if mesh is not None and xy_mesh.size and draw_mesh:
        # ensure bins argument correct
        if isinstance(bins, int):
            bins_arg: int | tuple[int, int] = bins
        elif (
            isinstance(bins, tuple)
            and len(bins) == 2
            and all(isinstance(b, int) for b in bins)
        ):
            bins_arg = (int(bins[0]), int(bins[1]))
        else:
            raise ValueError("bins must be an int or a tuple of two ints")

        from scipy.stats import binned_statistic_2d  # heavy import lazily

        hist, xedges, yedges, _ = binned_statistic_2d(
            xy_mesh[:, 0],
            xy_mesh[:, 1],
            None,
            statistic="count",
            bins=bins_arg,
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
    if hist is not None and draw_mesh:
        ax.imshow(
            hist,
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            origin="lower",
            cmap=_as_cmap(mesh_cmap),
            vmax=hist.max() * vmax_fraction,
            alpha=1.0,
            rasterized=rasterized > 1,
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
            edgecolors=col_nodes[1:]
            if isinstance(col_nodes, np.ndarray)
            else col_nodes,
            linewidths=1.0,
            alpha=circle_alpha,
            zorder=2,
            rasterized=rasterized > 0,
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
                    rasterized=rasterized > 1,
                )

    # ───────────────────────── soma shell & center (if possible) ───────────
    c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
    col_soma = swc_colors[1] if color_by == "ntype" else "pink"

    if soma_style == "filled":
        soma_fc = col_soma
        soma_ec = "k"
        soma_ls = "-"
        soma_mc = "none"
    else:
        ax.scatter(*c_xy, color="black", s=15, zorder=3)
        soma_fc = "none"
        soma_ec = "k"
        soma_ls = "--"
        soma_mc = col_soma

    if mesh is not None and skel.soma is not None and skel.soma.verts is not None:
        if draw_soma_mask:
            xy_soma = _project(mesh.vertices[np.asarray(skel.soma.verts, int)], ix, iy)
            xy_soma = xy_soma * scl_mesh
            xy_soma = xy_soma[_crop_window(xy_soma)]  # respect crop

            ax.scatter(
                xy_soma[:, 0],
                xy_soma[:, 1],
                s=1,
                c=[soma_mc],
                alpha=0.5,
                linewidths=0,
                label="soma surface",
                rasterized=rasterized > 0,
            )
        # dashed ellipse outline
        ell = _soma_ellipse2d(skel.soma, plane, scale=scl_skel)
        ell.set_edgecolor(soma_ec)
        ell.set_facecolor(soma_fc)
        ell.set_linestyle(soma_ls)
        ell.set_linewidth(0.8)
        ell.set_alpha(0.9)
        ax.add_patch(ell)
    else:
        soma_circle = Circle(
            c_xy,
            skel.soma.equiv_radius * scl_skel,
            facecolor=soma_fc,
            edgecolor=soma_ec,
            linewidth=0.8,
            linestyle=soma_ls,
            alpha=0.9,
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
                seg_end = xy_skel[idx_map[edges_kept[:, 1]]]
                segments = np.stack((seg_start, seg_end), axis=1)

                lc = LineCollection(
                    segments.tolist(),
                    colors="black",
                    linewidths=edge_lw,
                    alpha=cylinder_alpha,
                    rasterized=rasterized > 0,
                )
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
                        rr[i0],
                        rr[i1],
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

                    pc = PolyCollection(
                        quads,
                        facecolors="red",
                        edgecolors="red",
                        alpha=cylinder_alpha,
                        zorder=10,
                        rasterized=rasterized > 0,
                    )
                    ax.add_collection(pc)

    # ────────────────────────────── final cosmetics ────────────────────────
    if plane in ["xy", "yx"]:
        ax.set_aspect("equal", adjustable="box")

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
    mesh: Mesh | None = None,
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

    if unit is None:  # try to grab from metadata
        unit = skel.meta.get("unit", None)

    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    highlight_set = (
        set(map(int, np.atleast_1d(highlight_nodes)))
        if highlight_nodes is not None
        else set()
    )

    # ────────────── project skeleton (mandatory) ──────────────────────────
    xy_skel = _project(skel.nodes, ix, iy) * scl_skel
    rr = skel.radii[radius_metric] * scl_skel

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
    xy_skel = xy_skel[keep_skel]
    rr = rr[keep_skel]

    if have_mesh:
        keep_mesh = _mask_window(xy_mesh_all)
        xy_mesh_crop = xy_mesh_all[keep_mesh]
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
            xy_mesh_crop[:, 0],
            xy_mesh_crop[:, 1],
            None,
            statistic="count",
            bins=bins_arg,
        )
        hist = hist.T  # imshow rows=y
    else:
        hist = None

    # ────────────── component / cluster labels (optional) ─────────────────
    if have_mesh and skel.node2verts is not None and len(xy_mesh_crop):
        lab_full = _component_labels(len(mesh.vertices), skel.node2verts)
        in_cluster = lab_full >= 0
        mask_mesh = keep_mesh & in_cluster
        xy_mesh_scatter = xy_mesh_all[mask_mesh]
        lab_mesh = lab_full[mask_mesh]
        n_comp = int(lab_full.max() + 1)
    else:
        xy_mesh_scatter = xy_mesh_crop  # possibly empty
        lab_mesh = None
        n_comp = 0

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
            xy_mesh_scatter[:, 0],
            xy_mesh_scatter[:, 1],
            s=1.0,
            c=colours,
            alpha=0.75,
            linewidths=0,
            zorder=9,
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
            node_comp = np.array(
                [
                    lab_full[skel.node2verts[i][0]] if len(skel.node2verts[i]) else -1
                    for i in range(len(skel.nodes))
                ]
            )
            node_comp = node_comp[keep_skel]
            node_colors = lut[node_comp]
        else:
            # Keep fallback colors per-node so downstream slicing never
            # turns a color string (e.g. "red") into an invalid one ("ed").
            fallback = mcolors.to_rgba("red")
            node_colors = np.repeat([fallback], xy_skel.shape[0], axis=0)

        slicing = 0 if skel.soma.verts is None else 1
        ax.scatter(
            xy_skel[:, 0][slicing:],
            xy_skel[:, 1][slicing:],
            s=sizes[slicing:],
            facecolors="none",
            edgecolors=node_colors[slicing:],
            linewidths=0.9,
            alpha=circle_alpha,
            zorder=3,
        )
        ax.scatter(
            xy_skel[:, 0],
            xy_skel[:, 1],
            s=10,
            c=node_colors,
            alpha=circle_alpha,
            zorder=4,
            linewidths=0,
        )

        # optional node‑ID labels
        if show_node_ids:
            orig_ids = np.flatnonzero(keep_skel)
            dx, dy = id_offset
            for i_compressed, nid in enumerate(orig_ids):
                color = highlight_id_color if nid in highlight_set else id_color
                x, y = xy_skel[i_compressed]
                ax.text(
                    x + dx,
                    y + dy,
                    str(nid),
                    fontsize=id_fontsize,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=5,
                )

        # fill highlighted nodes
        if highlight_set:
            orig_ids = np.flatnonzero(keep_skel)
            hilite_mask = np.isin(orig_ids, list(highlight_set))
            if hilite_mask.any():
                ax.scatter(
                    xy_skel[hilite_mask, 0],
                    xy_skel[hilite_mask, 1],
                    s=sizes[hilite_mask],
                    facecolors=node_colors[hilite_mask] if n_comp else "green",
                    edgecolors=node_colors[hilite_mask] if n_comp else "green",
                    linewidths=0.9,
                    alpha=highlight_face_alpha,
                    zorder=4.5,
                )

    # ────────────── edges & cylinders (use skeleton only) ─────────────────
    if draw_edges and skel.edges.size:
        keep_flags = keep_skel
        ekeep = keep_flags[skel.edges[:, 0]] & keep_flags[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]
        if edges_kept.size:
            idx_map = -np.ones(len(keep_flags), int)
            idx_map[np.flatnonzero(keep_flags)] = np.arange(keep_flags.sum())
            seg_start = xy_skel[idx_map[edges_kept[:, 0]]]
            seg_end = xy_skel[idx_map[edges_kept[:, 1]]]
            segs = np.stack((seg_start, seg_end), axis=1)
            lc = LineCollection(
                segs,
                colors=edge_color,
                linewidths=edge_lw,
                alpha=circle_alpha * 0.9,
                zorder=2,
            )
            ax.add_collection(lc)

    if draw_nodes and draw_cylinders and skel.edges.size:
        ekeep = keep_skel[skel.edges[:, 0]] & keep_skel[skel.edges[:, 1]]
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
                    rr[i0],
                    rr[i1],
                    plane,
                )
                if quad is None:
                    continue
                quads.append(quad)
                facecols.append(node_colors[i0] if draw_nodes and n_comp else "0.25")

            if quads:
                pc = PolyCollection(
                    quads,
                    facecolors=facecols,
                    edgecolors="none",
                    alpha=cylinder_alpha,
                    zorder=2.8,
                )
                ax.add_collection(pc)

    # ────────────── optional soma shell (need mesh) ───────────────────────
    if (
        draw_soma_mask
        and have_mesh
        and skel.soma is not None
        and skel.soma.verts is not None
    ):
        xy_soma = _project(mesh.vertices[np.asarray(skel.soma.verts, int)], ix, iy)
        xy_soma = xy_soma * scl_mesh
        soma_keep = _mask_window(xy_soma)
        xy_soma = xy_soma[soma_keep]
        ax.scatter(
            xy_soma[:, 0],
            xy_soma[:, 1],
            s=1.0,
            c="C0",
            alpha=0.45,
            linewidths=0,
            zorder=9,
        )
        c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
        ax.scatter(*c_xy, c="k", s=16, zorder=9)
        ell = _soma_ellipse2d(skel.soma, plane, scale=scl_skel)
        ell.set_edgecolor("k")
        ell.set_facecolor("none")
        ell.set_linestyle("--")
        ell.set_linewidth(0.8)
        ax.add_patch(ell)

    # ────────────── cosmetics & labels ────────────────────────────────────
    if plane in ["xy", "yx"]:
        ax.set_aspect("equal", adjustable="box")

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
    gx = (v[:, 0].min(), v[:, 0].max())  # x-limits
    gy = (v[:, 1].min(), v[:, 1].max())  # y-limits
    gz = (v[:, 2].min(), v[:, 2].max())  # z-limits
    dx, dy, dz = np.ptp(v, axis=0)  # ranges
    return dict(x=gx, y=gy, z=gz), dict(x=dx, y=dy, z=dz)


def _plane_axes(plane: str) -> tuple[str, str]:
    """Return (horizontal_axis, vertical_axis) for a 2-letter plane code."""
    if len(plane) != 2 or any(c not in "xyz" for c in plane.lower()):
        raise ValueError(f"invalid plane spec '{plane}'")
    return plane[0].lower(), plane[1].lower()


def threeviews(
    skel: Skeleton,
    mesh: Mesh | None = None,
    *,
    planes: tuple[str, str, str] | list[str] = ["xy", "xz", "zy"],
    scale: float | tuple[float, float] = 1.0,  # nm → µm by default
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
        v_all = np.vstack((v_mesh, skel.nodes * scl_skel))
    else:
        v_all = skel.nodes * scl_skel

    lims, spans = _axis_extents(v_all)

    # helper – pick window limits for a given plane string
    def _limits(p: str):
        h, v = _plane_axes(p)
        return lims[h], lims[v]  # returns (xlim, ylim)

    # ── 1. gridspec ratios derived from the chosen planes ──────────────────
    A, B, C = planes  # unpack for readability
    _, vA = _plane_axes(A)
    _, vB = _plane_axes(B)
    hA, _ = _plane_axes(A)
    hC, _ = _plane_axes(C)

    height_ratios = [spans[vB], spans[vA]]  # row0, row1
    width_ratios = [spans[hA], spans[hC]]  # col0, col1

    mosaic = """
    B.
    AC
    """

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=figsize,
        gridspec_kw={
            "height_ratios": height_ratios,
            "width_ratios": width_ratios,
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
    scl_skel: float = 1.0,  # ← same scale you pass to `details`
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return (xlim, ylim) that encloses *multiplier × soma_radius* around
    *node_id* in the requested 2-D plane, **after** applying `scl_skel`.
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")

    ix, iy = _PLANE_AXES[plane]  # which coordinates are horizontal / vertical?
    r = skel.soma.spherical_radius * multiplier * scl_skel
    node = skel.nodes[node_id] * scl_skel

    xlim = (node[ix] - r, node[ix] + r)
    ylim = (node[iy] - r, node[iy] + r)
    return xlim, ylim


def node_details(
    skel: Skeleton,
    mesh: Mesh | None = None,
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
        skel,
        node_id,
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


# =====================================================================
#  Mesh cross-section visualization
# =====================================================================


def z_section(
    mesh: Mesh,
    z: float | None = None,
    *,
    z_tol: float = 150.0,
    organelles: Organelles | None = None,
    nucleus: dict | None = None,
    soma_hull: bool = False,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[Figure, Axes]:
    """Plot a Z cross-section of the mesh vertex cloud.

    Shows all mesh vertices near *z* as a 2D scatter, with a
    concave-hull outer contour, optional organelle overlay, and
    optional nucleus void circle from :func:`find_nucleus_center`.

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh.
    z : float
        Z-level to slice at.
    z_tol : float
        Half-width of the Z slab for collecting vertices.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.  If
        provided, organelle face centroids are drawn in red.
    nucleus : dict or None
        Result dict from :func:`find_nucleus_center`.  If provided,
        the per-Z void circle is drawn at the nearest matching slice.
    soma_hull : bool
        If True, apply fill + morphological opening to strip neurite
        protrusions before building the outer contour hull.  This
        matches the hull computation used by ``find_soma_via_z_contour``.
    ax : Axes or None
        Matplotlib axes to draw on.  Created if None.
    figsize : tuple
        Figure size when *ax* is None.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if z is None:
        if nucleus is not None:
            z = nucleus["center"][2]
        else:
            raise ValueError("z must be provided when nucleus is None")

    from scipy.ndimage import binary_dilation, label
    from scipy.spatial import ConvexHull

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    near_z = np.abs(verts[:, 2] - z) < z_tol
    pts_xy = verts[near_z, :2]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if len(pts_xy) < 3:
        ax.set_title(f"Z={z:.0f}: too few vertices")
        return fig, ax

    # All vertices — light gray
    ax.scatter(
        pts_xy[:, 0],
        pts_xy[:, 1],
        c="#cccccc",
        s=0.5,
        alpha=0.3,
        zorder=1,
        rasterized=True,
    )

    # Organelle overlay
    if organelles is not None:
        centroids = verts[faces].mean(axis=1)
        face_near = np.abs(centroids[:, 2] - z) < z_tol
        org_near = face_near & organelles.mask
        if org_near.sum() > 0:
            oc = centroids[org_near]
            ax.scatter(
                oc[:, 0],
                oc[:, 1],
                c="red",
                s=2,
                alpha=0.4,
                zorder=2,
                rasterized=True,
                label="organelles",
            )

    # Outer contour — rasterize, find largest cluster, convex hull.
    # When soma_hull=True, fill interior + morphological open to strip
    # neurite protrusions before hull (matches _soma_hulls()).
    soma_pts = None
    if len(pts_xy) >= 4:
        try:
            grid_res = 200.0
            struct3 = np.ones((3, 3), dtype=bool)
            xy_min = pts_xy.min(axis=0) - grid_res * 3
            nx = int((pts_xy[:, 0].max() - xy_min[0] + grid_res * 6) / grid_res) + 1
            ny = int((pts_xy[:, 1].max() - xy_min[1] + grid_res * 6) / grid_res) + 1
            occ = np.zeros((nx, ny), dtype=bool)
            ix = ((pts_xy[:, 0] - xy_min[0]) / grid_res).astype(int).clip(0, nx - 1)
            iy = ((pts_xy[:, 1] - xy_min[1]) / grid_res).astype(int).clip(0, ny - 1)
            occ[ix, iy] = True

            dilated = binary_dilation(occ, struct3)
            labeled, n_labels = label(dilated)
            if n_labels > 0:
                sizes = np.bincount(labeled.ravel())[1:]
                vert_labels = labeled[ix, iy]
                lid = int(np.argmax(sizes)) + 1
                soma_region = labeled == lid
                soma_mask = vert_labels == lid

                if soma_hull:
                    from scipy.ndimage import (
                        binary_erosion,
                        binary_fill_holes,
                        distance_transform_edt,
                    )

                    closed = binary_dilation(occ & soma_region, struct3, iterations=2)
                    closed = binary_erosion(closed, struct3, iterations=2)
                    filled = binary_fill_holes(closed)
                    dt = distance_transform_edt(filled)
                    peak_dist = dt.max()
                    if peak_dist >= 6:
                        r = max(int(peak_dist * 0.3), 3)
                        se = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
                        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
                        se[xx**2 + yy**2 <= r**2] = True
                        opened = binary_erosion(filled, se)
                        opened = binary_dilation(opened, se)
                        if opened.any():
                            gi, gj = np.where(opened)
                            hull_pts = np.column_stack(
                                [
                                    xy_min[0] + gi * grid_res + grid_res / 2,
                                    xy_min[1] + gj * grid_res + grid_res / 2,
                                ]
                            )
                            hull = ConvexHull(hull_pts)
                            hv = hull_pts[hull.vertices]
                            hv_closed = np.vstack([hv, hv[0:1]])
                            ax.plot(
                                hv_closed[:, 0],
                                hv_closed[:, 1],
                                color="green",
                                linewidth=2.5,
                                zorder=5,
                                label="outer contour",
                            )
                            soma_mask = soma_mask & opened[ix, iy]
                            soma_pts = pts_xy[soma_mask]

                # Fallback: raw convex hull of largest cluster
                if soma_pts is None:
                    soma_pts = pts_xy[soma_mask]
                    if len(soma_pts) >= 4:
                        hull = ConvexHull(soma_pts)
                        hv = soma_pts[hull.vertices]
                        hv_closed = np.vstack([hv, hv[0:1]])
                        ax.plot(
                            hv_closed[:, 0],
                            hv_closed[:, 1],
                            color="green",
                            linewidth=2.5,
                            zorder=5,
                            label="outer contour",
                        )
        except Exception:
            pass

    # Nucleus void — inner boundary of soma cluster from void center,
    # clipped to the soma convex hull so it can't escape through the
    # open pocket mouth.
    if nucleus is not None and len(pts_xy) >= 10:
        from shapely.geometry import Polygon

        slices = nucleus["slices"]
        dz = np.abs(slices[:, 0] - z)
        nearest = int(np.argmin(dz))
        z_step = np.median(np.diff(slices[:, 0])) if len(slices) > 1 else 500
        if dz[nearest] <= z_step * 0.6:
            vc = slices[nearest, 1:3]
            # Use soma cluster vertices only (soma_pts from above)
            sweep_pts = soma_pts if soma_pts is not None else pts_xy
            dx = sweep_pts[:, 0] - vc[0]
            dy = sweep_pts[:, 1] - vc[1]
            angles = np.arctan2(dy, dx)
            dists = np.sqrt(dx**2 + dy**2)
            n_bins = 72
            bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
            inner_pts = []
            for b in range(n_bins):
                mask = (angles >= bin_edges[b]) & (angles < bin_edges[b + 1])
                if mask.any():
                    idx = np.where(mask)[0]
                    closest = idx[np.argmin(dists[idx])]
                    inner_pts.append(sweep_pts[closest])
            if len(inner_pts) >= 3:
                inner_pts = np.array(inner_pts)
                inner_closed = np.vstack([inner_pts, inner_pts[0:1]])
                # Clip to soma convex hull
                try:
                    inner_poly = Polygon(inner_closed[:, :2])
                    if not inner_poly.is_valid:
                        inner_poly = inner_poly.buffer(0)
                    hull_poly = Polygon(hv) if soma_pts is not None else None
                    if hull_poly is not None and hull_poly.is_valid:
                        clipped = inner_poly.intersection(hull_poly)
                        if hasattr(clipped, "exterior"):
                            cc = np.array(clipped.exterior.coords)
                            ax.plot(
                                cc[:, 0],
                                cc[:, 1],
                                color="goldenrod",
                                linewidth=2.5,
                                zorder=5,
                                label="nucleus void",
                            )
                        else:
                            ax.plot(
                                inner_closed[:, 0],
                                inner_closed[:, 1],
                                color="goldenrod",
                                linewidth=2.5,
                                zorder=5,
                                label="nucleus void",
                            )
                    else:
                        ax.plot(
                            inner_closed[:, 0],
                            inner_closed[:, 1],
                            color="goldenrod",
                            linewidth=2.5,
                            zorder=5,
                            label="nucleus void",
                        )
                except Exception:
                    ax.plot(
                        inner_closed[:, 0],
                        inner_closed[:, 1],
                        color="goldenrod",
                        linewidth=2.5,
                        zorder=5,
                        label="nucleus void",
                    )

    # Auto-zoom centred on nucleus XY
    if nucleus is not None:
        nc = nucleus["center"]
        pad = nucleus["peak_r"] * 4
        ax.set_xlim(nc[0] - pad, nc[0] + pad)
        ax.set_ylim(nc[1] - pad, nc[1] + pad)
        ax.plot(
            nc[0], nc[1], "+", color="red", markersize=10, markeredgewidth=2, zorder=10
        )

    ax.set_aspect("equal")
    ax.set_title(f"Z = {z:.0f}  ({pts_xy.shape[0]} vertices)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
    return fig, ax


def z_section_grid(
    mesh: Mesh,
    nucleus: dict,
    *,
    z_step: float = 210.0,
    z_span: float | None = None,
    z_tol: float = 150.0,
    organelles: Organelles | None = None,
    soma_hull: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot a grid of Z cross-sections spanning the nucleus Z-range.

    The Z-levels are centred on the nucleus centre and span
    *z_span* (default: 1.5 × the detected nucleus Z-range).

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh.
    nucleus : dict
        Result dict from :func:`find_nucleus_center`.
    z_step : float
        Spacing between Z-levels in nm.
    z_span : float or None
        Total Z span to cover.  If None, uses 1.5 × the nucleus
        Z-range so the void appears and disappears within the grid.
    z_tol : float
        Half-width of the Z slab at each level.
    organelles : Organelles or None
        Forwarded to :func:`z_section`.
    figsize : tuple or None
        Figure size.  If None, auto-scaled from the grid dimensions.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
    """
    z_center = nucleus["center"][2]
    if z_span is None:
        z_lo, z_hi = nucleus["z_range"]
        z_span = (z_hi - z_lo) * 1.96
        z_span = max(z_span, 5000.0)  # at least 5 µm

    n_levels = max(4, int(np.ceil(z_span / z_step)))

    ncols = int(np.ceil(np.sqrt(n_levels)))
    nrows = int(np.ceil(n_levels / ncols))

    if figsize is None:
        figsize = (ncols * 3.5, nrows * 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.asarray(axes).ravel()

    z_levels = np.linspace(z_center - z_span / 2, z_center + z_span / 2, n_levels)

    for i, z_val in enumerate(z_levels):
        z_section(
            mesh,
            z_val,
            z_tol=z_tol,
            organelles=organelles,
            nucleus=nucleus,
            soma_hull=soma_hull,
            ax=axes_flat[i],
        )

    for j in range(n_levels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig, axes


def diagnose_soma(
    mesh: Mesh,
    *,
    nucleus: dict | None = None,
    soma: object = None,
    organelles: Organelles | None = None,
    planes: tuple[str, str, str] = ("xy", "xz", "zy"),
    figsize: tuple[float, float] = (12, 12),
) -> tuple[Figure, list[Axes]]:
    """Three orthogonal projections of soma detection results.

    Shows mesh vertices in gray, soma vertices highlighted, nucleus
    center marked, nucleus Z-range indicated, and soma ellipsoid
    outline.

    Colors match the web viewer: soma is light magenta, pocket
    organelles are red, isolated organelles are green, and expanded
    organelles are orange.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    nucleus : dict or None
        Result from :func:`find_nucleus_center`.
    soma : Soma or None
        Result from any soma detection method.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.
    planes : tuple of str
        Three plane codes for the panels.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes : Figure, list[Axes]
    """
    # Color scheme — matches skeliner.viewer
    SOMA_COLOR = (0.9, 0.5, 0.9)
    POCKET_COLOR = (1.0, 0.15, 0.15)
    ISOLATED_COLOR = (0.15, 0.8, 0.15)
    EXPANDED_COLOR = (1.0, 0.6, 0.15)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # 2×2 grid: top-left=XZ, top-right=empty, bottom-left=XY, bottom-right=ZY
    # Axes alignment: XY bottom-left shares X with XZ above, Y with ZY right.
    fig, ax_grid = plt.subplots(2, 2, figsize=figsize)
    # Map: XY→(1,0), XZ→(0,0), ZY→(1,1), empty→(0,1)
    panel_pos = {"xy": (1, 0), "xz": (0, 0), "zy": (1, 1)}
    axes = []
    for plane in planes:
        r, c = panel_pos[plane]
        axes.append(ax_grid[r, c])
    ax_grid[0, 1].set_visible(False)

    for ax, plane in zip(axes, planes):
        ix, iy = _PLANE_AXES[plane]
        xlab, ylab = _plane_axes(plane)

        # All mesh vertices — light gray
        ax.scatter(
            verts[:, ix],
            verts[:, iy],
            c="#dddddd",
            s=0.3,
            alpha=0.2,
            rasterized=True,
            zorder=1,
        )

        # Soma vertices
        if soma is not None and len(soma.verts) > 0:
            sv = verts[soma.verts]
            ax.scatter(
                sv[:, ix],
                sv[:, iy],
                color=SOMA_COLOR,
                s=0.8,
                alpha=0.6,
                rasterized=True,
                zorder=3,
                label="soma",
            )

        # Organelle face centroids — split by category, viewer colors
        if organelles is not None:
            face_centroids = verts[faces].mean(axis=1)
            org_layers = (
                ("pocket", organelles.pocket, POCKET_COLOR),
                ("isolated", organelles.isolated, ISOLATED_COLOR),
                ("expanded", organelles.expanded, EXPANDED_COLOR),
            )
            for label, mask, color in org_layers:
                if not mask.any():
                    continue
                fc = face_centroids[mask]
                ax.scatter(
                    fc[:, ix],
                    fc[:, iy],
                    color=color,
                    s=0.3,
                    alpha=0.25,
                    rasterized=True,
                    zorder=2,
                    label=f"organelle:{label}",
                )

        # Soma ellipsoid outline (drawn even when organelles is None)
        if soma is not None:
            ell = _soma_ellipse2d(soma, plane)
            ell.set_edgecolor(SOMA_COLOR)
            ell.set_linewidth(1.5)
            ax.add_patch(ell)

        # Nucleus center + z_range
        if nucleus is not None:
            nc = nucleus["center"]
            ax.plot(
                nc[ix],
                nc[iy],
                "+",
                color="cyan",
                markersize=12,
                markeredgewidth=2,
                zorder=10,
            )
            # Z-range lines on planes that show Z
            if 2 in (ix, iy):
                z_lo, z_hi = nucleus["z_range"]
                z_ax = ix if ix == 2 else iy
                for zv in (z_lo, z_hi):
                    if z_ax == ix:
                        ax.axvline(
                            zv, color="cyan", linewidth=0.8, alpha=0.6, linestyle="--"
                        )
                    else:
                        ax.axhline(
                            zv, color="cyan", linewidth=0.8, alpha=0.6, linestyle="--"
                        )

        # Zoom to soma region with generous padding
        if soma is not None and len(soma.verts) > 0:
            pad = max(soma.axes) * 1.5
            cx, cy = soma.center[ix], soma.center[iy]
        elif nucleus is not None:
            pad = nucleus["peak_r"] * 8
            cx, cy = nucleus["center"][ix], nucleus["center"][iy]
        else:
            pad = None

        if pad is not None:
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)

        ax.set_aspect("equal")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(plane.upper())
        ax.grid(True, alpha=0.15)

    axes[0].legend(fontsize=7, loc="upper right", markerscale=5)
    fig.tight_layout()
    return fig, axes


def diagnose_components(
    mesh: Mesh,
    components: MeshComponents,
    *,
    planes: tuple[str, str, str] = ("xy", "xz", "zy"),
    figsize: tuple[float, float] = (12, 12),
) -> tuple[Figure, list[Axes]]:
    """Three orthogonal projections of break_up_mesh component results.

    Same layout as :func:`diagnose_soma`, but colors faces by
    component (soma, organelles, neurites) using the same palette as
    the web viewer after running break_up_mesh. Discarded fragments
    are not plotted.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    components : MeshComponents
        Result of :func:`~skeliner.pre.break_up_mesh`.
    planes : tuple of str
        Three plane codes for the panels.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes : Figure, list[Axes]
    """
    # Color scheme — matches skeliner.viewer post break_up_mesh
    SOMA_COLOR = (0.9, 0.5, 0.9)
    ORG_COLOR = (1.0, 0.8, 0.0)
    NEURITE_COLORS = [
        (0.2, 0.6, 1.0),
        (0.3, 1.0, 0.3),
        (1.0, 0.4, 0.1),
        (0.0, 0.9, 0.9),
        (1.0, 0.2, 0.6),
    ]

    soma = components.soma
    organelles = components.organelles
    neurites = components.neurites
    nucleus = soma.nucleus if soma is not None else None

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    face_centroids = verts[faces].mean(axis=1)

    # Mirror the viewer: organelle faces that are also soma faces are
    # painted as soma, so subtract them from the org layer to avoid
    # double-coloring. A face is a soma face if >= 2 of its 3 verts
    # are in soma.verts.
    if soma is not None and len(soma.verts) > 0:
        soma_vmask = np.zeros(len(verts), dtype=bool)
        soma_vmask[np.asarray(soma.verts, dtype=np.int64)] = True
        soma_face_mask = soma_vmask[faces].sum(axis=1) >= 2
        org_only = organelles.mask & ~soma_face_mask
    else:
        org_only = organelles.mask

    # 2×2 grid: top-left=XZ, bottom-left=XY, bottom-right=ZY, top-right=empty
    fig, ax_grid = plt.subplots(2, 2, figsize=figsize)
    panel_pos = {"xy": (1, 0), "xz": (0, 0), "zy": (1, 1)}
    axes = []
    for plane in planes:
        r, c = panel_pos[plane]
        axes.append(ax_grid[r, c])
    ax_grid[0, 1].set_visible(False)

    for ax, plane in zip(axes, planes):
        ix, iy = _PLANE_AXES[plane]
        xlab, ylab = _plane_axes(plane)

        # All mesh vertices — light gray background
        ax.scatter(
            verts[:, ix],
            verts[:, iy],
            c="#dddddd",
            s=0.3,
            alpha=0.2,
            rasterized=True,
            zorder=1,
        )

        # Neurite face centroids — cycled palette.  The legend carries the
        # names once they are given: this figure is how you decide which
        # neurite is the axon, so it has to be able to show the answer.
        labels = neurites.labels if hasattr(neurites, "labels") else None
        for i, nf in enumerate(neurites):
            if len(nf) == 0:
                continue
            color = NEURITE_COLORS[i % len(NEURITE_COLORS)]
            fc = face_centroids[nf]
            ax.scatter(
                fc[:, ix],
                fc[:, iy],
                color=color,
                s=0.3,
                alpha=0.4,
                rasterized=True,
                zorder=2,
                label=f"[{i}] {labels[i]}" if labels is not None else f"neurite {i}",
            )

        # Organelle face centroids (excluding soma overlap)
        if org_only.any():
            fc = face_centroids[org_only]
            ax.scatter(
                fc[:, ix],
                fc[:, iy],
                color=ORG_COLOR,
                s=0.3,
                alpha=0.35,
                rasterized=True,
                zorder=3,
                label="organelles",
            )

        # Soma vertices
        if soma is not None and len(soma.verts) > 0:
            sv = verts[soma.verts]
            ax.scatter(
                sv[:, ix],
                sv[:, iy],
                color=SOMA_COLOR,
                s=0.8,
                alpha=0.6,
                rasterized=True,
                zorder=4,
                label="soma",
            )

        # Soma ellipsoid outline
        if soma is not None:
            ell = _soma_ellipse2d(soma, plane)
            ell.set_edgecolor(SOMA_COLOR)
            ell.set_linewidth(1.5)
            ax.add_patch(ell)

        # Nucleus center + z_range
        if nucleus is not None:
            nc = nucleus["center"]
            ax.plot(
                nc[ix],
                nc[iy],
                "+",
                color="cyan",
                markersize=12,
                markeredgewidth=2,
                zorder=10,
            )
            if 2 in (ix, iy):
                z_lo, z_hi = nucleus["z_range"]
                z_ax = ix if ix == 2 else iy
                for zv in (z_lo, z_hi):
                    if z_ax == ix:
                        ax.axvline(
                            zv,
                            color="cyan",
                            linewidth=0.8,
                            alpha=0.6,
                            linestyle="--",
                        )
                    else:
                        ax.axhline(
                            zv,
                            color="cyan",
                            linewidth=0.8,
                            alpha=0.6,
                            linestyle="--",
                        )

        # Zoom to soma region with generous padding
        if soma is not None and len(soma.verts) > 0:
            pad = max(soma.axes) * 1.5
            cx, cy = soma.center[ix], soma.center[iy]
        elif nucleus is not None:
            pad = nucleus["peak_r"] * 8
            cx, cy = nucleus["center"][ix], nucleus["center"][iy]
        else:
            pad = None

        if pad is not None:
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)

        ax.set_aspect("equal")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(plane.upper())
        ax.grid(True, alpha=0.15)

    axes[0].legend(fontsize=7, loc="upper right", markerscale=5)
    fig.tight_layout()
    return fig, axes


def diagnose_discarded(
    mesh: Mesh,
    components: MeshComponents,
    *,
    min_faces: int = 1000,
    planes: tuple[str, str, str] = ("xy", "xz", "zy"),
    figsize: tuple[float, float] = (12, 4),
) -> tuple[Figure, list[list[Axes]]]:
    """Row-per-component view of discarded fragments.

    Each discarded component with at least *min_faces* faces gets one
    row of three orthogonal projections.  All mesh vertices are shown
    in light gray; the discarded fragment is highlighted in red.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (after preprocessing).
    components : MeshComponents
        Result of :func:`~skeliner.pre.break_up_mesh`.
    min_faces : int, default 1000
        Only show fragments with at least this many faces.
    planes : tuple of str
        Three plane codes for the panels.
    figsize : tuple
        Figure size per row.

    Returns
    -------
    fig, axes : Figure, list[list[Axes]]
        One sub-list of axes per discarded fragment row.
    """
    DISCARD_COLOR = (0.9, 0.2, 0.2)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    face_centroids = verts[faces].mean(axis=1)

    big = [(i, d) for i, d in enumerate(components.discarded) if len(d) >= min_faces]
    if not big:
        print(f"No discarded fragments \u2265 {min_faces} faces")
        return None

    n_rows = len(big)
    n_cols = len(planes)
    fig, ax_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0], figsize[1] * n_rows),
        squeeze=False,
    )

    all_axes = []
    for row, (idx, d_faces) in enumerate(big):
        row_axes = []
        fc = face_centroids[d_faces]
        center = fc.mean(axis=0)
        span = fc.max(axis=0) - fc.min(axis=0)
        pad = max(span) * 0.6 + 500

        for col, plane in enumerate(planes):
            ax = ax_grid[row, col]
            row_axes.append(ax)
            ix, iy = _PLANE_AXES[plane]
            xlab, ylab = _plane_axes(plane)

            ax.scatter(
                verts[:, ix],
                verts[:, iy],
                c="#dddddd",
                s=0.3,
                alpha=0.15,
                rasterized=True,
                zorder=1,
            )
            ax.scatter(
                fc[:, ix],
                fc[:, iy],
                color=DISCARD_COLOR,
                s=0.8,
                alpha=0.6,
                rasterized=True,
                zorder=2,
            )

            ax.set_xlim(
                center[ix] - pad,
                center[ix] + pad,
            )
            ax.set_ylim(
                center[iy] - pad,
                center[iy] + pad,
            )
            ax.set_aspect("equal")
            ax.set_xlabel(xlab)
            if col == 0:
                ax.set_ylabel(
                    f"discarded {idx}\n({len(d_faces):,}f)\n\n{ylab}",
                )
            else:
                ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(plane.upper())
            ax.grid(True, alpha=0.15)

        all_axes.append(row_axes)

    fig.tight_layout()
    return fig, all_axes


# `skeliner.plot` held the web viewer too, and these figures as `vis2d`,
# until the viewer became `skeliner.viewer`
_MOVED = ("view", "view3d", "view_contacts", "stop_viewer")


def __getattr__(name: str):
    if name in _MOVED:
        warnings.warn(
            f"sk.plot.{name}() is deprecated, use sk.viewer.{name}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module("skeliner.viewer"), name)
    if name == "vis2d":
        warnings.warn(
            "sk.plot.vis2d is deprecated, its figures are sk.plot's own.",
            DeprecationWarning,
            stacklevel=2,
        )
        return sys.modules[__name__]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
