import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.stats import binned_statistic_2d

from .core import Skeleton

__all__ = ["plot_projection"]

_PLANE_AXES = {
    "xy": (0, 1), "yx": (1, 0),
    "xz": (0, 2), "zx": (2, 0),
    "yz": (1, 2), "zy": (2, 1),  
}

def _project(arr: np.ndarray, ix: int, iy: int, /) -> np.ndarray:
    """Return 2-column slice (arr[:, (ix, iy)])."""
    return arr[:, (ix, iy)].copy()

def _radii_to_scatter_size(rr: np.ndarray, ax: Axes) -> np.ndarray:
    """
    Convert radii in *data units* to matplotlib scatter sizes (points²) **in a
    way that is independent of the particular subplot’s width/height**.

    For axes that use ``ax.set_aspect("equal")`` the pixel–per–data-unit ratio
    is identical in *both* directions, but it can differ from one panel to the
    next if their x– or y-ranges (or their physical sizes on the canvas) are
    different.  We therefore

    1. work out the ratio in *both* directions,  
    2. take the *smaller* of the two (guaranteed to be the true isotropic
       scale), and  
    3. convert radii → pixels → points.

    This ensures that the same radius in data units is rendered with the same
    diameter in points in **every** subplot that belongs to the figure.
    """
    fig = ax.figure
    dpi = fig.dpi

    # current axis limits (data units)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # axis size on the canvas (pixels)
    bbox = ax.get_window_extent()

    # pixels per data-unit in each direction
    ppd_x = bbox.width  / abs(x1 - x0)
    ppd_y = bbox.height / abs(y1 - y0)

    # use the isotropic (smaller) scale so all panels agree
    ppd = min(ppd_x, ppd_y)

    # radius: data-units → pixels → points → area (points²)
    r_px = rr * ppd
    r_pt = r_px * 72.0 / dpi
    return np.pi * r_pt**2


def plot_projection(
    skel: Skeleton,
    mesh: "trimesh.Trimesh",
    *,
    plane: str = "xy",
    radius_metric: str | None = None,
    bins: int | tuple[int, int] = 800,
    scale: float | list = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    draw_skel: bool = True,
    draw_edges: bool = False,
    ax: Axes | None = None,
    cmap: str = "Blues",
    vmax_fraction: float = 0.10,
    circle_alpha: float = 0.25,
    line_alpha: float = 0.8,
    # --- soma ---
    draw_soma_mask: bool = True,
) -> tuple:
    """
    Plot a 2-D density map of *mesh* and overlay *skel* circles (and optionally
    edges) in the chosen *plane*.

    Parameters
    ----------
    plane : {\"xy\", \"xz\", \"yz\"}
        Projection plane.
    bins : int | (int,int)
        Resolution of the background histogram.
    scale : float or list, default 1
        Multiply coordinates & radii by this factor (e.g. ``1e-3`` for nm→µm).
        if a list, the first element is used for the skel and the second for the mesh.
    xlim, ylim : (min, max) or None
        Spatial extent to keep **before** plotting.
    draw_edges : bool
        If True, draw the skeleton graph edges.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw into (created automatically if ``None``).
    vmax_fraction : float, default 0.10
        Upper colour-limit for the histogram (as a fraction of its max).
    circle_alpha, line_alpha : float
        Transparencies of the skeleton glyphs.
    **imshow_kwargs
        Extra options passed to :pyfunc:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    (fig, ax)
        Figure and axes objects.
    """
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")

    ix, iy = _PLANE_AXES[plane]

    if type(scale) is not list:
        scale = [scale, scale]

    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    # ─────────────────── project & scale ──────────────────────────────────
    xy_mesh = _project(mesh.vertices, ix, iy) * scale[1]
    xy_skel = _project(skel.nodes, ix, iy) * scale[0]
    rr      = skel.radii[radius_metric] * scale[0]

    # ─────────────────── crop early to save work ──────────────────────────
    def _apply_window(xy: np.ndarray) -> np.ndarray:
        keep = np.ones(len(xy), dtype=bool)
        if xlim is not None:
            keep &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            keep &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        return keep

    keep_mesh = _apply_window(xy_mesh)
    xy_mesh   = xy_mesh[keep_mesh]

    keep_skel = _apply_window(xy_skel)
    xy_skel   = xy_skel[keep_skel]
    rr        = rr[keep_skel]

    # ─────────────────── density histogram ────────────────────────────────
    # Ensure bins is either an int or a tuple of two ints
    if isinstance(bins, int):
        bins_arg: int | tuple[int, int] = bins
    elif isinstance(bins, tuple) and len(bins) == 2 and all(isinstance(b, int) for b in bins):
        bins_arg = (int(bins[0]), int(bins[1]))
    else:
        raise ValueError("bins must be an int or a tuple of two ints")
    hist, xedges, yedges, _ = binned_statistic_2d(
        xy_mesh[:, 0],
        xy_mesh[:, 1],
        None,
        statistic="count",
        bins=bins_arg,  # type: ignore
    )
    hist = hist.T  # transpose for imshow (row = y)

    # ─────────────────── optional soma overlay ────────────────────────────
    if draw_soma_mask and skel.soma_verts is not None:
        xy_soma = _project(
            mesh.vertices[np.asarray(skel.soma_verts, dtype=np.int64)], ix, iy
        ) * scale[1]                                 # note: mesh scale!
        keep_soma = _apply_window(xy_soma)           # respect crop
        xy_soma   = xy_soma[keep_soma]
    else:
        xy_soma = None

    # ─────────────────── figure / axes ────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.imshow(
        hist,
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        origin="lower",
        cmap=cmap,
        vmax=hist.max() * vmax_fraction,
        alpha=1.0,
    )

    # ─────────────────── circles ──────────────────────────────────────────
    if draw_skel:
        sizes = _radii_to_scatter_size(rr, ax)
        ax.scatter(
            xy_skel[:, 0],
            xy_skel[:, 1],
            s=sizes,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
            alpha=circle_alpha,
        )

    # ─── highlight the soma if requested ───────────────────────────────────
    if draw_skel and draw_soma_mask and xy_soma is not None and len(xy_soma):
        ax.scatter(
            xy_soma[:, 0], xy_soma[:, 1],
            s=4, c="red", marker="o",
            linewidths=0, alpha=0.9, label="soma surface"
        )

        # centroid + dashed outline for radius readability
        c_xy = _project(skel.nodes[[0]] * scale[0], ix, iy).ravel()
        ax.scatter(*c_xy, c="k", s=15, zorder=3, label="soma centre")
        ax.add_patch(
            Circle(
                (c_xy[0], c_xy[1]),
                skel.radii[radius_metric][0] * scale[0],           # physical size
                facecolor="none", edgecolor="black",
                linestyle="--", linewidth=1.3, zorder=2,
            )
        )

    # ─────────────────── optional edges ───────────────────────────────────
    if draw_skel and draw_edges and skel.edges.size:
        keep = keep_skel                              # local alias
        # 1. filter edge list to kept endpoints
        ekeep = keep[skel.edges[:, 0]] & keep[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]

        if edges_kept.size:
            # 2. build old-id → compressed-id lookup once
            idx_map = -np.ones(len(keep), dtype=int)
            idx_map[np.flatnonzero(keep)] = np.arange(keep.sum())

            # 3. gather coordinates for both endpoints in one NumPy call
            seg_start = xy_skel[idx_map[edges_kept[:, 0]]]   # (E', 2)
            seg_end   = xy_skel[idx_map[edges_kept[:, 1]]]   # (E', 2)
            segments  = np.stack((seg_start, seg_end), axis=1)  # (E', 2, 2)

            # 4. hand over to Matplotlib
            lc = LineCollection(
                segments.tolist(),
                colors="black",
                linewidths=1.0,
                alpha=line_alpha,
            )
            ax.add_collection(lc)
    # ─────────────────── final tweaks ─────────────────────────────────────
    ax.set_aspect("equal")
    ax.set_xlabel(f"{plane[0]} (scaled units)")
    ax.set_ylabel(f"{plane[1]} (scaled units)")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    return fig, ax