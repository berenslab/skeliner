from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Circle
from scipy.stats import binned_statistic_2d

from .core import Skeleton

__all__ = ["plot_projection"]

_PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

def _project(arr: np.ndarray, ix: int, iy: int, /) -> np.ndarray:
    """Return 2-column slice (arr[:, (ix, iy)])."""
    return arr[:, (ix, iy)].copy()

def plot_projection(
    skel: Skeleton,
    mesh: "trimesh.Trimesh",
    *,
    plane: str = "xy",
    bins: int | tuple[int, int] = 800,
    scale: float | list = 1.0,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    draw_edges: bool = False,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "Blues",
    vmax_fraction: float = 0.10,
    circle_alpha: float = 0.25,
    line_alpha: float = 0.8,
    **imshow_kwargs,
) -> tuple["plt.Figure", "plt.Axes"]:
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

    # ─────────────────── project & scale ──────────────────────────────────
    xy_mesh = _project(mesh.vertices, ix, iy) * scale[1]
    xy_skel = _project(skel.nodes, ix, iy) * scale[0]
    rr      = skel.radii * scale[0]

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
    hist, xedges, yedges, _ = binned_statistic_2d(
        xy_mesh[:, 0],
        xy_mesh[:, 1],
        None,
        statistic="count",
        bins=bins,
    )
    hist = hist.T  # transpose for imshow (row = y)

    # ─────────────────── figure / axes ────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.imshow(
        hist,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        cmap=cmap,
        vmax=hist.max() * vmax_fraction,
        alpha=1.0,
        **imshow_kwargs,
    )

    # ─────────────────── circles ──────────────────────────────────────────
    for (x, y), r in zip(xy_skel, rr):
        circ = Circle(
            (x, y),
            r,
            facecolor="none",
            edgecolor="red",
            linewidth=1.0,
            alpha=circle_alpha,
        )
        ax.add_patch(circ)

    # ─────────────────── optional edges ───────────────────────────────────
    if draw_edges:
        for (i, j) in skel.edges:
            if keep_skel[i] and keep_skel[j]:            # both endpoints kept
                x1, y1 = xy_skel[np.searchsorted(np.flatnonzero(keep_skel), i)]
                x2, y2 = xy_skel[np.searchsorted(np.flatnonzero(keep_skel), j)]
                ax.plot([x1, x2], [y1, y2],
                        color="black", linewidth=1.0, alpha=line_alpha)

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