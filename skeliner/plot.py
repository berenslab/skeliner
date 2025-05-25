import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from scipy.stats import binned_statistic_2d

from .core import Skeleton

__all__ = ["projection", "threeviews", "diagnostic"]


_PLANE_AXES = {
    "xy": (0, 1), "yx": (1, 0),
    "xz": (0, 2), "zx": (2, 0),
    "yz": (1, 2), "zy": (2, 1),  
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
    centre_xy      = soma.centre[[ix, iy]] * scale

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
    unit: str | None = None,
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
        sizes, ppd = _radii_to_sizes(rr, ax)
        ax.scatter(
            xy_skel[:, 0],
            xy_skel[:, 1],
            s=sizes,
            facecolors="none",
            edgecolors="red",
            linewidths=1.0,
            alpha=circle_alpha,
            zorder=2,
        )

    # ─── highlight the soma if requested ───────────────────────────────────
    if draw_soma_mask and skel.soma is not None and skel.soma.verts is not None:

        xy_soma = _project(
            mesh.vertices[np.asarray(skel.soma.verts, dtype=np.int64)], ix, iy
        ) * scale[1]                                 # note: mesh scale!
        keep_soma = _apply_window(xy_soma)           # respect crop
        xy_soma   = xy_soma[keep_soma]

        ax.scatter(
            xy_soma[:, 0], xy_soma[:, 1],
            s=1, c="pink", marker="o",
            linewidths=0, alpha=0.5, label="soma surface"
        )

        # centroid + dashed outline for radius readability
        c_xy = _project(skel.nodes[[0]] * scale[0], ix, iy).ravel()
        ax.scatter(*c_xy, c="k", s=15, zorder=3, label="soma centre")

        ell = _soma_ellipse2d(skel.soma, plane, scale=scale[0])
                            # dashed outline as before
        ell.set_edgecolor("k")
        ell.set_facecolor("none")
        ell.set_linestyle("--")
        ell.set_linewidth(0.8)
        ell.set_alpha(0.9)
        ax.add_patch(ell)

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
                linewidths=.5,
                alpha=line_alpha,
            )
            ax.add_collection(lc)
    # ─────────────────── final tweaks ─────────────────────────────────────
    ax.set_aspect("equal")
    if unit is None:
        if scale[0] == 1.0:
            unit_str = ""
        else:
            unit_str = f"(Scaled by {scale[0]})"
    else:
        unit_str = f"({unit})"
    ax.set_xlabel(f"{plane[0]} {unit_str}")
    ax.set_ylabel(f"{plane[1]} {unit_str}")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    return fig, ax


def diagnostic(
    mesh: "trimesh.Trimesh",
    skel: Skeleton,
    *,
    plane: str = "xy",
    # background histogram --------------------------------------------------- #
    bins: int | tuple[int, int] = 800,
    hist_cmap: str = "Blues",
    vmax_fraction: float = 0.10,
    # overlays --------------------------------------------------------------- #
    draw_nodes: bool = True,
    draw_edges: bool = False,
    draw_soma_mask: bool = True,
    show_node_ids: bool = False,
    radius_metric: str | None = None,
    # appearance ------------------------------------------------------------- #
    cluster_cmap: str = "tab20",
    circle_alpha: float = 0.9,
    edge_color: str = "0.25",
    edge_lw: float = 0.8,
    id_fontsize: int = 6,
    id_color: str = "black",
    id_offset: tuple[float, float] = (0.0, 0.0),
    # geometry ---------------------------------------------------------------- #
    scale: float | tuple[float, float] | list[float] = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    # title ------------------------------------------------------------------ #
    title: str | None = None,
    unit: str | None = None,
    # axes                                                                     #
    ax: Axes | None = None,
):
    """
    2-D overview plot – mesh density + coloured clusters + optional skeleton.

    Parameters
    ----------
    plane
        Projection plane, one of ``{"xy","xz","yz","yx","zx","zy"}``.
    bins
        Background histogram resolution (passed to
        :pyfunc:`scipy.stats.binned_statistic_2d`).
    scale
        Either a single factor applied to *both* mesh and skeleton or a pair
        ``(skel_scale, mesh_scale)``.
    draw_nodes, draw_edges
        Overlay the skeleton circles and/or edges.
    cluster_cmap
        Matplotlib colormap for the clusters (colours are shuffled using the
        golden-ratio trick so neighbouring IDs differ).
    xlim, ylim
        Optional crop window **before** any rendering work is done.
    """
    # ------------- housekeeping / defaults ----------------------------------
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")

    ix, iy = _PLANE_AXES[plane]

    if isinstance(scale, (int, float)):
        scale = (float(scale),) * 2
    if len(scale) != 2:
        raise ValueError("scale must be a scalar or a pair/list of two scalars")
    scl_skel, scl_mesh = map(float, scale)

    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    # ------------- project & crop -------------------------------------------
    xy_mesh_all = _project(mesh.vertices.view(np.ndarray), ix, iy) * scl_mesh
    xy_skel = _project(skel.nodes, ix, iy) * scl_skel
    rr      = skel.radii[radius_metric] * scl_skel

    def _mask_window(xy: np.ndarray) -> np.ndarray:
        keep = np.ones(len(xy), bool)
        if xlim is not None:
            keep &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            keep &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        return keep

    keep_mesh     = _mask_window(xy_mesh_all)
    xy_mesh_crop  = xy_mesh_all[keep_mesh]

    keep_skel = _mask_window(xy_skel)
    xy_skel   = xy_skel[keep_skel]
    rr        = rr[keep_skel]

    # ------------- density histogram ----------------------------------------
    # Import here to keep hard deps minimal
    from scipy.stats import binned_statistic_2d  # local import

    if isinstance(bins, int):
        bins_arg: int | tuple[int, int] = bins
    else:
        if (not isinstance(bins, tuple)) or len(bins) != 2:
            raise ValueError("bins must be int or (int, int)")
        bins_arg = tuple(map(int, bins))

    hist, xedges, yedges, _ = binned_statistic_2d(
        xy_mesh_crop[:, 0], xy_mesh_crop[:, 1], None,
        statistic="count", bins=bins_arg,
    )
    hist = hist.T                                      # imshow(rows=y)

    # -------- component labels for every *mesh* vertex ----------------------
    # -------- prepare vertex–cluster labels (if we have a skeleton) ---------
    if skel is not None and skel.node2verts is not None:
        lab_full   = _component_labels(len(mesh.vertices), skel.node2verts)
        # restrict to vertices that belong to *some* node
        in_cluster = lab_full >= 0
        mask_mesh  = keep_mesh & in_cluster

        xy_mesh_scatter = xy_mesh_all[mask_mesh]
        lab_mesh        = lab_full[mask_mesh]
        n_comp          = int(lab_full.max() + 1)
    else:
        # no skeleton → fall back to “plot-everything” but colour uniformly
        xy_mesh_scatter = xy_mesh_crop
        lab_mesh        = None          # single colour
        n_comp          = 0             # disables LUT
 
    # ------------- figure / axes --------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Show the blue density map only if we do NOT have node-based clusters
    if skel is None or skel.node2verts is None:
        ax.imshow(
            hist,
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            origin="lower",
            cmap=hist_cmap,
            vmax=hist.max() * vmax_fraction,
            alpha=1.0,
            zorder=0,
        )

   # ------------- vertex cloud overlay -------------------------------------
    if n_comp:                               # skeleton → coloured clusters
        lut = _make_lut(cluster_cmap, n_comp)
        colours = lut[lab_mesh]
    else:                                    # plain mesh → uniform grey
        colours = "0.6"


    ax.scatter(
        xy_mesh_scatter[:, 0], xy_mesh_scatter[:, 1],
        s=1.0, c=colours, alpha=0.75, linewidths=0, zorder=9
    )

    sizes, ppd = _radii_to_sizes(rr, ax)
    # ------------- skeleton circles & centres -------------------------------
    if draw_nodes and len(xy_skel):
        

        # per-node colour uses *first vertex* of the cluster as label
        node_comp   = np.array([
            lab_full[skel.node2verts[i][0]] if len(skel.node2verts[i]) else -1
            for i in range(len(skel.nodes))
        ])
        node_comp   = node_comp[keep_skel]
        node_colors = lut[node_comp]

        # circles (facecolor none, coloured edge)
        ax.scatter(
            xy_skel[:, 0], xy_skel[:, 1],
            s=sizes, facecolors="none", edgecolors=node_colors,
            linewidths=0.9, alpha=circle_alpha, zorder=3,
        )
        # centre points
        ax.scatter(
            xy_skel[:, 0], xy_skel[:, 1],
            s=10, c=node_colors, alpha=circle_alpha, zorder=4,
            linewidths=0,
        )


        # optional node-ID labels
        if show_node_ids:
            orig_ids = np.flatnonzero(keep_skel)
            dx, dy   = id_offset
            for nid, (x, y) in zip(orig_ids, xy_skel):
                ax.text(
                    x + dx, y + dy, str(nid),
                    fontsize=id_fontsize, color=id_color,
                    ha="center", va="center", zorder=5)

    # ------------- edges -----------------------------------------------------
    if draw_edges and skel.edges.size:
        # keep edges whose *both* endpoints survived cropping
        keep_flags = keep_skel
        ekeep      = keep_flags[skel.edges[:, 0]] & keep_flags[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]

        if edges_kept.size:
            # compress indices once
            idx_map = -np.ones(len(keep_flags), dtype=int)
            idx_map[np.flatnonzero(keep_flags)] = np.arange(keep_flags.sum())

            seg_start = xy_skel[idx_map[edges_kept[:, 0]]]
            seg_end   = xy_skel[idx_map[edges_kept[:, 1]]]
            segs      = np.stack((seg_start, seg_end), axis=1)

            lc = LineCollection(
                segs, colors=edge_color, linewidths=edge_lw,
                alpha=circle_alpha * 0.9, zorder=2)
            ax.add_collection(lc)

    # ------------- optional soma shell --------------------------------------
    if draw_soma_mask and skel.soma is not None and skel.soma.verts is not None:
        xy_soma = _project(
            mesh.vertices[np.asarray(skel.soma.verts, dtype=np.int64)], ix, iy
        ) * scl_mesh
        soma_keep = _mask_window(xy_soma)
        xy_soma   = xy_soma[soma_keep]

        ax.scatter(
            xy_soma[:, 0], xy_soma[:, 1],
            s=1.0, c="C0", alpha=0.45, linewidths=0, zorder=9,
            label="soma surface",
        )
        # centre + outline
        c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
        ax.scatter(*c_xy, c="k", s=16, zorder=9)

        # dashed outline matching circle size
        ell = _soma_ellipse2d(skel.soma, plane, scale=scale[0])
        ell.set_edgecolor("k")
        ell.set_facecolor("none")
        ell.set_linestyle("--")
        ell.set_linewidth(0.8)
        ax.add_patch(ell)                         # dashed outline as before

    # ------------- cosmetics -------------------------------------------------
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
    mesh: trimesh.Trimesh,
    *,
    planes: tuple[str, str, str] | list[str] = ["xy", "xz", "zy"],
    scale: float = 1e-3,                 # nm → µm by default
    title: str | None = None,
    figsize: tuple[int, int] = (8, 8),
    draw_edges: bool = True,
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

    # ── 0. global bounding box (already scaled) ────────────────────────────
    v_scaled = mesh.vertices.view(np.ndarray) * scale
    lims, spans = _axis_extents(v_scaled)

    # helper: pick limits for a given plane string
    def _limits(p: str):
        h, v = _plane_axes(p)
        return lims[h], lims[v]            # (xlim, ylim)

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