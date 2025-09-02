# Portions derived from vis3d module in `seung-lab/microviewer``
# (c) 2025 William Silversmith <william.silversmith@gmail.com> originally under LGPL-2.1.
# This copy is distributed under the terms of the GNU GPL v3 (per LGPL-2.1 §3).

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import trimesh

from skeliner.core import Skeleton

# ---------------------------------------------------------------------
# Minimal palettes (no matplotlib dependency)
# ---------------------------------------------------------------------
COLORS: List[Tuple[float, float, float]] = [
    (1.00, 0.64, 0.00),  # amber
    (0.00, 0.62, 0.99),  # blue
    (0.92, 0.96, 1.00),  # pale blue
    (0.89, 0.58, 0.62),  # salmon
    (0.94, 0.22, 0.42),  # cerise
    (0.60, 0.87, 0.78),  # mint
    (0.81, 0.69, 0.72),  # rose
    (0.90, 0.33, 0.51),  # blush
    (0.76, 0.76, 0.90),  # periwinkle
]
BBOX_COLORS: List[Tuple[float, float, float]] = [
    (0.91, 0.84, 0.35),  # yellow
    (0.02, 0.84, 0.63),  # emerald
    (0.00, 0.67, 0.91),  # picton blue
    (0.96, 0.52, 0.29),  # orange
    (0.93, 0.79, 1.00),  # mauve
]

# Only radii: accepted synonyms on Skeleton
RADII_KEYS = ("r", "radius", "radii")


# ---------------------------------------------------------------------
# Simple bbox type (replacement for osteoid.Bbox)
# ---------------------------------------------------------------------
@dataclass(slots=True)
class BBox:
    minpt: np.ndarray  # (3,)
    maxpt: np.ndarray  # (3,)

    @classmethod
    def from_arrays(cls, arrs: Sequence[np.ndarray]) -> "BBox":
        mins = np.min([a.min(axis=0) for a in arrs if a.size], axis=0)
        maxs = np.max([a.max(axis=0) for a in arrs if a.size], axis=0)
        return cls(mins.astype(float), maxs.astype(float))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _as_iter(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _get_label(obj):
    for key in ("id", "segid", "label"):
        if hasattr(obj, key):
            return getattr(obj, key)
    return None


def _get_radii(skel):
    for key in RADII_KEYS:
        if hasattr(skel, key):
            r = getattr(skel, key)
            if r is not None:
                return np.asarray(r, dtype=float)
    return None


def _drop_soma(vertices: np.ndarray, edges: np.ndarray, radii: np.ndarray | None):
    """Remove node 0 and its incident edges, then compact indices."""
    if len(vertices) == 0:
        return vertices, edges, radii
    keep_edges = (edges[:, 0] != 0) & (edges[:, 1] != 0)
    edges = edges[keep_edges] - 1
    vertices = vertices[1:]
    if radii is not None:
        radii = radii[1:]
    return vertices, edges, radii


# ---------------------------------------------------------------------
# Core rendering helpers (VTK only; imported lazily)
# ---------------------------------------------------------------------
def _create_lut_adv(
    vmin: float,
    vmax: float,
    *,
    base: str = "teal-amber",
    stretch: str = "gamma",  # "linear" | "gamma" | "log" | "equalize"
    gamma: float = 0.5,
    eq_map: np.ndarray | None = None,  # length N in [0,1], only for "equalize"
    n: int = 256,
):
    """
    Build a LookupTable with a perceptual-ish base ramp and a value-space stretch.
    - base: "teal-amber" is designed to brighten early (good for small values).
    - stretch:
        linear:   u' = u
        gamma:    u' = u**gamma            (gamma<1 -> more low-end contrast)
        log:      u' = log10(1 + 9u)/log10(10)  (more aggressive near zero)
        equalize: u' = eq_map[i], empirical CDF (pass from data)
    """
    import vtk

    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToRGB()

    # Custom base ramp with early luminance rise (no viridis/magma).
    if base == "teal-amber":
        stops = [
            (0.00, (0.055, 0.070, 0.110)),  # deep slate
            (0.08, (0.000, 0.350, 0.500)),  # teal
            (0.25, (0.180, 0.640, 0.640)),  # aqua
            (0.55, (0.850, 0.850, 0.400)),  # sand
            (1.00, (1.000, 0.980, 0.800)),  # pale amber
        ]
    else:
        # Simple gray ramp fallback (high-contrast, non-offensive)
        stops = [
            (0.00, (0.10, 0.10, 0.10)),
            (1.00, (1.00, 1.00, 1.00)),
        ]

    for t, (r, g, b) in stops:
        ctf.AddRGBPoint(float(t), float(r), float(g), float(b))

    lut = vtk.vtkLookupTable()
    lut.SetRange(float(vmin), float(vmax))
    lut.SetNumberOfTableValues(int(n))

    for i in range(n):
        u = i / (n - 1)
        if stretch == "gamma":
            u = u ** float(gamma)
        elif stretch == "log":
            # log10 map of [0,1] -> [0,1], well-defined at 0
            u = np.log10(1.0 + 9.0 * u) / np.log10(10.0)
        elif stretch == "equalize" and eq_map is not None and len(eq_map) == n:
            u = float(eq_map[i])
        # else: linear

        r, g, b = ctf.GetColor(float(u))
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)

    lut.Build()
    return lut


def _vtk_mesh_actor(
    mesh: trimesh.Trimesh, color=(1.0, 1.0, 1.0), opacity=0.2, scale=1.0
):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    verts = (mesh.vertices * float(scale)).astype(np.float32)
    faces = mesh.faces.astype(np.int64)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(verts))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)

    counts = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    vtk_faces = vtk.vtkCellArray()
    vtk_faces.SetCells(
        faces.shape[0], numpy_to_vtkIdTypeArray(np.hstack([counts, faces]).ravel())
    )
    poly.SetPolys(vtk_faces)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*map(float, color))
    actor.GetProperty().SetOpacity(float(opacity))
    return actor


def _vtk_skeleton_actor(
    verts: np.ndarray,
    edges: np.ndarray,
    *,
    radii: np.ndarray | None = None,
    scale: float = 1.0,
    lut=None,
    default_color=(0.95, 0.95, 0.95),
):
    import vtk
    import vtk.util.numpy_support

    pv = np.asarray(verts, dtype=np.float32) * float(scale)
    pe = np.asarray(edges, dtype=np.int64)

    points = vtk.vtkPoints()
    points.SetData(vtk.util.numpy_support.numpy_to_vtk(pv))

    lines = vtk.vtkCellArray()
    for u, v in pe:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(u))
        line.GetPointIds().SetId(1, int(v))
        lines.InsertNextCell(line)

    polyline = vtk.vtkPolyData()
    polyline.SetPoints(points)
    polyline.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyline)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)

    if radii is not None and len(radii) == len(verts):
        s = vtk.util.numpy_support.numpy_to_vtk(np.asarray(radii, dtype=np.float32))
        s.SetName("Radii")
        polyline.GetPointData().SetScalars(s)
        mapper.SetScalarVisibility(1)
        mapper.SetScalarModeToUsePointData()
        if lut is not None:
            mapper.SetLookupTable(lut)
            mapper.UseLookupTableScalarRangeOn()
    else:
        actor.GetProperty().SetColor(*default_color)

    return actor


def _vtk_bbox_actor(box: BBox, color=(0.91, 0.84, 0.35), line_width: float = 2.0):
    import vtk

    p0 = np.asarray(box.minpt, dtype=float)
    p7 = np.asarray(box.maxpt, dtype=float)
    corners = np.array(
        [
            [p0[0], p0[1], p0[2]],
            [p7[0], p0[1], p0[2]],
            [p0[0], p7[1], p0[2]],
            [p0[0], p0[1], p7[2]],
            [p7[0], p7[1], p0[2]],
            [p7[0], p0[1], p7[2]],
            [p0[0], p7[1], p7[2]],
            [p7[0], p7[1], p7[2]],
        ],
        dtype=float,
    )

    edges = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [7, 6],
            [7, 5],
            [7, 4],
            [6, 2],
            [6, 3],
            [3, 5],
            [5, 1],
            [2, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )

    points = vtk.vtkPoints()
    for c in corners:
        points.InsertNextPoint(*c)

    lines = vtk.vtkCellArray()
    for u, v in edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(u))
        line.GetPointIds().SetId(1, int(v))
        lines.InsertNextCell(line)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*map(float, color))
    actor.GetProperty().SetLineWidth(float(line_width))
    return actor


def _add_recenter_controls(picker, interactor, renderer):
    def recenter(_obj, evt):
        if evt not in ("RightButtonPressEvent", "LeftButtonPressEvent"):
            return
        if evt == "LeftButtonPressEvent" and not interactor.GetControlKey():
            return
        x, y = interactor.GetEventPosition()
        if picker.Pick(x, y, 0, renderer):
            fp = np.array(picker.GetPickPosition())
            cam = renderer.GetActiveCamera()
            offset = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
            cam.SetFocalPoint(*fp)
            cam.SetPosition(*(fp + offset))
            renderer.ResetCameraClippingRange()
            interactor.Render()

    interactor.AddObserver("RightButtonPressEvent", recenter)
    interactor.AddObserver("LeftButtonPressEvent", recenter)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def view3d(
    skels: List["Skeleton"] | "Skeleton",
    meshes: List[trimesh.Trimesh] | trimesh.Trimesh,
    *,
    include_soma: bool = False,
    scale: float | Sequence[float] = 1.0,
    box: Sequence[float] | None = None,  # [x0,y0,z0,x1,y1,z1] or None
    mesh_color: str | Sequence[Tuple[float, float, float]] = "diff",
    mesh_opacity: float = 0.2,
    stretch: str = "linear",  # "linear" | "gamma" | "log" | "equalize"
    gamma: float = 0.55,  # only if stretch="gamma" (0.35–0.75 are good)
    clip_percentiles: tuple[float, float] = (0.0, 100.0),  # e.g., (1, 99)
) -> None:
    """
    Visualise skeletons + meshes with VTK, coloring skeleton vertices by radii.
    Radii are soma-dropped and scaled before LUT stats, then a value stretch is applied.

    stretch:
        "gamma": u' = u**gamma (gamma<1 expands low end)
        "log":   strong emphasis near zero; radii must be > 0
        "equalize": histogram-equalized mapping from your data (adapts to skew)
        "linear": plain linear
    clip_percentiles:
        ignore outliers when setting the colorbar (nice for spiky radii).
    """
    try:
        import vtk  # noqa: F401
    except Exception as e:
        raise ImportError("This viewer requires VTK. Try: pip install vtk") from e

    skels = _as_iter(skels)
    meshes = _as_iter(meshes)

    # Parse scaling
    if isinstance(scale, (int, float)):
        skel_scale = float(scale)
        mesh_scale = float(scale)
    else:
        if len(scale) != 2:
            raise ValueError("scale must be a scalar or a pair/list of two scalars")
        skel_scale, mesh_scale = map(float, scale)

    # ---- Prep skeletons once (apply include_soma here) & collect scaled radii for LUT ----
    prepped = []  # list of (verts, edges, radii_scaled)
    all_r_scaled = []
    for s in skels:
        verts = np.asarray(s.nodes, dtype=float)
        edges = np.asarray(s.edges, dtype=np.int64)
        r = _get_radii(s)

        if not include_soma:
            verts, edges, r = _drop_soma(verts, edges, r)

        r_scaled = None if r is None else (np.asarray(r, dtype=float) * skel_scale)
        prepped.append((verts, edges, r_scaled))
        if r_scaled is not None and r_scaled.size:
            all_r_scaled.append(r_scaled)

    # Concatenate for stats (may be empty)
    lut = None
    if all_r_scaled:
        vals = np.concatenate(all_r_scaled)
        # clip outliers for nicer bar if requested
        p_lo, p_hi = clip_percentiles
        if p_lo > 0.0 or p_hi < 100.0:
            vmin = float(np.percentile(vals, p_lo))
            vmax = float(np.percentile(vals, p_hi))
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

        # guard degenerate ranges
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            vmin, vmax = 0.0, 1.0
        if not (vmax > vmin):
            eps = 1e-9 if vmin == 0 else abs(vmin) * 1e-6
            vmax = vmin + eps

        # optional histogram equalization map
        eq_map = None
        if stretch == "equalize":
            # normalize (clipped) values to [0,1], then sample their CDF at 256 levels
            v = np.clip(vals, vmin, vmax)
            v = (v - vmin) / (vmax - vmin + 1e-15)
            eq_map = np.quantile(v, np.linspace(0, 1, 256))

        lut = _create_lut_adv(
            vmin,
            vmax,
            base="teal-amber",
            stretch=stretch,
            gamma=gamma,
            eq_map=eq_map,
            n=256,
        )

    # Mesh color handling
    if isinstance(mesh_color, str):
        if mesh_color == "same":
            mesh_palette = None  # all white
        elif mesh_color == "diff":
            mesh_palette = COLORS
        else:
            raise ValueError(
                "mesh_color must be 'same', 'diff', or a list of RGB triples"
            )
    else:
        palette = list(mesh_color)
        if not palette:
            raise ValueError("mesh_color iterable is empty")
        mesh_palette = [tuple(map(float, c)) for c in palette]

    # Build mesh actors
    mesh_actors = []
    for i, m in enumerate(meshes):
        color = (
            (1.0, 1.0, 1.0)
            if mesh_palette is None
            else mesh_palette[i % len(mesh_palette)]
        )
        mesh_actors.append(
            _vtk_mesh_actor(m, color=color, opacity=mesh_opacity, scale=mesh_scale)
        )

    # Build skeleton actors from the prepped data
    skel_actors = []
    for verts, edges, r_scaled in prepped:
        skel_actors.append(
            _vtk_skeleton_actor(verts, edges, radii=r_scaled, scale=skel_scale, lut=lut)
        )

    # Compute / normalize bbox
    if box is None:
        all_geom = []
        for m in meshes:
            if m.vertices.size:
                all_geom.append((m.vertices * mesh_scale).astype(float))
        for verts, _, _ in prepped:
            v = verts * skel_scale
            if v.size:
                all_geom.append(v)
        bbox = BBox.from_arrays(all_geom) if all_geom else None
    else:
        if len(box) != 6:
            raise ValueError("box must be [x0,y0,z0,x1,y1,z1]")
        bbox = BBox(np.array(box[:3], float), np.array(box[3:], float))

    # --------------------- VTK window ---------------------------------
    import vtk

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.062745, 0.129412)  # deep blue-gray

    # Draw bbox (optional)
    if bbox is not None:
        renderer.AddActor(_vtk_bbox_actor(bbox, color=BBOX_COLORS[0]))

    # Heuristic: if skeleton+mesh share a label, nudge mesh to (at most) 0.2 opacity
    labels = {}
    for m in meshes:
        lab = _get_label(m)
        if lab is not None:
            labels.setdefault(lab, []).append(("mesh", m))
    for s in skels:
        lab = _get_label(s)
        if lab is not None:
            labels.setdefault(lab, []).append(("skel", s))

    translucent_mesh_ids = set()
    for _, group in labels.items():
        if any(kind == "skel" for kind, _ in group):
            for kind, obj in group:
                if kind == "mesh":
                    translucent_mesh_ids.add(id(obj))

    # Add actors
    for m, actor in zip(meshes, mesh_actors):
        if id(m) in translucent_mesh_ids:
            actor.GetProperty().SetOpacity(min(actor.GetProperty().GetOpacity(), 0.2))
        renderer.AddActor(actor)

    for a in skel_actors:
        renderer.AddActor(a)

    # Scalar bar for radii
    if lut is not None:
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle("Radius")
        bar.SetWidth(0.1)
        bar.SetHeight(0.3)
        renderer.AddActor2D(bar)

    # Render window & interactor
    win = vtk.vtkRenderWindow()
    win.AddRenderer(renderer)
    win.SetSize(2000, 1500)
    win.SetWindowName("Skeliner 3D Viewer")

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    picker = vtk.vtkCellPicker()
    _add_recenter_controls(picker, iren, renderer)
    iren.AddObserver("ExitEvent", lambda *_: iren.TerminateApp())

    win.Render()
    iren.Start()


# visualize contact sites
def _bbox_from_faces(
    mesh: trimesh.Trimesh,
    faces_idx: np.ndarray,
    *,
    scale: float = 1.0,
    margin: float = 0.0,
) -> BBox | None:
    """
    AABB from a subset of faces. Returns coordinates in *scaled* space.
    margin: expand box by this absolute amount in the same (scaled) units.
    """
    faces_idx = np.asarray(faces_idx, np.int64)
    if faces_idx.size == 0:
        return None
    vidx = np.unique(mesh.faces[faces_idx].ravel())
    if vidx.size == 0:
        return None
    V = mesh.vertices[vidx] * float(scale)
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    if margin > 0.0:
        m = float(margin)
        lo = lo - m
        hi = hi + m
    return BBox(lo.astype(float), hi.astype(float))


def _bbox_union(a: BBox | None, b: BBox | None) -> BBox | None:
    if a is None:
        return b
    if b is None:
        return a
    lo = np.minimum(a.minpt, b.minpt)
    hi = np.maximum(a.maxpt, b.maxpt)
    return BBox(lo, hi)


def _add_mesh_toggles(interactor, renderer, actor_A, actor_B):
    import vtk

    # On-screen hint (optional)
    hud = vtk.vtkTextActor()
    hud.SetInput("[a] toggle A   [b] toggle B")
    hudprop = hud.GetTextProperty()
    hudprop.SetFontSize(20)
    hudprop.SetColor(1.0, 1.0, 1.0)
    hud.SetDisplayPosition(10, 10)
    renderer.AddActor2D(hud)

    def on_keypress(obj, evt):
        key = obj.GetKeySym()
        if key == "a":
            actor_A.SetVisibility(0 if actor_A.GetVisibility() else 1)
            renderer.GetRenderWindow().Render()
        elif key == "b":
            actor_B.SetVisibility(0 if actor_B.GetVisibility() else 1)
            renderer.GetRenderWindow().Render()

    interactor.AddObserver("KeyPressEvent", on_keypress)


def _vtk_faces_actor(
    mesh: trimesh.Trimesh,
    faces_idx: np.ndarray,
    color=(1.0, 0.0, 0.0),
    opacity: float = 1.0,
    scale: float = 1.0,
    patch_lift: float = 0.0,  # << new: in display units
    show_edges: bool = False,
):
    faces_idx = np.asarray(faces_idx, np.int64)
    if faces_idx.size == 0:
        return None

    sub = mesh.submesh([faces_idx], append=True, repair=False)

    # Micro-lift along normals (convert display -> model units)
    if patch_lift > 0.0 and sub.vertices.size:
        lift_model = float(patch_lift) / float(scale)
        # trimesh computes normals lazily; ensure availability
        n = sub.vertex_normals
        sub = trimesh.Trimesh(
            vertices=sub.vertices + lift_model * n, faces=sub.faces, process=False
        )

    actor = _vtk_mesh_actor(sub, color=color, opacity=opacity, scale=scale)

    # Optional edge overlay; if on, also offset lines to avoid z-fight
    actor.GetProperty().SetEdgeVisibility(bool(show_edges))
    if show_edges:
        actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
        actor.GetProperty().SetLineWidth(1.0)

    return actor


def view_contacts(
    A: trimesh.Trimesh,
    B: trimesh.Trimesh,
    contacts,  # ContactSitesResult
    *,
    scale: float = 1.0,
    color_A: tuple[float, float, float] = (0.82, 0.86, 1.00),  # light bluish
    color_B: tuple[float, float, float] = (1.00, 0.85, 0.85),  # light reddish
    site_opacity: float = 1.0,
    sides: str = "A",  # 'A', 'B', or 'both'
    window_size: tuple[int, int] = (2000, 1500),
    background: tuple[float, float, float] = (0.0, 0.062745, 0.129412),
    # --- NEW: AABB controls ---
    show_aabb: bool = True,
    aabb_mode: str = "union",  # "union" or "split"
    aabb_margin: float = 0.0,  # in scaled units (e.g., µm if scale=1e-3)
    aabb_color: str | tuple[float, float, float] = "match",  # "match" or fixed RGB
    aabb_line_width: float = 2.5,
):
    """
    Visualize two opaque meshes and overlay merged contact patches (A/B).
    Optionally draw an AABB per contact site:
      - union: one box per site around faces_A ∪ faces_B
      - split: one box per site per-side (A and/or B)
    """
    try:
        import vtk  # noqa: F401
    except Exception as e:
        raise ImportError("This viewer requires VTK. Try: pip install vtk") from e

    s = sides.lower()
    doA = s in ("a", "both")
    doB = s in ("b", "both")

    # ----------------- base scene -----------------
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*background)

    # Base meshes (opaque)
    actor_A = _vtk_mesh_actor(A, color=color_A, opacity=1.0, scale=scale)
    actor_B = _vtk_mesh_actor(B, color=color_B, opacity=1.0, scale=scale)
    renderer.AddActor(actor_A)
    renderer.AddActor(actor_B)

    # ----------------- contact patches -----------------
    palette = COLORS + BBOX_COLORS
    n_sites = max(len(contacts.faces_A), len(contacts.faces_B))

    # Precompute “fixed” aabb color if requested
    fixed_box_color = None
    if isinstance(aabb_color, (tuple, list)):
        fixed_box_color = tuple(map(float, aabb_color))

    for i in range(n_sites):
        site_color = palette[i % len(palette)]

        # ----- draw patches -----
        if doA and i < len(contacts.faces_A):
            fa = contacts.faces_A[i]
            act = _vtk_faces_actor(
                A,
                fa,
                color=site_color,
                opacity=site_opacity,
                scale=scale,
                patch_lift=0.05,
            )

            if act is not None:
                renderer.AddActor(act)

        if doB and i < len(contacts.faces_B):
            fb = contacts.faces_B[i]
            act = _vtk_faces_actor(
                B,
                fb,
                color=site_color,
                opacity=site_opacity,
                scale=scale,
                patch_lift=0.05,
            )
            if act is not None:
                renderer.AddActor(act)

        # ----- draw AABBs (optional) -----
        if not show_aabb:
            continue

        # choose color
        box_color = (
            site_color
            if (aabb_color == "match")
            else (fixed_box_color or BBOX_COLORS[i % len(BBOX_COLORS)])
        )

        if aabb_mode.lower() == "split":
            if doA and i < len(contacts.faces_A):
                boxA = _bbox_from_faces(
                    A, contacts.faces_A[i], scale=scale, margin=aabb_margin
                )
                if boxA is not None:
                    actor_boxA = _vtk_bbox_actor(
                        boxA, color=box_color, line_width=aabb_line_width
                    )
                    renderer.AddActor(actor_boxA)
            if doB and i < len(contacts.faces_B):
                boxB = _bbox_from_faces(
                    B, contacts.faces_B[i], scale=scale, margin=aabb_margin
                )
                if boxB is not None:
                    actor_boxB = _vtk_bbox_actor(
                        boxB, color=box_color, line_width=aabb_line_width
                    )
                    renderer.AddActor(actor_boxB)
        else:  # union (default)
            boxA = (
                _bbox_from_faces(
                    A, contacts.faces_A[i], scale=scale, margin=aabb_margin
                )
                if (doA and i < len(contacts.faces_A))
                else None
            )
            boxB = (
                _bbox_from_faces(
                    B, contacts.faces_B[i], scale=scale, margin=aabb_margin
                )
                if (doB and i < len(contacts.faces_B))
                else None
            )
            boxU = _bbox_union(boxA, boxB)
            if boxU is not None:
                actor_boxU = _vtk_bbox_actor(
                    boxU, color=box_color, line_width=aabb_line_width
                )
                renderer.AddActor(actor_boxU)

    # ----------------- window / interaction -----------------
    win = vtk.vtkRenderWindow()
    win.AddRenderer(renderer)
    win.SetSize(*window_size)
    win.SetWindowName("Skeliner Contact Patches")

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    picker = vtk.vtkCellPicker()
    _add_recenter_controls(picker, iren, renderer)
    iren.AddObserver("ExitEvent", lambda *_: iren.TerminateApp())

    _add_mesh_toggles(iren, renderer, actor_A, actor_B)

    win.Render()
    iren.Start()
