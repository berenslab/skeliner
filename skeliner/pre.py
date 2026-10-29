"""skeliner.pre – mesh preprocessing utilities."""

import copy
import functools
import time
import warnings
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from skeliner._backend import kernels
from skeliner.python import mesh as _py_mesh
from skeliner.python import pre as _py_pre
from skeliner.dataclass import (
    Discarded,
    Mesh,
    MeshComponents,
    MeshStats,
    Neurites,
    Organelles,
    Reassignment,
    Soma,
    mesh_args,
)

if TYPE_CHECKING:
    from scipy.spatial import KDTree

__all__ = [
    "apply_reassignment",
    "break_up_mesh",
    "compact_mesh",
    "compute_mesh_stats",
    "fill_holes",
    "find_disconnected",
    "find_fragments",
    "find_fusions",
    "find_gaps",
    "find_holes",
    "find_nucleus_center",
    "find_organelles",
    "find_parallel_patches",
    "find_soma",
    "find_soma_via_ring_cutoff",
    "preprocess",
    "preview_reassignment",
    "remove_fins",
    "remove_fragments",
    "remove_fusions",
    "remove_gaps",
    "remove_islands",
    "remove_organelles",
    "remove_parallel_patches",
    "soma_face_mask",
]


# ---------------------------------------------------------------------
#  Verbose timing helper (same style as skeletonize._timed)
# ---------------------------------------------------------------------


@contextmanager
def _timed(label: str, *, verbose: bool):
    """Context manager printing ``↳  label … N.NN s``."""
    if not verbose:
        yield lambda *_: None
        return

    _PAD = 47
    print(f" {label:<{_PAD}} …", end="", flush=True)
    t0 = time.perf_counter()
    _msgs: list[str] = []

    def log(msg: str) -> None:
        _msgs.append(str(msg))

    try:
        yield log
    finally:
        dt = time.perf_counter() - t0
        print(f" {dt:.2f} s")
        for m in _msgs:
            print(f"      └─ {m}")


def _deprecated(instead: str | None = None):
    """Mark a public function deprecated: each call warns, at the caller's
    line, naming *instead* when there is something to use in its place.
    The outermost decorator, so that the warning skips the others' frames
    (a warning placed inside skeliner is not shown by default)."""

    def wrap(fn):
        message = f"sk.pre.{fn.__name__}() is deprecated and will be removed"
        if instead:
            message += f"; use {instead} instead"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        return wrapper

    return wrap


_non_degenerate = _py_mesh.non_degenerate


def _edge_to_faces(mesh: Mesh) -> dict[tuple[int, int], list[int]]:
    """Build edge→face adjacency, skipping degenerate faces."""
    return _py_mesh.edge_to_faces(np.asarray(mesh.faces))


def _edge_map(mesh: Mesh):
    """Edge → faces of *mesh* as the active path's ``Topology``.

    It reads like the dict :func:`_edge_to_faces` builds — keys, face lists
    and iteration order — so code written against the dict takes it.
    """
    return kernels.mesh.Topology(
        np.asarray(mesh.faces, dtype=np.int64), len(mesh.vertices)
    )


_row_dot = _py_mesh.row_dot


def _svd_rows(x: np.ndarray, full_matrices: bool) -> np.ndarray:
    """``np.linalg.svd(x, full_matrices=...)[2]``.  The Rust core calls back
    for every SVD: the platform's LAPACK rounds as no Rust solver does."""
    return np.linalg.svd(x, full_matrices=full_matrices)[2]


_nearest = _py_mesh.nearest


def _median_edge(mesh) -> float:
    """``np.median(mesh.edges_unique_length)``, kept on a :class:`Mesh` once
    taken."""
    if not isinstance(mesh, Mesh):
        return float(np.median(mesh.edges_unique_length))
    return float(mesh._get("median_edge", lambda: np.median(mesh.edges_unique_length)))


def _face_edge_components(mesh: Mesh) -> tuple[np.ndarray, int]:
    """Connected components of faces using edge adjacency.

    Two faces are in the same component only if they share an edge
    (not just a vertex).  Faces with a repeated vertex are labelled -2 and
    connect nothing; components are numbered by their lowest face.

    Returns
    -------
    labels : np.ndarray
        (nFaces,) int array — component id for each face.
    main : int
        Component id of the largest component, ties to the lowest id.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return kernels.pre.face_edge_components(faces, len(mesh.vertices))


def _main_component(
    mesh: Mesh, labels: np.ndarray, largest: int, soma: "Soma | None"
) -> int:
    """The cell's main component: the one its soma lies on, else *largest*.

    A soma need not lie on the largest component — a segmentation can cut
    it from its arbor, which is then the larger piece.  Its vertices vote
    with their last face's component, as the ring kernel counts them, so
    the surfaces it holds inside are outvoted by its membrane; ties go to
    the lowest label.
    """
    if soma is None or soma.verts is None or len(soma.verts) == 0:
        return int(largest)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    last = np.full(len(mesh.vertices), -1, dtype=np.int64)
    last[faces.ravel()] = np.repeat(np.arange(len(faces)), 3)
    on = last[np.asarray(soma.verts, dtype=np.int64)]
    comp = np.asarray(labels)[on[on >= 0]]
    comp = comp[comp >= 0]
    if len(comp) == 0:
        return int(largest)
    return int(np.bincount(comp).argmax())


@_deprecated()
@mesh_args()
def find_holes(
    mesh: Mesh,
    *,
    verbose: bool = False,
) -> list[list[int]]:
    """Find boundary loops (holes) in the main surface of the mesh.

    .. deprecated::
        Neither the pipeline nor the viewer uses it; it will be removed.

    A hole is a closed loop of boundary edges — edges used by exactly
    one face.  Only holes in the main edge-adjacency component are
    returned; floating fragments (faces connected by at most a single
    vertex) are ignored.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    list[list[int]]
        Each element is an ordered list of vertex indices forming one
        boundary loop.
    """
    return _find_holes(mesh, verbose)


def _find_holes(mesh: Mesh, verbose: bool) -> list[list[int]]:
    """The boundary loops :func:`find_holes` returns."""
    face_comp, main_comp = _face_edge_components(mesh)

    # Build edge → face mapping and identify boundary edges
    edge_to_faces = _edge_to_faces(mesh)

    # Boundary edges whose face is in the main component
    boundary_edges = []
    for e, fis in edge_to_faces.items():
        if len(fis) == 1 and face_comp[fis[0]] == main_comp:
            boundary_edges.append(e)

    if verbose:
        print(f"[skeliner.pre] Holes: {len(boundary_edges)} boundary edges")

    if not boundary_edges:
        if verbose:
            print("[skeliner.pre] No holes found")
        return []

    # Trace closed loops from boundary edges
    adj: dict[int, set[int]] = defaultdict(set)
    for v1, v2 in boundary_edges:
        adj[v1].add(v2)
        adj[v2].add(v1)

    visited: set[int] = set()
    loops: list[list[int]] = []

    for start in sorted(adj):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current = start
        prev = -1
        while True:
            nbs = [n for n in adj[current] if n != prev]
            if not nbs:
                break
            closed = False
            for n in nbs:
                if n == start and len(loop) > 2:
                    loops.append(loop)
                    closed = True
                    break
            if closed:
                break
            nxt = next((n for n in nbs if n not in visited), None)
            if nxt is None:
                break
            visited.add(nxt)
            loop.append(nxt)
            prev = current
            current = nxt

    if verbose:
        print(f"[skeliner.pre] Found {len(loops)} holes")
        for i, loop in enumerate(loops):
            pts = mesh.vertices[loop]
            perimeter = sum(
                np.linalg.norm(pts[(j + 1) % len(pts)] - pts[j])
                for j in range(len(pts))
            )
            print(f"  Hole {i}: {len(loop)} vertices, perimeter={perimeter:.1f}")

    return loops


def _ear_clip_2d(pts_2d: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a 2D polygon.

    Parameters
    ----------
    pts_2d : np.ndarray
        (N, 2) ordered polygon vertices.

    Returns
    -------
    list[tuple[int, int, int]]
        Triangle vertex indices into *pts_2d*.
    """
    n = len(pts_2d)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Ensure CCW winding
    indices = list(range(n))
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += pts_2d[i, 0] * pts_2d[j, 1]
        signed_area -= pts_2d[j, 0] * pts_2d[i, 1]
    if signed_area < 0:
        indices = indices[::-1]

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _pt_in_tri(p, a, b, c):
        d1, d2, d3 = _cross(p, a, b), _cross(p, b, c), _cross(p, c, a)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    def _is_ear(idxs, i):
        m = len(idxs)
        p_i, c_i, n_i = (i - 1) % m, i, (i + 1) % m
        a, b, c = pts_2d[idxs[p_i]], pts_2d[idxs[c_i]], pts_2d[idxs[n_i]]
        if _cross(a, b, c) <= 0:
            return False
        for j in range(m):
            if j in (p_i, c_i, n_i):
                continue
            if _pt_in_tri(pts_2d[idxs[j]], a, b, c):
                return False
        return True

    triangles: list[tuple[int, int, int]] = []
    for _ in range(n * n):
        m = len(indices)
        if m == 3:
            triangles.append((indices[0], indices[1], indices[2]))
            break
        if m < 3:
            break
        ear_found = False
        for i in range(m):
            if _is_ear(indices, i):
                p_i, n_i = (i - 1) % m, (i + 1) % m
                triangles.append((indices[p_i], indices[i], indices[n_i]))
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            for i in range(1, m - 1):
                triangles.append((indices[0], indices[i], indices[i + 1]))
            break

    return triangles


def _fill_single_hole(
    mesh: Mesh,
    loop: list[int],
    target_edge: float,
    vert_offset: int,
    vtree: "KDTree | None" = None,
    vert_to_faces: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill one boundary loop with curvature-aware tessellation.

    Returns (new_vertices (N,3), new_faces (M,3)).
    Face indices use original mesh indices for boundary vertices
    and ``vert_offset + i`` for newly created vertices.
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import KDTree

    verts = mesh.vertices
    boundary_pts = verts[loop]
    n_bnd = len(loop)

    # ── Local 2D frame via SVD ────────────────────────────────────────
    centroid_3d = boundary_pts.mean(axis=0)
    centered = boundary_pts - centroid_3d
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    u_ax, v_ax, n_ax = Vt[0], Vt[1], Vt[2]

    bnd_2d = centered @ np.column_stack([u_ax, v_ax])
    bnd_h = centered @ n_ax

    # ── RBF surface from boundary + nearby mesh vertices ──────────────
    if vtree is None:
        vtree = KDTree(verts)
    nearby: set[int] = set()
    for vi in loop:
        nearby.update(vtree.query_ball_point(verts[vi], 3.0 * target_edge))
    nearby -= set(loop)

    if nearby:
        nrb_centered = verts[sorted(nearby)] - centroid_3d
        nrb_2d = nrb_centered @ np.column_stack([u_ax, v_ax])
        nrb_h = nrb_centered @ n_ax
        # Filter: keep only nearby vertices that project OUTSIDE the
        # boundary polygon in 2D.  Vertices from the opposite side of
        # a tube project inside and cause RBF overshoot (spikes).
        n_poly = len(bnd_2d)
        inside = np.zeros(len(nrb_2d), dtype=bool)
        px, py = nrb_2d[:, 0], nrb_2d[:, 1]
        j = n_poly - 1
        for ii in range(n_poly):
            xi, yi = bnd_2d[ii]
            xj, yj = bnd_2d[j]
            cond = ((yi > py) != (yj > py)) & (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            )
            inside ^= cond
            j = ii
        outside = ~inside
        nrb_2d = nrb_2d[outside]
        nrb_h = nrb_h[outside]
        if len(nrb_2d) > 0:
            rbf_pts = np.vstack([bnd_2d, nrb_2d])
            rbf_vals = np.concatenate([bnd_h, nrb_h])
        else:
            rbf_pts = bnd_2d
            rbf_vals = bnd_h
    else:
        rbf_pts = bnd_2d
        rbf_vals = bnd_h

    try:
        rbf = RBFInterpolator(rbf_pts, rbf_vals, kernel="thin_plate_spline")
    except np.linalg.LinAlgError:
        rbf = None

    # ── Ear-clip triangulation in 2D ──────────────────────────────────
    tri_list = _ear_clip_2d(bnd_2d)

    # ── Refine by longest-edge bisection ──────────────────────────────
    # Boundary edges (shared with original mesh) must not be split to
    # avoid T-junctions.
    bnd_edge_set: set[tuple[int, int]] = set()
    for idx in range(n_bnd):
        a, b = idx, (idx + 1) % n_bnd
        bnd_edge_set.add((min(a, b), max(a, b)))

    pts: list[np.ndarray] = list(bnd_2d)
    faces = [list(t) for t in tri_list]
    threshold = target_edge * 1.5
    edge_midpoints: dict[tuple[int, int], int] = {}

    for _ in range(200):
        new_faces: list[list[int]] = []
        changed = False
        for f in faces:
            i, j, k = f
            pi, pj, pk = pts[i], pts[j], pts[k]
            edges_sorted = sorted(
                [
                    (float(np.linalg.norm(pj - pi)), i, j),
                    (float(np.linalg.norm(pk - pj)), j, k),
                    (float(np.linalg.norm(pi - pk)), k, i),
                ],
                reverse=True,
            )
            split_done = False
            for elen, va, vb in edges_sorted:
                if elen <= threshold:
                    break
                ekey = (min(va, vb), max(va, vb))
                if ekey in bnd_edge_set:
                    continue
                if ekey in edge_midpoints:
                    mi = edge_midpoints[ekey]
                else:
                    mi = len(pts)
                    pts.append((pts[va] + pts[vb]) / 2.0)
                    edge_midpoints[ekey] = mi
                vc = [v for v in (i, j, k) if v != va and v != vb][0]
                new_faces.append([va, mi, vc])
                new_faces.append([mi, vb, vc])
                split_done = True
                changed = True
                break
            if not split_done:
                new_faces.append(f)
        faces = new_faces
        if not changed:
            break

    pts_arr = np.array(pts)
    faces_arr = np.array(faces, dtype=np.intp)
    n_new = len(pts_arr) - n_bnd

    # ── Map new 2D points → 3D via RBF height ────────────────────────
    if n_new > 0:
        new_2d = pts_arr[n_bnd:]
        if rbf is not None:
            new_h = np.clip(rbf(new_2d), bnd_h.min(), bnd_h.max())
        else:
            # Fallback: inverse-distance weighted interpolation from boundary
            dists = np.linalg.norm(
                new_2d[:, np.newaxis, :] - bnd_2d[np.newaxis, :, :], axis=2
            )
            weights = 1.0 / np.maximum(dists, 1e-10) ** 2
            new_h = (weights * bnd_h[np.newaxis, :]).sum(axis=1) / weights.sum(axis=1)
        new_3d = (
            centroid_3d
            + new_2d[:, 0:1] * u_ax
            + new_2d[:, 1:2] * v_ax
            + new_h[:, np.newaxis] * n_ax
        )
    else:
        new_3d = np.empty((0, 3))

    # ── Remap face indices ────────────────────────────────────────────
    loop_arr = np.array(loop, dtype=np.intp)
    remapped = np.empty_like(faces_arr)
    bnd_mask = faces_arr < n_bnd
    remapped[bnd_mask] = loop_arr[faces_arr[bnd_mask]]
    remapped[~bnd_mask] = vert_offset + (faces_arr[~bnd_mask] - n_bnd)

    # ── Consistent face orientation ───────────────────────────────────
    local_v = np.vstack([boundary_pts, new_3d]) if n_new > 0 else boundary_pts
    tri_normals = np.cross(
        local_v[faces_arr[:, 1]] - local_v[faces_arr[:, 0]],
        local_v[faces_arr[:, 2]] - local_v[faces_arr[:, 0]],
    )
    # Find adjacent faces via vert_to_faces index (fast) or fallback
    adj_fi: list[int] = []
    if vert_to_faces is not None:
        seen: set[int] = set()
        for vi in loop:
            for fi in vert_to_faces[vi]:
                if fi not in seen:
                    adj_fi.append(fi)
                    seen.add(fi)
                    if len(seen) > 50:  # enough for a good reference normal
                        break
            if len(seen) > 50:
                break
    else:
        loop_set = set(loop)
        _good = _non_degenerate(mesh.faces)
        adj_fi = [
            fi
            for fi, face in enumerate(mesh.faces)
            if _good[fi] and set(int(v) for v in face) & loop_set
        ]
    if adj_fi:
        ref_n = mesh.face_normals[adj_fi].mean(axis=0)
        if np.dot(tri_normals.mean(axis=0), ref_n) < 0:
            remapped = remapped[:, ::-1]

    return new_3d, remapped


def _fill_advancing_front(
    loop: list[int],
    vertices: np.ndarray,
    face_normals: np.ndarray,
    vert_to_faces: list[list[int]] | None = None,
) -> np.ndarray:
    """Fill a boundary loop using advancing front.

    Repeatedly picks the sharpest angle on the boundary and closes it
    with a triangle. No new vertices are created — only existing
    boundary vertices are used.

    Returns (M, 3) int array of new faces.
    """
    n = len(loop)
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)
    if n == 3:
        tri = np.array([[loop[0], loop[1], loop[2]]], dtype=np.int64)
        # Orient consistently with neighbors
        if vert_to_faces is not None:
            ref_fi = []
            for vi in loop[:1]:
                ref_fi.extend(vert_to_faces[vi][:5])
            if ref_fi:
                ref_n = face_normals[ref_fi].mean(axis=0)
                v0, v1, v2 = vertices[loop[0]], vertices[loop[1]], vertices[loop[2]]
                tri_n = np.cross(v1 - v0, v2 - v0)
                if np.dot(tri_n, ref_n) < 0:
                    tri = tri[:, ::-1]
        return tri

    # Determine consistent winding from adjacent mesh faces
    flip = False
    if vert_to_faces is not None:
        ref_fi = []
        for vi in loop[:3]:
            ref_fi.extend(vert_to_faces[vi][:5])
        if ref_fi:
            ref_n = face_normals[ref_fi].mean(axis=0)
            v0, v1, v2 = vertices[loop[0]], vertices[loop[1]], vertices[loop[2]]
            tri_n = np.cross(v1 - v0, v2 - v0)
            if np.dot(tri_n, ref_n) < 0:
                flip = True

    # Work with a mutable list of boundary vertex indices
    bnd = list(loop)
    new_faces = []

    for _ in range(n * 2):  # safety limit
        m = len(bnd)
        if m < 3:
            break
        if m == 3:
            if flip:
                new_faces.append([bnd[0], bnd[2], bnd[1]])
            else:
                new_faces.append([bnd[0], bnd[1], bnd[2]])
            break

        # Find the ear with the smallest interior angle
        # (sharpest corner = best triangle to close)
        best_i = -1
        best_angle = float("inf")

        for i in range(m):
            p = vertices[bnd[(i - 1) % m]]
            c = vertices[bnd[i]]
            n_pt = vertices[bnd[(i + 1) % m]]

            e1 = p - c
            e2 = n_pt - c
            len1 = np.linalg.norm(e1)
            len2 = np.linalg.norm(e2)
            if len1 < 1e-10 or len2 < 1e-10:
                best_i = i
                best_angle = 0
                break
            cos_a = np.dot(e1, e2) / (len1 * len2)
            cos_a = np.clip(cos_a, -1, 1)
            angle = np.arccos(cos_a)

            if angle < best_angle:
                best_angle = angle
                best_i = i

        if best_i < 0:
            break

        # Create triangle at this ear
        prev_i = (best_i - 1) % m
        next_i = (best_i + 1) % m
        if flip:
            new_faces.append([bnd[prev_i], bnd[next_i], bnd[best_i]])
        else:
            new_faces.append([bnd[prev_i], bnd[best_i], bnd[next_i]])

        # Remove the vertex from the boundary
        bnd.pop(best_i)

    if not new_faces:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(new_faces, dtype=np.int64)


def _fill_dome(
    loop: list[int],
    vertices: np.ndarray,
    face_normals: np.ndarray,
    vert_to_faces: list[list[int]] | None = None,
    dome_factor: float = 0.5,
    vert_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill a boundary loop with a convex dome cap.

    Creates a center vertex pushed outward along the average boundary
    normal, with concentric rings for larger holes.

    Returns ``(new_vertices (K, 3), new_faces (M, 3))``.  Face indices
    use original mesh indices for boundary vertices and
    ``vert_offset + i`` for newly created vertices.
    """
    n = len(loop)
    if n < 3:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    boundary_pts = vertices[loop]
    centroid = boundary_pts.mean(axis=0)

    # ── Outward normal from adjacent faces ────────────────────────────
    avg_normal = None
    if vert_to_faces is not None:
        adj_normals = []
        for vi in loop:
            for fi in vert_to_faces[vi]:
                adj_normals.append(face_normals[fi])
        if adj_normals:
            avg_normal = np.mean(adj_normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-10:
                avg_normal = avg_normal / norm
            else:
                avg_normal = None

    if avg_normal is None:
        centered = boundary_pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        avg_normal = Vt[2]
        if vert_to_faces is not None and vert_to_faces[loop[0]]:
            ref = face_normals[vert_to_faces[loop[0]][0]]
            if np.dot(avg_normal, ref) < 0:
                avg_normal = -avg_normal

    # ── Small holes: flat fill (no dome needed) ───────────────────────
    if n == 3:
        tri = np.array([[loop[0], loop[1], loop[2]]], dtype=np.int64)
        v0, v1, v2 = vertices[loop[0]], vertices[loop[1]], vertices[loop[2]]
        tri_n = np.cross(v1 - v0, v2 - v0)
        if np.dot(tri_n, avg_normal) < 0:
            tri = tri[:, ::-1]
        return np.empty((0, 3)), tri

    # ── Dome geometry ─────────────────────────────────────────────────
    radii = np.linalg.norm(boundary_pts - centroid, axis=1)
    avg_radius = float(radii.mean())
    dome_height = avg_radius * dome_factor

    # Intermediate rings for larger holes (better triangle quality)
    n_rings = max(0, (n - 4) // 15)

    new_verts_list: list[np.ndarray] = []
    ring_indices: list[list[int]] = []

    for r in range(1, n_rings + 1):
        t = r / (n_rings + 1)  # 0 → boundary, 1 → center
        frac = 1.0 - t  # radial fraction
        h = dome_height * (1.0 - frac * frac)  # parabolic profile
        ring_pts = centroid + (boundary_pts - centroid) * frac + avg_normal * h
        base = vert_offset + len(new_verts_list)
        ring_indices.append(list(range(base, base + n)))
        for pt in ring_pts:
            new_verts_list.append(pt)

    # Center vertex
    center = centroid + avg_normal * dome_height
    center_idx = vert_offset + len(new_verts_list)
    new_verts_list.append(center)

    # ── Triangulation ─────────────────────────────────────────────────
    faces: list[list[int]] = []
    prev = loop  # original vertex indices

    # Boundary → rings (quad strips → 2 triangles each)
    for ring in ring_indices:
        for i in range(n):
            j = (i + 1) % n
            faces.append([prev[i], prev[j], ring[j]])
            faces.append([prev[i], ring[j], ring[i]])
        prev = ring

    # Last ring (or boundary) → center fan
    for i in range(n):
        j = (i + 1) % n
        faces.append([prev[i], prev[j], center_idx])

    new_verts = np.array(new_verts_list)

    # ── Consistent orientation ────────────────────────────────────────
    def _pt(idx: int) -> np.ndarray:
        return vertices[idx] if idx < vert_offset else new_verts[idx - vert_offset]

    f0 = faces[0]
    tri_n = np.cross(_pt(f0[1]) - _pt(f0[0]), _pt(f0[2]) - _pt(f0[0]))
    if np.dot(tri_n, avg_normal) < 0:
        faces = [[f[1], f[0], f[2]] for f in faces]

    return new_verts, np.array(faces, dtype=np.int64)


@_deprecated()
@mesh_args()
def fill_holes(
    mesh: Mesh,
    *,
    holes: list[list[int]] | None = None,
    method: str = "advancing_front",
    max_perimeter_mult: float | None = None,
    dome_factor: float = 0.5,
    verbose: bool = False,
) -> Mesh:
    """Fill holes in the mesh.

    .. deprecated::
        Neither the pipeline nor the viewer uses it; it will be removed.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (typically after ``remove_organelles``).
    holes : list[list[int]] or None
        Pre-computed boundary loops from :func:`find_holes`.  If
        provided, detection is skipped and these loops are used directly.
    method : str, default "advancing_front"
        Filling method:
        - ``"advancing_front"`` — fast, closes boundary edges one by one
          by picking the sharpest angle. No new vertices. Works for all
          hole sizes.
        - ``"dome"`` — fills with a convex dome cap pushed outward along
          the boundary normal. Creates new vertices (center + optional
          concentric rings). Fast, O(n). Best for neurite tip holes.
        - ``"rbf"`` — projects to 2D, ear-clips, refines, lifts with RBF.
          Better surface quality for small planar holes but slow and fails
          for large curved holes.
    max_perimeter_mult : float or None
        Skip holes whose perimeter exceeds this multiple of the median
        edge length.  ``None`` = no limit (fill all holes).
        Default ``None`` for advancing_front/dome, ``50.0`` for rbf.
    dome_factor : float, default 0.5
        Height of dome as fraction of hole radius (only for ``"dome"``).
        0 = flat, 0.5 = gentle dome, 1.0 = hemisphere.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    Mesh
        Mesh with holes filled.
    """
    if holes is not None:
        loops = holes
        if verbose:
            print(f"[skeliner.pre] Using provided holes ({len(loops)} loops)")
    else:
        loops = _find_holes(mesh, verbose)
    if not loops:
        return mesh

    target_edge = _median_edge(mesh)

    # Default max_perimeter depends on method
    if max_perimeter_mult is None:
        if method == "rbf":
            max_perimeter_mult = 50.0
        # advancing_front: no limit
    max_perimeter = (
        max_perimeter_mult * target_edge
        if max_perimeter_mult is not None
        else float("inf")
    )

    if verbose:
        print(
            f"[skeliner.pre] Method: {method}, target edge: {target_edge:.1f}"
            + (
                f", max perimeter: {max_perimeter:.0f}"
                if max_perimeter < float("inf")
                else ", no perimeter limit"
            )
        )

    # Pre-build vert_to_faces for orientation (skip degenerate faces)
    _good = _non_degenerate(mesh.faces)
    vert_to_faces: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for fi, face in enumerate(mesh.faces):
        if _good[fi]:
            for v in face:
                vert_to_faces[int(v)].append(fi)

    # Filter loops by perimeter
    valid_loops: list[list[int]] = []
    n_skipped = 0
    for loop in loops:
        pts = mesh.vertices[loop]
        perimeter = float(
            np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum()
        )
        if perimeter > max_perimeter:
            n_skipped += 1
        else:
            valid_loops.append(loop)

    if verbose:
        print(
            f"[skeliner.pre] Filling {len(valid_loops)} holes"
            + (f" (skipped {n_skipped} too large)" if n_skipped else "")
        )

    if method == "advancing_front":
        new_faces: list[np.ndarray] = []
        for loop in valid_loops:
            nf = _fill_advancing_front(
                loop, mesh.vertices, mesh.face_normals, vert_to_faces
            )
            if len(nf):
                new_faces.append(nf)

        if not new_faces:
            return mesh

        all_faces = np.vstack([mesh.faces] + new_faces)
        result = Mesh(mesh.vertices, all_faces)

    elif method == "dome":
        new_verts: list[np.ndarray] = []
        new_face_list: list[np.ndarray] = []
        offset = len(mesh.vertices)

        for loop in valid_loops:
            nv, nf = _fill_dome(
                loop,
                mesh.vertices,
                mesh.face_normals,
                vert_to_faces,
                dome_factor=dome_factor,
                vert_offset=offset,
            )
            if len(nv) > 0:
                new_verts.append(nv)
            if len(nf) > 0:
                new_face_list.append(nf)
            offset += len(nv)

        if not new_face_list:
            return mesh

        v_parts = [mesh.vertices] + new_verts
        f_parts = [mesh.faces] + new_face_list
        result = Mesh(np.vstack(v_parts), np.vstack(f_parts))

    elif method == "rbf":
        from scipy.spatial import KDTree

        vtree = KDTree(mesh.vertices)
        new_verts_rbf: list[np.ndarray] = []
        new_face_rbf: list[np.ndarray] = []
        offset = len(mesh.vertices)

        for loop in valid_loops:
            nv, nf = _fill_single_hole(
                mesh, loop, target_edge, offset, vtree, vert_to_faces
            )
            new_verts_rbf.append(nv)
            new_face_rbf.append(nf)
            offset += len(nv)

        v_parts = [mesh.vertices] + [v for v in new_verts_rbf if len(v) > 0]
        f_parts = [mesh.faces] + [f for f in new_face_rbf if len(f) > 0]
        result = Mesh(np.vstack(v_parts), np.vstack(f_parts))
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'advancing_front', 'dome', or 'rbf'."
        )

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(result.vertices):,} verts, "
            f"{len(result.faces):,} faces"
            + (f" ({n_skipped} holes skipped)" if n_skipped else "")
        )
    return result


# ── Merge selected faces ────────────────────────────────────────────


_border_edges = _py_mesh.border_edges


def _trace_border_loops(
    sel: set[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    faces: np.ndarray | None = None,
) -> list[list[int]]:
    """Trace closed loops along the border between *sel* and kept faces.

    Each loop starts at its smallest vertex and, wherever the border offers
    a choice, steps to the smallest unvisited neighbour, so the loops depend
    only on which edges are on the border.  Given a ``Topology`` (of
    either path, :func:`_edge_map`) it traces them; given the dict, pass
    *faces* to examine only the selection's edges (see
    :func:`_border_edges`).
    """
    if hasattr(edge_to_faces, "border_loops"):  # a Topology, of either path
        return edge_to_faces.border_loops(sel)
    return _trace_border_loops_py(sel, edge_to_faces, faces)


_trace_border_loops_py = _py_mesh.border_loops


_validate_loop_pair = _py_pre.validate_loop_pair


_removal_would_sever = _py_pre.removal_would_sever


def _sever_cost(
    sel: set[int],
    face_adj: dict[int, set[int]] | None,
    budget: int,
    *,
    topology=None,
) -> int:
    """How many faces removing *sel* would strand, capped at ``budget``.

    :func:`_removal_would_sever` only says a sever is *possible* — two
    border loops mean the patch wraps the surface.  It says nothing about
    the cost, and the two differ by orders of magnitude: on 564241053 an
    annulus stranded 13,438 faces, on 554656742 a 22-face patch of the
    same shape stranded **4**.  Deciding on the loop count alone throws
    away good bridges to avoid trivial ones.

    Everything but the largest surviving piece is stranded.  Each piece
    is grown separately and abandoned once it passes ``budget``, so the
    work is bounded by the answer rather than by the mesh: at most one
    piece can exceed the budget and still be the largest, and if two do
    the cost already exceeds it.  Returns ``budget + 1`` in that case.

    With a ``Topology`` (of either path) as *topology* it does it and
    *face_adj* is not read.
    """
    if topology is not None:
        return topology.sever_cost(sel, budget)
    return _sever_cost_py(sel, face_adj, budget)


_sever_cost_py = _py_pre.sever_cost


_rescue_size = _py_pre.rescue_size


def _expand_tip_to_good_rim(
    tip: list[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_adj: dict[int, set[int]] | None,
    *,
    target_verts: int = 6,
    max_iters: int = 20,
    max_eat_factor: int = 30,
    faces: np.ndarray | None = None,
) -> tuple[set[int], list[int] | None]:
    """Iteratively peel BFS rings from *tip* until its border has at least
    *target_verts* vertices.

    Returns ``(expanded_tip, best_loop)``. ``best_loop`` is the largest
    border loop found at the final tip (``None`` if no loop ever traced).

    The initial tip from :func:`find_gaps` can produce a degenerate
    "rim" (e.g. 3 verts when the tip is a tongue attached at a single
    triangle). Each expansion step adds the next BFS ring of
    edge-neighbors and re-traces the border. Expansion is capped at
    ``len(tip) * max_eat_factor`` faces to avoid swallowing the entire
    component.  *faces* is passed to :func:`_trace_border_loops`.  Given a
    ``Topology`` (of either path) it does it and *face_adj* is not read.
    """
    if hasattr(edge_to_faces, "expand_tip"):  # a Topology, of either path
        return edge_to_faces.expand_tip(tip, target_verts, max_iters, max_eat_factor)
    return _expand_tip_to_good_rim_py(
        tip,
        edge_to_faces,
        face_adj,
        target_verts=target_verts,
        max_iters=max_iters,
        max_eat_factor=max_eat_factor,
        faces=faces,
    )


_expand_tip_to_good_rim_py = _py_pre.expand_tip_to_good_rim


def _expand_selection_per_group(
    sel: list[int] | set[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_adj: dict[int, set[int]] | None,
    *,
    target_verts: int = 6,
    max_iters: int = 20,
    max_eat_factor: int = 30,
    faces: np.ndarray | None = None,
) -> tuple[set[int], list[list[int]]]:
    """Split *sel* into edge-connected sub-groups, expand each
    independently with :func:`_expand_tip_to_good_rim`, then re-trace
    the border loops on the merged expanded selection.

    This is the user-driven merge analogue of the per-side expansion
    in :func:`remove_gaps`. Expanding each connected sub-selection on
    its own (rather than the whole selection at once) lets each side's
    rim grow toward similar sizes — the zipper stitch's KDTree weld
    only behaves manifoldly when the two rims are matched in vertex
    count, so symmetric per-side expansion is what unlocks a clean
    bridge.  Given a ``Topology`` (of either path) it does it and
    *face_adj* is not read.
    """
    if hasattr(edge_to_faces, "expand_selection_per_group"):  # a Topology
        return edge_to_faces.expand_selection_per_group(
            sel, target_verts, max_iters, max_eat_factor
        )
    return _expand_selection_per_group_py(
        sel,
        edge_to_faces,
        face_adj,
        target_verts=target_verts,
        max_iters=max_iters,
        max_eat_factor=max_eat_factor,
        faces=faces,
    )


_expand_selection_per_group_py = _py_pre.expand_selection_per_group


def _fit_loop_circle(pts: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a circle to a 3D point loop.  Returns (center, radius, normal)."""
    center = pts.mean(axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    radius = float(np.mean(radii))
    # Normal from PCA: smallest eigenvector of covariance
    cov = np.cov((pts - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # smallest eigenvalue = normal direction
    return center, radius, normal


def _stitch_and_rebuild(
    mesh: Mesh,
    sel: set[int],
    loop_pairs: list[tuple[list[int], list[int]]],
) -> Mesh:
    """Remove *sel* faces, stitch each loop pair, rebuild mesh once."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    new_verts, faces = kernels.pre.stitch_and_rebuild(
        np.asarray(mesh.faces, dtype=np.int64),
        verts,
        np.asarray(mesh.face_normals, dtype=np.float64),
        _median_edge(mesh),
        sel,
        loop_pairs,
        [],
        lambda ids: _fit_loop_circle(verts[ids]),
    )
    if len(new_verts):
        verts = np.vstack([verts, new_verts])
    return Mesh(verts, faces)


@mesh_args()
def merge_selected_faces(
    mesh: Mesh,
    face_indices: list[int],
    *,
    verbose: bool = False,
) -> Mesh:
    """Remove selected faces and stitch the resulting boundary loops.

    Workflow: lasso-select faces at the tips of two disconnected
    components, then merge.  The selected faces are removed, the
    border edges (between selected and remaining faces) are traced
    into loops, the two closest loops are paired, and a zipper
    stitch connects them with new triangles.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    face_indices : list[int]
        Indices of faces to remove (the "bridge" region).
    verbose : bool

    Returns
    -------
    Mesh
        Mesh with selected faces replaced by a stitched strip.
    """
    sel_input = set(face_indices)
    if not sel_input:
        return mesh

    # the Topology answers face adjacency too, so no face map is built
    edge_to_faces = _edge_map(mesh)
    face_adj = None

    if verbose:
        print(f"[skeliner.pre] Merge: requested {len(sel_input)} faces")

    # Expand the user selection if needed so every border loop is a
    # real rim (not a 3-vert tongue base). Each edge-connected
    # sub-group of the selection is expanded independently — same fix
    # as remove_gaps's per-side expansion — so the resulting rims
    # converge to matching sizes (the zipper stitch only welds
    # manifoldly when both rims have similar vertex counts).
    sel, loops = _expand_selection_per_group(
        sel_input, edge_to_faces, face_adj, faces=np.asarray(mesh.faces)
    )

    if verbose:
        if len(sel) != len(sel_input):
            print(
                f"[skeliner.pre] Merge: expanded selection "
                f"{len(sel_input)} -> {len(sel)} faces"
            )
        for i, lp in enumerate(loops):
            pts = mesh.vertices[lp]
            perimeter = float(
                np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum()
            )
            print(
                f"[skeliner.pre]   Loop {i}: {len(lp)} verts, perimeter={perimeter:.1f}"
            )

    if len(loops) < 2:
        if verbose:
            print("[skeliner.pre] Fewer than 2 loops — cannot stitch")
        keep = np.ones(len(mesh.faces), dtype=bool)
        for fi in sel:
            keep[fi] = False
        return _rebuild_mesh(mesh, keep)

    # Pair loops by closest centroid
    centroids = [mesh.vertices[lp].mean(axis=0) for lp in loops]
    pairs: list[tuple[list[int], list[int]]] = []
    available = list(range(len(loops)))
    while len(available) >= 2:
        best_d = float("inf")
        best_i, best_j = 0, 1
        for ii in range(len(available)):
            for jj in range(ii + 1, len(available)):
                d = float(
                    np.linalg.norm(centroids[available[ii]] - centroids[available[jj]])
                )
                if d < best_d:
                    best_d = d
                    best_i, best_j = ii, jj
        la = loops[available[best_i]]
        lb = loops[available[best_j]]
        # Safety net: if expansion couldn't fix this pair, skip it
        # rather than introduce a fusion. Both loops still get their
        # tip faces removed (so they become unbridged holes the user
        # can inspect), unlike `remove_gaps` where we keep the tips.
        reason = _validate_loop_pair(la, lb)
        if reason is not None:
            if verbose:
                print(f"[skeliner.pre]   Pair skipped after expansion ({reason})")
        else:
            pairs.append((la, lb))
        available.pop(best_j)
        available.pop(best_i)

    if verbose:
        print(f"[skeliner.pre] Stitching {len(pairs)} loop pair(s) ...")

    return _stitch_and_rebuild(mesh, sel, pairs)


@mesh_args()
def remove_selected_faces(
    mesh: Mesh,
    face_indices: list[int],
    *,
    verbose: bool = False,
) -> Mesh:
    """Remove the selected faces from the mesh, leaving open holes.

    Companion to :func:`merge_selected_faces`.  Where
    ``merge_selected_faces`` bridges the rims of the removed region
    with new triangles, this function leaves the boundary loops as
    open holes — useful when the user just wants to delete a chunk
    and inspect the result, or feed it into other repair steps.

    Face and vertex indices are preserved (removed faces become
    degenerate ``[0, 0, 0]``) so face-based annotations remain valid.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    face_indices : list[int]
        Indices of faces to remove.
    verbose : bool

    Returns
    -------
    Mesh
        Mesh with selected faces removed.
    """
    if not face_indices:
        return mesh
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[np.asarray(face_indices, dtype=np.int64)] = False
    if verbose:
        n_remove = int((~keep).sum())
        print(f"[skeliner.pre] remove_selected_faces: {n_remove} faces")
    return _rebuild_mesh(mesh, keep)


def _find_island_faces(
    faces: np.ndarray,
    active: np.ndarray,
    min_faces: int = 3,
) -> np.ndarray:
    """Return mask of island faces among *active* faces.

    Islands are edge-connected components with fewer than *min_faces*.
    """
    active = active & _non_degenerate(faces)
    active_idx = np.where(active)[0]
    af = faces[active_idx]
    n_active = len(af)
    if n_active == 0:
        return np.zeros(len(faces), dtype=bool)

    # Build edge-to-face adjacency for active faces only
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for li, face in enumerate(af):
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edge_to_faces[(min(a, b), max(a, b))].append(li)

    # BFS connected components (local indices)
    labels = np.full(n_active, -1, dtype=np.intp)
    comp_id = 0
    best_id, best_size = 0, 0
    for seed in range(n_active):
        if labels[seed] >= 0:
            continue
        queue = [seed]
        labels[seed] = comp_id
        size = 0
        while queue:
            li = queue.pop()
            size += 1
            face = af[li]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                for nb in edge_to_faces[(min(a, b), max(a, b))]:
                    if labels[nb] < 0:
                        labels[nb] = comp_id
                        queue.append(nb)
        if size > best_size:
            best_id, best_size = comp_id, size
        comp_id += 1

    comp_ids, counts = np.unique(labels, return_counts=True)
    small = set(
        int(c) for c, n in zip(comp_ids, counts) if n < min_faces and c != best_id
    )
    if not small:
        return np.zeros(len(faces), dtype=bool)

    island_local = np.array([int(labels[li]) in small for li in range(n_active)])
    mask = np.zeros(len(faces), dtype=bool)
    mask[active_idx[island_local]] = True
    return mask


def _find_fin_faces(
    faces: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Return mask of fin faces among *active* faces (iterates until stable)."""
    work = active.copy() & _non_degenerate(faces)
    max_v = int(faces.max()) + 1
    total_mask = np.zeros(len(faces), dtype=bool)

    for _ in range(100):
        active_idx = np.where(work)[0]
        af = faces[active_idx]
        n_active = len(af)
        if n_active == 0:
            break

        v0, v1, v2 = af[:, 0], af[:, 1], af[:, 2]
        e_a = np.stack([np.minimum(v0, v1), np.maximum(v0, v1)], axis=1)
        e_b = np.stack([np.minimum(v1, v2), np.maximum(v1, v2)], axis=1)
        e_c = np.stack([np.minimum(v2, v0), np.maximum(v2, v0)], axis=1)
        all_edges = np.concatenate([e_a, e_b, e_c], axis=0)

        edge_keys = all_edges[:, 0].astype(np.int64) * max_v + all_edges[:, 1]
        _, inverse, counts = np.unique(
            edge_keys, return_inverse=True, return_counts=True
        )
        edge_counts = counts[inverse]

        is_boundary = edge_counts.reshape(3, n_active) == 1
        fin_local = is_boundary.sum(axis=0) >= 2
        n_fins = int(fin_local.sum())
        if n_fins == 0:
            break

        fin_global_idx = active_idx[fin_local]
        total_mask[fin_global_idx] = True
        work[fin_global_idx] = False

    return total_mask


@_deprecated()
@mesh_args()
def find_fragments(
    mesh: Mesh,
    *,
    min_faces: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    """Detect all fragment faces (islands and fins) without modifying the mesh.

    .. deprecated::
        Neither the pipeline nor the viewer uses it; it will be removed.

    Alternates island and fin detection on a boolean mask until stable,
    exactly mirroring the logic of :func:`remove_fragments`.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    min_faces : int, default 3
        Edge-connected components with fewer faces than this are islands.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — fragment faces (islands + fins).
    """
    return _find_fragments(mesh, min_faces, verbose)


def _find_fragments(mesh: Mesh, min_faces: int, verbose: bool) -> np.ndarray:
    """The fragment mask :func:`find_fragments` returns."""
    faces = mesh.faces
    active = np.ones(len(faces), dtype=bool)
    fragment = np.zeros(len(faces), dtype=bool)

    for _ in range(100):
        prev = int(active.sum())
        islands = _find_island_faces(faces, active, min_faces)
        active &= ~islands
        fragment |= islands

        fins = _find_fin_faces(faces, active)
        active &= ~fins
        fragment |= fins

        if int(active.sum()) == prev:
            break

    if verbose:
        n_frag = int(fragment.sum())
        print(
            f"[skeliner.pre] Fragments: {n_frag:,} faces "
            f"({int(islands.sum()):,} islands, {int(fins.sum()):,} fins "
            f"in last pass)"
        )
    return fragment


_otsu_threshold = _py_pre.otsu_threshold


@mesh_args()
def find_soma(
    mesh: Mesh,
    *,
    organelles: Organelles | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Soma | None:
    """Find the soma by its ring cutoff: rings of vertices grown over the
    surface from the nucleus widen through the soma and narrow into the
    neurites, and the soma ends where they narrow.

    1. The nucleus (:func:`find_nucleus_center`): its void in the mesh's
       Z-slices gives the centre, the nucleus' height, and the seeds — the
       soma surface's vertices at the nucleus' middle Z-level.
    2. Rings: breadth-first over the component the seeds' outline lies on
       — the soma's outer membrane, where the slice also cuts surfaces
       inside the soma that are not joined to it, as in EyeWire 1's
       meshes — from its seeds (or from its vertex nearest the centre),
       one ring per step.
    3. The cutoff: the largest connected piece of each ring grows to a
       peak inside the soma and falls once the rings pass into the
       neurites; the soma ends at the first ring after the peak whose
       piece is under Otsu's threshold of the sizes from the peak on.  The
       peak is sought within three nucleus heights of rings, and the soma
       spans at least one.
    4. The soma: an ellipsoid fitted to the rings up to the cutoff, which
       then claims the vertices inside it, absorbs small pockets, sheds the
       neurite stubs at its exits and lets go of vertices that only
       organelle faces hold, refitted as its vertices change.

    All thresholds are derived from the mesh.  It was
    ``find_soma_via_ring_cutoff``, now a deprecated alias.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (before organelle removal).
    organelles : Organelles or None
        From :func:`find_organelles`: the organelles the soma leaves out.
        None for none — a mesh that traces no organelles, such as
        hemibrain's; they are not looked for here.
    verbose : bool, default False
        Print summary.
    mesh_stats : MeshStats or None
        From :func:`compute_mesh_stats`.  Reuses ``face_comp`` and
        ``main_ci`` to skip redundant component detection.

    Returns
    -------
    Soma or None
        Fitted ellipsoidal soma, or *None* if no nucleus is found, too
        few indicators exist, or fewer than four of the vertices it claims
        lie on anything but organelle faces.
    """
    if verbose:
        print("[skeliner.pre] Finding soma …")
        _t0 = time.perf_counter()

    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # ── 1. Find soma centre via nucleus detection ──────────────
    nuc = find_nucleus_center(mesh)
    if nuc is None:
        if verbose:
            dt = time.perf_counter() - _t0
            print(f"[skeliner.pre] Soma: no nucleus found ({dt:.1f}s)")
        return None
    center = nuc["center"]

    soma, cutoff, max_ring = _ring_cutoff_soma(mesh, nuc, labels, main, organelles)
    if len(soma.verts) < 4:
        # it let go of what it claimed: only organelle faces held those
        if verbose:
            dt = time.perf_counter() - _t0
            print(f"[skeliner.pre] Soma: none off the organelles ({dt:.1f}s)")
        return None

    if verbose:
        dt = time.perf_counter() - _t0
        zlo, zhi = nuc["z_range"]
        print(
            f"[skeliner.pre] Soma: "
            f"nucleus=[{center[0]:.0f}, "
            f"{center[1]:.0f}, {center[2]:.0f}] "
            f"r={nuc['peak_r']:.0f} "
            f"z=[{zlo:.0f}–{zhi:.0f}]; "
            f"ellipsoid axes=[{soma.axes[0]:.0f}, "
            f"{soma.axes[1]:.0f}, {soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} verts, "
            f"cutoff ring {cutoff}/{max_ring} ({dt:.1f}s)"
        )

    soma.nucleus = {
        "center": nuc["center"],
        "peak_r": nuc["peak_r"],
        "z_range": nuc["z_range"],
        "slices": nuc["slices"],
    }
    return soma


def _ring_cutoff_soma(
    mesh: Mesh,
    nuc: dict,
    labels: np.ndarray,
    main: int,
    organelles: Organelles | None,
) -> tuple["Soma", int, int]:
    """The soma :func:`find_soma` grows from nucleus *nuc*:
    ``(soma, cutoff ring, last ring)``.

    The rings, vertex assignment and pruning run in Rust; each ellipsoid
    fit is :meth:`Soma.fit`, called back from there.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    fits: list[Soma] = []

    def fit(ids, refit):
        try:
            soma = Soma.fit(verts[ids], verts=ids if refit else None)
        except ValueError:
            if not refit:
                raise
            return None
        fits.append(soma)
        return soma.center, soma._W, float(soma.axes.min())

    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    seeds = np.asarray(nuc.get("soma_seed_vi", []), dtype=np.int64)
    if organelles is None:
        organelles = Organelles.empty(len(faces))
    k, soma_verts, cutoff, max_ring = kernels.pre.find_soma_ring_cutoff(
        faces,
        verts,
        labels,
        int(main),
        _outline_seeds(verts, faces, labels, seeds),
        np.asarray(nuc["center"], dtype=np.float64),
        float(nuc["z_range"][1] - nuc["z_range"][0]),
        float(mesh.edges_unique_length.mean()),
        np.asarray(organelles.mask, dtype=bool),
        fit,
    )
    soma = fits[k]
    soma.verts = soma_verts
    return soma, cutoff, max_ring


def _outline_seeds(
    verts: np.ndarray, faces: np.ndarray, labels: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    """The *seeds* on the component their XY outline lies on.

    The seeds are a slice through the soma.  Its outline is the outer
    membrane; most of the seeds can lie inside it, on a surface the
    membrane encloses without being joined to it, and the rings would
    then grow over that surface instead.  Each vertex takes its last
    face's component, as the ring kernel counts them.
    """
    if len(seeds) < 3:
        return seeds
    from scipy.spatial import ConvexHull, QhullError

    try:
        rim = ConvexHull(verts[seeds, :2]).vertices
    except QhullError:
        return seeds
    last = np.full(len(verts), -1, dtype=np.int64)
    last[faces.ravel()] = np.repeat(np.arange(len(faces)), 3)
    comp = np.where(last[seeds] >= 0, labels[last[seeds]], -1)
    on_rim = comp[rim][comp[rim] >= 0]
    if len(on_rim) == 0:
        return seeds
    return seeds[comp == np.bincount(on_rim).argmax()]


_build_edge_to_faces = _py_pre.build_edge_to_faces


def _face_components_fast(faces, face_indices) -> list[np.ndarray]:
    """Connected components of *face_indices*, largest first.

    Vectorised equivalent of :func:`_face_components`, which walks the
    mesh in Python and dominates :func:`break_up_mesh`'s runtime.

    Order matters, because callers index the result: neurite ids come
    from this position.  :func:`_face_components` discovers components
    by BFS from the lowest unvisited face, so they arrive in ascending
    order of their smallest face id, and its stable sort by size leaves
    equal-sized ones in that order.  Sorting on ``(-size, min face id)``
    reproduces it without relying on the labeller's numbering.
    """
    return kernels.pre.face_components(
        np.asarray(faces, dtype=np.int64), np.asarray(face_indices, dtype=np.int64)
    )


def _face_components(faces, edge_to_faces, face_indices):
    """BFS connected components on a subset of faces."""
    fi_set = (
        set(face_indices.tolist())
        if hasattr(face_indices, "tolist")
        else set(face_indices)
    )
    visited: set[int] = set()
    components: list[np.ndarray] = []
    for fi in face_indices:
        fi = int(fi)
        if fi in visited:
            continue
        comp: list[int] = []
        q = deque([fi])
        visited.add(fi)
        while q:
            cur = q.popleft()
            comp.append(cur)
            face = faces[cur]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nb in edge_to_faces[e]:
                    if nb in fi_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        components.append(np.array(comp, dtype=np.intp))
    components.sort(key=len, reverse=True)
    return components


def _usable_face_mask(mesh: Mesh) -> np.ndarray:
    """Faces :func:`break_up_mesh` will consider: 3 distinct verts, non-zero area."""
    return kernels.pre.usable_faces(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def soma_face_mask(
    faces: np.ndarray,
    soma_verts: np.ndarray | None,
) -> np.ndarray:
    """Faces the soma owns: those with at least 2 of 3 soma vertices.

    Soma membership is stored per *vertex*
    (:attr:`~skeliner.dataclass.Soma.verts`); this majority rule is what
    turns it into the face set that :func:`break_up_mesh` cuts the mesh
    along.  A face straddling the boundary with exactly one soma vertex
    stays with the neurite.

    Parameters
    ----------
    faces : (nFaces, 3) int
    soma_verts : (n,) int or None
        Vertex ids belonging to the soma.  ``None`` or empty yields an
        all-False mask, which is the no-soma case.

    Returns
    -------
    (nFaces,) bool
    """
    faces = np.asarray(faces)
    if soma_verts is None or len(soma_verts) == 0:
        return np.zeros(len(faces), dtype=bool)
    return np.isin(faces, np.asarray(soma_verts)).sum(axis=1) >= 2


def _build_org_output(
    organelles: Organelles,
    expanded_mask: np.ndarray,
) -> Organelles:
    """Build an Organelles dataclass from break_up_mesh results."""
    detected = organelles.pocket | organelles.isolated | organelles.manual
    return Organelles(
        pocket=organelles.pocket,
        isolated=organelles.isolated,
        expanded=expanded_mask & ~detected,
        manual=organelles.manual,
    )


@mesh_args()
def break_up_mesh(
    mesh: Mesh,
    soma: Soma | None,
    organelles: Organelles | None = None,
    *,
    rescued=None,
    released=None,
    claim_min_faces: int = 0,
    verbose: bool = False,
) -> MeshComponents:
    """Break the mesh using soma and organelles, classify the pieces.

    Removes soma + organelle faces, finds connected components of
    the remainder, and classifies each as:

    - **missed organelle** — not reachable from the main mesh body
      without crossing organelle faces (topologically trapped).
    - **missed soma** — reachable, its boundary mostly soma faces, no
      bigger than the soma, and not standing out of it like a neurite.
    - **neurite** — reachable, large enough to be a real branch.
    - **discarded** — reachable, but too small (below auto threshold).

    A component carrying a face named in *rescued* skips all three
    demotions and is a neurite regardless.

    The discard threshold is auto-inferred: components are sorted by
    size descending; once the cumulative face count reaches 95% of
    the total, the remaining components are discarded — but for those
    at least a quarter the size of the smallest kept, as a neuron with
    many primary dendrites has several of a size.

    The soma test is a *boundary* test: it measures what fraction of a
    component's outer edges are shared with soma faces, not how far the
    component reaches.  A process leaving the soma is capped at its far
    end, so the junction is nearly all the boundary it has, and a real
    dendritic stub can score well over 0.5 — which is what *rescued* is
    for.  A component with more faces than the soma is never absorbed: a
    patch of the soma ``find_soma`` missed is smaller than the soma, while
    on an organelle-free cell (fly neurons) the whole arbor can be one
    component that scores as soma.  Nor is one that stands out of the soma
    like a neurite: by more than half the soma's radius beyond its
    ellipsoid, and by more than three times the radius of its junction with
    the soma faces (the mean distance of the junction's vertices from their
    centroid).  A patch of the soma lies on it; a neurite's stump
    reaches out further than it is wide.  A stump shorter than that — a
    truncated axon, a tiny branch — is absorbed as before.

    When ``soma`` is ``None`` (no soma in this mesh — e.g. a fragment
    or a non-somal cell type), the soma face mask is empty, the
    "missed soma" classification is skipped, and every reachable
    non-organelle component flows straight into the neurite/discarded
    split.

    Parameters
    ----------
    mesh : Mesh
    soma : Soma or None
        From :func:`find_soma` (must have ``.verts``).
        Pass ``None`` if the mesh has no soma.
    organelles : Organelles or None
        From :func:`find_organelles`; None for a mesh that traces no
        organelles (hemibrain's, EyeWire 1's): nothing is cut out as one,
        and with nothing to be trapped behind, no piece is trapped.
    rescued : array-like of int, optional
        Face ids the user has declared arbor, naming a component they
        had in hand.  Any component carrying one is a neurite: not
        absorbed into the soma, not expanded into the organelles, not
        discarded by the size threshold, at any size.
    released : array-like of int, optional
        The same claim made with a lasso instead.  It escapes absorption
        and trapping just as *rescued* does, but carries a component past
        the size threshold only if the piece has at least
        *claim_min_faces*.  The two are separate because they differ in
        how well the user could aim: a lasso grazes slivers it did not
        mean, and releasing a patch strands isolated triangles at the
        boundary the >=2-of-3 face rule then moves, so an unfiltered
        lasso claim produces two-face neurites.
    claim_min_faces : int, default 0
        The floor for *released*.  Under it a claimed component still
        leaves the soma and the organelles — it lands in ``discarded``,
        where it is visible, rather than becoming a neurite.
    verbose : bool

    Returns
    -------
    MeshComponents
    """
    if verbose:
        print("[skeliner.pre] Breaking up mesh …")
        _t0 = time.perf_counter()

    faces = np.asarray(mesh.faces)
    verts = mesh.vertices
    nF = len(faces)

    def _face_claim(ids) -> np.ndarray:
        m = np.zeros(nF, dtype=bool)
        if ids is None:
            return m
        fi = np.unique(np.asarray(ids, dtype=np.int64))
        m[fi[(fi >= 0) & (fi < nF)]] = True
        return m

    # Naming a face claims its whole component as arbor.  Both kinds of
    # claim outrank organelle trapping and soma absorption — each is a
    # different reason a component stops being a neurite, and "this is
    # arbor" answers both, so an override covering only one of them would
    # be undone by the other on the next re-derive.  They part company at
    # the size threshold; see `released`.
    rescued_face = _face_claim(rescued)
    released_face = _face_claim(released)
    claimed_face = rescued_face | released_face

    if organelles is None:
        organelles = Organelles.empty(nF)
    # Mesh-mutating steps (e.g. remove_gaps) may have appended faces
    # since find_organelles ran. The new faces are stitch geometry, never
    # organelle membrane — pad with False so the masks align with `nF`.
    n_org = len(organelles.pocket)
    if n_org < nF:
        pad = np.zeros(nF - n_org, dtype=bool)
        organelles = Organelles(
            pocket=np.concatenate([organelles.pocket, pad]),
            isolated=np.concatenate([organelles.isolated, pad]),
            expanded=np.concatenate([organelles.expanded, pad]),
            manual=np.concatenate([organelles.manual, pad]),
        )

    usable = _usable_face_mask(mesh)
    soma_face = soma_face_mask(faces, soma.verts if soma is not None else None)

    components, kinds = _break_up_pieces(
        faces,
        verts,
        usable,
        organelles.mask,
        soma_face,
        claimed_face,
        soma is not None,
        centers=mesh.triangles_center,
        normals=mesh.face_normals,
    )
    if soma is not None:
        kinds = _neurites_left_out(
            components, kinds, faces, verts, soma, usable & soma_face
        )
    if len(components) == 0:
        org_out = _build_org_output(organelles, organelles.mask)
        return MeshComponents(
            soma=soma,
            organelles=org_out,
            neurites=Neurites([]),
            discarded=Discarded([]),
        )

    # --- classify ---
    neurite_candidates: list[np.ndarray] = []
    extra_soma_vi: list[int] = []
    organelles_expanded = organelles.mask.copy()
    n_trapped = 0
    n_soma_absorbed = 0
    n_claimed = 0

    for comp, kind in zip(components, kinds):
        n_claimed += bool(claimed_face[comp].any())
        if kind == _PIECE_TRAPPED:
            # trapped by organelles
            organelles_expanded[comp] = True
            n_trapped += len(comp)
        elif kind == _PIECE_SOMA:
            comp_vi = np.unique(faces[comp])
            extra_soma_vi.extend(comp_vi.tolist())
            n_soma_absorbed += len(comp)
        else:
            neurite_candidates.append(comp)

    # --- refit soma if we absorbed extra verts ---
    if soma is not None and extra_soma_vi:
        all_soma_vi = np.union1d(soma.verts, np.array(extra_soma_vi, dtype=np.intp))
        prev_nucleus = soma.nucleus
        soma = Soma.fit(verts[all_soma_vi], verts=all_soma_vi)
        soma.nucleus = prev_nucleus

    # --- auto-threshold: keep components covering 95% of total faces ---
    # and any as big as a quarter of the smallest of those: a neuron with
    # many primary dendrites, one piece each once the soma is cut out, has
    # several of a size, and the 95% cut otherwise lands among them
    neurite_candidates.sort(key=len, reverse=True)
    total_faces = sum(len(c) for c in neurite_candidates)
    cumsum = 0
    split_idx = len(neurite_candidates)
    for i, comp in enumerate(neurite_candidates):
        cumsum += len(comp)
        if cumsum >= total_faces * 0.95:
            split_idx = i + 1
            break
    if 0 < split_idx < len(neurite_candidates):
        floor = len(neurite_candidates[split_idx - 1]) / 4
        while split_idx < len(neurite_candidates) and (
            len(neurite_candidates[split_idx]) >= floor
        ):
            split_idx += 1

    neurites = neurite_candidates[:split_idx]
    discarded = neurite_candidates[split_idx:]

    # --- re-apply the hand overrides the threshold knows nothing about ---
    # Only here do the two claims differ.  Escaping the soma is a
    # judgement about *what the surface is* and holds at any size; being a
    # neurite is a judgement about whether there is enough of it to be a
    # branch, and only a claim the user could aim precisely earns that
    # unconditionally.  A lassoed piece under the floor still left the
    # soma — it lands in `discarded`, visible, and can be rescued
    # deliberately from there.
    promoted: list[np.ndarray] = []
    if claimed_face.any() and discarded:
        still_discarded: list[np.ndarray] = []
        for comp in discarded:
            keep = rescued_face[comp].any() or (
                len(comp) >= claim_min_faces and released_face[comp].any()
            )
            (promoted if keep else still_discarded).append(comp)
        if promoted:
            discarded = still_discarded
            neurites = sorted(neurites + promoted, key=len, reverse=True)

    if verbose:
        dt = time.perf_counter() - _t0
        n_disc_faces = sum(len(c) for c in discarded)
        parts = [f"{len(neurites)} neurites"]
        if discarded:
            parts.append(f"{len(discarded)} discarded ({n_disc_faces:,}f)")
        if n_claimed:
            parts.append(f"claimed {n_claimed}")
        if n_trapped:
            parts.append(f"trapped +{n_trapped:,}f")
        if n_soma_absorbed:
            parts.append(f"soma +{n_soma_absorbed:,}f")
        print(f"[skeliner.pre] break_up_mesh: {', '.join(parts)} ({dt:.1f}s)")

    org_out = _build_org_output(organelles, organelles_expanded)
    return MeshComponents(
        soma=soma,
        organelles=org_out,
        neurites=Neurites(neurites),
        discarded=Discarded(discarded),
    )


# Keep old name as alias
break_at_soma = break_up_mesh

# A soma piece standing out of the soma's ellipsoid this far, in its radii,
# and this many times further than the radius of its junction with the soma,
# is a neurite: a patch of the soma find_soma missed lies on the soma, a stub
# of a neurite stands out of it further than it is wide.  Soma.fit's √5
# envelope sits ~1.3× out from the surface, so half its radius past it is a
# stump about the soma's own radius long.
_NEURITE_REACH = 0.5
_NEURITE_ASPECT = 3.0


def _neurites_left_out(components, kinds, faces, verts, soma, soma_faces):
    """*kinds* with every soma piece that stands out of the soma like a
    neurite a candidate again (:data:`_NEURITE_REACH`,
    :data:`_NEURITE_ASPECT`): its farthest vertex outside the soma's
    ellipsoid by more than half the soma's equivalent radius, and by more
    than three times the radius of its junction — the mean distance from
    their centroid of the vertices of the edges it shares with the soma
    faces, which is a ring's radius however the ring is triangulated."""
    soma_ids = np.flatnonzero(np.asarray(kinds) == _PIECE_SOMA)
    if len(soma_ids) == 0:
        return kinds
    kinds = np.array(kinds)
    width = len(verts) + 1

    def edges(fs):
        e = np.sort(
            np.concatenate([fs[:, [0, 1]], fs[:, [1, 2]], fs[:, [2, 0]]]), axis=1
        )
        return e, e[:, 0] * width + e[:, 1]

    soma_keys = np.unique(edges(faces[soma_faces])[1])
    for k in soma_ids:
        comp = faces[components[k]]
        reach = float(soma.distance_to_surface(verts[np.unique(comp)]).max())
        if reach <= _NEURITE_REACH * soma.equiv_radius:
            continue
        e, keys = edges(comp)
        ring = verts[np.unique(e[np.isin(keys, soma_keys)])]
        junction = (
            np.linalg.norm(ring - ring.mean(axis=0), axis=1).mean()
            if len(ring)
            else 0.0
        )
        if reach > _NEURITE_ASPECT * junction:
            kinds[k] = _PIECE_CANDIDATE
    return kinds


# What :func:`_break_up_pieces` makes of a piece.
_PIECE_CANDIDATE = _py_pre.PIECE_CANDIDATE
_PIECE_TRAPPED = _py_pre.PIECE_TRAPPED
_PIECE_SOMA = _py_pre.PIECE_SOMA


def _break_up_pieces(
    faces,
    vertices,
    usable,
    organelle,
    soma_face,
    claimed_face,
    has_soma,
    *,
    centers,
    normals,
) -> tuple[list[np.ndarray], np.ndarray]:
    """The pieces :func:`break_up_mesh` classifies: components of the usable
    faces that are neither soma nor organelle, largest first, and the kind
    of each — :data:`_PIECE_TRAPPED` when only organelle faces connect it to
    its mesh component's body and it is an organelle's inside: no bigger
    than the organelles walling it, or more than half its faces looking into
    the cell (a neurite cut off by a ring of folds is bigger and looks out);
    :data:`_PIECE_SOMA` when more than half its outer edges meet soma faces
    and it has no more faces than the soma (only with a soma); else
    :data:`_PIECE_CANDIDATE`.  Claimed pieces are never trapped or soma.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    return kernels.pre.break_up_classify(
        np.asarray(faces, dtype=np.int64),
        len(vertices),
        np.asarray(usable, dtype=bool),
        np.asarray(organelle, dtype=bool),
        np.asarray(soma_face, dtype=bool),
        np.asarray(claimed_face, dtype=bool),
        bool(has_soma),
        vertices=vertices,
        centers=np.asarray(centers, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
    )


# ---------------------------------------------------------------------
#  Manual component reassignment
# ---------------------------------------------------------------------


def _cleared(mask, fi):
    out = mask.copy()
    out[fi] = False
    return out


def _component_effects(
    before: MeshComponents,
    after: MeshComponents,
    n_faces: int,
) -> list[tuple[str, str]]:
    """Cross-tabulate two component sets over the same mesh.

    Both sides index the same faces, so correspondence is exact: label
    every face with the component owning it on each side and read the
    mapping off.  One before-component reaching several after-components
    is a split; several reaching one is a merge.

    Pure renumbering is not reported.  ``break_up_mesh`` sorts by
    descending size, so ids shift whenever anything changes size and
    saying so every time would bury the real news.  A component crossing
    between neurites and discarded *is* reported — that is a verdict, not
    a position.  With nothing before (a freshly loaded cell) there is
    nothing to report: every component would be "new".
    """
    b_comps = list(before.neurites) + list(before.discarded)
    if not b_comps:
        # a first derive (a freshly loaded cell): nothing to compare against
        return []
    a_comps = list(after.neurites) + list(after.discarded)
    b_kinds = ["neurite"] * len(before.neurites) + ["discarded"] * len(before.discarded)
    a_kinds = ["neurite"] * len(after.neurites) + ["discarded"] * len(after.discarded)
    b_names = [f"{k} {i}" for k, i in _numbered(b_kinds)]
    a_names = [f"{k} {i}" for k, i in _numbered(a_kinds)]

    b_owner = np.full(n_faces, -1, dtype=np.intp)
    for i, arr in enumerate(b_comps):
        b_owner[arr] = i
    a_owner = np.full(n_faces, -1, dtype=np.intp)
    for i, arr in enumerate(a_comps):
        a_owner[arr] = i

    effects: list[tuple[str, str]] = []
    for i, arr in enumerate(b_comps):
        landed = a_owner[arr]
        landed = np.unique(landed[landed >= 0])
        if len(landed) == 0:
            effects.append((b_names[i], "dissolved into soma/organelles"))
            continue
        if len(landed) > 1:
            effects.append((b_names[i], f"split into {len(landed)}"))
            continue

        j = int(landed[0])
        fed = b_owner[a_comps[j]]
        fed = set(np.unique(fed[fed >= 0]).tolist())
        if len(fed) > 1:
            others = ", ".join(b_names[k] for k in sorted(fed - {i}))
            effects.append((b_names[i], f"merged with {others}"))
            continue

        delta = len(a_comps[j]) - len(arr)
        if b_kinds[i] != a_kinds[j]:
            note = f"reclassified as {a_names[j]}"
            if delta:
                note += f" ({delta:+,}f)"
            effects.append((b_names[i], note))
        elif delta > 0:
            effects.append((b_names[i], f"grown +{delta:,}f"))
        elif delta < 0:
            effects.append((b_names[i], f"shrunk {delta:,}f"))

    for j, arr in enumerate(a_comps):
        fed = b_owner[arr]
        if not (fed >= 0).any():
            effects.append((a_names[j], f"new ({len(arr):,}f)"))

    return effects


def _numbered(kinds: list[str]):
    """``['neurite', 'neurite', 'discarded'] -> [(neurite,0), (neurite,1), ...]``."""
    seen: dict[str, int] = {}
    for k in kinds:
        yield k, seen.get(k, 0)
        seen[k] = seen.get(k, 0) + 1


@mesh_args()
def preview_reassignment(
    mesh: Mesh,
    components: MeshComponents,
    faces,
    *,
    to: str,
    rescued=None,
    released=None,
    claim_min_faces: int = 0,
    verbose: bool = False,
) -> Reassignment:
    """Work out what assigning *faces* to *to* would do, without doing it.

    Only two things are stored: which vertices are soma, and which faces
    are organelles.  Neurites and discarded fragments are *derived* from
    those by :func:`break_up_mesh`, so every reassignment is expressed as
    a change to the soma vertex set or to an organelle mask, and the
    components follow.

    That is also why ``to='remainder'`` rather than a named neurite: the
    faces are released from the soma and organelle masks, and which
    neurite they join is decided by what they touch.  Releasing is only
    half of it — see *rescued*.

    The preview runs the real :func:`break_up_mesh` on the edited state
    rather than modelling it.  That matters — connectivity is only its
    second phase.  It also decides what is topologically trapped behind
    organelle faces, absorbs any component whose boundary is more than
    half soma back *into* the soma, and re-infers the neurite/discarded
    size threshold.  Cutting a band near the soma leaves crumbs that the
    absorption pass reclaims, so a connectivity-only forecast reports a
    split into eight where the answer is two.

    Because soma membership is per vertex under a >=2-of-3 majority rule
    (:func:`soma_face_mask`), the faces that actually change hands are
    the selection plus a fringe of neighbours sharing two of its
    vertices.  *entering* and *leaving* report that effective set;
    showing it is what keeps the difference between what was drawn and
    what moved from being a mystery.

    Parameters
    ----------
    mesh : Mesh
    components : MeshComponents
        Current state, from :func:`break_up_mesh`.
    faces : array-like of int
        Face ids to reassign.
    to : {'soma', 'organelle', 'remainder'}
        ``'soma'`` adds the faces' vertices to the soma — or, on components
        with none (find_soma found none), makes their largest connected
        piece the soma, its ellipsoid fitted to it; ``'organelle'`` sets them in
        :attr:`~skeliner.dataclass.Organelles.manual`; ``'remainder'``
        releases them from both, back to the arbor.
    rescued : array-like of int, optional
        The component-level arbor claims in force, passed straight to
        :func:`break_up_mesh`.  Without them the re-derive this preview
        runs undoes them: a released stub is still surrounded by soma, so
        absorption reclaims it, and a released speck is still small, so
        the threshold re-discards it.  Without the sets below,
        the release and its undo would happen in the same call.
    released : array-like of int, optional
        The lasso-drawn claims, likewise passed straight through.  A
        caller releasing faces with ``to='remainder'`` normally wants the
        selection *in this* set — otherwise the release and its undo
        happen in the same call, which is what "the reassignment did
        nothing" looks like from the outside.
    claim_min_faces : int, default 0
        Passed to :func:`break_up_mesh`, where it floors *released*.
    verbose : bool
        Passed to :func:`break_up_mesh`.

    Returns
    -------
    Reassignment

    See Also
    --------
    apply_reassignment : commit the result.
    """
    if to not in ("soma", "organelle", "remainder"):
        raise ValueError(f"to must be 'soma', 'organelle' or 'remainder', got {to!r}")

    mesh_faces = np.asarray(mesh.faces)
    sel = np.unique(np.asarray(faces, dtype=np.intp))
    if len(sel) == 0:
        raise ValueError("no faces selected")
    if sel[0] < 0 or sel[-1] >= len(mesh_faces):
        raise ValueError(
            f"face ids must lie in [0, {len(mesh_faces)}), got [{sel[0]}, {sel[-1]}]"
        )

    soma = components.soma
    org = components.organelles
    old_soma_verts = soma.verts if soma is not None else None
    sel_verts = np.unique(mesh_faces[sel])

    # --- the state change: soma vertices and/or organelle masks ---
    if to == "soma":
        if soma is None:
            soma = _soma_from_selection(mesh, sel)
            new_soma_verts = soma.verts
        else:
            new_soma_verts = np.union1d(old_soma_verts, sel_verts)
        new_org = org
    elif to == "organelle":
        manual = org.manual.copy()
        manual[sel] = True
        new_soma_verts = old_soma_verts
        new_org = Organelles(
            pocket=org.pocket,
            isolated=org.isolated,
            expanded=org.expanded,
            manual=manual,
        )
    else:
        # Release: drop every vertex of the selection from the soma, and
        # clear the selection from all organelle masks — including the
        # detected ones, which the user is overriding on purpose.
        new_soma_verts = (
            None if old_soma_verts is None else np.setdiff1d(old_soma_verts, sel_verts)
        )
        new_org = Organelles(
            pocket=_cleared(org.pocket, sel),
            isolated=_cleared(org.isolated, sel),
            expanded=_cleared(org.expanded, sel),
            manual=_cleared(org.manual, sel),
        )

    new_soma = None
    if soma is not None:
        new_soma = copy.copy(soma)
        new_soma.verts = new_soma_verts

    # --- what actually changes hands, fringe included ---
    usable = _usable_face_mask(mesh)
    before_arbor = usable & ~soma_face_mask(mesh_faces, old_soma_verts) & ~org.mask
    after_arbor = usable & ~soma_face_mask(mesh_faces, new_soma_verts) & ~new_org.mask
    leaving = np.flatnonzero(before_arbor & ~after_arbor)
    entering = np.flatnonzero(~before_arbor & after_arbor)

    after = break_up_mesh(
        mesh,
        new_soma,
        new_org,
        rescued=rescued,
        released=released,
        claim_min_faces=claim_min_faces,
        verbose=verbose,
    )

    return Reassignment(
        target=to,
        selected=sel,
        entering=entering,
        leaving=leaving,
        components=after,
        effects=_component_effects(components, after, len(mesh_faces)),
    )


def _soma_from_selection(mesh: Mesh, sel: np.ndarray) -> Soma:
    """The soma a hand selection of faces makes where none was found: its
    largest connected piece (faces joined by a shared vertex, so a lasso
    that took every other triangle is still one) — a soma is one body, and
    a lasso drawn through the mesh catches faces behind it — with an
    ellipsoid fitted to that piece's vertices as find_soma fits one
    (:meth:`Soma.fit`).  The re-derive that follows absorbs what the
    selection missed of the soma's surface, as it does for a detected soma.
    A piece with too few vertices, or flat, spans no ellipsoid.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    verts, local = np.unique(np.asarray(mesh.faces)[sel], return_inverse=True)
    local = local.reshape(-1, 3)
    a = np.concatenate([local[:, 0], local[:, 1]])
    b = np.concatenate([local[:, 1], local[:, 2]])
    n = len(verts)
    _, piece = connected_components(
        coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False
    )
    verts = verts[piece == np.bincount(piece).argmax()]
    if len(verts) >= 4:
        pts = np.asarray(mesh.vertices, dtype=np.float64)[verts]
        try:
            # a flat patch's least variance is zero or a rounding below it
            with np.errstate(invalid="ignore", divide="ignore"):
                soma = Soma.fit(pts, verts=verts)
        except ValueError:
            soma = None
        if soma is not None and np.isfinite(soma.axes).all() and soma.axes[-1] > 0:
            return soma
    raise ValueError(
        "these faces span no soma: select faces round it, not a flat patch"
    )


def apply_reassignment(
    components: MeshComponents,
    reassignment: Reassignment,
) -> None:
    """Commit a :class:`Reassignment` to *components*, in place.

    The preview already ran :func:`break_up_mesh`, so this only installs
    the result — soma, organelle masks and the re-derived components all
    come from that one run and cannot disagree with what was shown.

    Note that ``break_up_mesh`` refits the soma ellipsoid when its
    absorption pass pulls extra vertices in, so ``center``/``axes``/``R``
    can move.  A reassignment that absorbs nothing leaves the model
    untouched and changes only ``verts``.
    """
    after = reassignment.components
    components.soma = after.soma
    components.organelles = after.organelles
    components.neurites = after.neurites
    components.discarded = after.discarded


@_deprecated("find_soma()")
@mesh_args()
def find_soma_via_neurite_exclusion(
    mesh: Mesh,
    *,
    organelles: Organelles | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Soma | None:
    """Estimate soma by per-tip neurite exclusion.

    .. deprecated::
        Use :func:`find_soma`.

    Approach:
      0. Organelle clustering → soma center
      1. BFS on external surface (no organelle faces) from center
      2. Keep all rings up to ~1.5× peak ring, find tip clusters at
         the outer boundary, filter false tips by equator distance
      3. Per-tip neurite stub removal: for each tip, BFS inward with
         perpendicular ring seeding, detect tube→soma transition via
         sustained positive velocity + acceleration in the ring-size
         profile, exclude everything on the tip side.  Cumulative
         exclusion handles shared branches automatically.
      4. Fit ellipsoid to the cleaned set
    """
    from scipy.spatial import KDTree

    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # ── 0. Organelle clustering → soma center ─────────────────────
    if organelles is None:
        organelles = find_organelles(mesh, verbose=verbose)
    if organelles.mask.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma: no organelles found")
        return None

    org_fi = np.where(organelles.mask)[0]
    org_labels, _ = _face_edge_components(Mesh(mesh.vertices, mesh.faces[org_fi]))
    centroids = []
    for cid in np.unique(org_labels):
        local_fi = org_fi[org_labels == cid]
        verts = np.unique(mesh.faces[local_fi])
        centroids.append(mesh.vertices[verts].mean(axis=0))
    centroids = np.asarray(centroids)

    if verbose:
        print(
            f"[skeliner.pre] Soma: {organelles.mask.sum():,} organelle faces, "
            f"{len(centroids)} clusters"
        )
    if len(centroids) < 3:
        if verbose:
            print("[skeliner.pre] Soma: too few organelle clusters")
        return None

    tree = KDTree(centroids)
    dd, _ = tree.query(centroids, k=min(2, len(centroids)))
    nn_dists = dd[:, 1] if dd.ndim > 1 else dd
    nn_thresh, _ = _otsu_threshold(nn_dists)
    pairs = tree.query_pairs(r=nn_thresh)
    prox_adj: dict[int, set[int]] = defaultdict(set)
    for i, j in pairs:
        prox_adj[i].add(j)
        prox_adj[j].add(i)

    visited_prox: set[int] = set()
    best_comp: list[int] = []
    for start in range(len(centroids)):
        if start in visited_prox:
            continue
        comp: list[int] = []
        pq = deque([start])
        while pq:
            v = pq.popleft()
            if v in visited_prox:
                continue
            visited_prox.add(v)
            comp.append(v)
            for nv in prox_adj[v]:
                if nv not in visited_prox:
                    pq.append(nv)
        if len(comp) > len(best_comp):
            best_comp = comp

    core = centroids[best_comp]
    if len(core) < 3:
        if verbose:
            print("[skeliner.pre] Soma: no dense fragment cluster found")
        return None

    center = np.median(core, axis=0)
    if verbose:
        print(
            f"[skeliner.pre] Soma: dense cluster {len(core)}/{len(centroids)} fragments"
        )

    # ── 1. BFS on external surface from center ────────────────────
    main_fi = np.where(labels == main)[0]
    non_org_main = main_fi[~organelles.mask[main_fi]]

    adj_bfs: dict[int, list[int]] = defaultdict(list)
    for fi in non_org_main:
        v = mesh.faces[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj_bfs[a].append(b)
            adj_bfs[b].append(a)

    bfs_verts = np.fromiter(adj_bfs.keys(), dtype=np.intp)
    seed = int(
        bfs_verts[np.argmin(np.linalg.norm(mesh.vertices[bfs_verts] - center, axis=1))]
    )

    ring_level: dict[int, int] = {seed: 0}
    queue: deque[int] = deque([seed])
    ring_verts: dict[int, list[int]] = defaultdict(list)
    ring_verts[0].append(seed)
    while queue:
        v = queue.popleft()
        lv = ring_level[v]
        for nv in adj_bfs[v]:
            if nv not in ring_level:
                ring_level[nv] = lv + 1
                queue.append(nv)
                ring_verts[lv + 1].append(nv)

    # Find peak ring (largest connected component width)
    max_ring = max(ring_verts.keys())
    largest_comp_size = np.zeros(max_ring + 1)
    for lv in range(max_ring + 1):
        vr = ring_verts.get(lv, [])
        if not vr:
            continue
        ring_set = set(vr)
        vis_r: set[int] = set()
        mx = 0
        for st in vr:
            if st in vis_r:
                continue
            sz = 0
            rq = deque([st])
            while rq:
                u = rq.popleft()
                if u in vis_r:
                    continue
                vis_r.add(u)
                sz += 1
                for nu in adj_bfs[u]:
                    if nu in ring_set and nu not in vis_r:
                        rq.append(nu)
            if sz > mx:
                mx = sz
        largest_comp_size[lv] = mx

    peak_ring = int(np.argmax(largest_comp_size))

    # Reject cells with no localised bulge (no soma).
    # spread_ratio: how concentrated the wide rings are.  For a soma
    # the wide rings cluster at one spot (ratio < 1/3).  For a
    # uniform-width cell the wide rings are everywhere (ratio ~ 1).
    nonzero_mask = largest_comp_size > 0
    nonzero = largest_comp_size[nonzero_mask]
    spread_ratio = 1.0
    if len(nonzero) >= 3:
        width_thresh, _ = _otsu_threshold(nonzero)
        above_idx = np.where(nonzero_mask & (largest_comp_size > width_thresh))[
            0
        ].astype(float)
        all_idx = np.where(nonzero_mask)[0].astype(float)
        if len(above_idx) >= 2 and np.std(all_idx) > 0:
            spread_ratio = float(np.std(above_idx) / np.std(all_idx))

    if spread_ratio > 1.0 / 3:
        if verbose:
            print(
                f"[skeliner.pre] Soma: no localised bulge "
                f"(spread_ratio={spread_ratio:.3f})"
            )
        return None

    # ── 2. Keep rings 0 to 1.5× peak, find tip clusters ─────────
    boundary = peak_ring + peak_ring // 2
    soma_set: set[int] = set()
    for lv in range(min(boundary + 1, max_ring + 1)):
        soma_set.update(ring_verts[lv])

    # Outer edge: soma_set verts adjacent to non-soma_set verts
    outer_edge: set[int] = set()
    for v in soma_set:
        for nv in adj_bfs[v]:
            if nv not in soma_set:
                outer_edge.add(v)
                break

    # Dilated clustering: expand outer_edge by 1 ring for connectivity,
    # then cluster — keeps only original outer_edge verts per cluster.
    dilated = set(outer_edge)
    for v in outer_edge:
        for nv in adj_bfs[v]:
            if nv in soma_set:
                dilated.add(nv)

    visited_tip: set[int] = set()
    tip_clusters: list[list[int]] = []
    for start in outer_edge:
        if start in visited_tip:
            continue
        cluster: list[int] = []
        tq = deque([start])
        while tq:
            v = tq.popleft()
            if v in visited_tip:
                continue
            visited_tip.add(v)
            if v in outer_edge:
                cluster.append(v)
            for nv in adj_bfs[v]:
                if nv in dilated and nv not in visited_tip:
                    tq.append(nv)
        if cluster:
            tip_clusters.append(cluster)
    tip_clusters.sort(key=len, reverse=True)

    # Filter false tips: remove clusters whose median Euclidean distance
    # from soma center is below the equator distance (median distance at
    # peak ring).  These are BFS-inflated verts inside the soma body.
    equator_dists = [
        float(np.linalg.norm(mesh.vertices[v] - center))
        for v in ring_verts.get(peak_ring, [])
    ]
    equator_dist = float(np.median(equator_dists)) if equator_dists else 0.0

    filtered_tips: list[list[int]] = []
    for cluster in tip_clusters:
        med_dist = float(
            np.median([np.linalg.norm(mesh.vertices[v] - center) for v in cluster])
        )
        if med_dist >= equator_dist:
            filtered_tips.append(cluster)

    if verbose:
        print(
            f"[skeliner.pre] Soma: peak ring {peak_ring}, "
            f"boundary ring {boundary}, "
            f"{len(soma_set):,} verts, "
            f"{len(tip_clusters)} tip clusters → "
            f"{len(filtered_tips)} after equator filter"
        )

    # ── 3. Per-tip neurite stub removal ─────────────────────────
    #
    # For each tip, BFS inward through the domain with perpendicular
    # ring seeding.  Detect where the ring-size profile transitions
    # from tube (flat) to soma (sustained climb) using the first
    # derivative (velocity) and second derivative (acceleration).
    # Exclude everything on the tip side of the cut.  Cumulative
    # exclusion handles shared branches automatically.

    _PERP_REF = 10  # rings into tube before computing perpendicular seed
    _SMOOTH_W = 5  # smoothing window for ring-size profile

    def _perp_seed(tip_verts, domain):
        """BFS from tip, find perpendicular cross-section at _PERP_REF."""
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in tip_verts if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        ref = _PERP_REF
        if len(rngs) < ref + 3:
            return tip_verts  # tube too short, use original
        ctrs = np.array(
            [mesh.vertices[[v for v in r]].mean(axis=0) for r in rngs[: ref + 5]]
        )
        lo = max(0, ref - 2)
        hi = min(len(ctrs) - 1, ref + 2)
        axis = ctrs[hi] - ctrs[lo]
        nrm = np.linalg.norm(axis)
        if nrm == 0:
            return tip_verts
        axis /= nrm
        ctr = ctrs[ref]
        nbr: set[int] = set()
        for b in range(max(0, ref - 3), min(len(rngs), ref + 4)):
            nbr.update(rngs[b])
        el = float(
            np.median(
                [
                    np.linalg.norm(mesh.vertices[v] - mesh.vertices[nv])
                    for v in list(nbr)[:100]
                    for nv in adj_bfs[v]
                    if nv in nbr
                ]
            )
        )
        half = el * 1.5
        return [
            v for v in nbr if abs(float(np.dot(mesh.vertices[v] - ctr, axis))) < half
        ]

    def _bfs_from(seeds, domain):
        """BFS from seed verts, return list of rings."""
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in seeds if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        return rngs

    def _find_cut(ring_sizes):
        """Detect tube→soma transition via d1 > 0 AND d2 > 0 with
        adaptive sustain derived from tube noise level."""
        skip = _PERP_REF
        w = _SMOOTH_W
        if len(ring_sizes) < skip + 15:
            return skip
        smooth = np.convolve(ring_sizes, np.ones(w) / w, mode="valid")
        d1 = np.diff(smooth)
        d2 = np.diff(d1)
        off1 = w // 2 + 1
        off2 = w // 2 + 2
        start = max(skip, off2)
        # Measure tube noise: max consecutive d1>0 AND d2>0 in tube
        tube_end = min(start + 20, len(ring_sizes))
        max_con = 0
        con = 0
        for b in range(start, tube_end):
            i1, i2 = b - off1, b - off2
            if 0 <= i1 < len(d1) and 0 <= i2 < len(d2) and d1[i1] > 0 and d2[i2] > 0:
                con += 1
                if con > max_con:
                    max_con = con
            else:
                con = 0
        sustain = max_con + 1
        # Forward scan
        for b in range(start, len(ring_sizes) - sustain):
            i1, i2 = b - off1, b - off2
            if i1 + sustain > len(d1) or i2 + sustain > len(d2):
                break
            if all(d1[i1 + k] > 0 and d2[i2 + k] > 0 for k in range(sustain)):
                return b
        return skip

    domain = set(soma_set)
    n_excluded = 0
    for ti, tip in enumerate(filtered_tips):
        pseed = _perp_seed(tip, domain)
        rngs = _bfs_from(pseed, domain)
        reached: set[int] = set()
        for r in rngs:
            reached.update(r)
        if seed not in reached:
            # Dead-end: tip is on an already-excluded branch
            domain -= reached
            n_excluded += len(reached)
            continue
        rsizes = [len(r) for r in rngs]
        cut = _find_cut(rsizes)
        to_rm: set[int] = set()
        for b in range(cut):
            to_rm.update(rngs[b])
        domain -= to_rm
        n_excluded += len(to_rm)

    soma_set = domain

    if verbose:
        print(
            f"[skeliner.pre] Soma: excluded {n_excluded:,} neurite verts "
            f"from {len(filtered_tips)} tips, {len(soma_set):,} soma verts remain"
        )

    # ── 4. Fit ellipsoid ──────────────────────────────────────────
    if len(soma_set) < 4:
        if verbose:
            print("[skeliner.pre] Soma: too few verts")
        return None

    soma_arr = np.fromiter(sorted(soma_set), dtype=np.intp)
    soma = Soma.fit(mesh.vertices[soma_arr], verts=soma_arr)

    if verbose:
        print(
            f"[skeliner.pre] Soma: center=["
            f"{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} surface verts"
        )

    return soma


@_deprecated("find_soma()")
@mesh_args()
def find_soma_via_geodesic(
    mesh: Mesh,
    *,
    organelles: Organelles | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Soma | None:
    """Experimental soma detection — geodesic proximity + mass boundary.

    .. deprecated::
        Use :func:`find_soma`.

    Work-in-progress.  Key differences from
    :func:`find_soma_via_neurite_exclusion`:

      0. Organelle clustering uses **geodesic** (surface) proximity via
         multi-source BFS instead of Euclidean NN.  This prevents linking
         organelles across membranes.
      1. BFS from center on non-organelle surface (same).
      2. Candidate set boundary determined by **neighborhood mass** of
         soma organelle clusters (Otsu on own_size + neighbor_sizes),
         converted to a BFS ring boundary.  No fixed multiplier.
      3. Per-tip neurite exclusion (same).
      4. Fit ellipsoid (same).
    """
    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # ── 0. Organelle clustering via geodesic proximity ───────────
    if organelles is None:
        organelles = find_organelles(mesh, verbose=verbose)
    if organelles.mask.sum() == 0:
        if verbose:
            print("[skeliner.pre] Soma-alt: no organelles found")
        return None

    org_fi = np.where(organelles.mask)[0]
    org_labels, _ = _face_edge_components(Mesh(mesh.vertices, mesh.faces[org_fi]))
    clusters: list[dict] = []
    for cid in np.unique(org_labels):
        local_fi = org_fi[org_labels == cid]
        verts = set(int(v) for v in np.unique(mesh.faces[local_fi]))
        clusters.append(
            {
                "verts": verts,
                "centroid": mesh.vertices[list(verts)].mean(axis=0),
                "size": int(len(local_fi)),
            }
        )

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: {organelles.mask.sum():,} organelle faces, "
            f"{len(clusters)} clusters"
        )
    if len(clusters) < 3:
        if verbose:
            print("[skeliner.pre] Soma-alt: too few organelle clusters")
        return None

    # Build non-organelle surface adjacency
    main_fi = np.where(labels == main)[0]
    non_org_main = main_fi[~organelles.mask[main_fi]]

    adj_bfs: dict[int, list[int]] = defaultdict(list)
    faces_arr = np.asarray(mesh.faces)
    for fi in non_org_main:
        v = faces_arr[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj_bfs[a].append(b)
            adj_bfs[b].append(a)

    # Border verts: non-organelle surface verts touching each cluster
    border_verts: dict[int, set[int]] = {}
    for ci, cl in enumerate(clusters):
        borders: set[int] = set()
        for v in cl["verts"]:
            if v in adj_bfs:
                borders.add(v)
            for nv in adj_bfs.get(v, []):
                if nv not in cl["verts"]:
                    borders.add(nv)
        border_verts[ci] = borders

    # Multi-source BFS: label every non-organelle vert by nearest cluster
    vert_label: dict[int, int] = {}
    vert_dist: dict[int, int] = {}
    queue: deque[int] = deque()
    for ci, borders in border_verts.items():
        for v in borders:
            if v not in vert_dist:
                vert_dist[v] = 0
                vert_label[v] = ci
                queue.append(v)
    while queue:
        v = queue.popleft()
        d = vert_dist[v]
        for nv in adj_bfs[v]:
            if nv not in vert_dist:
                vert_dist[nv] = d + 1
                vert_label[nv] = vert_label[v]
                queue.append(nv)

    # Voronoi neighbors: clusters whose territories are adjacent
    neighbors: dict[int, set[int]] = defaultdict(set)
    for v in vert_dist:
        lv = vert_label[v]
        for nv in adj_bfs[v]:
            if nv in vert_label:
                lnv = vert_label[nv]
                if lv != lnv:
                    neighbors[lv].add(lnv)
                    neighbors[lnv].add(lv)

    # Neighborhood mass: own size + sum of neighbor sizes
    sizes = np.array([cl["size"] for cl in clusters])
    nbr_mass = np.zeros(len(clusters))
    for ci in range(len(clusters)):
        nbr_mass[ci] = sizes[ci] + sum(int(sizes[nj]) for nj in neighbors[ci])

    has_nbr = np.array([len(neighbors[ci]) > 0 for ci in range(len(clusters))])
    if has_nbr.sum() < 3:
        if verbose:
            print("[skeliner.pre] Soma-alt: too few connected clusters")
        return None

    mass_thresh, _ = _otsu_threshold(nbr_mass[has_nbr])
    soma_clusters = has_nbr & (nbr_mass > mass_thresh)

    # Soma center = median of soma cluster centroids
    centroids = np.array([cl["centroid"] for cl in clusters])
    core = centroids[soma_clusters]
    if len(core) < 3:
        if verbose:
            print("[skeliner.pre] Soma-alt: no dense cluster found")
        return None

    center = np.median(core, axis=0)
    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: {soma_clusters.sum()}/{len(clusters)} "
            f"soma clusters (mass > {mass_thresh:.0f})"
        )

    # ── 1. BFS from center ───────────────────────────────────────
    bfs_verts = np.fromiter(adj_bfs.keys(), dtype=np.intp)
    seed = int(
        bfs_verts[np.argmin(np.linalg.norm(mesh.vertices[bfs_verts] - center, axis=1))]
    )

    ring_level: dict[int, int] = {seed: 0}
    bfs_q: deque[int] = deque([seed])
    ring_verts: dict[int, list[int]] = defaultdict(list)
    ring_verts[0].append(seed)
    while bfs_q:
        v = bfs_q.popleft()
        lv = ring_level[v]
        for nv in adj_bfs[v]:
            if nv not in ring_level:
                ring_level[nv] = lv + 1
                bfs_q.append(nv)
                ring_verts[lv + 1].append(nv)

    max_ring = max(ring_verts.keys())

    # Peak ring (largest connected component width)
    largest_comp_size = np.zeros(max_ring + 1)
    for lv in range(max_ring + 1):
        vr = ring_verts.get(lv, [])
        if not vr:
            continue
        ring_set = set(vr)
        vis_r: set[int] = set()
        mx = 0
        for st in vr:
            if st in vis_r:
                continue
            sz = 0
            rq = deque([st])
            while rq:
                u = rq.popleft()
                if u in vis_r:
                    continue
                vis_r.add(u)
                sz += 1
                for nu in adj_bfs[u]:
                    if nu in ring_set and nu not in vis_r:
                        rq.append(nu)
            if sz > mx:
                mx = sz
        largest_comp_size[lv] = mx

    peak_ring = int(np.argmax(largest_comp_size))

    # ── 2. Boundary from soma cluster extent ─────────────────────
    # Map soma cluster Voronoi territory to BFS ring levels.
    # Boundary = max ring reached by any soma cluster territory.
    soma_territory = {v for v, ci in vert_label.items() if soma_clusters[ci]}
    soma_ring_vals = [ring_level[v] for v in soma_territory if v in ring_level]
    boundary = max(soma_ring_vals) if soma_ring_vals else peak_ring

    soma_set: set[int] = set()
    for lv in range(min(boundary + 1, max_ring + 1)):
        soma_set.update(ring_verts[lv])

    # Outer edge → tip clusters
    outer_edge: set[int] = set()
    for v in soma_set:
        for nv in adj_bfs[v]:
            if nv not in soma_set:
                outer_edge.add(v)
                break

    dilated = set(outer_edge)
    for v in outer_edge:
        for nv in adj_bfs[v]:
            if nv in soma_set:
                dilated.add(nv)

    visited_tip: set[int] = set()
    tip_clusters: list[list[int]] = []
    for start in outer_edge:
        if start in visited_tip:
            continue
        cluster: list[int] = []
        tq = deque([start])
        while tq:
            v = tq.popleft()
            if v in visited_tip:
                continue
            visited_tip.add(v)
            if v in outer_edge:
                cluster.append(v)
            for nv in adj_bfs[v]:
                if nv in dilated and nv not in visited_tip:
                    tq.append(nv)
        if cluster:
            tip_clusters.append(cluster)
    tip_clusters.sort(key=len, reverse=True)

    # Filter false tips by equator distance
    equator_dists = [
        float(np.linalg.norm(mesh.vertices[v] - center))
        for v in ring_verts.get(peak_ring, [])
    ]
    equator_dist = float(np.median(equator_dists)) if equator_dists else 0.0

    filtered_tips: list[list[int]] = []
    for cluster in tip_clusters:
        med_dist = float(
            np.median([np.linalg.norm(mesh.vertices[v] - center) for v in cluster])
        )
        if med_dist >= equator_dist:
            filtered_tips.append(cluster)

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: peak ring {peak_ring}, "
            f"boundary ring {boundary}, "
            f"{len(soma_set):,} verts, "
            f"{len(tip_clusters)} tip clusters → "
            f"{len(filtered_tips)} after equator filter"
        )

    # ── 3. Per-tip neurite stub removal ──────────────────────────
    _PERP_REF = 10
    _SMOOTH_W = 5

    def _perp_seed(tip_verts, domain):
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in tip_verts if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        ref = _PERP_REF
        if len(rngs) < ref + 3:
            return tip_verts
        ctrs = np.array(
            [mesh.vertices[[v for v in r]].mean(axis=0) for r in rngs[: ref + 5]]
        )
        lo = max(0, ref - 2)
        hi = min(len(ctrs) - 1, ref + 2)
        axis = ctrs[hi] - ctrs[lo]
        nrm = np.linalg.norm(axis)
        if nrm == 0:
            return tip_verts
        axis /= nrm
        ctr = ctrs[ref]
        nbr: set[int] = set()
        for b in range(max(0, ref - 3), min(len(rngs), ref + 4)):
            nbr.update(rngs[b])
        el = float(
            np.median(
                [
                    np.linalg.norm(mesh.vertices[v] - mesh.vertices[nv])
                    for v in list(nbr)[:100]
                    for nv in adj_bfs[v]
                    if nv in nbr
                ]
            )
        )
        half = el * 1.5
        return [
            v for v in nbr if abs(float(np.dot(mesh.vertices[v] - ctr, axis))) < half
        ]

    def _bfs_from(seeds, domain):
        vis: set[int] = set()
        rngs: list[list[int]] = []
        cur = [v for v in seeds if v in domain]
        for v in cur:
            vis.add(v)
        if cur:
            rngs.append(cur)
        while cur:
            nx = []
            for v in cur:
                for nv in adj_bfs[v]:
                    if nv in domain and nv not in vis:
                        vis.add(nv)
                        nx.append(nv)
            if not nx:
                break
            rngs.append(nx)
            cur = nx
        return rngs

    def _find_cut(ring_sizes):
        skip = _PERP_REF
        w = _SMOOTH_W
        if len(ring_sizes) < skip + 15:
            return skip
        smooth = np.convolve(ring_sizes, np.ones(w) / w, mode="valid")
        d1 = np.diff(smooth)
        d2 = np.diff(d1)
        off1 = w // 2 + 1
        off2 = w // 2 + 2
        start = max(skip, off2)
        tube_end = min(start + 20, len(ring_sizes))
        max_con = 0
        con = 0
        for b in range(start, tube_end):
            i1, i2 = b - off1, b - off2
            if 0 <= i1 < len(d1) and 0 <= i2 < len(d2) and d1[i1] > 0 and d2[i2] > 0:
                con += 1
                if con > max_con:
                    max_con = con
            else:
                con = 0
        sustain = max_con + 1
        for b in range(start, len(ring_sizes) - sustain):
            i1, i2 = b - off1, b - off2
            if i1 + sustain > len(d1) or i2 + sustain > len(d2):
                break
            if all(d1[i1 + k] > 0 and d2[i2 + k] > 0 for k in range(sustain)):
                return b
        return skip

    domain = set(soma_set)
    n_excluded = 0
    for ti, tip in enumerate(filtered_tips):
        pseed = _perp_seed(tip, domain)
        rngs = _bfs_from(pseed, domain)
        reached: set[int] = set()
        for r in rngs:
            reached.update(r)
        if seed not in reached:
            domain -= reached
            n_excluded += len(reached)
            continue
        rsizes = [len(r) for r in rngs]
        cut = _find_cut(rsizes)
        to_rm: set[int] = set()
        for b in range(cut):
            to_rm.update(rngs[b])
        domain -= to_rm
        n_excluded += len(to_rm)

    soma_set = domain

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: excluded {n_excluded:,} neurite verts "
            f"from {len(filtered_tips)} tips, {len(soma_set):,} soma verts remain"
        )

    # ── 4. Fit ellipsoid ─────────────────────────────────────────
    if len(soma_set) < 4:
        if verbose:
            print("[skeliner.pre] Soma-alt: too few verts")
        return None

    soma_arr = np.fromiter(sorted(soma_set), dtype=np.intp)
    soma = Soma.fit(mesh.vertices[soma_arr], verts=soma_arr)

    if verbose:
        print(
            f"[skeliner.pre] Soma-alt: center=["
            f"{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}], "
            f"{len(soma.verts):,} surface verts"
        )

    return soma


def _refine_components_with_fusions(
    mesh: Mesh,
    labels: np.ndarray,
    main: int,
    fusions: list[list[int]],
) -> tuple[np.ndarray, int]:
    """Re-label every component, treating fusion-cluster faces as walls.

    Two branches glued together by non-manifold fusion edges merge into
    one component in :func:`_face_edge_components` (BFS walks straight
    through the shared edge).  Treating the fusion-cluster faces as
    breakpoints and re-running BFS splits each merged blob into separate
    logical branches.

    This runs over *all* components, not just the main one.  A
    disconnected component can carry a fusion internally; if it is left
    whole, :func:`find_gaps` bridges it as a single unit and
    :func:`remove_fusions` then splits it — orphaning whichever half did
    not receive the bridge (together with anything chained behind it).

    Fusion-cluster faces are assigned the sentinel label ``-3`` so they
    are excluded from any reported component (and won't be picked up by
    downstream rim/gap detection).

    Parameters
    ----------
    mesh : Mesh
    labels : np.ndarray
        Output of :func:`_face_edge_components`.
    main : int
        Main component label.
    fusions : list[list[int]]
        Fusion clusters from :func:`find_fusions`.  Each cluster is a
        list of face indices.

    Returns
    -------
    new_labels : np.ndarray
        Refined labels.  Fusion faces are -3.
    new_main : int
        Largest sub-component after refinement, or unchanged ``main`` if
        nothing was relabelled.
    """
    labels = np.asarray(labels)
    fusion = np.fromiter((int(f) for c in fusions for f in c), dtype=np.int64)
    new_labels, new_main = kernels.pre.refine_components_with_fusions(
        np.asarray(mesh.faces, dtype=np.int64),
        labels.astype(np.int64, copy=False),
        int(main),
        fusion,
    )
    return new_labels.astype(labels.dtype, copy=False), new_main


@mesh_args()
def find_disconnected(
    mesh: Mesh,
    *,
    verbose: bool = False,
    soma: Soma | None = None,
    organelles: Organelles | None = None,
    mesh_stats: MeshStats | None = None,
    fusions: list[list[int]] | None = None,
) -> list[list[int]]:
    """Detect disconnected mesh components from segmentation errors.

    Returns disconnected components — broken neurite segments that are
    separate from the main mesh: the component the soma lies on, or the
    largest when there is no soma.  Soma-region components, organelle
    components, and components enclosed by the main mesh are excluded.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    verbose : bool, default False
        Print summary.
    soma : Soma or None
        Pre-computed soma from any ``find_soma_via_*`` function.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.
        Components that are entirely organelle are skipped.
    mesh_stats : MeshStats or None
        From :func:`compute_mesh_stats`.  Reuses ``face_comp`` and
        ``main_ci`` to skip redundant component detection.
    fusions : list[list[int]] or None
        Fusion clusters from :func:`find_fusions`.  When provided, the
        main component is re-split treating each fusion cluster as a
        wall, so two branches that are glued by non-manifold fusion
        edges surface as separate components.  This function does NOT
        run :func:`find_fusions` itself — pass the result explicitly.

    Returns
    -------
    list[list[int]]
        Each element is a list of face indices for one disconnected
        component, sorted largest-first.
    """
    if verbose:
        print("[skeliner.pre] Finding disconnected components …")
        _t0 = time.perf_counter()

    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # Optional fusion-aware refinement: split the main blob along
    # fusion clusters so glued branches become separate components.
    if fusions:
        labels, main = _refine_components_with_fusions(mesh, labels, main, fusions)
    # Locate soma so we can exclude components inside it
    if soma is None:
        soma = find_soma(
            mesh,
            organelles=organelles,
            mesh_stats=mesh_stats,
        )
    main = _main_component(mesh, labels, main, soma)

    found = kernels.pre.find_disconnected(
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles_center, dtype=np.float64),
        np.asarray(mesh.face_normals, dtype=np.float64),
        _median_edge(mesh),
        np.asarray(labels, dtype=np.int64),
        int(main),
        None if organelles is None else np.asarray(organelles.mask, dtype=bool),
        None if soma is None else np.asarray(soma.center, dtype=np.float64),
        None if soma is None else np.asarray(soma.R / soma.axes, dtype=np.float64),
    )
    (
        components,
        n_total_non_main,
        n_soma_excluded,
        n_organelle_excluded,
        n_enclosed_excluded,
    ) = found

    if verbose:
        dt = time.perf_counter() - _t0
        total = sum(len(fis) for fis in components)
        excluded = []
        if n_soma_excluded:
            excluded.append(f"{n_soma_excluded} soma")
        if n_organelle_excluded:
            excluded.append(f"{n_organelle_excluded} organelle")
        if n_enclosed_excluded:
            excluded.append(f"{n_enclosed_excluded} enclosed")
        exc_str = f"; excluded {'+'.join(excluded)}" if excluded else ""
        print(
            f"[skeliner.pre] Disconnected: "
            f"{len(components)} of {n_total_non_main} "
            f"non-main components, {total:,} faces"
            f"{exc_str} ({dt:.1f}s)"
        )

    return components


@mesh_args()
def find_gaps(
    mesh: Mesh,
    *,
    tip_rings: int = 3,
    verbose: bool = False,
    soma: Soma | None = None,
    disconnected: list[list[int]] | None = None,
    organelles: Organelles | None = None,
    mesh_stats: MeshStats | None = None,
    fusions: list[list[int]] | None = None,
    fusion_clearance: float | None = None,
    max_bridge_ratio: float = 1.0,
    min_bridge_gap: float | None = None,
    max_bridge_dist: float | None = None,
    kiss_radius: float = 6.0,
    kiss_penalty: float = 10.0,
    relay_min_faces: int = 100,
    repair_stranded: bool = True,
) -> list[tuple[list[int], list[int], float]]:
    """Detect gaps between disconnected components and the main mesh.

    Each gap is a pair of face groups — the tip faces on each side of
    the break — ready for bridging.  The main mesh is the component the
    soma lies on, or the largest when there is no soma.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    tip_rings : int, default 3
        Number of face-graph BFS rings from the closest face to
        collect as tip faces on each side of a gap.
    verbose : bool, default False
        Print summary.
    soma : Soma or None
        Pre-computed soma from any ``find_soma_via_*`` function; found
        when neither it nor *disconnected* is given.
    disconnected : list[list[int]] or None
        Pre-computed disconnected components from :func:`find_disconnected`.
    fusions : list[list[int]] or None
        Fusion clusters from :func:`find_fusions`.  When provided,
        vertices touched by any fusion-cluster face are excluded from
        each component's KDTree so the gap query finds the real
        geometric gap rather than the zero-distance fusion overlap.
        Pass-through to :func:`find_disconnected` when ``disconnected``
        is ``None``.
    fusion_clearance : float or None
        Only used when ``fusions`` is provided.  Candidate gap face
        pairs whose endpoints lie within this distance of any
        fusion-cluster face are dropped — they represent "kissing
        points" near fusion zones (artifacts of the wrong fusion),
        not real gaps to bridge.  Defaults to ``10 * median_edge``.
    max_bridge_ratio : float, default 1.0
        Size-aware cap.  A gap is only bridged if its distance is at
        most ``max_bridge_ratio`` times the bounding-box diagonal of
        the *smaller* of the two pieces.  A real neurite break leaves
        the two ends close relative to the piece's own extent; a
        fragment sitting far compared to its own size is debris, not a
        break, and is left disconnected (later discarded, recoverable
        via the viewer lasso).  Set to ``float("inf")`` to disable.
    min_bridge_gap : float or None
        Gaps at or below this distance are always bridged regardless of
        the size ratio — small breaks on small fragments are real.
        Defaults to ``8 * median_edge``.
    max_bridge_dist : float or None
        Absolute ceiling: no gap longer than this is ever bridged, even
        for a large piece with a low size ratio (a straight tube across
        that much empty space fabricates geometry).  Defaults to
        ``75 * median_edge``.
    kiss_radius : float, default 6.0
        Radius (in median edges) of the local neighbourhood used to score
        whether a candidate contact sits at a piece's end/cut (a real
        break) or on its flank (a "kiss").
    kiss_penalty : float, default 10.0
        Weight of the kiss penalty on the MST edge cost: a candidate is
        scored ``distance * (1 + kiss_penalty * flank)``, where ``flank``
        is the smaller of the two sides' local two-sidedness (~0 at a
        tip/cut, ~1 at a side-by-side crossing).  This makes a real break
        outrank a closer kiss so a piece bridges at its tip, not where its
        wall grazes another surface.  ``0`` disables it.
    relay_min_faces : int, default 100
        Smallest component allowed to carry a bridge chain onward.
        :func:`remove_gaps` peels a tip patch off each end of every
        bridge, so a fragment used as an intermediate hop loses two
        patches and is erased — breaking the chain routed through it.
        Components below this size may still be bridged, but only as
        chain endpoints.  ``0`` disables the restriction.
    repair_stranded : bool, default True
        Retry a component whose MST tree edge ``_add_gap`` refused.
        Prim's minimises the kiss-penalised cost without knowing which
        edges are acceptable, and a refused edge is otherwise never
        retried, so the component — and everything hanging off it —
        loses its only path back to main.  The retry takes the cheapest
        edge that *is* acceptable, and leaves a cluster with no
        acceptable edge disconnected.  Set ``False`` to keep only the
        bridges Prim's itself chose: fewer pieces are reconnected, and
        pieces that should stay separate are less likely to be joined.

    Notes
    -----
    The closest clear pair between two components can be a "kiss" — where
    a branch's side wall grazes another surface — rather than a real break
    where the two pieces END at the contact (tip-to-tip or a planar cut).
    Each candidate's MST cost is multiplied by ``1 + kiss_penalty * flank``,
    where ``flank`` (0…1) measures whether the pieces CONTINUE past the
    contact (a kiss) or end there (a break), in a local ``kiss_radius``
    neighbourhood so a piece's overall length/branchiness can't fool it.
    A kiss is thus penalised and a real break outranks a closer kiss, so
    the piece bridges at its tip instead of where its wall grazes.

    Returns
    -------
    list[tuple[list[int], list[int], float, int, int]]
        Each element is ``(faces_a, faces_b, gap_distance, disc_a, disc_b)``
        where *faces_a* and *faces_b* are face-index lists on each side of
        the gap, *disc_a* and *disc_b* are disconnected component indices
        (``-1`` for main), sorted by gap distance (smallest first).
    """
    if verbose:
        print("[skeliner.pre] Finding gaps …")
        _t0 = time.perf_counter()

    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels, main = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        labels, main = _face_edge_components(mesh)

    # When fusions are provided, refine the components the same way
    # find_disconnected does — otherwise a label still spans both glued
    # branches and we'd be querying nearest neighbors within a single
    # merged blob.  Keep the pre-refinement labels: two components that
    # share one are *siblings* split apart by a fusion wall, and only
    # those may be re-joined across the fusion (see `fusion_join`).
    labels_pre_fusion = labels
    if fusions:
        labels, main = _refine_components_with_fusions(mesh, labels, main, fusions)
    # found here when find_disconnected would find it, so both name the
    # same main component
    if soma is None and disconnected is None:
        soma = find_soma(mesh, organelles=organelles, mesh_stats=mesh_stats)
    main = _main_component(mesh, labels, main, soma)

    # Get disconnected components (reuse filtering logic)
    if disconnected is not None:
        disc = disconnected
    else:
        disc = find_disconnected(
            mesh,
            soma=soma,
            organelles=organelles,
            mesh_stats=mesh_stats,
            fusions=fusions,
        )

    if not disc:
        if verbose:
            dt = time.perf_counter() - _t0
            print(
                f"[skeliner.pre] Gaps: 0 gaps, no disconnected components ({dt:.1f}s)"
            )
        return []

    median_edge = _median_edge(mesh)
    fusion_faces = list(dict.fromkeys(int(f) for c in fusions or () for f in c))
    gaps, n_capped, n_repaired = kernels.pre.find_gaps(
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles_center, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(labels_pre_fusion, dtype=np.int64),
        int(main),
        [[int(f) for f in d] for d in disc],
        np.asarray(fusion_faces, dtype=np.int64),
        tip_rings,
        float(fusion_clearance) if fusion_clearance is not None else 10.0 * median_edge,
        6.0 * median_edge,
        float(max_bridge_ratio),
        float(min_bridge_gap) if min_bridge_gap is not None else 8.0 * median_edge,
        float(max_bridge_dist) if max_bridge_dist is not None else 75.0 * median_edge,
        kiss_radius * median_edge,
        float(kiss_penalty),
        int(relay_min_faces),
        bool(repair_stranded),
        _svd_rows,
    )
    if verbose:
        _report_gaps(gaps, disc, n_capped, n_repaired, _t0)
    return gaps


def _report_gaps(gaps, disc, n_capped: int, n_repaired: int, t0: float) -> None:
    dt = time.perf_counter() - t0
    if gaps:
        dists = [g[2] for g in gaps]
        dist_str = (
            f", dist {min(dists):.0f}–{max(dists):.0f}"
            if len(dists) > 1
            else f", dist {dists[0]:.0f}"
        )
    else:
        dist_str = ""
    cap_str = f", {n_capped} too-far dropped" if n_capped else ""
    rep_str = f", {n_repaired} stranded re-bridged" if n_repaired else ""
    print(
        f"[skeliner.pre] Gaps: {len(gaps)} gaps "
        f"across {len(disc)} disconnected components"
        f"{dist_str}{cap_str}{rep_str} ({dt:.1f}s)"
    )


@mesh_args()
def remove_gaps(
    mesh: Mesh,
    *,
    verbose: bool = False,
    soma: Soma | None = None,
    gaps: list | None = None,
    organelles: Organelles | None = None,
    mesh_stats: MeshStats | None = None,
) -> Mesh:
    """Bridge all detected gaps in a single mesh rebuild.

    All gap tip faces are removed at once and boundary loops from each
    gap are paired and zipper-stitched, then the mesh is rebuilt once.
    Vertex indices are preserved.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    verbose : bool, default False
        Print progress.
    soma : Soma or None
        Pre-computed soma.
    gaps : list or None
        Pre-computed gaps from :func:`find_gaps`.  Tuning knobs live on
        that function; call it yourself and pass the result here.

    Returns
    -------
    Mesh
        Mesh with gaps bridged.
    """
    if verbose:
        print("[skeliner.pre] Removing gaps …")
        _t0 = time.perf_counter()

    if gaps is None:
        gaps = find_gaps(
            mesh,
            verbose=verbose,
            soma=soma,
            organelles=organelles,
            mesh_stats=mesh_stats,
        )

    if not gaps:
        if verbose:
            dt = time.perf_counter() - _t0
            print(f"[skeliner.pre] remove_gaps: no gaps to bridge ({dt:.1f}s)")
        return mesh

    labels = None
    if mesh_stats is not None and mesh_stats.face_comp is not None:
        labels = mesh_stats.face_comp
    result, n_bridged, n_removed, n_skipped = _bridge_gaps(mesh, gaps, labels)
    if result is None:
        if verbose:
            dt = time.perf_counter() - _t0
            print(f"[skeliner.pre] remove_gaps: no valid loop pairs ({dt:.1f}s)")
        return mesh

    # Invalidate topology (components merged); pad outward_dots for appended faces
    n_added = len(result.faces) - len(mesh.faces)
    if mesh_stats is not None:
        if n_added > 0 and mesh_stats.outward_dots is not None:
            mesh_stats.outward_dots = np.concatenate(
                [
                    mesh_stats.outward_dots,
                    np.ones(
                        n_added,
                        dtype=mesh_stats.outward_dots.dtype,
                    ),
                ]
            )
        mesh_stats.invalidate_topology()

    if verbose:
        dt = time.perf_counter() - _t0
        parts = [f"bridged {n_bridged}/{len(gaps)} gaps"]
        parts.append(f"removed {n_removed:,}f")
        if n_added > 0:
            parts.append(f"stitched +{n_added:,}f")
        if n_skipped:
            parts.append(f"{n_skipped} skipped")
        print(f"[skeliner.pre] remove_gaps: {', '.join(parts)} ({dt:.1f}s)")

    return result


def _bridge_gaps(mesh, gaps, labels):
    """Peel each gap's tips to clean rims, skip pairs that cannot be stitched
    or would strand more than they win back, and stitch the rest in one
    rebuild: ``(mesh or None, n_bridged, n_removed, n_skipped)``.

    *labels* are the raw face-edge components rescue sizes are read from
    (computed on first need when None).  Each loop is fitted by
    :func:`_fit_loop_circle`, called back from the Rust core.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    def fit(ids):
        center, radius, normal = _fit_loop_circle(verts[ids])
        return center, radius, normal

    new_verts, faces, n_bridged, n_removed, n_skipped = kernels.pre.remove_gaps(
        np.asarray(mesh.faces, dtype=np.int64),
        verts,
        np.asarray(mesh.face_normals, dtype=np.float64),
        _median_edge(mesh),
        None if labels is None else np.asarray(labels, dtype=np.int64),
        [
            (list(map(int, faces_a)), list(map(int, faces_b)))
            for faces_a, faces_b, *_ in gaps
        ],
        fit,
    )
    if faces is None:
        return None, 0, 0, n_skipped
    if len(new_verts):
        verts = np.vstack([verts, new_verts])
    result = Mesh(verts, faces)
    return result, n_bridged, n_removed, n_skipped


@_deprecated()
@mesh_args()
def remove_islands(
    mesh: Mesh,
    *,
    min_faces: int = 3,
    verbose: bool = False,
) -> Mesh:
    """Remove small edge-disconnected components (islands).

    .. deprecated::
        Neither the pipeline nor the viewer uses it; it will be removed.

    An island is a cluster of faces that shares no edge with the main
    surface — it may touch via a vertex, but has no edge connectivity.
    This performs a single pass: all islands present at call time are
    removed.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    min_faces : int, default 3
        Edge-connected components with fewer faces than this are removed.
    verbose : bool, default False
        Print summary.

    Returns
    -------
    Mesh
        Mesh with islands removed.
    """
    active = np.ones(len(mesh.faces), dtype=bool)
    islands = _find_island_faces(mesh.faces, active, min_faces)
    n_removed = int(islands.sum())

    if n_removed == 0:
        if verbose:
            print("[skeliner.pre] No islands to remove")
        return mesh

    clean = _rebuild_mesh(mesh, ~islands)

    if verbose:
        print(f"[skeliner.pre] Removed {n_removed} island faces")
    return clean


@_deprecated()
@mesh_args()
def remove_fins(
    mesh: Mesh,
    *,
    verbose: bool = False,
) -> Mesh:
    """Remove fin faces — faces with 2+ boundary edges.

    .. deprecated::
        Neither the pipeline nor the viewer uses it; it will be removed.

    A fin is a face hanging off the surface by a single edge (2 of its
    3 edges are boundary edges).  This iterates until no more fins
    remain, since removing a fin can expose new fins.
    """
    active = np.ones(len(mesh.faces), dtype=bool)
    fins = _find_fin_faces(mesh.faces, active)
    n_removed = int(fins.sum())

    if n_removed == 0:
        if verbose:
            print("[skeliner.pre] No fins to remove")
        return mesh

    result = _rebuild_mesh(mesh, ~fins)

    if verbose:
        print(f"[skeliner.pre] Removed {n_removed} fin faces")

    return result


@_deprecated()
@mesh_args()
def remove_fragments(
    mesh: Mesh,
    *,
    min_faces: int = 3,
    verbose: bool = False,
    fragments: np.ndarray | None = None,
) -> Mesh:
    """Remove all fragments (islands and fins) by alternating until convergence.

    .. deprecated::
        Neither the pipeline nor the viewer uses it; it will be removed.

    Islands are small edge-disconnected components; fins are faces
    hanging by a single edge.  Removing one type can create the other,
    so this alternates island and fin detection on a mask until stable,
    then rebuilds the mesh once.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    min_faces : int, default 3
        Passed to island detection.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    Mesh
        Mesh with all fragments removed.
    """
    if fragments is not None:
        fragment = fragments
        if verbose:
            print(
                f"[skeliner.pre] Using provided fragment mask ({int(fragment.sum()):,} faces)"
            )
    else:
        fragment = _find_fragments(mesh, min_faces, verbose)

    n_removed = int(fragment.sum())
    if n_removed == 0:
        return mesh

    result = _rebuild_mesh(mesh, ~fragment)

    if verbose:
        print(f"[skeliner.pre] Removed {n_removed:,} fragment faces")

    return result


def _outward_dot(
    mesh: Mesh,
    radius: float,
    vert_comp: np.ndarray | None = None,
) -> np.ndarray:
    """Per-face outward score: dot(face_normal, direction_from_local_COM).

    For each face, finds all vertices within *radius* of its centroid
    **on the same component**, computes their center of mass, and dots
    the face normal against the direction from that COM to the face
    centroid.

    Surface faces point outward (positive), internal faces point
    inward (negative).

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    radius : float
        Ball-query radius for local centre-of-mass.
    vert_comp : np.ndarray or None
        Per-vertex component ID ``(nVertices,)``.  When provided, only
        vertices from the same component are used for the local COM.
        This gives correct outward-dot values for disconnected
        components.

    Returns
    -------
    np.ndarray
        (nFaces,) float64 array of outward dot products.
    """
    return kernels.pre.outward_dots(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.triangles_center, dtype=np.float64),
        np.asarray(mesh.face_normals, dtype=np.float64),
        float(radius),
        None if vert_comp is None else np.asarray(vert_comp, dtype=np.int64),
    )


def _filter_small_clusters(
    mesh: Mesh,
    face_mask: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    """Remove connected components of flagged faces below a size threshold.

    Parameters
    ----------
    mesh
        The mesh.
    face_mask
        Boolean mask (nFaces,) — True for faces to consider.
    min_cluster_size
        Clusters with fewer faces than this are dropped.

    Returns
    -------
    np.ndarray
        Filtered boolean mask.
    """
    if not np.any(face_mask):
        return face_mask
    return kernels.pre.filter_small_clusters(
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(face_mask, dtype=bool),
        int(min_cluster_size),
    )


@mesh_args()
def compute_mesh_stats(
    mesh: Mesh,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
):
    """Shared precomputation for organelle detection.

    Returns a :class:`~skeliner.dataclass.MeshStats`.
    """
    if verbose:
        print("[skeliner.pre] Computing mesh stats …")
        _t0 = time.perf_counter()

    if radius is None:
        median_edge = _median_edge(mesh)
        radius = radius_multiplier * median_edge

    outward_dots, face_comp, main_ci = kernels.pre.mesh_stats(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.triangles_center, dtype=np.float64),
        np.asarray(mesh.face_normals, dtype=np.float64),
        float(radius),
    )
    if verbose:
        _report_mesh_stats(face_comp, outward_dots, radius, len(mesh.faces), _t0)
    return MeshStats(
        outward_dots=outward_dots, face_comp=face_comp, main_ci=int(main_ci)
    )


def _report_mesh_stats(face_comp, outward_dots, radius, n_faces, t0) -> None:
    dt = time.perf_counter() - t0
    n_comps = int(face_comp.max()) + 1 if len(face_comp) else 0
    raw_neg = int((outward_dots < 0).sum())
    pct = 100 * raw_neg / n_faces
    print(
        f"[skeliner.pre] Mesh stats: "
        f"{n_comps} components, radius={radius:.0f}, "
        f"raw internal={raw_neg:,} ({pct:.1f}%) "
        f"({dt:.1f}s)"
    )


@_deprecated()
@mesh_args()
def find_pocket_mouths(
    mesh: Mesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
    _non_main_mask: np.ndarray | None = None,
) -> list[list[tuple[int, int]]]:
    """Find mouth edges — boundaries of negative-dot face clusters.

    .. deprecated::
        Neither the pipeline nor the viewer uses it; it will be removed.

    A valid pocket must satisfy:

    1. Multiple boundary loops (not a flat patch with a single outline).
    2. Fold ratio > *min_fold_ratio*: the pocket surface area must be
       much larger than the mouth's enclosed planar area, indicating the
       surface folds inward through a small opening.

    Returns
    -------
    list[list[tuple[int, int]]]
        One list of edges per pocket mouth.
    """

    if mesh_stats is not None and mesh_stats.outward_dots is not None:
        outward_dots = mesh_stats.outward_dots
        if mesh_stats.face_comp is not None:
            main_face_mask = mesh_stats.main_face_mask
        else:
            _fc, _mc = _face_edge_components(mesh)
            main_face_mask = _fc == _mc
    else:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
        outward_dots = mesh_stats.outward_dots
        main_face_mask = mesh_stats.main_face_mask
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return kernels.pre.pocket_mouths(
        faces,
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.area_faces, dtype=np.float64),
        np.asarray(outward_dots, dtype=np.float64),
        np.asarray(main_face_mask, dtype=bool),
        kernels.mesh.Topology(faces, len(mesh.vertices)),
        _svd_rows,
        min_pocket_size=min_pocket_size,
        min_fold_ratio=min_fold_ratio,
        non_main_mask=_non_main_mask,
    )


def _face_adjacency(
    mesh: Mesh,
    edge_to_face: dict[tuple[int, int], list[int]] | None = None,
) -> dict[int, set[int]]:
    """Build face adjacency map (edge-connected neighbors)."""
    if edge_to_face is None:
        edge_to_face = _edge_to_faces(mesh)
    return _py_mesh.face_adjacency(edge_to_face)


@mesh_args()
def find_pocket_organelles(
    mesh: Mesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_threshold: float = 0.1,
    bridge_threshold: float = 0.3,
    max_hole_size: int = 500,
    hole_enclosure_ratio: float = 0.5,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
    _non_main_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Detect pocket organelles — membrane folds connected to the neuron surface.

    Uses mouths (boundaries of negative-dot clusters) as seeds:

    1. Call :func:`find_pocket_mouths` to get mouth edges for each pocket.
    2. Seed from the **negative-dot faces** of each mouth's pocket cluster.
    3. Flood-fill from seeds, stopping at mouth edges and faces with
       ``outward_dot > grow_threshold``.
    4. Bridging: flood-fill from pocket boundary using the relaxed
       ``bridge_threshold`` to cross narrow positive-dot barriers.
    5. Hole filling: small non-pocket clusters mostly enclosed by pocket
       faces are filled in, if they lie inside the cell — more than half
       their faces look into it, the ray along a face's normal meeting
       another face: a fold's inside faces into the cell, while a neurite's
       end ringed by a fold at its base faces out of it.
    6. Shallow dips dropped: a pocket with less than *min_fold_ratio*
       times the area of the faces ringing it, unless more than half its
       faces look into the cell — a small fold does, a dip in the outer
       membrane looks out.

    Only regions behind a mouth get detected — curved surfaces without a
    mouth are correctly excluded.

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh.
    radius : float or None
        Radius for outward_dot computation. Auto-computed if None.
    radius_multiplier : float
        Multiplier for auto radius.
    grow_threshold : float
        Flood-fill will not enter faces with ``outward_dot > grow_threshold``.
    bridge_threshold : float
        Relaxed threshold used in a second flood-fill from the pocket
        boundary to cross narrow positive-dot barriers within a pocket.
    max_hole_size : int
        Non-pocket clusters with at most this many faces that are
        mostly enclosed by pocket faces are filled in.
    hole_enclosure_ratio : float
        Minimum fraction of a hole cluster's boundary neighbors that
        must be pocket faces for the hole to be filled.
    min_pocket_size : int
        Minimum negative-face cluster size to produce a mouth / be detected.
    min_cluster_size : int
        Final pocket clusters smaller than this are discarded.
    verbose : bool

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — pocket organelle faces.
    """
    if verbose:
        print("[skeliner.pre] Finding pocket organelles …")
        _t0 = time.perf_counter()

    if mesh_stats is not None and mesh_stats.outward_dots is not None:
        outward_dots = mesh_stats.outward_dots
        if mesh_stats.face_comp is not None:
            main_face_mask = mesh_stats.main_face_mask
        else:
            _fc, _mc = _face_edge_components(mesh)
            main_face_mask = _fc == _mc
    else:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
        outward_dots = mesh_stats.outward_dots
        main_face_mask = mesh_stats.main_face_mask
    faces = np.asarray(mesh.faces, dtype=np.int64)
    pocket, counts = kernels.pre.pocket_organelles(
        faces,
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.area_faces, dtype=np.float64),
        np.asarray(outward_dots, dtype=np.float64),
        np.asarray(main_face_mask, dtype=bool),
        kernels.mesh.Topology(faces, len(mesh.vertices)),
        _svd_rows,
        centers=np.asarray(mesh.triangles_center, dtype=np.float64),
        normals=np.asarray(mesh.face_normals, dtype=np.float64),
        grow_threshold=grow_threshold,
        bridge_threshold=bridge_threshold,
        max_hole_size=max_hole_size,
        hole_enclosure_ratio=hole_enclosure_ratio,
        min_pocket_size=min_pocket_size,
        min_fold_ratio=min_fold_ratio,
        min_cluster_size=min_cluster_size,
        non_main_mask=_non_main_mask,
    )

    if verbose:
        dt = time.perf_counter() - _t0
        if not counts["n_mouths"]:
            print(f"[skeliner.pre] Pocket organelles: 0 faces (no mouths) ({dt:.1f}s)")
        else:
            print(
                f"[skeliner.pre] Pocket organelles: "
                f"{pocket.sum():,} faces from {counts['n_mouths']} mouths "
                f"(fill {counts['n_initial']:,}, "
                f"bridge +{counts['n_bridged']:,}, "
                f"holes +{counts['n_holes']:,}, "
                f"rejected {counts['n_rejected']}) ({dt:.1f}s)"
            )
    return pocket


@mesh_args()
def find_isolated_organelles(
    mesh: Mesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> np.ndarray:
    """Detect vertex-disconnected internal fragments.

    These are organelle membranes that form entirely separate connected
    components enclosed within the neuron body.

    Returns
    -------
    np.ndarray
        Boolean mask ``(nFaces,)`` — isolated organelle faces.
    """
    if verbose:
        print("[skeliner.pre] Finding isolated organelles …")
        _t0 = time.perf_counter()

    if mesh_stats is None or mesh_stats.outward_dots is None:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
        )
    outward_dots = mesh_stats.outward_dots
    if mesh_stats.face_comp is not None:
        face_comp = mesh_stats.face_comp
        main_ci = mesh_stats.main_ci
    else:
        face_comp, main_ci = _face_edge_components(mesh)
    isolated, counts = kernels.pre.isolated_organelles(
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles_center, dtype=np.float64),
        np.asarray(mesh.face_normals, dtype=np.float64),
        np.asarray(outward_dots, dtype=np.float64),
        np.asarray(face_comp, dtype=np.int64),
        int(main_ci),
    )

    if verbose:
        dt = time.perf_counter() - _t0
        print(
            f"[skeliner.pre] Isolated organelles: "
            f"{counts['n_internal']:,} fragments, "
            f"{counts['n_internal_faces']:,} faces "
            f"({counts['n_kept']:,} external kept; "
            f"{counts['n_via_dots']:,} by dots, "
            f"{counts['n_via_geometry']:,} by geometry) "
            f"({dt:.1f}s)"
        )
    return isolated


@mesh_args()
def find_organelles(
    mesh: Mesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
    return_mesh_stats: bool = False,
) -> Organelles | tuple[Organelles, MeshStats]:
    """Detect internal mesh fragments (organelle membranes) in a neuron mesh.

    Returns an :class:`~skeliner.dataclass.Organelles` with two
    non-overlapping masks:

    * **pocket** — membrane folds connected to the neuron surface,
      detected via gradient-based mouth finding + flood-fill, then grown
      on across every face that looks into the cell (its normal ray meets
      another face), which the outer membrane, looking out, stops.
    * **isolated** — entire vertex-disconnected components whose
      mean outward dot is negative (fully enclosed fragments).

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh.
    radius : float or None, default None
        Radius (in mesh units, typically nm) for the local center-of-mass
        neighbourhood.  If None, automatically set to
        ``radius_multiplier * median_edge_length``.
    radius_multiplier : float, default 5.0
        Multiplier for automatic radius computation.  Only used when
        *radius* is None.
    min_cluster_size : int, default 5
        Connected components of internal faces smaller than this are
        kept (assumed to be noise).
    verbose : bool, default False
        Print summary statistics.
    mesh_stats : MeshStats or None, default None
        Pre-computed :func:`compute_mesh_stats` output to reuse.  If
        None, fresh stats are computed internally.
    return_mesh_stats : bool, default False
        If True, also return the :class:`MeshStats` used during
        detection (freshly computed when *mesh_stats* was not provided).

    Returns
    -------
    Organelles
        Always returned.
    MeshStats
        Only returned when ``return_mesh_stats=True``.
    """
    if verbose:
        print("[skeliner.pre] Finding organelles …")
    t_total = time.perf_counter()

    # ── 1. Precompute outward dots and components ─────────────────
    if mesh_stats is None or mesh_stats.outward_dots is None:
        mesh_stats = compute_mesh_stats(
            mesh,
            radius,
            radius_multiplier,
            verbose,
        )
    if mesh_stats.face_comp is not None:
        face_comp, main_ci = mesh_stats.face_comp, mesh_stats.main_ci
    else:
        face_comp, main_ci = _face_edge_components(mesh)

    if verbose:
        print("[skeliner.pre] Finding isolated organelles and pockets …")
    pocket, isolated, rep = kernels.pre.find_organelles(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.triangles_center, dtype=np.float64),
        np.asarray(mesh.face_normals, dtype=np.float64),
        np.asarray(mesh.area_faces, dtype=np.float64),
        np.asarray(mesh_stats.outward_dots, dtype=np.float64),
        np.asarray(face_comp, dtype=np.int64),
        int(main_ci),
        np.asarray(mesh.face_adjacency, dtype=np.int64),
        _svd_rows,
        int(min_cluster_size),
    )
    n_structural, enclosed_count = rep["n_structural"], rep["n_enclosed"]
    if verbose:
        print(
            f"[skeliner.pre] Isolated organelles: "
            f"{rep['n_internal']:,} fragments, "
            f"{rep['n_internal_faces']:,} faces "
            f"({rep['n_kept']:,} external kept; "
            f"{rep['n_via_dots']:,} by dots, "
            f"{rep['n_via_geometry']:,} by geometry)"
        )
        n_pocket = int(pocket.sum()) - enclosed_count - rep["n_grown"]
        print(
            f"[skeliner.pre] Pocket organelles: "
            f"{n_pocket:,} faces from {rep['n_mouths']} mouths "
            f"(fill {rep['n_initial']:,}, "
            f"bridge +{rep['n_bridged']:,}, "
            f"holes +{rep['n_holes']:,}, "
            f"rejected {rep['n_rejected']})"
        )

    dt_total = time.perf_counter() - t_total
    if verbose:
        parts = [
            f"pocket={int(pocket.sum()):,}",
            f"isolated={int(isolated.sum()):,}",
        ]
        if n_structural > 1:
            parts.append(f"{n_structural} structural components")
        if enclosed_count:
            parts.append(f"enclosed +{enclosed_count:,}")
        if rep["n_grown"]:
            parts.append(f"grown +{rep['n_grown']:,}")
        print(f"[skeliner.pre] find_organelles: {', '.join(parts)} ({dt_total:.1f}s)")

    org = Organelles(
        pocket=pocket,
        isolated=isolated,
        expanded=np.zeros(len(mesh.faces), dtype=bool),
    )
    if return_mesh_stats:
        return org, mesh_stats
    return org


def _find_nonmanifold_fusions(
    mesh: Mesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_rings: int = 20,
    min_branch_size: int = 5,
    mesh_stats: MeshStats | None = None,
) -> list[list[int]]:
    """Detect non-manifold fusions (shared edges, duplicate faces, pinch vertices)."""
    return kernels.pre.find_nonmanifold_fusions(
        np.asarray(mesh.faces, dtype=np.int64),
        len(mesh.vertices),
        np.asarray(mesh.area_faces, dtype=np.float64),
        _outward_dots_for(mesh, radius, radius_multiplier, mesh_stats),
        grow_rings,
        min_branch_size,
    )


def _outward_dots_for(mesh, radius, radius_multiplier, mesh_stats) -> np.ndarray:
    """The outward dots in *mesh_stats*, or computed at *radius* (default
    ``radius_multiplier`` median edges)."""
    if mesh_stats is not None and mesh_stats.outward_dots is not None:
        dots = mesh_stats.outward_dots
    else:
        dots = _outward_dot(
            mesh,
            radius if radius is not None else radius_multiplier * _median_edge(mesh),
        )
    return np.asarray(dots, dtype=np.float64)


@mesh_args()
def find_fusions(
    mesh: Mesh,
    *,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    grow_rings: int = 20,
    min_branch_size: int = 5,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> list[list[int]]:
    """Detect non-manifold fusion points where two branches are wrongly connected.

    Detects non-manifold edges, duplicate faces, and pinch vertices.

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh (ideally after organelle removal).
    radius : float or None
        Radius for outward_dot computation. Auto-computed if None.
    radius_multiplier : float
        Multiplier for auto radius.
    grow_rings : int
        Maximum rings to grow around each seed cluster.
    min_branch_size : int
        Minimum faces in a component to count as a branch.
    verbose : bool
    mesh_stats : tuple or None
        Precomputed ``(outward_dots, face_comp, main_ci, main_face_mask)``
        from :func:`compute_mesh_stats`.  Reuses ``outward_dots`` to skip
        redundant computation.

    Returns
    -------
    list[list[int]]
        Each inner list is one fusion cluster (face indices at the
        fusion boundary).

    Notes
    -----
    This detects **non-manifold fusions** only (shared edges, duplicate
    faces, pinch vertices).  **Loop fusions** — manifold cycles where
    two branches diverge and reconverge — are not detected here.  Loop
    fusion detection is planned for the skeletonization refactoring
    (detect cycles in the pre-MST skeleton graph).
    """
    if verbose:
        print("[skeliner.pre] Finding fusions …")
        _t0 = time.perf_counter()

    clusters = kernels.pre.find_nonmanifold_fusions(
        np.asarray(mesh.faces, dtype=np.int64),
        len(mesh.vertices),
        np.asarray(mesh.area_faces, dtype=np.float64),
        _outward_dots_for(mesh, radius, radius_multiplier, mesh_stats),
        grow_rings,
        min_branch_size,
        sort=True,
    )

    if verbose:
        dt = time.perf_counter() - _t0
        n_faces = sum(len(c) for c in clusters)
        print(
            f"[skeliner.pre] Fusions: {len(clusters)} regions, "
            f"{n_faces:,} boundary faces ({dt:.1f}s)"
        )

    return clusters


def _split_fan_vertices(
    mesh: Mesh,
    candidate_verts: set[int] | None = None,
    verbose: bool = False,
) -> Mesh:
    """Split vertices whose face fan is disconnected by face-edge adjacency.

    At a manifold fusion, two branches share vertices but not face-edges.
    Duplicating these vertices so each face-edge component gets its own
    copy disconnects the branches in the vertex graph.

    Returns a new mesh with split vertices (more vertices, same faces).
    """
    candidates = (
        None
        if candidate_verts is None
        else np.fromiter(sorted(candidate_verts), dtype=np.int64)
    )
    faces, sources = kernels.pre.split_fan_vertices(
        np.asarray(mesh.faces, dtype=np.int64), len(mesh.vertices), candidates
    )
    if len(sources) == 0:
        return mesh
    verts = np.asarray(mesh.vertices)
    result = Mesh(np.vstack([verts, verts[sources]]), faces)
    if verbose:
        print(
            f"[skeliner.pre] Split {len(sources)} fan vertices "
            f"({len(verts):,} -> {len(result.vertices):,} vertices)"
        )
    return result


@mesh_args()
def remove_fusions(
    mesh: Mesh,
    *,
    fusions: list[list[int]] | None = None,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Mesh:
    """Remove fusion faces and split shared vertices between branches.

    Two-step process:

    1. **Remove non-manifold fusion faces** — detected by
       :func:`find_fusions` (non-manifold edges, duplicate faces).
    2. **Split shared vertices** — vertices whose face fan is
       disconnected by face-edge adjacency, indicating two branches
       share a vertex without sharing face-edges.

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh (ideally after organelle removal).
    fusion_clusters : list[list[int]] or None
        Pre-computed clusters from :func:`find_fusions`.  If provided,
        detection is skipped and these clusters are used directly.
    radius : float or None
        Radius for outward_dot computation. Auto-computed if None.
    radius_multiplier : float
        Multiplier for auto radius.
    verbose : bool

    Returns
    -------
    Mesh
        Mesh with fusions removed and shared vertices split.
    """
    if verbose:
        print("[skeliner.pre] Removing fusions …")
        _t0 = time.perf_counter()

    # Step 1: remove non-manifold fusion faces
    if fusions is None:
        fusions = find_fusions(
            mesh,
            radius=radius,
            radius_multiplier=radius_multiplier,
            verbose=verbose,
        )

    all_fusions: set[int] = set()
    for c in fusions:
        all_fusions.update(c)

    n_verts_before = len(mesh.vertices)
    if all_fusions:
        faces, sources = kernels.pre.remove_fusions(
            np.asarray(mesh.faces, dtype=np.int64),
            len(mesh.vertices),
            [list(map(int, c)) for c in fusions],
        )
        verts = np.asarray(mesh.vertices)
        if len(sources):
            verts = np.vstack([verts, verts[sources]])
        mesh = Mesh(verts, faces)
    n_split = len(mesh.vertices) - n_verts_before

    if verbose:
        dt = time.perf_counter() - _t0
        parts = [f"removed {len(all_fusions):,}f"]
        if n_split:
            parts.append(f"split {n_split} fan vertices")
        print(f"[skeliner.pre] remove_fusions: {', '.join(parts)} ({dt:.1f}s)")

    # Invalidate topology (components split at fan vertices)
    if mesh_stats is not None:
        mesh_stats.invalidate_topology()

    return mesh


def _rebuild_mesh(
    mesh: Mesh,
    keep_mask: np.ndarray,
) -> Mesh:
    """Remove faces from a mesh, preserving all indices.

    Removed faces are replaced with degenerate triangles ``[0, 0, 0]``
    so that both vertex and face indices remain stable.  Downstream
    data (``soma.verts``, face-based annotations) stays valid without
    any remapping.

    Parameters
    ----------
    mesh : Mesh
        Original mesh.
    keep_mask : np.ndarray
        Boolean mask ``(nFaces,)`` — True for faces to keep.

    Returns
    -------
    Mesh
        Mesh with removed faces degenerated.  Same vertex and face
        array sizes as the input.
    """
    new_faces = mesh.faces.copy()
    new_faces[~keep_mask] = 0
    return Mesh(mesh.vertices, new_faces)


@mesh_args()
def remove_organelles(
    mesh: Mesh,
    *,
    organelles: Organelles | None = None,
    radius: float | None = None,
    radius_multiplier: float = 5.0,
    min_cluster_size: int = 5,
    verbose: bool = False,
) -> Mesh:
    """Remove internal mesh fragments (organelle membranes) from a neuron mesh.

    Organelle membranes (mitochondria, ER, etc.) often appear as
    connected or semi-connected components sitting *inside* the
    neuron body.  They bias skeleton-node positions and radius estimates
    and should be removed before skeletonisation.

    Vertex indices are preserved (unreferenced vertices are left in
    the array) so that vertex-based data like ``soma.verts`` remains
    valid without remapping.

    Note: this does NOT remove the nucleus membrane inside the soma.
    Use :func:`remove_nucleus` after skeletonisation for that.

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.  If
        provided, detection is skipped.
    radius : float or None, default None
        Radius (in mesh units, typically nm) for the local center-of-mass
        neighbourhood.  If None, automatically set to
        ``radius_multiplier * median_edge_length``.
    radius_multiplier : float, default 5.0
        Multiplier for automatic radius computation.  Only used when
        *radius* is None.
    min_cluster_size : int, default 5
        Connected components of internal faces smaller than this are
        kept (assumed to be noise).
    verbose : bool, default False
        Print summary statistics.

    Returns
    -------
    Mesh
        Cleaned mesh with internal fragments removed.  Vertex indices
        are unchanged from the input mesh.

    Examples
    --------
    org = find_organelles(mesh, verbose=True)
    soma = find_soma(mesh, organelles=org)
    clean = remove_organelles(mesh, organelles=org)
    # soma.verts is still valid on clean — no remapping needed
    """
    if organelles is None:
        organelles = find_organelles(
            mesh,
            radius=radius,
            radius_multiplier=radius_multiplier,
            min_cluster_size=min_cluster_size,
            verbose=verbose,
        )
    elif verbose:
        print(
            f"[skeliner.pre] Using provided organelles mask "
            f"({int(organelles.mask.sum()):,} faces)"
        )

    if not organelles.mask.any():
        if verbose:
            print("[skeliner.pre] Nothing to remove")
        return mesh

    clean = _rebuild_mesh(mesh, ~organelles.mask)

    if verbose:
        print(
            f"[skeliner.pre] Result: {len(clean.faces):,} faces "
            f"(removed {organelles.mask.sum():,} faces)"
        )

    return clean


# ── Chunk boundary / sandwich detection ────────────────────────────


@mesh_args()
def find_chunk_boundaries(
    mesh: Mesh,
    *,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Detect chunk boundary planes from vertex coordinate clustering.

    EM meshes are built by running marching cubes per chunk and merging
    at chunk boundaries.  At each boundary plane, many vertices share
    the exact same coordinate on the boundary axis.  This function finds
    those planes by detecting local vertex-count spikes on each axis.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    verbose : bool, default False
        Print detected boundaries.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``{axis: array_of_boundary_coordinates}`` where *axis*
        is 0 (X), 1 (Y), or 2 (Z).  Each array is sorted.
    """
    boundaries: dict[int, np.ndarray] = {}
    found = kernels.pre.find_chunk_boundaries(
        np.asarray(mesh.vertices, dtype=np.float64)
    )
    for axis, bvals, counts, spikes in found:
        boundaries[axis] = bvals
        if verbose:
            name = "XYZ"[axis]
            print(f"[skeliner.pre] Chunk boundaries {name}: {len(bvals)} planes")
            for val, count, spike in zip(bvals, counts, spikes, strict=True):
                print(f"  {name}={val}: {count} verts (spike {spike:.1f}x)")

    return boundaries


@mesh_args()
def find_parallel_patches(
    mesh: Mesh,
    *,
    boundaries: dict[int, np.ndarray] | None = None,
    normal_thresh: float = 0.99,
    boundary_dist: float = 30.0,
    expand_tilted: bool = True,
    loose_normal_thresh: float = 0.9,
    min_sandwich_overlap: float = 0.3,
    verbose: bool = False,
) -> list[dict]:
    """Detect parallel-patch artifacts at chunk boundaries.

    At chunk merge failures, marching cubes produces pairs of perfectly
    axis-aligned face patches on opposite sides of the boundary plane.
    This function finds connected patches of such faces and flags those
    that have an opposing patch (same boundary, overlapping in the
    perpendicular plane).

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    boundaries : dict or None
        Output of :func:`find_chunk_boundaries`.  Computed if *None*.
    normal_thresh : float
        Minimum |normal·axis| to count as axis-aligned (default 0.99).
    boundary_dist : float
        Max distance from a boundary plane in nm (default 30).
    expand_tilted : bool, default True
        Grow each seed patch at the end to absorb the tilted transition
        faces at its rim.
    loose_normal_thresh : float, default 0.9
        Looser threshold used during that rim expansion.
    min_sandwich_overlap : float, default 0.3
        After detection, drop patch pairs whose triangle-area overlap
        (in the perpendicular plane) is below this fraction of the
        smaller partner's area.  Filters out organelle-to-main
        parallels that are not marching-cubes sandwich artifacts.
        Set to ``0.0`` to disable.
    verbose : bool
        Print summary.

    Returns
    -------
    list[dict]
        Each entry: ``{"axis": int, "bval": int, "faces": list[int],
        "up_faces": list[int], "down_faces": list[int]}``.
    """
    if verbose:
        print("[skeliner.pre] Finding parallel patches …")
        _t0 = time.perf_counter()

    if boundaries is None:
        boundaries = find_chunk_boundaries(mesh)

    found = kernels.pre.find_parallel_patches(
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.face_normals, dtype=np.float64),
        [(int(ax), [float(b) for b in bvals]) for ax, bvals in boundaries.items()],
        normal_thresh,
        boundary_dist,
        expand_tilted,
        loose_normal_thresh,
    )
    results = [
        {"axis": a, "bval": b, "faces": f, "up_faces": up, "down_faces": dn}
        for a, b, f, up, dn in found
    ]

    if min_sandwich_overlap > 0:
        results = _filter_weak_sandwich_pairs(
            mesh,
            results,
            min_overlap=min_sandwich_overlap,
        )

    results.sort(key=lambda r: (r["axis"], r["bval"], -len(r["faces"])))

    if verbose:
        dt = time.perf_counter() - _t0
        n_bnd = sum(len(bv) for bv in boundaries.values())
        n_faces = sum(len(r["faces"]) for r in results)
        # Compact per-axis summary
        ax_counts = Counter(r["axis"] for r in results)
        ax_str = ", ".join(f"{'XYZ'[a]}:{c}" for a, c in sorted(ax_counts.items()))
        print(
            f"[skeliner.pre] Parallel patches: "
            f"{len(results)} patches, {n_faces:,} faces "
            f"({ax_str}) at {n_bnd} boundaries "
            f"({dt:.1f}s)"
        )

    return results


def _trim_patches_to_overlap(
    mesh: Mesh,
    patches: list[dict],
) -> list[dict]:
    """Trim each patch to the subset of faces whose projection onto
    the plane perpendicular to the patch axis intersects an opposing
    partner at the same ``(axis, bval)``.

    Only the portion of a parallel patch that actually sits above an
    opposing sandwich partner is a marching-cubes double-layer.  The
    rest of the same connected flat region (a large axis-aligned sheet
    that happens to be at a chunk boundary) is load-bearing topology
    and must not be removed — otherwise the mesh gets disconnected.

    Patches with no opposing partner in their group are dropped
    (empty ``faces``).  Mixed patches with both up and down faces are
    self-opposing: each sign is trimmed against the other sign's
    union.
    """

    if not patches:
        return patches

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    verts = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals
    perp = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    groups: dict[tuple, list[int]] = defaultdict(list)
    for pi, p in enumerate(patches):
        groups[(int(p["axis"]), int(p["bval"]))].append(pi)

    new_patches: list[dict] = []
    for (axis, bval), group_pis in groups.items():
        perp_axes = perp[axis]

        # Union polygon per sign, pooled from all patches in the group.
        sign_union: dict[int, object] = {}
        for sign_val in (1, -1):
            polys = []
            for pj in group_pis:
                p = patches[pj]
                side_faces = p["up_faces"] if sign_val > 0 else p["down_faces"]
                if not side_faces:
                    continue
                for tri in verts[faces[side_faces]][:, :, perp_axes]:
                    try:
                        poly = Polygon(tri)
                        if poly.is_valid and poly.area > 0:
                            polys.append(poly)
                    except Exception:
                        continue
            if polys:
                sign_union[sign_val] = unary_union(polys)

        for pi in group_pis:
            p = patches[pi]
            kept_faces: list[int] = []
            kept_up: list[int] = []
            kept_down: list[int] = []

            for fi in p["faces"]:
                n_ax = normals[fi, axis]
                is_up = n_ax > 0
                opp_sign = -1 if is_up else 1
                opp = sign_union.get(opp_sign)
                if opp is None:
                    continue
                tri = verts[faces[fi]][:, perp_axes]
                try:
                    face_poly = Polygon(tri)
                    if not face_poly.is_valid:
                        continue
                    if face_poly.intersects(opp):
                        fi_int = int(fi)
                        kept_faces.append(fi_int)
                        if is_up:
                            kept_up.append(fi_int)
                        else:
                            kept_down.append(fi_int)
                except Exception:
                    continue

            if kept_faces:
                new_patches.append(
                    {
                        "axis": axis,
                        "bval": bval,
                        "faces": kept_faces,
                        "up_faces": kept_up,
                        "down_faces": kept_down,
                    }
                )

    return new_patches


def _sandwich_overlap_ratio(
    mesh: Mesh,
    faces_a: list[int],
    faces_b: list[int],
    axis: int,
) -> float:
    """Triangle-area overlap of two patches projected onto the plane
    perpendicular to *axis*, normalised by the smaller patch's area.

    Returns 0.0 if either side is empty or the polygons are invalid.
    """
    if not faces_a or not faces_b:
        return 0.0
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    perp = [i for i in range(3) if i != axis]
    verts = mesh.vertices
    tris_a = verts[mesh.faces[faces_a]][:, :, perp]
    tris_b = verts[mesh.faces[faces_b]][:, :, perp]

    polys_a = [Polygon(t) for t in tris_a if Polygon(t).is_valid]
    polys_b = [Polygon(t) for t in tris_b if Polygon(t).is_valid]
    if not polys_a or not polys_b:
        return 0.0

    union_a = unary_union(polys_a)
    union_b = unary_union(polys_b)
    if union_a.is_empty or union_b.is_empty:
        return 0.0

    area_a = float(union_a.area)
    area_b = float(union_b.area)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter_area = float(union_a.intersection(union_b).area)
    return inter_area / min(area_a, area_b)


def _filter_weak_sandwich_pairs(
    mesh: Mesh,
    patches: list[dict],
    *,
    min_overlap: float = 0.3,
) -> list[dict]:
    """Drop patches whose best opposing partner at the same (axis,bval)
    has triangle-area overlap below *min_overlap* in the perpendicular
    plane.  Keeps only patches that are genuine marching-cubes sandwich
    sides — filters out organelle-to-main parallels where the two
    sheets barely intersect.
    """

    if not patches or min_overlap <= 0:
        return patches

    groups: dict[tuple, list[int]] = defaultdict(list)
    for pi, p in enumerate(patches):
        groups[(int(p["axis"]), int(p["bval"]))].append(pi)

    keep = [False] * len(patches)

    for (axis, _bval), group_pis in groups.items():
        # For each patch, find the best opposing overlap ratio.
        for pi in group_pis:
            p = patches[pi]
            up_i = p["up_faces"]
            dn_i = p["down_faces"]
            # Self-opposing (mixed patch): its own up vs down
            best = 0.0
            if up_i and dn_i:
                best = max(best, _sandwich_overlap_ratio(mesh, up_i, dn_i, axis))
            # Pure plus or minus: look at other patches at same boundary
            for pj in group_pis:
                if pj == pi:
                    continue
                q = patches[pj]
                if up_i and q["down_faces"]:
                    best = max(
                        best,
                        _sandwich_overlap_ratio(mesh, up_i, q["down_faces"], axis),
                    )
                if dn_i and q["up_faces"]:
                    best = max(
                        best,
                        _sandwich_overlap_ratio(mesh, dn_i, q["up_faces"], axis),
                    )
            if best >= min_overlap:
                keep[pi] = True

    return [p for pi, p in enumerate(patches) if keep[pi]]


@mesh_args()
def remove_parallel_patches(
    mesh: Mesh,
    *,
    patches: list[dict] | None = None,
    boundaries: dict[int, np.ndarray] | None = None,
    verbose: bool = False,
    mesh_stats: MeshStats | None = None,
) -> Mesh:
    """Remove parallel-patch artifacts at chunk boundaries.

    1. Classify each patch as Mode A (has fold edges, where an up face
       shares an edge with a down face) or Mode B (no fold edges).
    2. Remove all detected patch faces.
    3. Find new disconnected components created by the removal.
    4. For each new component, check which removed patches it borders:
       - borders only Mode A → orphan from fold removal → remove
       - borders any Mode B → real neurite piece → keep (for stitching)

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    patches : list[dict] or None
        Output of :func:`find_parallel_patches`.  Computed if *None*.
    boundaries : dict or None
        Passed through to :func:`find_parallel_patches` if *patches* is None.
    verbose : bool
        Print summary.

    Returns
    -------
    Mesh
        Mesh with patch artifacts and Mode A orphans removed.
    """
    if verbose:
        print("[skeliner.pre] Removing parallel patches …")
        _t0 = time.perf_counter()

    if patches is None:
        patches = find_parallel_patches(mesh, boundaries=boundaries, verbose=verbose)

    if not patches:
        return mesh

    # ── Step 0: trim patches to overlap with opposing partner ─────
    # Only the portion of a parallel patch that sits above an opposing
    # sandwich partner is a marching-cubes double-layer.  Trimming
    # avoids disconnecting the mesh when a large flat axis-aligned
    # sheet is load-bearing topology with only a small MC sandwich
    # sitting in one corner (see site_02 case).
    n_faces_before_trim = sum(len(p["faces"]) for p in patches)
    patches = _trim_patches_to_overlap(mesh, patches)
    n_faces_after_trim = sum(len(p["faces"]) for p in patches)
    if not patches:
        return mesh

    faces_out, n_removed, n_a, n_b, n_orphan, n_stitched = (
        kernels.pre.remove_parallel_patches(
            np.asarray(mesh.faces, dtype=np.int64),
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.face_normals, dtype=np.float64),
            [
                (p["axis"], p["bval"], p["faces"], p["up_faces"], p["down_faces"])
                for p in patches
            ],
        )
    )
    result = Mesh(mesh.vertices, faces_out)
    n_trimmed = n_faces_before_trim - n_faces_after_trim

    # Invalidate topology; pad outward_dots for appended stitch faces
    if mesh_stats is not None:
        if n_stitched > 0 and mesh_stats.outward_dots is not None:
            mesh_stats.outward_dots = np.concatenate(
                [
                    mesh_stats.outward_dots,
                    np.ones(
                        n_stitched,
                        dtype=mesh_stats.outward_dots.dtype,
                    ),
                ]
            )
        mesh_stats.invalidate_topology()

    if verbose:
        parts = [
            f"removed {n_removed:,}f ({n_a} fold + {n_b} parallel)",
        ]
        if n_trimmed:
            parts.append(
                f"trimmed to overlap {n_faces_before_trim:,}→{n_faces_after_trim:,}f"
            )
        if n_orphan:
            parts.append(f"+{n_orphan:,}f orphans removed")
        if n_stitched:
            parts.append(f"+{n_stitched:,}f stitched")
        dt = time.perf_counter() - _t0
        print(f"[skeliner.pre] remove_parallel_patches: {'; '.join(parts)} ({dt:.1f}s)")

    return result


@mesh_args()
def compact_mesh(
    mesh: Mesh,
    components: MeshComponents,
    *,
    return_maps: bool = False,
    verbose: bool = False,
) -> tuple[Mesh, MeshComponents] | tuple[Mesh, MeshComponents, np.ndarray, np.ndarray]:
    """Remove degenerate faces, drop unreferenced vertices, reindex.

    Call once at the end of preprocessing, after all face removals.
    Remaps all face/vertex-indexed data inside *components*.

    Parameters
    ----------
    mesh : Mesh
        Mesh after face removals.
    components : MeshComponents
        All face-indexed data is remapped to match the compacted mesh.
    return_maps : bool, default False
        If True, also return ``vert_map`` and ``face_map`` arrays for
        remapping external data (e.g. viewer annotations).
    verbose : bool, default False
        Print summary.

    Returns
    -------
    clean : Mesh
        Compacted mesh with no unreferenced vertices.
    components : MeshComponents
        Remapped components.
    vert_map : np.ndarray *(only when return_maps=True)*
        ``(nOldVerts,)`` int64 — ``vert_map[old]`` → new index, or -1.
    face_map : np.ndarray *(only when return_maps=True)*
        ``(nOldFaces,)`` int64 — ``face_map[old]`` → new index, or -1.
    """
    # Strip the collapsed faces (from _rebuild_mesh) and compact vertices
    good, vert_map, face_map, verts, faces = kernels.pre.compact_mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )
    n_old = len(mesh.vertices)
    clean = Mesh._adopt(verts, faces)

    if verbose:
        n_removed_v = n_old - len(clean.vertices)
        n_removed_f = len(mesh.faces) - len(clean.faces)
        print(
            f"[skeliner.pre] Compact: {n_old:,} → {len(clean.vertices):,} verts "
            f"({n_removed_v:,} removed), "
            f"{len(mesh.faces):,} → {len(clean.faces):,} faces "
            f"({n_removed_f:,} removed)"
        )

    # Remap soma
    remapped_soma = None
    if components.soma is not None:
        remapped_soma = components.soma.remap(vert_map)
        if verbose:
            n_before = (
                len(components.soma.verts) if components.soma.verts is not None else 0
            )
            n_after = len(remapped_soma.verts) if remapped_soma.verts is not None else 0
            print(
                f"[skeliner.pre] Soma remap: {n_before:,} → {n_after:,} verts "
                f"({n_before - n_after:,} dropped)"
            )

    def _remap_face_list(arrays):
        out = []
        for arr in arrays:
            mapped = face_map[arr]
            mapped = mapped[mapped >= 0]
            if len(mapped) > 0:
                out.append(mapped)
        return out

    organelles = components.organelles
    remapped_org = Organelles(
        pocket=organelles.pocket[good],
        isolated=organelles.isolated[good],
        expanded=organelles.expanded[good],
        manual=organelles.manual[good],
    )

    # Names ride through the remap.  Compaction reindexes faces and changes
    # nothing about the partition — same pieces, same order — so unlike a
    # re-derive it is not a reason to drop what they were called.
    src_labels = components.neurites.labels
    src_types = components.neurites.swc_types
    remapped = MeshComponents(
        soma=remapped_soma,
        organelles=remapped_org,
        neurites=Neurites(
            _remap_face_list(components.neurites),
            labels=None if src_labels is None else list(src_labels),
            swc_types=None if src_types is None else list(src_types),
        ),
        discarded=Discarded(_remap_face_list(components.discarded)),
    )

    if return_maps:
        return clean, remapped, vert_map, face_map
    return clean, remapped


# -----------------------------------------------------------------------------
# High-level preprocessing pipeline
# -----------------------------------------------------------------------------


@mesh_args()
def preprocess(
    mesh: Mesh,
    *,
    organelles: bool = True,
    soma: Soma | None = None,
    rescued=None,
    released=None,
    claim_min_faces: int = 0,
    compact: bool = False,
    verbose: bool = False,
) -> tuple[Mesh, MeshComponents]:
    """Run the full preprocessing pipeline in one call.

    Pipeline order:

      1. find_parallel_patches → remove_parallel_patches
      2. compute_mesh_stats
      3. find_organelles (unless *organelles* is False)
      4. find_soma (unless *soma* is given)
      5. find_fusions → find_disconnected → find_gaps
      6. remove_gaps → remove_fusions
      7. break_up_mesh
      8. compact_mesh (optional)

    Each ``find_*`` step is skipped if it returns nothing.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    organelles : bool, default True
        False for a mesh that traces no organelles (hemibrain's): what
        :func:`find_organelles` would flag there is fragments and patches
        of the cell, so it is not run and no face is an organelle.
    soma : Soma or None
        A soma to take instead of :func:`find_soma`'s: one assigned by hand
        where it found none.  Its ``verts`` name this mesh's vertices,
        which the step before it (parallel patches) keeps.
    rescued : array-like of int, optional
        Face ids the caller has declared arbor, passed through to
        :func:`break_up_mesh` in step 7.  The neurite/discarded split is
        re-derived from scratch here, so an override that is not fed back
        in is undone by the run.
    released : array-like of int, optional
        Lasso-drawn arbor claims, likewise passed through to step 7.
    claim_min_faces : int, default 0
        Passed to :func:`break_up_mesh` in step 7.
    compact : bool, default False
        If True, run :func:`compact_mesh` as a final step to drop
        unreferenced vertices and degenerate faces.  This invalidates
        original mesh indices, so external data tied to them (e.g.
        viewer annotations) will need to be remapped.  Skeletonization
        works correctly on uncompacted meshes.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    mesh : Mesh
        Cleaned mesh (gaps stitched, fusions removed, optionally
        compacted).
    components : MeshComponents
        Classified pieces: soma, organelles, neurites, discarded.
    """
    if verbose:
        print(
            f"[skeliner.pre] starting preprocessing "
            f"({len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} faces)"
        )
        _t0 = time.perf_counter()

    # 1. Parallel patches
    with _timed("↳  find & remove parallel patches", verbose=verbose):
        patches = find_parallel_patches(mesh)
        if patches:
            mesh = remove_parallel_patches(mesh, patches=patches)

    # 2. Mesh stats — outward dots and components, which the later steps
    #    read whether or not organelles are looked for
    with _timed("↳  mesh stats", verbose=verbose):
        mesh_stats = compute_mesh_stats(mesh)

    # 3. Organelles
    with _timed("↳  find organelles", verbose=verbose) as log:
        if organelles:
            org = find_organelles(mesh, mesh_stats=mesh_stats)
            log(
                f"pocket={int(org.pocket.sum()):,}, "
                f"isolated={int(org.isolated.sum()):,}"
            )
        else:
            org = Organelles.empty(len(mesh.faces))
            log("none: organelles=False")

    # 4. Soma
    with _timed("↳  find soma", verbose=verbose) as log:
        if soma is not None:
            log(f"given: center={soma.center.round(0).tolist()}")
        else:
            soma = find_soma(mesh, organelles=org, mesh_stats=mesh_stats)
            if soma is not None:
                log(f"center={soma.center.round(0).tolist()}")
            else:
                log("no soma detected")

    # 5. Fusions → disconnected → gaps
    with _timed("↳  find fusions", verbose=verbose) as log:
        fusions = find_fusions(mesh, mesh_stats=mesh_stats)
        log(f"{len(fusions)} fusions")

    with _timed("↳  find disconnected", verbose=verbose) as log:
        disconnected = find_disconnected(
            mesh,
            soma=soma,
            organelles=org,
            mesh_stats=mesh_stats,
            fusions=fusions,
        )
        log(f"{len(disconnected)} disconnected components")

    with _timed("↳  find & remove gaps", verbose=verbose) as log:
        gaps = find_gaps(
            mesh,
            soma=soma,
            disconnected=disconnected,
            mesh_stats=mesh_stats,
            fusions=fusions,
        )
        log(f"{len(gaps)} gaps")
        if gaps:
            mesh = remove_gaps(mesh, gaps=gaps, mesh_stats=mesh_stats)

    # 6. Fusions
    with _timed("↳  remove fusions", verbose=verbose):
        if fusions:
            mesh = remove_fusions(mesh, fusions=fusions, mesh_stats=mesh_stats)

    # 7. Break up mesh
    with _timed("↳  break up mesh", verbose=verbose) as log:
        components = break_up_mesh(
            mesh,
            soma,
            org,
            rescued=rescued,
            released=released,
            claim_min_faces=claim_min_faces,
        )
        log(f"{len(components.neurites)} neurites")

    # 8. Compact (optional — remaps everything in MeshComponents)
    if compact:
        with _timed("↳  compact mesh", verbose=verbose):
            mesh, components = compact_mesh(mesh, components=components)

    if verbose:
        dt = time.perf_counter() - _t0
        print(
            f"[skeliner.pre] preprocessing done ({len(mesh.faces):,} faces, {dt:.1f}s)"
        )

    return mesh, components


# =====================================================================
#  Z-slice helpers (shared by find_nucleus_center & find_soma_via_z_contour)
# =====================================================================


#  find_nucleus_center — nucleus detection via Z-slice void analysis
# =====================================================================


@mesh_args()
def find_nucleus_center(
    mesh: Mesh,
    *,
    grid_res: float | None = None,
    z_tol: float | None = None,
    max_shift: float = 3000.0,
    min_void_r: float | None = None,
    verbose: bool = False,
) -> dict | None:
    """Locate the nucleus center from the mesh geometry alone.

    The nucleus membrane folds inward from the soma surface, creating
    a large void visible in Z-slice cross-sections.  At each Z-level
    the algorithm rasterizes vertex positions into a 2D occupancy grid,
    isolates the largest connected cluster (the soma cross-section),
    and measures the maximum distance-to-nearest-vertex within a
    4-ray-enclosed region.  The nucleus shows up as a sustained void
    across many consecutive Z-levels at a stable XY location.

    **Algorithm**

    1. For each Z-level, rasterize mesh vertex XY positions into a
       2D grid at *grid_res* resolution.
    2. Dilate by 1 px and take connected components; keep only the
       largest (the soma cross-section, which is bigger than any
       neurite cross-section).
    3. On the *original* (undilated) grid within that component, run a
       4-ray enclosure test (occupied cell must exist in +x, −x, +y,
       −y).  Enclosed empty cells are void candidates.
    4. Distance-transform the void candidates; the peak gives the void
       centre and radius for that Z-level.
    5. Chain consecutive Z-levels whose void centres are within
       *max_shift* of each other.  The longest chain = nucleus.
    6. Return the mean void centre across the chain.

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh (any cell type).
    grid_res : float or None
        2D rasterization cell size in nm.  It must be fine enough to
        resolve the nucleus void (~2 000–3 000 nm radius) and coarse
        enough that the gaps between the soma surface's vertices close.
        ``None`` (default) takes the mesh's median edge length, at least
        200 nm — which is 200 nm on full-resolution meshes (edges of
        ~105 nm) and on a vertex cloud without faces; a coarser mesh
        needs a coarser grid.
    z_tol : float or None
        Half-width of the Z slab for collecting vertices at each level.
        ``None`` (default) takes 0.75 × the median edge, at least 150 nm.
    max_shift : float
        Maximum XY displacement (nm) between consecutive Z-levels for
        a void to be considered part of the same chain.
    min_void_r : float or None
        Minimum void radius (nm) for a Z-level to join the chain.  ``None``
        (default) takes it from the mesh: a nucleus is the cell's one void
        no neurite can hold, so the chain holding the cell's biggest void
        must beat every void elsewhere in the cell — outside its own body,
        which a neurite leaving it is not — by more than one grid cell (what
        the grid resolves), and the chain's levels must beat that biggest
        void elsewhere by a cell too.  A cell whose biggest void does not
        stand out has no nucleus.  A fixed size cannot do this: EyeWire 2's
        500 nm let the lumens of hemibrain's and EyeWire 1's thick neurites
        pass.
    verbose : bool
        Print progress.

    Returns
    -------
    dict or None
        ``None`` if no sustained void is found.  Otherwise a dict with:

        - ``center`` — ``(3,)`` nucleus centre in world coordinates.
        - ``z_range`` — ``(z_lo, z_hi)`` Z extent of the detected void.
        - ``peak_r`` — peak void radius (nm) across the chain.
        - ``slices`` — ``(N, 4)`` array where each row is
          ``(z, cx, cy, void_r)`` for the N Z-levels in the best chain.
          *cx, cy* are the void centre at each level and *void_r* is
          the void radius at that level.
    """
    _p = "[find_nucleus_center]"

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if grid_res is None or z_tol is None:
        # a vertex cloud has no edges to measure: full resolution's settings
        edge = _median_edge(mesh) if len(mesh.faces) else 0.0
        grid_res = max(200.0, edge) if grid_res is None else grid_res
        z_tol = max(150.0, 0.75 * edge) if z_tol is None else z_tol
    if verbose:
        z_unique = np.unique(verts[:, 2])
        n_levels = int((z_unique.max() - z_unique.min()) / 210)
        print(
            f"{_p} {len(verts):,} vertices, ~{n_levels} Z-levels,"
            f" grid {grid_res:.0f} nm, slab ±{z_tol:.0f} nm"
        )

    raw, best, center = _z_scan(verts, grid_res, z_tol, min_void_r, max_shift)

    if best is None:
        if verbose:
            print(f"{_p} no sustained void found")
        return None

    slices_numeric = np.array(
        [(raw[i][0], raw[i][1], raw[i][2], raw[i][3]) for i in best]
    )
    contours = {raw[i][0]: raw[i][4] for i in best}

    peak_r = float(slices_numeric[:, 3].max())
    z_lo = float(slices_numeric[0, 0])
    z_hi = float(slices_numeric[-1, 0])

    if verbose:
        print(
            f"{_p} nucleus: center=({center[0]:.0f}, {center[1]:.0f}, "
            f"{center[2]:.0f}), Z=[{z_lo:.0f},{z_hi:.0f}], "
            f"peak_r={peak_r:.0f} nm, {len(best)} Z-levels"
        )

    # Soma-cluster vertex indices at the mid-chain Z-level.
    # These are outer-surface vertices (not void/pocket), suitable
    # as BFS seeds for soma detection.
    mid_i = best[len(best) // 2]
    mid_vi = raw[mid_i][5]  # vert_indices at this Z
    mid_sm = raw[mid_i][6]  # soma_mask (bool over vert_indices)
    soma_seed_vi = mid_vi[mid_sm] if mid_sm is not None else np.array([], dtype=np.intp)

    return {
        "center": center,
        "z_range": (z_lo, z_hi),
        "peak_r": peak_r,
        "slices": slices_numeric,
        "contours": contours,
        "soma_seed_vi": soma_seed_vi,
    }


#  find_soma_via_z_contour — soma detection using Z-slice contours
# =====================================================================


def _z_scan(
    verts: np.ndarray,
    grid_res: float = 200.0,
    z_tol: float = 150.0,
    min_void_r: float | None = None,
    max_shift: float = 3000.0,
    retry_broken: bool = True,
) -> tuple[list[tuple], list[int] | None, np.ndarray | None]:
    """Shared Z-scan: cluster + void detection + spatial coherence.

    Two-pass approach:

    1. Per-Z independent clustering + void detection → find best chain,
       past *min_void_r* or, when None, the mesh's own void minimum.
    2. Hull-guided void detection for Z-levels where the soma ring has
       holes (mesh artifacts / missing faces).  The gap breaks the
       4-ray enclosure test, so the void goes undetected.  Pass 2
       borrows the convex hull from the nearest good neighbor to close
       the gap and recover the void.

    Returns ``(raw, best_run, nucleus_center)`` where *raw* is the
    per-Z-level data ``[(z, cx, cy, void_r, void_xy, vert_indices,
    soma_mask), ...]``, *best_run* is the indices into *raw* of the
    nucleus chain (or None), and *nucleus_center* is ``(3,)`` or None.
    """
    verts = np.asarray(verts, dtype=np.float64)
    scan = kernels.pre.ZScan(verts, grid_res, z_tol, min_void_r, max_shift)
    if scan.best is not None and retry_broken:
        # Pass 2 needs the convex hull of each chain end's soma cluster.
        hulls = []
        for ref_i in scan.chain_ends():
            sv = scan.soma_vertices(ref_i)
            ok = sv is not None and len(sv) >= 4
            hulls.append(kernels.mesh.hull_equations(verts[sv, :2]) if ok else None)
        scan.retry(*hulls)
    return scan.raw(), scan.best, scan.center


def _soma_hulls(
    raw: list[tuple],
    best: list[int],
    center: np.ndarray,
    verts: np.ndarray,
    grid_res: float = 200.0,
) -> tuple[dict, bool, bool]:
    """Build per-Z soma hulls with neurite protrusions stripped.

    For each Z-level with a soma cluster, rasterizes the soma vertices,
    fills the ring interior to form a solid disk, then applies
    morphological opening to remove thin structures (neurite
    protrusions).  The convex hull is built from the surviving
    (soma-only) grid cells.

    Returns ``(hulls, extend_lo, extend_hi)`` where *hulls* is a dict
    mapping raw-index → Shapely Polygon, and *extend_lo* / *extend_hi*
    indicate whether the soma extends to the mesh boundary on each side
    (True = nothing beyond the soma, extend to mesh edge; False = soma
    is bounded by neurites, don't extend).
    """
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        distance_transform_edt,
    )
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon as _Poly

    soma_levels = [
        i for i in range(len(raw)) if raw[i][6] is not None and raw[i][6].sum() >= 4
    ]

    hulls: dict[int, _Poly] = {}
    struct3 = np.ones((3, 3), dtype=bool)

    for i in soma_levels:
        sm = raw[i][6]
        vi = raw[i][5]
        soma_pts = verts[vi[sm], :2]
        if len(soma_pts) < 4:
            continue

        # Rasterize soma vertices
        xy_min = soma_pts.min(axis=0) - grid_res * 3
        nx = int((soma_pts[:, 0].max() - xy_min[0] + grid_res * 6) / grid_res) + 1
        ny = int((soma_pts[:, 1].max() - xy_min[1] + grid_res * 6) / grid_res) + 1
        occ = np.zeros((nx, ny), dtype=bool)
        ix = ((soma_pts[:, 0] - xy_min[0]) / grid_res).astype(int).clip(0, nx - 1)
        iy = ((soma_pts[:, 1] - xy_min[1]) / grid_res).astype(int).clip(0, ny - 1)
        occ[ix, iy] = True

        # Close ring gaps → fill interior → open to strip neurites.
        closed = binary_dilation(occ, struct3, iterations=2)
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
                # Build hull from opened grid cell centres
                gi, gj = np.where(opened)
                hull_pts = np.column_stack(
                    [
                        xy_min[0] + gi * grid_res + grid_res / 2,
                        xy_min[1] + gj * grid_res + grid_res / 2,
                    ]
                )
                try:
                    h = ConvexHull(hull_pts)
                    p = _Poly(hull_pts[h.vertices])
                    if p.is_valid:
                        hulls[i] = p
                        continue
                except Exception:
                    pass

        # Fallback: raw convex hull (opening too small or failed)
        try:
            h = ConvexHull(soma_pts)
            p = _Poly(soma_pts[h.vertices])
            if p.is_valid:
                hulls[i] = p
        except Exception:
            pass

    # Walk outward from the nucleus chain in both Z-directions.
    # Stop when the hull area drops well below the peak (soma is
    # tapering into neurites).  If the area is still large at the
    # mesh boundary, include everything — the soma is cut off by
    # the segmentation volume, not by a natural taper.
    chain_areas = [hulls[i].area for i in best if i in hulls]
    peak_area = max(chain_areas) if chain_areas else 0
    max_r = np.sqrt(peak_area / np.pi) if peak_area > 0 else 5000
    max_dist = max_r * 1.5
    # Collect all levels with hulls, sorted by index
    hull_levels = sorted(hulls.keys())

    # Find the chain's position in hull_levels
    chain_set = set(best)
    chain_positions = [pos for pos, i in enumerate(hull_levels) if i in chain_set]
    if not chain_positions:
        return {}, False, False
    mid_lo = min(chain_positions)
    mid_hi = max(chain_positions)

    def _walk(cutoff):
        """Walk outward from the chain, return (accepted, ext_lo, ext_hi)."""
        acc = set()
        for pos in range(mid_lo, mid_hi + 1):
            acc.add(hull_levels[pos])

        e_lo = True
        for pos in range(mid_lo - 1, -1, -1):
            i = hull_levels[pos]
            p = hulls[i]
            if p.area < cutoff:
                e_lo = False
                break
            cx, cy = p.centroid.x, p.centroid.y
            dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            if dist > max_dist:
                e_lo = False
                break
            acc.add(i)

        e_hi = True
        for pos in range(mid_hi + 1, len(hull_levels)):
            i = hull_levels[pos]
            p = hulls[i]
            if p.area < cutoff:
                e_hi = False
                break
            cx, cy = p.centroid.x, p.centroid.y
            dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            if dist > max_dist:
                e_hi = False
                break
            acc.add(i)

        return acc, e_lo, e_hi

    # Two area cutoffs derived from the data:
    # - Low cutoff (5% of peak): for sides where the soma extends to
    #   the mesh boundary — keeps levels that are still clearly soma.
    # - High cutoff (50% of min chain area): for sides bounded by
    #   neurites — stops before neurite-sized levels.
    cutoff_lo = peak_area * 0.05
    min_chain_area = min(chain_areas) if chain_areas else peak_area
    cutoff_hi = min_chain_area * 0.5

    # First walk with low cutoff to determine which sides are bounded.
    acc_probe, extend_lo, extend_hi = _walk(cutoff_lo)

    # Override: if the soma walk reached close to the mesh boundary,
    # the soma is at the edge of the segmentation volume — extend
    # regardless of why the walk stopped.
    mesh_z_lo = float(verts[:, 2].min())
    mesh_z_hi = float(verts[:, 2].max())
    z_step = 210.0
    if not extend_lo and acc_probe:
        lo_z = min(raw[i][0] for i in acc_probe)
        if lo_z - mesh_z_lo < z_step * 3:
            extend_lo = True
    if not extend_hi and acc_probe:
        hi_z = max(raw[i][0] for i in acc_probe)
        if mesh_z_hi - hi_z < z_step * 3:
            extend_hi = True

    # Second walk: use tight cutoff for bounded sides, low for unbounded.
    # If both sides have the same cutoff, one walk suffices.
    if extend_lo == extend_hi:
        cutoff = cutoff_lo if extend_lo else cutoff_hi
        accepted, _, _ = _walk(cutoff)
    else:
        # Different cutoffs per side — walk with tight cutoff, then
        # re-extend the unbounded side with the low cutoff.
        accepted, _, _ = _walk(cutoff_hi)
        # Re-walk the unbounded side with the low cutoff
        if extend_lo:
            for pos in range(mid_lo - 1, -1, -1):
                i = hull_levels[pos]
                p = hulls[i]
                if p.area < cutoff_lo:
                    break
                cx, cy = p.centroid.x, p.centroid.y
                dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
                if dist > max_dist:
                    break
                accepted.add(i)
        if extend_hi:
            for pos in range(mid_hi + 1, len(hull_levels)):
                i = hull_levels[pos]
                p = hulls[i]
                if p.area < cutoff_lo:
                    break
                cx, cy = p.centroid.x, p.centroid.y
                dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
                if dist > max_dist:
                    break
                accepted.add(i)

    return {i: hulls[i] for i in accepted}, extend_lo, extend_hi


@_deprecated("find_soma()")
@mesh_args()
def find_soma_via_z_contour(
    mesh: Mesh,
    *,
    organelles: Organelles | None = None,
    grid_res: float = 200.0,
    z_tol: float = 150.0,
    verbose: bool = False,
) -> "Soma | None":
    """Detect the soma by first locating the nucleus void.

    .. deprecated::
        Use :func:`find_soma`.

    Uses :func:`_z_scan` (fast) to find the nucleus, then builds
    per-Z soma hulls with neurite protrusions stripped via
    morphological opening.  Classifies faces whose centroids fall
    inside the per-Z hulls as soma.

    Parameters
    ----------
    mesh : Mesh
        Input neuron mesh.
    organelles : Organelles or None
        Pre-computed organelles from :func:`find_organelles`.  If
        provided, organelle faces are excluded from the soma vertex set.
    grid_res : float
        2D rasterization cell size in nm.
    z_tol : float
        Half-width of the Z slab.
    verbose : bool
        Print progress.

    Returns
    -------
    Soma or None
    """
    from shapely.geometry import Point

    _p = "[find_soma_via_z_contour]"

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verbose:
        print(f"{_p} {len(verts):,} vertices")

    raw, best, nc = _z_scan(verts, grid_res, z_tol)

    if nc is None:
        if verbose:
            print(f"{_p} no nucleus found")
        return None

    if verbose:
        n_chain = len(best)
        peak_r = max(raw[i][3] for i in best)
        print(
            f"{_p} nucleus: ({nc[0]:.0f}, {nc[1]:.0f}, {nc[2]:.0f}), "
            f"r={peak_r:.0f}nm, {n_chain} Z-levels"
        )

    soma_hulls, extend_lo, extend_hi = _soma_hulls(raw, best, nc, verts, grid_res)

    from shapely import prepare

    hull_polys: list[tuple[float, object]] = []
    for i in sorted(soma_hulls.keys()):
        poly = soma_hulls[i]
        if poly is None or poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_valid:
            prepare(poly)
            hull_polys.append((raw[i][0], poly))

    if not hull_polys:
        if verbose:
            print(f"{_p} no soma contours found")
        return None

    # Small expansion to account for grid-resolution tolerance.
    # The hull is built from grid cell centres (grid_res=200nm), so
    # surface faces up to ~1 cell away may be just outside.
    expanded_polys: list[tuple[float, object]] = []
    for z_val, poly in hull_polys:
        ep = poly.buffer(grid_res)
        if not ep.is_valid:
            ep = ep.buffer(0)
        prepare(ep)
        expanded_polys.append((z_val, ep))

    hull_z = np.array([h[0] for h in expanded_polys])
    z_lo, z_hi = hull_z.min(), hull_z.max()
    n_z_hit = len(expanded_polys)

    # Extend the classification range to the mesh boundary on sides
    # where the soma walk didn't encounter a natural taper (nothing
    # beyond the soma — it's cut off by the segmentation volume).
    # Faces beyond the last hull are tested against the nearest hull.
    if extend_lo:
        z_lo = float(verts[:, 2].min())
    if extend_hi:
        z_hi = float(verts[:, 2].max())

    # Classify faces: centroid inside the nearest Z-level's contour → soma.
    faces = np.asarray(mesh.faces)
    centroids = verts[faces].mean(axis=1)  # (n_faces, 3)

    z_in_range = (centroids[:, 2] >= z_lo) & (centroids[:, 2] <= z_hi)
    cand_fi = np.where(z_in_range)[0]

    soma_face = np.zeros(len(faces), dtype=bool)
    cand_z = centroids[cand_fi, 2]
    nearest_idx = np.searchsorted(hull_z, cand_z).clip(0, len(hull_z) - 1)

    # Check nearest and the one before (searchsorted gives insertion point)
    for offset in [0, -1]:
        idx = (nearest_idx + offset).clip(0, len(hull_z) - 1)
        for hi in range(n_z_hit):
            mask = idx == hi
            if not mask.any():
                continue
            fi_batch = cand_fi[mask]
            pts_xy = centroids[fi_batch, :2]
            inside = np.array(
                [expanded_polys[hi][1].contains(Point(p)) for p in pts_xy]
            )
            soma_face[fi_batch] |= inside

    # Exclude organelle faces from soma
    if organelles is not None:
        soma_face &= ~organelles.mask

    # Keep only the largest connected component of soma faces

    adj = _face_adjacency(mesh)
    soma_idx = set(np.where(soma_face)[0].tolist())
    visited: set[int] = set()
    largest_comp: list[int] = []
    for fi in soma_idx:
        if fi in visited:
            continue
        comp: list[int] = []
        q = deque([fi])
        while q:
            c = q.popleft()
            if c in visited:
                continue
            visited.add(c)
            comp.append(c)
            for nfi in adj.get(c, set()):
                if nfi in soma_idx and nfi not in visited:
                    q.append(nfi)
        if len(comp) > len(largest_comp):
            largest_comp = comp
    soma_face[:] = False
    for fi in largest_comp:
        soma_face[fi] = True

    n_soma_faces = int(soma_face.sum())
    if n_soma_faces < 4:
        if verbose:
            print(f"{_p} too few soma faces ({n_soma_faces})")
        return None

    # Soma vertices = all vertices of soma faces
    soma_vert_set = set(faces[soma_face].ravel().tolist())
    soma_arr = np.fromiter(sorted(soma_vert_set), dtype=np.intp)
    soma = Soma.fit(mesh.vertices[soma_arr], verts=soma_arr)

    # Build nucleus dict from _z_scan results
    nuc_slices = np.array([(raw[i][0], raw[i][1], raw[i][2], raw[i][3]) for i in best])
    soma.nucleus = {
        "center": nc,
        "peak_r": float(nuc_slices[:, 3].max()),
        "z_range": (float(nuc_slices[0, 0]), float(nuc_slices[-1, 0])),
        "slices": nuc_slices,
    }

    if verbose:
        print(
            f"{_p} soma: {int(soma_face.sum()):,} faces, "
            f"{len(soma.verts):,} verts from {n_z_hit} Z-levels, "
            f"center=[{soma.center[0]:.0f}, {soma.center[1]:.0f}, "
            f"{soma.center[2]:.0f}], "
            f"axes=[{soma.axes[0]:.0f}, {soma.axes[1]:.0f}, "
            f"{soma.axes[2]:.0f}]"
        )

    return soma


# Backward-compat wrapper — will be removed in a future release.


def _deprecated_alias(old_name: str, new_func):
    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"sk.pre.{old_name}() is deprecated and will be removed; "
            f"use {new_func.__name__}() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_func(*args, **kwargs)

    return wrapper


find_soma_via_ring_cutoff = _deprecated_alias("find_soma_via_ring_cutoff", find_soma)
