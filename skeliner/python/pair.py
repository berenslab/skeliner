"""``skeliner.pair`` on the Python path: the faces of one mesh that touch
another, and how far rays travel before they first hit a mesh.

Rays use trimesh's ``ray_triangle`` hit test and nearest points Ericson's
region test (Real-Time Collision Detection, 5.1.5), computed as the Rust
path computes them: products rounded, then summed left to right. Candidate
triangles come from a KD-tree over their centroids — a triangle within *r*
of a point has its centroid within *r* plus the largest centroid-to-vertex
distance — and every candidate is tested exactly, so the answers are the
Rust path's.
"""

from __future__ import annotations

import math

import numpy as np

from .mesh import total_key
from scipy.spatial import cKDTree


def _dd(a, b):
    """trimesh's ``diagonal_dot``: rowwise products, summed left to right."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _triangles(triangles) -> np.ndarray:
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 2 or tris.shape[1] != 9:
        raise ValueError("triangles must be (F, 9)")
    return tris.reshape(-1, 3, 3)


def _rows3(a, what: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{what} must be (N, 3)")
    return a


def hit_distances(o, d, tris, normals) -> np.ndarray:
    """Where the ray from *o* along *d* hits each triangle (normals
    *normals*), as the ray parameter; NaN for the triangles it misses.

    A triangle is hit when the ray crosses its plane (the direction not
    within 1e-5 of parallel to it), the crossing's barycentric coordinates
    (Cramer's rule) all lie within 1e-13 of [0, 1], and the crossing is no
    more than 1e-6 behind the origin.
    """
    proj_ori = _dd(tris[:, 0] - o, normals)
    proj_dir = _dd(d[None, :], normals)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = proj_ori / proj_dir
        p = d[None, :] * distance[:, None] + o[None, :]
        e0 = tris[:, 1] - tris[:, 0]
        e1 = tris[:, 2] - tris[:, 0]
        w = p - tris[:, 0]
        d00, d01, d02 = _dd(e0, e0), _dd(e0, e1), _dd(e0, w)
        d11, d12 = _dd(e1, e1), _dd(e1, w)
        inv = 1.0 / (d00 * d11 - d01 * d01)
        b2 = (d00 * d12 - d01 * d02) * inv
        b1 = (d11 * d02 - d01 * d12) * inv
        b0 = 1.0 - b1 - b2
        zero = 1e-13
        inside = np.ones(len(tris), dtype=bool)
        for b in (b0, b1, b2):
            inside &= (b > -zero) & (b < 1.0 + zero)
        forward = _dd(p - o[None, :], d[None, :]) > -1e-6
    hit = (np.abs(proj_dir) > 1e-5) & inside & forward
    return np.where(hit, distance, np.nan)


def closest_point_on_triangle(p, t) -> np.ndarray:
    """The point of triangle *t* (rows a, b, c) closest to *p*."""
    a, b, c = t
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dd(ab, ap), _dd(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3, d4 = _dd(ab, bp), _dd(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = _dd(ab, cp), _dd(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w
    denom = va + vb + vc
    if denom == 0.0 or not math.isfinite(denom):
        # a degenerate triangle: the nearest point of its edges
        best = None
        for u, v in ((a, b), (b, c), (c, a)):
            e = v - u
            length2 = _dd(e, e)
            s = _dd(p - u, e) / length2 if length2 > 0.0 else 0.0
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            q = u + e * s
            key = total_key(float(_dd(p - q, p - q)))
            if best is None or key < best[0]:
                best = (key, q)
        return best[1]
    return a + (ab * (vb / denom) + ac * (vc / denom))


def _squared_distances(p, tris, ids) -> list[tuple[int, float]]:
    out = []
    for t in ids:
        q = closest_point_on_triangle(p, tris[t])
        out.append((int(t), float(_dd(q - p, q - p))))
    return out


class _Centroids:
    """A KD-tree over triangle centroids that returns, for a point and a
    radius, every triangle that could lie within that radius."""

    def __init__(self, tris: np.ndarray):
        self.centroids = tris.mean(axis=1)
        reach = np.linalg.norm(tris - self.centroids[:, None, :], axis=2).max(axis=1)
        self.reach = float(reach.max()) if len(tris) else 0.0
        self.tree = cKDTree(self.centroids) if len(tris) else None

    def near(self, p, r: float) -> list[int]:
        if self.tree is None:
            return []
        # the margin covers the rounding of the tree's own distances
        found = self.tree.query_ball_point(p, r + self.reach + 1e-9 * (1 + r))
        return sorted(found)


def contact_faces(centers, normals, triangles, other_normals, tol, max_dot):
    """Which faces of one mesh touch another: per face, the other mesh's
    triangle closest to its *center* — of several equally close (meeting at
    the closest point), the one whose normal most opposes the face's, then
    the lowest index — kept when it lies within *tol* and ``dot <= max_dot``.
    Returns ``(faces, other_faces, distances)`` for the faces kept.
    *triangles* are the other mesh's ``(G, 3, 3)`` flattened to ``(G, 9)``.
    """
    centers = _rows3(centers, "centers")
    normals = _rows3(normals, "normals")
    if len(normals) != len(centers):
        raise ValueError("one normal per face")
    tris = _triangles(triangles)
    other_normals = _rows3(other_normals, "other normals")
    if len(other_normals) != len(tris):
        raise ValueError("one normal per triangle")
    tol = float(tol)
    index = _Centroids(tris)
    faces, others, dist = [], [], []
    if not (tol >= 0.0):  # NaN or negative: nothing is within it
        return (np.array(faces, np.int64), np.array(others, np.int64), np.array(dist))
    limit = tol * tol
    for f, (c, n) in enumerate(zip(centers, normals)):
        near = [(t, d2) for t, d2 in _squared_distances(c, tris, index.near(c, tol))]
        near = [(t, d2) for t, d2 in near if d2 <= limit]
        if not near:
            continue
        t0, d2 = min(near, key=lambda td: (td[1], td[0]))
        # the triangles that share the closest distance, rounding aside
        tied = d2 + 1e-8
        r = math.sqrt(tied)
        within = r * r
        cands = [
            (t, e2)
            for t, e2 in _squared_distances(c, tris, index.near(c, r))
            if e2 <= within and e2 <= tied
        ]
        if not cands:
            continue
        best_dot, best = min(
            ((float(_dd(n, other_normals[t])), t) for t, _ in cands),
            key=lambda dt: (total_key(dt[0]), dt[1]),
        )
        if best_dot <= max_dot:
            faces.append(f)
            others.append(best)
            dist.append(math.sqrt(d2))
    return (
        np.array(faces, dtype=np.int64),
        np.array(others, dtype=np.int64),
        np.array(dist, dtype=np.float64),
    )


class TriangleBvh:
    """Triangles (``(F, 3, 3)`` flattened to ``(F, 9)``, *normals* their
    face normals), indexed once to cast many rays at, with trimesh's
    ``ray_triangle`` test."""

    def __init__(self, triangles, normals):
        self.tris = _triangles(triangles)
        self.normals = _rows3(normals, "normals")
        if len(self.normals) != len(self.tris):
            raise ValueError("one normal per triangle")
        self._index = _Centroids(self.tris)
        if len(self.tris):
            self._span = float(np.linalg.norm(np.ptp(self.tris.reshape(-1, 3), axis=0)))
        else:
            self._span = 0.0

    def _first_hit(self, o, d, min_t: float, skip: int | None):
        """``(triangle, ray parameter)`` of the first hit past *min_t*,
        leaving out *skip*; ties to the lowest index; None for no hit."""
        norm_d = float(np.linalg.norm(d))
        if not len(self.tris) or not (norm_d > 0.0):
            return None
        radius = max(self._index.reach, 1e-12)
        while True:
            # a hit at ray parameter h lies h * |d| from the origin
            ids = self._index.near(o, radius)
            ids = [t for t in ids if t != skip]
            best = None
            if ids:
                h = hit_distances(o, d, self.tris[ids], self.normals[ids])
                for t, ht in zip(ids, h):
                    if not (ht > min_t):  # a miss (NaN) or not past min_t
                        continue
                    if best is None or ht < best[1] or (ht == best[1] and t < best[0]):
                        best = (t, float(ht))
            # every hit up to the best one lies within the radius searched
            if best is not None and max(abs(best[1]), abs(min_t)) * norm_d <= radius:
                return best
            if radius > 2.0 * (self._span + float(np.linalg.norm(o - self.tris[0, 0]))):
                return best
            radius *= 2.0

    def first_hits(self, origins, directions, skip, min_t):
        """How far each ray travels before it first hits a triangle: per ray
        from *origins* along *directions*, the distance (ray parameter) to the
        first triangle past *min_t*, leaving out triangle *skip* (the one the
        ray starts on; -1 for none); NaN when it hits nothing."""
        o = _rows3(origins, "origins")
        d = _rows3(directions, "directions")
        skip = np.asarray(skip, dtype=np.int64)
        if len(o) != len(d) or len(o) != len(skip):
            raise ValueError("one direction and one skip per origin")
        out = np.full(len(o), np.nan)
        for k in range(len(o)):
            s = int(skip[k]) if skip[k] >= 0 else None
            hit = self._first_hit(o[k], d[k], float(min_t), s)
            if hit is not None:
                out[k] = hit[1]
        return out


# ---------------------------------------------------------------------
# the seeded path's nearest points (trimesh's ProximityQuery.on_surface)
# ---------------------------------------------------------------------
_TRIMESH_ZERO = 1e-13
_TRIMESH_MERGE = 1e-8


def closest_point_trimesh(tris: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """trimesh's ``triangles.closest_point``: the point of each triangle
    ``(n, 3, 3)`` closest to each point ``(n, 3)`` — Ericson's regions tested
    against trimesh's zero (1e-13) in trimesh's order, every dot product
    summed in axis order, the face case through ``1 / (va + vb + vc)``."""
    z = _TRIMESH_ZERO
    result = np.zeros_like(pts)
    remain = np.ones(len(pts), dtype=bool)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac, ap = b - a, c - a, pts - a
    d1, d2 = _dd(ab, ap), _dd(ac, ap)
    is_a = (d1 < z) & (d2 < z)
    result[is_a] = a[is_a]
    remain[is_a] = False
    bp = pts - b
    d3, d4 = _dd(ab, bp), _dd(ac, bp)
    is_b = (d3 > -z) & (d4 <= d3) & remain
    result[is_b] = b[is_b]
    remain[is_b] = False
    vc = d1 * d4 - d3 * d2
    is_ab = (vc < z) & (d1 > -z) & (d3 < z) & remain
    v = (d1[is_ab] / (d1[is_ab] - d3[is_ab]))[:, None]
    result[is_ab] = a[is_ab] + v * ab[is_ab]
    remain[is_ab] = False
    cp = pts - c
    d5, d6 = _dd(ab, cp), _dd(ac, cp)
    is_c = (d6 > -z) & (d5 <= d6) & remain
    result[is_c] = c[is_c]
    remain[is_c] = False
    vb = d5 * d2 - d1 * d6
    is_ac = (vb < z) & (d2 > -z) & (d6 < z) & remain
    w = (d2[is_ac] / (d2[is_ac] - d6[is_ac]))[:, None]
    result[is_ac] = a[is_ac] + w * ac[is_ac]
    remain[is_ac] = False
    va = d3 * d6 - d5 * d4
    is_bc = (va < z) & ((d4 - d3) > -z) & ((d5 - d6) > -z) & remain
    d43 = d4[is_bc] - d3[is_bc]
    w = (d43 / (d43 + (d5[is_bc] - d6[is_bc])))[:, None]
    result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
    remain[is_bc] = False
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
    v = (vb[remain] * denom)[:, None]
    w = (vc[remain] * denom)[:, None]
    result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)
    return result


def closest_points(vertices, faces, normals, points):
    """trimesh's ``ProximityQuery.on_surface`` of the mesh *vertices*,
    *faces* with its face *normals*, for each of *points*: ``(closest points,
    distances, faces)``, each face's nearest point by
    :func:`closest_point_trimesh`. Where faces tie — squared distances
    within 1e-8 of the nearest, itself over 1e-8 — the one whose normal best
    faces the point wins (trimesh's rule, over every tied face rather than
    the two its sort puts first), then the nearer, then the lower id."""
    vertices = _rows3(vertices, "vertices")
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces name vertices out of range")
    normals = _rows3(normals, "normals")
    if len(normals) != len(faces):
        raise ValueError("one normal per face")
    pts = _rows3(points, "points")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")
    if not len(faces) and len(pts):
        raise ValueError("the mesh has no faces")
    tris = vertices[faces]
    index = _Centroids(tris)
    # every face that may tie with the nearest: the nearest is no farther
    # than the nearest centroid, a point of the surface
    d0 = index.tree.query(pts)[0] if len(pts) else np.empty(0)
    radius = np.sqrt(d0 * d0 + _TRIMESH_MERGE) * (1 + 1e-12)
    groups = [index.near(p, float(r)) for p, r in zip(pts, radius, strict=True)]
    owner = np.repeat(np.arange(len(pts)), [len(g) for g in groups])
    cand = (
        np.concatenate([np.asarray(g, np.int64) for g in groups])
        if groups
        else (np.empty(0, np.int64))
    )
    q = closest_point_trimesh(tris[cand], pts[owner])
    vec = pts[owner] - q
    d2 = _dd(vec, vec)
    measured = ~np.isnan(d2)
    best = np.full(len(pts), np.inf)
    np.minimum.at(best, owner[measured], d2[measured])
    tied = measured & (d2 - best[owner] < _TRIMESH_MERGE)
    facing = np.zeros(len(cand))
    lean = tied & (best[owner] > _TRIMESH_MERGE)
    facing[lean] = _dd(normals[cand[lean]], vec[lean] / np.sqrt(d2[lean])[:, None])
    # every distance NaN: the lowest face, unmeasured
    unmeasured = ~np.isin(np.arange(len(pts)), owner[tied])
    keep = tied | (unmeasured[owner] & ~measured)
    order = np.lexsort((cand, d2, -facing, owner))
    order = order[keep[order]]
    first = order[np.r_[True, owner[order][1:] != owner[order][:-1]]]
    out = np.empty(len(pts), dtype=np.int64)
    out[owner[first]] = first
    dist = np.where(np.isnan(d2[out]), d2[out], np.sqrt(d2[out]))
    return q[out], dist, cand[out]


def face_reach(vertices, faces) -> float:
    """The farthest any face's corner lies from its centre (``(0 + a + b +
    c) / 3``, as trimesh places it), over the mesh *vertices* and *faces*;
    0 without faces, NaN when a coordinate is NaN."""
    vertices = _rows3(vertices, "vertices")
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces name vertices out of range")
    if not len(faces):
        return 0.0
    tris = vertices[faces]
    center = (((0.0 + tris[:, 0]) + tris[:, 1]) + tris[:, 2]) / 3.0
    d = tris - center[:, None]
    return float(np.sqrt(_dd(d, d)).max())
