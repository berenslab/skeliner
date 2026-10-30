"""``skeliner.dataclass`` on the Python path: the soma's ellipsoid
(``Soma``'s measures) and the arrays a skeleton keeps in step
(``skeliner._state``)."""

from __future__ import annotations

import numpy as np

from .mesh import node_pairs, rows3

#: Eberly's bound on the halvings of a bisection: as many as a double has
#: values between two of its powers of two, down to the subnormals.
MAX_BISECTIONS = 1074


def soma_contains(center, axes, rotation, points, limit: float) -> np.ndarray:
    """``Soma.contains``: which of *points* ``(N, 3)`` lie inside the
    ellipsoid (*center*, *axes*, *rotation*), their body coordinates'
    squared length at most *limit*."""
    points = rows3("N", points)
    w = np.asarray(rotation, dtype=np.float64) / np.asarray(axes, dtype=np.float64)
    xi = (points - np.asarray(center, dtype=np.float64)) @ w
    return (xi**2).sum(axis=-1) <= limit


def soma_distance_to_center(center, points) -> np.ndarray:
    """``Soma.distance_to_center``: each of *points* ``(N, 3)``'s distance
    to *center*."""
    points = rows3("N", points)
    return np.linalg.norm(points - np.asarray(center, dtype=np.float64), axis=1)


def soma_distance_to_surface(
    center, axes, rotation, points, atol: float, max_iter: int
) -> np.ndarray:
    """``Soma.distance_to_surface``: each of *points* ``(N, 3)``'s signed
    Euclidean distance to the ellipsoid's surface, negative inside.

    Outside (``|body / axes|^2 > 1 + 1e-12``), Newton's iteration on the
    Lagrange multiplier *t* of the nearest point, one loop for all of them,
    stopped once every step is below *atol* (*max_iter* steps at most);
    inside, Eberly's bisection (:func:`inside_distance`); within *atol* of
    the centre, in body units, the smallest semi-axis."""
    points = rows3("N", points)
    a = np.asarray(axes, dtype=np.float64)
    # --- body-coordinates: align to principal axes --------------------
    p = (points - np.asarray(center, dtype=np.float64)) @ np.asarray(
        rotation, dtype=np.float64
    )
    a2 = a * a
    r2 = (p**2 / a2).sum(axis=1)  # ‖p‖² in unit-sphere space
    out = r2 > 1.0 + 1e-12
    dist = np.empty(len(p), dtype=np.float64)

    # ---------------- OUTSIDE points  ---------------------------------
    if out.any():
        po = p[out]
        t = np.zeros(len(po))
        for _ in range(max_iter):
            denom = t[:, None] + a2
            f = (a2 * po**2 / denom**2).sum(1) - 1.0
            fp = (-2.0 * a2 * po**2 / denom**3).sum(1)
            dt = -f / fp
            t += dt
            if np.all(np.abs(dt) < atol):
                break
        xs = a2 * po / (t[:, None] + a2)  # nearest surface points
        dist[out] = np.linalg.norm(xs - po, axis=1)

    # ---------------- INSIDE points  ----------------------------------
    inn = np.flatnonzero(~out)
    if len(inn):
        nz = np.sqrt(r2[inn]) > atol  # not at the centre
        dist[inn[nz]] = -inside_distance(a, np.abs(p[inn[nz]]))
        # the centre → the shortest half-axis
        dist[inn[~nz]] = -a.min()
    return dist


def inside_distance(e, y) -> np.ndarray:
    """Each of *y* ``(K, 3)``'s distance to the ellipsoid of semi-axes *e*,
    largest first, *y* in its frame and in its first octant (``y >= 0``):
    Eberly's robust bisection ("Distance from a point to an ellipse, an
    ellipsoid, or a hyperellipsoid", Geometric Tools), which needs no
    tolerance — it halves until no double lies between its ends.  A
    coordinate of *y* that is zero leaves the ellipse of the other two, or
    the foot of a perpendicular off the plane."""
    e0, e1, e2 = (float(v) for v in e)
    y0, y1, y2 = (np.ascontiguousarray(y[:, k], dtype=np.float64) for k in range(3))
    d = np.empty(len(y0))
    p0, p1, p2 = y0 > 0, y1 > 0, y2 > 0

    m = p2 & p1 & p0
    z0, z1, z2 = y0[m] / e0, y1[m] / e1, y2[m] / e2
    g = z0 * z0 + z1 * z1 + z2 * z2 - 1.0
    q0, q1 = e0 / e2, e1 / e2
    r0, r1 = q0 * q0, q1 * q1
    s = _bisect([r0 * z0, r1 * z1], [r0, r1], z2, g)
    x0, x1, x2 = r0 * y0[m] / (s + r0), r1 * y1[m] / (s + r1), y2[m] / (s + 1.0)
    d0, d1, d2 = x0 - y0[m], x1 - y1[m], x2 - y2[m]
    d[m] = np.where(g == 0.0, 0.0, np.sqrt(d0 * d0 + d1 * d1 + d2 * d2))

    m = p2 & p1 & ~p0
    d[m] = ellipse_distance(e1, e2, y1[m], y2[m])
    m = p2 & ~p1 & p0
    d[m] = ellipse_distance(e0, e2, y0[m], y2[m])
    m = p2 & ~p1 & ~p0
    d[m] = np.abs(y2[m] - e2)

    # on the plane of the two longer axes: the foot of a perpendicular off
    # it where there is one, else the ellipse in it
    m = np.flatnonzero(~p2)
    denom0, denom1 = e0 * e0 - e2 * e2, e1 * e1 - e2 * e2
    numer0, numer1 = e0 * y0[m], e1 * y1[m]
    cand = np.flatnonzero((numer0 < denom0) & (numer1 < denom1))
    xde0, xde1 = numer0[cand] / denom0, numer1[cand] / denom1
    discr = 1.0 - xde0 * xde0 - xde1 * xde1
    ok = discr > 0.0
    foot = m[cand[ok]]
    x0, x1 = e0 * xde0[ok], e1 * xde1[ok]
    x2 = e2 * np.sqrt(discr[ok])
    d0, d1 = x0 - y0[foot], x1 - y1[foot]
    d[foot] = np.sqrt(d0 * d0 + d1 * d1 + x2 * x2)
    rest = np.setdiff1d(m, foot, assume_unique=True)
    d[rest] = ellipse_distance(e0, e1, y0[rest], y1[rest])
    return d


def ellipse_distance(e0: float, e1: float, y0, y1) -> np.ndarray:
    """:func:`inside_distance` in two dimensions: each point to the ellipse
    of semi-axes ``e0 >= e1``."""
    d = np.empty(len(y0))
    p0, p1 = y0 > 0, y1 > 0

    m = p1 & p0
    z0, z1 = y0[m] / e0, y1[m] / e1
    g = z0 * z0 + z1 * z1 - 1.0
    q0 = e0 / e1
    r0 = q0 * q0
    s = _bisect([r0 * z0], [r0], z1, g)
    d0, d1 = r0 * y0[m] / (s + r0) - y0[m], y1[m] / (s + 1.0) - y1[m]
    d[m] = np.where(g == 0.0, 0.0, np.sqrt(d0 * d0 + d1 * d1))

    m = p1 & ~p0
    d[m] = np.abs(y1[m] - e1)

    m = np.flatnonzero(~p1)
    numer0, denom0 = e0 * y0[m], e0 * e0 - e1 * e1
    inner = numer0 < denom0
    xde0 = numer0[inner] / denom0
    x0, x1 = e0 * xde0, e1 * np.sqrt(1.0 - xde0 * xde0)
    d0 = x0 - y0[m[inner]]
    d[m[inner]] = np.sqrt(d0 * d0 + x1 * x1)
    d[m[~inner]] = np.abs(y0[m[~inner]] - e0)
    return d


def _robust_length(v: list) -> np.ndarray:
    """``|v|`` of the vectors whose coordinates are the arrays *v*, scaled by
    the largest magnitude so no square overflows."""
    m = np.zeros_like(v[0])
    for x in v:
        m = np.fmax(m, np.abs(x))
    total = np.zeros_like(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        for x in v:
            total = total + (x / m) * (x / m)
        return np.where(m > 0.0, m * np.sqrt(total), 0.0)


def _bisect(n: list, r: list, z, g) -> np.ndarray:
    """For each point, the root of ``sum_i (n_i / (s + r_i))^2 + (z / (s +
    1))^2 = 1`` that :func:`inside_distance` bisects for, the bracket from
    *g*'s sign; every point halves its own bracket until no double lies
    between its ends or it lands on the root."""
    lo = z - 1.0
    hi = np.where(g < 0.0, 0.0, _robust_length([*n, z]) - 1.0)
    s = np.zeros_like(z)
    live = np.ones(z.shape, dtype=bool)
    for _ in range(MAX_BISECTIONS):
        if not live.any():
            break
        mid = (lo + hi) / 2.0
        s = np.where(live, mid, s)
        live &= (mid != lo) & (mid != hi)
        total = np.zeros_like(mid)
        for ni, ri in zip(n, r):
            ratio = ni / (mid + ri)
            total = total + ratio * ratio
        ratio = z / (mid + 1.0)
        gm = total + ratio * ratio - 1.0
        lo = np.where(live & (gm > 0.0), mid, lo)
        hi = np.where(live & (gm < 0.0), mid, hi)
        live &= (gm > 0.0) | (gm < 0.0)
    return s


def node_neighbors(edges, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """``Skeleton``'s adjacency: each of *n_nodes* nodes' neighbours as a
    CSR ``(ptr, idx)``, in the order *edges* ``(E, 2)`` name them, an edge's
    first end before its second (a self-loop lists its node twice)."""
    e = node_pairs(edges, n_nodes)
    order = np.argsort(e.reshape(-1), kind="stable")
    idx = e[:, ::-1].reshape(-1)[order]
    ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(e.reshape(-1), minlength=n_nodes), out=ptr[1:])
    return ptr, idx


def remap_edges(edges, old2new) -> np.ndarray:
    """``_state.remap_edges``: the *edges* ``(E, 2)`` between the nodes
    *old2new* keeps (``>= 0``), renumbered by it, each sorted low to high,
    without self-loops, unique and in order."""
    old2new = np.asarray(old2new, dtype=np.int64).reshape(-1)
    remapped = old2new[node_pairs(edges, len(old2new))]
    remapped = remapped[(remapped[:, 0] >= 0) & (remapped[:, 1] >= 0)]
    remapped = remapped[remapped[:, 0] != remapped[:, 1]]
    remapped.sort(axis=1)
    return np.unique(remapped, axis=0).reshape(-1, 2)


def vertex_map(
    ptr, idx, n_vertices: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``vert2node`` from the bins (a CSR *ptr*, *idx*, nodes in order):
    ``(keys, node)`` — each vertex the first time a bin lists it, and by
    vertex the node of the last bin listing it (-1 for none), *n_vertices*
    long (one past the highest listed when not given). The dict
    ``zip(vertices, nodes)`` makes, as arrays."""
    ptr = np.asarray(ptr, dtype=np.int64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if not (
        ptr.size and ptr[0] == 0 and (np.diff(ptr) >= 0).all() and ptr[-1] == idx.size
    ):
        raise ValueError("the bins must be a CSR (ptr, idx)")
    if (idx < 0).any():
        raise ValueError("vertex ids must not be negative")
    if n_vertices is None:
        n_vertices = int(idx.max()) + 1 if idx.size else 0
    elif idx.size and idx.max() >= n_vertices:
        v = int(idx[np.argmax(idx >= n_vertices)])
        raise ValueError(
            f"the bins name vertex {v}, but there are {n_vertices} vertices"
        )
    nodes = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    mapping = dict(zip(idx.tolist(), nodes.tolist()))
    keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    node = np.full(n_vertices, -1, dtype=np.int64)
    node[keys] = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    return keys, node
