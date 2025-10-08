"""skeliner.batch – helpers for querying *multiple* skeletons at once."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence, Tuple

import numpy as np

from . import dx
if TYPE_CHECKING:
    from .core import Skeleton

__all__ = ["distance_matrix", "nearest_skeletons"]


def _ensure_points(points: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, bool]:
    """Normalise point input to a (M, 3) float64 array."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        if pts.shape[0] != 3:
            raise ValueError("points must be a 3-vector or an array of shape (M, 3)")
        pts = pts[None, :]
        return pts, True
    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts, False
    raise ValueError("points must be a 3-vector or an array of shape (M, 3)")


def _ensure_skeletons(skeletons: Iterable["Skeleton"]) -> Tuple[Tuple["Skeleton", ...], int]:
    """Freeze iterable of skeletons and validate that each has nodes."""
    skels = tuple(skeletons)
    if not skels:
        raise ValueError("At least one skeleton is required")
    for idx, skel in enumerate(skels):
        if skel.nodes.size == 0:
            raise ValueError(f"Skeleton at index {idx} has no nodes")
    return skels, len(skels)


def distance_matrix(
    skeletons: Iterable["Skeleton"],
    points: Sequence[float] | np.ndarray,
    *,
    point_unit: str | None = None,
    k_nearest: int = 4,
    radius_metric: str | None = None,
    mode: str = "surface",
) -> np.ndarray:
    """
    Distance from one or more points to *each* skeleton in ``skeletons``.

    Parameters
    ----------
    skeletons
        Iterable of :class:`skeliner.Skeleton` objects to query.
    points
        Query location(s) as a single 3-vector or an array of shape (M, 3).
    point_unit
        Unit of the input coordinates and returned distances. When ``None``
        the underlying :func:`skeliner.dx.distance` helper performs no unit
        conversion.
    k_nearest
        Passed through to :func:`skeliner.dx.distance`.
    radius_metric
        Radius column name forwarded to :func:`skeliner.dx.distance`.
    mode
        Either ``"surface"`` (default) or ``"centerline"`` – same semantics
        as :func:`skeliner.dx.distance`.

    Returns
    -------
    ndarray
        Array of shape ``(M, S)`` with distances in ``point_unit`` where
        ``S`` is the number of skeletons and ``M`` the number of query points.
        A single input point returns an array of shape ``(S,)``.
    """
    pts, single = _ensure_points(points)
    skels, n_skels = _ensure_skeletons(skeletons)

    distances = np.empty((pts.shape[0], n_skels), dtype=np.float64)

    for j, skel in enumerate(skels):
        d = dx.distance(
            skel,
            pts,
            point_unit=point_unit,
            k_nearest=k_nearest,
            radius_metric=radius_metric,
            mode=mode,
        )
        distances[:, j] = d  # d is (M,)

    return distances[0] if single else distances


def nearest_skeletons(
    skeletons: Iterable["Skeleton"],
    points: Sequence[float] | np.ndarray,
    *,
    k: int = 1,
    point_unit: str | None = None,
    k_nearest: int = 4,
    radius_metric: str | None = None,
    mode: str = "surface",
    return_all: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Query the *k* closest skeletons for each point.

    Parameters
    ----------
    skeletons
        Iterable of :class:`skeliner.Skeleton` objects.
    points
        Single 3-vector or array of shape (M, 3).
    k
        Number of closest skeletons to return per point. Clamped to the
        number of available skeletons.
    point_unit, k_nearest, radius_metric, mode
        Forwarded to :func:`distance_matrix`.
    return_all
        When *True* also return the full distance matrix.

    Returns
    -------
    distances, indices [, all_distances]
        Distances and skeleton indices of shape ``(k,)`` for a single point
        or ``(M, k)`` for multiple points. ``indices`` refer to the order of
        ``skeletons``. When ``return_all`` is *True* the full distance matrix
        (``(S,)`` or ``(M, S)``) is appended to the return tuple.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    skels, n_skels = _ensure_skeletons(skeletons)

    dist = distance_matrix(
        skels,
        points,
        point_unit=point_unit,
        k_nearest=k_nearest,
        radius_metric=radius_metric,
        mode=mode,
    )

    k = min(k, n_skels)

    single = dist.ndim == 1
    if single:
        # 1D case: distances per skeleton
        if k == n_skels:
            idx = np.arange(n_skels, dtype=np.int64)
            dists = np.asarray(dist, dtype=np.float64)
        else:
            part = np.argpartition(dist, k - 1)[:k]
            dists = dist[part]
            order = np.argsort(dists)
            idx = part[order]
            dists = dists[order]

        return (dists, idx) + ((dist,) if return_all else ())

    # M>1 points – work row-wise
    if k == n_skels:
        order = np.argsort(dist, axis=1)
        idx = np.tile(np.arange(n_skels, dtype=np.int64), (dist.shape[0], 1))
        idx = np.take_along_axis(idx, order, axis=1)
        dists = np.take_along_axis(dist, order, axis=1)
    else:
        part_idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist, part_idx, axis=1)
        order = np.argsort(part_dist, axis=1)
        idx = np.take_along_axis(part_idx, order, axis=1)
        dists = np.take_along_axis(part_dist, order, axis=1)
        return (dists, idx) + ((dist,) if return_all else ())

    return (dists, idx) + ((dist,) if return_all else ())
