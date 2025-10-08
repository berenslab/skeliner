"""skeliner.batch – helpers for querying *multiple* skeletons at once."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import dx

if TYPE_CHECKING:
    from .core import Skeleton

__all__ = ["distance_matrix", "nearest_skeletons"]

StructuredMatches = Dict[str, Any] | List[Dict[str, Any]]


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


def _ensure_skeletons(
    skeletons: Iterable["Skeleton"],
) -> Tuple[Tuple["Skeleton", ...], int]:
    """Freeze iterable of skeletons and validate that each has nodes."""
    skels = tuple(skeletons)
    if not skels:
        raise ValueError("At least one skeleton is required")
    for idx, skel in enumerate(skels):
        if skel.nodes.size == 0:
            raise ValueError(f"Skeleton at index {idx} has no nodes")
    return skels, len(skels)


def _build_structured_matches(
    *,
    points: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    skeletons: Sequence["Skeleton"],
    single_point: bool,
    id_field: str,
    all_distances: np.ndarray | None = None,
) -> StructuredMatches:
    """Create a user friendly summary of nearest skeleton matches."""

    dist_rows = np.atleast_2d(distances)
    idx_rows = np.atleast_2d(indices)
    all_rows = np.atleast_2d(all_distances) if all_distances is not None else None

    point_matches: List[Dict[str, Any]] = []
    for i, point in enumerate(points):
        matches: List[Dict[str, Any]] = []
        for dist, idx in zip(dist_rows[i], idx_rows[i], strict=False):
            skel = skeletons[int(idx)]
            meta = getattr(skel, "meta", {}) or {}
            matches.append(
                {
                    "skeleton_index": int(idx),
                    "skeleton_id": meta.get(id_field),
                    "distance": float(dist),
                }
            )
        entry: Dict[str, Any] = {
            "point_index": i,
            "point": point.tolist(),
            "matches": matches,
        }
        if all_rows is not None:
            entry["all_distances"] = all_rows[i].tolist()
        point_matches.append(entry)

    return point_matches[0] if single_point else point_matches


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
    structured: bool = True,
    id_field: str = "id",
) -> (
    Tuple[np.ndarray, np.ndarray]
    | Tuple[np.ndarray, np.ndarray, np.ndarray]
    | StructuredMatches
):
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
    structured
        When *True* (default) return a human-friendly summary that lists the nearest
        skeletons per point together with their meta IDs instead of the raw arrays.
        Pass ``False`` to preserve the legacy tuple return style.
    id_field
        Key looked up on :attr:`Skeleton.meta` when building the structured
        summary. Only consulted when ``structured`` is *True*.

    Returns
    -------
    distances, indices [, all_distances]
        When ``structured`` is *False* distances and skeleton indices of shape
        ``(k,)`` for a single point or ``(M, k)`` for multiple points. ``indices``
        refer to the order of ``skeletons``. When ``return_all`` is *True* the full
        distance matrix (``(S,)`` or ``(M, S)``) is appended to the return tuple.
    structured
        When ``structured`` is *True* the function returns a dictionary (single point)
        or a list of dictionaries (multiple points) describing the closest skeletons.
        Each dictionary includes the matched skeleton IDs, indices, and distances. When
        ``return_all`` is *True* each entry also contains an ``"all_distances"`` list
        covering the distances to every skeleton in the input.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    pts, single_input = _ensure_points(points)
    skels, n_skels = _ensure_skeletons(skeletons)

    dist = distance_matrix(
        skels,
        pts[0] if single_input else pts,
        point_unit=point_unit,
        k_nearest=k_nearest,
        radius_metric=radius_metric,
        mode=mode,
    )

    k = min(k, n_skels)

    single = dist.ndim == 1
    if single:
        if k == n_skels:
            idx = np.arange(n_skels, dtype=np.int64)
            dists = np.asarray(dist, dtype=np.float64)
        else:
            part = np.argpartition(dist, k - 1)[:k]
            dists = dist[part]
            order = np.argsort(dists)
            idx = part[order]
            dists = dists[order]
    else:
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

    if structured:
        structured_matches = _build_structured_matches(
            points=pts,
            distances=dists,
            indices=idx,
            skeletons=skels,
            single_point=single,
            id_field=id_field,
            all_distances=dist if return_all else None,
        )
        return structured_matches

    result: Tuple[np.ndarray, ...] = (dists, idx)
    if return_all:
        result = result + (dist,)
    return result
