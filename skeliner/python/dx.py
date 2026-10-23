"""``skeliner.dx`` on the Python path: distances from points to a skeleton."""

from __future__ import annotations

import numpy as np


def point_segment_distance(point, start, end) -> float:
    """Euclidean distance from *point* to the segment [*start*, *end*]."""
    vec = end - start
    seg_len2 = float(np.dot(vec, vec))
    if seg_len2 <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, vec) / seg_len2)
    t = min(1.0, max(0.0, t))  # a NaN becomes 0
    closest = start + t * vec
    return float(np.linalg.norm(point - closest))


def segment_distances(points, nodes, centres, segments) -> np.ndarray:
    """Per point, the distance to the nearest of the *centres* (node ids) or
    of the *segments* (node-id pairs); infinity when there are neither.

    ``skeliner.dx.distance`` in centreline mode against a whitelist.
    """
    points = np.asarray(points, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    ids = [*centres, *(i for pair in segments for i in pair)]
    if any(i >= len(nodes) for i in ids):
        raise ValueError("node index out of range")
    out = np.empty(len(points), dtype=np.float64)
    for k, p in enumerate(points):
        best = float("inf")
        for c in centres:
            d = float(np.linalg.norm(nodes[c] - p))
            if d < best:
                best = d
        for a, b in segments:
            d = point_segment_distance(p, nodes[a], nodes[b])
            if d < best:
                best = d
        out[k] = best
    return out
