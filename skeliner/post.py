
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree as _KDTree

from skeliner.core import Skeleton


def _closest_pair(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """
    Return indices (ia, ib) of the closest points in *a* and *b* plus distance.
    """
    if _KDTree is not None:
        ta = _KDTree(a)
        d, ia = ta.query(b)          # distance from each b-point to nearest a
        ib = int(d.argmin())
        ia = int(ia[ib])
        return ia, ib, float(d[ib])

    # ── SciPy not available – brute force (still fine for few-hundred nodes)
    dmat = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ia, ib = np.unravel_index(dmat.argmin(), dmat.shape)
    return int(ia), int(ib), float(dmat[ia, ib])


def merge(
    skels: Sequence[Skeleton],
    *,
    max_link_distance: float | None = None,
) -> Skeleton:
    """
    Hierarchically merge disconnected skeleton fragments.

    *Iteratively* find the two components whose nearest nodes are closest
    together, connect them with one edge, and repeat until a single skeleton
    remains.  The largest skeleton is **not** privileged beyond being part of
    the distance search, so fragments can attach to each other first.

    Parameters
    ----------
    skels : Sequence[Skeleton]
        At least two disconnected skeletons.
    max_link_distance : float, optional
        If set, raise ``ValueError`` when **no** pair of components is closer
        than this threshold (prevents accidental mega-merges).

    Returns
    -------
    Skeleton
        The unified skeleton.
    """
    if len(skels) == 0:
        raise ValueError("merge() received an empty sequence")
    if len(skels) == 1:
        return skels[0]

    # ── find the largest skeleton and use its soma as reference
    main_idx = max(range(len(skels)), key=lambda i: len(skels[i].nodes))
    soma_ref = skels[main_idx].nodes[0].copy()

    # start with *copies* so we never mutate callers’ objects
    components: list[Skeleton] = [
        Skeleton(s.nodes.copy(), s.radii.copy(), s.edges.copy()) for s in skels
    ]

    while len(components) > 1:
        # ── search all unordered pairs for the minimum inter-component distance
        best = None  # (dist, ia, ib, pa, pb)
        for ia, A in enumerate(components[:-1]):
            for ib, B in enumerate(components[ia + 1 :], start=ia + 1):
                pa, pb, dist = _closest_pair(A.nodes, B.nodes)
                if best is None or dist < best[0]:
                    best = (dist, ia, ib, pa, pb)

        assert best is not None  # len>1 ⇒ we found at least one pair
        dist, ia, ib, pa, pb = best

        if max_link_distance is not None and dist > max_link_distance:
            raise ValueError(
                "No component pairs lie within max_link_distance "
                f"({dist:.1f} > {max_link_distance})."
            )

        # ── merge components[ib] INTO components[ia]  (ia < ib)
        A = components[ia]
        B = components[ib]
        offset = len(A.nodes)

        merged_nodes  = np.vstack([A.nodes,  B.nodes]).astype(np.float32, copy=False)
        merged_radii  = np.concatenate([A.radii, B.radii]).astype(np.float32, copy=False)
        merged_edges  = np.vstack(
            [
                A.edges,
                B.edges + offset,
                np.array([[pa, pb + offset]], dtype=np.int64),  # the graft edge
            ]
        ).astype(np.int64, copy=False)

        components[ia] = Skeleton(merged_nodes, merged_radii, merged_edges)
        components.pop(ib)           # remove the consumed component

    final = components[0]
    soma_idx = int(
        np.linalg.norm(final.nodes - soma_ref[None, :], axis=1).argmin()
    )

    if soma_idx != 0:
        order = np.arange(len(final.nodes))
        order[0], order[soma_idx] = order[soma_idx], order[0]

        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        nodes  = final.nodes[order]
        radii  = final.radii[order]
        edges  = inv[final.edges]          # remap every endpoint

        final = Skeleton(nodes, radii, edges)

    return final
