from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from .core import Skeleton


@dataclass(slots=True)
class ContactSet:
    """
    Pairwise geometrical contacts between two Skeletons (node-to-node).
    idx_a, idx_b : (K,) int64  Node indices in A and B.
    pos_a, pos_b : (K,3) float64  Closest points on the node spheres along the line of centers.
    pos         : (K,3) float64  Midpoint between pos_a and pos_b.
    center_gap  : (K,) float64   ||xa-xb|| - (ra+rb).
    meta        : dict           Aux info.
    """
    idx_a: np.ndarray
    idx_b: np.ndarray
    pos_a: np.ndarray
    pos_b: np.ndarray
    pos:   np.ndarray
    center_gap: np.ndarray | None
    meta: dict[str, object]

    @property
    def n(self) -> int:
        return int(len(self.idx_a))

# --- helpers -----------------------------------------------------------

def _empty_contactset(unit: str | None, key: str, delta: float) -> ContactSet:
    z0 = np.zeros(0, np.int64)
    z3 = np.zeros((0, 3), np.float64)
    return ContactSet(
        idx_a=z0, idx_b=z0, pos_a=z3, pos_b=z3, pos=z3,
        center_gap=np.zeros(0, np.float64),
        meta={"unit": unit, "delta": float(delta), "radius_key": key, "engine": "kdtree"}
    )

# --- core: simple, robust node↔node contacts via KD-tree ---------------

def find_contacts(
    A: "Skeleton",
    B: "Skeleton",
    *,
    delta: float = 0.0,
    radius_key: str | None = None,
    exclude_soma: bool = True, 
) -> "ContactSet":
    """
    1) Filter (optional) soma / near-soma nodes.
    2) KD-tree on B to get a cheap superset candidate set per A-node using radius (ra + RB_margin + delta).
    3) Exact pairwise filter: keep pairs with ||xa-xb|| <= ra+rb+delta.
    4) For each A-node keep the nearest valid B-node ("first hit").
    5) Compute pos_a/pos_b on node spheres and the midpoint pos; return center_gap.
    """

    # --- data & radii
    xa = np.asarray(A.nodes, float)
    xb = np.asarray(B.nodes, float)

    if len(xa) == 0 or len(xb) == 0:
        raise ValueError(f"Empty skeletons (len(A)={len(xa)}, len(B)={len(xb)})")

    key = radius_key or A.recommend_radius()[0]
    ra = np.asarray(A.radii[key], float)
    rb = np.asarray(B.radii[key], float)

    # --- optional soma filtering
    mask_a = np.ones(len(xa), bool)
    mask_b = np.ones(len(xb), bool)
    if exclude_soma:
        mask_a[0] = False
        mask_b[0] = False

    idx_a0 = np.where(mask_a)[0]
    idx_b0 = np.where(mask_b)[0]
    XA, RA = xa[idx_a0], ra[idx_a0]
    XB, RB = xb[idx_b0], rb[idx_b0]

    # --- cheap mutual AABB overlap prefilter (pad by robust RB/RA margin)
    padA = float(RB.max()) + float(delta)
    padB = float(RA.max()) + float(delta)

    Alo, Ahi = XA.min(0), XA.max(0)
    Blo, Bhi = XB.min(0), XB.max(0)

    keepA = np.all((XA >= Blo - padA) & (XA <= Bhi + padA), axis=1)
    keepB = np.all((XB >= Alo - padB) & (XB <= Ahi + padB), axis=1)
    if not keepA.any() or not keepB.any():
        return _empty_contactset(A.meta.get("unit"), key, delta)

    XA, RA, idx_a0 = XA[keepA], RA[keepA], idx_a0[keepA]
    XB, RB, idx_b0 = XB[keepB], RB[keepB], idx_b0[keepB]

    if XA.size == 0 or XB.size == 0:
        return _empty_contactset(A.meta.get("unit"), key, delta)

    # --- KD-tree superset search
    tree = KDTree(XB)
    rb_margin = float(RB.max())  # safe, tight-ish
    radii_superset = RA + rb_margin + float(delta)    # per-A query radius

    # query_ball_point supports vector radii when x is an array
    nbrs = tree.query_ball_point(XA, radii_superset)

    # flatten candidate pairs
    counts = np.fromiter((len(js) for js in nbrs), dtype=np.int64, count=len(nbrs))
    if counts.sum() == 0:
        return _empty_contactset(A.meta.get("unit"), key, delta)

    ii = np.repeat(np.arange(XA.shape[0], dtype=np.int64), counts)
    jj = np.concatenate([np.asarray(js, dtype=np.int64) for js in nbrs])

    # exact filter: ||xa - xb|| <= ra + rb + delta
    Ai = XA[ii]
    Bj = XB[jj]
    RAi = RA[ii]
    RBj = RB[jj]

    # squared distances for stability
    d2 = np.einsum('ij,ij->i', Ai - Bj, Ai - Bj)
    thresh = (RAi + RBj + float(delta))**2
    keep = d2 <= thresh
    if not keep.any():
        return _empty_contactset(A.meta.get("unit"), key, delta)

    ii, jj, d2 = ii[keep], jj[keep], d2[keep]

    # first-hit semantics: pick nearest B per A
    order = np.argsort(ii, kind="mergesort")
    ii_s, jj_s, d2_s = ii[order], jj[order], d2[order]
    cut = np.r_[True, ii_s[1:] != ii_s[:-1]]
    starts = np.flatnonzero(cut)
    ends = np.r_[starts[1:], len(ii_s)]

    best_idx = []
    for s, e in zip(starts, ends):
        k = s + np.argmin(d2_s[s:e])
        best_idx.append(k)
    best_idx = np.asarray(best_idx, dtype=np.int64)

    ia_loc = ii_s[best_idx]
    ib_loc = jj_s[best_idx]

    # dedupe in case multiple A map to same B and you want uniqueness of (ia,ib)
    pairs = np.unique(np.stack([ia_loc, ib_loc], axis=1), axis=0)
    ia_loc, ib_loc = pairs[:, 0], pairs[:, 1]

    ia = idx_a0[ia_loc]
    ib = idx_b0[ib_loc]

    # contact loci on node spheres along the line of centers
    ca = xa[ia]
    cb = xb[ib]
    ra_sel = ra[ia]
    rb_sel = rb[ib]

    v = cb - ca
    d = np.linalg.norm(v, axis=1)
    safe = np.maximum(d, 1e-12)

    pa = ca + (ra_sel / safe)[:, None] * v
    pb = cb - (rb_sel / safe)[:, None] * v
    pos = 0.5 * (pa + pb)
    gap = d - (ra_sel + rb_sel)

    return ContactSet(
        idx_a=ia, idx_b=ib,
        pos_a=pa, pos_b=pb, pos=pos,
        center_gap=gap.astype(np.float64),
        meta={"unit": A.meta.get("unit"),
              "delta": float(delta),
              "radius_key": key,
              "exclude_soma": exclude_soma,
        }
    )