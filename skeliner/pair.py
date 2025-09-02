"""skeliner.pair – pairwise contact detection between two skeletons and meshes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from scipy.spatial import KDTree

from .core import Skeleton

####################
## Skeleton-based ##
####################

# --- contact seeds ---


@dataclass(slots=True)
class ContactSeeds:
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
    pos: np.ndarray
    center_gap: np.ndarray | None
    meta: dict[str, object]

    @property
    def n(self) -> int:
        return int(len(self.idx_a))


# --- helpers -----------------------------------------------------------


def _empty_contactset(unit: str | None, key: str, delta: float) -> ContactSeeds:
    z0 = np.zeros(0, np.int64)
    z3 = np.zeros((0, 3), np.float64)
    return ContactSeeds(
        idx_a=z0,
        idx_b=z0,
        pos_a=z3,
        pos_b=z3,
        pos=z3,
        center_gap=np.zeros(0, np.float64),
        meta={
            "unit": unit,
            "delta": float(delta),
            "radius_key": key,
            "engine": "kdtree",
        },
    )


# --- core: simple, robust node↔node contacts via KD-tree ---------------


def find_contact_seeds(
    A: "Skeleton",
    B: "Skeleton",
    *,
    delta: float = 0.0,
    radius_key: str | None = None,
    exclude_soma: bool = True,
) -> "ContactSeeds":
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
    radii_superset = RA + rb_margin + float(delta)  # per-A query radius

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
    d2 = np.einsum("ij,ij->i", Ai - Bj, Ai - Bj)
    thresh = (RAi + RBj + float(delta)) ** 2
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

    return ContactSeeds(
        idx_a=ia,
        idx_b=ib,
        pos_a=pa,
        pos_b=pb,
        pos=pos,
        center_gap=gap.astype(np.float64),
        meta={
            "unit": A.meta.get("unit"),
            "delta": float(delta),
            "radius_key": key,
            "exclude_soma": exclude_soma,
        },
    )


# --- approximate contact sites purely based on skeleton


@dataclass(slots=True)
class ProxySites:
    seed_groups: list[np.ndarray]
    center: np.ndarray  # (M,3) mean of seed midpoints (µm)
    area_A: np.ndarray  # (M,) in chosen unit
    area_B: np.ndarray  # (M,) in chosen unit
    area_mean: np.ndarray
    seed_to_site: np.ndarray  # (K,)
    meta: dict


def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    # stable ||xi-xj|| without pdist
    G = X @ X.T
    n = np.einsum("ii->i", G)
    D2 = n[:, None] + n[None, :] - 2 * G
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2, dtype=float)
    np.fill_diagonal(D, 0.0)
    return D


def _cap_cos_theta(d, r, r_other_plus_tol):
    num = d * d + r * r - r_other_plus_tol * r_other_plus_tol
    den = 2.0 * np.maximum(d * r, 1e-12)
    return np.clip(num / den, -1.0, 1.0)


def _cap_area_and_chord_radius(r, cos_th):
    area = 2.0 * np.pi * r * r * (1.0 - cos_th)
    a = r * np.sqrt(np.maximum(0.0, 1.0 - cos_th * cos_th))
    return area, a


def approximate_contact_sites(
    A: "Skeleton",
    B: "Skeleton",
    seeds: "ContactSeeds",
    *,
    tol_nm: float = 60.0,  # match your mesh tol if you have it
    radius_key: str | None = None,
    gamma_mid: float = 2.0,  # looseness for midpoint discs
    gamma_side: float = 2.0,  # looseness for pos_a/pos_b discs
    beta_node: float = 0.35,  # node-scale floor fraction
    eps_floor_um: float = 0.08,  # absolute floor (~80 nm)
    theta_max_deg: float | None = None,  # set e.g. 60 for orientation gating
    require_shared_node: str = "none",  # "none" | "a" | "b" | "either"
    area_unit: str = "um^2",
) -> ProxySites:
    K = len(seeds.idx_a)
    if K == 0:
        return ProxySites(
            [],
            np.empty((0, 3), float),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            -np.ones(0, np.int64),
            dict(K_seeds=0),
        )

    key = radius_key or seeds.meta.get("radius_key") or A.recommend_radius()[0]
    ia = np.asarray(seeds.idx_a, np.int64)
    ib = np.asarray(seeds.idx_b, np.int64)
    ra_all = np.asarray(A.radii[key], float)
    rb_all = np.asarray(B.radii[key], float)
    ra = ra_all[ia]
    rb = rb_all[ib]

    # seed geometry (µm)
    Pm = np.asarray(seeds.pos, float)  # midpoints
    Pa = np.asarray(seeds.pos_a, float)  # on sphere A
    Pb = np.asarray(seeds.pos_b, float)  # on sphere B
    gap = np.asarray(seeds.center_gap, float)
    d = np.maximum(ra + rb + gap, 0.0)

    # tolerance in µm
    t_um = float(tol_nm) * 1e-3

    # spherical-cap areas/chord radii
    cosA = _cap_cos_theta(d, ra, rb + t_um)
    cosB = _cap_cos_theta(d, rb, ra + t_um)
    areaA_um2, aA = _cap_area_and_chord_radius(ra, cosA)
    areaB_um2, aB = _cap_area_and_chord_radius(rb, cosB)

    # zero-out when not touching under tolerance
    no_touch = d >= (ra + rb + t_um) - 1e-12
    areaA_um2[no_touch] = 0.0
    areaB_um2[no_touch] = 0.0
    aA[no_touch] = 0.0
    aB[no_touch] = 0.0

    # --- merge radii (adaptive, never collapses) ---
    node_floorA = beta_node * ra
    node_floorB = beta_node * rb
    R_mid = gamma_mid * (0.5 * (aA + aB) + t_um)
    R_mid = np.maximum(R_mid, np.maximum(node_floorA, node_floorB))
    R_mid = np.maximum(R_mid, eps_floor_um)

    R_A = gamma_side * (aA + t_um)
    R_A = np.maximum(R_A, node_floorA)
    R_A = np.maximum(R_A, eps_floor_um)

    R_B = gamma_side * (aB + t_um)
    R_B = np.maximum(R_B, node_floorB)
    R_B = np.maximum(R_B, eps_floor_um)

    # distances
    Dm = _pairwise_dists(Pm)
    Da = _pairwise_dists(Pa)
    Db = _pairwise_dists(Pb)

    # disc-overlap on any of the three spaces
    M_mid = Dm <= (R_mid[:, None] + R_mid[None, :])
    M_a = Da <= (R_A[:, None] + R_A[None, :])
    M_b = Db <= (R_B[:, None] + R_B[None, :])
    M_any = M_mid | M_a | M_b
    np.fill_diagonal(M_any, False)

    # optional orientation gating
    if theta_max_deg is not None:
        # line-of-centers
        ca = np.asarray(A.nodes, float)[ia]
        cb = np.asarray(B.nodes, float)[ib]
        v = cb - ca
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        C = v @ v.T  # cosine matrix
        C_ok = C >= np.cos(np.deg2rad(theta_max_deg))
        np.fill_diagonal(C_ok, False)
        M_any &= C_ok

    # optional node-sharing gating
    if require_shared_node != "none":
        shareA = ia[:, None] == ia[None, :]
        shareB = ib[:, None] == ib[None, :]
        if require_shared_node == "a":
            M_any &= shareA
        elif require_shared_node == "b":
            M_any &= shareB
        else:  # "either"
            M_any &= shareA | shareB

    # union-find from adjacency
    parent = np.arange(K, dtype=np.int64)
    rank = np.zeros(K, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1

    II, JJ = np.nonzero(M_any)
    for i, j in zip(II, JJ):
        if i < j:
            union(i, j)

    roots = np.fromiter((find(i) for i in range(K)), np.int64)
    uniq, inv = np.unique(roots, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]

    # per-site areas: sum over nodes of max cap per node (A and B separately)
    areaA_site_um2, areaB_site_um2, centers = [], [], []
    for g in groups:
        centers.append(Pm[g].mean(axis=0))
        # A side
        ia_g = ia[g]
        aA_g = areaA_um2[g]
        if ia_g.size:
            order = np.argsort(ia_g)
            ia_s = ia_g[order]
            a_s = aA_g[order]
            starts = np.r_[0, 1 + np.flatnonzero(ia_s[1:] != ia_s[:-1])]
            ends = np.r_[starts[1:], ia_s.size]
            areaA_site_um2.append(
                sum(float(np.max(a_s[s:e])) for s, e in zip(starts, ends))
            )
        else:
            areaA_site_um2.append(0.0)
        # B side
        ib_g = ib[g]
        bB_g = areaB_um2[g]
        order = np.argsort(ib_g)
        ib_s = ib_g[order]
        b_s = bB_g[order]
        starts = np.r_[0, 1 + np.flatnonzero(ib_s[1:] != ib_s[:-1])]
        ends = np.r_[starts[1:], ib_s.size]
        areaB_site_um2.append(
            sum(float(np.max(b_s[s:e])) for s, e in zip(starts, ends))
        )

    areaA_site_um2 = np.asarray(areaA_site_um2, float)
    areaB_site_um2 = np.asarray(areaB_site_um2, float)
    area_mean_um2 = 0.5 * (areaA_site_um2 + areaB_site_um2)
    centers = np.vstack(centers) if centers else np.empty((0, 3), float)

    # unit conversion
    if area_unit == "um^2":
        conv = 1.0
    elif area_unit == "nm^2":
        conv = 1e6  # 1 µm² = 10^6 nm²
    else:
        raise ValueError("area_unit must be 'um^2' or 'nm^2'")

    area_A = areaA_site_um2 * conv
    area_B = areaB_site_um2 * conv
    area_M = area_mean_um2 * conv

    seed_to_site = -np.ones(K, np.int64)
    for s, g in enumerate(groups):
        seed_to_site[g] = s

    meta = dict(
        engine="seed-only: discs @ mid/pos_a/pos_b + UF",
        K_seeds=int(K),
        K_sites=int(len(groups)),
        tol_nm=float(tol_nm),
        gamma_mid=float(gamma_mid),
        gamma_side=float(gamma_side),
        beta_node=float(beta_node),
        eps_floor_um=float(eps_floor_um),
        theta_max_deg=theta_max_deg,
        require_shared_node=require_shared_node,
        area_unit=area_unit,
    )

    return ProxySites(
        seed_groups=[np.asarray(g, np.int64) for g in groups],
        center=centers,
        area_A=area_A,
        area_B=area_B,
        area_mean=area_M,
        seed_to_site=seed_to_site,
        meta=meta,
    )


################
## Mesh-based ##
################

# ------------------------ helpers --------------------------------------


def _nearest_on_surface(
    mesh: trimesh.Trimesh, pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust batched projection using one ProximityQuery, returns (points, dists, face_ids).
    """
    pts = np.asarray(pts, np.float64)
    pq = trimesh.proximity.ProximityQuery(mesh)
    cp, dist, fid = pq.on_surface(pts)
    return np.asarray(cp, float), np.asarray(dist, float), np.asarray(fid, np.int64)


def _sample_points_on_faces(
    mesh: trimesh.Trimesh, faces: np.ndarray, scheme: str = "7"
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return (points, local_face_index_per_point, S) for a list of face indices.
    scheme '7' = centroid + 3 verts + 3 edge midpoints.
    """
    faces = np.asarray(faces, np.int64)
    tri = mesh.triangles[faces]  # (F,3,3)
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    centroid = (a + b + c) / 3.0

    if scheme == "1c":
        pts = centroid[:, None, :]  # (F,1,3)
        S = 1
    elif scheme == "4":
        pts = np.stack([centroid, a, b, c], axis=1)  # (F,4,3)
        S = 4
    elif scheme == "7":
        m_ab = (a + b) / 2.0
        m_bc = (b + c) / 2.0
        m_ca = (c + a) / 2.0
        pts = np.stack([centroid, a, b, c, m_ab, m_bc, m_ca], axis=1)  # (F,7,3)
        S = 7
    else:
        raise ValueError("Unsupported sample_scheme")

    F = pts.shape[0]
    P = pts.reshape(F * S, 3)
    face_of_point = np.repeat(np.arange(F, dtype=np.int64), S)
    return P, face_of_point, S


def median_edge_length(mesh: trimesh.Trimesh, max_faces: int = 50000) -> float:
    """
    Approximate median edge length without touching edges_unique_length.
    Samples up to max_faces triangles and takes median of their 3 edge lengths.
    """
    tri = np.asarray(mesh.triangles)
    if tri.size == 0:
        return 1.0
    if tri.shape[0] > max_faces:
        # deterministic sample for reproducibility
        rng = np.random.RandomState(0)
        idx = rng.choice(tri.shape[0], size=max_faces, replace=False)
        tri = tri[idx]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    L = np.concatenate(
        [
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ]
    )
    return float(np.median(L)) if L.size else 1.0


def _build_face_kdt(
    mesh: trimesh.Trimesh,
) -> tuple[KDTree, np.ndarray, np.ndarray, np.ndarray]:
    """
    KD-tree on triangle centroids + cached face normals/areas.
    """
    centers = mesh.triangles_center
    kdt = KDTree(centers) if len(centers) else None
    normals = mesh.face_normals
    areas = mesh.area_faces
    return kdt, centers, normals, areas


def _faces_within_radius(kdt: KDTree, points: np.ndarray, radius: float) -> np.ndarray:
    """
    Union of face indices whose centroids fall within 'radius' of any point in 'points'.
    Robust to KDTree returning list-of-lists or numpy object arrays.
    """
    if kdt is None:
        return np.empty(0, np.int64)

    pts = np.asarray(points, float)
    out = kdt.query_ball_point(
        pts, r=float(radius)
    )  # could be list-of-lists or ndarray(object)

    # Normalize to python list-of-lists
    if isinstance(out, np.ndarray):
        out = out.tolist()  # ndarray(object) -> list
    if not isinstance(out, list):
        out = [out]  # single list/int case

    flat: list[int] = []
    for grp in out:
        if grp is None:
            continue
        if isinstance(grp, (list, tuple, np.ndarray)):
            if len(grp):
                flat.extend(np.asarray(grp, np.int64).tolist())
        else:
            flat.append(int(grp))

    if not flat:
        return np.empty(0, np.int64)
    return np.unique(np.asarray(flat, np.int64))


def _submesh_from_faces(
    mesh: trimesh.Trimesh, faces_idx: np.ndarray
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    Build a submesh from explicit face indices and return (submesh, face_map_to_global).
    Ordering of submesh faces matches 'faces_idx' exactly.
    """
    faces_idx = np.asarray(faces_idx, np.int64)
    if faces_idx.size == 0:
        return trimesh.Trimesh(), faces_idx
    # Use index list (not boolean mask) so face i in submesh == faces_idx[i]
    sub = mesh.submesh([faces_idx], append=True, repair=False)
    return sub, faces_idx


def _cluster_points(P: np.ndarray, eps: float) -> list[np.ndarray]:
    """
    Simple density-based clustering with KDTree (DBSCAN-ish, but tiny and dependency-free).
    """
    if len(P) == 0:
        return []
    kdt = KDTree(P)
    unvisited = np.ones(len(P), dtype=bool)
    clusters: list[np.ndarray] = []
    for i in range(len(P)):
        if not unvisited[i]:
            continue
        queue = [i]
        unvisited[i] = False
        cur = [i]
        while queue:
            j = queue.pop()
            nbrs = kdt.query_ball_point(P[j], r=float(eps))
            for n in nbrs:
                if unvisited[n]:
                    unvisited[n] = False
                    queue.append(n)
                    cur.append(n)
        clusters.append(np.asarray(cur, np.int64))
    return clusters


def _aabbs_for_patches(
    mesh: trimesh.Trimesh, patch_faces: list[np.ndarray]
) -> np.ndarray:
    """
    Compute per-patch AABBs from lists of face indices.
    Returns (M,2,3) array; rows with no faces are filled with NaNs.
    """
    M = len(patch_faces)
    out = np.full((M, 2, 3), np.nan, dtype=float)
    if M == 0:
        return out
    tri = np.asarray(mesh.triangles)  # (F,3,3)
    for i, faces in enumerate(patch_faces):
        if len(faces):
            pts = tri[np.asarray(faces, np.int64)].reshape(-1, 3)
            out[i, 0] = pts.min(axis=0)
            out[i, 1] = pts.max(axis=0)
    return out


def _union_aabbs(bA: np.ndarray, bB: np.ndarray) -> np.ndarray:
    """
    Union two (M,2,3) AABB arrays (NaN-safe).
    If only one side is present (finite), union == that side.
    If both missing, union is NaN.
    """
    out = np.full_like(bA, np.nan)
    hasA = np.all(np.isfinite(bA), axis=(1, 2))
    hasB = np.all(np.isfinite(bB), axis=(1, 2))
    both = hasA & hasB

    # both sides present
    out[both, 0] = np.minimum(bA[both, 0], bB[both, 0])
    out[both, 1] = np.maximum(bA[both, 1], bB[both, 1])

    # only A
    onlyA = hasA & (~hasB)
    out[onlyA] = bA[onlyA]

    # only B
    onlyB = hasB & (~hasA)
    out[onlyB] = bB[onlyB]

    return out


# ------------------------ result struct ---------------------------------


@dataclass(slots=True)
class ContactSites:
    faces_A: list[np.ndarray]  # per-contact face indices on A
    faces_B: list[np.ndarray]  # per-contact face indices on B
    area_A: np.ndarray  # per-contact area on A
    area_B: np.ndarray  # per-contact area on B
    area_mean: np.ndarray  # (A+B)/2
    seeds_A: np.ndarray  # projected seed on A
    seeds_B: np.ndarray  # projected seed on B
    pairs_AB: list[np.ndarray] | None
    bbox_A: np.ndarray  # (M,2,3) [min,max] AABBs on A in mesh units
    bbox_B: np.ndarray  # (M,2,3) [min,max] AABBs on B in mesh units
    bbox: np.ndarray  # (M,2,3) union(A,B) in mesh units
    meta: dict[str, object]

    def to_npz(self, path: str | Path, *, compress: bool = True) -> None:
        # Lazy import to avoid module-level circular imports
        from .io import save_contact_sites_npz

        save_contact_sites_npz(self, path, compress=compress)


# ------------------------ fast extractor --------------------------------


def map_contact_sites(
    A: trimesh.Trimesh,
    B: trimesh.Trimesh,
    contact_pos: np.ndarray,
    *,
    tol: float | None = None,  # gap tolerance in mesh unit (nm in your case)
    radius: float | None = None,  # search radius around each seed
    sample_scheme: str = "1c",  # '1c', '4', or '7'
    fractional: bool = True,
    normal_opposition_dot: float | None = -0.2,
    cluster_eps: float | None = None,
    cluster_halo: float = 1.25,
    return_pairs: bool = False,
    workers: int = 16,
    sides: str = "both",
    unit: str = "nm^2",
) -> ContactSites:
    """
    Extract touching-face patches near seeds, then MERGE overlapping per-seed patches
    into consolidated contact patches (no seed→patch 1:1). Empty patches are pruned.

    Steps:
      1) project seeds;
      2) cluster seeds (to reuse local submeshes);
      3) per-seed faces via batched PQ;
      4) EARLY prune empty seeds (no faces/area);
      5) union-find over seeds using face overlap (A or B);
      6) per-group union of faces with capped fractional area;
      7) prune groups with no area on both sides.
    """
    P = np.asarray(contact_pos, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("contact_pos must be (K,3)")
    K = len(P)
    K0 = K  # original seed count

    s = sides.lower()
    doA = s in ("a", "both")
    doB = s in ("b", "both")
    if not (doA or doB):
        raise ValueError("sides must be 'A', 'B', or 'both'")

    # Resolution-aware defaults (cheap median edge)
    med_edgeA = median_edge_length(A)
    med_edgeB = median_edge_length(B)
    med_edge = float(np.median([med_edgeA, med_edgeB]))
    if tol is None:
        tol = 0.5 * med_edge
    if radius is None:
        radius = 6.0 * med_edge
    if cluster_eps is None:
        cluster_eps = max(radius, 4.0 * med_edge)

    tol = float(tol)
    radius = float(radius)
    cluster_eps = float(cluster_eps)

    # Per-mesh caches
    kdtA, _, nA, areaA = _build_face_kdt(A)
    kdtB, _, nB, areaB = _build_face_kdt(B)

    # Seed projections
    if doA:
        seeds_A, _, _ = _nearest_on_surface(A, P)
    else:
        seeds_A = np.full((K, 3), np.nan, float)
    if doB:
        seeds_B, _, _ = _nearest_on_surface(B, P)
    else:
        seeds_B = np.full((K, 3), np.nan, float)

    # Per-seed outputs
    out_faces_A: list[np.ndarray] = [np.empty(0, np.int64) for _ in range(K)]
    out_faces_B: list[np.ndarray] = [np.empty(0, np.int64) for _ in range(K)]
    out_frac_A: list[np.ndarray] = [
        np.empty(0, np.float64) for _ in range(K)
    ]  # per-face fractional cover
    out_frac_B: list[np.ndarray] = [np.empty(0, np.float64) for _ in range(K)]
    out_area_A = np.zeros(K, float)
    out_area_B = np.zeros(K, float)
    out_pairs: list[np.ndarray] | None = [] if return_pairs else None

    # Seed clustering (reuse crops)
    clusters = _cluster_points(P, eps=cluster_eps)

    # ---- inner worker over one cluster ---------------------------------
    def process_cluster(
        seed_idx: Iterable[int],
    ) -> list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            float,
            float,
            np.ndarray | None,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        seed_idx = np.asarray(list(seed_idx), np.int64)
        if seed_idx.size == 0:
            return []

        cluster_pts = P[seed_idx]
        crop_r = cluster_halo * (radius + tol)

        if doA:
            facesB_crop = _faces_within_radius(kdtB, cluster_pts, crop_r)
            subB, mapB = _submesh_from_faces(B, facesB_crop)
            pq_subB = (
                trimesh.proximity.ProximityQuery(subB) if len(subB.faces) else None
            )
        else:
            mapB = np.empty(0, np.int64)
            pq_subB = None

        if doB:
            facesA_crop = _faces_within_radius(kdtA, cluster_pts, crop_r)
            subA, mapA = _submesh_from_faces(A, facesA_crop)
            pq_subA = (
                trimesh.proximity.ProximityQuery(subA) if len(subA.faces) else None
            )
        else:
            mapA = np.empty(0, np.int64)
            pq_subA = None

        results: list[
            tuple[
                int,
                np.ndarray,
                np.ndarray,
                float,
                float,
                np.ndarray | None,
                np.ndarray,
                np.ndarray,
            ]
        ] = []

        # ---------- A-side batch ----------
        if doA:
            fa_list, pts_list, face_of_pt_list, S_A_list = [], [], [], []
            for k in seed_idx:
                fa = (
                    np.asarray(kdtA.query_ball_point(seeds_A[k], r=radius), np.int64)
                    if kdtA is not None
                    else np.empty(0, np.int64)
                )
                fa_list.append(fa)
                if fa.size:
                    ptsA, face_of_ptA, S_A = _sample_points_on_faces(
                        A, fa, sample_scheme
                    )
                else:
                    ptsA = np.empty((0, 3), float)
                    face_of_ptA = np.empty(0, np.int64)
                    S_A = 1
                pts_list.append(ptsA)
                face_of_pt_list.append(face_of_ptA)
                S_A_list.append(S_A)

            offs = np.cumsum([0] + [len(p) for p in pts_list[:-1]])
            ptsA_all = (
                np.vstack(pts_list)
                if len(pts_list) and len(pts_list[0])
                else np.empty((0, 3), float)
            )

            if pq_subB is not None and len(ptsA_all):
                _, dAB_all, fB_sub_all = pq_subB.on_surface(ptsA_all)
                dAB_all = np.asarray(dAB_all, float)
                fB_sub_all = np.asarray(fB_sub_all, np.int64)
            else:
                dAB_all = np.empty(0, float)
                fB_sub_all = np.empty(0, np.int64)

            for i, k in enumerate(seed_idx):
                fa = fa_list[i]
                pts_len = len(pts_list[i])
                if pts_len == 0:
                    facesA_k = np.empty(0, np.int64)
                    fracA_k = np.empty(0, np.float64)
                    areaA_k = 0.0
                    prs = None
                else:
                    sl = slice(offs[i], offs[i] + pts_len)
                    dAB = dAB_all[sl]
                    fB_sub = fB_sub_all[sl]
                    okA = dAB <= tol
                    if normal_opposition_dot is not None and okA.any():
                        gfa = fa[face_of_pt_list[i]]
                        nd = np.einsum("ij,ij->i", nA[gfa], nB[mapB[fB_sub]])
                        okA &= nd <= float(normal_opposition_dot)
                    if okA.any():
                        # per-face fractions
                        src_faces = fa[face_of_pt_list[i][okA]]
                        uniqA, countsA = np.unique(src_faces, return_counts=True)
                        S_A = float(S_A_list[i])
                        fracA_k = countsA.astype(np.float64) / S_A
                        facesA_k = uniqA
                        if fractional:
                            areaA_k = float(np.dot(fracA_k, areaA[facesA_k]))
                        else:
                            areaA_k = float(areaA[facesA_k].sum())
                        prs = None
                        if return_pairs:
                            w = areaA[fa[face_of_pt_list[i][okA]]] / S_A
                            prs = np.stack(
                                [fa[face_of_pt_list[i][okA]], mapB[fB_sub[okA]], w],
                                axis=1,
                            ).astype(np.float64)
                    else:
                        facesA_k = np.empty(0, np.int64)
                        fracA_k = np.empty(0, np.float64)
                        areaA_k = 0.0
                        prs = None
                results.append(
                    (
                        int(k),
                        facesA_k,
                        np.empty(0, np.int64),
                        float(areaA_k),
                        0.0,
                        prs,
                        fracA_k,
                        np.empty(0, np.float64),
                    )
                )

        # ---------- B-side batch ----------
        if doB:
            fb_list, pts_list, face_of_pt_list, S_B_list = [], [], [], []
            for k in seed_idx:
                fb = (
                    np.asarray(kdtB.query_ball_point(seeds_B[k], r=radius), np.int64)
                    if kdtB is not None
                    else np.empty(0, np.int64)
                )
                fb_list.append(fb)
                if fb.size:
                    ptsB, face_of_ptB, S_B = _sample_points_on_faces(
                        B, fb, sample_scheme
                    )
                else:
                    ptsB = np.empty((0, 3), float)
                    face_of_ptB = np.empty(0, np.int64)
                    S_B = 1
                pts_list.append(ptsB)
                face_of_pt_list.append(face_of_ptB)
                S_B_list.append(S_B)

            offs = np.cumsum([0] + [len(p) for p in pts_list[:-1]])
            ptsB_all = (
                np.vstack(pts_list)
                if len(pts_list) and len(pts_list[0])
                else np.empty((0, 3), float)
            )

            if pq_subA is not None and len(ptsB_all):
                _, dBA_all, fA_sub_all = pq_subA.on_surface(ptsB_all)
                dBA_all = np.asarray(dBA_all, float)
                fA_sub_all = np.asarray(fA_sub_all, np.int64)
            else:
                dBA_all = np.empty(0, float)
                fA_sub_all = np.empty(0, np.int64)

            pos_by_seed = {res[0]: i for i, res in enumerate(results)}
            for i, k in enumerate(seed_idx):
                fb = fb_list[i]
                pts_len = len(pts_list[i])
                facesB_k = np.empty(0, np.int64)
                fracB_k = np.empty(0, np.float64)
                areaB_k = 0.0
                prs2 = None
                if pts_len:
                    sl = slice(offs[i], offs[i] + pts_len)
                    dBA = dBA_all[sl]
                    fA_sub = fA_sub_all[sl]
                    okB = dBA <= tol
                    if normal_opposition_dot is not None and okB.any():
                        gfb = fb[face_of_pt_list[i]]
                        nd = np.einsum("ij,ij->i", nB[gfb], nA[mapA[fA_sub]])
                        okB &= nd <= float(normal_opposition_dot)
                    if okB.any():
                        src_facesB = fb[face_of_pt_list[i][okB]]
                        uniqB, countsB = np.unique(src_facesB, return_counts=True)
                        S_B = float(S_B_list[i])
                        fracB_k = countsB.astype(np.float64) / S_B
                        facesB_k = uniqB
                        if fractional:
                            areaB_k = float(np.dot(fracB_k, areaB[facesB_k]))
                        else:
                            areaB_k = float(areaB[facesB_k].sum())
                        if return_pairs:
                            w = areaB[fb[face_of_pt_list[i][okB]]] / S_B
                            prs2 = np.stack(
                                [mapA[fA_sub[okB]], fb[face_of_pt_list[i][okB]], w],
                                axis=1,
                            ).astype(np.float64)

                j = pos_by_seed.get(int(k))
                if j is None:
                    results.append(
                        (
                            int(k),
                            np.empty(0, np.int64),
                            facesB_k,
                            0.0,
                            float(areaB_k),
                            prs2,
                            np.empty(0, np.float64),
                            fracB_k,
                        )
                    )
                else:
                    (
                        k_id,
                        fa_prev,
                        fb_prev,
                        aA_prev,
                        aB_prev,
                        prs_prev,
                        frA_prev,
                        frB_prev,
                    ) = results[j]
                    results[j] = (
                        k_id,
                        fa_prev,
                        facesB_k,
                        aA_prev,
                        float(areaB_k),
                        prs2
                        if prs_prev is None
                        else (
                            prs_prev if prs2 is None else np.vstack([prs_prev, prs2])
                        ),
                        frA_prev,
                        fracB_k,
                    )
        return results

    # ---- run (optionally parallel) --------------------------------------
    if workers and len(clusters) > 1:
        import concurrent.futures as cf

        all_results = []
        with cf.ThreadPoolExecutor(max_workers=int(workers)) as ex:
            for res in ex.map(process_cluster, clusters):
                all_results.extend(res)
    else:
        all_results = []
        for cl in clusters:
            all_results.extend(process_cluster(cl))

    # write back per-seed
    for k, fa, fb, aA, aB, prs, frA, frB in all_results:
        out_faces_A[k] = fa
        out_faces_B[k] = fb
        out_frac_A[k] = frA
        out_frac_B[k] = frB
        out_area_A[k] = aA
        out_area_B[k] = aB
        if return_pairs:
            out_pairs.append(prs if prs is not None else np.empty((0, 3), np.float64))

    # -------- EARLY PRUNE empty seeds (before merging) --------------------
    EPS = 1e-12
    if doA:
        has_faces_A = np.fromiter(
            (len(f) > 0 for f in out_faces_A), dtype=bool, count=K
        )
        has_area_A = out_area_A > EPS
        keepA = has_faces_A & has_area_A
    else:
        keepA = np.zeros(K, dtype=bool)

    if doB:
        has_faces_B = np.fromiter(
            (len(f) > 0 for f in out_faces_B), dtype=bool, count=K
        )
        has_area_B = out_area_B > EPS
        keepB = has_faces_B & has_area_B
    else:
        keepB = np.zeros(K, dtype=bool)

    keep_seed = keepA | keepB
    orig_idx = np.flatnonzero(keep_seed)

    # compact per-seed structures
    out_faces_A = [out_faces_A[i] for i in orig_idx]
    out_faces_B = [out_faces_B[i] for i in orig_idx]
    out_frac_A = [out_frac_A[i] for i in orig_idx]
    out_frac_B = [out_frac_B[i] for i in orig_idx]
    out_area_A = out_area_A[orig_idx]
    out_area_B = out_area_B[orig_idx]
    seeds_A = seeds_A[orig_idx]
    seeds_B = seeds_B[orig_idx]
    if return_pairs:
        out_pairs = [out_pairs[i] for i in orig_idx]

    K = orig_idx.size
    if K == 0:
        meta = dict(
            tol=tol,
            radius=radius,
            fractional=bool(fractional),
            sample_scheme=sample_scheme,
            normal_opposition_dot=normal_opposition_dot,
            cluster_eps=cluster_eps,
            cluster_halo=float(cluster_halo),
            engine="seed-cluster → local-submesh PQ → union-by-face",
            sides=(
                "A" if (doA and not doB) else ("B" if (doB and not doA) else "both")
            ),
            unit=unit,
            K_seeds_in=int(K0),
            K_seeds_kept=0,
            K_groups_before_prune=0,
            K_patches=0,
            seed_to_patch=(-np.ones(K0, dtype=np.int64)),
        )
        empty_bbox = np.empty((0, 2, 3), float)
        return ContactSites(
            faces_A=[],
            faces_B=[],
            area_A=np.zeros(0, float),
            area_B=np.zeros(0, float),
            area_mean=np.zeros(0, float),
            seeds_A=np.empty((0, 3), float),
            seeds_B=np.empty((0, 3), float),
            pairs_AB=([] if return_pairs else None),
            bbox_A=empty_bbox,
            bbox_B=empty_bbox,
            bbox=empty_bbox,
            meta=meta,
        )

    # -------- MERGE overlapping patches (by face overlap on A or B) --------
    parent = np.arange(K, dtype=np.int64)
    rank = np.zeros(K, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    def _union_from_faces_list(faces_list: list[np.ndarray]):
        total = sum(len(f) for f in faces_list)
        if total == 0:
            return
        face_all = np.concatenate([f for f in faces_list if len(f)])
        seed_all = np.concatenate(
            [
                np.full(len(f), i, dtype=np.int64)
                for i, f in enumerate(faces_list)
                if len(f)
            ]
        )
        order = np.argsort(face_all)
        face_sorted = face_all[order]
        seed_sorted = seed_all[order]
        if len(face_sorted) == 0:
            return
        starts = np.r_[0, 1 + np.flatnonzero(face_sorted[1:] != face_sorted[:-1])]
        ends = np.r_[starts[1:], len(face_sorted)]
        for s0, s1 in zip(starts, ends):
            if s1 - s0 <= 1:
                continue
            base = seed_sorted[s0]
            for u in seed_sorted[s0 + 1 : s1]:
                union(base, u)

    if doA:
        _union_from_faces_list(out_faces_A)
    if doB:
        _union_from_faces_list(out_faces_B)

    roots = np.fromiter((find(i) for i in range(K)), dtype=np.int64, count=K)
    uniq_roots, inv = np.unique(roots, return_inverse=True)
    groups = [
        np.flatnonzero(inv == gi) for gi in range(len(uniq_roots))
    ]  # list of compacted-seed index arrays

    # Build merged outputs
    new_faces_A: list[np.ndarray] = []
    new_faces_B: list[np.ndarray] = []
    new_area_A_list: list[float] = []
    new_area_B_list: list[float] = []
    new_pairs: list[np.ndarray] | None = [] if return_pairs else None
    new_seeds_A: list[np.ndarray] = []
    new_seeds_B: list[np.ndarray] = []

    for members in groups:
        # ---- A side merge (guarded)
        if doA:
            fa_seqs = [out_faces_A[i] for i in members if len(out_faces_A[i])]
            fr_seqs = [out_frac_A[i] for i in members if len(out_frac_A[i])]
            if fa_seqs:
                fa_cat = np.concatenate(fa_seqs)
                fr_cat = np.concatenate(fr_seqs)
                uA, invA = np.unique(fa_cat, return_inverse=True)
                frac_sum = np.bincount(invA, weights=fr_cat, minlength=uA.size)
                frac_sum = np.minimum(1.0, frac_sum)
                areaA_val = (
                    float(np.dot(frac_sum, areaA[uA]))
                    if fractional
                    else float(areaA[uA].sum())
                )
                facesA_m = uA
            else:
                facesA_m = np.empty(0, np.int64)
                areaA_val = 0.0
        else:
            facesA_m = np.empty(0, np.int64)
            areaA_val = 0.0

        # ---- B side merge (guarded)
        if doB:
            fb_seqs = [out_faces_B[i] for i in members if len(out_faces_B[i])]
            frb_seqs = [out_frac_B[i] for i in members if len(out_frac_B[i])]
            if fb_seqs:
                fb_cat = np.concatenate(fb_seqs)
                frb_cat = np.concatenate(frb_seqs)
                uB, invB = np.unique(fb_cat, return_inverse=True)
                frac_sumB = np.bincount(invB, weights=frb_cat, minlength=uB.size)
                frac_sumB = np.minimum(1.0, frac_sumB)
                areaB_val = (
                    float(np.dot(frac_sumB, areaB[uB]))
                    if fractional
                    else float(areaB[uB].sum())
                )
                facesB_m = uB
            else:
                facesB_m = np.empty(0, np.int64)
                areaB_val = 0.0
        else:
            facesB_m = np.empty(0, np.int64)
            areaB_val = 0.0

        # seeds: mean of projected seeds in the group (NaN-safe)
        if doA:
            sA = seeds_A[members]
            new_seeds_A.append(np.nanmean(sA, axis=0))
        else:
            new_seeds_A.append(np.array([np.nan, np.nan, np.nan], dtype=float))
        if doB:
            sB = seeds_B[members]
            new_seeds_B.append(np.nanmean(sB, axis=0))
        else:
            new_seeds_B.append(np.array([np.nan, np.nan, np.nan], dtype=float))

        # pairs merge (sum duplicate weights)
        if return_pairs:
            pcats = [
                out_pairs[i]
                for i in members
                if out_pairs[i] is not None and out_pairs[i].size
            ]
            if pcats:
                cat = np.vstack(pcats)
                keys = cat[:, :2].astype(np.int64)
                w = cat[:, 2].astype(float)
                uniq_keys, invk = np.unique(keys, axis=0, return_inverse=True)
                wsum = np.bincount(invk, weights=w, minlength=uniq_keys.shape[0])
                merged_pairs = np.column_stack([uniq_keys, wsum]).astype(np.float64)
            else:
                merged_pairs = np.empty((0, 3), np.float64)
            new_pairs.append(merged_pairs)

        new_faces_A.append(facesA_m)
        new_faces_B.append(facesB_m)
        new_area_A_list.append(areaA_val)
        new_area_B_list.append(areaB_val)

    # Convert lists to arrays
    out_faces_A = new_faces_A
    out_faces_B = new_faces_B
    out_area_A = np.asarray(new_area_A_list, float)
    out_area_B = np.asarray(new_area_B_list, float)
    seeds_A = np.vstack(new_seeds_A) if new_seeds_A else np.empty((0, 3), float)
    seeds_B = np.vstack(new_seeds_B) if new_seeds_B else np.empty((0, 3), float)
    bbox_A = (
        _aabbs_for_patches(A, out_faces_A)
        if doA
        else np.full((len(out_faces_A), 2, 3), np.nan)
    )
    bbox_B = (
        _aabbs_for_patches(B, out_faces_B)
        if doB
        else np.full((len(out_faces_B), 2, 3), np.nan)
    )
    bbox_union = _union_aabbs(bbox_A, bbox_B)

    K_groups = len(out_faces_A)  # == len(groups)

    # Map from original seed index -> patch index (after group pruning below)
    # Start with compacted-seed -> group index:
    seed_to_group = inv.copy()  # length K (compacted)
    # We'll finalize to original length after group pruning.

    # -------- ALWAYS prune empty groups -----------------------------------
    eps = 1e-12
    hasA = (
        (np.array([len(fa) > 0 for fa in out_faces_A], dtype=bool) & (out_area_A > eps))
        if doA
        else np.zeros(K_groups, bool)
    )
    hasB = (
        (np.array([len(fb) > 0 for fb in out_faces_B], dtype=bool) & (out_area_B > eps))
        if doB
        else np.zeros(K_groups, bool)
    )
    keep_mask_groups = hasA | hasB
    keep_idx = np.flatnonzero(keep_mask_groups)

    if keep_idx.size != K_groups:
        out_faces_A = [out_faces_A[i] for i in keep_idx]
        out_faces_B = [out_faces_B[i] for i in keep_idx]
        out_area_A = out_area_A[keep_idx]
        out_area_B = out_area_B[keep_idx]
        seeds_A = seeds_A[keep_idx]
        seeds_B = seeds_B[keep_idx]
        if return_pairs:
            new_pairs = [new_pairs[i] for i in keep_idx]
        # remap compacted-seed -> new group indices, else -1
        new_index_of_group = -np.ones(K_groups, dtype=np.int64)
        new_index_of_group[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)
        seed_to_group = np.where(
            keep_mask_groups[seed_to_group], new_index_of_group[seed_to_group], -1
        )

    # area_mean after pruning
    if doA and doB:
        area_mean = 0.5 * (out_area_A + out_area_B)
    elif doA:
        area_mean = out_area_A.copy()
    else:
        area_mean = out_area_B.copy()

    # Build seed_to_patch array in ORIGINAL seed indexing
    seed_to_patch = -np.ones(K0, dtype=np.int64)
    seed_to_patch[orig_idx] = seed_to_group  # compacted seeds correspond to orig_idx

    meta = dict(
        tol=tol,
        radius=radius,
        fractional=bool(fractional),
        sample_scheme=sample_scheme,
        normal_opposition_dot=normal_opposition_dot,
        cluster_eps=cluster_eps,
        cluster_halo=float(cluster_halo),
        engine="seed-cluster → local-submesh PQ → union-by-face",
        sides=("A" if (doA and not doB) else ("B" if (doB and not doA) else "both")),
        unit=unit,
        K_seeds_in=int(K0),
        K_seeds_kept=int(len(orig_idx)),
        K_groups_before_prune=int(K_groups),
        K_patches=int(len(out_faces_A)),
        seed_to_patch=seed_to_patch,
    )

    return ContactSites(
        faces_A=out_faces_A,
        faces_B=out_faces_B,
        area_A=out_area_A,
        area_B=out_area_B,
        area_mean=area_mean,
        seeds_A=seeds_A,
        seeds_B=seeds_B,
        pairs_AB=(new_pairs if return_pairs else None),
        bbox_A=bbox_A,
        bbox_B=bbox_B,
        bbox=bbox_union,
        meta=meta,
    )
