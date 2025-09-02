import math
from copy import deepcopy

import numpy as np
import pytest

import skeliner as sk
from skeliner.core import Skeleton, Soma

# ======================== analytic helpers ========================


def V_capsule(L: float, r0: float, r1: float) -> float:
    """
    Volume of a capsule with linear radius taper (frustum) and hemispherical end-caps.
    NOTE: For L=0 and r0!=r1 the *true union* is a single ball of radius max(r0, r1),
    while this formula degenerates to sum of hemispheres. Only use this formula
    for L>0 or the r0=r1 special case.
    """
    return np.pi * L * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0 + (2.0 / 3.0) * np.pi * (
        r0**3 + r1**3
    )


def V_cylinder(L: float, r: float) -> float:
    return np.pi * r * r * L


def V_ball(r: float) -> float:
    return (4.0 / 3.0) * np.pi * r**3


def V_ellipsoid(a: float, b: float, c: float) -> float:
    return (4.0 / 3.0) * np.pi * a * b * c


# ======================== builders & transforms ========================


def tiny_soma(center=(0, 0, 0), eps=1e-9) -> Soma:
    # Near-zero sphere; keeps Skeleton happy without affecting volume
    return Soma.from_sphere(np.asarray(center, float), float(eps), verts=None)


def skel_capsule(
    L: float, r0: float, r1: float, origin=(0, 0, 0), direction=(1, 0, 0)
) -> Skeleton:
    """
    Single edge with radii r0 -> r1 along straight segment of length L.
    Node 0 is a tiny soma placeholder at the origin by default.
    """
    origin = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    a = origin
    b = origin + L * d

    nodes = np.vstack([np.zeros(3), a, b])  # node 0 = soma placeholder
    radii_vec = np.array([0.0, r0, r1], dtype=float)
    edges = np.array([[1, 2]], dtype=np.int64)  # single segment

    return Skeleton(
        soma=tiny_soma(),
        nodes=nodes.astype(np.float64),
        radii={"median": radii_vec.copy(), "mean": radii_vec.copy()},
        edges=edges,
        ntype=None,
        node2verts=None,
        vert2node=None,
        meta={},
        extra={},
    )


def skel_capsule_with_mean_scale(L, r0, r1, mean_scale=1.1) -> Skeleton:
    """Same as skel_capsule but 'mean' radii are scaled vs 'median'."""
    skel = skel_capsule(L, r0, r1)
    skel.radii["mean"] = skel.radii["median"] * float(mean_scale)
    return skel


def skel_two_capsules() -> Skeleton:
    # Capsule A: L=8, r=0.8 at y=0
    A = skel_capsule(L=8.0, r0=0.8, r1=0.8, origin=(0, 0, 0), direction=(1, 0, 0))
    # Capsule B: L=5, r=0.5 far away at y=100 (no union overlap)
    B = skel_capsule(L=5.0, r0=0.5, r1=0.5, origin=(0, 100, 0), direction=(1, 0, 0))

    # Merge into a single skeleton (share tiny soma at 0)
    nodes = np.vstack([A.nodes[0:1], A.nodes[1:], B.nodes[1:]])
    r = np.concatenate(
        [A.radii["median"][0:1], A.radii["median"][1:], B.radii["median"][1:]]
    )
    edges = np.vstack([A.edges, B.edges + (len(A.nodes) - 1)])

    return Skeleton(
        soma=tiny_soma(),
        nodes=nodes,
        radii={"median": r.copy(), "mean": r.copy()},
        edges=edges.astype(np.int64),
        ntype=None,
        node2verts=None,
        vert2node=None,
        meta={},
        extra={},
    )


def skel_branch_y(L=6.0, r=0.8, origin=(20.0, 0.0, 0.0)) -> Skeleton:
    """
    A simple Y-junction: two edges meeting at a junction node.
    Geometry is placed away from the tiny soma to avoid overlap.
    """
    o = np.asarray(origin, float)
    p0 = np.zeros(3, float)  # soma placeholder
    p1 = o
    p2 = o + np.array([L, 0.0, 0.0])
    p3 = o + np.array([0.0, L, 0.0])

    nodes = np.vstack([p0, p1, p2, p3])
    rvec = np.array([0.0, r, r, r], float)
    edges = np.array([[1, 2], [1, 3]], np.int64)

    return Skeleton(
        soma=tiny_soma(),
        nodes=nodes,
        radii={"median": rvec.copy(), "mean": rvec.copy()},
        edges=edges,
        ntype=None,
        node2verts=None,
        vert2node=None,
        meta={},
        extra={},
    )


def skel_soma_only(a=3.0, b=2.0, c=1.0, center=(0, 0, 0)) -> Skeleton:
    nodes = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    r = np.array([0.0], dtype=float)
    R = np.eye(3, dtype=float)
    soma = Soma(center=np.asarray(center, float), axes=np.array([a, b, c], float), R=R)
    return Skeleton(
        soma=soma,
        nodes=nodes,
        radii={"median": r.copy(), "mean": r.copy()},
        edges=np.empty((0, 2), dtype=np.int64),
        ntype=None,
        node2verts=None,
        vert2node=None,
        meta={},
        extra={},
    )


def translate_skeleton(skel: Skeleton, delta) -> Skeleton:
    """Return a deep-copied skeleton translated by 'delta' (3-vector)."""
    delta = np.asarray(delta, float).reshape(3)
    sk2 = deepcopy(skel)
    sk2.nodes = sk2.nodes + delta
    if sk2.soma is not None:
        sk2.soma = Soma(
            center=sk2.soma.center + delta,
            axes=sk2.soma.axes.copy(),
            R=sk2.soma.R.copy(),
            verts=None if sk2.soma.verts is None else None,
        )
    return sk2


def rotation_matrix_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotate_skeleton(skel: Skeleton, R: np.ndarray, about=(0, 0, 0)) -> Skeleton:
    """Rigidly rotate skeleton and soma in world coordinates."""
    about = np.asarray(about, float)
    sk2 = deepcopy(skel)
    sk2.nodes = ((sk2.nodes - about) @ R.T) + about
    if sk2.soma is not None:
        new_center = ((sk2.soma.center - about) @ R.T) + about
        new_R = R @ sk2.soma.R
        sk2.soma = Soma(
            center=new_center, axes=sk2.soma.axes.copy(), R=new_R, verts=None
        )
    return sk2


# ======================== tests: correctness ========================


@pytest.mark.parametrize("direction", [(1, 0, 0), (1, 1, 1)])
def test_capsule_constant_radius_matches_analytic(direction):
    L, r = 10.0, 1.0
    voxel_size = 0.12  # ~16–17 voxels across diameter → sub-2% expected
    skel = skel_capsule(L=L, r0=r, r1=r, origin=(0, 0, 0), direction=direction)
    V_true = V_capsule(L, r, r)
    V_est = sk.dx.volume(
        skel, include_soma=False, voxel_size=voxel_size, radius_metric="median"
    )
    rel_err = abs(V_est - V_true) / V_true
    assert rel_err <= 0.018


def test_capsule_tapered_matches_analytic():
    L, r0, r1 = 10.0, 1.5, 0.5
    voxel_size = 0.12
    skel = skel_capsule(L=L, r0=r0, r1=r1)
    V_true = V_capsule(L, r0, r1)
    V_est = sk.dx.volume(
        skel, include_soma=False, voxel_size=voxel_size, radius_metric="median"
    )
    rel_err = abs(V_est - V_true) / V_true
    assert rel_err <= 0.005


def test_two_disjoint_capsules_sum():
    voxel_size = 0.12
    skel = skel_two_capsules()
    V_true = V_capsule(8.0, 0.8, 0.8) + V_capsule(5.0, 0.5, 0.5)
    V_est = sk.dx.volume(
        skel, include_soma=False, voxel_size=voxel_size, radius_metric="median"
    )
    rel_err = abs(V_est - V_true) / V_true
    # Slightly looser: big span inflates grid; still should be close with fixed h
    assert rel_err <= 0.02


def test_soma_only_matches_analytic():
    a, b, c = 3.0, 2.0, 1.0
    voxel_size = 0.12
    skel = skel_soma_only(a, b, c)
    V_true = V_ellipsoid(a, b, c)
    V_est = sk.dx.volume(
        skel, include_soma=True, voxel_size=voxel_size, radius_metric="median"
    )
    rel_err = abs(V_est - V_true) / V_true
    assert rel_err <= 5e-4


# ======================== tests: bbox slicing ========================


def test_capsule_bbox_interior_slice_equals_cylinder_chunk():
    """
    Capsule aligned with +x. Choose a bbox that slices only the interior
    cylinder (stay ≥ r away from caps), so analytic volume is π r^2 Δx.
    """
    L, r = 10.0, 1.0
    voxel_size = 0.10
    skel = skel_capsule(L=L, r0=r, r1=r, origin=(0, 0, 0), direction=(1, 0, 0))
    bbox = [3.0, 7.0, -2.5, 2.5, -2.5, 2.5]  # Interior slice: x ∈ [3,7]
    V_true = V_cylinder(7.0 - 3.0, r)
    V_est = sk.dx.volume(
        skel,
        bbox=bbox,
        include_soma=False,
        voxel_size=voxel_size,
        radius_metric="median",
    )
    rel_err = abs(V_est - V_true) / V_true
    assert rel_err <= 0.015


def test_bbox_outside_returns_zero():
    skel = skel_capsule(L=10.0, r0=1.0, r1=1.0)
    bbox = [100.0, 105.0, 100.0, 103.0, 100.0, 103.0]  # far away
    V = sk.dx.volume(
        skel, bbox=bbox, include_soma=False, voxel_size=0.12, radius_metric="median"
    )
    assert V == pytest.approx(0.0, abs=1e-12)


def test_bbox_monotonicity_and_enclosing_equivalence():
    skel = skel_capsule(L=10.0, r0=1.0, r1=1.0, origin=(0, 0, 0), direction=(1, 0, 0))
    h = 0.12
    V_no_bbox = sk.dx.volume(
        skel, include_soma=False, voxel_size=h, radius_metric="median"
    )

    # nested bboxes
    small = [-1.0, 3.0, -2.0, 2.0, -2.0, 2.0]
    large = [-2.5, 12.5, -5.0, 5.0, -5.0, 5.0]
    V_small = sk.dx.volume(
        skel, bbox=small, include_soma=False, voxel_size=h, radius_metric="median"
    )
    V_large = sk.dx.volume(
        skel, bbox=large, include_soma=False, voxel_size=h, radius_metric="median"
    )

    assert 0.0 <= V_small <= V_large
    # Enclosing bbox should be very close to no-bbox computation
    rel_err = abs(V_large - V_no_bbox) / V_no_bbox
    assert rel_err <= 0.01


# ======================== tests: edge cases ========================


def test_zero_length_edge_becomes_ball():
    """L=0, r0=r1 ⇒ union is a single ball of radius r."""
    R = 1.2
    voxel_size = 0.08
    skel = skel_capsule(L=0.0, r0=R, r1=R)  # both nodes coincide
    V_true = V_ball(R)
    V_est = sk.dx.volume(
        skel, include_soma=False, voxel_size=voxel_size, radius_metric="median"
    )
    rel_err = abs(V_est - V_true) / V_true
    assert rel_err <= 0.02


def test_zero_length_edge_unequal_radii_becomes_max_ball():
    """L=0, r0!=r1 ⇒ union is a single ball of radius max(r0,r1)."""
    r0, r1 = 0.8, 1.2
    skel = skel_capsule(L=0.0, r0=r0, r1=r1)
    V_true = V_ball(max(r0, r1))
    V_est = sk.dx.volume(
        skel, include_soma=False, voxel_size=0.08, radius_metric="median"
    )
    rel_err = abs(V_est - V_true) / V_true
    assert rel_err <= 0.02


def test_translation_invariance_up_to_phase_jitter():
    """
    Translate by half-voxel along axes; the range across phases should be tiny.
    Useful to detect grid-phase aliasing regressions.
    """
    L, r = 10.0, 1.0
    voxel_size = 0.12
    skel = skel_capsule(L=L, r0=r, r1=r)
    base = sk.dx.volume(
        skel, include_soma=False, voxel_size=voxel_size, radius_metric="median"
    )

    phases = [
        (voxel_size / 2, 0, 0),
        (0, voxel_size / 2, 0),
        (0, 0, voxel_size / 2),
        (voxel_size / 2, voxel_size / 2, voxel_size / 2),
    ]
    vals = [base]
    for d in phases:
        tr = translate_skeleton(skel, d)
        vals.append(
            sk.dx.volume(
                tr, include_soma=False, voxel_size=voxel_size, radius_metric="median"
            )
        )

    V_true = V_capsule(L, r, r)
    rel_range = (max(vals) - min(vals)) / V_true
    # With this resolution, expect < ~1.0% spread across phases
    assert rel_range <= 0.01


def test_refinement_reduces_error_on_average():
    """
    Coarse h vs fine h: fine should be closer to truth (allow tiny slack for jitter).
    """
    L, r = 10.0, 1.0
    skel = skel_capsule(L=L, r0=r, r1=r)
    V_true = V_capsule(L, r, r)
    V_coarse = sk.dx.volume(
        skel, include_soma=False, voxel_size=0.20, radius_metric="median"
    )
    V_fine = sk.dx.volume(
        skel, include_soma=False, voxel_size=0.10, radius_metric="median"
    )
    err_c = abs(V_coarse - V_true) / V_true
    err_f = abs(V_fine - V_true) / V_true
    assert err_f <= err_c + 0.002  # allow 0.2% wiggle for phase jitter


def test_unit_scaling_cubic_consistency():
    """
    Convert µm → nm (×1000), compute volume (nm^3), convert back (÷1e9) and compare.
    """
    L, r = 10.0, 1.0
    voxel_size = 0.12
    sk_um = skel_capsule(L=L, r0=r, r1=r)
    sk_um.set_unit("µm")
    V_um = sk.dx.volume(
        sk_um, include_soma=False, voxel_size=voxel_size, radius_metric="median"
    )

    sk_nm = deepcopy(sk_um)
    sk_nm.convert_unit("nm")  # now coordinates/radii are in nm
    V_nm = sk.dx.volume(
        sk_nm, include_soma=False, voxel_size=voxel_size * 1000, radius_metric="median"
    )
    V_back_um = V_nm / 1e9

    rel_err = abs(V_back_um - V_um) / V_um
    assert rel_err <= 0.01  # loose; grids differ slightly after scaling


def test_include_soma_flag_difference_approx_soma_volume():
    """
    Edge far from soma → volume(include_soma=True) - volume(False) ≈ V_ellipsoid.
    """
    a, b, c = 3.0, 2.0, 1.0
    voxel_size = 0.12
    # Soma at origin; put a capsule far enough away to avoid overlap
    skel = skel_soma_only(a, b, c)
    far = skel_capsule(L=6.0, r0=0.5, r1=0.5, origin=(0, 50, 0))
    nodes = np.vstack([skel.nodes, far.nodes[1:]])
    r = np.concatenate([skel.radii["median"], far.radii["median"][1:]])
    edges = far.edges + len(skel.nodes) - 1
    sk_union = Skeleton(
        soma=skel.soma,
        nodes=nodes,
        radii={"median": r.copy(), "mean": r.copy()},
        edges=edges.astype(np.int64),
        ntype=None,
        node2verts=None,
        vert2node=None,
        meta={},
        extra={},
    )

    V_with = sk.dx.volume(
        sk_union, include_soma=True, voxel_size=voxel_size, radius_metric="median"
    )
    V_without = sk.dx.volume(
        sk_union, include_soma=False, voxel_size=voxel_size, radius_metric="median"
    )
    diff = V_with - V_without
    V_soma = V_ellipsoid(a, b, c)
    rel_err = abs(diff - V_soma) / V_soma
    assert rel_err <= 0.02
    assert diff > 0.0  # including soma should increase volume


def test_include_soma_no_effect_if_bbox_excludes_soma():
    """
    If bbox excludes the soma region, include_soma should be a no-op.
    """
    a, b, c = 2.0, 1.5, 1.0
    # Put a capsule far from soma and then slice tightly around the capsule
    cap = skel_capsule(L=6.0, r0=0.5, r1=0.5, origin=(100, 0, 0))
    soma = skel_soma_only(a, b, c)
    nodes = np.vstack([soma.nodes, cap.nodes[1:]])
    r = np.concatenate([soma.radii["median"], cap.radii["median"][1:]])
    edges = cap.edges + len(soma.nodes) - 1
    sk_union = Skeleton(
        soma=soma.soma,
        nodes=nodes,
        radii={"median": r.copy(), "mean": r.copy()},
        edges=edges.astype(np.int64),
        ntype=None,
        node2verts=None,
        vert2node=None,
        meta={},
        extra={},
    )
    # Bbox around the capsule only
    bbox = [98.0, 106.0, -2.0, 2.0, -2.0, 2.0]
    h = 0.12
    V_true = sk.dx.volume(
        sk_union, include_soma=False, voxel_size=h, bbox=bbox, radius_metric="median"
    )
    V_with = sk.dx.volume(
        sk_union, include_soma=True, voxel_size=h, bbox=bbox, radius_metric="median"
    )
    assert abs(V_with - V_true) / max(V_true, 1e-12) <= 0.01


# ======================== tests: orientation robustness ========================


def test_soma_rotation_invariance():
    """
    Rotating an ellipsoid does not change its volume; estimator should be stable.
    """
    a, b, c = 3.0, 2.0, 1.0
    h = 0.12
    base = skel_soma_only(a, b, c)
    R = rotation_matrix_xyz(0.3, -0.2, 0.5)
    rot = rotate_skeleton(base, R)
    V0 = sk.dx.volume(base, include_soma=True, voxel_size=h, radius_metric="median")
    V1 = sk.dx.volume(rot, include_soma=True, voxel_size=h, radius_metric="median")
    rel_err = abs(V1 - V0) / V0
    assert rel_err <= 2e-3  # tight — soma volume should be robust


def test_capsule_orientation_robustness():
    """
    Change capsule direction; volume should be essentially unchanged.
    """
    L, r = 12.0, 0.9
    h = 0.12
    a = skel_capsule(L=L, r0=r, r1=r, direction=(1, 0, 0))
    b = skel_capsule(
        L=L, r0=r, r1=r, direction=(0.769, 0.535, 0.352)
    )  # normalized inside builder
    Va = sk.dx.volume(a, include_soma=False, voxel_size=h, radius_metric="median")
    Vb = sk.dx.volume(b, include_soma=False, voxel_size=h, radius_metric="median")
    rel_err = abs(Va - Vb) / Va
    assert rel_err <= 0.02


# ======================== tests: branching overlap ========================


def test_branching_subadditivity_and_bounds():
    """
    For a Y junction, union volume must be:
        max(edge volumes) ≤ union ≤ sum(edge volumes).
    This detects double-counting near junctions.
    """
    L, r = 6.0, 0.8
    h = 0.12
    br = skel_branch_y(L=L, r=r)
    V_union = sk.dx.volume(br, include_soma=False, voxel_size=h, radius_metric="median")

    # Volumes of each edge as isolated capsules (same geometry & radii).
    edge1 = skel_capsule(L=L, r0=r, r1=r, origin=(20.0, 0.0, 0.0), direction=(1, 0, 0))
    edge2 = skel_capsule(L=L, r0=r, r1=r, origin=(20.0, 0.0, 0.0), direction=(0, 1, 0))
    V1 = sk.dx.volume(edge1, include_soma=False, voxel_size=h, radius_metric="median")
    V2 = sk.dx.volume(edge2, include_soma=False, voxel_size=h, radius_metric="median")

    # The union should not exceed the sum, and not be less than the larger part.
    assert V_union <= (V1 + V2) + 1e-6
    assert V_union + 1e-6 >= max(V1, V2)


# ======================== tests: radius metric handling ========================


def test_radius_metric_changes_result_when_arrays_differ():
    """
    If 'mean' radii are uniformly scaled vs 'median', the computed volume should change.
    """
    skel = skel_capsule_with_mean_scale(L=10.0, r0=1.0, r1=1.0, mean_scale=1.15)
    h = 0.12
    V_med = sk.dx.volume(skel, include_soma=False, voxel_size=h, radius_metric="median")
    V_mean = sk.dx.volume(skel, include_soma=False, voxel_size=h, radius_metric="mean")
    assert V_mean > V_med  # thicker radii ⇒ larger volume


def test_radius_metric_none_uses_recommendation_if_supported():
    """
    If dx.volume accepts radius_metric=None, ensure it uses Skeleton.recommend_radius().
    Skip gracefully if not supported.
    """
    import inspect

    sig = inspect.signature(sk.dx.volume)
    if "radius_metric" not in sig.parameters:
        pytest.skip("dx.volume has no 'radius_metric' parameter.")
    # Some implementations may not accept None explicitly; try and skip if TypeError.
    skel = skel_capsule_with_mean_scale(L=10.0, r0=1.0, r1=1.0, mean_scale=1.0)
    try:
        V_none = sk.dx.volume(
            skel, include_soma=False, voxel_size=0.12, radius_metric=None
        )
    except TypeError:
        pytest.skip("dx.volume(radius_metric=None) not supported.")
    # With equal mean/median here, recommendation commonly picks 'mean' (p75<1.02) or 'median' —
    # either way V_none should equal V_med within numerical noise.
    V_med = sk.dx.volume(
        skel, include_soma=False, voxel_size=0.12, radius_metric="median"
    )
    assert abs(V_none - V_med) / V_med <= 0.005


# ======================== tests: component selection via bbox ========================


def test_bbox_selects_single_component_in_forest():
    """
    Two well-separated components; bbox around one should recover that component's volume.
    """
    forest = skel_two_capsules()
    h = 0.08
    V_all = sk.dx.volume(
        forest, include_soma=False, voxel_size=h, radius_metric="median"
    )
    # Bbox around the first capsule near y=0 with slack
    bbox_A = [-2.0, 10.0, -2.0, 2.0, -2.0, 2.0]
    V_A = sk.dx.volume(
        forest, bbox=bbox_A, include_soma=False, voxel_size=h, radius_metric="median"
    )
    # Bbox around the second capsule near y=100
    bbox_B = [-2.0, 8.0, 98.0, 102.0, -2.0, 2.0]
    V_B = sk.dx.volume(
        forest, bbox=bbox_B, include_soma=False, voxel_size=h, radius_metric="median"
    )
    V_true_A = V_capsule(8.0, 0.8, 0.8)
    V_true_B = V_capsule(5.0, 0.5, 0.5)
    assert abs(V_A - V_true_A) / V_true_A <= 0.02
    assert abs(V_B - V_true_B) / V_true_B <= 0.02
    # The sum of sliced parts should be ≈ total (tiny differences from bbox padding)
    assert abs((V_A + V_B) - V_all) / V_all <= 0.02


# ======================== tests: API guardrails ========================


def test_invalid_voxel_size_raises():
    skel = skel_capsule(L=5.0, r0=0.5, r1=0.5)
    with pytest.raises(ValueError):
        sk.dx.volume(skel, include_soma=False, voxel_size=0.0, radius_metric="median")
    with pytest.raises(ValueError):
        sk.dx.volume(skel, include_soma=False, voxel_size=-0.1, radius_metric="median")


def test_invalid_bbox_raises():
    skel = skel_capsule(L=5.0, r0=0.5, r1=0.5)
    # Wrong length
    with pytest.raises(ValueError):
        sk.dx.volume(
            skel,
            include_soma=False,
            voxel_size=0.12,
            bbox=[0, 1, 0, 1, 0],
            radius_metric="median",
        )
    # Ill-ordered bounds
    with pytest.raises(ValueError):
        sk.dx.volume(
            skel,
            include_soma=False,
            voxel_size=0.12,
            bbox=[1, 0, -1, 1, -1, 1],
            radius_metric="median",
        )


# ======================== tests: additional taper limit ========================


def test_cone_limit_r1_zero_matches_analytic():
    """
    One end radius is zero: union = cone (frustum) + hemisphere at the wide end.
    """
    L, r0, r1 = 10.0, 1.2, 0.0
    h = 0.12
    skel = skel_capsule(L=L, r0=r0, r1=r1)
    V_true = V_capsule(L, r0, r1)  # valid in this regime
    V_est = sk.dx.volume(skel, include_soma=False, voxel_size=h, radius_metric="median")
    rel_err = abs(V_est - V_true) / V_true
    assert rel_err <= 0.008
