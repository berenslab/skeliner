import numpy as np
import pytest

from skeliner import dx


class MiniSkel:
    """Minimal skeleton carrying only what's needed by dx.total_path_length."""

    def __init__(self, nodes, edges):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.edges = np.asarray(edges, dtype=int)


# ---------------------------
# Helpers for brute checking
# ---------------------------


def _brute_seg_len_in_box(a, b, lo, hi, n=4000):
    """
    Approximate length of segment [a,b] inside [lo,hi] by midpoint sampling
    with n equal subsegments. Error <= |a-b| / n.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)

    v = b - a
    L = np.linalg.norm(v)
    if L == 0.0:
        return 0.0
    step = L / n
    # midpoints along the segment
    ts = (np.arange(n) + 0.5) / n
    mids = a[None, :] + ts[:, None] * v[None, :]
    inside = np.all((mids >= lo) & (mids <= hi), axis=1)
    return float(inside.sum()) * step


# ---------------------------
# Exact, closed-form cases
# ---------------------------


def test_tpl_full_no_bbox_single_edge():
    nodes = [(0, 0, 0), (10, 0, 0)]
    edges = [(0, 1)]
    sk = MiniSkel(nodes, edges)
    L = dx.total_path_length(sk)
    assert L == pytest.approx(10.0, abs=1e-12)


def test_tpl_bbox_simple_clipping():
    nodes = [(0, 0, 0), (10, 0, 0)]
    edges = [(0, 1)]
    sk = MiniSkel(nodes, edges)
    # keep only x in [2, 8] → length 6
    bbox = [2, 8, -1, 1, -1, 1]
    L = dx.total_path_length(sk, bbox=bbox)
    assert L == pytest.approx(6.0, abs=1e-12)


def test_tpl_bbox_no_intersection_zero():
    nodes = [(0, 0, 0), (10, 0, 0)]
    edges = [(0, 1)]
    sk = MiniSkel(nodes, edges)
    bbox = [20, 30, -1, 1, -1, 1]
    L = dx.total_path_length(sk, bbox=bbox)
    assert L == 0.0


def test_tpl_multi_edge_mixed_clipping_and_outside():
    # Segments:
    # 1) (0,0,0) -> (3,4,0)  length 5; inside bbox only t in [0.75,1] → 0.25*5 = 1.25
    # 2) (3,4,0) -> (3,4,12) length 12; inside z in [0,10] → 10
    # 3) (3,4,12) -> (-1,4,12) length 4; z=12 outside → 0
    nodes = [(0, 0, 0), (3, 4, 0), (3, 4, 12), (-1, 4, 12)]
    edges = [(0, 1), (1, 2), (2, 3)]
    sk = MiniSkel(nodes, edges)
    bbox = [2, 3.5, 3, 5, 0, 10]
    L, info = dx.total_path_length(sk, bbox=bbox, return_details=True)
    assert L == pytest.approx(11.25, abs=1e-12)
    assert info["clipped"] is True
    assert info["edges_total"] == 3
    assert info["edges_intersected"] == 2


def test_tpl_zero_length_edges_are_ignored():
    nodes = [(1, 2, 3), (1, 2, 3), (1, 2, 3)]
    edges = [(0, 1), (1, 2)]
    sk = MiniSkel(nodes, edges)
    assert dx.total_path_length(sk) == 0.0
    assert dx.total_path_length(sk, bbox=[0, 2, 0, 3, 0, 4]) == 0.0


def test_tpl_edges_on_bbox_boundary_count_as_inside():
    # Segment lies exactly on y=0 plane which is the bbox boundary.
    nodes = [(0, 0, 0), (1, 0, 0)]
    edges = [(0, 1)]
    sk = MiniSkel(nodes, edges)
    bbox = [0, 1, 0, 1, -1, 1]
    L = dx.total_path_length(sk, bbox=bbox)
    assert L == pytest.approx(1.0, abs=1e-12)


def test_tpl_bbox_forms_equivalent():
    nodes = [(0, 0, 0), (10, 0, 0)]
    edges = [(0, 1)]
    sk = MiniSkel(nodes, edges)
    bbox_list = [2, 8, -1, 1, -1, 1]
    bbox_tuple = ((2, -1, -1), (8, 1, 1))
    L1 = dx.total_path_length(sk, bbox=bbox_list)
    L2 = dx.total_path_length(sk, bbox=bbox_tuple)
    assert L1 == pytest.approx(6.0, abs=1e-12)
    assert L2 == pytest.approx(6.0, abs=1e-12)


# ---------------------------
# Randomized brute-force cross-check
# ---------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_tpl_random_segments_vs_bruteforce(seed):
    rng = np.random.default_rng(seed)

    # random AABB
    c = rng.uniform(-2.0, 2.0, size=3)
    half = rng.uniform(0.5, 3.0, size=3)
    lo = c - half
    hi = c + half
    bbox = (lo, hi)

    # test 10 random segments in this box
    for _ in range(10):
        a = rng.uniform(-5.0, 5.0, size=3)
        b = rng.uniform(-5.0, 5.0, size=3)
        sk = MiniSkel([a, b], [(0, 1)])

        L_exact = dx.total_path_length(sk, bbox=bbox)
        L_brute = _brute_seg_len_in_box(a, b, lo, hi, n=4000)

        # error <= |a-b|/n; be a bit generous for numeric noise
        L = float(np.linalg.norm(b - a))
        tol = max(1e-5, L / 2000.0)
        assert L_exact == pytest.approx(L_brute, abs=tol)
