import numpy as np

from skeliner import post, Skeleton


def make_simple_skeleton(n_nodes=3):
    # Straight line nodes, arbitrary positions
    nodes = np.stack([np.arange(n_nodes), np.zeros(n_nodes), np.zeros(n_nodes)], axis=1).astype(float)
    edges = np.array([[i, i + 1] for i in range(n_nodes - 1)], dtype=np.int64)
    # Provide a fallback radius vector
    radii = {
        "median": np.linspace(0.4, 0.6, n_nodes).astype(float),
    }
    # node2verts: assign 50 verts to each non-soma, soma gets some too
    node2verts = [
        np.arange(0, 40, dtype=np.int64),        # node 0
        np.arange(40, 140, dtype=np.int64),      # node 1
        np.arange(140, 200, dtype=np.int64),     # node 2
    ]
    # mesh vertices cloud (values unused by mocked distance but must exist)
    V = 200
    mesh_vertices = np.random.RandomState(0).randn(V, 3).astype(float)

    skel = Skeleton(
        soma=None,  # keep soma None to exercise standard path for node 0
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=np.full(n_nodes, 3, np.int8),
        node2verts=node2verts,
        vert2node={int(v): int(i) for i, vs in enumerate(node2verts) for v in vs},
        meta={},
        extra={"mesh_vertices": mesh_vertices},
    )
    return skel


def test_calibrate_radii_bimodal_outer_tail(monkeypatch):
    skel = make_simple_skeleton(n_nodes=3)

    # Fallback radii for reference
    r_fb = skel.radii[skel.recommend_radius()[0]].copy()

    # Synthetic distance generator depending on allowed_nodes
    def fake_distance(skel_arg, pts, *, mode, radius_metric, allowed_nodes):
        # Identify target node index
        node = int(allowed_nodes[0])
        rng = np.random.RandomState(42 + node)
        n = len(pts)
        if node == 1:
            # Bimodal: inner cloud ~0.2, outer shell ~1.0 (dominant outer few)
            k_outer = max(10, n // 5)
            outer = rng.normal(loc=1.0, scale=0.05, size=k_outer)
            inner = rng.normal(loc=0.20, scale=0.03, size=n - k_outer)
            vals = np.concatenate([inner, outer])
            rng.shuffle(vals)
            return vals
        else:
            # Unimodal around ~0.5
            return rng.normal(loc=0.50, scale=0.03, size=n)

    # Monkeypatch dx.distance
    import skeliner.dx as dx_mod

    monkeypatch.setattr(dx_mod, "distance", fake_distance)

    post.calibrate_radii(skel, points=None, store_key="calibrated", mode="centerline")

    r_cal = skel.radii["calibrated"]

    # Node 1 should be close to outer mode (~1.0) and never below fallback
    assert r_cal[1] >= r_fb[1] - 1e-6
    assert 0.85 <= r_cal[1] <= 1.1

    # Other nodes should stay near ~0.5 and >= fallback due to clamp
    assert r_cal[0] >= r_fb[0] - 1e-6
    assert 0.4 <= r_cal[0] <= 0.7
    assert r_cal[2] >= r_fb[2] - 1e-6
    assert 0.4 <= r_cal[2] <= 0.7


def test_calibrate_radii_copy_if_missing():
    # Skeleton without node2verts or points → should copy fallback
    n_nodes = 2
    nodes = np.array([[0, 0, 0], [1, 0, 0]], float)
    edges = np.array([[0, 1]], np.int64)
    radii = {"median": np.array([0.7, 0.5], float)}

    skel = Skeleton(
        soma=None,
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=np.array([3, 3], np.int8),
        node2verts=None,
        vert2node=None,
        meta={},
        extra={},
    )

    post.calibrate_radii(skel, points=None, store_key="calibrated", copy_if_missing=True)

    assert "calibrated" in skel.radii
    assert np.allclose(skel.radii["calibrated"], skel.radii["median"]) 
    assert skel.extra.get("calibration", {}).get("status") == "copied"
