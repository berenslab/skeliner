from __future__ import annotations

import numpy as np
import pytest

from skeliner import batch, dx
from skeliner.core import Skeleton, Soma


def make_line_skeleton(base: np.ndarray, *, skel_id: str | None = None) -> Skeleton:
    base = np.asarray(base, dtype=np.float64)
    nodes = np.vstack([base, base + np.array([10.0, 0.0, 0.0])])
    radii = np.array([1.0, 1.0], dtype=np.float64)
    edges = np.array([[0, 1]], dtype=np.int64)
    soma = Soma.from_sphere(center=base, radius=1.0, verts=None)
    meta = {"unit": "nm"}
    if skel_id is not None:
        meta["id"] = skel_id
    return Skeleton(
        soma=soma,
        nodes=nodes,
        radii={"median": radii.copy(), "mean": radii.copy()},
        edges=edges,
        ntype=None,
        node2verts=None,
        vert2node=None,
        meta=meta,
        extra={},
    )


@pytest.fixture()
def simple_skeletons() -> list[Skeleton]:
    return [
        make_line_skeleton(np.array([0.0, 0.0, 0.0]), skel_id="proximal"),
        make_line_skeleton(np.array([100.0, 0.0, 0.0]), skel_id="distal"),
        make_line_skeleton(np.array([-80.0, 0.0, 0.0]), skel_id="left"),
    ]


def test_distance_matrix_matches_manual(simple_skeletons):
    points = np.array(
        [
            [5.0, 0.0, 0.0],
            [105.0, 0.0, 0.0],
            [-75.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    dm = batch.distance_matrix(simple_skeletons, points, point_unit="nm")
    assert dm.shape == (4, 3)

    # Compare against direct dx.distance calls
    manual = np.vstack(
        [dx.distance(skel, points, point_unit="nm") for skel in simple_skeletons]
    ).T
    assert np.allclose(dm, manual)


def test_nearest_single_point(simple_skeletons):
    distance, index = batch.nearest_skeletons(
        simple_skeletons,
        np.array([102.5, 0.0, 0.0]),
        point_unit="nm",
        structured=False,
    )
    assert distance.shape == (1,)
    assert index.shape == (1,)
    assert index[0] == 1  # middle skeleton
    expected = dx.distance(
        simple_skeletons[1], np.array([102.5, 0.0, 0.0]), point_unit="nm"
    )
    assert distance[0] == pytest.approx(expected)


def test_nearest_multiple_points(simple_skeletons):
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
            [-60.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    distances, indices = batch.nearest_skeletons(
        simple_skeletons,
        points,
        k=2,
        point_unit="nm",
        structured=False,
    )
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2)

    # For each point, ensure returned skeletons are sorted by distance
    full = batch.distance_matrix(simple_skeletons, points, point_unit="nm")
    expected_idx = np.argsort(full, axis=1)[:, :2]
    assert np.array_equal(indices, expected_idx)
    expected_dist = np.take_along_axis(full, expected_idx, axis=1)
    assert np.allclose(distances, expected_dist)


def test_structured_single_point(simple_skeletons):
    summary = batch.nearest_skeletons(
        simple_skeletons,
        np.array([102.5, 0.0, 0.0]),
        point_unit="nm",
        k=2,
        structured=True,
    )
    assert isinstance(summary, dict)
    assert summary["point_index"] == 0
    assert summary["matches"][0]["skeleton_id"] == "distal"
    assert summary["matches"][1]["skeleton_id"] == "proximal"
    assert summary["matches"][0]["skeleton_index"] == 1
    assert summary["matches"][0]["distance"] < summary["matches"][1]["distance"]


def test_structured_multiple_points(simple_skeletons):
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    summary = batch.nearest_skeletons(
        simple_skeletons,
        points,
        k=1,
        point_unit="nm",
        structured=True,
    )
    assert isinstance(summary, list)
    assert summary[0]["matches"][0]["skeleton_id"] == "proximal"
    assert summary[1]["matches"][0]["skeleton_id"] == "distal"
    assert "all_distances" not in summary[0]
    assert summary[0]["matches"][0]["skeleton_index"] == 0
    assert summary[1]["matches"][0]["skeleton_index"] == 1


def test_structured_with_full_distances(simple_skeletons):
    points = np.array([[10.0, 0.0, 0.0], [110.0, 0.0, 0.0]])
    summary = batch.nearest_skeletons(
        simple_skeletons,
        points,
        structured=True,
        return_all=True,
        point_unit="nm",
    )
    assert isinstance(summary, list)
    assert np.allclose(
        np.asarray(summary[0]["all_distances"]),
        batch.distance_matrix(simple_skeletons, points[0], point_unit="nm"),
    )
    assert len(summary[0]["all_distances"]) == len(simple_skeletons)
    assert len(summary[1]["all_distances"]) == len(simple_skeletons)


def test_nearest_k_clamped(simple_skeletons):
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    distances, indices = batch.nearest_skeletons(
        simple_skeletons,
        points,
        k=10,
        point_unit="nm",
        structured=False,
    )
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    # Should be sorted distances covering all skeletons
    full = batch.distance_matrix(simple_skeletons, points, point_unit="nm")
    expected_idx = np.argsort(full, axis=1)
    assert np.array_equal(indices, expected_idx)
    expected_dist = np.take_along_axis(full, expected_idx, axis=1)
    assert np.allclose(distances, expected_dist)


def test_return_all_matrix(simple_skeletons):
    points = np.array([[10.0, 0.0, 0.0], [110.0, 0.0, 0.0]])
    distances, indices, matrix = batch.nearest_skeletons(
        simple_skeletons,
        points,
        k=1,
        return_all=True,
        structured=False,
        point_unit="nm",
    )
    assert matrix.shape == (2, 3)
    assert np.allclose(distances[:, 0], np.min(matrix, axis=1))
    assert np.array_equal(indices[:, 0], np.argmin(matrix, axis=1))


def test_invalid_inputs(simple_skeletons):
    with pytest.raises(ValueError):
        batch.nearest_skeletons(simple_skeletons, np.array([0.0]), point_unit="nm")

    with pytest.raises(ValueError):
        batch.nearest_skeletons([], np.zeros(3), point_unit="nm")

    with pytest.raises(ValueError):
        batch.nearest_skeletons(simple_skeletons, np.zeros(3), k=0, point_unit="nm")

    empty_skel = make_line_skeleton(np.array([0.0, 0.0, 0.0]))
    empty_skel.nodes = np.empty((0, 3))
    empty_skel.radii["median"] = np.empty(0, dtype=float)
    empty_skel.radii["mean"] = np.empty(0, dtype=float)
    empty_skel.edges = np.empty((0, 2), dtype=np.int64)
    empty_skel.ntype = np.empty(0, dtype=np.int8)
    with pytest.raises(ValueError):
        batch.distance_matrix([empty_skel], np.zeros(3), point_unit="nm")
