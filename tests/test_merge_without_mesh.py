import numpy as np
import pytest

from skeliner import post
from skeliner.dataclass import Skeleton, Soma


def _make_simple_skeleton() -> Skeleton:
    nodes = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float64)
    radii = {
        "median": np.asarray([2.0, 0.8], dtype=np.float64),
    }
    edges = np.asarray([[0, 1]], dtype=np.int64)
    soma = Soma.from_sphere(center=np.zeros(3), radius=2.0, verts=None)
    return Skeleton(soma=soma, nodes=nodes, radii=radii, edges=edges, ntype=None)


def test_merge_near_soma_without_mesh_logs_and_merges():
    skel = _make_simple_skeleton()

    merged = post.merge_near_soma_nodes(skel, mesh_vertices=None)

    assert merged.nodes.shape[0] == 1
    assert merged.node2verts is None
    assert merged.vert2node is None
    np.testing.assert_allclose(merged.nodes[0], np.zeros(3))
    np.testing.assert_allclose(
        merged.soma.axes, np.full(3, merged.soma.spherical_radius)
    )
