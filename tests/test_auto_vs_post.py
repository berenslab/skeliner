from pathlib import Path

import numpy as np
import pytest

from skeliner import post, skeletonize
from skeliner.io import load_mesh

DATA_PATH = Path(__file__).parent / "data" / "60427.obj"


@pytest.fixture(scope="module")
def mesh_60427():
    return load_mesh(DATA_PATH)


def _manual_post_pipeline(mesh):
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()

    skel_core = skeletonize(mesh, postprocess=False, verbose=False)
    skel_detected = post.detect_soma(
        skel_core,
        radius_key="median",
        verbose=False,
        mesh_vertices=mesh_vertices,
    )
    has_soma = skel_detected is not skel_core
    skel_cur = skel_detected

    if has_soma:
        skel_cur = post.merge_near_soma_nodes(
            skel_cur,
            mesh_vertices=mesh_vertices,
            radius_key="median",
            near_factor=1.2,
            fat_factor=0.20,
            verbose=False,
        )

    post.bridge_gaps(
        skel_cur,
        bridge_max_factor=None,
        bridge_recalc_after=None,
        rebuild_mst=False,
        verbose=False,
    )
    skel_cur = post.rebuild_mst(skel_cur, verbose=False)

    if has_soma:
        skel_cur = post.prune_neurites(
            skel_cur,
            mesh_vertices=mesh_vertices,
            tip_extent_factor=1.2,
            stem_extent_factor=3.0,
            drop_single_node_branches=True,
            verbose=False,
        )

    return skel_cur


def _assert_soma_equal(auto_soma, post_soma):
    np.testing.assert_allclose(auto_soma.center, post_soma.center)
    np.testing.assert_allclose(auto_soma.axes, post_soma.axes)
    np.testing.assert_allclose(auto_soma.R, post_soma.R)
    if auto_soma.verts is None or post_soma.verts is None:
        assert auto_soma.verts is None and post_soma.verts is None
    else:
        np.testing.assert_array_equal(
            np.sort(np.asarray(auto_soma.verts)), np.sort(np.asarray(post_soma.verts))
        )


def _assert_skeleton_equal(auto_skel, post_skel):
    np.testing.assert_allclose(auto_skel.nodes, post_skel.nodes)
    np.testing.assert_array_equal(auto_skel.edges, post_skel.edges)

    assert set(auto_skel.radii) == set(post_skel.radii)
    for key in auto_skel.radii:
        np.testing.assert_allclose(auto_skel.radii[key], post_skel.radii[key])

    if auto_skel.ntype is None or post_skel.ntype is None:
        assert auto_skel.ntype is None and post_skel.ntype is None
    else:
        np.testing.assert_array_equal(auto_skel.ntype, post_skel.ntype)

    if auto_skel.node2verts is None or post_skel.node2verts is None:
        assert auto_skel.node2verts is None and post_skel.node2verts is None
    else:
        assert len(auto_skel.node2verts) == len(post_skel.node2verts)
        for arr_auto, arr_post in zip(auto_skel.node2verts, post_skel.node2verts):
            np.testing.assert_array_equal(
                np.asarray(arr_auto, dtype=np.int64),
                np.asarray(arr_post, dtype=np.int64),
            )

    if auto_skel.vert2node is None or post_skel.vert2node is None:
        assert auto_skel.vert2node is None and post_skel.vert2node is None
    else:
        assert auto_skel.vert2node == post_skel.vert2node

    _assert_soma_equal(auto_skel.soma, post_skel.soma)


def test_manual_pipeline_matches_auto(mesh_60427):
    skel_auto = skeletonize(mesh_60427.copy(), verbose=False)
    skel_post = _manual_post_pipeline(mesh_60427.copy())
    _assert_skeleton_equal(skel_auto, skel_post)


def test_prune_neurites_without_mesh_vertices_logs_warning(mesh_60427, capsys):
    skel_auto = skeletonize(mesh_60427.copy(), verbose=False)
    skel_repruned = post.prune_neurites(
        skel_auto,
        mesh_vertices=None,
        tip_extent_factor=1.2,
        stem_extent_factor=3.0,
        drop_single_node_branches=True,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "mesh_vertices not provided" in captured.out
    assert len(skel_repruned.nodes) > 0
