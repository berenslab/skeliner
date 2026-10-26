import matplotlib
import numpy as np
import pytest

import skeliner as sk
from skeliner.dataclass import Skeleton, Soma
from skeliner.plot import details

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _toy_skeleton() -> Skeleton:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    radii = np.array([1.0, 0.6, 0.4], dtype=float)
    soma = Soma.from_sphere(
        center=nodes[0],
        radius=1.0,
        verts=np.array([0, 1, 2], dtype=np.int64),
    )
    return Skeleton(
        soma=soma,
        nodes=nodes,
        radii={"mean": radii.copy(), "median": radii.copy()},
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        ntype=None,
        node2verts=None,
        vert2node=None,
    )


def test_details_with_mesh_none_and_soma_verts_does_not_raise():
    skel = _toy_skeleton()
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        details(
            skel,
            mesh=None,
            draw_nodes=True,
            draw_edges=True,
            xlim=(-1.0, 3.0),
            ylim=(-1.0, 2.0),
            ax=ax,
        )
    finally:
        plt.close(fig)


def test_the_viewer_s_old_names_warn_and_reach_sk_viewer():
    for name in ("view", "view3d", "view_contacts", "stop_viewer"):
        with pytest.warns(DeprecationWarning, match=f"sk.viewer.{name}"):
            moved = getattr(sk.plot, name)
        assert moved is getattr(sk.viewer, name)
    with pytest.warns(DeprecationWarning, match="vis2d"):
        assert sk.plot.vis2d.diagnose_soma is sk.plot.diagnose_soma
    with pytest.raises(AttributeError):
        sk.plot.no_such_figure
