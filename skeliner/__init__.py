from importlib.metadata import PackageNotFoundError, version

from . import batch, dx, io, pair, plot, post
from .dataclass import Skeleton, register_skeleton_methods
from .plot.vis2d import projection as plot2d
from .plot.vis2d import threeviews as plot3v
from .plot.vis3d import view3d
from .skeletonize import skeletonize

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

register_skeleton_methods(dx, getattr(dx, "__skeleton__", None))
register_skeleton_methods(post, getattr(post, "__skeleton__", None))

__all__ = [
    "Skeleton",
    "skeletonize",
    "plot2d",
    "plot3v",
    "view3d",
    "io",
    "dx",
    "batch",
    "post",
    "pair",
    "plot",
]
