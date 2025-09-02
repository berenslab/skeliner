from importlib.metadata import PackageNotFoundError, version

from . import dx, io, pair, plot, post
from .core import Skeleton, skeletonize
from .plot.vis2d import projection as plot2d
from .plot.vis2d import threeviews as plot3v
from .plot.vis3d import view3d

try:
    __version__ = version(__name__)       
except PackageNotFoundError:              
    __version__ = "0.0.0.dev0"

__all__ = [
    "Skeleton",
    "skeletonize",
    "plot2d",
    "plot3v",
    "view3d",
    "io",
    "dx",
    "post",
    "pair",
    "plot",
]