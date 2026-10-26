import importlib
from importlib.metadata import PackageNotFoundError, version

from . import batch, dx, io, pair, post, pre
from .dataclass import Mesh, Skeleton, Soma, register_skeleton_methods
from .pre import preprocess
from ._backend import backend
from ._threads import threads
from .skeletonize import skeletonize

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

register_skeleton_methods(dx, getattr(dx, "__skeleton__", None))
register_skeleton_methods(post, getattr(post, "__skeleton__", None))

# the plotting functions, from `plot` (it imports matplotlib) on first use
_PLOTS = {
    "view3d": ("plot", "view3d"),
    "plot2d": ("plot.vis2d", "projection"),
    "plot3v": ("plot.vis2d", "threeviews"),
}


def __getattr__(name: str):
    if name == "plot":
        return importlib.import_module(".plot", __name__)
    if name in _PLOTS:
        module, attr = _PLOTS[name]
        value = getattr(importlib.import_module(f".{module}", __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "plot", *_PLOTS})


__all__ = [
    "Mesh",
    "Skeleton",
    "Soma",
    "skeletonize",
    "plot2d",
    "plot3v",
    "view3d",
    "io",
    "dx",
    "batch",
    "post",
    "pair",
    "pre",
    "plot",
    "preprocess",
    "threads",
    "backend",
]
