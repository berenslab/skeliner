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

# the figures (`plot` imports matplotlib) and the viewer, on first use
_SUBMODULES = ("plot", "viewer")
_PLOTS = {
    "view3d": ("viewer", "view3d"),
    "plot2d": ("plot", "projection"),
    "plot3v": ("plot", "threeviews"),
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _PLOTS:
        module, attr = _PLOTS[name]
        value = getattr(importlib.import_module(f".{module}", __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBMODULES, *_PLOTS})


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
    "viewer",
    "preprocess",
    "threads",
    "backend",
]
