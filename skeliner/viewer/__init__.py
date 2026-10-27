"""The interactive web viewer: meshes, skeletons and their annotations in
the browser, served by a small local app.  Its dependencies load on first
use."""

import warnings

__all__ = [
    "start",
    "stop",
    "view3d",
    "view_contacts",
]


def start(mesh_path=None, **kwargs):
    """Start the viewer, empty or with the mesh at *mesh_path*; in a
    notebook it runs in the background until :func:`stop`."""
    from .viewer import start as _start

    return _start(mesh_path, **kwargs)


def stop(port=None):
    """Stop the background viewer on *port*, or every one."""
    from .viewer import stop as _stop

    _stop(port)


def view3d(skels=None, meshes=None, **kwargs):
    """Launch viewer with pre-loaded skeletons and/or meshes."""
    from .viewer import view3d as _view3d

    return _view3d(skels, meshes, **kwargs)


def view_contacts(A, B, contacts=None, **kwargs):
    """Visualize two meshes with contact-site overlays."""
    from .viewer import view_contacts as _view_contacts

    return _view_contacts(A, B, contacts, **kwargs)


# what `start` and `stop` were called before
_RENAMED = {"view": "start", "stop_viewer": "stop"}


def __getattr__(name: str):
    if name in _RENAMED:
        new = _RENAMED[name]
        warnings.warn(
            f"sk.viewer.{name}() is deprecated, use sk.viewer.{new}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
