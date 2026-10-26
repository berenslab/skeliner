"""The interactive web viewer: meshes, skeletons and their annotations in
the browser, served by a small local app.  Its dependencies load on first
use."""

__all__ = [
    "view",
    "view3d",
    "view_contacts",
    "stop_viewer",
]


def view(mesh_path=None, **kwargs):
    """Launch interactive mesh viewer. Lazy import to avoid heavy deps."""
    from .viewer import view as _view

    return _view(mesh_path, **kwargs)


def view3d(skels=None, meshes=None, **kwargs):
    """Launch viewer with pre-loaded skeletons and/or meshes."""
    from .viewer import view3d as _view3d

    return _view3d(skels, meshes, **kwargs)


def view_contacts(A, B, contacts, **kwargs):
    """Visualize two meshes with contact-site overlays."""
    from .viewer import view_contacts as _view_contacts

    return _view_contacts(A, B, contacts, **kwargs)


def stop_viewer():
    """Stop the background viewer server (Jupyter only)."""
    from .viewer import stop_viewer as _stop

    _stop()
