"""Shared structural dataclasses used across skeliner."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Tuple

import numpy as np

from ._state import VertexMap

if TYPE_CHECKING:
    import igraph as ig

    from . import dx as _dx_mod
    from . import post as _post_mod

__all__ = [
    "Mesh",
    "Discarded",
    "MeshComponents",
    "Neurites",
    "Soma",
    "Skeleton",
    "ContactSeeds",
    "ProxySites",
    "ContactSites",
    "register_skeleton_methods",
]


class Mesh:
    """A triangle mesh: *vertices* ``(N, 3)`` float64 and *faces* ``(F, 3)``
    int64 vertex ids, with the arrays skeliner derives from them — face
    normals, centres and areas, the unique edges and their lengths, the face
    adjacency — each computed once, on the active path, equal to trimesh's.

    The arrays are copies and read-only, so what is derived from them stays
    true: edit a mesh by making a new one. :meth:`Mesh.of` makes one of
    anything with ``vertices`` and ``faces`` (a ``trimesh.Trimesh``, say);
    :meth:`to_trimesh` goes the other way when trimesh is installed.
    """

    __slots__ = ("_vertices", "_faces", "_derived")

    def __init__(self, vertices, faces):
        vertices = np.array(vertices, dtype=np.float64, order="C")
        if vertices.size == 0:
            vertices = vertices.reshape(0, 3)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), not {vertices.shape}")
        faces = np.array(faces, dtype=np.int64, order="C")
        if faces.size == 0:
            faces = faces.reshape(0, 3)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"expected (F, 3) array, got shape {list(faces.shape)}")
        faces, _ = _kernels().mesh.checked_faces(faces, len(vertices))
        vertices.flags.writeable = False
        faces.flags.writeable = False
        self._vertices = vertices
        self._faces = faces
        self._derived: dict[str, np.ndarray] = {}

    @classmethod
    def _adopt(cls, vertices: np.ndarray, faces: np.ndarray) -> "Mesh":
        """A mesh of arrays made for it — float64 ``(N, 3)`` vertices and
        int64 ``(F, 3)`` faces already checked against them — without
        copying them."""
        mesh = cls.__new__(cls)
        vertices.flags.writeable = False
        faces.flags.writeable = False
        mesh._vertices = vertices
        mesh._faces = faces
        mesh._derived = {}
        return mesh

    @classmethod
    def of(cls, mesh) -> "Mesh":
        """*mesh* as a :class:`Mesh`: itself when it is one, else a copy of its
        ``vertices`` and ``faces`` with the copies of a vertex along a seam
        merged, as :func:`skeliner.io.load_mesh` reads a file — every face
        names the first vertex at each position that a seam joins it to (a
        mesh stored in chunks copies the vertices along every seam, which
        cuts it there; vertices that only touch stay apart), and a face the
        merge made a copy of another collapses to ``[0, 0, 0]``. Vertices and
        faces keep their places."""
        if isinstance(mesh, cls):
            return mesh
        try:
            vertices, faces = mesh.vertices, mesh.faces
        except AttributeError:
            raise TypeError(
                f"a mesh has vertices and faces; {type(mesh).__name__} has not"
            ) from None
        made = cls(vertices, faces)
        merged = coincident_merged(made._vertices, made._faces)
        if merged is made._faces:
            return made
        return cls._adopt(made._vertices, merged)

    def __reduce__(self):
        return (Mesh, (self._vertices, self._faces))

    def __repr__(self) -> str:
        return (
            f"<skeliner.Mesh: {len(self._vertices):,} vertices, "
            f"{len(self._faces):,} faces>"
        )

    @property
    def vertices(self) -> np.ndarray:
        """``(N, 3)`` float64 vertex positions, read-only."""
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        """``(F, 3)`` int64 vertex ids per face, read-only."""
        return self._faces

    def _get(self, name: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
        out = self._derived.get(name)
        if out is None:
            out = np.asarray(compute())
            out.flags.writeable = False
            self._derived[name] = out
        return out

    @property
    def triangles(self) -> np.ndarray:
        """``(F, 3, 3)`` each face's vertex positions."""
        return self._get(
            "triangles",
            lambda: _kernels().mesh.triangles(self._vertices, self._faces),
        )

    @property
    def triangles_cross(self) -> np.ndarray:
        """``(F, 3)`` each face's ``(v1 - v0) x (v2 - v1)``."""
        return self._get(
            "triangles_cross",
            lambda: _kernels().mesh.face_crosses(self._vertices, self._faces),
        )

    @property
    def face_normals(self) -> np.ndarray:
        """``(F, 3)`` unit normals, zero for a face too small to have one."""
        return self._get(
            "face_normals", lambda: _kernels().mesh.face_normals(self.triangles_cross)
        )

    @property
    def area_faces(self) -> np.ndarray:
        """``(F,)`` each face's area."""
        return self._get(
            "area_faces", lambda: _kernels().mesh.face_areas(self.triangles_cross)
        )

    @property
    def triangles_center(self) -> np.ndarray:
        """``(F, 3)`` each face's centroid."""
        return self._get(
            "triangles_center",
            lambda: _kernels().mesh.face_centers(self._vertices, self._faces),
        )

    @property
    def edges_unique(self) -> np.ndarray:
        """``(E, 2)`` every distinct edge as ``[low, high]``, by ``high``
        then ``low``."""
        return self._get(
            "edges_unique",
            lambda: _kernels().mesh.unique_edges(self._faces, len(self._vertices)),
        )

    @property
    def edges_unique_length(self) -> np.ndarray:
        """``(E,)`` the length of each of :attr:`edges_unique`."""
        return self._get(
            "edges_unique_length",
            lambda: _kernels().mesh.edge_lengths(self._vertices, self.edges_unique),
        )

    @property
    def face_adjacency(self) -> np.ndarray:
        """``(M, 2)`` the two faces of each edge exactly two faces share,
        lower first, by the edge's higher end and then its lower."""
        return self._get(
            "face_adjacency",
            lambda: _kernels().mesh.adjacent_faces(self._faces, len(self._vertices)),
        )

    def submesh(self, faces) -> "Mesh":
        """The *faces* (ids or a mask) as a mesh of their own, with the
        vertices they use, ascending, renumbered: trimesh's ``submesh(...,
        append=True)``."""
        current = self._faces[np.asarray(faces)].reshape(-1, 3)
        return Mesh._adopt(*_kernels().mesh.submesh(self._vertices, current))

    def copy(self) -> "Mesh":
        """A mesh of the same vertices and faces."""
        return Mesh._adopt(self._vertices.copy(), self._faces.copy())

    def to_trimesh(self):
        """This mesh as a ``trimesh.Trimesh`` (trimesh must be installed)."""
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "Mesh.to_trimesh needs trimesh: pip install 'skeliner[trimesh]'"
            ) from None
        return trimesh.Trimesh(
            vertices=self._vertices.copy(), faces=self._faces.copy(), process=False
        )

    def export(self, path: str | Path) -> None:
        """Write the mesh to *path*, the format by its suffix
        (:func:`skeliner.io.save_mesh`)."""
        from .io import save_mesh

        save_mesh(self, path)


def coincident_merged(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """*faces* with the copies of a vertex along a seam merged
    (:meth:`Mesh.of`); *faces* itself when no vertex is merged."""
    merged, _, _ = _kernels().mesh.merge_coincident_vertices(vertices, faces)
    return faces if merged is None else merged


def _kernels():
    from ._backend import kernels

    return kernels


def mesh_args(*names: str):
    """Decorate a function so that its parameters *names* (``mesh`` when none
    are named) arrive as :class:`Mesh` — anything with ``vertices`` and
    ``faces`` converted once, ``None`` left alone."""

    def wrap(fn):
        params = list(inspect.signature(fn).parameters)
        at = [(params.index(name), name) for name in names or ("mesh",)]

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if args:
                args = list(args)
                for i, name in at:
                    if i < len(args):
                        if args[i] is not None:
                            args[i] = Mesh.of(args[i])
                    elif kwargs.get(name) is not None:
                        kwargs[name] = Mesh.of(kwargs[name])
            else:
                for _, name in at:
                    if kwargs.get(name) is not None:
                        kwargs[name] = Mesh.of(kwargs[name])
            return fn(*args, **kwargs)

        return inner

    return wrap


class _SkeletonModuleView:
    """Expose module functions as bound methods on a Skeleton instance."""

    __slots__ = ("_skel", "_module")

    def __init__(self, skel: "Skeleton", module: Any) -> None:
        self._skel = skel
        self._module = module

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module, name)
        if callable(attr):
            return MethodType(attr, self._skel)
        return attr

    def __dir__(self) -> list[str]:
        names = set(dir(self._module))
        skeleton_names = getattr(self._module, "__skeleton__", None)
        if skeleton_names is not None:
            names.update(skeleton_names)
        return sorted(names)


# -----------------------------------------------------------------------------
# MeshStats dataclass
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class MeshStats:
    """Precomputed per-face mesh statistics.

    Returned by :func:`~skeliner.pre.compute_mesh_stats`.  Lifecycle is
    tied to the mesh, not to organelles — mesh-mutating operations call
    :meth:`invalidate_topology` / :meth:`invalidate_geometry` to mark the
    cached arrays as stale.  Persist standalone via
    :func:`~skeliner.io.save_mesh_stats_npz`.

    Parameters
    ----------
    outward_dots : (nFaces,) float
        Per-face outward score (dot of face normal vs direction from
        local center-of-mass).  Positive = surface, negative = internal.
    face_comp : (nFaces,) int
        Connected component label per face.
    main_ci : int
        Largest component ID.
    """

    outward_dots: np.ndarray | None = None
    face_comp: np.ndarray | None = None
    main_ci: int | None = None

    @property
    def main_face_mask(self) -> np.ndarray:
        """Bool mask for faces in the main component."""
        if self.face_comp is None or self.main_ci is None:
            raise AttributeError(
                "face_comp/main_ci invalidated; recompute via _face_edge_components"
            )
        return self.face_comp == self.main_ci

    def invalidate_topology(self) -> None:
        """Mark face_comp/main_ci as stale (after connectivity changes)."""
        self.face_comp = None
        self.main_ci = None

    def invalidate_geometry(self) -> None:
        """Mark outward_dots as stale (after vertex position changes)."""
        self.outward_dots = None

    def as_tuple(self) -> tuple:
        """Backward-compatible tuple ``(outward_dots, face_comp, main_ci, main_face_mask)``."""
        return (self.outward_dots, self.face_comp, self.main_ci, self.main_face_mask)


# -----------------------------------------------------------------------------
# Organelles dataclass
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class Organelles:
    """Organelle masks.

    Returned by :func:`~skeliner.pre.find_organelles` (alongside the
    associated :class:`MeshStats`) and by
    :func:`~skeliner.io.load_organelles_npz`.

    Parameters
    ----------
    pocket : (nFaces,) bool
        Pocket organelle faces.
    isolated : (nFaces,) bool
        Isolated (disconnected) organelle faces.
    expanded : (nFaces,) bool
        Faces added by :func:`~skeliner.pre.break_up_mesh`.
    manual : (nFaces,) bool, optional
        Faces assigned to the organelles by hand.  Kept apart from the
        detected masks because those are all recomputed:
        :func:`~skeliner.pre.break_up_mesh` rebuilds *expanded* from
        scratch every run, so a manual assignment stored there would be
        erased by the same re-derive meant to apply it.  Defaults to all
        False.
    """

    pocket: np.ndarray
    isolated: np.ndarray
    expanded: np.ndarray
    manual: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.manual is None:
            self.manual = np.zeros(len(self.pocket), dtype=bool)

    @classmethod
    def empty(cls, n_faces: int) -> "Organelles":
        """No organelle faces: a mesh that traces none (hemibrain's,
        EyeWire 1's), where what :func:`~skeliner.pre.find_organelles`
        flags is not one."""
        return cls(*(np.zeros(n_faces, dtype=bool) for _ in range(3)))

    @property
    def mask(self) -> np.ndarray:
        """Combined bool mask (pocket | isolated | expanded | manual)."""
        return self.pocket | self.isolated | self.expanded | self.manual


# -----------------------------------------------------------------------------
# Neurites / Discarded / MeshComponents dataclasses
# -----------------------------------------------------------------------------
#: SWC structure codes, by the leading word of a neurite's name.  Only the
#: standard ones are here; anything else resolves to 0 ("undefined"), which
#: is what every neurite exports as today.
SWC_TYPES: dict[str, int] = {
    "soma": 1,
    "axon": 2,
    "dendrite": 3,
    "basal": 3,
    "apical": 4,
    "custom": 5,
    "unspecified": 6,
    "glia": 7,
}


def swc_type_for(label: str) -> int:
    """The SWC code a neurite name implies, or 0 if it implies none.

    Reads the leading word, so ``"dendrite 2"`` and ``"apical tuft"`` work
    and an index can distinguish two neurites of the same kind.  Only ever
    a *default*: the code is stored explicitly beside the name, because a
    name is free text and an export that silently retypes itself when
    someone writes "Axon (?)" is worse than one that asks.
    """
    head = label.strip().lower().split()
    return SWC_TYPES.get(head[0], 0) if head else 0


@dataclass(slots=True)
class Neurites:
    """Neurite components from :func:`~skeliner.pre.break_up_mesh`.

    Each element of *components* is a face-index array for one neurite,
    sorted by descending face count.

    *labels* and *swc_types* are the hand-given identity of each neurite —
    ``"dendrite 1"``, ``"axon"`` — and the SWC structure code it exports
    as.  Both are ``None`` on anything :func:`~skeliner.pre.break_up_mesh`
    produces, which is the whole design: a name is pinned to a *position*
    in this list, and a re-derive re-sorts by size, splits and merges, so
    a name that survived one would end up on a different piece of surface.
    Naming is therefore the last step, after the mesh is final; if you have
    to break the mesh again, the names go and you name it again.
    """

    components: list[np.ndarray]
    labels: list[str] | None = None
    swc_types: list[int] | None = None

    def __post_init__(self) -> None:
        n = len(self.components)
        if (self.labels is None) != (self.swc_types is None):
            raise ValueError("labels and swc_types must be given together")
        if self.labels is None:
            return
        if len(self.labels) != n or len(self.swc_types) != n:
            raise ValueError(
                f"labels and swc_types must have one entry per component "
                f"(got {len(self.labels)}, {len(self.swc_types)} for {n})"
            )
        self.labels = [str(x) for x in self.labels]
        self.swc_types = [int(x) for x in self.swc_types]

    @property
    def named(self) -> bool:
        return self.labels is not None

    def name(
        self,
        index: int,
        label: str,
        *,
        swc_type: int | None = None,
    ) -> None:
        """Name one neurite, in place.

        Naming any neurite names them all: the rest default to
        ``neurite {i}`` with code 0, so ``labels`` never has holes and the
        skeleton always has a code to stamp.

        Parameters
        ----------
        index : int
            Position in ``components``.
        label : str
            Free text, e.g. ``"dendrite 1"``.
        swc_type : int, optional
            The SWC code.  Defaults to :func:`swc_type_for` of *label*.
        """
        n = len(self.components)
        if not -n <= index < n:
            raise IndexError(f"neurite index {index} out of range for {n}")
        if self.labels is None:
            self.labels = [f"neurite {i}" for i in range(n)]
            self.swc_types = [0] * n
        self.labels[index] = str(label)
        self.swc_types[index] = (
            swc_type_for(label) if swc_type is None else int(swc_type)
        )

    def rename(self, names, *, swc_types=None) -> "Neurites":
        """Name several neurites at once, in place.

        Parameters
        ----------
        names : Mapping[int, str] or Sequence[str]
            ``{0: "axon", 2: "dendrite 1"}`` names those two and leaves the
            rest at their defaults; a sequence names them in order and must
            have one entry per component.
        swc_types : Mapping[int, int] or Sequence[int], optional
            Explicit codes, in the same shape.  Anything not given is
            defaulted from the name by :func:`swc_type_for`.

        Returns
        -------
        Neurites
            *self*, so this chains off ``components.neurites``.

        Examples
        --------
        >>> components.neurites.rename(["dendrite 0", "dendrite 1", "axon"])
        >>> components.neurites.rename({2: "axon"})
        """
        n = len(self.components)
        if hasattr(names, "items"):
            pairs = {int(k): str(v) for k, v in names.items()}
        else:
            names = list(names)
            if len(names) != n:
                raise ValueError(
                    f"expected one name per component ({n}), got {len(names)}"
                )
            pairs = {i: str(v) for i, v in enumerate(names)}

        if swc_types is None:
            codes = {}
        elif hasattr(swc_types, "items"):
            codes = {int(k): int(v) for k, v in swc_types.items()}
        else:
            swc_types = list(swc_types)
            if len(swc_types) != n:
                raise ValueError(
                    f"expected one code per component ({n}), got {len(swc_types)}"
                )
            codes = {i: int(v) for i, v in enumerate(swc_types)}

        for i, label in pairs.items():
            self.name(i, label, swc_type=codes.get(i))
        return self

    def index_of(self, label: str) -> int:
        """The position of the neurite called *label*.

        Raises ``KeyError`` if nothing is called that, or if two are —
        names are free text and nothing enforces that they are unique, so
        an ambiguous lookup is an error rather than a silent first match.
        """
        if self.labels is None:
            raise KeyError("these neurites have no names")
        hits = [i for i, x in enumerate(self.labels) if x == label]
        if not hits:
            raise KeyError(f"no neurite called {label!r}; have {self.labels}")
        if len(hits) > 1:
            raise KeyError(f"{len(hits)} neurites are called {label!r}: {hits}")
        return hits[0]

    def summary(self) -> str:
        """One line per neurite: index, name, SWC code, face count."""
        rows = []
        for i, comp in enumerate(self.components):
            if self.labels is None:
                rows.append(f"  [{i}] {len(comp):>9,} f")
            else:
                rows.append(
                    f"  [{i}] {len(comp):>9,} f  {self.labels[i]} "
                    f"(SWC {self.swc_types[i]})"
                )
        head = f"{len(self.components)} neurites"
        return "\n".join([head, *rows]) if rows else head

    def clear_names(self) -> None:
        """Drop every name, back to the unnamed state."""
        self.labels = None
        self.swc_types = None

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, idx):
        return self.components[idx]

    def to_npz(self, path: str | Path) -> None:
        from . import io

        io.save_neurites_npz(self, path)

    @classmethod
    def from_npz(cls, path: str | Path) -> "Neurites":
        from . import io

        return io.load_neurites_npz(path)


@dataclass(slots=True)
class Discarded:
    """Small fragments below the auto threshold from :func:`~skeliner.pre.break_up_mesh`."""

    components: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, idx):
        return self.components[idx]

    def to_npz(self, path: str | Path) -> None:
        from . import io

        io.save_discarded_npz(self, path)

    @classmethod
    def from_npz(cls, path: str | Path) -> "Discarded":
        from . import io

        return io.load_discarded_npz(path)


@dataclass(slots=True)
class MeshComponents:
    """Result of :func:`~skeliner.pre.break_up_mesh`.

    Holds the four classified pieces of a neuron mesh after breaking
    at the soma and organelle boundaries.  Only *soma* and *organelles*
    are stored inputs; *neurites* and *discarded* are derived from them
    by :func:`~skeliner.pre.break_up_mesh`.
    """

    soma: "Soma | None"
    organelles: Organelles
    neurites: Neurites
    discarded: Discarded

    def rescue_discarded(self, indices: int | list[int]) -> None:
        """Move discarded fragments to neurites, in place.

        The neurite/discarded split is *derived* — see
        :func:`~skeliner.pre.break_up_mesh` — so this move lasts only
        until the next re-derive.  To make it stick, pass the fragments'
        faces to that function as ``rescued=``.

        Parameters
        ----------
        indices : int or list[int]
            Index (or indices) into ``self.discarded`` to rescue.
        """
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        n = len(self.discarded.components)
        picked = set()
        for i in indices:
            i = int(i)
            if not -n <= i < n:
                raise IndexError(f"discarded index {i} out of range for {n}")
            picked.add(i % n)
        # Pop from the back so the earlier indices stay valid, then
        # append in the original order: `discarded` is sorted by
        # descending size, and so is `neurites`.
        moved = [self.discarded.components.pop(i) for i in sorted(picked, reverse=True)]
        for comp in reversed(moved):
            self.neurites.components.append(comp)
            # Appending leaves the existing positions alone, so the names
            # already given still point at the same surface; the new
            # arrival just needs one of its own.
            if self.neurites.labels is not None:
                self.neurites.labels.append(f"neurite {len(self.neurites.labels)}")
                self.neurites.swc_types.append(0)

    def to_npz(self, path: str | Path) -> None:
        from . import io

        io.save_components_npz(self, path)

    @classmethod
    def from_npz(cls, path: str | Path) -> "MeshComponents":
        from . import io

        return io.load_components_npz(path)


@dataclass(slots=True)
class Reassignment:
    """A previewed hand reassignment of faces between components.

    Produced by :func:`~skeliner.pre.preview_reassignment` and applied by
    :func:`~skeliner.pre.apply_reassignment`.  Holds the exact components
    the edit produces, so what is shown before committing and what lands
    afterwards are the same object rather than a forecast and a result.

    Parameters
    ----------
    target : {'soma', 'organelle', 'remainder'}
        Where the selected faces go.
    selected : (n,) int
        The faces picked by the user.
    entering, leaving : (k,) int
        Faces joining and leaving the arbor — the unnamed remainder that
        neurites are derived from.  These are the *effective* face sets,
        so they include the one-ring fringe the ≥2-of-3 soma rule drags
        along with the selection.
    components : MeshComponents
        The mesh components after the edit.
    effects : list of (str, str)
        Per affected component, its label before the edit and what
        becomes of it: ``grown``, ``shrunk``, ``split into N``,
        ``merged with …``, ``dissolved …``, ``reclassified as …`` or
        ``new``.  Empty when no component changed.  Pure renumbering is
        not listed — every re-derive re-sorts by size, so ids shift
        routinely and reporting that would drown the real changes.
    """

    target: str
    selected: np.ndarray
    entering: np.ndarray
    leaving: np.ndarray
    components: "MeshComponents"
    effects: list[tuple[str, str]]

    @property
    def summary(self) -> str:
        """One line: the size of the move and what it does to components."""
        moved = f"{len(self.selected):,} selected → {self.target}"
        net = f"-{len(self.leaving):,}/+{len(self.entering):,} arbor faces"
        if not self.effects:
            return f"{moved} ({net}, no component change)"
        what = "; ".join(f"{name} {eff}" for name, eff in self.effects)
        return f"{moved} ({net}): {what}"


# -----------------------------------------------------------------------------
# Soma dataclass
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class Soma:
    """
    Ellipsoidal soma model.

    The ellipsoid is defined in *world coordinates* by the triple
    `(center, axes, R)` that satisfies

        **world = R @ body + center**

    where *body* points live inside the unit sphere ``‖body‖ ≤ 1``.

    Parameters
    ----------
    center : (3,) float
        XYZ world-space coordinates of the ellipsoid center.
    axes   : (3,) float
        Semi-axis lengths **sorted** as  a ≥ b ≥ c.
    R      : (3,3) float
        Right-handed rotation matrix whose *columns* are the principal
        axes expressed in world space.
    verts  : optional (N,) int64
        Mesh-vertex IDs belonging to the soma surface.
    nucleus : optional dict
        Nucleus void info from detection.  Keys:

        - ``center`` — ``(3,)`` XYZ world coordinates of nucleus void.
        - ``peak_r`` — ``float`` peak void radius (nm).
        - ``z_range`` — ``(z_lo, z_hi)`` Z extent of the void.
        - ``slices`` — ``(N, 4)`` per-Z data ``(z, cx, cy, void_r)``.

        Compatible with the ``nucleus`` parameter of
        :func:`~skeliner.plot.z_section` and
        :func:`~skeliner.plot.diagnose_soma`.
    """

    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3,)
    R: np.ndarray  # (3,3)
    verts: np.ndarray | None = None  # (N,)
    nucleus: dict | None = None

    # ---------------------------------------------------------------------
    # dataclass life-cycle
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.axes = np.asarray(self.axes, dtype=np.float64).reshape(3)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)

        # ---- fast safety checks -----------------------------------------
        if not np.all(np.diff(self.axes) <= 0):
            raise ValueError("axes must be sorted a ≥ b ≥ c")

    def __setstate__(self, state) -> None:
        # a pickle from when the map to the unit sphere was stored carries
        # it as `_W`; it is derived now
        _, slots = state if isinstance(state, tuple) else (None, state)
        for name, value in slots.items():
            if name != "_W":
                object.__setattr__(self, name, value)

    @property
    def _W(self) -> np.ndarray:
        """``(3, 3)`` the map to the unit sphere, ``ξ = (x − c) @ W``: ``R``
        over the axes, from the current ones."""
        return self.R / self.axes

    # ---------------------------------------------------------------------
    # geometry
    # ---------------------------------------------------------------------
    def _body_coords(self, x: np.ndarray) -> np.ndarray:
        """World ➜ body coords where the ellipsoid becomes the *unit sphere*."""
        x = np.asarray(x, dtype=np.float64)
        return (x - self.center) @ self._W

    def contains(self, x: np.ndarray, *, inside_frac: float = 1.0) -> np.ndarray:
        """
        Boolean mask telling whether points lie **inside** the scaled ellipsoid
        (‖ξ‖ ≤ inside_frac).
        """
        x = np.asarray(x, dtype=np.float64)
        inside = _kernels().dataclass.soma_contains(
            self.center, self.axes, self.R, x.reshape(-1, 3), inside_frac**2
        )
        return inside.reshape(x.shape[:-1])[()]

    def remap(self, vert_map: np.ndarray) -> "Soma":
        """Return a copy with vertex indices translated by *vert_map*.

        Parameters
        ----------
        vert_map : (nOldVerts,) int64
            ``vert_map[old_idx]`` gives the new index, or ``-1`` if the
            vertex was removed.

        Returns
        -------
        Soma
            New Soma with the same geometry and nucleus but remapped verts.
        """
        if self.verts is None:
            new_verts = None
        else:
            mapped = vert_map[self.verts]
            new_verts = mapped[mapped >= 0]
            if len(new_verts) == 0:
                new_verts = None
        return Soma(
            center=self.center.copy(),
            axes=self.axes.copy(),
            R=self.R.copy(),
            verts=new_verts,
            nucleus=None if self.nucleus is None else dict(self.nucleus),
        )

    def _scaled(self, scale: Callable[[Any], Any]) -> "Soma":
        """A copy with every length — centre, axes, the nucleus' centre,
        radius, z range and slices — passed through *scale*."""
        nucleus = None
        if self.nucleus is not None:
            nucleus = dict(self.nucleus)
            for key in ("center", "slices"):
                if key in nucleus:
                    nucleus[key] = scale(np.asarray(nucleus[key], dtype=np.float64))
            if "peak_r" in nucleus:
                nucleus["peak_r"] = float(scale(float(nucleus["peak_r"])))
            if "z_range" in nucleus:
                nucleus["z_range"] = tuple(
                    float(scale(float(z))) for z in nucleus["z_range"]
                )
        return Soma(
            center=scale(self.center),
            axes=scale(self.axes),
            R=self.R.copy(),
            verts=None if self.verts is None else self.verts.copy(),
            nucleus=nucleus,
        )

    def distance(self, x, to="center"):
        """
        Compute the distance from *x* to the soma.

        Parameters
        ----------
        x : (N, 3) or (3,) array-like
            Points in world coordinates.
        to : {'center', 'surface'}
            Whether to compute the distance to the center or to the surface.

        Returns
        -------
        (N,) or float
            Unsigned Euclidean distance from *x* to the soma.
        """
        if to == "center":
            return self.distance_to_center(x)
        elif to == "surface":
            return self.distance_to_surface(x)
        else:
            raise ValueError(f"Unknown distance target '{to}'.")

    def distance_to_center(self, x: np.ndarray) -> np.ndarray | float:
        """Unsigned Euclidean distance from *x* to the soma *center*."""
        x = np.asanyarray(x, dtype=np.float64)
        single_input = x.ndim == 1
        if single_input:
            x = x[None, :]
        d = _kernels().dataclass.soma_distance_to_center(self.center, x)
        return d[0] if single_input else d

    def distance_to_surface(
        self, x: np.ndarray, *, atol: float = 1e-9, max_iter: int = 64
    ) -> np.ndarray | float:
        """
        Exact signed Euclidean distance to the ellipsoid surface
        ( > 0 outside | ≈ 0 on surface | < 0 inside ).

        Outside, Newton's iteration on the nearest point's Lagrange
        multiplier, one loop for all of *x*, stopped once every step is
        below *atol* (*max_iter* steps at most); inside, Eberly's bisection,
        to the last bit.
        """
        x = np.asanyarray(x, dtype=np.float64)
        single_input = x.ndim == 1
        if single_input:
            x = x[None, :]
        dist = _kernels().dataclass.soma_distance_to_surface(
            self.center, self.axes, self.R, x, float(atol), int(max_iter)
        )
        return dist[0] if single_input else dist

    # ---------------------------------------------------------------------
    # derived scalars
    # ---------------------------------------------------------------------
    @property
    def spherical_radius(self) -> float:
        """Radius of the sphere which encloses the ellipsoid."""
        return max(self.axes)

    @property
    def equiv_radius(self) -> float:
        """Radius of the sphere of the ellipsoid's volume, ``(abc)^(1/3)``."""
        a, b, c = self.axes
        return float((a * b * c) ** (1.0 / 3.0))

    # ---------------------------------------------------------------------
    # constructors
    # ---------------------------------------------------------------------
    @classmethod
    def fit(cls, pts: np.ndarray, verts=None) -> "Soma":
        """
        Fast PCA-based ellipsoid fit to ≥ 3×`axes` sample points.
        Rough 95 %-mass envelope, same idea as the original *sphere* fit.
        """
        pts = np.asarray(pts, dtype=np.float64)
        center = pts.mean(axis=0)
        cov = np.cov(pts - center, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)  # λ₁ ≤ λ₂ ≤ λ₃
        axes = np.sqrt(evals * 5.0)[::-1]  # 95 % of mass → 2 σ ≈ √5
        R = evecs[:, ::-1]  # reorder to a ≥ b ≥ c
        if np.dot(np.cross(R[:, 0], R[:, 1]), R[:, 2]) < 0:
            # a rotation: an axis' direction is either way, so the smallest
            # one turns
            R[:, 2] = -R[:, 2]
        return cls(center, axes, R, verts=verts)

    @classmethod
    def from_sphere(
        cls, center: np.ndarray, radius: float, verts: np.ndarray | None
    ) -> "Soma":
        """Backward-compat helper – treat a sphere as a = b = c = radius."""
        center = np.asarray(center, dtype=np.float64)
        axes = np.full(3, float(radius), dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        return cls(center, axes, R, verts=verts)

    # ---------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------
    def to_npz(self, path: str | Path) -> None:
        """Write the soma to a compressed NumPy archive."""
        from . import io

        io.save_soma_npz(self, path)

    @classmethod
    def from_npz(cls, path: str | Path) -> "Soma":
        """Load a soma from a ``.npz`` archive."""
        from . import io

        return io.load_soma_npz(path)


# -----------------------------------------------------------------------------
# Skeleton dataclass
# -----------------------------------------------------------------------------
#: Each unit a skeleton converts between, as a power of ten of a metre.
_UNIT_POWERS: dict[str, int] = {
    "nm": -9,
    "nanometer": -9,
    "µm": -6,
    "μm": -6,
    "um": -6,
    "micron": -6,
    "micrometer": -6,
    "mm": -3,
    "millimeter": -3,
    "cm": -2,
    "centimeter": -2,
    "m": 0,
    "meter": 0,
}


@dataclass(slots=True)
class Skeleton:
    """Light-weight skeleton graph."""

    # ---- mandatory soma data ---------------------------------
    soma: Soma

    # ---- mandatory skeleton data (except ntype)---------------
    nodes: np.ndarray  # (N, 3) float64
    radii: dict[str, np.ndarray]  # (N,) float64
    edges: np.ndarray  # (E, 2) int64 – undirected, **sorted** pairs
    ntype: np.ndarray | None  # (N,) int64, node type

    # ---- optional mesh data ----------------------------------
    node2verts: list[np.ndarray] | None = None
    vert2node: VertexMap | None = None  # a dict given is converted
    # ---- optional dictionary for meta data and future extras -
    meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    # ---- cached spatial helpers ------------------------------
    # (ptr, idx): each node's neighbours as a CSR (`_ensure_adjacency`)
    _adjacency: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # (path, index): `dx.distance`'s index, built by the path then active
    _distance_index: tuple[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ---------------------------------------------------------------------
    # sanity checks
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate basic shape constraints."""
        nodes = np.asarray(self.nodes)
        if nodes.size == 0:
            nodes = nodes.reshape(0, 3)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError(f"nodes must be (N, 3), not {nodes.shape}")
        self.nodes = nodes
        N = nodes.shape[0]

        # ---- vert2node --------------------------------------------------
        if self.vert2node is not None and not isinstance(self.vert2node, VertexMap):
            self.vert2node = VertexMap(self.vert2node)

        # ---- radii ---------------------------------------------------
        if any(len(r) != N for r in self.radii.values()):
            raise ValueError("All radius arrays must match the number of nodes")

        # ---- edges ---------------------------------------------------
        edges = np.asarray(self.edges)
        if edges.size == 0:
            edges = edges.reshape(0, 2).astype(np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Edges must be of shape (E, 2)")
        self.edges = edges

        # ---- ntype ---------------------------------------------------
        if self.ntype is None:
            # create default label vector: root=-1, rest=unknown (0)
            ntype = np.full(N, 0, dtype=np.int8)
            if N:
                ntype[0] = -1
        else:
            ntype = np.asanyarray(self.ntype, dtype=np.int8).reshape(-1)
            if len(ntype) != N:
                raise ValueError("ntype length must match number of nodes")
            if N and ntype[0] not in (-1, 1):
                # root must be "root" or "soma"; the caller's array stays
                ntype = ntype.copy()
                ntype[0] = -1
        self.ntype = ntype

        if self.soma is not None:
            if self.soma.verts is not None and self.soma.verts.ndim != 1:
                raise ValueError("soma_verts must be 1-D")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in ("nodes", "edges"):
            # what was built from them is stale
            self._invalidate_spatial_index()

    def __getstate__(self) -> dict[str, Any]:
        # the fields only: the caches are rebuilt when asked for
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def __setstate__(self, state) -> None:
        # a pickle from another version may carry caches this one keeps
        # differently, or none: the fields come back, the caches do not
        _, slots = state if isinstance(state, tuple) else (None, state)
        self._invalidate_spatial_index()
        for f in fields(self):
            if f.init and f.name in slots:
                object.__setattr__(self, f.name, slots[f.name])

    # ---------------------------------------------------------------------
    # spatial helpers (distance index + adjacency cache)
    # ---------------------------------------------------------------------
    def _invalidate_spatial_index(self) -> None:
        """Drop cached spatial structures (distance index, adjacency)."""
        self._adjacency = None
        self._distance_index = None

    def _ensure_distance_index(self):
        """The active path's ``SkeletonIndex`` over the nodes and their
        neighbours, which ``dx.distance`` measures against; cached, and built
        again when the path changes."""
        from ._backend import backend, kernels

        path = backend.get()
        if self._distance_index is None or self._distance_index[0] != path:
            ptr, idx = self._ensure_adjacency()
            index = kernels.dx.SkeletonIndex(
                np.asarray(self.nodes, dtype=np.float64), ptr, idx
            )
            self._distance_index = (path, index)
        return self._distance_index[1]

    def _ensure_adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        """Each node's neighbours as a CSR ``(ptr, idx)``, int64: in the
        order the edges name them, an edge's first end before its second (a
        self-loop lists its node twice); cached."""
        if self._adjacency is None:
            e = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
            self._adjacency = _kernels().dataclass.node_neighbors(e, len(self.nodes))
        return self._adjacency

    def _ensure_node_neighbors(self) -> tuple[np.ndarray, ...]:
        """Each node's neighbours (:meth:`_ensure_adjacency`), one array per
        node."""
        ptr, idx = self._ensure_adjacency()
        return tuple(idx[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1))

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _igraph(self) -> ig.Graph:
        """Return an :class:`igraph.Graph` view of self (undirected)."""
        import igraph as ig

        return ig.Graph(
            n=len(self.nodes),
            edges=np.asarray(self.edges, dtype=np.int64).reshape(-1, 2).tolist(),
            directed=False,
        )

    # ---------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------
    def to_swc(
        self,
        path: str | Path,
        include_header: bool = True,
        scale: float = 1.0,
        radius_metric: str | None = None,
        axis_order: tuple[int, int, int] | str = (0, 1, 2),
    ) -> None:
        """Write the skeleton to SWC."""
        from . import io

        io.save_skeleton_swc(
            self,
            path,
            include_header=include_header,
            scale=scale,
            radius_metric=radius_metric,
            axis_order=axis_order,
        )

    def to_npz(self, path: str | Path) -> None:
        """Write the skeleton to a compressed NumPy archive."""
        from . import io

        io.save_skeleton_npz(self, path)

    @classmethod
    def from_npz(cls, path: str | Path) -> "Skeleton":
        """Load a skeleton from a ``.npz`` archive."""
        from . import io

        return io.load_skeleton_npz(path)

    @classmethod
    def from_swc(
        cls,
        path: str | Path,
        *,
        scale: float = 1.0,
        keep_types: Iterable[int] | None = None,
    ) -> "Skeleton":
        """Load a skeleton from an SWC file."""
        from . import io

        return io.load_skeleton_swc(path, scale=scale, keep_types=keep_types)

    # ------------------------------------------------------------------
    # radius recommendation
    # ------------------------------------------------------------------
    def recommend_radius(self) -> Tuple[str, str, Dict[str, float]]:
        """Recommend the best available radius key.

        Returns ``"centerline"`` when available (perpendicular
        distance to the skeleton path from the second-pass
        re-binning).  Falls back to the legacy heuristic among
        mean / trim / median.
        """
        if "centerline" in self.radii:
            return (
                "centerline",
                "Centerline radius from perpendicular re-binning.",
                {},
            )

        # Legacy fallback
        mean = self.radii.get("mean")
        median = self.radii.get("median")
        if mean is None or median is None:
            if median is not None or mean is not None:
                key = "median" if median is not None else "mean"
            else:
                key = next(iter(self.radii), "median")
            return key, f"No mean and median to compare; using {key!r}.", {}

        ok = (mean > 0) & (median > 0)
        if not np.all(ok):
            bad = np.count_nonzero(~ok)
            print(
                f"[skeliner] Warning: {bad} nodes have zero radius; "
                "they were ignored when picking the estimator."
            )
            mean, median = mean[ok], median[ok]

        if mean.size == 0:
            return "median", "All radii are zero; using median by convention.", {}

        ratio = mean / median
        p50 = float(np.percentile(ratio, 50))
        p75 = float(np.percentile(ratio, 75))
        pmax = float(ratio.max())

        if p75 < 1.02:
            choice, reason = (
                "mean",
                "Bias ≤ 2% for 75% of nodes – distribution symmetric.",
            )
        elif p50 < 1.05 and "trim" in self.radii:
            choice, reason = (
                "trim",
                "Moderate tails; 5% trimmed mean is robust and less biased.",
            )
        else:
            choice, reason = "median", "Long positive tails detected; median is safest."

        return choice, reason, {"p50": p50, "p75": p75, "max": pmax}

    def set_unit(self, unit: str | None = None):
        """Set the unit of the skeleton."""
        if unit is None:
            raise ValueError("unit must be specified")
        self.meta["unit"] = unit

    def convert_unit(self, target_unit: str, current_unit: str | None = None):
        """Convert every length to *target_unit*: the nodes, the radii, the
        soma (centre, axes, nucleus) and the centreline distances kept in
        ``extra``.  Each is replaced, never scaled in place, so a skeleton
        sharing an array or the soma with this one keeps its own."""
        if current_unit is None:
            current_unit = self.meta.get("unit", None)
            if current_unit is None:
                raise ValueError("current_unit must be specified")

        k = self._unit_power(current_unit, target_unit)
        if k:
            # by a power of ten held exactly, so each value is rounded once
            step = 10.0 ** abs(k)

            def scale(x):
                return x * step if k > 0 else x / step

            self.nodes = scale(self.nodes)
            self.radii = {key: scale(r) for key, r in self.radii.items()}
            if self.soma is not None:
                self.soma = self.soma._scaled(scale)
            vals = self.extra.get("cl_dist_vals")
            if vals is not None:
                self.extra = {**self.extra, "cl_dist_vals": scale(np.asarray(vals))}
        self.meta = {**self.meta, "unit": target_unit}

    def _get_unit_conversion_factor(self, current_unit: str, target_unit: str) -> float:
        """Return the conversion factor from current_unit to target_unit."""
        k = self._unit_power(current_unit, target_unit)
        return 10.0**k if k >= 0 else 1.0 / 10.0**-k

    @staticmethod
    def _unit_power(current_unit: str, target_unit: str) -> int:
        """The power of ten from *current_unit* to *target_unit*."""
        if current_unit not in _UNIT_POWERS or target_unit not in _UNIT_POWERS:
            raise ValueError(
                f"Unsupported unit conversion from {current_unit} to {target_unit}. "
                "Supported units: " + ", ".join(_UNIT_POWERS)
            )
        return _UNIT_POWERS[current_unit] - _UNIT_POWERS[target_unit]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def r(self) -> np.ndarray:
        """Return radii based on the recommended estimator."""
        choice = self.recommend_radius()[0]
        return self.radii[choice]

    @property
    def dx(self) -> "_dx_mod":
        """Bound view over :mod:`skeliner.dx`, e.g. ``skel.dx.check_connectivity()``."""
        from . import dx as dx_mod

        return _SkeletonModuleView(self, dx_mod)

    @property
    def post(self) -> "_post_mod":
        """Bound view over :mod:`skeliner.post`, e.g. ``skel.post.clip(...)``."""
        from . import post as post_mod

        return _SkeletonModuleView(self, post_mod)

    # ------------------------------------------------------------------
    # Type Checking block to make pylance happy
    # ------------------------------------------------------------------
    if TYPE_CHECKING:
        # diagnostics
        check_connectivity = _dx_mod.check_connectivity
        connectivity = _dx_mod.connectivity
        check_acyclicity = _dx_mod.check_acyclicity
        acyclicity = _dx_mod.acyclicity
        degree = _dx_mod.degree
        neighbors = _dx_mod.neighbors
        nodes_of_degree = _dx_mod.nodes_of_degree
        branches_of_length = _dx_mod.branches_of_length
        twigs_of_length = _dx_mod.twigs_of_length
        suspicious_tips = _dx_mod.suspicious_tips
        distance = _dx_mod.distance
        node_summary = _dx_mod.node_summary
        extract_neurites = _dx_mod.extract_neurites
        neurites_out_of_bounds = _dx_mod.neurites_out_of_bounds
        volume = _dx_mod.volume
        total_path_length = _dx_mod.total_path_length

        # post-processing
        graft = _post_mod.graft
        clip = _post_mod.clip
        prune = _post_mod.prune
        bridge_gaps = _post_mod.bridge_gaps
        merge_near_soma_nodes = _post_mod.merge_near_soma_nodes
        prune_neurites = _post_mod.prune_neurites
        rebuild_mst = _post_mod.rebuild_mst
        downsample = _post_mod.downsample
        set_ntype = _post_mod.set_ntype
        reroot = _post_mod.reroot
        detect_soma = _post_mod.detect_soma


# -----------------------------------------------------------------------------
# Pairwise contact dataclasses
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class ContactSeeds:
    """
    Pairwise geometrical contacts between two Skeletons (node-to-node).
    idx_a, idx_b : (K,) int64  Node indices in A and B.
    pos_a, pos_b : (K,3) float64  Closest points on the node spheres.
    pos         : (K,3) float64  Midpoint between pos_a and pos_b.
    center_gap  : (K,) float64   ||xa-xb|| - (ra+rb).
    meta        : dict           Aux info.
    """

    idx_a: np.ndarray
    idx_b: np.ndarray
    pos_a: np.ndarray
    pos_b: np.ndarray
    pos: np.ndarray
    center_gap: np.ndarray | None
    meta: dict[str, object]

    @property
    def n(self) -> int:
        return int(len(self.idx_a))


@dataclass(slots=True)
class ProxySites:
    seed_groups: list[np.ndarray]
    center: np.ndarray  # (M,3)
    area_A: np.ndarray  # (M,)
    area_B: np.ndarray  # (M,)
    area_mean: np.ndarray
    seed_to_site: np.ndarray  # (K,)
    meta: dict


@dataclass(slots=True)
class ContactSites:
    faces_A: list[np.ndarray]
    faces_B: list[np.ndarray]
    area_A: np.ndarray
    area_B: np.ndarray
    area_mean: np.ndarray
    seeds_A: np.ndarray
    seeds_B: np.ndarray
    bbox_A: np.ndarray  # (M,2,3)
    bbox_B: np.ndarray  # (M,2,3)
    bbox: np.ndarray  # (M,2,3)
    meta: dict[str, object]
    pairs_AB: list[np.ndarray] | None
    stats_A: dict[str, np.ndarray] | None = None
    stats_B: dict[str, np.ndarray] | None = None
    stats_pair: dict[str, np.ndarray] | None = None
    # what kind of contact each site is ("blob", "crossing", "other") and
    # the branch geometry that says so; set by
    # `skeliner.pair.label_contacts`, None until then
    kind: np.ndarray | None = None
    geometry: dict[str, np.ndarray] | None = None

    def to_npz(self, path: str | Path, *, compress: bool = True) -> None:
        from .io import save_contact_sites_npz

        save_contact_sites_npz(self, path, compress=compress)


def register_skeleton_methods(module: Any, names: Iterable[str] | None = None) -> None:
    """Attach functions from *module* as bound methods on :class:`Skeleton`."""
    if names is None and hasattr(module, "__skeleton__"):
        names = module.__skeleton__
    if names is None:
        raise ValueError("names must be provided when module lacks '__skeleton__'")

    for name in names:
        func = getattr(module, name, None)
        if not callable(func):
            continue
        setattr(Skeleton, name, func)
