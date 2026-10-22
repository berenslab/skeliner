"""Shared structural dataclasses used across skeliner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Tuple

import igraph as ig
import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from . import dx as _dx_mod
    from . import post as _post_mod

__all__ = [
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
        # Pop from the back so the earlier indices stay valid, then
        # append in the original order: `discarded` is sorted by
        # descending size, and so is `neurites`.
        moved = [
            self.discarded.components.pop(i) for i in sorted(indices, reverse=True)
        ]
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
        :func:`~skeliner.plot.vis2d.z_section` and
        :func:`~skeliner.plot.vis2d.diagnose_soma`.
    """

    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3,)
    R: np.ndarray  # (3,3)
    verts: np.ndarray | None = None  # (N,)
    nucleus: dict | None = None

    # ---- cached helper (not part of the public API) -----------------------
    _W: np.ndarray = field(init=False, repr=False)  # (3,3) affine map

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

        # ---- pre-compute affine map  ξ = (x−c) @ W -----------------------
        self._W = (self.R / self.axes).astype(np.float64)

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
        ξ = self._body_coords(x)
        ρ2 = (ξ**2).sum(axis=-1)
        return ρ2 <= inside_frac**2

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
            New Soma with the same geometry but remapped verts.
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
        d = np.linalg.norm(x - self.center, axis=1)
        return d[0] if single_input else d

    def distance_to_surface(
        self, x: np.ndarray, *, atol: float = 1e-9, max_iter: int = 64
    ) -> np.ndarray | float:
        """
        Exact signed Euclidean distance to the ellipsoid surface
        ( > 0 outside | ≈ 0 on surface | < 0 inside ).
        """
        x = np.asanyarray(x, dtype=np.float64)
        single_input = x.ndim == 1
        if single_input:
            x = x[None, :]

        # --- body-coordinates: align to principal axes --------------------
        p = (x - self.center) @ self.R  # (N,3)
        a = self.axes
        a2 = a * a
        r2 = (p**2 / a2).sum(axis=1)  # ‖p‖² in unit-sphere space
        out = r2 > 1.0 + 1e-12  # bool mask
        dist = np.empty(len(p), dtype=np.float64)

        # ---------------- OUTSIDE points  ---------------------------------
        if out.any():
            po = p[out]
            t = np.zeros(len(po))
            for _ in range(max_iter):
                denom = t[:, None] + a2
                f = (a2 * po**2 / denom**2).sum(1) - 1.0
                fp = (-2.0 * a2 * po**2 / denom**3).sum(1)
                dt = -f / fp
                t += dt
                if np.all(np.abs(dt) < atol):
                    break
            xs = a2 * po / (t[:, None] + a2)  # nearest surface points
            dist[out] = np.linalg.norm(xs - po, axis=1)

        # ---------------- INSIDE points  ----------------------------------
        inn = ~out
        if inn.any():
            idx_inn = np.where(inn)[0]
            pi = p[inn]
            s = np.sqrt(r2[inn])  # radial factor
            nz = s > atol  # not at exact center

            # general interior points
            if nz.any():
                xs = pi[nz] / s[nz, None]  # radial projection
                dist[idx_inn[nz]] = -np.linalg.norm(xs - pi[nz], axis=1)

            # exact center → shortest half-axis
            if (~nz).any():
                dist[idx_inn[~nz]] = -a.min()

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
        """Equivalent radius of the ellipsoid (mean of semi-axes)."""
        """Sphere radius of equal volume ( (abc)^{1/3} )."""
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
    vert2node: dict[int, int] | None = None
    # ---- optional dictionary for meta data and future extras -
    meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    # ---- cached spatial helpers ------------------------------
    _nodes_kdtree: KDTree | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _node_neighbors: tuple[np.ndarray, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ---------------------------------------------------------------------
    # sanity checks
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate basic shape constraints."""
        N = self.nodes.shape[0]

        # ---- radii ---------------------------------------------------
        if any(len(r) != N for r in self.radii.values()):
            raise ValueError("All radius arrays must match the number of nodes")

        # ---- edges ---------------------------------------------------
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("Edges must be of shape (E, 2)")

        # ---- ntype ---------------------------------------------------
        if self.ntype is None:
            # create default label vector: root=-1, rest=unknown (0)
            ntype = np.full(N, 0, dtype=np.int8)
            if N:
                ntype[0] = -1
                self.ntype = ntype
        else:
            self.ntype = np.asanyarray(self.ntype, dtype=np.int8).reshape(-1)
            if len(self.ntype) != N:
                raise ValueError("ntype length must match number of nodes")
            self.ntype[0] = (
                -1 if self.ntype[0] not in [-1, 1] else self.ntype[0]
            )  # root must be "root" or "soma"

        if self.soma is not None:
            if self.soma.verts is not None and self.soma.verts.ndim != 1:
                raise ValueError("soma_verts must be 1-D")

    # ---------------------------------------------------------------------
    # spatial helpers (KD-tree + adjacency cache)
    # ---------------------------------------------------------------------
    def _invalidate_spatial_index(self) -> None:
        """Drop cached spatial structures (KD-tree, adjacency)."""
        self._nodes_kdtree = None
        self._node_neighbors = None

    def _ensure_nodes_kdtree(self, *, rebuild: bool = False) -> KDTree:
        """Return a cached KD-tree over node coordinates."""
        if rebuild:
            self._nodes_kdtree = None
        if self._nodes_kdtree is None:
            if self.nodes.size == 0:
                raise ValueError("Cannot build KD-tree: skeleton has no nodes.")
            self._nodes_kdtree = KDTree(self.nodes)
        return self._nodes_kdtree

    def _ensure_node_neighbors(self) -> tuple[np.ndarray, ...]:
        """Return cached neighbour lists for every node."""
        if self._node_neighbors is None:
            neighbours = [[] for _ in range(len(self.nodes))]
            for u, v in self.edges:
                neighbours[u].append(v)
                neighbours[v].append(u)
            self._node_neighbors = tuple(
                np.asarray(nbrs, dtype=np.int64) if nbrs else np.empty(0, np.int64)
                for nbrs in neighbours
            )
        return self._node_neighbors

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _igraph(self) -> ig.Graph:
        """Return an :class:`igraph.Graph` view of self (undirected)."""
        return ig.Graph(
            n=len(self.nodes),
            edges=[tuple(map(int, e)) for e in self.edges],
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
            return "median", "Only one radius column available; using it.", {}

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
        """Convert all coordinates/radii to a new unit."""
        if current_unit is None:
            current_unit = self.meta.get("unit", None)
            if current_unit is None:
                raise ValueError("current_unit must be specified")

        if current_unit == target_unit:
            return

        factor = self._get_unit_conversion_factor(current_unit, target_unit)
        if factor is None:
            raise ValueError(f"Cannot convert from {current_unit} to {target_unit}")

        self.nodes *= factor
        for key in self.radii.keys():
            self.radii[key] *= factor
        if self.soma is not None:
            self.soma.axes *= factor

        self.meta["unit"] = target_unit
        self._invalidate_spatial_index()

    def _get_unit_conversion_factor(
        self, current_unit: str, target_unit: str
    ) -> float | None:
        """Return the conversion factor from current_unit to target_unit."""
        conversion_factors = {
            "nm": 1e-9,
            "nanometer": 1e-9,
            "µm": 1e-6,
            "μm": 1e-6,
            "um": 1e-6,
            "micron": 1e-6,
            "micrometer": 1e-6,
            "mm": 1e-3,
            "millimeter": 1e-3,
            "cm": 1e-2,
            "centimeter": 1e-2,
            "m": 1.0,
            "meter": 1.0,
        }

        if (
            current_unit not in conversion_factors
            or target_unit not in conversion_factors
        ):
            raise ValueError(
                f"Unsupported unit conversion from {current_unit} to {target_unit}. "
                "Supported units: " + ", ".join(conversion_factors.keys())
            )

        return conversion_factors[current_unit] / conversion_factors[target_unit]

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
    # what kind of contact each site is ("crossing", "parallel", "tip",
    # "blob", "other") and the branch geometry that says so; set by
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
