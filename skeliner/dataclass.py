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
    """

    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3,)
    R: np.ndarray  # (3,3)
    verts: np.ndarray | None = None  # (N,)

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
            # create default label vector: soma=1, rest=dendrite (3)
            ntype = np.full(N, 3, dtype=np.int8)
            if N:
                ntype[0] = 1
                self.ntype = ntype
        else:
            self.ntype = np.asanyarray(self.ntype, dtype=np.int8).reshape(-1)
            if len(self.ntype) != N:
                raise ValueError("ntype length must match number of nodes")
            self.ntype[0] = 1  # always enforce soma label

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

        io.to_swc(
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

        io.to_npz(self, path)

    # ------------------------------------------------------------------
    # radius recommendation
    # ------------------------------------------------------------------
    def recommend_radius(self) -> Tuple[str, str, Dict[str, float]]:
        """Heuristic choice among mean / trim / median with explanation."""
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
