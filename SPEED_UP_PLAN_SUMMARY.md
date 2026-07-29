# `calibrate_radii` speed-up: summary

## Context

`calibrate_radii` ([skeliner/post.py](skeliner/post.py)) is correct but slow. Two concrete
inefficiencies were identified and fixed:

1. **`submesh_by_vertices`** ([skeliner/post.py](skeliner/post.py)) scanned the *entire* mesh
   face array (`np.isin(mesh.faces, vertex_indices)`) for every "large" node needing the
   outer-shell check — O(F_total) per node instead of O(node's own vertex-bin size).
2. **`dx.distance`'s whitelist path** ([skeliner/dx.py](skeliner/dx.py)), called once per node
   with a fixed `allowed_nodes=[i]`, recomputed the candidate-edge set on *every point* inside a
   pure-Python loop, even though that set is identical for the whole call.

Scope, agreed up front: no threading/multiprocessing inside `calibrate_radii`/`dx.distance`
(parallelized externally across whole skeletons already); no changes to ray casting
(`filter_inner_surfaces_raycast`) or its algorithm; no new dependencies.

## What changed

- **`submesh_by_vertices`**: restricts the face scan to faces incident to the node's vertex bin,
  via a plain-numpy vertex→face incidence map (`vertex_face_incidence`, argsort/searchsorted)
  built **once** in `calibrate_radii` and passed into every call. Output is bit-identical to the
  old implementation (proved and tested).
- **`dx.distance`**: the whitelist branch (`allowed_nodes`/`allowed_edges`) now computes its
  candidate centres/edges once per call and vectorizes the point-to-centre / point-to-segment
  distance math across all points via two new batched helpers
  (`_batched_point_segment_distance`, `_batched_point_segment_capsule_distance`). The
  non-whitelist (KD-tree) path is untouched.
- **`filter_inner_surfaces_raycast`**: the `trimesh.ray.ray_pyembree` import is hoisted to module
  scope instead of being re-attempted inside the hot loop (mechanical, no behavior change).

## A real pitfall found along the way

The first version of the `submesh_by_vertices` fix used trimesh's cached `mesh.vertex_faces`
property. Micro-benchmarks looked great, but an end-to-end profile showed the win nearly
vanishing: `mesh.vertex_faces` is a `@cache_decorator` property, and **every access** re-verifies
the mesh's cache validity by hashing its vertex/face data with `blake2b` (since the optional
`xxhash` package isn't installed) — not just on the first access. That hashing tax ate almost the
entire gain. Switching to a plain-numpy incidence map (no trimesh cache machinery involved)
fixed it for real. Lesson: trimesh cached properties are not "free after first touch" without
`xxhash` installed.

## Verification

- All fixes are covered by tests in `tests/test_post.py` and `tests/test_dx.py`:
  - `submesh_by_vertices` bit-identical parity against the old `np.isin`-based reference, across
    real node2verts bins, isolated-vertex, no-full-face, and full-vertex-set cases.
  - `dx.distance`'s whitelist path matches a literal reference of the old per-point loop
    (`np.allclose`, rtol/atol 1e-9) across `surface`/`centerline` × `allowed_nodes`/`allowed_edges`/both.
  - End-to-end `calibrate_radii` parity test: old vs. new implementation (via monkeypatch) on the
    existing `60427.obj` fixture — identical `radius_method` classification, `allclose` radii.
- Full suite: **93/93 tests pass.**

## Performance results

**Synthetic benchmark** (60K-face tube mesh, 300 nodes, 150 qualifying for outer-shell check,
`rays_num_outer=20`):

| phase | old | new | speedup |
|---|---|---|---|
| `submesh_by_vertices` | 0.345s | 0.056s | 6.2x |
| ray casting (untouched) | 0.512s | 0.495s | ~1x |
| `dx.distance` | 0.965s | 0.042s | 23x |
| **total** | **1.865s** | **0.647s** | **~2.9x** |

**Real production data** (`720575940548230344`: 357K vertices / 713K faces, 7,957 nodes, 800
qualifying nodes at the production `min_verts_q_outer=90`, `rays_num_outer=30`):

| phase | old | new |
|---|---|---|
| `submesh_by_vertices` | ~22s (est.) | **1.52s** |
| ray casting (untouched) | ~68s | **67.87s** (95% of total) |
| **total** | ~84s | ~71s → **1.15x** |

Correctness on the real skeleton: `radius_method` classification identical, calibrated radii
`allclose`.

## Bottom line

Both fixes work exactly as designed and are safe to keep (proven correctness, real measured
speedups in isolation: 6-12x on `submesh_by_vertices`, ~20x+ on `dx.distance`). But on real
production meshes, **ray casting dominates completely (~95% of total runtime)** — individual
qualifying nodes cost up to ~2.5s each because their local submeshes have hundreds to ~1,500
faces at `rays_num_outer=30`, cast through trimesh's non-embree ray-triangle intersector rebuilt
fresh per node. This was already the largest cost before these changes and was explicitly kept
out of scope. As a result, the *net* end-to-end win on real data is only ~1.15x, even though the
two targeted fixes each deliver an order-of-magnitude improvement on their own piece.

**If a bigger real-world win is wanted later**, ray casting is where the remaining time is:
options include installing/using `embreex` for a faster intersector, reusing one intersector
across nodes instead of rebuilding per submesh, reducing `rays_num_outer`, or reworking the
outer-shell detection algorithm itself. None of that was in scope for this round.

## Files touched

- `skeliner/post.py` — `submesh_by_vertices`, new `vertex_face_incidence`, `calibrate_radii`,
  `filter_inner_surfaces_raycast`, module-level import
- `skeliner/dx.py` — `distance`, new `_batched_point_segment_distance`,
  `_batched_point_segment_capsule_distance`, `_whitelist_distances`
- `tests/test_post.py`, `tests/test_dx.py` — new regression/parity tests
