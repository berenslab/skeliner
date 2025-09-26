# skeliner 

[![PyPI version](https://badge.fury.io/py/skeliner.svg)](https://badge.fury.io/py/skeliner)

A lightweight skeletonizer that converts neuron meshes into biophysical‑modelling‑ready morphologies. It extracts an acyclic center‑line skeleton, estimates per‑node radii, detects the soma, and bridges small gaps. It also provides robust contact‑site mapping between pairs of cells using both skeletons and meshes.

![](.github/banner.png)

## Features

- Mesh → SWC skeletons 
  - Center‑line, acyclic skeleton.
  - Per‑node radii with multiple estimators (median/mean/trim) and an automatic recommendation.
  - Ellipsoidal soma detection near the soma centroid.
  - Bridges disconnected mesh components back to the soma (gap‑closing) and prunes tiny peri‑soma artefacts.
  - SWC and NPZ export with metadata;

- Pairwise contact sites (two cells)
  - Skeleton‑based seeding: fast KD‑tree search finds node↔node proximity pairs with tolerance; returns contact midpoints and per‑node loci.
  - Mesh‑based patches: around seeds, projects to both meshes and extracts touching triangle patches; returns per‑site areas on A/B and their mean, plus compact AABBs for downstream use.
  - Optional skeleton‑only approximation of contact sites/areas when meshes are unavailable.

- I/O, visualization, diagnostics and post-processing
  - Read/write SWC and compact NPZ; load meshes via `trimesh`.
  - 2D projections and 3D viewers for meshes, skeletons, and contact patches.
  - Various diagnostics and post-processing tools.

## Installation

```bash
pip install skeliner
```

or from source:

```bash
git clone https://github.com/berenslab/skeliner.git

# with uv
uv sync --all-extras

# or not
pip install -e "skeliner[dev]"
```

## Quickstart

### Skeletonize a mesh and save SWC

```python
import skeliner as sk

mesh = sk.io.load_mesh("cellA.obj")  # or trimesh.load_mesh directly
skel = sk.skeletonize(mesh, unit="nm", verbose=True)

# Choose the recommended radius column (median/mean/trim)
print("Recommended radius:", skel.recommend_radius())

# Save SWC (e.g. convert nm → µm on write)
skel.to_swc("cellA.swc", scale=1e-3)
# Also persist a compact NPZ for fast reload
skel.to_npz("cellA.npz")
```

### Find contact sites between two cells (skeleton + mesh)

```python
import skeliner as sk

meshA = sk.io.load_mesh("cellA.obj")
meshB = sk.io.load_mesh("cellB.obj")
A = sk.skeletonize(meshA)
B = sk.skeletonize(meshB)

# 1) skeleton-based seeds (delta in the same unit as the skeletons/meshes)
seeds = pair.find_contact_seeds(A, B, exclude_soma=True)
print("seed count:", seeds.n)

# 2) mesh-based contact patches around the seed midpoints
sites = pair.map_contact_sites(meshA, meshB, seeds.pos, unit="nm^2")
print("contact patches:", len(sites.faces_A))
print("mean areas (mesh units^2):", sites.area_mean)

# Optional: visualize patches
# from skeliner.plot.vis3d import view_contacts
# sk.plot.view_contacts(meshA, meshB, sites, sides="A")
```

<p align="center">
  <img src=".github/media/contacts-meshes-with-patches.png" width="30%">
  <img src=".github/media/contacts-one-mesh-with-patches.png" width="30%">
  <img src=".github/media/contacts-no-meshes-patches-only.png" width="30%">
</p>

---

See [example notebooks](https://github.com/berenslab/skeliner/tree/main/notebooks) for more usage. 
