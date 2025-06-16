# skeliner

A lightweight skeletonizer that converts neuron meshes into biophysical‑modelling‑ready SWC morphologies. It heuristically detects the soma, extracts an acyclic center‑line skeleton, estimates per‑node radii, and bridges small gaps.

![](./.github/banner.png)

## Installation

```bash
pip install skeliner
```

or

```bash
git clone https://github.com/berenslab/skeliner.git
pip install -e "skeliner[dev]"
```

## Usage

See [example notebooks](https://github.com/berenslab/skeliner/tree/main/notebooks) for usage. 