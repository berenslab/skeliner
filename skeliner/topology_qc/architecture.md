# Architecture

## Guiding Principle

The toolkit separates **topology analysis** from **visualization**.

Every script builds on the same reusable analysis engine.

The workflow is intentionally one-directional.

```
SWC
 │
 ▼
Graph construction
 │
 ▼
NodeClassifier
 │
 ├──────────────┐
 │              │
 ▼              ▼
Region QC    Local QC
 │              │
 └──────┬───────┘
        ▼
Topology Viewer
```

---

# Core Layer

## NodeClassifier.py

This is the computational core of the toolkit.

Responsibilities include

- constructing graph context
- neighborhood extraction
- component decomposition
- branch geometry measurements
- distal component statistics
- branch pairing
- node classification

Every downstream tool imports this module rather than reimplementing topology analysis.

Keeping the analysis centralized ensures that improvements to the classification logic automatically propagate throughout the toolkit.

---

# Validation Layer

## CheckSWCTree.py

This module verifies basic graph correctness before biological analysis begins.

Checks include

- one root
- connected tree
- E = N − 1

If these conditions fail, topology QC results are not meaningful.

---

# Analysis Layer

## FindAnomalousRegions.py

Consumes node-level analysis and performs region-level aggregation.

Pipeline

```
Node scores

↓

Suspicious nodes

↓

Connected components

↓

Anomalous regions
```

Outputs include CSV summaries and an interactive HTML viewer.

---

# Inspection Layer

## NeighborhoodTree.py

Designed for interactive debugging.

Rather than viewing an entire neuron, this tool focuses on a local neighborhood surrounding a selected node.

Displayed information includes

- upstream branch
- downstream subtree
- suspicious neighboring nodes
- local topology metrics

This serves as the "microscope" of the toolkit.

---

# Abstraction Layer

## BuildTopologyGraphViewer.py

Produces a higher-level representation of the neuron.

Long degree-2 chains are contracted into corridors connecting biologically meaningful topology nodes.

The resulting graph is much easier to inspect than the original skeleton.

Importantly, this is an analysis transform only.

The source SWC remains unchanged.

---

# Data Flow

```
             SWC

              │

              ▼

      build_graph_context()

              │

              ▼

      NodeClassifier.py

              │

     ┌────────┼─────────┐

     ▼        ▼         ▼

 Regions   Neighborhood  Topology

              │

              ▼

      Interactive HTML
```

---

# Design Goals

The architecture emphasizes

- reusable analysis
- modular visualization
- non-destructive processing
- reproducible measurements
- extensibility

New visualization tools should consume the outputs of `NodeClassifier.py` rather than introducing independent topology logic.

Similarly, future topology metrics should be added to `NodeClassifier.py` so they become immediately available throughout the toolkit.

---

# Future Extensions

The intent is for this repository to evolve into a general framework for biological topology quality control of neuronal skeletons rather than a collection of standalone scripts.