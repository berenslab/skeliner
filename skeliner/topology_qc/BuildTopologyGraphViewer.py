#!/usr/bin/env python3
"""
BuildTopologyGraphViewer.py

Contract an SWC tree into a higher-level topology graph and generate:

  - topology_nodes.csv
  - topology_edges.csv
  - topology_summary.txt
  - topology_graph.html

The HTML viewer shows:
  - the original SWC skeleton in Viridis,
  - contracted topology nodes as labeled markers,
  - contracted corridors as colored overlay polylines.

Interesting nodes (by default):
  - root
  - terminals (degree 1)
  - branch points (degree >= 3)
  - any user-specified keep nodes (useful for preserving nodes like 1758)

This is an analysis-layer transform only. The input SWC is never modified.

Usage:
  python BuildTopologyGraphViewer.py --swc skeleton.swc --keep-node 1758

Dependencies:
  numpy
  pandas
  plotly
  NodeClassifier.py (same directory or importable on PYTHONPATH)
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale

from NodeClassifier import (
    DEFAULT_MIN_DISTAL_COMPONENT_SIZE_UM,
    DEFAULT_MIN_NODE_DEGREE,
    DEFAULT_MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
    analyze_node,
    build_graph_context,
)

DEFAULT_SWC = Path("/Users/jhsinger/Documents/nNOS_AC_2026/EyewireAnalysis/720575940551376557/720575940562533849/skeleton.swc")
DEFAULT_OUTDIR = Path("/Users/jhsinger/Documents/nNOS_AC_2026/EyewireAnalysis/720575940551376557/720575940562533849/topology_graph2")

SMOOTH_PATH_UM = 30.0
N_COLOR_BINS = 24
MIN_LINE_WIDTH = 2
MAX_LINE_WIDTH = 6
OFFSET = 20

# Visual encoding for topology nodes.
ROOT_COLOR = "#F1C40F"         # gold
BRANCH_COLOR = "#27AE60"       # green
TERMINAL_COLOR = "#3498DB"     # blue
KEPT_DEG2_COLOR = "#E67E22"    # orange
SUSPICIOUS_COLOR = "#E74C3C"   # red


def read_swc(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append(
                (
                    int(float(parts[0])),
                    int(float(parts[1])),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    int(float(parts[6])),
                )
            )

    if not rows:
        raise ValueError(f"No SWC rows found in {path}")

    arr = np.array(rows, dtype=object)
    ids = arr[:, 0].astype(int)
    xyz = np.column_stack(
        [
            arr[:, 2].astype(float),
            arr[:, 3].astype(float),
            arr[:, 4].astype(float),
        ]
    )
    radii = arr[:, 5].astype(float)
    parents = arr[:, 6].astype(int)
    return ids, xyz, radii, parents


def choose_important_nodes(graph, keep_set: Set[int]) -> Set[int]:
    """Important nodes are root, terminals, branchpoints, and kept nodes."""
    important = set(keep_set)
    if graph.soma_idx is not None:
        important.add(int(graph.soma_idx))

    for idx in range(len(graph.ids)):
        deg = len(graph.adj[idx])
        if deg != 2:
            important.add(idx)

    return important


def node_kind(idx: int, root_idx: Optional[int], degree: int, keep_set: Set[int]) -> str:
    parts: List[str] = []
    if root_idx is not None and idx == root_idx:
        parts.append("root")

    if degree == 1:
        parts.append("terminal")
    elif degree == 2:
        if idx in keep_set:
            parts.append("kept_degree2")
        else:
            parts.append("degree2")
    elif degree >= 3:
        parts.append("branchpoint")
    else:
        parts.append(f"degree{degree}")

    return "+".join(parts)


def corridor_path(
    start_idx: int,
    next_idx: int,
    important_set: Set[int],
    adj: List[List[int]],
    visited_edges: Set[Tuple[int, int]],
) -> List[int]:
    """
    Traverse from start_idx through next_idx until another important node is reached.
    Returns full corridor node-index list including both endpoints.
    """
    path = [start_idx, next_idx]
    visited_edges.add(tuple(sorted((start_idx, next_idx))))
    prev = start_idx
    cur = next_idx

    while cur not in important_set:
        nbrs = [v for v in adj[cur] if v != prev]
        if len(nbrs) != 1:
            break
        nxt = nbrs[0]
        visited_edges.add(tuple(sorted((cur, nxt))))
        path.append(nxt)
        prev, cur = cur, nxt

    return path


def corridor_stats(path: Sequence[int], graph) -> dict:
    """Compute simple measurements along a contracted corridor."""
    if len(path) < 2:
        return {
            "n_swc_nodes": len(path),
            "cable_length_um": 0.0,
            "euclidean_length_um": 0.0,
            "length_ratio": np.nan,
            "mean_radius_um": np.nan,
            "median_radius_um": np.nan,
            "min_radius_um": np.nan,
            "max_radius_um": np.nan,
        }

    cable = 0.0
    radii = graph.radii[np.array(path, dtype=int)]
    for a, b in zip(path[:-1], path[1:]):
        cable += float(np.linalg.norm(graph.xyz[a] - graph.xyz[b]))

    euclid = float(np.linalg.norm(graph.xyz[path[-1]] - graph.xyz[path[0]]))
    ratio = float(cable / euclid) if euclid > 0 else np.nan

    return {
        "n_swc_nodes": int(len(path)),
        "cable_length_um": float(cable),
        "euclidean_length_um": float(euclid),
        "length_ratio": ratio,
        "mean_radius_um": float(np.mean(radii)),
        "median_radius_um": float(np.median(radii)),
        "min_radius_um": float(np.min(radii)),
        "max_radius_um": float(np.max(radii)),
    }


def build_contracted_graph(graph, keep_node_ids: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return node_df and edge_df for the contracted topology graph.
    """
    keep_set: Set[int] = set()
    for nid in keep_node_ids:
        if nid in graph.id_to_idx:
            keep_set.add(int(graph.id_to_idx[nid]))

    important = choose_important_nodes(graph, keep_set)
    root_idx = graph.soma_idx

    # Analyze important nodes.
    node_results: Dict[int, dict] = {}
    for idx in sorted(important):
        node_results[idx] = analyze_node(
            idx,
            graph,
            MIN_DISTAL_COMPONENT_SIZE_UM=DEFAULT_MIN_DISTAL_COMPONENT_SIZE_UM,
            MIN_NODE_DEGREE=DEFAULT_MIN_NODE_DEGREE,
            MIN_SUBSTANTIAL_DISTAL_COMPONENTS=DEFAULT_MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
        )

    # Build corridors between important nodes.
    visited_edges: Set[Tuple[int, int]] = set()
    corridors: List[dict] = []

    for start in sorted(important):
        for nbr in graph.adj[start]:
            edge_key = tuple(sorted((int(start), int(nbr))))
            if edge_key in visited_edges:
                continue
            path = corridor_path(start, nbr, important, graph.adj, visited_edges)
            if len(path) < 2:
                continue
            end = path[-1]
            if start == end:
                continue
            corridors.append(
                {
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "path": path,
                }
            )

    # Node rows first; topology_node_id will be assigned after sorting.
    node_rows: List[dict] = []
    for idx in sorted(important):
        a = node_results[idx]
        deg = int(a["degree"])
        kind = node_kind(idx, root_idx, deg, keep_set)
        node_rows.append(
            {
                "swc_node_index": int(idx),
                "swc_node_id": int(graph.ids[idx]),
                "kind": kind,
                "is_root": int(root_idx is not None and idx == root_idx),
                "is_terminal": int(deg == 1),
                "is_branchpoint": int(deg >= 3),
                "is_kept_node": int(idx in keep_set),
                "degree": int(deg),
                "n_incident_corridors": 0,  # filled below
                "root_distance_um": float(a["root_distance_um"]),
                "depth_edges": int(a["depth_edges"]),
                "n_components": int(a["n_components"]),
                "n_soma_components": int(a["n_soma_components"]),
                "n_terminal_components": int(a["n_terminal_components"]),
                "n_branchpoint_components": int(a["n_branchpoint_components"]),
                "soma_component_cable_um": float(a["soma_component_cable_um"]),
                "largest_distal_component_cable_um": float(a["largest_distal_component_cable_um"]),
                "second_distal_component_cable_um": float(a["second_distal_component_cable_um"])
                if np.isfinite(a["second_distal_component_cable_um"])
                else np.nan,
                "largest_distal_fraction": float(a["largest_distal_fraction"])
                if np.isfinite(a["largest_distal_fraction"])
                else np.nan,
                "second_distal_fraction": float(a["second_distal_fraction"])
                if np.isfinite(a["second_distal_fraction"])
                else np.nan,
                "largest_to_second_distal_ratio": float(a["largest_to_second_distal_ratio"])
                if np.isfinite(a["largest_to_second_distal_ratio"])
                else np.nan,
                "n_distal_components_ge_um": int(a["n_distal_components_ge_um"]),
                "badness_score": float(a["badness_score"]),
                "is_suspicious": int(bool(a["is_suspicious"])),
                "reason": str(a["reason"]),
                "pairing_margin": float(a["pairing_margin"]) if np.isfinite(a["pairing_margin"]) else np.nan,
                "best_pairing": str(a["best_pairing"]),
                "neighbor_ids": ";".join(map(str, a["neighbor_ids"])),
            }
        )

    # Incident corridor counts.
    incident_counts = {idx: 0 for idx in important}
    for c in corridors:
        incident_counts[c["start_idx"]] = incident_counts.get(c["start_idx"], 0) + 1
        incident_counts[c["end_idx"]] = incident_counts.get(c["end_idx"], 0) + 1

    for row in node_rows:
        row["n_incident_corridors"] = int(incident_counts.get(int(row["swc_node_index"]), 0))

    node_df = pd.DataFrame(node_rows)

    # Stable sort then assign topology IDs.
    node_df = node_df.sort_values(
        ["is_root", "is_branchpoint", "is_terminal", "is_kept_node", "badness_score", "degree"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    node_df.insert(0, "topology_node_id", np.arange(1, len(node_df) + 1, dtype=int))

    swc_idx_to_topology_id = {
        int(row["swc_node_index"]): int(row["topology_node_id"])
        for _, row in node_df.iterrows()
    }

    # Corridor table.
    edge_rows: List[dict] = []
    for corridor_id, c in enumerate(corridors, start=1):
        path = c["path"]
        start_idx = int(c["start_idx"])
        end_idx = int(c["end_idx"])

        stats = corridor_stats(path, graph)
        path_results = [node_results[idx] for idx in path if idx in node_results]
        susp_ids = [int(graph.ids[idx]) for idx in path if idx in node_results and bool(node_results[idx]["is_suspicious"])]
        max_badness = max([float(r["badness_score"]) for r in path_results], default=0.0)
        mean_badness = float(np.mean([float(r["badness_score"]) for r in path_results])) if path_results else 0.0

        edge_rows.append(
            {
                "corridor_id": corridor_id,
                "start_topology_node_id": int(swc_idx_to_topology_id[start_idx]),
                "end_topology_node_id": int(swc_idx_to_topology_id[end_idx]),
                "start_swc_node_id": int(graph.ids[start_idx]),
                "end_swc_node_id": int(graph.ids[end_idx]),
                "start_swc_node_index": int(start_idx),
                "end_swc_node_index": int(end_idx),
                "n_swc_nodes": int(stats["n_swc_nodes"]),
                "swc_node_ids": ";".join(str(int(graph.ids[idx])) for idx in path),
                "swc_node_indices": ";".join(map(str, path)),
                "cable_length_um": float(stats["cable_length_um"]),
                "euclidean_length_um": float(stats["euclidean_length_um"]),
                "length_ratio": float(stats["length_ratio"]) if np.isfinite(stats["length_ratio"]) else np.nan,
                "mean_radius_um": float(stats["mean_radius_um"]),
                "median_radius_um": float(stats["median_radius_um"]),
                "min_radius_um": float(stats["min_radius_um"]),
                "max_radius_um": float(stats["max_radius_um"]),
                "max_badness_score_on_corridor": float(max_badness),
                "mean_badness_score_on_corridor": float(mean_badness),
                "n_suspicious_nodes_on_corridor": int(sum(bool(r["is_suspicious"]) for r in path_results)),
                "suspicious_node_ids": ";".join(map(str, susp_ids)),
                "contains_root": int(root_idx in path if root_idx is not None else 0),
            }
        )

    edge_df = pd.DataFrame(edge_rows)
    if not edge_df.empty:
        edge_df = edge_df.sort_values(
            ["max_badness_score_on_corridor", "cable_length_um", "length_ratio", "n_swc_nodes"],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    return node_df, edge_df


def write_summary(out_txt: Path, swc_path: Path, node_df: pd.DataFrame, edge_df: pd.DataFrame, keep_node_ids: Sequence[int]) -> None:
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("Contracted topology graph summary")
    lines.append("=" * 88)
    lines.append(f"SWC: {swc_path}")
    lines.append(f"Topology nodes: {len(node_df)}")
    lines.append(f"Topology corridors: {len(edge_df)}")
    lines.append(f"Kept node IDs: {', '.join(map(str, keep_node_ids)) if keep_node_ids else '(none)'}")
    lines.append("")

    if not node_df.empty:
        lines.append("Topological nodes")
        lines.append("-" * 88)
        cols = [
            c for c in [
                "topology_node_id",
                "swc_node_id",
                "kind",
                "degree",
                "n_incident_corridors",
                "badness_score",
                "is_suspicious",
                "reason",
            ]
            if c in node_df.columns
        ]
        lines.append(node_df[cols].head(30).to_string(index=False))
        lines.append("")

    if not edge_df.empty:
        lines.append("Corridors")
        lines.append("-" * 88)
        cols = [
            c for c in [
                "corridor_id",
                "start_swc_node_id",
                "end_swc_node_id",
                "n_swc_nodes",
                "cable_length_um",
                "euclidean_length_um",
                "length_ratio",
                "max_badness_score_on_corridor",
                "n_suspicious_nodes_on_corridor",
            ]
            if c in edge_df.columns
        ]
        lines.append(edge_df[cols].head(30).to_string(index=False))
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def smooth_radii_along_tree(graph) -> np.ndarray:
    r = graph.radii
    r_smooth = np.zeros_like(r, dtype=float)

    for i in range(len(graph.ids)):
        samples = [(0.0, float(r[i]))]

        dist = 0.0
        cur = i
        while True:
            parent_id = int(graph.parents[cur])
            if parent_id == -1:
                break
            j = graph.id_to_idx.get(parent_id)
            if j is None:
                break
            dist += float(graph.seg_len[cur])
            if dist > SMOOTH_PATH_UM:
                break
            samples.append((dist, float(r[j])))
            cur = j

        dist = 0.0
        cur = i
        while True:
            if len(graph.children[cur]) != 1:
                break
            c = graph.children[cur][0]
            dist += float(graph.seg_len[c])
            if dist > SMOOTH_PATH_UM:
                break
            samples.append((dist, float(r[c])))
            cur = c

        dists = np.array([d for d, _ in samples], dtype=float)
        vals = np.array([v for _, v in samples], dtype=float)

        sigma = SMOOTH_PATH_UM / 2.0
        if sigma > 0:
            weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
            r_smooth[i] = np.sum(weights * vals) / np.sum(weights)
        else:
            r_smooth[i] = r[i]

    return r_smooth


def radius_to_t(radius: float, rmin: float, rmax: float) -> float:
    if rmax == rmin:
        return 0.5
    clipped = min(max(radius, rmin), rmax)
    return (clipped - rmin) / (rmax - rmin)


def radius_to_color(radius: float, rmin: float, rmax: float) -> str:
    return sample_colorscale("Viridis", [radius_to_t(radius, rmin, rmax)])[0]


def radius_to_width(radius: float, rmin: float, rmax: float) -> float:
    t = radius_to_t(radius, rmin, rmax)
    return MIN_LINE_WIDTH + t * (MAX_LINE_WIDTH - MIN_LINE_WIDTH)


def kind_to_color(kind: str) -> str:
    if "root" in kind:
        return ROOT_COLOR
    if "branchpoint" in kind:
        return BRANCH_COLOR
    if "terminal" in kind:
        return TERMINAL_COLOR
    if "kept_degree2" in kind:
        return KEPT_DEG2_COLOR
    return "#95A5A6"


def kind_to_size(kind: str, suspicious: bool) -> int:
    if suspicious:
        return 12
    if "root" in kind:
        return 14
    if "branchpoint" in kind:
        return 11
    if "terminal" in kind:
        return 9
    if "kept_degree2" in kind:
        return 10
    return 8


def make_html_viewer(graph, node_df: pd.DataFrame, edge_df: pd.DataFrame, out_html: Path) -> None:
    r_smooth = smooth_radii_along_tree(graph)

    r_nonzero = r_smooth[r_smooth > 0]
    if len(r_nonzero) == 0:
        raise ValueError("All smoothed radii are zero.")

    rmin = float(np.percentile(r_nonzero, 5))
    rmax = float(np.percentile(r_nonzero, 95))

    xmin, xmax = float(graph.xyz[:, 0].min()), float(graph.xyz[:, 0].max())
    ymin, ymax = float(graph.xyz[:, 1].min()), float(graph.xyz[:, 1].max())
    zmin, zmax = float(graph.xyz[:, 2].min()), float(graph.xyz[:, 2].max())

    raw_len = (xmax - xmin) * 0.1
    scale_len = round(raw_len / 10) * 10
    scale_len = max(20, min(200, scale_len))

    x0 = xmin + OFFSET
    y0 = ymin + OFFSET
    z0 = zmin + OFFSET

    fig = go.Figure()

    # Background skeleton in Viridis by radius.
    edges = np.linspace(rmin, rmax, N_COLOR_BINS + 1)
    for b in range(N_COLOR_BINS):
        low = edges[b]
        high = edges[b + 1]
        xs, ys, zs = [], [], []

        for i, pid in enumerate(graph.parents):
            if pid == -1:
                continue
            j = graph.id_to_idx.get(int(pid))
            if j is None:
                continue

            child_radius = min(max(float(r_smooth[i]), rmin), rmax)
            in_bin = (low <= child_radius < high) or (b == N_COLOR_BINS - 1 and child_radius == high)
            if not in_bin:
                continue

            xs.extend([float(graph.xyz[i, 0]), float(graph.xyz[j, 0]), None])
            ys.extend([float(graph.xyz[i, 1]), float(graph.xyz[j, 1]), None])
            zs.extend([float(graph.xyz[i, 2]), float(graph.xyz[j, 2]), None])

        if xs:
            mid = 0.5 * (low + high)
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(
                        width=radius_to_width(mid, rmin, rmax),
                        color=radius_to_color(mid, rmin, rmax),
                    ),
                    hoverinfo="skip",
                    opacity=0.25,
                    showlegend=False,
                )
            )

    # Colorbar handle.
    fig.add_trace(
        go.Scatter3d(
            x=graph.xyz[:, 0],
            y=graph.xyz[:, 1],
            z=graph.xyz[:, 2],
            mode="markers",
            marker=dict(
                size=0.1,
                color=np.clip(r_smooth, rmin, rmax),
                colorscale="Viridis",
                cmin=rmin,
                cmax=rmax,
                colorbar=dict(
                    title="Radius (µm)",
                    x=1.0,
                    xanchor="left",
                    y=0.5,
                    len=0.5,
                    thickness=20,
                ),
                showscale=True,
                opacity=0,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Corridor overlays.
    if not edge_df.empty:
        score_min = float(edge_df["max_badness_score_on_corridor"].min())
        score_max = float(edge_df["max_badness_score_on_corridor"].max())

        for _, row in edge_df.iterrows():
            path_idx = [int(x) for x in str(row["swc_node_indices"]).split(";")]
            if len(path_idx) < 2:
                continue

            t = 0.5 if score_max == score_min else (float(row["max_badness_score_on_corridor"]) - score_min) / (score_max - score_min)
            t = min(max(t, 0.0), 1.0)
            corridor_color = sample_colorscale("Viridis", [t])[0]

            xs, ys, zs = [], [], []
            for a, b in zip(path_idx[:-1], path_idx[1:]):
                xs.extend([float(graph.xyz[a, 0]), float(graph.xyz[b, 0]), None])
                ys.extend([float(graph.xyz[a, 1]), float(graph.xyz[b, 1]), None])
                zs.extend([float(graph.xyz[a, 2]), float(graph.xyz[b, 2]), None])

            ratio_val = row["length_ratio"]
            ratio_txt = "nan" if pd.isna(ratio_val) else f"{float(ratio_val):.3f}"

            hover = (
                f"Corridor {int(row['corridor_id'])}<br>"
                f"Start topology node: {int(row['start_topology_node_id'])}<br>"
                f"End topology node: {int(row['end_topology_node_id'])}<br>"
                f"Start SWC node: {int(row['start_swc_node_id'])}<br>"
                f"End SWC node: {int(row['end_swc_node_id'])}<br>"
                f"Nodes on corridor: {int(row['n_swc_nodes'])}<br>"
                f"Cable length: {float(row['cable_length_um']):.3f} µm<br>"
                f"Length ratio: {ratio_txt}<br>"
                f"Max badness: {float(row['max_badness_score_on_corridor']):.3f}"
            )

            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=corridor_color, width=8),
                    hovertext=hover,
                    hoverinfo="text",
                    showlegend=False,
                    opacity=0.95,
                )
            )

    # Topology nodes.
    if not node_df.empty:
        for kind, group in node_df.groupby("kind", sort=False):
            color = kind_to_color(kind)
            idxs = group["swc_node_index"].astype(int).tolist()
            sizes = [kind_to_size(kind, bool(s)) for s in group["is_suspicious"].astype(bool).tolist()]

            hover = []
            text = []
            for _, row in group.iterrows():
                hover.append(
                    f"Topology node {int(row['topology_node_id'])}<br>"
                    f"SWC node: {int(row['swc_node_id'])}<br>"
                    f"Kind: {row['kind']}<br>"
                    f"Degree: {int(row['degree'])}<br>"
                    f"Badness score: {float(row['badness_score']):.3f}<br>"
                    f"Suspicious: {bool(row['is_suspicious'])}<br>"
                    f"Reason: {row['reason']}<br>"
                    f"Incident corridors: {int(row['n_incident_corridors'])}"
                )
                text.append(str(int(row["swc_node_id"])))

            fig.add_trace(
                go.Scatter3d(
                    x=[float(graph.xyz[i, 0]) for i in idxs],
                    y=[float(graph.xyz[i, 1]) for i in idxs],
                    z=[float(graph.xyz[i, 2]) for i in idxs],
                    mode="markers+text",
                    marker=dict(size=sizes, color=color, line=dict(color="black", width=2), opacity=0.98),
                    text=text,
                    textposition="top center",
                    hovertext=hover,
                    hoverinfo="text",
                    name=kind,
                    showlegend=False,
                )
            )

    # Soma sphere.
    soma_indices = np.where(graph.parents == -1)[0]
    if len(soma_indices) > 0:
        sidx = int(soma_indices[0])
        cx, cy, cz = graph.xyz[sidx]
        soma_radius = float(max(r_smooth[sidx], 1e-6))

        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        sx = cx + soma_radius * np.outer(np.cos(u), np.sin(v))
        sy = cy + soma_radius * np.outer(np.sin(u), np.sin(v))
        sz = cz + soma_radius * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=sx,
                y=sy,
                z=sz,
                opacity=0.95,
                colorscale=[[0, "black"], [1, "black"]],
                showscale=False,
                name="Soma",
            )
        )

    # Scale bars.
    fig.add_trace(
        go.Scatter3d(
            x=[x0, x0 + scale_len],
            y=[y0, y0],
            z=[z0, z0],
            mode="lines+text",
            line=dict(width=6, color="black"),
            text=["", f"{scale_len} µm"],
            textposition="top center",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x0, x0],
            y=[y0, y0 + scale_len],
            z=[z0, z0],
            mode="lines+text",
            line=dict(width=6, color="black"),
            text=["", f"{scale_len} µm"],
            textposition="middle right",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x0, x0],
            y=[y0, y0],
            z=[z0, z0 + scale_len],
            mode="lines+text",
            line=dict(width=6, color="black"),
            text=["", f"{scale_len} µm"],
            textposition="middle left",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Topology graph | nodes={len(node_df)} | corridors={len(edge_df)}",
        scene=dict(
            xaxis=dict(title="", showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="", showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title="", showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="white",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )

    pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a contracted topology graph and HTML viewer.")
    p.add_argument("--swc", type=Path, default=DEFAULT_SWC, help="Path to SWC file.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    p.add_argument(
        "--keep-node",
        type=int,
        action="append",
        default=[],
        help="SWC node ID to preserve as a topology node even if degree 2. Repeatable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ids, xyz, radii, parents = read_swc(args.swc)
    graph = build_graph_context(ids, xyz, radii, parents)

    node_df, edge_df = build_contracted_graph(graph, args.keep_node)

    node_csv = args.outdir / "topology_nodes.csv"
    edge_csv = args.outdir / "topology_edges.csv"
    summary_txt = args.outdir / "topology_summary.txt"
    html_path = args.outdir / "topology_graph.html"

    node_df.to_csv(node_csv, index=False)
    edge_df.to_csv(edge_csv, index=False)
    write_summary(summary_txt, args.swc, node_df, edge_df, args.keep_node)
    make_html_viewer(graph, node_df, edge_df, html_path)

    print(f"Wrote: {node_csv}")
    print(f"Wrote: {edge_csv}")
    print(f"Wrote: {summary_txt}")
    print(f"Wrote: {html_path}")


if __name__ == "__main__":
    main()
