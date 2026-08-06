#!/usr/bin/env python3
"""
FindAnomalousRegions.py

Scan an SWC tree with NodeClassifier.py and write:

  1) anomalous_nodes.csv
       One row per suspicious node.

  2) anomalous_regions.csv
       Connected groups of suspicious nodes.

  3) anomalous_regions.html
       Interactive Plotly viewer showing the Viridis skeleton with the
       anomalous regions overlaid.

This is a topology-quality map for the entire skeleton.

A node is considered suspicious according to NodeClassifier.classify_node()
using the feature values from NodeClassifier.analyze_node().

Region grouping is conservative:
  suspicious nodes are grouped into a region when they are connected by
  tree edges and both endpoints are suspicious.

Dependencies:
  numpy
  pandas
  plotly
  NodeClassifier.py (same directory or importable on PYTHONPATH)
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale

from NodeClassifier import build_graph_context, analyze_node

# ============================================================
# USER INPUTS
# ============================================================

SWC_PATH = Path("/Users/jhsinger/Documents/nNOS_AC_2026/EyewireAnalysis/720575940584776059/skeleton.swc")
OUTPUT_DIR = Path("/Users/jhsinger/Documents/nNOS_AC_2026/EyewireAnalysis/720575940584776059/anomaly_map")

# Optional override of NodeClassifier defaults.
MIN_DISTAL_COMPONENT_SIZE_UM = 250.0
MIN_NODE_DEGREE = 4
MIN_SUBSTANTIAL_DISTAL_COMPONENTS = 2

# If True, write a short summary to stdout.
VERBOSE = True

# Rendering controls
SMOOTH_PATH_UM = 30.0
N_COLOR_BINS = 24
MIN_LINE_WIDTH = 2
MAX_LINE_WIDTH = 6
OFFSET = 20

REGION_MARKER_SIZE = 7
REPRESENTATIVE_MARKER_SIZE = 12
REGION_LINE_WIDTH = 5
# ============================================================


def read_swc_for_graph(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read an SWC into arrays suitable for NodeClassifier.build_graph_context()."""
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


def connected_components_on_suspicious_subgraph(
    suspicious_indices: List[int],
    graph_adj: List[List[int]],
) -> List[List[int]]:
    """Return connected components in the induced subgraph of suspicious nodes."""
    suspicious_set = set(suspicious_indices)
    comps: List[List[int]] = []
    seen = set()

    for start in suspicious_indices:
        if start in seen:
            continue
        q = deque([start])
        comp = []

        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)

            for v in graph_adj[u]:
                if v in suspicious_set and v not in seen:
                    q.append(v)

        comps.append(sorted(comp))

    comps.sort(key=lambda c: (-len(c), min(c) if c else 10**18))
    return comps


def summarise_region(region_id: int, comp: List[int], node_results: Dict[int, dict], graph) -> dict:
    """Summarize a connected suspicious-node component."""
    rows = [node_results[i] for i in comp]
    badness_scores = np.array([float(r["badness_score"]) for r in rows], dtype=float)
    node_ids = [int(r["node_id"]) for r in rows]
    degrees = [int(r["degree"]) for r in rows]

    xyz = graph.xyz[np.array(comp, dtype=int)]
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    centroid = xyz.mean(axis=0)

    rep = max(
        rows,
        key=lambda r: (float(r["badness_score"]), int(r["n_distal_components_ge_um"])),
    )

    reason_counts: Dict[str, int] = {}
    for r in rows:
        reason = str(r.get("reason", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "region_id": region_id,
        "n_nodes": int(len(comp)),
        "representative_node_id": int(rep["node_id"]),
        "representative_node_index": int(rep["node_index"]),
        "max_badness_score": float(badness_scores.max()) if len(badness_scores) else np.nan,
        "mean_badness_score": float(badness_scores.mean()) if len(badness_scores) else np.nan,
        "median_badness_score": float(np.median(badness_scores)) if len(badness_scores) else np.nan,
        "node_ids": ";".join(map(str, node_ids)),
        "node_indices": ";".join(map(str, comp)),
        "degrees": ";".join(map(str, degrees)),
        "centroid_x_um": float(centroid[0]),
        "centroid_y_um": float(centroid[1]),
        "centroid_z_um": float(centroid[2]),
        "bbox_min_x_um": float(bbox_min[0]),
        "bbox_min_y_um": float(bbox_min[1]),
        "bbox_min_z_um": float(bbox_min[2]),
        "bbox_max_x_um": float(bbox_max[0]),
        "bbox_max_y_um": float(bbox_max[1]),
        "bbox_max_z_um": float(bbox_max[2]),
        "reasons_json": str(reason_counts),
    }


def smooth_radii_along_tree(graph) -> np.ndarray:
    """Smooth the radius along the tree, following the same logic as the node viewer."""
    r = graph.radii
    r_smooth = np.zeros_like(r, dtype=float)

    for i in range(len(graph.ids)):
        samples = [(0.0, float(r[i]))]

        # Upstream.
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

        # Downstream only along single-child continuation.
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


def make_html_viewer(graph, node_df: pd.DataFrame, region_df: pd.DataFrame, out_html: Path) -> None:
    """Write a browser-friendly HTML viewer showing anomalous regions on the skeleton."""
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

    # Skeleton edges in Viridis bins.
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
                    showlegend=False,
                )
            )

    # Invisible marker cloud to attach a colorbar.
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

    # Region overlays.
    if not region_df.empty and not node_df.empty:
        region_score_min = float(region_df["max_badness_score"].min())
        region_score_max = float(region_df["max_badness_score"].max())

        region_lookup = {int(row["region_id"]): row for _, row in region_df.iterrows()}
        node_groups = {}
        for _, row in node_df.iterrows():
            rid = int(row["region_id"])
            node_groups.setdefault(rid, []).append(row)

        for region_id, rows in sorted(node_groups.items(), key=lambda kv: kv[0]):
            region = region_lookup.get(region_id)
            if region is None:
                continue

            score = float(region["max_badness_score"])
            t = 0.5 if region_score_max == region_score_min else (score - region_score_min) / (region_score_max - region_score_min)
            region_color = sample_colorscale("Viridis", [np.clip(t, 0.0, 1.0)])[0]

            region_indices = [int(r["node_index"]) for r in rows]
            region_set = set(region_indices)

            # Edges inside the region.
            edge_x, edge_y, edge_z = [], [], []
            seen_edges = set()
            for u in region_indices:
                for v in graph.adj[u]:
                    if v not in region_set:
                        continue
                    edge = tuple(sorted((int(u), int(v))))
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    edge_x.extend([float(graph.xyz[u, 0]), float(graph.xyz[v, 0]), None])
                    edge_y.extend([float(graph.xyz[u, 1]), float(graph.xyz[v, 1]), None])
                    edge_z.extend([float(graph.xyz[u, 2]), float(graph.xyz[v, 2]), None])

            if edge_x:
                fig.add_trace(
                    go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode="lines",
                        line=dict(color=region_color, width=REGION_LINE_WIDTH),
                        hoverinfo="skip",
                        name=f"Region {region_id}",
                        showlegend=False,
                    )
                )

            # Node markers for the region.
            marker_x, marker_y, marker_z = [], [], []
            marker_sizes = []
            marker_hover = []
            marker_text = []
            rep_idx = int(region["representative_node_index"])

            for row in rows:
                idx = int(row["node_index"])
                marker_x.append(float(graph.xyz[idx, 0]))
                marker_y.append(float(graph.xyz[idx, 1]))
                marker_z.append(float(graph.xyz[idx, 2]))
                marker_sizes.append(REPRESENTATIVE_MARKER_SIZE if idx == rep_idx else REGION_MARKER_SIZE)
                marker_text.append(f"R{region_id}" if idx == rep_idx else "")

                pm = row["pairing_margin"]
                pm_txt = "nan" if pd.isna(pm) else f"{float(pm):.3f}"
                marker_hover.append(
                    f"Region {region_id}<br>"
                    f"Node {int(row['node_id'])}<br>"
                    f"Badness score: {float(row['badness_score']):.3f}<br>"
                    f"Degree: {int(row['degree'])}<br>"
                    f"Reason: {row['reason']}<br>"
                    f"Pairing margin: {pm_txt}"
                )

            fig.add_trace(
                go.Scatter3d(
                    x=marker_x,
                    y=marker_y,
                    z=marker_z,
                    mode="markers+text",
                    marker=dict(
                        size=marker_sizes,
                        color=region_color,
                        opacity=0.95,
                        line=dict(color="black", width=2),
                    ),
                    text=marker_text,
                    textposition="top center",
                    hovertext=marker_hover,
                    hoverinfo="text",
                    name=f"Region {region_id}",
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
        title=f"Anomalous regions | nodes={len(node_df)} | regions={len(region_df)}",
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


def main() -> None:
    if not SWC_PATH.exists():
        raise FileNotFoundError(f"SWC not found: {SWC_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ids, xyz, radii, parents = read_swc_for_graph(SWC_PATH)
    graph = build_graph_context(ids, xyz, radii, parents)

    node_results: Dict[int, dict] = {}
    suspicious_indices: List[int] = []

    for node_idx in range(len(graph.ids)):
        analysis = analyze_node(
            node_idx,
            graph,
            MIN_DISTAL_COMPONENT_SIZE_UM=MIN_DISTAL_COMPONENT_SIZE_UM,
            MIN_NODE_DEGREE=MIN_NODE_DEGREE,
            MIN_SUBSTANTIAL_DISTAL_COMPONENTS=MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
        )
        node_results[node_idx] = analysis
        if bool(analysis.get("is_suspicious", False)):
            suspicious_indices.append(node_idx)

    suspicious_indices.sort()

    # Regions first so node table can reference region_id.
    region_comps = connected_components_on_suspicious_subgraph(suspicious_indices, graph.adj)

    node_to_region: Dict[int, int] = {}
    region_rows: List[dict] = []
    for region_id, comp in enumerate(region_comps, start=1):
        for idx in comp:
            node_to_region[int(idx)] = int(region_id)
        region_rows.append(summarise_region(region_id, comp, node_results, graph))

    node_rows: List[dict] = []
    for node_idx in suspicious_indices:
        a = node_results[node_idx]
        pm = a["pairing_margin"]
        node_rows.append(
            {
                "node_index": int(a["node_index"]),
                "node_id": int(a["node_id"]),
                "region_id": int(node_to_region.get(int(node_idx), -1)),
                "degree": int(a["degree"]),
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
                "third_distal_component_cable_um": float(a["third_distal_component_cable_um"])
                if np.isfinite(a["third_distal_component_cable_um"])
                else np.nan,
                "largest_distal_fraction": float(a["largest_distal_fraction"])
                if np.isfinite(a["largest_distal_fraction"])
                else np.nan,
                "second_distal_fraction": float(a["second_distal_fraction"])
                if np.isfinite(a["second_distal_fraction"])
                else np.nan,
                "third_distal_fraction": float(a["third_distal_fraction"])
                if np.isfinite(a["third_distal_fraction"])
                else np.nan,
                "largest_to_second_distal_ratio": float(a["largest_to_second_distal_ratio"])
                if np.isfinite(a["largest_to_second_distal_ratio"])
                else np.nan,
                "n_distal_components_ge_um": int(a["n_distal_components_ge_um"]),
                "badness_score": float(a["badness_score"]),
                "reason": str(a["reason"]),
                "is_suspicious": int(bool(a["is_suspicious"])),
                "pairing_score": float(a["pairing_score"])
                if np.isfinite(a["pairing_score"])
                else np.nan,
                "pairing_second_score": float(a["pairing_second_score"])
                if np.isfinite(a["pairing_second_score"])
                else np.nan,
                "pairing_margin": float(pm) if np.isfinite(pm) else np.nan,
                "best_pairing": str(a["best_pairing"]),
                "neighbor_ids": ";".join(map(str, a["neighbor_ids"])),
                "distal_component_cables_um": ";".join(f"{x:.3f}" for x in a["distal_component_cables_um"]),
                "distal_component_fractions": ";".join(
                    f"{x:.4f}" for x in a["distal_component_fractions"] if np.isfinite(x)
                ),
            }
        )

    node_columns = [
        "node_index",
        "node_id",
        "region_id",
        "degree",
        "root_distance_um",
        "depth_edges",
        "n_components",
        "n_soma_components",
        "n_terminal_components",
        "n_branchpoint_components",
        "soma_component_cable_um",
        "largest_distal_component_cable_um",
        "second_distal_component_cable_um",
        "third_distal_component_cable_um",
        "largest_distal_fraction",
        "second_distal_fraction",
        "third_distal_fraction",
        "largest_to_second_distal_ratio",
        "n_distal_components_ge_um",
        "badness_score",
        "reason",
        "is_suspicious",
        "pairing_score",
        "pairing_second_score",
        "pairing_margin",
        "best_pairing",
        "neighbor_ids",
        "distal_component_cables_um",
        "distal_component_fractions",
    ]
    region_columns = [
        "region_id",
        "n_nodes",
        "representative_node_id",
        "representative_node_index",
        "max_badness_score",
        "mean_badness_score",
        "median_badness_score",
        "node_ids",
        "node_indices",
        "degrees",
        "centroid_x_um",
        "centroid_y_um",
        "centroid_z_um",
        "bbox_min_x_um",
        "bbox_min_y_um",
        "bbox_min_z_um",
        "bbox_max_x_um",
        "bbox_max_y_um",
        "bbox_max_z_um",
        "reasons_json",
    ]

    node_df = pd.DataFrame(node_rows, columns=node_columns)
    region_df = pd.DataFrame(region_rows, columns=region_columns)

    node_df = node_df.sort_values(
        ["badness_score", "degree", "pairing_margin"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    if not region_df.empty:
        region_df = region_df.sort_values(
            ["max_badness_score", "n_nodes"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    node_csv = OUTPUT_DIR / "anomalous_nodes.csv"
    region_csv = OUTPUT_DIR / "anomalous_regions.csv"
    html_path = OUTPUT_DIR / "anomalous_regions.html"

    node_df.to_csv(node_csv, index=False)
    region_df.to_csv(region_csv, index=False)
    make_html_viewer(graph, node_df, region_df, html_path)

    if not html_path.exists():
        raise RuntimeError(f"HTML write failed: {html_path}")

    if VERBOSE:
        print(f"Scanned {len(graph.ids)} nodes")
        print(f"Suspicious nodes: {len(node_df)}")
        print(f"Anomalous regions: {len(region_df)}")
        print(f"Wrote: {node_csv}")
        print(f"Wrote: {region_csv}")
        print(f"Wrote: {html_path}")

        if not node_df.empty:
            cols = [
                c for c in [
                    "node_id",
                    "region_id",
                    "degree",
                    "badness_score",
                    "reason",
                    "pairing_margin",
                    "n_distal_components_ge_um",
                    "largest_distal_component_cable_um",
                    "second_distal_component_cable_um",
                ]
                if c in node_df.columns
            ]
            print("\nTop suspicious nodes:")
            print(node_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
