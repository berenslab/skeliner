#!/usr/bin/env python3
"""
NeighborhoodTree.py

Inspect the local topology around a chosen SWC node and generate:

  - <stem>_neighborhood.txt
  - <stem>_neighborhood.html

The HTML viewer uses Plotly and shows:
  - the full skeleton in Viridis,
  - the upstream chain toward the soma,
  - the downstream neighborhood away from the soma,
  - the chosen center node highlighted,
  - suspicious nodes highlighted using NodeClassifier.

Example:
  python NeighborhoodTree.py --node 1758 --up 100 --down 100

Dependencies:
  numpy
  plotly
  NodeClassifier.py (same directory or importable on PYTHONPATH)
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
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

# --------------------------------------------------
# Defaults
# --------------------------------------------------
DEFAULT_SWC = Path("/Users/jhsinger/Documents/nNOS_AC_2026/EyewireAnalysis/720575940584776059/skeleton.swc")
DEFAULT_OUTDIR = Path("/Users/jhsinger/Documents/nNOS_AC_2026/EyewireAnalysis/720575940584776059/neighborhood")

SMOOTH_PATH_UM = 30.0
N_COLOR_BINS = 24
MIN_LINE_WIDTH = 2
MAX_LINE_WIDTH = 6
OFFSET = 20

CENTER_COLOR = "#FFD54F"
UPSTREAM_COLOR = "#F39C12"
DOWNSTREAM_COLOR = "#2ECC71"
SUSPICIOUS_COLOR = "#E74C3C"


# --------------------------------------------------
# SWC helpers
# --------------------------------------------------
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


def smooth_radii_along_tree(graph) -> np.ndarray:
    r = graph.radii
    r_smooth = np.zeros_like(r, dtype=float)

    for i in range(len(graph.ids)):
        samples = [(0.0, float(r[i]))]

        # upstream
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

        # downstream only along single-child continuation
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


# --------------------------------------------------
# Neighborhood extraction
# --------------------------------------------------
def upstream_chain(center_idx: int, parents: np.ndarray, id_to_idx: Dict[int, int], max_edges: int) -> List[int]:
    chain = [center_idx]
    cur = center_idx
    steps = 0
    while steps < max_edges:
        pid = int(parents[cur])
        if pid == -1:
            break
        nxt = id_to_idx.get(pid)
        if nxt is None:
            break
        chain.append(nxt)
        cur = nxt
        steps += 1
    return chain


def downstream_subtree(center_idx: int, children: List[List[int]], max_edges: int) -> Tuple[Set[int], Dict[int, List[int]]]:
    included: Set[int] = {center_idx}
    restricted_children: Dict[int, List[int]] = {}
    q = deque([(center_idx, 0)])

    while q:
        u, d = q.popleft()
        if d >= max_edges:
            continue
        restricted_children.setdefault(u, [])
        for v in children[u]:
            included.add(v)
            restricted_children[u].append(v)
            q.append((v, d + 1))

    return included, restricted_children


def format_node_line(idx: int, graph, node_results: Dict[int, dict], prefix: str = "", marker: str = "") -> str:
    a = node_results[idx]
    deg = int(a["degree"])
    score = float(a["badness_score"])
    susp = "YES" if bool(a["is_suspicious"]) else "no"
    reason = str(a["reason"])
    node_id = int(a["node_id"])
    return f"{prefix}{marker}{node_id}  (idx={idx}, deg={deg}, score={score:.3f}, susp={susp}, {reason})"


def subtree_text(
    root_idx: int,
    restricted_children: Dict[int, List[int]],
    node_results: Dict[int, dict],
    graph,
    max_depth: int,
    depth: int = 0,
    is_last: bool = True,
    prefix: str = "",
) -> List[str]:
    lines = []
    branch = "└── " if is_last else "├── "
    if depth == 0:
        lines.append(format_node_line(root_idx, graph, node_results))
    else:
        lines.append(prefix + branch + format_node_line(root_idx, graph, node_results))

    children = restricted_children.get(root_idx, [])
    if not children or depth >= max_depth:
        return lines

    next_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        child_last = i == len(children) - 1
        lines.extend(
            subtree_text(
                child,
                restricted_children,
                node_results,
                graph,
                max_depth,
                depth=depth + 1,
                is_last=child_last,
                prefix=next_prefix,
            )
        )
    return lines


def build_node_results(center_nodes: Set[int], graph) -> Dict[int, dict]:
    node_results: Dict[int, dict] = {}
    for idx in sorted(center_nodes):
        node_results[idx] = analyze_node(
            idx,
            graph,
            MIN_DISTAL_COMPONENT_SIZE_UM=DEFAULT_MIN_DISTAL_COMPONENT_SIZE_UM,
            MIN_NODE_DEGREE=DEFAULT_MIN_NODE_DEGREE,
            MIN_SUBSTANTIAL_DISTAL_COMPONENTS=DEFAULT_MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
        )
    return node_results


# --------------------------------------------------
# Plotly viewer
# --------------------------------------------------
def make_html_viewer(
    graph,
    center_idx: int,
    up_chain: List[int],
    downstream_nodes: Set[int],
    restricted_children: Dict[int, List[int]],
    node_results: Dict[int, dict],
    out_html: Path,
) -> None:
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

    # Full skeleton in Viridis.
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

    # Upstream chain overlay.
    up_x, up_y, up_z = [], [], []
    for a, b in zip(up_chain[:-1], up_chain[1:]):
        up_x.extend([float(graph.xyz[a, 0]), float(graph.xyz[b, 0]), None])
        up_y.extend([float(graph.xyz[a, 1]), float(graph.xyz[b, 1]), None])
        up_z.extend([float(graph.xyz[a, 2]), float(graph.xyz[b, 2]), None])
    if up_x:
        fig.add_trace(
            go.Scatter3d(
                x=up_x,
                y=up_y,
                z=up_z,
                mode="lines",
                line=dict(color=UPSTREAM_COLOR, width=7),
                hoverinfo="skip",
                name="Upstream chain",
                showlegend=False,
            )
        )

    # Downstream overlay edges.
    down_x, down_y, down_z = [], [], []
    seen_edges = set()
    for u in sorted(downstream_nodes):
        for v in graph.adj[u]:
            if v not in downstream_nodes:
                continue
            edge = tuple(sorted((int(u), int(v))))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            down_x.extend([float(graph.xyz[u, 0]), float(graph.xyz[v, 0]), None])
            down_y.extend([float(graph.xyz[u, 1]), float(graph.xyz[v, 1]), None])
            down_z.extend([float(graph.xyz[u, 2]), float(graph.xyz[v, 2]), None])

    if down_x:
        fig.add_trace(
            go.Scatter3d(
                x=down_x,
                y=down_y,
                z=down_z,
                mode="lines",
                line=dict(color=DOWNSTREAM_COLOR, width=5),
                hoverinfo="skip",
                name="Downstream neighborhood",
                showlegend=False,
            )
        )

    # Marker groups: center, suspicious, upstream-only, downstream-only.
    all_nodes = set(up_chain) | set(downstream_nodes)
    suspicious_nodes = {idx for idx in all_nodes if bool(node_results[idx]["is_suspicious"])}
    upstream_only = [idx for idx in up_chain if idx not in suspicious_nodes and idx != center_idx]
    downstream_only = sorted([idx for idx in downstream_nodes if idx not in suspicious_nodes and idx != center_idx and idx not in set(up_chain)])
    suspicious_sorted = sorted(suspicious_nodes - {center_idx})

    def hover_for_idx(idx: int) -> str:
        a = node_results[idx]
        return (
            f"Node {int(a['node_id'])}<br>"
            f"Index: {int(a['node_index'])}<br>"
            f"Degree: {int(a['degree'])}<br>"
            f"Badness score: {float(a['badness_score']):.3f}<br>"
            f"Suspicious: {bool(a['is_suspicious'])}<br>"
            f"Reason: {a['reason']}<br>"
            f"Root distance: {float(a['root_distance_um']):.3f} µm"
        )

    # Center node.
    fig.add_trace(
        go.Scatter3d(
            x=[float(graph.xyz[center_idx, 0])],
            y=[float(graph.xyz[center_idx, 1])],
            z=[float(graph.xyz[center_idx, 2])],
            mode="markers+text",
            marker=dict(size=14, color=CENTER_COLOR, line=dict(color="black", width=2)),
            text=[str(int(graph.ids[center_idx]))],
            textposition="top center",
            hovertext=[hover_for_idx(center_idx)],
            hoverinfo="text",
            name="Center",
            showlegend=False,
        )
    )

    # Suspicious nodes.
    if suspicious_sorted:
        fig.add_trace(
            go.Scatter3d(
                x=[float(graph.xyz[idx, 0]) for idx in suspicious_sorted],
                y=[float(graph.xyz[idx, 1]) for idx in suspicious_sorted],
                z=[float(graph.xyz[idx, 2]) for idx in suspicious_sorted],
                mode="markers+text",
                marker=dict(size=10, color=SUSPICIOUS_COLOR, line=dict(color="black", width=2)),
                text=[str(int(graph.ids[idx])) for idx in suspicious_sorted],
                textposition="top center",
                hovertext=[hover_for_idx(idx) for idx in suspicious_sorted],
                hoverinfo="text",
                name="Suspicious",
                showlegend=False,
            )
        )

    # Upstream-only nodes.
    if upstream_only:
        fig.add_trace(
            go.Scatter3d(
                x=[float(graph.xyz[idx, 0]) for idx in upstream_only],
                y=[float(graph.xyz[idx, 1]) for idx in upstream_only],
                z=[float(graph.xyz[idx, 2]) for idx in upstream_only],
                mode="markers",
                marker=dict(size=7, color=UPSTREAM_COLOR, line=dict(color="black", width=1)),
                hovertext=[hover_for_idx(idx) for idx in upstream_only],
                hoverinfo="text",
                name="Upstream nodes",
                showlegend=False,
            )
        )

    # Downstream-only nodes.
    if downstream_only:
        fig.add_trace(
            go.Scatter3d(
                x=[float(graph.xyz[idx, 0]) for idx in downstream_only],
                y=[float(graph.xyz[idx, 1]) for idx in downstream_only],
                z=[float(graph.xyz[idx, 2]) for idx in downstream_only],
                mode="markers",
                marker=dict(size=7, color=DOWNSTREAM_COLOR, line=dict(color="black", width=1)),
                hovertext=[hover_for_idx(idx) for idx in downstream_only],
                hoverinfo="text",
                name="Downstream nodes",
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
        title=f"Neighborhood around node {int(graph.ids[center_idx])} | suspicious={len(suspicious_nodes)}",
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


def report_for_node(
    swc_path: Path,
    center_id: int,
    upstream_edges: int,
    downstream_edges: int,
    out_txt: Path,
    out_html: Path,
) -> None:
    ids, xyz, radii, parents = read_swc(swc_path)
    graph = build_graph_context(ids, xyz, radii, parents)

    if center_id not in graph.id_to_idx:
        raise ValueError(f"Center node {center_id} not found in SWC")

    center_idx = graph.id_to_idx[center_id]
    up_chain = upstream_chain(center_idx, graph.parents, graph.id_to_idx, upstream_edges)
    downstream_nodes, restricted_children = downstream_subtree(center_idx, graph.children, downstream_edges)

    neighborhood_nodes = set(up_chain) | set(downstream_nodes)
    node_results = build_node_results(neighborhood_nodes, graph)

    make_html_viewer(
        graph=graph,
        center_idx=center_idx,
        up_chain=up_chain,
        downstream_nodes=downstream_nodes,
        restricted_children=restricted_children,
        node_results=node_results,
        out_html=out_html,
    )

    lines: List[str] = []
    lines.append("=" * 88)
    lines.append(f"Neighborhood around node {center_id}")
    lines.append("=" * 88)
    lines.append(f"SWC: {swc_path}")
    lines.append(f"Center node id: {center_id}")
    lines.append(f"Center node index: {center_idx}")
    lines.append(f"Upstream edges: {upstream_edges}")
    lines.append(f"Downstream edges: {downstream_edges}")
    lines.append(f"Neighborhood node count: {len(neighborhood_nodes)}")
    lines.append("")

    a = node_results[center_idx]
    lines.append("CENTER SUMMARY")
    lines.append("-" * 88)
    lines.append(f"node_id: {int(a['node_id'])}")
    lines.append(f"index: {int(a['node_index'])}")
    lines.append(f"degree: {int(a['degree'])}")
    lines.append(f"root_distance_um: {float(a['root_distance_um']):.3f}")
    lines.append(f"depth_edges: {int(a['depth_edges'])}")
    lines.append(f"suspicious: {bool(a['is_suspicious'])}")
    lines.append(f"badness_score: {float(a['badness_score']):.3f}")
    lines.append(f"reason: {a['reason']}")
    lines.append(f"neighbor_ids: {a['neighbor_ids']}")
    lines.append("")

    lines.append("UPSTREAM CHAIN (toward soma)")
    lines.append("-" * 88)
    for step, idx in enumerate(up_chain):
        marker = "CENTER: " if step == 0 else f"-{step:03d}:   "
        lines.append(format_node_line(idx, graph, node_results, prefix=marker))
    lines.append("")

    lines.append("DOWNSTREAM TREE (away from soma)")
    lines.append("-" * 88)
    lines.extend(
        subtree_text(
            center_idx,
            restricted_children,
            node_results,
            graph,
            max_depth=downstream_edges,
        )
    )
    lines.append("")

    suspicious_nodes = [idx for idx in sorted(neighborhood_nodes) if bool(node_results[idx]["is_suspicious"])]
    lines.append("SUSPICIOUS NODES IN NEIGHBORHOOD")
    lines.append("-" * 88)
    if suspicious_nodes:
        for idx in suspicious_nodes:
            lines.append(format_node_line(idx, graph, node_results))
    else:
        lines.append("(none)")
    lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print a local neighborhood around an SWC node.")
    p.add_argument("--swc", type=Path, default=DEFAULT_SWC, help="Path to the SWC file.")
    p.add_argument("--node", type=int, default=1758, help="Center SWC node ID.")
    p.add_argument("--up", type=int, default=100, help="Number of edges to walk toward the soma.")
    p.add_argument("--down", type=int, default=100, help="Number of edges to traverse away from the center node.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    stem = f"node_{args.node}"
    out_txt = args.outdir / f"{stem}_neighborhood.txt"
    out_html = args.outdir / f"{stem}_neighborhood.html"

    report_for_node(
        swc_path=args.swc,
        center_id=args.node,
        upstream_edges=args.up,
        downstream_edges=args.down,
        out_txt=out_txt,
        out_html=out_html,
    )

    print(f"Wrote: {out_txt}")
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
