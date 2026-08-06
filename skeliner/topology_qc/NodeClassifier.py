#!/usr/bin/env python3
"""
NodeClassifier.py

Reusable node-level topology analysis for SWC trees.

This module is intentionally I/O-free. It exposes small, composable
functions that can be imported by BranchDistance.py and future scripts.

Current heuristic:
  A node is suspicious when it has degree >= 4 and at least two distal
  components each exceed a size threshold (default: 250 um).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math

import numpy as np

DEFAULT_MIN_DISTAL_COMPONENT_SIZE_UM = 250.0
DEFAULT_MIN_NODE_DEGREE = 4
DEFAULT_MIN_SUBSTANTIAL_DISTAL_COMPONENTS = 2


@dataclass
class SWCGraph:
    ids: np.ndarray
    xyz: np.ndarray
    radii: np.ndarray
    parents: np.ndarray
    id_to_idx: Dict[int, int]
    children: List[List[int]]
    adj: List[List[int]]
    seg_len: np.ndarray
    root_dist: np.ndarray
    depth_edges: np.ndarray
    soma_idx: Optional[int]


def build_children(parents: np.ndarray, id_to_idx: Dict[int, int]) -> List[List[int]]:
    children = [[] for _ in range(len(parents))]
    for child_idx, pid in enumerate(parents):
        if pid == -1:
            continue
        parent_idx = id_to_idx.get(int(pid))
        if parent_idx is not None:
            children[parent_idx].append(child_idx)
    return children


def build_undirected_adj(parents: np.ndarray, id_to_idx: Dict[int, int]) -> List[List[int]]:
    adj = [[] for _ in range(len(parents))]
    for child_idx, pid in enumerate(parents):
        if pid == -1:
            continue
        parent_idx = id_to_idx.get(int(pid))
        if parent_idx is None:
            continue
        adj[parent_idx].append(child_idx)
        adj[child_idx].append(parent_idx)
    return adj


def compute_seg_lengths(xyz: np.ndarray, parents: np.ndarray, id_to_idx: Dict[int, int]) -> np.ndarray:
    seg_len = np.zeros(len(parents), dtype=float)
    for child_idx, pid in enumerate(parents):
        if pid == -1:
            continue
        parent_idx = id_to_idx.get(int(pid))
        if parent_idx is not None:
            seg_len[child_idx] = np.linalg.norm(xyz[child_idx] - xyz[parent_idx])
    return seg_len


def compute_root_dist(parents: np.ndarray, seg_len: np.ndarray, id_to_idx: Dict[int, int]) -> np.ndarray:
    root_dist = np.full(len(parents), np.nan, dtype=float)

    def compute(i: int) -> float:
        if not np.isnan(root_dist[i]):
            return float(root_dist[i])

        pid = int(parents[i])
        if pid == -1:
            root_dist[i] = 0.0
            return 0.0

        j = id_to_idx.get(pid)
        if j is None:
            root_dist[i] = np.nan
            return float("nan")

        pd = compute(j)
        if math.isnan(pd):
            root_dist[i] = np.nan
            return float("nan")

        root_dist[i] = pd + seg_len[i]
        return float(root_dist[i])

    for i in range(len(parents)):
        compute(i)

    return root_dist


def build_depth_edges(parents: np.ndarray, id_to_idx: Dict[int, int]) -> np.ndarray:
    depth = np.full(len(parents), -1, dtype=int)

    def compute(i: int) -> int:
        if depth[i] >= 0:
            return int(depth[i])

        pid = int(parents[i])
        if pid == -1:
            depth[i] = 0
            return 0

        j = id_to_idx.get(pid)
        if j is None:
            depth[i] = -1
            return -1

        d = compute(j)
        depth[i] = d + 1 if d >= 0 else -1
        return int(depth[i])

    for i in range(len(parents)):
        compute(i)

    return depth


def get_soma_idx(parents: np.ndarray) -> Optional[int]:
    roots = np.where(parents == -1)[0]
    return int(roots[0]) if len(roots) else None


def build_graph_context(
    ids: np.ndarray,
    xyz: np.ndarray,
    radii: np.ndarray,
    parents: np.ndarray,
) -> SWCGraph:
    id_to_idx = {int(nid): int(i) for i, nid in enumerate(ids)}
    children = build_children(parents, id_to_idx)
    adj = build_undirected_adj(parents, id_to_idx)
    seg_len = compute_seg_lengths(xyz, parents, id_to_idx)
    root_dist = compute_root_dist(parents, seg_len, id_to_idx)
    depth_edges = build_depth_edges(parents, id_to_idx)
    soma_idx = get_soma_idx(parents)
    return SWCGraph(
        ids=ids,
        xyz=xyz,
        radii=radii,
        parents=parents,
        id_to_idx=id_to_idx,
        children=children,
        adj=adj,
        seg_len=seg_len,
        root_dist=root_dist,
        depth_edges=depth_edges,
        soma_idx=soma_idx,
    )


def incident_neighbors(node_idx: int, graph: SWCGraph) -> List[int]:
    neigh = []
    pid = int(graph.parents[node_idx])
    if pid != -1 and pid in graph.id_to_idx:
        neigh.append(graph.id_to_idx[pid])
    neigh.extend(graph.children[node_idx])

    out, seen = [], set()
    for n in neigh:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def ancestor_chain(i: int, parents: np.ndarray, id_to_idx: Dict[int, int]) -> List[int]:
    chain = [i]
    while parents[i] != -1:
        i = id_to_idx[int(parents[i])]
        chain.append(i)
    return chain


def lowest_common_ancestor(i: int, j: int, parents: np.ndarray, id_to_idx: Dict[int, int]) -> Optional[int]:
    seen = set(ancestor_chain(i, parents, id_to_idx))
    while j != -1:
        if j in seen:
            return j
        pj = int(parents[j])
        if pj == -1:
            break
        j = id_to_idx.get(pj, -1)
    return None


def node_path_between(i: int, j: int, parents: np.ndarray, id_to_idx: Dict[int, int]) -> List[int]:
    ai = ancestor_chain(i, parents, id_to_idx)
    aj = ancestor_chain(j, parents, id_to_idx)
    ai_set = set(ai)

    lca = None
    for node in aj:
        if node in ai_set:
            lca = node
            break
    if lca is None:
        return []

    part_i = []
    cur = i
    while cur != lca:
        part_i.append(cur)
        cur = id_to_idx[int(parents[cur])]
    part_i.append(lca)

    part_j = []
    cur = j
    while cur != lca:
        part_j.append(cur)
        cur = id_to_idx[int(parents[cur])]

    return part_i + part_j[::-1]


def bfs_component(start: int, blocked: int, adj: List[List[int]]) -> List[int]:
    q = deque([start])
    seen = {blocked}
    comp = []

    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        comp.append(u)
        for v in adj[u]:
            if v not in seen:
                q.append(v)

    return comp


def component_metrics(comp: List[int], graph: SWCGraph, blocked: int) -> dict:
    comp_set = set(comp)
    if not comp_set:
        return {
            "contains_soma": False,
            "n_nodes": 0,
            "n_branchpoints": 0,
            "n_leaf_nodes": 0,
            "cable_length_um": 0.0,
            "max_depth_edges": 0,
            "bbox_min_x": np.nan,
            "bbox_min_y": np.nan,
            "bbox_min_z": np.nan,
            "bbox_max_x": np.nan,
            "bbox_max_y": np.nan,
            "bbox_max_z": np.nan,
            "mean_radius_um": np.nan,
            "median_radius_um": np.nan,
            "terminal_ids": "",
        }

    deg_local = {i: 0 for i in comp_set}
    cable = 0.0

    for child_idx in comp_set:
        pid = int(graph.parents[child_idx])
        if pid != -1 and pid in graph.id_to_idx:
            pidx = graph.id_to_idx[pid]
            if pidx in comp_set:
                cable += float(np.linalg.norm(graph.xyz[child_idx] - graph.xyz[pidx]))
                deg_local[child_idx] += 1
                deg_local[pidx] += 1

    for u in comp_set:
        pid = int(graph.parents[u])
        if pid != -1 and graph.id_to_idx.get(pid, None) == blocked:
            deg_local[u] += 1

    start = next(iter(comp_set))
    q = deque([(start, 0)])
    seen = set()
    dist = {}

    while q:
        u, d = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        dist[u] = d
        for v in graph.adj[u]:
            if v in comp_set and v not in seen:
                q.append((v, d + 1))

    xs = graph.xyz[list(comp_set), 0]
    ys = graph.xyz[list(comp_set), 1]
    zs = graph.xyz[list(comp_set), 2]
    rs = graph.radii[list(comp_set)]

    terminal_ids = sorted(int(graph.ids[i]) for i, d in deg_local.items() if d == 1)

    return {
        "contains_soma": bool(graph.soma_idx is not None and graph.soma_idx in comp_set),
        "n_nodes": int(len(comp_set)),
        "n_branchpoints": int(sum(1 for d in deg_local.values() if d >= 3)),
        "n_leaf_nodes": int(sum(1 for d in deg_local.values() if d == 1)),
        "cable_length_um": float(cable),
        "max_depth_edges": int(max(dist.values())) if dist else 0,
        "bbox_min_x": float(xs.min()),
        "bbox_min_y": float(ys.min()),
        "bbox_min_z": float(zs.min()),
        "bbox_max_x": float(xs.max()),
        "bbox_max_y": float(ys.max()),
        "bbox_max_z": float(zs.max()),
        "mean_radius_um": float(np.mean(rs)),
        "median_radius_um": float(np.median(rs)),
        "terminal_ids": ";".join(map(str, terminal_ids)),
    }


def unit_vector(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return vec / n


def arm_pair_score(u: np.ndarray, v: np.ndarray) -> float:
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        return float("inf")
    return float(1.0 + np.dot(u, v))


def best_4way_pairing(node_idx: int, neighbors: List[int], graph: SWCGraph) -> dict:
    if len(neighbors) != 4:
        return {}

    pairings = [
        [(0, 1), (2, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
    ]

    units = [unit_vector(graph.xyz[n] - graph.xyz[node_idx]) for n in neighbors]
    scored = []
    for pairing in pairings:
        score = sum(arm_pair_score(units[i], units[j]) for i, j in pairing)
        scored.append((float(score), pairing))

    scored.sort(key=lambda t: t[0])
    best_score, best_pairing = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else np.nan

    return {
        "best_score": float(best_score),
        "second_score": float(second_score),
        "pairing_margin": float(second_score - best_score) if np.isfinite(second_score) else np.nan,
        "best_pairing": ";".join(f"({neighbors[i]},{neighbors[j]})" for i, j in best_pairing),
        "all_pairings_json": json.dumps(
            [
                {
                    "score": float(score),
                    "pairs": [[int(neighbors[i]), int(neighbors[j])] for i, j in pairing],
                }
                for score, pairing in scored
            ]
        ),
    }


def classify_node(
    features: Dict[str, Any],
    MIN_DISTAL_COMPONENT_SIZE_UM: float = DEFAULT_MIN_DISTAL_COMPONENT_SIZE_UM,
    MIN_NODE_DEGREE: int = DEFAULT_MIN_NODE_DEGREE,
    MIN_SUBSTANTIAL_DISTAL_COMPONENTS: int = DEFAULT_MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
) -> dict:
    degree = int(features.get("degree", 0))
    n_large = int(features.get("n_distal_components_ge_um", 0))

    if degree >= MIN_NODE_DEGREE and n_large >= MIN_SUBSTANTIAL_DISTAL_COMPONENTS:
        reason = f"degree{degree}_distal_components_ge_{int(MIN_DISTAL_COMPONENT_SIZE_UM)}um_{n_large}"
        score = float(max(0, n_large - 1))
        suspicious = True
    else:
        reason = f"degree{degree}_distal_components_ge_{int(MIN_DISTAL_COMPONENT_SIZE_UM)}um_{n_large}_below_threshold"
        score = 0.0
        suspicious = False

    return {
        "is_suspicious": bool(suspicious),
        "badness_score": float(score),
        "reason": reason,
    }


def analyze_node(
    node_idx: int,
    graph: SWCGraph,
    MIN_DISTAL_COMPONENT_SIZE_UM: float = DEFAULT_MIN_DISTAL_COMPONENT_SIZE_UM,
    MIN_NODE_DEGREE: int = DEFAULT_MIN_NODE_DEGREE,
    MIN_SUBSTANTIAL_DISTAL_COMPONENTS: int = DEFAULT_MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
) -> dict:
    neighbors = incident_neighbors(node_idx, graph)
    comps = [bfs_component(nbr, node_idx, graph.adj) for nbr in neighbors]
    comp_metrics = [component_metrics(comp, graph, node_idx) for comp in comps]

    pairing = best_4way_pairing(node_idx, neighbors, graph) if len(neighbors) == 4 else {}

    arm_summaries = []
    for arm_i, (nbr_idx, comp, met) in enumerate(zip(neighbors, comps, comp_metrics)):
        arm_summaries.append(
            {
                "arm_index": arm_i,
                "neighbor_index": int(nbr_idx),
                "neighbor_id": int(graph.ids[nbr_idx]),
                "contains_soma": bool(met["contains_soma"]),
                "n_nodes": int(met["n_nodes"]),
                "n_branchpoints": int(met["n_branchpoints"]),
                "n_leaf_nodes": int(met["n_leaf_nodes"]),
                "cable_length_um": float(met["cable_length_um"]),
                "max_depth_edges": int(met["max_depth_edges"]),
                "mean_radius_um": float(met["mean_radius_um"]),
                "median_radius_um": float(met["median_radius_um"]),
                "terminal_ids": met["terminal_ids"],
                "arm_vector_um": (
                    float(graph.xyz[nbr_idx, 0] - graph.xyz[node_idx, 0]),
                    float(graph.xyz[nbr_idx, 1] - graph.xyz[node_idx, 1]),
                    float(graph.xyz[nbr_idx, 2] - graph.xyz[node_idx, 2]),
                ),
            }
        )

    soma_component = np.nan
    distal_components = []
    for met in comp_metrics:
        if met["contains_soma"]:
            soma_component = float(met["cable_length_um"])
        else:
            distal_components.append(float(met["cable_length_um"]))

    distal_components.sort(reverse=True)
    distal_total = sum(distal_components)
    distal_fractions = [x / distal_total for x in distal_components] if distal_total > 0 else [np.nan] * len(distal_components)

    n_large_distal = int(sum(c >= MIN_DISTAL_COMPONENT_SIZE_UM for c in distal_components))

    analysis = {
        "node_index": int(node_idx),
        "node_id": int(graph.ids[node_idx]),
        "degree": int(len(neighbors)),
        "root_distance_um": float(graph.root_dist[node_idx]),
        "depth_edges": int(graph.depth_edges[node_idx]),
        "neighbor_indices": [int(n) for n in neighbors],
        "neighbor_ids": [int(graph.ids[n]) for n in neighbors],
        "n_components": int(len(comps)),
        "n_soma_components": int(sum(m["contains_soma"] for m in comp_metrics)),
        "n_terminal_components": int(sum(m["n_leaf_nodes"] > 0 for m in comp_metrics)),
        "n_branchpoint_components": int(sum(m["n_branchpoints"] > 0 for m in comp_metrics)),
        "soma_component_cable_um": soma_component,
        "distal_component_cables_um": distal_components,
        "distal_component_fractions": distal_fractions,
        "largest_distal_component_cable_um": distal_components[0] if len(distal_components) > 0 else np.nan,
        "second_distal_component_cable_um": distal_components[1] if len(distal_components) > 1 else np.nan,
        "third_distal_component_cable_um": distal_components[2] if len(distal_components) > 2 else np.nan,
        "largest_distal_fraction": distal_fractions[0] if len(distal_fractions) > 0 else np.nan,
        "second_distal_fraction": distal_fractions[1] if len(distal_fractions) > 1 else np.nan,
        "third_distal_fraction": distal_fractions[2] if len(distal_fractions) > 2 else np.nan,
        "largest_to_second_distal_ratio": (
            distal_components[0] / distal_components[1]
            if len(distal_components) >= 2 and distal_components[1] > 0
            else np.nan
        ),
        "n_distal_components_ge_um": n_large_distal,
        "MIN_DISTAL_COMPONENT_SIZE_UM": float(MIN_DISTAL_COMPONENT_SIZE_UM),
        "pairing_score": float(pairing.get("best_score", np.nan)) if pairing else np.nan,
        "pairing_second_score": float(pairing.get("second_score", np.nan)) if pairing else np.nan,
        "pairing_margin": float(pairing.get("pairing_margin", np.nan)) if pairing else np.nan,
        "best_pairing": pairing.get("best_pairing", "") if pairing else "",
        "pairing_json": pairing.get("all_pairings_json", "") if pairing else "",
        "arm_summaries": arm_summaries,
    }

    analysis.update(
        classify_node(
            analysis,
            MIN_DISTAL_COMPONENT_SIZE_UM=MIN_DISTAL_COMPONENT_SIZE_UM,
            MIN_NODE_DEGREE=MIN_NODE_DEGREE,
            MIN_SUBSTANTIAL_DISTAL_COMPONENTS=MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
        )
    )
    return analysis


def classify_path(
    path_node_indices: Sequence[int],
    graph: SWCGraph,
    MIN_DISTAL_COMPONENT_SIZE_UM: float = DEFAULT_MIN_DISTAL_COMPONENT_SIZE_UM,
    MIN_NODE_DEGREE: int = DEFAULT_MIN_NODE_DEGREE,
    MIN_SUBSTANTIAL_DISTAL_COMPONENTS: int = DEFAULT_MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
    node_cache: Optional[Dict[int, dict]] = None,
) -> dict:
    cache = node_cache if node_cache is not None else {}
    node_results: List[dict] = []

    for node_idx in path_node_indices:
        if node_idx in cache:
            node_result = cache[node_idx]
        else:
            node_result = analyze_node(
                int(node_idx),
                graph,
                MIN_DISTAL_COMPONENT_SIZE_UM=MIN_DISTAL_COMPONENT_SIZE_UM,
                MIN_NODE_DEGREE=MIN_NODE_DEGREE,
                MIN_SUBSTANTIAL_DISTAL_COMPONENTS=MIN_SUBSTANTIAL_DISTAL_COMPONENTS,
            )
            cache[node_idx] = node_result
        node_results.append(node_result)

    suspicious = [r for r in node_results if r.get("is_suspicious", False)]

    if suspicious:
        worst = max(suspicious, key=lambda r: (r.get("badness_score", 0.0), r.get("n_distal_components_ge_um", 0)))
        return {
            "bad_path": True,
            "n_suspicious_nodes": int(len(suspicious)),
            "suspicious_node_ids": [int(r["node_id"]) for r in suspicious],
            "suspicious_node_indices": [int(r["node_index"]) for r in suspicious],
            "suspicious_node_scores": [float(r["badness_score"]) for r in suspicious],
            "max_badness_score": float(worst.get("badness_score", 0.0)),
            "worst_node_id": int(worst.get("node_id")),
            "worst_node_index": int(worst.get("node_index")),
            "worst_reason": str(worst.get("reason", "")),
            "node_results": node_results,
        }

    return {
        "bad_path": False,
        "n_suspicious_nodes": 0,
        "suspicious_node_ids": [],
        "suspicious_node_indices": [],
        "suspicious_node_scores": [],
        "max_badness_score": 0.0,
        "worst_node_id": None,
        "worst_node_index": None,
        "worst_reason": "",
        "node_results": node_results,
    }
