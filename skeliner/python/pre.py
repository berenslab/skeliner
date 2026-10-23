"""``skeliner.pre`` on the Python path: the preprocessing stages, as the
Rust crate ``pre`` runs them."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np
from scipy.spatial import KDTree

from .mesh import (
    border_loops,
    checked_faces,
    edge_to_faces,
    face_adjacency,
    nearest,
    non_degenerate,
    row_dot,
    rows3,
)


def sever_cost(sel: set[int], face_adj: dict[int, set[int]], budget: int) -> int:
    """How many faces removing *sel* would strand, capped at ``budget``.

    Everything but the largest surviving piece is stranded.  Each piece is
    grown separately and abandoned once it passes ``budget``: at most one
    piece can exceed the budget and still be the largest, and if two do the
    cost already exceeds it.  Returns ``budget + 1`` in that case.
    """
    seeds: set[int] = set()
    for fi in sel:
        seeds.update(f for f in face_adj[fi] if f not in sel)
    if not seeds:
        return 0

    seen: set[int] = set()
    closed: list[int] = []
    n_over = 0
    for seed in sorted(seeds):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        size = 0
        over = False
        while stack:
            u = stack.pop()
            size += 1
            if size > budget:
                over = True
                # u is popped but not yet walked; the drain below walks it
                stack.append(u)
                break
            for v in face_adj[u]:
                if v not in sel and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if over:
            n_over += 1
            if n_over >= 2:
                return budget + 1
            # drain the rest of this piece so its faces are not revisited
            while stack:
                u = stack.pop()
                for v in face_adj[u]:
                    if v not in sel and v not in seen:
                        seen.add(v)
                        stack.append(v)
        else:
            closed.append(size)

    if n_over:
        # the oversized piece is the largest; everything closed is lost
        return sum(closed)
    if not closed:
        return 0
    return sum(closed) - max(closed)


def expand_tip_to_good_rim(
    tip: list[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_adj: dict[int, set[int]],
    *,
    target_verts: int = 6,
    max_iters: int = 20,
    max_eat_factor: int = 30,
    faces: np.ndarray | None = None,
) -> tuple[set[int], list[int] | None]:
    """Peel BFS rings from *tip* until its border has at least
    *target_verts* vertices: ``(expanded tip, largest border loop found or
    None)``.  Growth is capped at ``len(tip) * max_eat_factor`` faces."""
    sel: set[int] = set(int(fi) for fi in tip)
    if not sel:
        return sel, None
    max_size = max(len(sel) * max_eat_factor, target_verts * 4)

    best_loop: list[int] | None = None
    for _ in range(max_iters):
        loops = border_loops(sel, edge_to_faces, faces)
        loops = [lp for lp in loops if len(lp) >= 3]
        if loops:
            biggest = max(loops, key=len)
            best_loop = biggest
            if len(biggest) >= target_verts:
                return sel, biggest
        # Expand by one BFS ring
        next_sel = set(sel)
        for fi in sel:
            for nfi in face_adj.get(fi, ()):
                next_sel.add(nfi)
        if next_sel == sel or len(next_sel) > max_size:
            break
        sel = next_sel

    return sel, best_loop


def expand_selection_per_group(
    sel: list[int] | set[int],
    edge_to_faces: dict[tuple[int, int], list[int]],
    face_adj: dict[int, set[int]],
    *,
    target_verts: int = 6,
    max_iters: int = 20,
    max_eat_factor: int = 30,
    faces: np.ndarray | None = None,
) -> tuple[set[int], list[list[int]]]:
    """Split *sel* into edge-connected groups, expand each on its own with
    :func:`expand_tip_to_good_rim`, then trace the border loops of the
    merged selection: ``(selection, loops)``."""
    sel_set: set[int] = set(int(fi) for fi in sel)
    if not sel_set:
        return sel_set, []

    # Split sel into edge-connected sub-groups via BFS over face_adj,
    # restricted to faces inside sel_set.
    groups: list[set[int]] = []
    visited: set[int] = set()
    for seed in sel_set:
        if seed in visited:
            continue
        group: set[int] = set()
        stack = [seed]
        while stack:
            f = stack.pop()
            if f in visited:
                continue
            visited.add(f)
            group.add(f)
            for nf in face_adj.get(f, ()):
                if nf in sel_set and nf not in visited:
                    stack.append(nf)
        groups.append(group)

    # Expand each group independently
    expanded: set[int] = set()
    for grp in groups:
        sub_sel, _ = expand_tip_to_good_rim(
            list(grp),
            edge_to_faces,
            face_adj,
            target_verts=target_verts,
            max_iters=max_iters,
            max_eat_factor=max_eat_factor,
            faces=faces,
        )
        expanded |= sub_sel

    # Re-trace loops on the final merged selection
    loops = border_loops(expanded, edge_to_faces, faces)
    loops = [lp for lp in loops if len(lp) >= 3]
    return expanded, loops


def face_edge_components(
    faces, n_vertices: int | None = None
) -> tuple[np.ndarray, int]:
    """Edge-connected components of the non-degenerate faces: ``(labels,
    main)``, components numbered by their lowest face, ``-2`` for a face
    with a repeated vertex, *main* the largest with ties to the lowest id."""
    faces, _ = checked_faces(faces, n_vertices)
    n = len(faces)

    # Mark degenerate faces (duplicate vertices) so they are skipped
    degen = ~non_degenerate(faces)

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(n):
        if degen[fi]:
            continue
        face = faces[fi]
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    # -1 = unlabeled, -2 = degenerate (permanently skipped)
    labels = np.full(n, -1, dtype=np.int64)
    labels[degen] = -2
    comp_id = 0
    best_id, best_size = 0, 0

    for seed in range(n):
        if labels[seed] >= 0 or labels[seed] == -2:
            continue
        queue = [seed]
        labels[seed] = comp_id
        size = 0
        while queue:
            fi = queue.pop()
            size += 1
            face = faces[fi]
            for i in range(3):
                e = (
                    min(int(face[i]), int(face[(i + 1) % 3])),
                    max(int(face[i]), int(face[(i + 1) % 3])),
                )
                for nb in edge_to_faces[e]:
                    if labels[nb] < 0:
                        labels[nb] = comp_id
                        queue.append(nb)
        if size > best_size:
            best_id, best_size = comp_id, size
        comp_id += 1

    return labels, best_id


def _vertex_ids(ids, n_vertices: int) -> np.ndarray:
    """*ids* as int64, each checked to name one of *n_vertices* vertices."""
    ids = np.asarray(ids, dtype=np.int64).ravel()
    bad = (ids < 0) | (ids >= n_vertices)
    if bad.any():
        raise ValueError(f"vertex {ids[bad][0]} is out of range")
    return ids


def _face_ids(ids, n_faces: int) -> np.ndarray:
    """*ids* as int64, each checked to name one of *n_faces* faces."""
    ids = np.asarray(ids, dtype=np.int64)
    bad = (ids < 0) | (ids >= n_faces)
    if bad.any():
        raise ValueError(f"face index {ids[bad][0]} is out of range")
    return ids


def _face_adjacency_pairs(faces: np.ndarray, fi: np.ndarray) -> np.ndarray:
    """``(m, 2)`` pairs of edge-sharing faces drawn from *fi*.

    Every face on an edge links to the first face on that edge, non-manifold
    edges included, which spans the same connected components as the full
    clique would at a fraction of the pairs.
    """
    fi = np.asarray(fi, dtype=np.intp)
    if len(fi) == 0:
        return np.empty((0, 2), dtype=np.intp)

    gf = np.asarray(faces)[fi]
    e = np.sort(
        np.concatenate([gf[:, [0, 1]], gf[:, [1, 2]], gf[:, [0, 2]]], axis=0),
        axis=1,
    )
    owner = np.concatenate([fi, fi, fi])

    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]

    first = np.ones(len(e), dtype=bool)
    first[1:] = (e[1:] != e[:-1]).any(axis=1)
    root = owner[np.flatnonzero(first)][np.cumsum(first) - 1]

    rest = ~first
    return np.column_stack([root[rest], owner[rest]])


def _label_face_components(faces: np.ndarray, fi: np.ndarray) -> np.ndarray:
    """Connected-component label per entry of *fi*, by edge adjacency."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    fi = np.asarray(fi, dtype=np.intp)
    if len(fi) == 0:
        return np.empty(0, dtype=np.intp)

    local = np.full(int(fi.max()) + 1, -1, dtype=np.intp)
    local[fi] = np.arange(len(fi))
    pairs = _face_adjacency_pairs(faces, fi)
    graph = coo_matrix(
        (
            np.ones(len(pairs), dtype=np.int8),
            (local[pairs[:, 0]], local[pairs[:, 1]]),
        ),
        shape=(len(fi), len(fi)),
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def face_components(faces, face_indices) -> list[np.ndarray]:
    """Edge-connected components of the listed faces: largest first, ties to
    the lower lowest face, each an int64 array of face ids in the order
    listed."""
    faces = np.asarray(faces, dtype=np.int64)
    fi = _face_ids(face_indices, len(faces))
    labels = _label_face_components(faces, fi)
    if len(labels) == 0:
        return []
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    comps = np.split(fi[order], bounds)
    comps.sort(key=lambda c: (-len(c), int(c.min())))
    return comps


def refine_components_with_fusions(
    faces, labels, main: int, fusion
) -> tuple[np.ndarray, int]:
    """Split each face-edge component at its fusion faces (*fusion*, ids past
    the last face ignored) and non-manifold edges: ``(labels, main)``, the
    pieces numbered after the highest old label, fusion faces ``-3``, *main*
    the largest piece (unchanged when nothing is relabelled)."""
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(faces):
        raise ValueError(f"{len(labels)} labels for {len(faces)} faces")
    fusion_faces: set[int] = set(int(f) for f in np.asarray(fusion).ravel())

    # Every real component (skip the degenerate-face sentinel labels).
    comp_face_idx = np.where(labels >= 0)[0]
    comp_set = set(int(i) for i in comp_face_idx)
    fusion_in_comps = fusion_faces & comp_set

    # Edge bookkeeping over the WHOLE mesh:
    # `edge_count` knows the total face uses (used to detect
    # non-manifold edges that must be severed).  `edge_to_faces` maps
    # each edge to its non-fusion faces (the BFS-walkable ones).
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in comp_face_idx:
        ifi = int(fi)
        face = faces[ifi]
        is_fusion = ifi in fusion_faces
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            edge_count[e] += 1
            if not is_fusion:
                edge_to_faces[e].append(ifi)

    # BFS each component.  Walls:
    #   1. Fusion-cluster faces (skipped as seeds and as neighbors)
    #   2. Non-manifold edges (face-count > 2 in the original mesh):
    #      these are the actual fusion structure; severing them is
    #      what splits the glued branches.
    #
    # Severing a non-manifold edge does not over-segment same-branch
    # faces in practice — two faces of one branch that share a
    # non-manifold edge always have other manifold paths between them
    # (otherwise the branch would be 2 faces, well below the < 7 filter).
    new_labels = labels.copy()
    visited: set[int] = set()
    next_id = int(labels.max()) + 1 if len(labels) else 0
    sub_sizes: dict[int, int] = {}
    for seed in comp_face_idx:
        seed = int(seed)
        if seed in visited or seed in fusion_faces:
            continue
        cid = next_id
        next_id += 1
        size = 0
        stack = [seed]
        while stack:
            fi = stack.pop()
            if fi in visited:
                continue
            visited.add(fi)
            new_labels[fi] = cid
            size += 1
            face = faces[fi]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                if edge_count[e] > 2:
                    continue  # non-manifold — sever
                for nb in edge_to_faces.get(e, ()):
                    if nb in visited or nb in fusion_faces:
                        continue
                    stack.append(nb)
        sub_sizes[cid] = size

    # Mark fusion faces with sentinel label -3.
    for fi in fusion_in_comps:
        new_labels[fi] = -3

    new_main = max(sub_sizes, key=sub_sizes.get) if sub_sizes else int(main)
    return new_labels, new_main


#: Faces per chunk of :func:`outward_dots`: each chunk materialises all of its
#: (face, vertex) pairs within the radius, about 170 per face at the default.
_OUTWARD_DOT_CHUNK = 50_000


def outward_dots(
    vertices, faces, centers, normals, radius: float, vert_comp=None
) -> np.ndarray:
    """Per face, the dot of its normal with the direction from the centroid
    of the vertices within *radius* of its centre (of its own component's,
    given *vert_comp*) to that centre; 0 where fewer than four vertices are
    near or the direction vanishes."""
    from scipy.sparse import csr_matrix

    verts = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(verts))
    face_centers = rows3("F", centers)
    face_normals = rows3("F", normals)
    if len(face_centers) != len(faces) or len(face_normals) != len(faces):
        raise ValueError("centers and normals need one row per face")
    if vert_comp is not None:
        vert_comp = np.asarray(vert_comp, dtype=np.int64)
        if len(vert_comp) != len(verts):
            raise ValueError("vert_comp needs one label per vertex")
    n_faces = len(faces)
    dots_out = np.zeros(n_faces, dtype=np.float64)

    def _compute_chunk(vtree, v, face_idx):
        fc = face_centers[face_idx]
        fn = face_normals[face_idx]

        # All (face, vert) pairs within radius — returned as sparse matrix
        sdm = KDTree(fc).sparse_distance_matrix(vtree, radius, output_type="coo_matrix")
        if sdm.nnz == 0:
            return

        # Count neighbors per face
        row = sdm.row
        counts = np.bincount(row, minlength=len(face_idx))

        # Build weight matrix: each entry = 1/count (for mean)
        valid_entries = counts[row] >= 4
        weights = np.zeros(len(row), dtype=np.float64)
        weights[valid_entries] = 1.0 / counts[row[valid_entries]]

        W = csr_matrix(
            (weights, (row, sdm.col)),
            shape=(len(face_idx), len(v)),
            dtype=np.float64,
        )

        # COM = W @ v  (weighted mean of neighbor vertices)
        local_com = W @ v  # (n_faces, 3)

        # Direction
        direction = fc - local_com
        norms = np.linalg.norm(direction, axis=1)
        valid = (counts >= 4) & (norms > 1e-10)

        # Normalize and dot
        direction[valid] /= norms[valid, None]
        dots = row_dot(fn, direction)

        dots_out[face_idx[valid]] = dots[valid]

    def _compute_batch(vtree, v, face_idx):
        # a face's result depends only on its own pairs, so chunking does
        # not change it
        for lo in range(0, len(face_idx), _OUTWARD_DOT_CHUNK):
            _compute_chunk(vtree, v, face_idx[lo : lo + _OUTWARD_DOT_CHUNK])

    if vert_comp is None:
        vtree = KDTree(verts)
        _compute_batch(vtree, verts, np.arange(n_faces))
    else:
        face_comp = vert_comp[faces[:, 0]]
        for ci in np.unique(face_comp):
            ci = int(ci)
            comp_verts_idx = np.where(vert_comp == ci)[0]
            if len(comp_verts_idx) < 4:
                continue
            comp_verts = verts[comp_verts_idx]
            vtree = KDTree(comp_verts)
            comp_face_idx = np.where(face_comp == ci)[0]
            _compute_batch(vtree, comp_verts, comp_face_idx)

    return dots_out


def mesh_stats(
    vertices, faces, centers, normals, radius: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(outward_dots, face_comp, main_ci)``: the face-edge components, and
    each face's outward dot within its own component."""
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    face_comp, main_ci = face_edge_components(faces, len(vertices))

    # Vertex component labels, for the per-component centroid; a vertex on
    # several components takes the last face's
    vert_comp = np.full(len(vertices), -1, dtype=np.int64)
    for fi in range(len(faces)):
        ci = int(face_comp[fi])
        if ci < 0:
            continue
        for v in faces[fi]:
            vert_comp[int(v)] = ci

    dots = outward_dots(vertices, faces, centers, normals, radius, vert_comp=vert_comp)
    return dots, face_comp, main_ci


def filter_small_clusters(faces, face_mask, min_cluster_size: int) -> np.ndarray:
    """*face_mask* without its edge-connected clusters of fewer than
    *min_cluster_size* faces."""
    import igraph as ig

    faces_arr = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool)
    if len(face_mask) != len(faces_arr):
        raise ValueError("face_mask needs one value per face")
    flagged = set(int(fi) for fi in np.where(face_mask)[0])
    if not flagged:
        return face_mask.copy()

    # Build face adjacency restricted to flagged faces
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in flagged:
        f = faces_arr[fi]
        for i in range(3):
            e = (
                min(int(f[i]), int(f[(i + 1) % 3])),
                max(int(f[i]), int(f[(i + 1) % 3])),
            )
            edge_to_faces[e].append(fi)

    int_list = sorted(flagged)
    int_remap = {fi: i for i, fi in enumerate(int_list)}
    edges = set()
    for fi in int_list:
        f = faces_arr[fi]
        for i in range(3):
            e = (
                min(int(f[i]), int(f[(i + 1) % 3])),
                max(int(f[i]), int(f[(i + 1) % 3])),
            )
            for nfi in edge_to_faces[e]:
                if nfi != fi:
                    a, b = int_remap[fi], int_remap[nfi]
                    edges.add((min(a, b), max(a, b)))

    g = ig.Graph(n=len(int_list), edges=list(edges), directed=False)
    keep = set()
    for cl in g.connected_components():
        if len(cl) >= min_cluster_size:
            keep.update(int_list[i] for i in cl)

    filtered = np.zeros(len(faces_arr), dtype=bool)
    filtered[list(keep)] = True
    return filtered


def find_nonmanifold_fusions(
    faces,
    n_vertices: int,
    areas,
    outward_dots,
    grow_rings: int = 20,
    min_branch_size: int = 5,
    sort: bool = False,
) -> list[list[int]]:
    """Fusion clusters, as face lists: around faces that point inward at a
    non-manifold edge or duplicate another, the boundary between the two
    largest manifold pieces a few rings of growth split them into; and every
    vertex whose fan of faces falls apart.  With *sort*, largest first."""
    faces_arr, n_verts = checked_faces(faces, n_vertices)
    areas = np.asarray(areas, dtype=np.float64)
    outward_dots = np.asarray(outward_dots, dtype=np.float64)
    if len(areas) != len(faces_arr) or len(outward_dots) != len(faces_arr):
        raise ValueError("areas and outward_dots need one value per face")
    zero_faces = set(np.where(areas < 1e-6)[0].tolist())

    edge_to_face = edge_to_faces(faces_arr)

    def _get_neighbors(fi: int) -> set[int]:
        f = faces_arr[fi]
        nb: set[int] = set()
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            for nfi in edge_to_face[(min(a, b), max(a, b))]:
                if nfi != fi:
                    nb.add(nfi)
        return nb

    # Signal 1: negative-dot faces at non-manifold edges
    nm_neg_faces: set[int] = set()
    for e, fs in edge_to_face.items():
        if len(fs) > 2:
            for fi in fs:
                if outward_dots[fi] < 0:
                    nm_neg_faces.add(fi)

    # Signal 2: exact duplicate faces with >3 neighbors
    face_tuples = [tuple(sorted(int(v) for v in f)) for f in faces_arr]
    dupe_set = {ft for ft, c in Counter(face_tuples).items() if c > 1}
    dupe_faces: set[int] = set()
    for fi, ft in enumerate(face_tuples):
        if ft in dupe_set and fi not in zero_faces:
            nb: set[int] = set()
            f = faces_arr[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                for nfi in edge_to_face[(min(a, b), max(a, b))]:
                    if nfi != fi:
                        nb.add(nfi)
            if len(nb) > 3:
                dupe_faces.add(fi)

    seed_faces = nm_neg_faces | dupe_faces

    # Signal 3: fan vertices
    _good = non_degenerate(faces_arr)
    vert_to_face: list[list[int]] = [[] for _ in range(n_verts)]
    for fi, f in enumerate(faces_arr):
        if not _good[fi] or fi in zero_faces:
            continue
        for v in f:
            vert_to_face[int(v)].append(fi)

    fan_vertex_clusters: list[list[int]] = []
    for vid in range(n_verts):
        fan = vert_to_face[vid]
        if len(fan) < 2:
            continue
        fan_set = set(fan)
        fan_adj: dict[int, set[int]] = defaultdict(set)
        for fi in fan:
            f = faces_arr[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nfi in edge_to_face[e]:
                    if nfi in fan_set and nfi != fi:
                        fan_adj[fi].add(nfi)
        vis: set[int] = set()
        q = deque([fan[0]])
        while q:
            curr = q.popleft()
            if curr in vis:
                continue
            vis.add(curr)
            for nfi in fan_adj.get(curr, set()):
                if nfi not in vis:
                    q.append(nfi)
        if len(vis) < len(fan_set):
            fan_vertex_clusters.append(sorted(fan))

    if not seed_faces and not fan_vertex_clusters:
        return []

    # Cluster seeds
    seed_adj: dict[int, set[int]] = defaultdict(set)
    for e, fs in edge_to_face.items():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                if fs[i] in seed_faces and fs[j] in seed_faces:
                    seed_adj[fs[i]].add(fs[j])
                    seed_adj[fs[j]].add(fs[i])

    visited: set[int] = set()
    seed_clusters: list[list[int]] = []
    for fi in sorted(seed_faces):
        if fi in visited:
            continue
        cluster: list[int] = []
        queue = deque([fi])
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)
            for nfi in seed_adj.get(curr, set()):
                if nfi not in visited:
                    queue.append(nfi)
        seed_clusters.append(cluster)

    # Grow region & split per cluster
    def _manifold_components(region: set[int]) -> list[set[int]]:
        m_adj: dict[int, set[int]] = defaultdict(set)
        for fi in region:
            f = faces_arr[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                if len(edge_to_face[e]) == 2:
                    for nfi in edge_to_face[e]:
                        if nfi != fi and nfi in region:
                            m_adj[fi].add(nfi)
        vis: set[int] = set()
        comps: list[set[int]] = []
        for fi in sorted(region):
            if fi in vis:
                continue
            comp: set[int] = set()
            q = deque([fi])
            while q:
                curr = q.popleft()
                if curr in vis:
                    continue
                vis.add(curr)
                comp.add(curr)
                for nfi in m_adj[curr]:
                    if nfi not in vis:
                        q.append(nfi)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        return comps

    result_clusters: list[list[int]] = []
    for seed_cluster in seed_clusters:
        region = set(seed_cluster)
        for ring in range(1, grow_rings + 1):
            boundary: set[int] = set()
            for fi in region:
                boundary.update(_get_neighbors(fi))
            region = (region | boundary) - zero_faces
            if ring < 3:
                continue
            comps = _manifold_components(region)
            big = [c for c in comps if len(c) >= min_branch_size]
            if len(big) >= 2:
                branch0 = big[0]
                branch1 = big[1]
                fusion_boundary: set[int] = set()
                for fi in branch0:
                    for nfi in _get_neighbors(fi):
                        if nfi in branch1:
                            fusion_boundary.add(fi)
                            fusion_boundary.add(nfi)
                result_clusters.append(sorted(fusion_boundary))
                break

    result_clusters.extend(fan_vertex_clusters)
    if sort:
        result_clusters.sort(key=len, reverse=True)
    return result_clusters


def split_fan_vertices(
    faces, n_vertices: int, candidates=None
) -> tuple[np.ndarray, np.ndarray]:
    """Give each edge-connected piece of a vertex's fan of faces, but the
    first, a copy of the vertex: ``(faces, sources)``, *sources* the vertex
    each new vertex (numbered from *n_vertices* on) copies — empty when no
    fan of the *candidates* (default all vertices, in order) falls apart.

    Splitting a vertex is a pure relabel that keeps face-edge adjacency, so
    every fan is read on the original vertex ids.
    """
    orig_faces, n_verts = checked_faces(faces, n_vertices)
    if candidates is not None:
        check = _vertex_ids(candidates, n_verts).tolist()
    else:
        check = range(n_verts)
    _good = non_degenerate(orig_faces)
    vert_to_face: list[list[int]] = [[] for _ in range(n_verts)]
    for fi, f in enumerate(orig_faces):
        if _good[fi]:
            for v in f:
                vert_to_face[int(v)].append(fi)
    edge_to_face = edge_to_faces(orig_faces)

    out = orig_faces.copy()
    sources: list[int] = []
    for vid in check:
        fan = vert_to_face[vid]
        if len(fan) < 2:
            continue

        fan_set = set(fan)
        fan_adj: dict[int, set[int]] = defaultdict(set)
        for fi in fan:
            f = orig_faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nfi in edge_to_face[e]:
                    if nfi in fan_set and nfi != fi:
                        fan_adj[fi].add(nfi)

        # connected pieces of the fan, in the order their first face is met
        vis: set[int] = set()
        comps: list[set[int]] = []
        for fi in fan:
            if fi in vis:
                continue
            comp: set[int] = set()
            q = deque([fi])
            while q:
                curr = q.popleft()
                if curr in vis:
                    continue
                vis.add(curr)
                comp.add(curr)
                for nfi in fan_adj[curr]:
                    if nfi not in vis:
                        q.append(nfi)
            comps.append(comp)

        # the first piece keeps the vertex, each other one gets a copy
        for comp_faces in comps[1:]:
            new_vid = n_verts + len(sources)
            sources.append(vid)
            for fi in comp_faces:
                out[fi][out[fi] == vid] = new_vid

    return out, np.asarray(sources, dtype=np.int64)


def remove_fusions(faces, n_vertices: int, fusions) -> tuple[np.ndarray, np.ndarray]:
    """Degenerate the *fusions*' faces to ``[0, 0, 0]``, then split the fans
    of the vertices they touched: :func:`split_fan_vertices`' result."""
    faces, n_verts = checked_faces(faces, n_vertices)
    clusters = [_face_ids(c, len(faces)) for c in fusions]
    out = faces.copy()
    touched: set[int] = set()
    for c in clusters:
        for f in c.tolist():
            touched.update(faces[f].tolist())
            out[f] = 0
    return split_fan_vertices(out, n_verts, np.asarray(sorted(touched), np.int64))


def find_disconnected(
    faces,
    vertices,
    centers,
    normals,
    median_edge: float,
    labels,
    main: int,
    organelle=None,
    soma_center=None,
    soma_w=None,
) -> tuple[list[list[int]], int, int, int, int]:
    """The components other than *main* worth bridging back: ``(components,
    n_non_main, n_soma, n_organelle, n_enclosed)``, the components as face
    lists, largest first, and how many were set aside — every component of
    seven faces or more is kept unless it is all organelle, its vertex
    centroid lies in the soma (*soma_w* is ``R / axes``, which maps the soma
    to the unit sphere), or it sits against the inside of the main surface.
    """
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    centers = rows3("F", centers)
    normals = rows3("F", normals)
    labels = np.asarray(labels, dtype=np.int64)
    if len(centers) != len(faces) or len(normals) != len(faces):
        raise ValueError("centers, normals and labels need one row per face")
    if len(labels) != len(faces):
        raise ValueError("centers, normals and labels need one row per face")
    if organelle is not None:
        organelle = np.asarray(organelle, dtype=bool)
        if len(organelle) != len(faces):
            raise ValueError("organelle needs one value per face")
    if (soma_center is None) != (soma_w is None):
        raise ValueError("give both soma_center and soma_w, or neither")
    if soma_center is not None:
        soma_center = np.asarray(soma_center, dtype=np.float64)
        soma_w = rows3("3", soma_w)
        if soma_center.shape != (3,) or len(soma_w) != 3:
            raise ValueError("soma_center is (3,), soma_w (3, 3)")

    # Build KD-tree of main-component face centroids + normals
    # for inside/outside classification
    main_face_idx = np.where(labels == main)[0]
    main_centroids = centers[main_face_idx]
    main_normals = normals[main_face_idx]
    main_tree = KDTree(main_centroids)

    # Collect non-main components (skip degenerate faces with label -2)
    comp_faces: dict[int, list[int]] = {}
    for fi in range(len(faces)):
        cid = int(labels[fi])
        if cid == main or cid < 0:
            continue
        comp_faces.setdefault(cid, []).append(fi)

    n_total_non_main = len(comp_faces)
    components = []
    n_soma_excluded = 0
    n_organelle_excluded = 0
    n_enclosed_excluded = 0
    for cid, fis in comp_faces.items():
        # Need at least 7 faces: 3 for each tip + 1 body face to
        # bridge back to two other parts
        if len(fis) < 7:
            continue

        # Skip components that are entirely organelle
        if organelle is not None and organelle[np.asarray(fis)].all():
            n_organelle_excluded += 1
            continue

        coords = vertices[np.unique(faces[fis])]
        centroid = coords.mean(axis=0)

        # Exclude components whose centroid falls inside the soma ellipsoid,
        # in the soma's body coordinates, as `Soma.contains` tests it
        if soma_center is not None:
            xi = (centroid.reshape(1, -1) - soma_center) @ soma_w
            if (xi**2).sum(axis=-1)[0] <= 1.0:
                n_soma_excluded += 1
                continue

        # Exclude components enclosed by the main mesh.
        # For each component vertex, find the nearest main mesh face and
        # check which side of that face the vertex is on: if the vector
        # from the main face centroid to the vertex is opposite the main
        # face normal, the vertex is on the interior side.
        #
        # The nearest-face-normal vote is only reliable when the
        # component sits close to main: a true enclosed organelle is
        # right against the inner surface. For components far from main,
        # the nearest face may lie on a curved/concave region whose
        # normal points the wrong way, producing false "inside" votes.
        # Skip the test entirely when the component is clearly external.
        _, nn_idx = nearest(main_tree, coords)
        vecs = coords - main_centroids[nn_idx]
        dists = np.linalg.norm(vecs, axis=1)
        if np.median(dists) <= 5.0 * median_edge:
            dots = row_dot(vecs, main_normals[nn_idx])
            # Component is enclosed if the majority of vertices are on
            # the inward side (dot < 0)
            if (dots < 0).sum() > len(dots) / 2:
                n_enclosed_excluded += 1
                continue

        components.append(fis)

    # Sort largest-first
    components.sort(key=lambda fis: -len(fis))
    return (
        components,
        n_total_non_main,
        n_soma_excluded,
        n_organelle_excluded,
        n_enclosed_excluded,
    )


def rim_enclosed_area(boundary_edges, vertices, svd_rows) -> float:
    """Total planar area enclosed by the closed loops among *boundary_edges*:
    each loop projected onto its best-fit plane (``svd_rows(x,
    full_matrices)`` is numpy's SVD's ``Vt``), its area by the shoelace
    formula; a piece that is not one closed loop counts nothing."""
    if not boundary_edges:
        return 0.0

    vert_to_be: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(boundary_edges):
        vert_to_be[e[0]].append(i)
        vert_to_be[e[1]].append(i)

    visited: set[int] = set()
    total_area = 0.0

    for start in range(len(boundary_edges)):
        if start in visited:
            continue
        comp: list[int] = []
        queue = deque([start])
        while queue:
            ei = queue.popleft()
            if ei in visited:
                continue
            visited.add(ei)
            comp.append(ei)
            for v in boundary_edges[ei]:
                for nei in vert_to_be[v]:
                    if nei not in visited:
                        queue.append(nei)

        # Check closed ring
        vcount: dict[int, int] = defaultdict(int)
        for ei in comp:
            for v in boundary_edges[ei]:
                vcount[v] += 1
        if not all(c == 2 for c in vcount.values()):
            continue

        # Order vertices into a loop
        edge_list = [boundary_edges[ei] for ei in comp]
        adj_v: dict[int, list[int]] = defaultdict(list)
        for a, b in edge_list:
            adj_v[a].append(b)
            adj_v[b].append(a)

        ordered = [edge_list[0][0]]
        while len(ordered) < len(edge_list):
            curr = ordered[-1]
            prev = ordered[-2] if len(ordered) > 1 else -1
            for nb in adj_v[curr]:
                if nb != prev:
                    ordered.append(nb)
                    break
            else:
                break

        if len(ordered) < 3:
            continue

        # Project onto best-fit plane and compute area
        pts = vertices[ordered]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        vh = svd_rows(centered, True)
        u_ax, v_ax = vh[0], vh[1]
        pts_2d = np.column_stack([centered @ u_ax, centered @ v_ax])
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        area = 0.5 * abs(
            np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]
        )
        total_area += area

    return total_area


def pocket_mouths(
    faces,
    vertices,
    areas,
    outward_dots,
    main_face_mask,
    edge_to_face: dict[tuple[int, int], list[int]],
    adj: dict[int, set[int]],
    svd_rows,
    hull_area,
    *,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    non_main_mask=None,
) -> list[list[tuple[int, int]]]:
    """Mouth edges — the boundaries of negative-dot face clusters on the
    faces of *main_face_mask* — one list per pocket.

    A valid pocket must satisfy:

    1. Multiple boundary loops (not a flat patch with a single outline).
    2. Fold ratio > *min_fold_ratio*: the pocket surface area must be
       much larger than the mouth's enclosed planar area, indicating the
       surface folds inward through a small opening.

    *hull_area(pts)* is the convex-hull area of points on their best-fit
    plane, the opening when no rim loop closes.
    """
    faces_arr = np.asarray(faces)
    area_arr = np.asarray(areas)
    verts_arr = np.asarray(vertices)

    # Connected components of negative-dot faces on main component
    neg_idx = set(np.where((outward_dots < 0) & main_face_mask)[0].tolist())
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for fi in neg_idx:
        if fi in visited:
            continue
        cluster: list[int] = []
        queue = deque([fi])
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)
            for nfi in adj.get(curr, set()):
                if nfi in neg_idx and nfi not in visited:
                    queue.append(nfi)
        clusters.append(cluster)

    # For each cluster, collect boundary edges and count boundary loops.
    # A real pocket has multiple boundary loops (multiple openings).
    # A flat concave patch has exactly one boundary loop — not a pocket.
    mouths: list[list[tuple[int, int]]] = []
    for cluster in clusters:
        if len(cluster) < min_pocket_size:
            continue
        cset = set(cluster)
        boundary_edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        for fi in cluster:
            f = faces_arr[fi]
            for i in range(3):
                e = (
                    min(int(f[i]), int(f[(i + 1) % 3])),
                    max(int(f[i]), int(f[(i + 1) % 3])),
                )
                if e in seen_edges:
                    continue
                seen_edges.add(e)
                faces_on_edge = edge_to_face[e]
                if len(faces_on_edge) == 2:
                    other = (
                        faces_on_edge[1]
                        if faces_on_edge[0] in cset
                        else faces_on_edge[0]
                    )
                    if other not in cset:
                        boundary_edges.append(e)
        if not boundary_edges:
            continue

        # Fold ratio: pocket surface area / mouth enclosed planar area.
        # A real pocket folds inward through a small opening (high ratio).
        # A flat patch has ratio near 1.
        pocket_area = float(area_arr[np.sort(cluster)].sum())
        opening_area = rim_enclosed_area(boundary_edges, verts_arr, svd_rows)
        if opening_area <= 0:
            # Boundary loops with non-manifold junctions (vertex
            # shared by 4+ boundary edges) get discarded by
            # rim_enclosed_area.  Fall back to the convex hull of
            # boundary vertices: hull_area >= true opening area, so
            # pocket_area / hull_area is a *lower bound* on the true
            # fold ratio — this can only recover missed mouths, never
            # reject valid ones.
            # Only apply on the main component — small disconnected
            # components have degenerate boundary topology that
            # produces false positives.
            on_non_main = non_main_mask is not None and non_main_mask[cluster[0]]
            if not on_non_main:
                bverts = sorted({v for e in boundary_edges for v in e})
                if len(bverts) >= 3:
                    opening_area = hull_area(verts_arr[bverts])
            if opening_area <= 0:
                continue
        fold_ratio = pocket_area / opening_area
        if fold_ratio < min_fold_ratio:
            continue

        mouths.append(boundary_edges)

    return mouths


def pocket_organelles(
    faces,
    vertices,
    areas,
    outward_dots,
    main_face_mask,
    edge_to_face: dict[tuple[int, int], list[int]],
    adj: dict[int, set[int]],
    svd_rows,
    hull_area,
    *,
    grow_threshold: float = 0.1,
    bridge_threshold: float = 0.3,
    max_hole_size: int = 500,
    hole_enclosure_ratio: float = 0.5,
    min_pocket_size: int = 5,
    min_fold_ratio: float = 3.0,
    min_cluster_size: int = 5,
    non_main_mask=None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Pocket organelles — membrane folds connected to the neuron surface —
    grown from the mouths of :func:`pocket_mouths`: ``(pocket, counts)``,
    *counts* the mouths, the faces the flood fill, the bridging and the hole
    filling claimed, and the components rejected as shallow dips.

    1. Seed from the **negative-dot faces** of each mouth's pocket cluster.
    2. Flood-fill from seeds, stopping at mouth edges and faces with
       ``outward_dot > grow_threshold``.
    3. Bridging: flood-fill from pocket boundary using the relaxed
       ``bridge_threshold`` to cross narrow positive-dot barriers.
    4. Hole filling: small non-pocket clusters mostly enclosed by pocket
       faces are filled in.

    Only regions behind a mouth get detected — curved surfaces without a
    mouth are correctly excluded.
    """
    n_faces = len(faces)
    counts = {
        "n_mouths": 0,
        "n_initial": 0,
        "n_bridged": 0,
        "n_holes": 0,
        "n_rejected": 0,
    }
    mouths = pocket_mouths(
        faces,
        vertices,
        areas,
        outward_dots,
        main_face_mask,
        edge_to_face,
        adj,
        svd_rows,
        hull_area,
        min_pocket_size=min_pocket_size,
        min_fold_ratio=min_fold_ratio,
        non_main_mask=non_main_mask,
    )
    if not mouths:
        return np.zeros(n_faces, dtype=bool), counts

    # Collect all mouth edges and seed faces
    # Seeds = negative-dot faces adjacent to mouth edges (inward side)
    mouth_edge_set: set[tuple[int, int]] = set()
    for mouth in mouths:
        mouth_edge_set.update(mouth)

    # Seeds: negative-dot faces touching mouth edges
    seeds: set[int] = set()
    for e in mouth_edge_set:
        for fi in edge_to_face.get(e, []):
            if main_face_mask[fi] and outward_dots[fi] < 0:
                seeds.add(fi)

    # Mouth faces: positive-dot faces touching mouth edges (block flood-fill)
    mouth_faces: set[int] = set()
    for e in mouth_edge_set:
        for fi in edge_to_face.get(e, []):
            if fi not in seeds:
                mouth_faces.add(fi)

    counts["n_mouths"] = len(mouths)

    # Flood-fill from seeds, blocked by mouth faces and grow_threshold
    pocket = np.zeros(n_faces, dtype=bool)
    visited = np.zeros(n_faces, dtype=bool)
    queue = deque(seeds)

    while queue:
        fi = queue.popleft()
        if visited[fi]:
            continue
        visited[fi] = True
        if fi in mouth_faces:
            continue
        if not main_face_mask[fi]:
            continue
        if outward_dots[fi] > grow_threshold:
            continue
        pocket[fi] = True
        for nfi in adj.get(fi, set()):
            if not visited[nfi]:
                queue.append(nfi)

    initial_count = int(pocket.sum())
    counts["n_initial"] = initial_count

    # Phase 1 — Bridging: flood-fill from pocket boundary using the
    # relaxed bridge_threshold to cross positive-dot barriers.
    bridge_queue = deque()
    for fi in np.where(pocket)[0]:
        for nfi in adj.get(int(fi), set()):
            if not pocket[nfi]:
                bridge_queue.append(nfi)

    while bridge_queue:
        fi = bridge_queue.popleft()
        if pocket[fi]:
            continue
        if fi in mouth_faces or not main_face_mask[fi]:
            continue
        if outward_dots[fi] > bridge_threshold:
            continue
        pocket[fi] = True
        for nfi in adj.get(fi, set()):
            if not pocket[nfi]:
                bridge_queue.append(nfi)

    counts["n_bridged"] = int(pocket.sum()) - initial_count

    # Phase 2 — Hole filling: small non-pocket clusters mostly enclosed
    # by pocket faces are filled in.  Sub-folds inside pockets can have
    # positive outward_dot, so we use enclosure ratio only.
    non_pocket_main = main_face_mask & ~pocket
    non_pocket_idx = set(np.where(non_pocket_main)[0].tolist())
    np_visited: set[int] = set()
    hole_count = 0

    for fi in non_pocket_idx:
        if fi in np_visited:
            continue
        cluster: list[int] = []
        n_pocket_boundary = 0
        n_total_boundary = 0
        adj_pocket_faces: set[int] = set()
        bfs_queue = deque([fi])
        while bfs_queue:
            curr = bfs_queue.popleft()
            if curr in np_visited:
                continue
            np_visited.add(curr)
            cluster.append(curr)
            for nfi in adj.get(curr, set()):
                if nfi in non_pocket_idx and nfi not in np_visited:
                    bfs_queue.append(nfi)
                elif nfi not in non_pocket_idx:
                    n_total_boundary += 1
                    if pocket[nfi]:
                        n_pocket_boundary += 1
                        adj_pocket_faces.add(nfi)
        if len(cluster) == 0:
            continue
        enclosure = n_pocket_boundary / n_total_boundary if n_total_boundary else 0
        # Small holes: fill if enclosure meets threshold.
        # Large holes: only fill if fully entrapped (enclosure ~1.0)
        #   and not the main mesh body (< 5% of faces).
        # On non-main structural components, a hole can't be bigger
        # than the pocket membrane enclosing it — prevents swallowing
        # entire small disconnected components.
        is_small = len(cluster) <= max_hole_size
        is_entrapped = enclosure >= 0.99 and len(cluster) < n_faces * 0.05
        if (
            non_main_mask is not None
            and non_main_mask[cluster[0]]
            and len(cluster) > len(adj_pocket_faces)
        ):
            continue
        if (is_small and enclosure >= hole_enclosure_ratio) or is_entrapped:
            for c in cluster:
                pocket[c] = True
            hole_count += len(cluster)
    counts["n_holes"] = hole_count

    # Cluster filter to remove noise
    pocket = filter_small_clusters(faces, pocket, min_cluster_size)

    # Post-validate: reject pocket components that are just surface dips.
    # A real pocket has much more surface area than its adjacent non-pocket
    # faces (the "cap").  A shallow dip has pocket_area ≈ cap_area.
    area_arr = np.asarray(areas)
    pocket_idx = set(np.where(pocket)[0].tolist())
    pocket_visited: set[int] = set()
    n_rejected = 0
    for start in list(pocket_idx):
        if start in pocket_visited:
            continue
        comp: list[int] = []
        pq = deque([start])
        while pq:
            fi = pq.popleft()
            if fi in pocket_visited:
                continue
            pocket_visited.add(fi)
            comp.append(fi)
            for nfi in adj.get(fi, set()):
                if nfi in pocket_idx and nfi not in pocket_visited:
                    pq.append(nfi)

        comp_set = set(comp)
        cap_faces: set[int] = set()
        for fi in comp:
            for nfi in adj.get(fi, set()):
                if nfi not in comp_set:
                    cap_faces.add(nfi)

        pocket_area = float(area_arr[np.sort(comp)].sum())
        cap_area = float(area_arr[sorted(cap_faces)].sum()) if cap_faces else 0.0
        if cap_area <= 0 or pocket_area / cap_area < min_fold_ratio:
            for fi in comp:
                pocket[fi] = False
            n_rejected += 1
    counts["n_rejected"] = n_rejected

    return pocket, counts


def isolated_organelles(
    faces, vertices, centers, normals, outward_dots, face_comp, main: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Vertex-disconnected internal fragments — organelle membranes that form
    separate components enclosed within the neuron body: ``(isolated,
    counts)``, *counts* the fragments marked and their faces, the components
    kept, and how many were judged by their outward dots and how many, all
    dots zero, by the side of the main surface they lie on."""
    faces = np.asarray(faces)
    vertices = np.asarray(vertices)
    face_comp = np.asarray(face_comp)
    n_comps = face_comp.max() + 1
    isolated = np.zeros(len(faces), dtype=bool)
    n_internal_frags = 0
    n_internal_frag_faces = 0
    n_kept_frags = 0
    n_via_dots = 0
    n_via_geo = 0

    # Classify each non-main component using outward_dots.
    # Organelle membranes have predominantly negative outward_dots
    # (faces point inward); structural components have positive.
    # For tiny components where outward_dots are all zero (too few
    # neighbours for valid local COM), fall back to the geometric
    # inside/outside test against the main component.
    main_face_idx = np.where(face_comp == main)[0]
    main_centroids = np.asarray(centers)[main_face_idx]
    main_normals = np.asarray(normals)[main_face_idx]
    main_tree = None  # built lazily for fallback

    for ci in range(n_comps):
        if ci == main:
            continue
        comp_face_idx = np.where(face_comp == ci)[0]
        if len(comp_face_idx) == 0:
            continue

        comp_dots = outward_dots[comp_face_idx]
        n_valid = np.count_nonzero(comp_dots)

        if n_valid > 0:
            # Use outward_dots: majority negative → isolated organelle
            is_internal = (comp_dots < 0).sum() > n_valid / 2
            n_via_dots += 1
        else:
            # Fallback: geometric test against main surface
            if main_tree is None:
                main_tree = KDTree(main_centroids)
            coords = vertices[np.unique(faces[comp_face_idx])]
            _, nn_idx = nearest(main_tree, coords)
            vecs = coords - main_centroids[nn_idx]
            dots = row_dot(vecs, main_normals[nn_idx])
            is_internal = (dots < 0).sum() > len(dots) / 2
            n_via_geo += 1

        if is_internal:
            isolated[comp_face_idx] = True
            n_internal_frags += 1
            n_internal_frag_faces += len(comp_face_idx)
        else:
            n_kept_frags += 1

    counts = {
        "n_internal": n_internal_frags,
        "n_internal_faces": n_internal_frag_faces,
        "n_kept": n_kept_frags,
        "n_via_dots": n_via_dots,
        "n_via_geometry": n_via_geo,
    }
    return isolated, counts


def enclosure_pass(
    adjacency, np_mask, pocket, min_cluster_size: int
) -> tuple[np.ndarray, int]:
    """Claim for *pocket* every organelle interior joined to the surface
    through a narrow neck: each subtree of *np_mask*'s faces hanging off an
    articulation face, whose boundary is at least 90% pocket and whose size
    is between *min_cluster_size* and the pocket's.  A single-pass Tarjan
    over *np_mask*'s faces (CSR from trimesh's manifold face *adjacency*)
    tracks each subtree's size and boundary inline: ``(pocket with the
    claimed faces, how many were added)``."""
    _np_mask = np.asarray(np_mask, dtype=bool)
    pocket = np.array(pocket, dtype=bool)
    if len(_np_mask) != len(pocket):
        raise ValueError("np_mask and pocket differ in length")
    _fa = np.asarray(adjacency, dtype=np.int64)
    if _fa.ndim != 2 or _fa.shape[1] != 2:
        raise ValueError("adjacency must be (m, 2)")
    nF = len(_np_mask)
    if _fa.size and (_fa.min() < 0 or _fa.max() >= nF):
        raise ValueError("adjacency names a face out of range")
    _np_arr = np.where(_np_mask)[0]
    enclosed_count = 0
    if _np_arr.size == 0:
        return pocket, enclosed_count
    _both = _np_mask[_fa[:, 0]] & _np_mask[_fa[:, 1]]
    _pairs = _fa[_both]

    # Per-face pocket boundary: edges crossing np ↔ non-np
    _cross_a = _np_mask[_fa[:, 0]] & ~_np_mask[_fa[:, 1]]
    _cross_b = _np_mask[_fa[:, 1]] & ~_np_mask[_fa[:, 0]]
    _pbnd = np.zeros(nF, dtype=np.int32)
    _tbnd = np.zeros(nF, dtype=np.int32)
    if _cross_a.any():
        np.add.at(_tbnd, _fa[_cross_a, 0], 1)
        _pocket_a = _cross_a & pocket[_fa[:, 1]]
        if _pocket_a.any():
            np.add.at(_pbnd, _fa[_pocket_a, 0], 1)
    if _cross_b.any():
        np.add.at(_tbnd, _fa[_cross_b, 1], 1)
        _pocket_b = _cross_b & pocket[_fa[:, 0]]
        if _pocket_b.any():
            np.add.at(_pbnd, _fa[_pocket_b, 1], 1)

    # Build CSR adjacency for non-pocket faces
    _edges = np.empty((len(_pairs) * 2, 2), dtype=np.int64)
    _edges[: len(_pairs)] = _pairs
    _edges[len(_pairs) :, 0] = _pairs[:, 1]
    _edges[len(_pairs) :, 1] = _pairs[:, 0]
    _ord = np.argsort(_edges[:, 0], kind="stable")
    _dst = _edges[_ord, 1]
    _usrc, _ucnt = np.unique(_edges[_ord, 0], return_counts=True)
    _off = np.zeros(nF + 1, dtype=np.int64)
    _off[_usrc + 1] = _ucnt
    np.cumsum(_off, out=_off)

    # Tarjan's with subtree accumulation
    _disc = np.full(nF, -1, dtype=np.int32)
    _low = np.full(nF, -1, dtype=np.int32)
    _sub_sz = np.ones(nF, dtype=np.int32)
    _sub_pb = _pbnd.copy()
    _sub_tb = _tbnd.copy()
    _tmr = 0
    _to_claim: list[tuple[int, int]] = []

    for _root in _np_arr:
        if _disc[_root] >= 0:
            continue
        _disc[_root] = _low[_root] = _tmr
        _tmr += 1
        # stack: (node, parent, edge_cursor)
        _stk: list[tuple[int, int, int]] = [(int(_root), -1, int(_off[_root]))]
        _ccnt = np.zeros(nF, dtype=np.int8)
        while _stk:
            u, pu, ei = _stk[-1]
            end_u = int(_off[u + 1])
            if ei < end_u:
                _stk[-1] = (u, pu, ei + 1)
                v = int(_dst[ei])
                if _disc[v] < 0:
                    _disc[v] = _low[v] = _tmr
                    _tmr += 1
                    _stk.append((v, u, int(_off[v])))
                elif v != pu:
                    dv = _disc[v]
                    if _low[u] > dv:
                        _low[u] = dv
            else:
                _stk.pop()
                if not _stk:
                    break
                pu2 = _stk[-1][0]
                lu = _low[u]
                if _low[pu2] > lu:
                    _low[pu2] = lu
                _sub_sz[pu2] += _sub_sz[u]
                _sub_pb[pu2] += _sub_pb[u]
                _sub_tb[pu2] += _sub_tb[u]
                _ccnt[pu2] += 1
                is_root = _stk[-1][1] == -1
                if lu >= _disc[pu2] and (not is_root or _ccnt[pu2] > 1):
                    sz = int(_sub_sz[u])
                    tb = int(_sub_tb[u])
                    pb = int(_sub_pb[u])
                    _pocket_n = int(pocket.sum())
                    if (
                        tb > 0
                        and pb >= tb * 0.9
                        and sz >= min_cluster_size
                        and sz <= _pocket_n
                    ):
                        _to_claim.append((int(u), int(_disc[u])))

    # Collect subtree faces for each claim via BFS
    for _u, _d_u in _to_claim:
        _q = deque([_u])
        _vis: set[int] = set()
        while _q:
            c = _q.popleft()
            if c in _vis:
                continue
            _vis.add(c)
            if not pocket[c]:
                pocket[c] = True
                enclosed_count += 1
            s = int(_off[c])
            e = int(_off[c + 1])
            for i in range(s, e):
                nb = int(_dst[i])
                if nb not in _vis and _disc[nb] >= _d_u:
                    _q.append(nb)
    return pocket, enclosed_count


def find_organelles(
    vertices,
    faces,
    centers,
    normals,
    areas,
    outward_dots,
    face_comp,
    main: int,
    adjacency,
    svd_rows,
    hull_area,
    min_cluster_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Organelle membranes, given the mesh stats: ``(pocket, isolated,
    report)``.  Isolated fragments first; then pockets on every structural
    component (one with at least *min_cluster_size* faces not isolated);
    then the organelle interiors behind a narrow neck.  *report* holds the
    counts each step found."""
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    centers = rows3("F", centers)
    normals = rows3("F", normals)
    areas = np.asarray(areas, dtype=np.float64)
    outward_dots = np.asarray(outward_dots, dtype=np.float64)
    face_comp = np.asarray(face_comp, dtype=np.int64)
    nf = len(faces)
    if any(len(a) != nf for a in (centers, normals, areas, outward_dots, face_comp)):
        raise ValueError("per-face arrays need one row per face")
    adjacency = np.asarray(adjacency, dtype=np.int64)
    if adjacency.ndim != 2 or adjacency.shape[1] != 2:
        raise ValueError("adjacency must be (m, 2)")
    if adjacency.size and (adjacency.min() < 0 or adjacency.max() >= nf):
        raise ValueError("adjacency names a face out of range")

    # ── 1. Isolated organelles (small internal components) ──────────
    isolated, iso_counts = isolated_organelles(
        faces, vertices, centers, normals, outward_dots, face_comp, main
    )

    # ── 2. Structural components (non-isolated): pockets are found on
    #       all of them in one pass (the outward dots are already per
    #       component).
    isolated_faces_set = set(np.where(isolated)[0].tolist())
    structural_comps = []
    for ci in np.unique(face_comp):
        comp_face_idx = np.where(face_comp == ci)[0]
        non_iso_count = sum(1 for fi in comp_face_idx if fi not in isolated_faces_set)
        if non_iso_count >= min_cluster_size:
            structural_comps.append(int(ci))

    structural_mask = np.zeros(nf, dtype=bool)
    for ci in structural_comps:
        structural_mask[face_comp == ci] = True
    non_main_mask = np.zeros(nf, dtype=bool)
    for ci in structural_comps:
        if ci != main:
            non_main_mask[face_comp == ci] = True

    # ── 3. Pocket detection on all structural components ────────────
    edge_to_face = edge_to_faces(faces)
    adj = face_adjacency(edge_to_face)
    pocket, pocket_counts = pocket_organelles(
        faces,
        vertices,
        areas,
        outward_dots,
        structural_mask,
        edge_to_face,
        adj,
        svd_rows,
        hull_area,
        min_cluster_size=min_cluster_size,
        non_main_mask=non_main_mask,
    )

    # ── 4. Enclosure pass — organelle interiors connected to the surface
    #       through a narrow neck (articulation face).
    np_mask = structural_mask & ~pocket & ~isolated
    pocket, enclosed_count = enclosure_pass(
        adjacency, np_mask, pocket, min_cluster_size
    )

    report = {
        "n_structural": len(structural_comps),
        "n_enclosed": enclosed_count,
        **iso_counts,
        **pocket_counts,
    }
    return pocket, isolated, report


def find_gaps(
    faces,
    vertices,
    centers,
    labels,
    labels_pre_fusion,
    main: int,
    disc,
    fusion_faces,
    tip_rings: int,
    fusion_clear: float,
    fusion_fallback: float,
    max_bridge_ratio: float,
    bridge_floor: float,
    bridge_abs_cap: float,
    kiss_r: float,
    kiss_penalty: float,
    relay_min_faces: int,
    repair_stranded: bool,
    svd_rows,
) -> tuple[list[tuple[list[int], list[int], float, int, int]], int, int]:
    """The bridges that join the disconnected components *disc* back to
    *main*: ``(gaps, n_capped, n_repaired)``, each gap ``(tip faces a, tip
    faces b, distance, disc index a, disc index b)`` (-1 for main), nearest
    first; the thresholds come resolved, in mesh units.

    Every pair of components gets its closest contact (vertices near a
    fusion face set aside, unless that leaves nothing closer than
    *fusion_fallback*); a spanning tree from *main* over those contacts,
    a contact where both surfaces continue past it (a kiss) penalised,
    picks the bridges; each bridge must be shorter than its cap, set by the
    smaller piece's extent; a component the tree left stranded is retried
    with its cheapest acceptable contact.
    """
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    centers = rows3("F", centers)
    labels = np.asarray(labels, dtype=np.int64)
    labels_pre_fusion = np.asarray(labels_pre_fusion, dtype=np.int64)
    nf = len(faces)
    if len(centers) != nf or len(labels) != nf or len(labels_pre_fusion) != nf:
        raise ValueError("per-face arrays need one row per face")
    disc = [list(_face_ids(c, nf)) for c in disc]
    if any(len(c) == 0 for c in disc):
        raise ValueError("a disconnected component has no faces")
    fusion_face_idx = _face_ids(fusion_faces, nf).tolist()

    # The vertices touched by any fusion-cluster face.  Components share
    # these verts (fusion = "glued at the rim"), so excluding them is what
    # reveals the real geometric gap distance.
    fusion_vert_set: set[int] = set()
    for ifi in fusion_face_idx:
        for v in faces[ifi]:
            fusion_vert_set.add(int(v))

    # Fusion proximity check: candidate gaps with both endpoints inside
    # the fusion blast radius are kissing points, not real gaps.
    # When the clearance-filtered distance exceeds the fallback
    # threshold, the near-fusion connection is used instead — the
    # component likely belongs at that fusion site.
    fusion_face_tree: KDTree | None = None
    if fusion_face_idx:
        fusion_face_tree = KDTree(centers[fusion_face_idx])

    def _component_verts(fis: np.ndarray) -> np.ndarray:
        verts = np.unique(faces[fis])
        if not fusion_vert_set:
            return verts
        mask = np.array([int(v) not in fusion_vert_set for v in verts], dtype=bool)
        kept = verts[mask]
        # Fall back to the unfiltered set if filtering would empty the
        # component (shouldn't happen for real branches but keeps the
        # KDTree well-defined).
        return kept if len(kept) > 0 else verts

    # Build KD-trees: main component + each disconnected component
    comp_data: dict[int, dict] = {}

    def _attach_fusion_dists(coords: np.ndarray) -> np.ndarray | None:
        if fusion_face_tree is None or len(coords) == 0:
            return None
        return fusion_face_tree.query(coords)[0]

    def _bbox_diag(coords: np.ndarray) -> float:
        if len(coords) == 0:
            return 0.0
        return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))

    def _data(fis: np.ndarray) -> dict:
        verts = _component_verts(fis)
        coords = vertices[verts]
        return {
            "fi": fis,
            "verts": verts,
            "coords": coords,
            "tree": KDTree(coords),
            "fusion_dist": _attach_fusion_dists(coords),
            "bbox": _bbox_diag(coords),
        }

    # Main component, then the disconnected ones, identified by their label
    comp_data[main] = _data(np.where(labels == main)[0])
    for fis in disc:
        comp_data[int(labels[fis[0]])] = _data(np.asarray(fis))

    # For each disconnected component, find its nearest neighbour.
    # Deduplicate: if A→B and B→A both exist, keep only one.
    gaps = []
    n_capped = [0]  # gaps skipped by the size-aware bridge cap
    seen_pairs: set[tuple[int, int]] = set()
    disc_cids = [int(labels[fis[0]]) for fis in disc]

    # Build face adjacency for BFS-based tip selection
    non_degen = non_degenerate(faces)
    edge_to_faces_gap: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(nf):
        if not non_degen[fi]:
            continue
        face = faces[fi]
        for i in range(3):
            e = (
                min(int(face[i]), int(face[(i + 1) % 3])),
                max(int(face[i]), int(face[(i + 1) % 3])),
            )
            edge_to_faces_gap[e].append(fi)

    face_adj: dict[int, list[int]] = defaultdict(list)
    for fis_e in edge_to_faces_gap.values():
        for i in range(len(fis_e)):
            for j in range(i + 1, len(fis_e)):
                face_adj[fis_e[i]].append(fis_e[j])
                face_adj[fis_e[j]].append(fis_e[i])

    def _tip_faces_bfs(comp_fi, tip_vert_idx, n_rings=tip_rings):
        """BFS on face graph from the face nearest to tip vertex."""
        comp_set = set(comp_fi.tolist())
        vcoord = vertices[tip_vert_idx]
        face_centroids = vertices[faces[comp_fi]].mean(axis=1)
        seed_local = int(np.argmin(np.linalg.norm(face_centroids - vcoord, axis=1)))
        seed = int(comp_fi[seed_local])
        visited: set[int] = {seed}
        frontier: set[int] = {seed}
        for _ in range(n_rings):
            next_frontier: set[int] = set()
            for fi in frontier:
                for nb in face_adj[fi]:
                    if nb not in visited and nb in comp_set:
                        visited.add(nb)
                        next_frontier.add(nb)
            if not next_frontier:
                break
            frontier = next_frontier
        return sorted(visited)

    def _bridge_cap(cid_a, cid_b):
        """Longest bridge allowed between two components.

        A real break leaves the ends close relative to the smaller
        piece's own extent (and never longer than an absolute ceiling).
        A gap that dwarfs its piece is a floating fragment, not a break.
        """
        smaller_bbox = min(comp_data[cid_a]["bbox"], comp_data[cid_b]["bbox"])
        return min(
            bridge_abs_cap,
            max(bridge_floor, max_bridge_ratio * smaller_bbox),
        )

    def _add_gap(cid_a, cid_b, idx_a, idx_b, dist, fusion_join=False):
        """Emit a bridge for one component pair.  True if one was emitted."""
        pair_key = (min(cid_a, cid_b), max(cid_a, cid_b))
        if pair_key in seen_pairs:
            return False
        seen_pairs.add(pair_key)
        # A sub-1nm contact normally means the two components already
        # share a vertex (a double layer) and need no bridge.  That does
        # NOT hold when the shared vertex sits on a fusion: the join is
        # about to be deleted by `remove_fusions`, so the pieces must be
        # stitched across the fusion site instead.
        if dist < 1.0 and not fusion_join:
            return False
        if dist > _bridge_cap(cid_a, cid_b):
            n_capped[0] += 1
            return False
        va = int(comp_data[cid_a]["verts"][idx_a])
        vb = int(comp_data[cid_b]["verts"][idx_b])
        fa = _tip_faces_bfs(comp_data[cid_a]["fi"], va)
        fb = _tip_faces_bfs(comp_data[cid_b]["fi"], vb)
        if not (fa and fb):
            return False
        # Map component IDs to disc indices (-1 for main)
        disc_idx_a = disc_cids.index(cid_a) if cid_a in disc_cids else -1
        disc_idx_b = disc_cids.index(cid_b) if cid_b in disc_cids else -1
        gaps.append((fa, fb, dist, disc_idx_a, disc_idx_b))
        return True

    # Precompute pairwise distances between all components (main + disc).
    # Then build an MST rooted at main so every disc component has a
    # bridge path back to main — no missing chain links, no duplicates.
    all_cids = [main] + disc_cids
    # edge_info keyed by sorted pair:
    #   (mst_cost, real_dist, a, ia, b, ib, fusion_join)
    edge_info: dict[tuple[int, int], tuple[float, float, int, int, int, int, bool]] = {}

    def _orig_label(cid: int) -> int:
        """Component's label *before* fusion-wall refinement.

        Refinement only splits, so every face of a refined component
        carries the same pre-refinement label; siblings share it.
        """
        return int(labels_pre_fusion[int(comp_data[cid]["fi"][0])])

    def _flank(cid: int, idx: int) -> float:
        """Local two-sidedness at a candidate contact: 0 = the piece ENDS at
        the contact (tip/cut → real break), ~1 = the piece CONTINUES past it
        (a side-by-side crossing → kiss).  Measured in a bounded neighborhood
        so branch length can't inflate it."""
        cd = comp_data[cid]
        p = cd["coords"][idx]
        nb = cd["tree"].query_ball_point(p, kiss_r, return_sorted=True)
        if len(nb) < 6:
            return 0.0
        nbr = cd["coords"][nb]
        vh = svd_rows(nbr - nbr.mean(0), False)
        t = (nbr - p) @ vh[0]
        return 2.0 * min(float(np.mean(t > 0)), float(np.mean(t < 0)))

    for i, cid_a in enumerate(all_cids):
        for cid_b in all_cids[i + 1 :]:
            da = comp_data[cid_a]
            db = comp_data[cid_b]
            dists, idxs = nearest(db["tree"], da["coords"])
            # Raw best match (before clearance masking).
            raw_min_i = int(np.argmin(dists))
            raw_dist = float(dists[raw_min_i])
            # Mask candidate pairs whose endpoints lie within
            # the fusion blast radius — those are kissing-point
            # artifacts, not real gaps.  Both sides must be
            # clear.  If the filtered distance exceeds the
            # fallback threshold (6x median_edge ≈ 95th pctile
            # of normal gaps), the near-fusion connection is
            # kept instead — the component likely belongs at
            # that fusion site.
            if (
                fusion_face_tree is not None
                and da["fusion_dist"] is not None
                and db["fusion_dist"] is not None
            ):
                a_clear = da["fusion_dist"] > fusion_clear
                b_clear = db["fusion_dist"][idxs] > fusion_clear
                valid = a_clear & b_clear
                if not valid.any():
                    # No candidates survive clearance — fall
                    # back to raw match.  dist < 1.0 (shared vert =
                    # already joined) keeps its real ~0 cost so the
                    # MST uses the join and _add_gap emits no bridge.
                    min_i = raw_min_i
                    cost = raw_dist
                else:
                    masked = np.where(valid, dists, np.inf)
                    filt_min_i = int(np.argmin(masked))
                    filt_dist = float(masked[filt_min_i])
                    if filt_dist > fusion_fallback and raw_dist >= 1.0:
                        # Filtered gap too far — the
                        # near-fusion connection is the real
                        # reconnection point.
                        min_i = raw_min_i
                        cost = raw_dist
                    else:
                        min_i = filt_min_i
                        # dist < 1.0 (shared vert) keeps its real ~0
                        # cost — see the no-fusion branch below.
                        cost = filt_dist
            else:
                min_i = raw_min_i
                # dist < 1.0 means the two components share a vertex —
                # they are already joined there (a double-layer, or a
                # non-manifold edge that find_fusions did not report).
                # Use the real (~0) distance as the cost so the MST
                # satisfies the component via that join; _add_gap emits
                # no bridge for dist < 1.0.  (Forcing inf here would
                # push the component onto a far finite edge instead —
                # the "stretched bridge across the cell" overshoot.)
                cost = raw_dist
            # Penalize a "kiss" candidate — one where BOTH surfaces continue
            # past the contact (a side-by-side crossing) — so a real break
            # (tip-to-tip or a planar cut, where at least one side ENDS at the
            # contact) outranks a closer kiss in the MST.  cost < 1.0 is a
            # shared-vertex join, not a gap, and is never penalized.
            mst_cost = cost
            if cost >= 1.0:
                flank = min(
                    _flank(cid_a, min_i),
                    _flank(cid_b, int(idxs[min_i])),
                )
                mst_cost = cost * (1.0 + kiss_penalty * flank)
            # Is a sub-1nm contact a durable join, or a fusion that
            # remove_fusions will delete?  Only the latter needs a
            # bridge, and only between SIBLINGS — two halves that a
            # fusion wall split out of one original component.  Any
            # other pair merely touches at a fusion; bridging there
            # would put a stitch on a tube's flank, and the tip removal
            # that precedes the stitch can sever the tube.
            fusion_join = False
            if (
                cost < 1.0
                and da["fusion_dist"] is not None
                and db["fusion_dist"] is not None
                and _orig_label(cid_a) == _orig_label(cid_b)
            ):
                fusion_join = (
                    min(
                        float(da["fusion_dist"][min_i]),
                        float(db["fusion_dist"][int(idxs[min_i])]),
                    )
                    <= fusion_clear
                )
            key = (min(cid_a, cid_b), max(cid_a, cid_b))
            edge_info[key] = (
                mst_cost,  # MST selection cost (kiss-penalized)
                cost,  # real bridge distance (for _add_gap)
                cid_a,
                min_i,  # idx into cid_a's coords
                cid_b,
                int(idxs[min_i]),  # idx into cid_b's coords
                fusion_join,  # bridge across a doomed fusion join
            )

    # Track which components the tree actually joined.  A tree edge is
    # *satisfied* when it produced a bridge, or when it is a sub-1nm
    # contact that already joins the two (no bridge needed).
    joined: dict[int, int] = {cid: cid for cid in all_cids}

    def _root(cid: int) -> int:
        r = cid
        while joined[r] != r:
            r = joined[r]
        while joined[cid] != r:
            joined[cid], cid = r, joined[cid]
        return r

    def _join(cid_a: int, cid_b: int) -> None:
        ra, rb = _root(cid_a), _root(cid_b)
        if ra != rb:
            joined[ra] = rb

    # Sub-1nm contacts join their components wherever they occur, not
    # only along tree edges — record them all so the repair below does
    # not mistake an already-joined component for a stranded one.
    for (ka, kb), info in edge_info.items():
        if info[1] < 1.0 and not info[6]:
            _join(ka, kb)

    # A fragment cannot carry a chain onward: remove_gaps peels a tip
    # patch off each end of every bridge, and two patches erase a small
    # component outright — destroying the very chain routed through it.
    # Such a component may still be bridged, but only as an endpoint.
    relay_ok = {
        cid: cid == main or len(comp_data[cid]["fi"]) >= relay_min_faces
        for cid in all_cids
    }

    # Prim's MST from main
    connected: set[int] = {main}
    remaining: set[int] = set(disc_cids)

    while remaining:
        best_edge = None
        best_cost = float("inf")
        for cid_r in sorted(remaining):
            for cid_c in sorted(connected):
                if not relay_ok[cid_c]:
                    continue
                key = (min(cid_r, cid_c), max(cid_r, cid_c))
                cost = edge_info[key][0]
                if cost < best_cost:
                    best_cost = cost
                    best_edge = (cid_r, cid_c, key)
        if best_edge is None:
            break
        cid_r, cid_c, key = best_edge
        connected.add(cid_r)
        remaining.discard(cid_r)

        _mst_cost, dist, ca, idx_a, cb, idx_b, fj = edge_info[key]
        if _add_gap(ca, cb, idx_a, idx_b, dist, fusion_join=fj):
            _join(ca, cb)

    # Prim's minimises the kiss-penalised cost without knowing which
    # edges `_add_gap` will accept, and a refused edge is never retried —
    # so a component whose one tree edge lands on a small fragment (whose
    # bbox sets a tight cap) loses its only path back to main, taking
    # everything hanging off it along.  Retry each stranded cluster with
    # the cheapest edge `_add_gap` does accept; a cluster with no
    # acceptable edge at all stays disconnected, as before.
    n_repaired = 0
    while repair_stranded:
        main_root = _root(main)
        stranded = [c for c in disc_cids if _root(c) != main_root]
        if not stranded:
            break
        clusters: dict[int, list[int]] = defaultdict(list)
        for cid in stranded:
            clusters[_root(cid)].append(cid)
        progress = False
        for members in clusters.values():
            inside = set(members)
            best_key = None
            best_cost = float("inf")
            for cid_in in members:
                for cid_out in all_cids:
                    if cid_out in inside:
                        continue
                    key = (min(cid_in, cid_out), max(cid_in, cid_out))
                    mst_cost, dist, ca, _ia, cb, _ib, fj = edge_info[key]
                    if dist < 1.0 and not fj:
                        continue
                    if dist > _bridge_cap(ca, cb):
                        continue
                    if mst_cost < best_cost:
                        best_cost = mst_cost
                        best_key = key
            if best_key is None:
                continue
            _mst_cost, dist, ca, idx_a, cb, idx_b, fj = edge_info[best_key]
            if _add_gap(ca, cb, idx_a, idx_b, dist, fusion_join=fj):
                _join(ca, cb)
                n_repaired += 1
                progress = True
        if not progress:
            break

    # Sort by gap distance
    gaps.sort(key=lambda x: x[2])
    return gaps, n_capped[0], n_repaired


def validate_loop_pair(
    la: list[int],
    lb: list[int],
    *,
    min_verts: int = 4,
    max_ratio: float = 5.0,
) -> str | None:
    """Return a reason string if the loop pair is unsuitable for stitching.

    Reject 3-vertex "loops" (degenerate triangle holes that no stitcher
    can bridge to a larger rim manifoldly), reject pairs whose size
    ratio exceeds *max_ratio* (highly mismatched rims force the zipper
    or the tube fitter to fan multiple ring verts onto a single rim
    vert, producing fan vertices and non-manifold edges), and reject a
    loop paired with itself — both tips grown onto the same rim, as when
    the two pieces are still glued — which the zipper folds into a
    zero-length tube of non-manifold edges inside that rim.

    Returns ``None`` if the pair passes.
    """
    na, nb = len(la), len(lb)
    if na < min_verts or nb < min_verts:
        return f"loop too small ({na}v + {nb}v, min {min_verts})"
    ratio = max(na, nb) / min(na, nb)
    if ratio > max_ratio:
        return (
            f"loop size mismatch ({na}v + {nb}v, ratio {ratio:.1f} > {max_ratio:.1f})"
        )
    if set(la) == set(lb):
        return f"both tips reached the same rim ({na}v)"
    return None


def rescue_size(
    labels: np.ndarray,
    counts: dict[int, int],
    fa: int,
    fb: int,
) -> int:
    """How many faces a stitch between faces *fa* and *fb* wins back.

    The counterpart to :func:`sever_cost`: what the stitch buys, to weigh
    against what removing its rims would strand.  Bridging joins two
    surface components, and the smaller of them is what would otherwise
    be left behind, so that is the benefit.

    *labels* and *counts* come from :func:`face_edge_components` and a
    ``Counter`` over it — raw connectivity, before fusions are parted.
    """
    la, lb = int(labels[fa]), int(labels[fb])
    if la == lb:
        # A fusion join: the two sides are still glued here and only
        # `remove_fusions` will part them, so raw connectivity cannot say
        # how big the far side is.  Claiming no benefit keeps the original
        # rule for these — never sever to bridge a fusion.
        return 0
    return min(counts.get(la, 0), counts.get(lb, 0))


def hermite_spline(
    p0: np.ndarray,
    t0: np.ndarray,
    p1: np.ndarray,
    t1: np.ndarray,
    n: int,
) -> np.ndarray:
    """Evaluate cubic Hermite spline at *n* evenly spaced stations.

    Returns ``(n, 3)`` array including endpoints.
    """
    s = np.linspace(0, 1, n)[:, None]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def zipper_stitch(
    vertices: np.ndarray,
    face_normals: np.ndarray,
    median_edge: float,
    loop_a: list[int],
    loop_b: list[int],
    vert_to_faces: list[list[int]],
    new_verts: list[np.ndarray] | None,
    fit,
) -> list[list[int]]:
    """Bridge two boundary loops with a tubular surface, returning new triangles.

    Generates circular cross-sections along a curved Hermite-spline
    centerline so the result looks like a realistic neurite segment.
    The radius interpolates smoothly between the two loop radii, and
    parallel transport keeps the frame twist-free.

    Parameters
    ----------
    new_verts : list or None
        Mutable list to which any newly created vertex coordinates are
        appended.  Vertex indices in the returned triangles reference
        *vertices* for existing verts and ``len(vertices) + i`` for
        ``new_verts[i]``.
    fit
        ``fit(ids)`` fits a circle to the loop of vertices *ids*:
        ``(center, radius, normal)``.
    """

    if new_verts is None:
        new_verts_local: list[np.ndarray] = []
    else:
        new_verts_local = new_verts

    la = list(loop_a)
    lb = list(loop_b)

    # ── Fit loops and build curved centerline ───────────────────
    ca_fit, ra, na = fit(np.asarray(la, dtype=np.int64))
    cb_fit, rb, nb = fit(np.asarray(lb, dtype=np.int64))

    direction = cb_fit - ca_fit
    gap_dist = float(np.linalg.norm(direction))
    if np.dot(na, direction) < 0:
        na = -na
    if np.dot(nb, direction) < 0:
        nb = -nb

    n_rings = max(1, int(round(gap_dist / median_edge)) - 1)

    # Hermite tangents: scale by gap distance for natural curvature
    tangent_a = na * gap_dist
    tangent_b = nb * gap_dist

    # n_rings intermediate + 2 endpoints = n_rings + 2 stations
    n_stations = n_rings + 2
    centers = hermite_spline(ca_fit, tangent_a, cb_fit, tangent_b, n_stations)
    n_ring_pts = max(len(la), len(lb))
    n_existing = len(vertices)

    # Tangent at each station via finite differences
    tangents = np.zeros_like(centers)
    tangents[0] = centers[1] - centers[0]
    tangents[-1] = centers[-1] - centers[-2]
    for i in range(1, len(centers) - 1):
        tangents[i] = centers[i + 1] - centers[i - 1]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-10

    # Parallel-transport a stable "up" vector along the centerline
    up = np.zeros_like(tangents)
    t0 = tangents[0]
    seed_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed_up, t0)) > 0.9:
        seed_up = np.array([0.0, 1.0, 0.0])
    up[0] = seed_up - np.dot(seed_up, t0) * t0
    up[0] /= np.linalg.norm(up[0]) + 1e-10
    for i in range(1, len(centers)):
        u = up[i - 1] - np.dot(up[i - 1], tangents[i]) * tangents[i]
        norm = np.linalg.norm(u)
        up[i] = u / norm if norm > 1e-10 else up[i - 1]

    # ── Resample both loops as polar offsets (angle, radius) ────────
    # Express each loop vertex as (angle, radius) in its station's
    # local frame, resample to n_ring_pts at evenly-spaced angles,
    # then blend between the two profiles.  Angle-based resampling
    # ensures the vertex correspondence is stable across the blend.
    def _to_polar_offsets(pts_3d, center, tangent, up_vec):
        """Project 3D loop to (angle, radius) in the local frame."""
        right_vec = np.cross(tangent, up_vec)
        right_vec /= np.linalg.norm(right_vec) + 1e-10
        rel = pts_3d - center
        u_coords = np.dot(rel, up_vec)
        r_coords = np.dot(rel, right_vec)
        angles = np.arctan2(r_coords, u_coords)
        radii = np.sqrt(u_coords**2 + r_coords**2)
        return angles, radii

    def _resample_polar(angles, radii, n):
        """Resample a polar profile to *n* evenly-spaced angles."""
        order = np.argsort(angles, kind="stable")
        a_sorted = angles[order]
        r_sorted = radii[order]
        # Close the loop
        a_closed = np.concatenate(
            [a_sorted - 2 * np.pi, a_sorted, a_sorted + 2 * np.pi]
        )
        r_closed = np.concatenate([r_sorted, r_sorted, r_sorted])
        target_angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
        resampled_r = np.interp(target_angles, a_closed, r_closed)
        return target_angles, resampled_r

    ang_a, rad_a = _to_polar_offsets(vertices[la], centers[0], tangents[0], up[0])
    ang_b, rad_b = _to_polar_offsets(vertices[lb], centers[-1], tangents[-1], up[-1])
    target_angles, profile_a = _resample_polar(ang_a, rad_a, n_ring_pts)
    _, profile_b = _resample_polar(ang_b, rad_b, n_ring_pts)

    # ── Generate ALL rings including at endpoint stations ─────────
    # We generate rings at every station (including 0 and N-1) so that
    # ALL ring-to-ring connections are between uniform vertex counts.
    # The boundary loops (la, lb) are stitched to their co-located
    # endpoint rings separately — this avoids the cross-opening fan
    # that occurs when stitching an irregular boundary loop directly
    # to a uniform ring at a different station.
    ring_ids: list[list[int]] = []

    for si in range(n_stations):
        t = si / (n_stations - 1)  # 0 at A, 1 at B
        c = centers[si]
        u_vec = up[si]
        right_vec = np.cross(tangents[si], u_vec)
        right_vec /= np.linalg.norm(right_vec) + 1e-10
        profile = profile_a * (1 - t) + profile_b * t
        ids = []
        for j in range(n_ring_pts):
            r = profile[j]
            angle = target_angles[j]
            pt = c + r * (np.cos(angle) * u_vec + np.sin(angle) * right_vec)
            ids.append(n_existing + len(new_verts_local))
            new_verts_local.append(pt)
        ring_ids.append(ids)

    # ── Stitch: boundary loops → endpoint rings → ring chain ─────
    # Sequence: la → ring[0] → ring[1] → ... → ring[-1] → lb
    triangles: list[list[int]] = []

    def _vpos(vid):
        if vid < n_existing:
            return vertices[vid]
        return new_verts_local[vid - n_existing]

    # Only stitch ring-to-ring (no boundary→ring bands that seal ends).
    all_bands: list[tuple[list[int], list[int]]] = []
    for ri in range(len(ring_ids) - 1):
        all_bands.append((ring_ids[ri], ring_ids[ri + 1]))

    for ra_ids, rb_ids in all_bands:
        na_r, nb_r = len(ra_ids), len(rb_ids)

        # Align start of rb to closest vertex in ra
        best_j = 0
        best_d = float("inf")
        p0 = _vpos(ra_ids[0])
        for j in range(nb_r):
            d = float(np.linalg.norm(_vpos(rb_ids[j]) - p0))
            if d < best_d:
                best_d = d
                best_j = j
        rb_ids = list(rb_ids[best_j:]) + list(rb_ids[:best_j])

        if na_r == nb_r:
            # Equal uniform rings (always the case here: every ring has
            # n_ring_pts verts).  A plain quad split never folds — each
            # vertical edge (ra[i], rb[i]) lands in exactly two triangles.
            # The greedy walk below can re-emit an edge at closure on a very
            # short (one-ring) tube, producing a non-manifold edge.
            for i in range(na_r):
                i2 = (i + 1) % na_r
                triangles.append([ra_ids[i], ra_ids[i2], rb_ids[i]])
                triangles.append([ra_ids[i2], rb_ids[i2], rb_ids[i]])
            continue

        ia, ib, sa, sb = 0, 0, 0, 0
        while sa < na_r or sb < nb_r:
            ia_n = (ia + 1) % na_r
            ib_n = (ib + 1) % nb_r
            can_a, can_b = sa < na_r, sb < nb_r
            if can_a and can_b:
                da = float(np.linalg.norm(_vpos(ra_ids[ia_n]) - _vpos(rb_ids[ib])))
                db = float(np.linalg.norm(_vpos(ra_ids[ia]) - _vpos(rb_ids[ib_n])))
                adv_a = da <= db
            else:
                adv_a = can_a
            if adv_a:
                triangles.append([ra_ids[ia], ra_ids[ia_n], rb_ids[ib]])
                ia = ia_n
                sa += 1
            else:
                triangles.append([ra_ids[ia], rb_ids[ib_n], rb_ids[ib]])
                ib = ib_n
                sb += 1

    # ── Weld endpoint ring verts to boundary loop verts ──────────
    # Replace each ring[0] / ring[-1] vertex ID in the triangles with a
    # boundary loop vertex.  This connects the tube to the existing mesh
    # without creating cap faces across the opening.
    def _build_weld_map(ring, loop):
        """Order-preserving surjective weld of the uniform endpoint ring
        onto the boundary loop.

        Consecutive ring verts map to the same or the *next* loop vert
        (cyclic step <= 1), so every welded seam edge coincides with an
        actual boundary edge ``(loop_i, loop_{i+1})`` — never an interior
        chord across the loop.  A nearest-neighbour weld can map two
        adjacent ring verts onto non-adjacent loop verts, dropping the seam
        triangle onto an existing two-face edge and creating a non-manifold
        edge; this monotone walk cannot.  The ring is at least as long as
        the loop (``n_ring_pts = max(len(la), len(lb))``), so exactly
        ``m - 1`` advances distribute over the ``n - 1`` transitions and the
        cyclic seam closes on a rim edge.
        """
        n, m = len(ring), len(loop)
        ring_pts = np.array([_vpos(v) for v in ring])
        loop_pts = np.array([_vpos(v) for v in loop])
        if m < 2 or n < m:
            # Cannot guarantee step <= 1; fall back to nearest (tiny loops).
            tree = KDTree(loop_pts)
            _, idxs = nearest(tree, ring_pts)
            return {ring[i]: loop[int(idxs[i])] for i in range(n)}
        cur = int(np.argmin(np.linalg.norm(loop_pts - ring_pts[0], axis=1)))
        wmap = {ring[0]: loop[cur]}
        adv = 0
        for j in range(n - 1):
            remaining_dec = (n - 1) - j
            remaining_adv = (m - 1) - adv
            if remaining_adv > 0:
                nxt = (cur + 1) % m
                closer = float(np.linalg.norm(loop_pts[nxt] - ring_pts[j + 1])) < float(
                    np.linalg.norm(loop_pts[cur] - ring_pts[j + 1])
                )
                # Force an advance once we can no longer afford to stay, so
                # all m-1 advances land and the seam covers every rim edge.
                if remaining_adv >= remaining_dec or closer:
                    cur = nxt
                    adv += 1
            wmap[ring[j + 1]] = loop[cur]
        return wmap

    weld = {}
    weld.update(_build_weld_map(ring_ids[0], la))
    weld.update(_build_weld_map(ring_ids[-1], lb))

    if weld:
        triangles = [[weld.get(v, v) for v in tri] for tri in triangles]

    # Orient consistently with surrounding mesh
    ref_fis: list[int] = []
    for vi in la[:5]:
        ref_fis.extend(vert_to_faces[vi][:5])
    if ref_fis:
        ref_n = face_normals[ref_fis].mean(axis=0)
        tri_normals = [
            np.cross(
                _vpos(t[1]) - _vpos(t[0]),
                _vpos(t[2]) - _vpos(t[0]),
            )
            for t in triangles
        ]
        if np.dot(np.mean(tri_normals, axis=0), ref_n) < 0:
            triangles = [[t[0], t[2], t[1]] for t in triangles]

    return triangles


def fan_fill(
    faces: np.ndarray,
    vertices: np.ndarray,
    lp: list[int],
    vert_to_faces: list[list[int]],
    new_verts: list[np.ndarray],
) -> list[list[int]]:
    """Close the hole bounded by the loop *lp* with a fan from its centroid,
    wound so that every border edge runs against the kept face beside it;
    the centre is appended to *new_verts*.  *vert_to_faces* lists the kept
    faces at each vertex."""
    n = len(lp)
    along = against = 0
    for i in range(n):
        u, v = lp[i], lp[(i + 1) % n]
        for f in vert_to_faces[u]:
            t = faces[f]
            for k in range(3):
                e = (int(t[k]), int(t[(k + 1) % 3]))
                along += e == (u, v)
                against += e == (v, u)
    ring = lp[::-1] if along >= against else list(lp)
    # the centroid, summed left to right
    c = np.zeros(3)
    for v in lp:
        c = c + vertices[v]
    centre = len(vertices) + len(new_verts)
    new_verts.append(c / n)
    return [[ring[i], ring[(i + 1) % n], centre] for i in range(n)]


def stitch_and_rebuild(
    faces,
    vertices,
    face_normals,
    median_edge: float,
    sel: set[int],
    loop_pairs: list[tuple[list[int], list[int]]],
    fills: list[list[int]],
    fit,
) -> tuple[np.ndarray, np.ndarray]:
    """Degenerate the faces of *sel* to ``[0, 0, 0]``, append a tube for each
    loop pair (:func:`zipper_stitch`), then a fan for each loop of *fills*
    (:func:`fan_fill`): ``(new vertices to append, faces)``."""
    faces = np.asarray(faces)
    vertices = np.asarray(vertices)
    # Vert-to-face for orientation (non-removed faces only)
    vert_to_faces: list[list[int]] = [[] for _ in range(len(vertices))]
    non_degen = non_degenerate(faces)
    for fi, face in enumerate(faces):
        if fi not in sel and non_degen[fi]:
            for v in face:
                vert_to_faces[int(v)].append(fi)

    all_stitch: list[list[int]] = []
    new_verts: list[np.ndarray] = []
    for loop_a, loop_b in loop_pairs:
        all_stitch.extend(
            zipper_stitch(
                vertices,
                face_normals,
                median_edge,
                loop_a,
                loop_b,
                vert_to_faces,
                new_verts,
                fit,
            )
        )
    for lp in fills:
        all_stitch.extend(fan_fill(faces, vertices, lp, vert_to_faces, new_verts))

    # Degenerate removed faces, append stitch faces at the end
    new_faces = faces.astype(np.int64, copy=True)
    for fi in sel:
        new_faces[fi] = 0
    if all_stitch:
        new_faces = np.vstack([new_faces, np.array(all_stitch, dtype=np.int64)])
    added = np.array(new_verts, dtype=np.float64).reshape(-1, 3)
    return added, new_faces


def removal_would_sever(sel: set[int], edge_to_faces, faces=None) -> bool:
    """Would removing the connected patch *sel* cut its surface in two?

    `expand_tip_to_good_rim` grows a patch by BFS, so *sel* is always
    connected and its border is a single closed curve when it sits at a
    tube END (a cap).  Two or more border loops mean the patch is an
    annulus that wraps the surface — removing it leaves the far side
    disconnected, severing whatever lies beyond.  *edge_to_faces* is the
    dict :func:`skeliner.python.mesh.edge_to_faces` builds or a
    ``Topology`` of either path.
    """
    if hasattr(edge_to_faces, "border_loops"):
        loops = edge_to_faces.border_loops(sel)
    else:
        loops = border_loops(set(int(f) for f in sel), edge_to_faces, faces)
    return sum(1 for lp in loops if len(lp) >= 3) >= 2


def remove_gaps(
    faces,
    vertices,
    face_normals,
    median_edge: float,
    labels,
    gaps,
    fit,
) -> tuple[np.ndarray, np.ndarray | None, int, int, int]:
    """Bridge the *gaps*, each a pair of tip-face lists: ``(new vertices,
    faces or None when nothing was stitched, n_bridged, n_removed,
    n_skipped)``.

    Each side's tip is peeled to a clean rim; a pair that cannot be
    stitched, or whose removal would strand more than the stitch wins back
    (sizes from the raw face-edge components *labels*, computed on first
    need when None), is skipped.  Rims that share two or more vertices
    belong to tips lying side by side, glued along those vertices (a fusion
    join): a tube from rim to rim would weld both of its ends onto the glue
    and fold over it, so both tips go and the one hole they leave is closed
    with a fan — or the gap is skipped when their union does not leave
    exactly one hole.  *fit* is :func:`zipper_stitch`'s.
    """
    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    face_normals = rows3("N", face_normals)
    nf = len(faces)
    if len(face_normals) != nf:
        raise ValueError("face_normals needs one row per face")
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) != nf:
            raise ValueError("labels needs one value per face")
    tips = []
    for side_a, side_b in gaps:
        sides = []
        for side in (side_a, side_b):
            if len(side) == 0:
                raise ValueError("a gap side has no faces")
            sides.append(_face_ids(side, nf).tolist())
        tips.append(sides)

    # Build edge / face adjacency once.
    edge_to_face = edge_to_faces(faces)
    face_adj = face_adjacency(edge_to_face)

    # Component labels for `rescue_size`.  Only a gap that actually trips
    # the sever check needs them, and labelling a multi-million-face mesh
    # costs ~19 s, so they are built on first use.
    sizes: dict = {}

    def _lazy_rescue(fa: int, fb: int) -> int:
        if not sizes:
            comp = labels
            if comp is None:
                comp, _ = face_edge_components(faces, len(vertices))
            sizes["labels"] = comp
            sizes["counts"] = Counter(int(x) for x in comp)
        return rescue_size(sizes["labels"], sizes["counts"], fa, fb)

    # For each gap, expand each side's tip until both rims are real
    # loops. The initial tip from find_gaps can produce a degenerate
    # "rim" (e.g. 3 verts when the tip is a tongue attached at a single
    # triangle); peeling more rings exposes a clean cross-section that
    # the stitcher can bridge manifoldly.
    faces_to_remove: set[int] = set()
    loop_pairs: list[tuple[list[int], list[int]]] = []
    fills: list[list[int]] = []
    n_skipped = 0
    for faces_a, faces_b in tips:
        sel_a, loop_a = expand_tip_to_good_rim(
            faces_a, edge_to_face, face_adj, faces=faces
        )
        sel_b, loop_b = expand_tip_to_good_rim(
            faces_b, edge_to_face, face_adj, faces=faces
        )
        if loop_a is None or loop_b is None:
            n_skipped += 1
            continue

        # Tips side by side, glued along the vertices their rims share:
        # remove both and close the one hole they leave.
        set_a, set_b = set(loop_a), set(loop_b)
        if len(set_a & set_b) >= 2 and set_a != set_b:
            union = sel_a | sel_b
            holes = [
                lp for lp in border_loops(union, edge_to_face, faces) if len(lp) >= 3
            ]
            if len(holes) == 1:
                fills.append(holes[0])
                faces_to_remove |= union
            else:
                n_skipped += 1
            continue

        # Final safety net: if expansion still didn't reach a clean
        # pair (degenerate or wildly mismatched), skip rather than
        # introduce a fusion.
        if validate_loop_pair(loop_a, loop_b) is not None:
            n_skipped += 1
            continue

        # A bridge that lands on a tube's FLANK rather than its end peels
        # a rim that wraps the circumference and strands everything
        # distal.  Whether that is worth avoiding depends on how much it
        # strands, not on the fact that it can: on 564241053 a 21 nm gap
        # strands 13,438 f to rescue 54 f, while on 554656742 a patch of
        # the same annular shape strands 4 f and rescues 4,160 f.  Skip
        # only when the stitch costs more than it saves — two face
        # counts, no distance scope and no threshold.
        sev_a = removal_would_sever(sel_a, edge_to_face, faces)
        sev_b = removal_would_sever(sel_b, edge_to_face, faces)
        if sev_a or sev_b:
            rescue = _lazy_rescue(faces_a[0], faces_b[0])
            if (sev_a and sever_cost(sel_a, face_adj, rescue) > rescue) or (
                sev_b and sever_cost(sel_b, face_adj, rescue) > rescue
            ):
                n_skipped += 1
                continue

        loop_pairs.append((loop_a, loop_b))
        faces_to_remove |= sel_a | sel_b

    if not loop_pairs and not fills:
        return np.empty((0, 3)), None, 0, 0, n_skipped
    new_verts, new_faces = stitch_and_rebuild(
        faces,
        vertices,
        face_normals,
        median_edge,
        faces_to_remove,
        loop_pairs,
        fills,
        fit,
    )
    n_bridged = len(loop_pairs) + len(fills)
    return new_verts, new_faces, n_bridged, len(faces_to_remove), n_skipped


def find_parallel_patches(
    faces,
    vertices,
    normals,
    boundaries,
    normal_thresh: float,
    boundary_dist: float,
    expand_tilted: bool,
    loose_normal_thresh: float,
) -> list[tuple[int, int, list[int], list[int], list[int]]]:
    """Patches of axis-aligned faces at the chunk *boundaries* (``(axis,
    planes)``) that face an opposing patch across the plane — marching
    cubes' double layer: ``(axis, bval, faces, up, down)`` each, before
    ``skeliner.pre.find_parallel_patches``' sandwich-overlap filter and
    final sort.  Faces next to two or more patch faces join, the patches
    are merged by connectivity and, with *expand_tilted*, grown by the
    tilted faces at their rims (:func:`expand_parallel_patches`)."""
    verts = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(verts))
    normals = rows3("N", normals)
    if len(normals) != len(faces):
        raise ValueError("normals needs one row per face")
    if any(int(ax) > 2 for ax, _ in boundaries):
        raise ValueError("boundary axes are 0, 1 or 2")
    boundaries = {
        int(ax): np.asarray(bvals, dtype=np.float64) for ax, bvals in boundaries
    }
    centroids = verts[faces].mean(axis=1)

    # Perpendicular axes for overlap check
    perp = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    results = []

    for axis, bvals in boundaries.items():
        n_ax = normals[:, axis]
        aligned = np.abs(n_ax) > normal_thresh

        # For each face, find nearest boundary on this axis
        near_mask = np.zeros(len(faces), dtype=bool)
        face_bval = np.full(len(faces), -1, dtype=np.int64)
        for bval in bvals:
            m = np.abs(centroids[:, axis] - bval) < boundary_dist
            near_mask |= m
            face_bval[m] = int(round(bval))

        target_mask = aligned & near_mask
        target = np.where(target_mask)[0]
        if len(target) < 2:
            continue
        target_set = set(int(fi) for fi in target)

        # Face adjacency among target faces
        edge_to_faces: dict[tuple, list[int]] = defaultdict(list)
        for fi in target:
            f = faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                edge_to_faces[e].append(int(fi))

        # Connected components — same boundary only
        visited: set[int] = set()
        patches: list[tuple[list[int], int]] = []
        for seed in target:
            seed = int(seed)
            if seed in visited:
                continue
            seed_bv = face_bval[seed]
            comp: list[int] = []
            queue = deque([seed])
            visited.add(seed)
            while queue:
                fi = queue.popleft()
                comp.append(fi)
                f = faces[fi]
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    e = (min(a, b), max(a, b))
                    for nb in edge_to_faces[e]:
                        if (
                            nb not in visited
                            and nb in target_set
                            and face_bval[nb] == seed_bv
                        ):
                            visited.add(nb)
                            queue.append(nb)
            patches.append((comp, int(seed_bv)))

        # Check each patch for opposing faces
        perp_axes = perp[axis]

        def _any_face_overlaps(faces_a, faces_b):
            """Check if any face from A overlaps any face from B in perp plane.

            Projects triangles onto the perpendicular plane and checks
            for vertex-in-triangle overlap.
            """
            # Get triangle vertices projected onto perp plane
            verts_a = verts[faces[faces_a]][:, :, perp_axes]  # (Na, 3, 2)
            verts_b = verts[faces[faces_b]][:, :, perp_axes]  # (Nb, 3, 2)

            # Quick bbox pre-filter
            a_min, a_max = verts_a.reshape(-1, 2).min(0), verts_a.reshape(-1, 2).max(0)
            b_min, b_max = verts_b.reshape(-1, 2).min(0), verts_b.reshape(-1, 2).max(0)
            if not (np.all(a_min <= b_max) and np.all(b_min <= a_max)):
                return False

            # Check if any vertex from A is strictly inside any triangle
            # of B (and vice versa).  Use cross-product sign test with
            # strict inequalities — points on edges/vertices don't count
            # as "inside" so that adjacent (but non-overlapping) triangles
            # sharing a vertex or edge are not flagged.
            def _point_in_tri(px, py, tri):
                """Test points strictly inside triangle tri (3,2)."""
                x0, y0 = tri[0]
                x1, y1 = tri[1]
                x2, y2 = tri[2]
                d00 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
                d01 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                d02 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)
                return ((d00 > 0) & (d01 > 0) & (d02 > 0)) | (
                    (d00 < 0) & (d01 < 0) & (d02 < 0)
                )

            # Exclude shared vertices between A and B
            vidx_a = set(int(v) for v in faces[faces_a].ravel())
            vidx_b = set(int(v) for v in faces[faces_b].ravel())
            shared_vidx = vidx_a & vidx_b

            # Collect vertices from A (excluding shared), test against B
            pts_a = verts_a.reshape(-1, 2)
            mask_a = np.ones(len(pts_a), dtype=bool)
            flat_vidx_a = faces[faces_a].ravel()
            for k, vi in enumerate(flat_vidx_a):
                if int(vi) in shared_vidx:
                    mask_a[k] = False
            pts_a_filtered = pts_a[mask_a]
            if len(pts_a_filtered) > 0:
                for tri in verts_b:
                    if np.any(
                        _point_in_tri(pts_a_filtered[:, 0], pts_a_filtered[:, 1], tri)
                    ):
                        return True

            # Vice versa
            pts_b = verts_b.reshape(-1, 2)
            mask_b = np.ones(len(pts_b), dtype=bool)
            flat_vidx_b = faces[faces_b].ravel()
            for k, vi in enumerate(flat_vidx_b):
                if int(vi) in shared_vidx:
                    mask_b[k] = False
            pts_b_filtered = pts_b[mask_b]
            if len(pts_b_filtered) > 0:
                for tri in verts_a:
                    if np.any(
                        _point_in_tri(pts_b_filtered[:, 0], pts_b_filtered[:, 1], tri)
                    ):
                        return True

            return False

        for comp, bv in patches:
            fi_arr = np.array(comp)
            n_ax_comp = n_ax[fi_arr]
            pos_fi = fi_arr[n_ax_comp > normal_thresh]
            neg_fi = fi_arr[n_ax_comp < -normal_thresh]

            has_overlap = False

            # Self-opposing: up/down within same patch
            if len(pos_fi) > 0 and len(neg_fi) > 0:
                if _any_face_overlaps(pos_fi, neg_fi):
                    has_overlap = True

            # Nearby opposing: other patches at same boundary
            if not has_overlap:
                comp_set = set(comp)
                others = [
                    int(fi)
                    for fi in target
                    if face_bval[fi] == bv and fi not in comp_set
                ]
                if others:
                    other_nax = n_ax[others]
                    mean_dir = n_ax_comp.mean()
                    if mean_dir > 0:
                        opp = [
                            others[j]
                            for j in range(len(others))
                            if other_nax[j] < -normal_thresh
                        ]
                    else:
                        opp = [
                            others[j]
                            for j in range(len(others))
                            if other_nax[j] > normal_thresh
                        ]
                    if opp:
                        if _any_face_overlaps(fi_arr, np.array(opp)):
                            has_overlap = True

            if has_overlap:
                results.append(
                    {
                        "axis": axis,
                        "bval": bv,
                        "faces": [int(f) for f in fi_arr],
                        "up_faces": [int(f) for f in pos_fi],
                        "down_faces": [int(f) for f in neg_fi],
                    }
                )

    # ── Neighbor expansion across all patches ──────────────────────
    # Include non-target faces adjacent to 2+ faces from ANY flagged
    # patch (catches slightly-tilted faces between patches).
    all_patch_faces: set[int] = set()
    for r in results:
        all_patch_faces.update(r["faces"])

    # Build full edge→face for faces near any boundary
    all_near_set: set[int] = set()
    for axis_k in boundaries:
        near_k = np.abs(centroids[:, axis_k, np.newaxis] - boundaries[axis_k]) < 30.0
        all_near_set.update(int(i) for i in np.where(near_k.any(axis=1))[0])

    full_e2f: dict[tuple, list[int]] = defaultdict(list)
    for fi in all_near_set:
        f = faces[fi]
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            full_e2f[(min(a, b), max(a, b))].append(fi)

    neighbor_count: dict[int, int] = defaultdict(int)
    for fi in all_patch_faces:
        f = faces[fi]
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            for nb in full_e2f[e]:
                if nb not in all_patch_faces:
                    neighbor_count[nb] += 1

    expanded = {nb for nb, cnt in neighbor_count.items() if cnt >= 2}
    all_patch_faces.update(expanded)

    # ── Merge into connected patches ──────────────────────────────
    # Re-run connected components on all flagged faces so that
    # neighboring patches (possibly linked through expanded faces)
    # form a single patch.
    merge_e2f: dict[tuple, list[int]] = defaultdict(list)
    for fi in all_patch_faces:
        f = faces[fi]
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            merge_e2f[(min(a, b), max(a, b))].append(fi)

    visited_merge: set[int] = set()
    merged: list[list[int]] = []
    for seed in sorted(all_patch_faces):
        if seed in visited_merge:
            continue
        comp: list[int] = []
        queue = deque([seed])
        visited_merge.add(seed)
        while queue:
            fi = queue.popleft()
            comp.append(fi)
            f = faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                e = (min(a, b), max(a, b))
                for nb in merge_e2f[e]:
                    if nb not in visited_merge and nb in all_patch_faces:
                        visited_merge.add(nb)
                        queue.append(nb)
        merged.append(sorted(comp))

    # Rebuild results from merged components.  Determine axis from
    # the dominant normal direction among perfectly-aligned faces.
    results = []
    for comp in merged:
        fi_arr = np.array(comp)
        # Find dominant axis: the one with most |n| > normal_thresh faces
        best_axis = 0
        best_count = 0
        for ax in range(3):
            cnt = int(np.sum(np.abs(normals[fi_arr, ax]) > normal_thresh))
            if cnt > best_count:
                best_count = cnt
                best_axis = ax
        # Determine boundary value from centroids
        bv = int(round(float(np.median(centroids[fi_arr, best_axis]))))
        # Find nearest actual boundary
        bvals_ax = boundaries.get(best_axis, np.array([]))
        if len(bvals_ax) > 0:
            idx = np.argmin(np.abs(bvals_ax - bv))
            bv = int(round(bvals_ax[idx]))

        n_ax_comp = normals[fi_arr, best_axis]
        pos_fi = fi_arr[n_ax_comp > 0.5]
        neg_fi = fi_arr[n_ax_comp < -0.5]
        results.append(
            {
                "axis": best_axis,
                "bval": bv,
                "faces": [int(f) for f in fi_arr],
                "up_faces": [int(f) for f in pos_fi],
                "down_faces": [int(f) for f in neg_fi],
            }
        )

    if expand_tilted:
        results = expand_parallel_patches(
            faces,
            verts,
            normals,
            results,
            loose_normal_thresh=loose_normal_thresh,
            boundary_dist=boundary_dist,
        )
    return [
        (r["axis"], r["bval"], r["faces"], r["up_faces"], r["down_faces"])
        for r in results
    ]


def expand_parallel_patches(
    faces,
    vertices,
    normals,
    patches: list[dict],
    *,
    loose_normal_thresh: float = 0.9,
    boundary_dist: float = 30.0,
) -> list[dict]:
    """Absorb tilted transition faces adjacent to strict-parallel seed
    patches.

    At the rim of a parallel sandwich patch, marching cubes often emits
    a few slightly-tilted triangles that bridge the flat sheet to the
    surrounding geometry.  These faces fail the strict ``normal_thresh``
    test but should still be treated as part of the patch, otherwise
    they split one logical sandwich into multiple disconnected seed
    patches (and leave the rim contour broken).

    For each seed patch this function BFS-expands through edge-adjacent
    faces whose axis-aligned normal component satisfies

    * same sign as the patch (``n_axis * sign > 0``),
    * ``|n_axis| > loose_normal_thresh`` (default 0.9 — about 26° tilt),
    * centroid within ``2 * boundary_dist`` of the patch's ``bval``,
    * projection (in the plane perpendicular to the axis) overlapping
      the bbox of an *opposing* patch at the same axis/bval.

    Patches whose expansions share any absorbed face get merged into
    one in the returned list.
    """

    if not patches:
        return patches

    verts = np.asarray(vertices)
    faces = np.asarray(faces)
    normals = np.asarray(normals)
    centroids = verts[faces].mean(axis=1)
    perp = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    expand_dist = 2.0 * boundary_dist

    # Mesh-wide edge adjacency
    full_ef: dict[tuple, list[int]] = defaultdict(list)
    for fi in range(len(faces)):
        f = faces[fi]
        if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
            continue
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            full_ef[(min(a, b), max(a, b))].append(int(fi))

    def patch_sign(p):
        if p["up_faces"]:
            return 1
        if p["down_faces"]:
            return -1
        return 0

    # Group patches by (axis, bval) and compute opposing XY bbox per sign.
    groups: dict[tuple, list[int]] = defaultdict(list)
    for pi, p in enumerate(patches):
        groups[(int(p["axis"]), int(p["bval"]))].append(pi)

    expansions: list[set[int]] = [set(p["faces"]) for p in patches]

    for (axis, bval), group_pis in groups.items():
        perp_axes = perp[axis]

        opp_bbox: dict[int, tuple] = {}
        for sign_val in (1, -1):
            opp_vs: list[np.ndarray] = []
            for pj in group_pis:
                if patch_sign(patches[pj]) == -sign_val:
                    opp_vs.append(verts[faces[patches[pj]["faces"]]].reshape(-1, 3))
            if not opp_vs:
                continue
            stacked = np.vstack(opp_vs)
            opp_bbox[sign_val] = (
                stacked[:, perp_axes].min(0),
                stacked[:, perp_axes].max(0),
            )

        for pi in group_pis:
            sign_p = patch_sign(patches[pi])
            if sign_p == 0 or sign_p not in opp_bbox:
                continue
            xy_min, xy_max = opp_bbox[sign_p]
            q = deque(patches[pi]["faces"])
            while q:
                fi = q.popleft()
                f = faces[fi]
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    e = (min(a, b), max(a, b))
                    for nb in full_ef[e]:
                        if nb in expansions[pi]:
                            continue
                        n_ax = normals[nb, axis]
                        if n_ax * sign_p <= 0:
                            continue
                        if abs(n_ax) < loose_normal_thresh:
                            continue
                        if abs(centroids[nb, axis] - bval) > expand_dist:
                            continue
                        v = verts[faces[nb]][:, perp_axes]
                        if not (
                            (v.min(0) < xy_max).all() and (v.max(0) > xy_min).all()
                        ):
                            continue
                        expansions[pi].add(int(nb))
                        q.append(int(nb))

    # Merge patches whose expansions share absorbed faces (same axis,
    # bval, and sign).
    parent = list(range(len(patches)))

    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uf_union(a, b):
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            if patches[i]["axis"] != patches[j]["axis"]:
                continue
            if patches[i]["bval"] != patches[j]["bval"]:
                continue
            if patch_sign(patches[i]) != patch_sign(patches[j]):
                continue
            if expansions[i] & expansions[j]:
                uf_union(i, j)

    merged_groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(patches)):
        merged_groups[uf_find(i)].append(i)

    new_patches: list[dict] = []
    for root, members in merged_groups.items():
        axis = int(patches[root]["axis"])
        bval = int(patches[root]["bval"])
        merged_faces: set[int] = set()
        merged_up: set[int] = set()
        merged_down: set[int] = set()
        orig: set[int] = set()
        for mi in members:
            merged_faces |= expansions[mi]
            merged_up.update(patches[mi]["up_faces"])
            merged_down.update(patches[mi]["down_faces"])
            orig.update(patches[mi]["faces"])
        # Classify absorbed faces by their signed normal component
        for fi in merged_faces - orig:
            if normals[fi, axis] > 0:
                merged_up.add(fi)
            else:
                merged_down.add(fi)
        new_patches.append(
            {
                "axis": axis,
                "bval": bval,
                "faces": sorted(int(f) for f in merged_faces),
                "up_faces": sorted(int(f) for f in merged_up),
                "down_faces": sorted(int(f) for f in merged_down),
            }
        )

    return new_patches


def fold_faces(faces, patch: dict) -> set[int]:
    """Find faces on fold edges within a patch.

    A fold edge is where an up face shares an edge with a down face —
    the surface turns 180°.  Returns the set of face indices directly
    on fold edges.
    """

    up_set = set(patch["up_faces"])
    down_set = set(patch["down_faces"])

    edge_faces: dict[tuple, list[int]] = defaultdict(list)
    for fi in patch["faces"]:
        f = faces[fi]
        for i in range(3):
            e = (
                min(int(f[i]), int(f[(i + 1) % 3])),
                max(int(f[i]), int(f[(i + 1) % 3])),
            )
            edge_faces[e].append(fi)

    fold: set[int] = set()
    for e, fis in edge_faces.items():
        has_up = any(fi in up_set for fi in fis)
        has_down = any(fi in down_set for fi in fis)
        if has_up and has_down:
            fold.update(fis)

    return fold


def trace_edge_loops(
    edges: list[tuple[int, int]],
) -> list[tuple[list[int], bool]]:
    """Trace a set of undirected edges into ordered vertex sequences.

    Returns a list of ``(vertex_sequence, is_closed)`` pairs.  For
    closed loops the last vertex is implicitly connected back to the
    first (i.e. the wrap-around edge exists in the input).  For open
    chains the first and last vertices are the dangling endpoints.

    Edges are taken in sorted order, so the result does not depend on the
    order of *edges*.  Open chains are emitted first, starting at any
    degree-1 vertex.
    Remaining unused edges form closed loops.  Non-manifold vertices
    (degree ≥ 3) are split — each incident edge is walked at most
    once, so a figure-8 becomes two separate loops touching at the
    shared vertex.
    """

    if not edges:
        return []

    ve: dict[int, list[tuple[int, int]]] = defaultdict(list)
    eset: set[tuple[int, int]] = set()
    for key in sorted((min(a, b), max(a, b)) for a, b in edges):
        if key in eset:
            continue
        eset.add(key)
        ve[key[0]].append(key)
        ve[key[1]].append(key)

    used: set[tuple[int, int]] = set()
    out: list[tuple[list[int], bool]] = []

    def walk(start_edge: tuple[int, int]) -> tuple[list[int], bool]:
        chain = [start_edge[0], start_edge[1]]
        # mark the edge itself, however the walk is oriented along it
        used.add((min(start_edge), max(start_edge)))
        curr = start_edge[1]
        closed = False
        while True:
            nxt_e = None
            for cand in ve[curr]:
                if cand not in used:
                    nxt_e = cand
                    break
            if nxt_e is None:
                break
            used.add(nxt_e)
            nxt = nxt_e[1] if nxt_e[0] == curr else nxt_e[0]
            if nxt == chain[0]:
                closed = True
                break
            chain.append(nxt)
            curr = nxt
        return chain, closed

    # Open chains: start at degree-1 endpoints
    endpoints = [v for v, es in ve.items() if len(es) == 1]
    for v in endpoints:
        for e in ve[v]:
            if e in used:
                continue
            oriented = e if e[0] == v else (e[1], e[0])
            chain, closed = walk(oriented)
            out.append((chain, closed))

    # Closed loops from any remaining unused edge
    for e in sorted(eset):
        if e in used:
            continue
        chain, closed = walk(e)
        out.append((chain, closed))

    return out


def zip_vertex_loops(
    loop_a: list[int],
    loop_b: list[int],
    verts: np.ndarray,
    *,
    closed_a: bool = True,
    closed_b: bool = True,
) -> list[tuple[int, int, int]]:
    """Zip two ordered vertex loops into a strip of triangles.

    Both loops are walked in lock-step with a "shortest diagonal"
    decision rule: at each step, advance whichever side produces the
    shorter bridge edge to the other side's current vertex.  Each
    source edge of loop_a and loop_b is consumed in exactly one new
    triangle, so the result is manifold by construction (no fans, no
    duplicate edges).

    Before walking, this tries every starting offset and both winding
    orders on ``loop_b`` and picks the alignment that minimises the
    sum of bridge-edge lengths.  This is O((m + n) * n) which is fine
    for small rim loops.
    """
    m = len(loop_a)
    n = len(loop_b)
    if m < 2 or n < 2:
        return []

    def walk(shifted_b: list[int]) -> tuple[float, list[tuple[int, int, int]]]:
        tris: list[tuple[int, int, int]] = []
        cost = 0.0
        i, j = 0, 0
        # Closed loops walk m + n steps total; open chains walk m-1 + n-1.
        steps_a = m if closed_a else m - 1
        steps_b = n if closed_b else n - 1
        while i < steps_a or j < steps_b:
            a_curr = loop_a[i % m]
            a_next = loop_a[(i + 1) % m]
            t_curr = shifted_b[j % n]
            t_next = shifted_b[(j + 1) % n]
            if i >= steps_a:
                tris.append((a_curr, t_next, t_curr))
                cost += float(np.linalg.norm(verts[t_next] - verts[a_curr]))
                j += 1
            elif j >= steps_b:
                tris.append((a_curr, a_next, t_curr))
                cost += float(np.linalg.norm(verts[a_next] - verts[t_curr]))
                i += 1
            else:
                d_adv_a = float(np.linalg.norm(verts[a_next] - verts[t_curr]))
                d_adv_b = float(np.linalg.norm(verts[a_curr] - verts[t_next]))
                if d_adv_a <= d_adv_b:
                    tris.append((a_curr, a_next, t_curr))
                    cost += d_adv_a
                    i += 1
                else:
                    tris.append((a_curr, t_next, t_curr))
                    cost += d_adv_b
                    j += 1
        return cost, tris

    best: tuple[float, list[tuple[int, int, int]]] | None = None
    # Try both winding directions and all starting offsets on loop_b
    # (for closed loops).  For open chains, only offset 0 is valid.
    reverse_options = [False, True]
    shift_options = range(n) if closed_b else [0]
    for reverse in reverse_options:
        base = list(loop_b[::-1] if reverse else loop_b)
        for shift in shift_options:
            shifted = base[shift:] + base[:shift]
            cost, tris = walk(shifted)
            if best is None or cost < best[0]:
                best = (cost, tris)

    return best[1] if best is not None else []


def stitch_mode_b(verts, faces, orig_faces, mode_b_faces, patches) -> np.ndarray:
    """Bridge gaps left by Mode B parallel-patch removal using
    per-sandwich rim tracing and contour zipping.

    For each ``(axis, bval)`` sandwich group in *patches*, collect the
    rim edges on each sign side (edges where an up-removed or
    down-removed face meets a surviving face), trace them into ordered
    vertex loops, pair up-side loops with down-side loops by spatial
    proximity, and zip each pair into a manifold triangle strip.

    Bridges are emitted with :func:`zip_vertex_loops`, which consumes
    each rim edge exactly once in exactly one new triangle — so the
    result has no fans, no duplicate faces, and no non-manifold edges
    by construction.  *faces* are the faces after the removal, the
    triangles appended to them returned.
    """

    degen = np.all(faces == 0, axis=1)

    # Edge → surviving faces map (excludes degenerates)
    surv_ef: dict[tuple[int, int], set[int]] = defaultdict(set)
    for fi in range(len(faces)):
        if degen[fi]:
            continue
        f = faces[fi]
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a == b or b == c or a == c:
            continue
        surv_ef[(min(a, b), max(a, b))].add(fi)
        surv_ef[(min(b, c), max(b, c))].add(fi)
        surv_ef[(min(a, c), max(a, c))].add(fi)

    mode_b_set = set(int(fi) for fi in mode_b_faces)

    def rim_edges_for(removed_side: set[int]) -> list[tuple[int, int]]:
        """Return edges of *orig_faces* in removed_side where the
        other side of the edge is a surviving face (not a degen or
        another removed face)."""
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for fi in removed_side:
            f = orig_faces[fi]
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                if a == b:
                    continue
                e = (min(a, b), max(a, b))
                if e in seen:
                    continue
                surv = surv_ef.get(e, set())
                if len(surv) == 1:
                    seen.add(e)
                    out.append(e)
        return out

    # Group Mode-B patches by (axis, bval)
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for p in patches:
        pf = set(p["faces"])
        if not (pf & mode_b_set):
            continue
        groups[(int(p["axis"]), int(p["bval"]))].append(p)

    if not groups:
        return faces

    all_new_tris: list[tuple[int, int, int]] = []
    n_pairs = 0
    n_skipped_loops = 0

    for (_axis, _bval), group in groups.items():
        up_removed: set[int] = set()
        dn_removed: set[int] = set()
        for p in group:
            up_removed.update(int(f) for f in p["up_faces"] if int(f) in mode_b_set)
            dn_removed.update(int(f) for f in p["down_faces"] if int(f) in mode_b_set)

        up_rim = rim_edges_for(up_removed)
        dn_rim = rim_edges_for(dn_removed)
        if not up_rim or not dn_rim:
            continue

        up_traces = trace_edge_loops(up_rim)
        dn_traces = trace_edge_loops(dn_rim)
        # A real contour needs at least 3 vertices (closed) or 2
        # vertices for a single-edge chain.  Drop anything smaller.
        up_traces = [
            (lp, c)
            for lp, c in up_traces
            if (c and len(lp) >= 3) or (not c and len(lp) >= 2)
        ]
        dn_traces = [
            (lp, c)
            for lp, c in dn_traces
            if (c and len(lp) >= 3) or (not c and len(lp) >= 2)
        ]
        if not up_traces or not dn_traces:
            continue

        # Pair loops greedily by centroid distance — largest loops first.
        # Perpendicular plane of the sandwich axis for bbox overlap check.
        perp_axes = [i for i in range(3) if i != _axis]
        up_info = []
        for lp, c in up_traces:
            pts = verts[lp]
            up_info.append(
                {
                    "loop": lp,
                    "closed": c,
                    "pts": pts,
                    "center": pts.mean(axis=0),
                    "bmin": pts[:, perp_axes].min(0),
                    "bmax": pts[:, perp_axes].max(0),
                }
            )
        dn_info = []
        for lp, c in dn_traces:
            pts = verts[lp]
            dn_info.append(
                {
                    "loop": lp,
                    "closed": c,
                    "pts": pts,
                    "center": pts.mean(axis=0),
                    "bmin": pts[:, perp_axes].min(0),
                    "bmax": pts[:, perp_axes].max(0),
                }
            )

        up_order = sorted(range(len(up_info)), key=lambda i: -len(up_info[i]["loop"]))
        dn_used: set[int] = set()

        for ui in up_order:
            u = up_info[ui]
            best_dn = -1
            best_d = float("inf")
            for di in range(len(dn_info)):
                if di in dn_used:
                    continue
                d_info = dn_info[di]
                # Require bbox overlap in the perpendicular plane —
                # without this, a mixed-patch rim whose up and down
                # sides cover different XY regions would get zipped
                # into a twisted strip crossing empty space.
                imin = np.maximum(u["bmin"], d_info["bmin"])
                imax = np.minimum(u["bmax"], d_info["bmax"])
                if not (imax > imin).all():
                    continue
                d = float(np.linalg.norm(u["center"] - d_info["center"]))
                if d < best_d:
                    best_d = d
                    best_dn = di
            if best_dn < 0:
                n_skipped_loops += 1
                continue
            dn_used.add(best_dn)
            d_info = dn_info[best_dn]
            tris = zip_vertex_loops(
                u["loop"],
                d_info["loop"],
                verts,
                closed_a=u["closed"],
                closed_b=d_info["closed"],
            )
            if tris:
                all_new_tris.extend(tris)
                n_pairs += 1
            else:
                n_skipped_loops += 1

        n_skipped_loops += len(dn_info) - len(dn_used)

    if not all_new_tris:
        return faces
    return np.vstack([faces, np.array(all_new_tris, dtype=np.int64)])


def remove_parallel_patches(
    faces, vertices, normals, patches
) -> tuple[np.ndarray, int, int, int, int, int]:
    """Remove the trimmed parallel *patches* (``(axis, bval, faces, up,
    down)`` each), drop the pieces that only fold patches cut off, and zip
    the sandwiches' rims together: ``(faces, n_removed, n_fold,
    n_parallel, n_orphan, n_stitched)``.  A patch with both sides, or
    without a fold (an up face edge to edge with a down face), is a
    parallel patch whose rims are zipped; one with a fold and one side is a
    fold patch, removed only."""
    verts = rows3("V", vertices)
    faces, n_vertices = checked_faces(faces, len(verts))
    normals = rows3("N", normals)
    if len(normals) != len(faces):
        raise ValueError("normals needs one row per face")
    checked = []
    for axis, bval, pf, up, down in patches:
        if int(axis) > 2:
            raise ValueError(f"no axis {axis}")
        pf, up, down = (_face_ids(ids, len(faces)).tolist() for ids in (pf, up, down))
        checked.append(
            {
                "axis": int(axis),
                "bval": int(bval),
                "faces": pf,
                "up_faces": up,
                "down_faces": down,
            }
        )
    patches = checked

    # ── Step 1: classify patches ──────────────────────────────────
    mode_a_faces: set[int] = set()  # fold patches
    mode_b_faces: set[int] = set()  # parallel patches
    all_removed: set[int] = set()
    n_a = 0
    n_b = 0

    for patch in patches:
        fold = fold_faces(faces, patch)
        patch_set = set(patch["faces"])
        all_removed.update(patch_set)

        has_up = bool(patch.get("up_faces"))
        has_down = bool(patch.get("down_faces"))
        if has_up and has_down:
            # Sandwich with both sides → always stitch (mode B).
            # Fold faces within the patch don't disqualify it —
            # they just mark local non-manifold edges that get
            # removed along with the rest.
            mode_b_faces.update(patch_set)
            n_b += 1
            if fold:
                mode_a_faces.update(fold)
        elif fold:
            mode_a_faces.update(patch_set)
            n_a += 1
        else:
            mode_b_faces.update(patch_set)
            n_b += 1

    # ── Step 2: remove all patch faces ────────────────────────────
    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[list(all_removed)] = False
    result = faces.copy()
    result[~keep_mask] = 0

    # ── Step 3: find NEWLY disconnected components ─────────────────
    # Only consider faces that were in the main component before
    # removal but ended up in a non-main component after.
    labels_before, main_before = face_edge_components(faces, n_vertices)
    was_main = set(int(i) for i in np.where(labels_before == main_before)[0])

    labels_after, main_after = face_edge_components(result, n_vertices)
    degen = np.all(result == 0, axis=1)

    # Faces that were main, survived removal, but are now non-main
    newly_disconnected: set[int] = set()
    for fi in was_main:
        if not degen[fi] and labels_after[fi] != main_after:
            newly_disconnected.add(fi)

    # ── Step 4: classify new orphan components ────────────────────
    # Only applies to *newly* disconnected pieces (those that were in
    # main_before but are non-main_after).  Mode B pre-existing
    # disconnects are handled by the stitch step unconditionally.
    if newly_disconnected:
        # Build edge→type map for removed faces
        removed_edge_type: dict[tuple, str] = {}
        for fi in all_removed:
            f = faces[fi]
            typ = "a" if fi in mode_a_faces else "b"
            for i in range(3):
                e = (
                    min(int(f[i]), int(f[(i + 1) % 3])),
                    max(int(f[i]), int(f[(i + 1) % 3])),
                )
                # If any bordering removed face is Mode B, mark as "b"
                if removed_edge_type.get(e) != "b":
                    removed_edge_type[e] = typ

        # Group newly disconnected faces by component
        new_comp_ids = set(int(labels_after[fi]) for fi in newly_disconnected)
        orphan_remove: set[int] = set()

        for comp_id in new_comp_ids:
            comp_face_idxs = [
                fi
                for fi in np.where(labels_after == comp_id)[0]
                if fi in newly_disconnected or not degen[fi]
            ]
            borders_mode_b = False

            for fi in comp_face_idxs:
                if borders_mode_b:
                    break
                f = faces[fi]
                for i in range(3):
                    e = (
                        min(int(f[i]), int(f[(i + 1) % 3])),
                        max(int(f[i]), int(f[(i + 1) % 3])),
                    )
                    if removed_edge_type.get(e) == "b":
                        borders_mode_b = True
                        break

            if not borders_mode_b:
                orphan_remove.update(int(fi) for fi in comp_face_idxs)

        if orphan_remove:
            keep_mask2 = np.ones(len(faces), dtype=bool)
            keep_mask2[list(all_removed)] = False
            keep_mask2[list(orphan_remove)] = False
            result = faces.copy()
            result[~keep_mask2] = 0
    n_orphan = len(orphan_remove) if newly_disconnected else 0

    # ── Step 5: stitch Mode B gaps ───────────────────────────────
    n_before = len(result)
    if mode_b_faces:
        result = stitch_mode_b(verts, result, faces, mode_b_faces, patches)
    n_stitched = len(result) - n_before
    return result, len(all_removed), n_a, n_b, n_orphan, n_stitched


def build_edge_to_faces(faces, mask) -> dict[tuple[int, int], list[int]]:
    """Edge→face map for faces where *mask* is True."""
    fi_idx = np.where(mask)[0]
    gf = faces[fi_idx]
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for col_a, col_b in ((0, 1), (1, 2), (0, 2)):
        va = gf[:, col_a].astype(np.intp)
        vb = gf[:, col_b].astype(np.intp)
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        for k in range(len(fi_idx)):
            result[(int(lo[k]), int(hi[k]))].append(int(fi_idx[k]))
    return result


#: What :func:`break_up_classify` makes of a piece.
PIECE_CANDIDATE, PIECE_TRAPPED, PIECE_SOMA = 0, 1, 2


def break_up_classify(
    faces, n_vertices: int, usable, organelle, soma_face, claimed, has_soma: bool
) -> tuple[list[np.ndarray], np.ndarray]:
    """The pieces ``skeliner.pre.break_up_mesh`` classifies: components of
    the usable faces that are neither soma nor organelle, largest first, and
    the kind of each — :data:`PIECE_TRAPPED` when only organelle faces
    connect it to its mesh component's body, :data:`PIECE_SOMA` when more
    than half its outer edges meet soma faces (only with a soma), else
    :data:`PIECE_CANDIDATE`.  Claimed pieces are never trapped or soma."""
    faces, _ = checked_faces(faces, n_vertices)
    nF = len(faces)
    flags = {}
    for name, a in [
        ("usable", usable),
        ("organelle", organelle),
        ("soma_face", soma_face),
        ("claimed", claimed),
    ]:
        flags[name] = np.asarray(a, dtype=bool)
        if len(flags[name]) != nF:
            raise ValueError(f"{name} needs one value per face")
    usable, organelle = flags["usable"], flags["organelle"]
    soma_face, claimed_face = flags["soma_face"], flags["claimed"]

    # --- Phase 1: reachability without crossing organelles ---
    # For each mesh component, find its largest non-organelle body.
    # Faces not in their component's body are trapped by organelles.
    # This is per-component so disconnected neurite fragments are
    # handled correctly (each has its own reachable body).
    non_org = usable & ~organelle
    non_org_fi = np.where(non_org)[0]
    non_org_comps = face_components(faces, non_org_fi)

    # Label every non-org face with its non-org component id.
    # The largest non-org component within each mesh component is
    # that component's "body".
    non_org_label = np.full(nF, -1, dtype=np.intp)
    for ci, comp in enumerate(non_org_comps):
        non_org_label[comp] = ci

    # For each mesh component, find its largest non-org sub-component.
    # Use usable-face edge components as the mesh components.
    usable_fi = np.where(usable)[0]
    mesh_comps = face_components(faces, usable_fi)

    body_labels: set[int] = set()
    for mc in mesh_comps:
        # non-org labels present in this mesh component
        labels_in_mc = non_org_label[mc]
        labels_in_mc = labels_in_mc[labels_in_mc >= 0]
        if len(labels_in_mc) == 0:
            continue
        unique, counts = np.unique(labels_in_mc, return_counts=True)
        body_labels.add(int(unique[counts.argmax()]))

    reachable = np.zeros(nF, dtype=bool)
    for ci, comp in enumerate(non_org_comps):
        if ci in body_labels:
            reachable[comp] = True

    # --- Phase 2: break at soma + organelles ---
    remain = usable & ~soma_face & ~organelle
    remain_fi = np.where(remain)[0]

    if len(remain_fi) == 0:
        return [], np.zeros(0, dtype=np.uint8)

    components = face_components(faces, remain_fi)
    # Still needed below, to count how much of each component's boundary
    # is soma.  That is the only remaining use of the edge map.
    ef_all = build_edge_to_faces(faces, usable)

    kinds = np.zeros(len(components), dtype=np.uint8)
    for k, comp in enumerate(components):
        claimed_piece = bool(claimed_face[comp].any())

        # Reachability: is any face in its mesh component's body?
        in_main = reachable[comp].any()

        if not in_main and not claimed_piece:
            kinds[k] = PIECE_TRAPPED
            continue

        # Boundary analysis for soma detection (skipped without a soma)
        if has_soma and not claimed_piece:
            comp_set = set(comp.tolist())
            n_soma = 0
            n_total = 0
            for fi in comp:
                face = faces[fi]
                for i in range(3):
                    a, b = int(face[i]), int(face[(i + 1) % 3])
                    e = (min(a, b), max(a, b))
                    for nb in ef_all[e]:
                        if nb not in comp_set:
                            n_total += 1
                            if soma_face[nb]:
                                n_soma += 1

            soma_frac = n_soma / n_total if n_total > 0 else 0.0
            if soma_frac > 0.5:
                kinds[k] = PIECE_SOMA
    return components, kinds


def otsu_threshold(values: np.ndarray) -> tuple[float, float]:
    """Otsu's method on continuous 1-D data.

    Returns ``(threshold, separability)`` where *separability* is the
    ratio of between-class variance to total variance (0 = no split,
    1 = perfect bimodal separation).
    """
    x = np.sort(values)
    n = len(x)
    if n < 2:
        return float(x[0]), 0.0

    total_var = float(np.var(x))
    if total_var < 1e-12:
        return float(x[0]), 0.0

    best_thresh = float(x[0])
    best_between = 0.0

    # Scan every midpoint between consecutive unique values
    for i in range(1, n):
        if x[i] == x[i - 1]:
            continue
        left, right = x[:i], x[i:]
        w0, w1 = i / n, (n - i) / n
        between = w0 * w1 * (left.mean() - right.mean()) ** 2
        if between > best_between:
            best_between = between
            best_thresh = float(x[i - 1] + x[i]) / 2.0

    return best_thresh, best_between / total_var


class _Fitted:
    """The soma ellipsoid as a fit returned it: fit number *k*, its centre,
    the map *w* to the unit sphere (as ``Soma._W``), its smallest axis, and
    the vertices it was fitted to (None for the first fit)."""

    def __init__(self, k: int, center, w, axes_min: float, verts):
        self.k = k
        self.center = np.asarray(center, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)
        self.axes_min = float(axes_min)
        self.verts = verts

    def body(self, x) -> np.ndarray:
        """Body coordinates, where the ellipsoid is the unit sphere, as
        ``Soma._body_coords`` computes them."""
        return (np.asarray(x, dtype=np.float64) - self.center) @ self.w

    def contains(self, x) -> np.ndarray:
        """Which points lie inside or on the ellipsoid, as ``Soma.contains``."""
        return (self.body(x) ** 2).sum(axis=-1) <= 1.0


def _assign_soma_verts(
    faces: np.ndarray,
    verts: np.ndarray,
    avg_edge: float,
    soma: _Fitted,
    main_fi: np.ndarray,
    adj: dict,
    refit,
) -> _Fitted:
    """Assign the vertices of the component *main_fi* (vertex adjacency
    *adj*) to the ellipsoid *soma*: containment, edge-dilation, pocket
    absorption, and disconnected-component absorption; then
    ``refit(soma, ids)`` the ellipsoid to the final vertex set."""
    # Containment: main-component vertices inside the ellipsoid
    all_main_verts = np.unique(faces[main_fi])
    inside = soma.contains(verts[all_main_verts])
    soma_set: set[int] = set(all_main_verts[inside].tolist())

    # Dilate along mesh edges (a few average edge lengths past the
    # ellipsoid surface, measured in body-coord units).
    edge_in_body = avg_edge / soma.axes_min
    dilation_limit = 1.0 + 3.0 * edge_in_body

    prev_size = 0
    while len(soma_set) != prev_size:
        prev_size = len(soma_set)
        frontier = set()
        for v in soma_set:
            for nv in adj.get(v, []):
                if nv not in soma_set:
                    frontier.add(nv)
        if not frontier:
            break
        frontier_arr = np.fromiter(frontier, dtype=np.intp)
        body = soma.body(verts[frontier_arr])
        body_dist = np.sqrt((body**2).sum(axis=1))
        accept = frontier_arr[body_dist < dilation_limit]
        if len(accept) == 0:
            break
        soma_set.update(accept.tolist())

    # Absorb pockets: connected components of non-soma verts that are
    # topologically trapped (only reachable through the soma).
    # A neuron can have multiple neurite trees stemming from the soma;
    # all of them are trapped by definition.  We separate neurites
    # (large, preserve) from pockets (small surface artifacts, absorb)
    # using Otsu on trapped component sizes.
    all_main_set = set(all_main_verts.tolist())
    for _iteration in range(10):
        outside = all_main_set - soma_set
        visited: set[int] = set()
        trapped_comps: list[list[int]] = []
        for start in outside:
            if start in visited:
                continue
            comp: list[int] = []
            pocket_queue = deque([start])
            while pocket_queue:
                v = pocket_queue.popleft()
                if v in visited:
                    continue
                visited.add(v)
                comp.append(v)
                for nv in adj[v]:
                    if nv in outside and nv not in visited:
                        pocket_queue.append(nv)
            comp_set = set(comp)
            trapped = all(
                nv in soma_set or nv in comp_set for v in comp for nv in adj.get(v, [])
            )
            if trapped and len(comp) < len(soma_set):
                trapped_comps.append(comp)

        if not trapped_comps:
            break

        # Otsu on component sizes to split neurites from pockets
        sizes = np.array([len(c) for c in trapped_comps], dtype=np.float64)
        if len(sizes) >= 2:
            size_thresh, _ = otsu_threshold(sizes)
        else:
            size_thresh = 0.0  # single component — absorb it

        absorbed = 0
        for comp in trapped_comps:
            if len(comp) <= size_thresh:
                soma_set.update(comp)
                absorbed += len(comp)
        if absorbed == 0:
            break

    # Absorb disconnected-component vertices inside the ellipsoid.
    all_verts = np.arange(len(verts))
    non_main_mask = np.ones(len(verts), dtype=bool)
    non_main_mask[all_main_verts] = False
    non_main_verts = all_verts[non_main_mask]
    if non_main_verts.size:
        inside_non_main = soma.contains(verts[non_main_verts])
        soma_set.update(non_main_verts[inside_non_main].tolist())

    # Refit ellipsoid to the final vertex set.
    return refit(soma, np.fromiter(sorted(soma_set), dtype=np.intp))


def find_soma_ring_cutoff(
    faces,
    vertices,
    labels,
    main: int,
    seed_vi,
    center,
    z_span: float,
    avg_edge: float,
    organelle,
    fit,
) -> tuple[int, np.ndarray, int, int]:
    """The soma grown from the nucleus at *center* (*z_span* tall, seeded
    from the vertices *seed_vi*): ``(the fit that stands, the soma's
    vertices, the cutoff ring, the last ring)``.

    BFS rings from the seeds over the soma's component, cut where the
    ring's largest connected piece shrinks (Otsu on the sizes after the
    peak), give the vertices of a first ellipsoid fit; the vertices are
    then assigned to it, neurite stubs pruned, small fragments dropped and
    organelle-only vertices left out, the ellipsoid refitted after each.
    ``fit(ids, refit)`` fits the soma to the vertices *ids* (ascending)
    and returns ``(center, W, smallest axis)``, ``W`` the map that takes the
    soma to the unit sphere, or None when a refit fails; fits that return
    something are counted from 0.
    """
    verts = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(verts))
    labels = np.asarray(labels, dtype=np.int64)
    organelle = np.asarray(organelle, dtype=bool)
    if len(labels) != len(faces) or len(organelle) != len(faces):
        raise ValueError("labels and organelle need one value per face")
    seed_vi = np.asarray(seed_vi, dtype=np.int64)
    bad = (seed_vi < 0) | (seed_vi >= len(verts))
    if bad.any():
        raise ValueError(f"vertex index {seed_vi[bad][0]} is out of range")
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("center needs three coordinates")
    n_fits = [0]

    def _fit(ids, refit: bool) -> _Fitted | None:
        out = fit(np.asarray(ids, dtype=np.int64), refit)
        if out is None:
            return None
        k, n_fits[0] = n_fits[0], n_fits[0] + 1
        c, w, axes_min = out
        return _Fitted(k, c, w, axes_min, ids if refit else None)

    def _refit(soma: _Fitted, sv: np.ndarray) -> _Fitted:
        # a failed refit keeps the ellipsoid and takes the vertices
        if len(sv) >= 4:
            new = _fit(sv, True)
            if new is not None:
                return new
        soma.verts = sv
        return soma

    # ── 2. Build soma-component vertex adjacency ─────────────────
    #       The soma is on the component that contains the nucleus
    #       seeds.  This may differ from the largest component when
    #       the soma is disconnected from the main neurite surface
    #       (a gap that remove_gaps would bridge later).
    faces_arr = faces

    # Determine which component the nucleus seeds belong to
    soma_ci = main
    if len(seed_vi) > 0:
        vert_comp = np.full(len(verts), -1, dtype=np.intp)
        for fi in range(len(faces)):
            ci = int(labels[fi])
            for v in faces_arr[fi]:
                vert_comp[int(v)] = ci
        seed_comps = vert_comp[seed_vi]
        valid = seed_comps >= 0
        if valid.any():
            soma_ci = Counter(seed_comps[valid].tolist()).most_common(1)[0][0]

    soma_fi = np.where(labels == soma_ci)[0]
    adj: dict[int, list[int]] = defaultdict(list)
    for fi in soma_fi:
        v = faces_arr[fi]
        for i in range(3):
            a, b = int(v[i]), int(v[(i + 1) % 3])
            adj[a].append(b)
            adj[b].append(a)

    # ── 3. BFS from soma surface at nucleus mid-Z ──────────────
    #       Seed ring 0 with soma-cluster vertices at the nucleus
    #       center Z-level (from find_nucleus_center).  These are
    #       outer-surface vertices, tilt-invariant, and form a
    #       symmetric band around the soma cross-section.
    all_soma_verts = np.fromiter(adj.keys(), dtype=np.intp)
    soma_set = set(adj.keys())
    # Keep only seeds that are on the soma component
    seed_verts = np.array([v for v in seed_vi if v in soma_set], dtype=np.intp)
    if len(seed_verts) == 0:
        # Fallback: single nearest vertex to nucleus center
        seed_verts = all_soma_verts[
            np.argmin(np.linalg.norm(verts[all_soma_verts] - center, axis=1))
        ].reshape(1)

    ring_level: dict[int, int] = {}
    queue: deque[int] = deque()
    ring_verts: dict[int, list[int]] = defaultdict(list)
    for sv in seed_verts:
        vi = int(sv)
        ring_level[vi] = 0
        queue.append(vi)
        ring_verts[0].append(vi)
    while queue:
        v = queue.popleft()
        lv = ring_level[v]
        for nv in adj[v]:
            if nv not in ring_level:
                ring_level[nv] = lv + 1
                queue.append(nv)
                ring_verts[lv + 1].append(nv)

    # ── 4. Find soma boundary: largest connected ring component ────
    #       Each BFS ring is split into connected components on the
    #       mesh surface.  On the soma the largest component dominates
    #       and grows; once the frontier enters neurites it fragments
    #       into narrow strips and the largest component shrinks.
    #       The cutoff is where the largest component peaks (Otsu on
    #       the post-peak sizes determines the boundary).
    max_ring = max(ring_verts.keys())
    largest_comp_size = np.zeros(max_ring + 1)

    for lv in range(max_ring + 1):
        verts_in_ring = ring_verts.get(lv, [])
        if not verts_in_ring:
            continue
        # Find connected components within this ring using mesh adjacency
        ring_set = set(verts_in_ring)
        visited_ring: set[int] = set()
        max_comp = 0
        for start in verts_in_ring:
            if start in visited_ring:
                continue
            size = 0
            rq = deque([start])
            while rq:
                v = rq.popleft()
                if v in visited_ring:
                    continue
                visited_ring.add(v)
                size += 1
                for nv in adj[v]:
                    if nv in ring_set and nv not in visited_ring:
                        rq.append(nv)
            if size > max_comp:
                max_comp = size
        largest_comp_size[lv] = max_comp

    # Skip ring 0 (the injected seed set, not a natural BFS ring).
    # Limit peak search to rings within 3× the nucleus Z-span
    # (converted to ring count via average edge length) — peaks
    # beyond that are neurite artifacts, not the soma.
    max_soma_rings = max(int(z_span * 3 / avg_edge), 10)
    search_end = min(max_soma_rings + 1, max_ring + 1)
    peak_ring = 1 + int(np.argmax(largest_comp_size[1:search_end]))
    post_peak = largest_comp_size[peak_ring:]
    if len(post_peak) > 1:
        count_thresh, _ = otsu_threshold(post_peak)
    else:
        count_thresh = 0.0

    cutoff = max_ring
    for lv in range(peak_ring, max_ring + 1):
        if largest_comp_size[lv] < count_thresh:
            cutoff = lv
            break

    # Under-detection guard: the soma must at least span the nucleus
    # Z-extent.  If it doesn't, extend the cutoff.
    min_cutoff = max(int(z_span / avg_edge), 1)
    if cutoff < min_cutoff:
        cutoff = min_cutoff

    # ── 5. Fit ellipsoid from BFS ring vertices ────────────────────
    #       Include ring 0 (nucleus seed) in the soma vertex set.
    bfs_set: set[int] = set()
    for lv in range(cutoff + 1):
        bfs_set.update(ring_verts[lv])
    bfs_verts_arr = np.fromiter(sorted(bfs_set), dtype=np.intp)

    initial_soma = _fit(bfs_verts_arr, False)

    soma = _assign_soma_verts(
        faces, verts, avg_edge, initial_soma, soma_fi, adj, _refit
    )

    # ── 6. Prune neurite tubes from absorbed soma ─────────────────
    #       Pocket absorption may swallow neurite stubs.  Find
    #       connected components of soma vertices outside the initial
    #       (pre-absorption) ellipsoid.  A component is a neurite
    #       tube if it has a narrow attachment to the soma body
    #       (ratio < 0.2) and is elongated (PCA: λ1 >> λ2 ≈ λ3).
    #       After pruning tubes, drop any fragments that become
    #       disconnected from the main soma body when the tube
    #       boundary (soma verts adjacent to pruned verts) is
    #       temporarily removed — these are axon sections inside
    #       the ellipsoid connected only through the pruned tube.
    soma_set = set(soma.verts.tolist())
    soma_arr = np.fromiter(soma_set, dtype=np.intp)

    if len(soma_arr) >= 4:
        body_dist = np.sqrt((initial_soma.body(verts[soma_arr]) ** 2).sum(axis=1))
        inside_set: set[int] = set()
        outside_set: set[int] = set()
        for i in range(len(soma_arr)):
            vi = int(soma_arr[i])
            (inside_set if body_dist[i] <= 1.0 else outside_set).add(vi)

        neurite_prune: set[int] = set()
        if outside_set:
            # ── 6a. Prune neurite stubs ──────────────────────────
            #        A stub connects the soma body to a neurite tree.
            #        Identify stubs as outside components that border
            #        a large non-soma connected component (the neurite
            #        tree).  Fins protrude into extracellular space
            #        and do NOT border a large non-soma CC.
            #        "Large" is determined by Otsu on non-soma CC
            #        sizes — fully data-driven, no hard thresholds.

            # Find non-soma CCs on the soma component
            all_comp_set = set(np.unique(faces[soma_fi]).tolist())
            non_soma = all_comp_set - soma_set
            vis_ns: set[int] = set()
            neurite_tree: set[int] = set()
            ns_sizes: list[int] = []
            ns_ccs: list[list[int]] = []
            for ns_start in non_soma:
                if ns_start in vis_ns:
                    continue
                cc: list[int] = []
                nsq = deque([ns_start])
                while nsq:
                    v = nsq.popleft()
                    if v in vis_ns:
                        continue
                    vis_ns.add(v)
                    cc.append(v)
                    for nv in adj.get(v, []):
                        if nv in non_soma and nv not in vis_ns:
                            nsq.append(nv)
                ns_ccs.append(cc)
                ns_sizes.append(len(cc))

            if len(ns_sizes) >= 2:
                ns_thresh, _ = otsu_threshold(np.array(ns_sizes, dtype=np.float64))
                for cc, sz in zip(ns_ccs, ns_sizes):
                    if sz > ns_thresh:
                        neurite_tree.update(cc)
            elif len(ns_sizes) == 1:
                neurite_tree.update(ns_ccs[0])

            # Find outside components; collect features
            n_tubes = 0
            outside_comps: list[tuple[list[int], bool, float]] = []
            if neurite_tree:
                visited_out: set[int] = set()
                for start in outside_set:
                    if start in visited_out:
                        continue
                    comp: list[int] = []
                    oq = deque([start])
                    while oq:
                        v = oq.popleft()
                        if v in visited_out:
                            continue
                        visited_out.add(v)
                        comp.append(v)
                        for nv in adj.get(v, []):
                            if nv in outside_set and nv not in visited_out:
                                oq.append(nv)

                    if len(comp) < 20:
                        continue

                    # Does this component border the neurite tree?
                    borders = any(
                        nv in neurite_tree for v in comp for nv in adj.get(v, [])
                    )
                    # How far does it extend in body coords?
                    comp_body = initial_soma.body(verts[np.array(comp)])
                    max_ext = float(np.sqrt((comp_body**2).sum(axis=1)).max())
                    outside_comps.append((comp, borders, max_ext))

                # Prune components extending far beyond the typical
                # near-surface bump.  Threshold = 2× the 25th
                # percentile of extents — data-relative, adapts to
                # ellipsoid fit quality.
                all_ext = np.array(
                    [e for _, _, e in outside_comps],
                    dtype=np.float64,
                )
                q1 = float(np.percentile(all_ext, 25))
                ext_thresh = 2.0 * q1

                for comp, borders, ext in outside_comps:
                    if ext > ext_thresh:
                        neurite_prune.update(comp)
                        n_tubes += 1

            if neurite_prune:
                soma_set -= neurite_prune

        # ── 6b. Per-exit stub erosion ─────────────────────────────
        #        For each neurite exit (cluster of boundary verts),
        #        BFS inward into the soma independently.  Ring widths
        #        start narrow (tube); stop when the ring widens into
        #        the soma body (Otsu).  Erode each exit separately so
        #        narrow tubes get deep erosion, wide junctions don't.
        comp_vert_set = set(adj.keys())
        non_comp_soma = soma_set - comp_vert_set
        all_comp_set = set(np.unique(faces[soma_fi]).tolist())
        non_soma = all_comp_set - soma_set

        # Find soma boundary verts adjacent to non-soma.
        # Only erode when step 6a pruned neurite chains.
        exit_verts: set[int] = set()
        if neurite_prune and non_soma:
            for v in soma_set & comp_vert_set:
                for nv in adj.get(v, []):
                    if nv in non_soma:
                        exit_verts.add(v)
                        break

        if exit_verts:
            soma_main = soma_set & comp_vert_set

            # Group exit verts into per-exit clusters
            vis_ex: set[int] = set()
            exit_clusters: list[set[int]] = []
            for ex_start in sorted(exit_verts):
                if ex_start in vis_ex:
                    continue
                cl: list[int] = []
                exq = deque([ex_start])
                while exq:
                    v = exq.popleft()
                    if v in vis_ex:
                        continue
                    vis_ex.add(v)
                    cl.append(v)
                    for nv in adj.get(v, []):
                        if nv in exit_verts and nv not in vis_ex:
                            exq.append(nv)
                exit_clusters.append(set(cl))

            # Erode each exit independently
            all_stub: set[int] = set()
            n_exits_eroded = 0
            global_visited: set[int] = set()

            for cluster in exit_clusters:
                visited_er = set(cluster)
                stub_verts = set(cluster)
                current_ring = cluster
                ring_sizes: list[int] = [len(current_ring)]

                for _depth in range(200):
                    next_ring: set[int] = set()
                    for v in current_ring:
                        for nv in adj.get(v, []):
                            if (
                                nv in soma_main
                                and nv not in visited_er
                                and nv not in global_visited
                            ):
                                next_ring.add(nv)
                                visited_er.add(nv)
                    if not next_ring:
                        break

                    # Stop at real branch (2+ CCs each ≥ 25%)
                    # or when ring doubles from entry (soma body).
                    if len(next_ring) > 2 * ring_sizes[0]:
                        break
                    vis_r: set[int] = set()
                    cc_sizes_r: list[int] = []
                    for rs in next_ring:
                        if rs in vis_r:
                            continue
                        sz = 0
                        rq = deque([rs])
                        while rq:
                            u = rq.popleft()
                            if u in vis_r:
                                continue
                            vis_r.add(u)
                            sz += 1
                            for nu in adj.get(u, []):
                                if nu in next_ring and nu not in vis_r:
                                    rq.append(nu)
                        cc_sizes_r.append(sz)
                    quarter = len(next_ring) * 0.25
                    if sum(1 for s in cc_sizes_r if s >= quarter) >= 2:
                        break

                    stub_verts.update(next_ring)
                    current_ring = next_ring
                    ring_sizes.append(len(next_ring))

                if next_ring:  # actually eroded
                    all_stub.update(stub_verts)
                    global_visited.update(stub_verts)
                    n_exits_eroded += 1

            if all_stub and len(all_stub) < len(soma_main) * 0.5:
                soma_set -= all_stub
                soma_set |= non_comp_soma

        # ── 6c. Drop small disconnected soma fragments ─────────────
        #        Axon sections inside the ellipsoid may form small
        #        CCs disconnected from the main soma body.
        main_soma_final = soma_set & comp_vert_set
        vis_final: set[int] = set()
        largest_final: list[int] = []
        for cc_start in sorted(main_soma_final):
            if cc_start in vis_final:
                continue
            cc: list[int] = []
            ccq = deque([cc_start])
            while ccq:
                v = ccq.popleft()
                if v in vis_final:
                    continue
                vis_final.add(v)
                cc.append(v)
                for nv in adj.get(v, []):
                    if nv in main_soma_final and nv not in vis_final:
                        ccq.append(nv)
            if len(cc) > len(largest_final):
                largest_final = cc

        n_frag = len(main_soma_final) - len(largest_final)
        if n_frag > 0:
            soma_set = set(largest_final) | non_comp_soma

        # Refit ellipsoid to final pruned vertex set
        if len(soma_set) != len(soma.verts):
            soma = _refit(soma, np.fromiter(sorted(soma_set), dtype=np.intp))

    # ── 7. Exclude organelle vertices from final soma ───────────
    #       Only remove vertices that appear exclusively on organelle
    #       faces.  Mouth vertices (shared between organelle and
    #       non-organelle faces) must stay in the soma.
    if organelle.any():
        org_verts = set(np.unique(faces[organelle]).tolist())
        non_org_verts = set(np.unique(faces[~organelle]).tolist())
        exclusive_org = org_verts - non_org_verts
        soma_set = set(soma.verts.tolist()) - exclusive_org
        n_removed = len(soma.verts) - len(soma_set)
        if n_removed > 0:
            soma = _refit(soma, np.fromiter(sorted(soma_set), dtype=np.intp))

    return soma.k, np.asarray(soma.verts, dtype=np.int64), cutoff, max_ring


def z_slice_cluster(
    verts_xy: np.ndarray,
    grid_res: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Rasterize 2D points, find the largest connected cluster.

    Returns ``(occ, soma_region, ix, iy, xy_min)`` or None if too few
    points.  *occ* is the raw occupancy grid, *soma_region* is the
    dilated largest-component mask, *ix/iy* are per-vertex grid indices,
    *xy_min* is the grid origin.
    """
    from scipy.ndimage import binary_dilation, label

    pts = verts_xy
    if len(pts) < 10:
        return None

    xy_min = pts.min(axis=0) - grid_res * 3
    nx = int((pts[:, 0].max() - xy_min[0] + grid_res * 6) / grid_res) + 1
    ny = int((pts[:, 1].max() - xy_min[1] + grid_res * 6) / grid_res) + 1

    occ = np.zeros((nx, ny), dtype=bool)
    ix = ((pts[:, 0] - xy_min[0]) / grid_res).astype(int).clip(0, nx - 1)
    iy = ((pts[:, 1] - xy_min[1]) / grid_res).astype(int).clip(0, ny - 1)
    occ[ix, iy] = True

    struct3 = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(occ, struct3)
    labeled, n_labels = label(dilated)
    if n_labels == 0:
        return None

    sizes = np.bincount(labeled.ravel())[1:]
    lid = int(np.argmax(sizes)) + 1
    soma_region = labeled == lid

    return occ, soma_region, ix, iy, xy_min


def z_slice_void(
    occ: np.ndarray,
    soma_region: np.ndarray,
    xy_min: np.ndarray,
    grid_res: float,
) -> tuple[float, float, float, np.ndarray | None] | None:
    """Detect the nucleus void within a soma cluster.

    Returns ``(cx, cy, void_r, void_xy)`` or None if no void.
    *void_xy* is ``(M, 2)`` world-coordinate points of the void region.
    """
    from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

    occ_s = occ & soma_region

    # 4-ray enclosure
    enc = np.ones_like(occ_s)
    for ax in range(2):
        enc &= np.maximum.accumulate(occ_s, axis=ax)
        enc &= np.flip(
            np.maximum.accumulate(np.flip(occ_s, axis=ax), axis=ax),
            axis=ax,
        )
    # Confine to filled soma cluster (can't leak through pocket mouth)
    soma_filled = binary_fill_holes(soma_region)
    enc &= soma_filled

    void = enc & ~occ_s
    if not void.any():
        return None

    dt = distance_transform_edt(~occ_s)
    dtv = dt * void
    pk = np.unravel_index(dtv.argmax(), dtv.shape)
    cx = float(xy_min[0] + pk[0] * grid_res + grid_res / 2)
    cy = float(xy_min[1] + pk[1] * grid_res + grid_res / 2)
    void_r = float(dtv.max()) * grid_res

    # Keep only the connected component containing the peak,
    # with dt > 2 to exclude small vertex-spacing gaps.
    deep_void = void & (dt > 2)
    if not deep_void.any():
        return cx, cy, void_r, None

    void_labeled, _ = label(deep_void)
    peak_label = void_labeled[pk[0], pk[1]]
    if peak_label == 0:
        deep_ij = np.argwhere(deep_void)
        dists = np.abs(deep_ij[:, 0] - pk[0]) + np.abs(deep_ij[:, 1] - pk[1])
        peak_label = void_labeled[
            deep_ij[np.argmin(dists)][0], deep_ij[np.argmin(dists)][1]
        ]
    nucleus_void = void_labeled == peak_label

    void_ij = np.argwhere(nucleus_void)
    void_xy = np.column_stack(
        [
            xy_min[0] + void_ij[:, 0] * grid_res + grid_res / 2,
            xy_min[1] + void_ij[:, 1] * grid_res + grid_res / 2,
        ]
    )
    return cx, cy, void_r, void_xy


class ZScan:
    """The Z-level scan of a mesh's vertices that finds the nucleus: per
    level, the largest cluster of the vertices near it and the void inside
    that cluster, then the longest chain of levels whose voids line up.

    Construction runs the first pass, each level on its own, and finds the
    chain; :meth:`retry` runs the second, on the levels next to the chain
    whose soma ring has a gap that breaks the void test, inside the convex
    hull of the chain end's soma cluster.
    """

    def __init__(
        self,
        vertices,
        grid_res: float,
        z_tol: float,
        min_void_r: float,
        max_shift: float,
    ):
        from scipy.ndimage import binary_dilation, label

        verts = rows3("V", vertices)
        self._verts = verts
        self._grid_res = grid_res
        self._min_void_r = min_void_r
        self._max_shift = max_shift

        z_unique = np.unique(verts[:, 2])
        z_step = 210.0
        z_levels = np.arange(z_unique.min() + z_step, z_unique.max() - z_step, z_step)

        all_vi = np.arange(len(verts))
        # Each entry: (z, cx, cy, void_r, void_xy, vert_indices, soma_mask)
        #   vert_indices: indices into the vertices near this Z
        #   soma_mask: bool array over vert_indices, True = in soma cluster
        raw: list[tuple] = []
        # the grid data, for pass 2: (occ, ix, iy, xy_min, labeled)
        grid_data: list[tuple | None] = []

        for z in z_levels:
            near_z = np.abs(verts[:, 2] - z) < z_tol
            vi = all_vi[near_z]
            pts = verts[vi, :2]

            cluster = z_slice_cluster(pts, grid_res)
            if cluster is None:
                raw.append((z, np.nan, np.nan, 0.0, None, vi, None))
                grid_data.append(None)
                continue

            occ, soma_region, ix, iy, xy_min = cluster
            soma_mask = soma_region[ix, iy]

            # Store labeled grid for pass 2
            struct3 = np.ones((3, 3), dtype=bool)
            dilated = binary_dilation(occ, struct3)
            labeled, _ = label(dilated)
            grid_data.append((occ, ix, iy, xy_min, labeled))

            void_result = z_slice_void(occ, soma_region, xy_min, grid_res)
            if void_result is None:
                raw.append((z, np.nan, np.nan, 0.0, None, vi, soma_mask))
            else:
                cx, cy, void_r, void_xy = void_result
                raw.append((z, cx, cy, void_r, void_xy, vi, soma_mask))

        self._raw = raw
        self._grid_data = grid_data
        self.best = self._chain()

    def _chain(self) -> list[int] | None:
        """The longest run of two or more consecutive levels with a void of
        at least ``min_void_r``, each centred within ``max_shift`` of the
        one before; None when there is none."""
        raw = self._raw
        runs: list[list[int]] = []
        cur: list[int] = []
        for i, entry in enumerate(raw):
            cx, cy, r = entry[1], entry[2], entry[3]
            if r < self._min_void_r or np.isnan(cx):
                if cur:
                    runs.append(cur)
                    cur = []
                continue
            if not cur:
                cur = [i]
            else:
                pcx, pcy = raw[cur[-1]][1], raw[cur[-1]][2]
                if np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2) < self._max_shift:
                    cur.append(i)
                else:
                    runs.append(cur)
                    cur = [i]
        if cur:
            runs.append(cur)

        runs = [r for r in runs if len(r) >= 2]
        return max(runs, key=len) if runs else None

    def chain_ends(self) -> tuple[int, int] | None:
        """The chain's first and last level, or None."""
        return None if self.best is None else (self.best[0], self.best[-1])

    def soma_vertices(self, i: int) -> np.ndarray | None:
        """Vertex ids of level *i*'s soma cluster, or None if it has none."""
        if i >= len(self._raw):
            raise ValueError(f"no level {i}")
        vi, mask = self._raw[i][5], self._raw[i][6]
        return None if mask is None else vi[mask]

    def retry(self, hull_lo=None, hull_hi=None) -> None:
        """Pass 2: hull-guided void detection for the broken levels next to
        the chain, below it inside *hull_lo* and above it inside *hull_hi*
        (scipy ``ConvexHull.equations`` of the chain end's soma cluster; a
        missing hull skips that side), then the chain again.

        When the soma ring has a gap (surface hole), the 4-ray enclosure and
        binary_fill_holes both fail; the hull from the nearest good level
        defines "inside" instead.
        """
        from scipy.ndimage import distance_transform_edt, label

        if self.best is None:
            return
        raw, grid_data, grid_res = self._raw, self._grid_data, self._grid_res
        best = self.best
        best_set = set(best)
        chain_lo, chain_hi = best[0], best[-1]

        # Scan levels adjacent to the chain in both directions
        retry_indices = []
        for i in range(chain_lo - 1, -1, -1):
            if grid_data[i] is None:
                break
            retry_indices.append(i)
        for i in range(chain_hi + 1, len(raw)):
            if grid_data[i] is None:
                break
            retry_indices.append(i)

        for i in retry_indices:
            if i in best_set:
                continue
            r_existing = raw[i][3]
            if r_existing >= self._min_void_r and not np.isnan(raw[i][1]):
                continue  # already good

            gd = grid_data[i]
            if gd is None:
                continue
            occ, ix, iy, xy_min, labeled = gd

            # the hull of the nearest end of the chain
            equations = hull_lo if i < chain_lo else hull_hi
            if equations is None:
                continue

            # Build hull mask on this level's grid: for each grid cell,
            # test if its centre falls inside the reference convex hull.
            nx, ny = occ.shape
            gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            gx = xy_min[0] + gi * grid_res + grid_res / 2
            gy = xy_min[1] + gj * grid_res + grid_res / 2
            grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
            inside_hull = np.ones(len(grid_pts), dtype=bool)
            for eq in equations:
                inside_hull &= grid_pts @ eq[:-1] + eq[-1] <= 0
            hull_mask = inside_hull.reshape(nx, ny)

            # Merge all clusters whose centroid is inside the hull
            n_labels = labeled.max()
            merged_region = np.zeros_like(occ)
            for lid in range(1, n_labels + 1):
                cij = np.argwhere(labeled == lid)
                centroid = np.array(
                    [
                        xy_min[0] + cij[:, 0].mean() * grid_res + grid_res / 2,
                        xy_min[1] + cij[:, 1].mean() * grid_res + grid_res / 2,
                    ]
                )
                if all(eq[:-1] @ centroid + eq[-1] <= grid_res for eq in equations):
                    merged_region |= labeled == lid
            if not merged_region.any():
                continue
            soma_mask_new = merged_region[ix, iy]

            # Hull-guided void detection: use hull_mask instead of
            # binary_fill_holes to define the interior.
            occ_h = occ & hull_mask
            enc = np.ones_like(occ_h)
            for ax in range(2):
                enc &= np.maximum.accumulate(occ_h, axis=ax)
                enc &= np.flip(
                    np.maximum.accumulate(np.flip(occ_h, axis=ax), axis=ax), axis=ax
                )
            enc &= hull_mask
            void = enc & ~occ_h
            if not void.any():
                continue

            dt = distance_transform_edt(~occ_h)
            dtv = dt * void
            pk = np.unravel_index(dtv.argmax(), dtv.shape)
            void_r = float(dtv.max()) * grid_res
            cx = float(xy_min[0] + pk[0] * grid_res + grid_res / 2)
            cy = float(xy_min[1] + pk[1] * grid_res + grid_res / 2)

            deep_void = void & (dt > 2)
            void_xy = None
            if deep_void.any():
                vl, _ = label(deep_void)
                pl = vl[pk[0], pk[1]]
                if pl == 0:
                    dij = np.argwhere(deep_void)
                    dd = np.abs(dij[:, 0] - pk[0]) + np.abs(dij[:, 1] - pk[1])
                    pl = vl[dij[dd.argmin()][0], dij[dd.argmin()][1]]
                nv = vl == pl
                vij = np.argwhere(nv)
                void_xy = np.column_stack(
                    [
                        xy_min[0] + vij[:, 0] * grid_res + grid_res / 2,
                        xy_min[1] + vij[:, 1] * grid_res + grid_res / 2,
                    ]
                )

            z = raw[i][0]
            vi = raw[i][5]
            raw[i] = (z, cx, cy, void_r, void_xy, vi, soma_mask_new)

        # Re-run spatial coherence with potentially new void detections
        chain = self._chain()
        if chain is not None:
            self.best = chain

    @property
    def center(self) -> np.ndarray | None:
        """Mean void centre and Z over the chain, or None."""
        if self.best is None:
            return None
        raw = self._raw
        slices = np.array(
            [(raw[i][0], raw[i][1], raw[i][2], raw[i][3]) for i in self.best]
        )
        return np.array(
            [np.nanmean(slices[:, 1]), np.nanmean(slices[:, 2]), np.mean(slices[:, 0])]
        )

    def raw(self) -> list[tuple]:
        """Per level ``(z, cx, cy, void_r, void_xy, vertex ids, soma_mask)``."""
        return list(self._raw)
