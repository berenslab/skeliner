//! `skeliner._core`'s stages, run by both tracks of `skeletonize` and by
//! `post`'s wrappers.

use std::collections::VecDeque;

use crate::skeleton::estimate_radius;

/// A skeleton's node-aligned arrays, as `_state.SkeletonState` holds them.
#[derive(Debug, Clone, PartialEq)]
pub struct Arrays {
    pub nodes: Vec<[f64; 3]>,
    /// One array per radius estimator, each one radius per node.
    pub radii: Vec<Vec<f64>>,
    /// Each node's mesh vertices.
    pub node2verts: Vec<Vec<usize>>,
    pub edges: Vec<[usize; 2]>,
}

impl Arrays {
    /// `compact_state`: the kept nodes in order with their radii and
    /// vertices, and the edges between kept nodes renumbered
    /// (`remap_edges`); `old2new`, −1 for a node dropped.
    pub fn compact(&mut self, keep: &[bool]) -> Vec<i64> {
        let mut old2new = vec![-1i64; keep.len()];
        let mut next = 0;
        for (i, &k) in keep.iter().enumerate() {
            if k {
                old2new[i] = next;
                next += 1;
            }
        }
        retain_kept(&mut self.nodes, keep);
        for r in &mut self.radii {
            retain_kept(r, keep);
        }
        retain_kept(&mut self.node2verts, keep);
        self.edges = remap_edges(&self.edges, &old2new);
        old2new
    }

    /// `_radii_from_distances` for node `nid` from its vertices' distances
    /// `d`: every radius a plain aggregate of them (`estimate_radius`)
    /// re-estimated; `centerline` and any other left as they are.
    fn reestimate(&mut self, keys: &[String], nid: usize, d: &[f64]) {
        for (key, r) in keys.iter().zip(&mut self.radii) {
            if let Some(v) = estimate_radius(d, key) {
                r[nid] = v;
            }
        }
    }

    /// Distances from node `nid` to its vertices, as `np.linalg.norm(...,
    /// axis=1)` sums each row: left to right.
    fn vertex_distances(&self, nid: usize, mesh_vertices: &[[f64; 3]]) -> Vec<f64> {
        let c = self.nodes[nid];
        self.node2verts[nid]
            .iter()
            .map(|&v| {
                let p = mesh_vertices[v];
                let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect()
    }
}

/// `v`'s items where `keep` is true, in order.
fn retain_kept<T>(v: &mut Vec<T>, keep: &[bool]) {
    let mut i = 0;
    v.retain(|_| {
        i += 1;
        keep[i - 1]
    });
}

/// `remap_edges`: the edges between kept nodes, renumbered, each sorted, no
/// self-loops, unique and in order.
pub fn remap_edges(edges: &[[usize; 2]], old2new: &[i64]) -> Vec<[usize; 2]> {
    let mut out: Vec<[usize; 2]> = edges
        .iter()
        .filter_map(|&[a, b]| {
            let (a, b) = (old2new[a], old2new[b]);
            (a >= 0 && b >= 0 && a != b).then(|| {
                let (a, b) = (a as usize, b as usize);
                [a.min(b), a.max(b)]
            })
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// `_merge_single_node_branches`: every leaf (one edge, not the root) whose
/// parent in a breadth-first tree from the root has `min_parent_degree`
/// edges or more is merged into that parent — its vertices appended to the
/// parent's, in leaf order, and with `mesh_vertices` the parent's radii
/// re-estimated from all its vertices — and dropped; again until no leaf
/// qualifies. A leaf the root does not reach stays. Returns, per node
/// kept, its index before.
pub fn merge_single_node_branches(
    s: &mut Arrays,
    keys: &[String],
    mesh_vertices: Option<&[[f64; 3]]>,
    min_parent_degree: usize,
) -> Vec<usize> {
    let mut orig: Vec<usize> = (0..s.nodes.len()).collect();
    loop {
        let n = s.nodes.len();
        let mut deg = vec![0usize; n];
        for &[a, b] in &s.edges {
            deg[a] += 1;
            deg[b] += 1;
        }
        let parent = bfs_parents(&s.edges, n, 0);
        let singles: Vec<(usize, usize)> = (1..n)
            .filter(|&i| deg[i] == 1)
            .filter_map(|leaf| {
                let p = usize::try_from(parent[leaf]).ok()?;
                (deg[p] >= min_parent_degree).then_some((leaf, p))
            })
            .collect();
        if singles.is_empty() {
            return orig;
        }
        let mut keep = vec![true; n];
        let mut touched = Vec::new();
        for &(leaf, p) in &singles {
            let moved = s.node2verts[leaf].clone();
            s.node2verts[p].extend(moved);
            keep[leaf] = false;
            touched.push(p);
        }
        if let Some(mv) = mesh_vertices {
            touched.sort_unstable();
            touched.dedup();
            for p in touched {
                if !s.node2verts[p].is_empty() {
                    let d = s.vertex_distances(p, mv);
                    s.reestimate(keys, p, &d);
                }
            }
        }
        s.compact(&keep);
        retain_kept(&mut orig, &keep);
    }
}

/// What `prune_neurites` changed: per node before, its index after (−1
/// for a node merged away), the soma's vertices to refit it from, and how
/// many nodes went into the soma.
#[derive(Debug, Clone, PartialEq)]
pub struct Pruned {
    pub old2new: Vec<i64>,
    pub soma_verts: Vec<usize>,
    pub n_merged: usize,
}

/// `_prune_neurites` but the soma's refit: each neurite at the root — the
/// tree from one of the root's neighbours, walked as `_prune_neurites`
/// walks it — whose nodes all lie within `stem_extent_factor × r_soma` of
/// the soma's `center` (`tip_extent_factor` for a neurite whose first node
/// is not the root's child in the tree) is merged into the root: its
/// vertices to the root's, which are made unique, and with
/// `mesh_vertices` the root's radii re-estimated. Then, with
/// `drop_single_node_branches`, `merge_single_node_branches` (parents of
/// three edges or more). The soma's vertices are the root's before that.
#[allow(clippy::too_many_arguments)]
pub fn prune_neurites(
    s: &mut Arrays,
    keys: &[String],
    center: [f64; 3],
    r_soma: f64,
    mesh_vertices: Option<&[[f64; 3]]>,
    tip_extent_factor: f64,
    stem_extent_factor: f64,
    drop_single_node_branches: bool,
) -> Pruned {
    let n = s.nodes.len();
    let d2c: Vec<f64> = s
        .nodes
        .iter()
        .map(|p| {
            let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .collect();
    let parent = bfs_parents(&s.edges, n, 0);
    let mut adj = vec![Vec::new(); n];
    for &[a, b] in &s.edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut merge = vec![false; n];
    let mut visited = vec![false; n];
    for &child in &adj[0] {
        if visited[child] {
            continue;
        }
        let mut comp = Vec::new();
        let (mut stack, mut prev) = (vec![child], 0usize);
        while let Some(v) = stack.pop() {
            if visited[v] {
                continue;
            }
            visited[v] = true;
            comp.push(v);
            for &nb in &adj[v] {
                if nb != prev && (parent[nb] == v as i64 || parent[v] == nb as i64) {
                    stack.push(nb);
                }
            }
            prev = v;
        }
        // as numpy's `max`: NaN if any is
        let max_d = comp
            .iter()
            .map(|&v| d2c[v])
            .fold(f64::NEG_INFINITY, |a, b| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        let factor = if parent[child] == 0 {
            stem_extent_factor
        } else {
            tip_extent_factor
        };
        if max_d <= factor * r_soma {
            for v in comp {
                merge[v] = true;
            }
        }
    }
    merge[0] = false;
    let n_merged = merge.iter().filter(|&&m| m).count();
    if n_merged > 0 {
        let mut root: Vec<usize> = s.node2verts[0].clone();
        for v in (0..n).filter(|&v| merge[v]) {
            root.extend_from_slice(&s.node2verts[v]);
        }
        root.sort_unstable();
        root.dedup();
        s.node2verts[0] = root;
        if let Some(mv) = mesh_vertices
            && !s.node2verts[0].is_empty()
        {
            let d = s.vertex_distances(0, mv);
            s.reestimate(keys, 0, &d);
        }
    }
    let keep: Vec<bool> = merge.iter().map(|&m| !m).collect();
    let mut old2new = s.compact(&keep);
    let mut soma_verts = s.node2verts[0].clone();
    soma_verts.sort_unstable();
    soma_verts.dedup();
    if drop_single_node_branches {
        let kept = merge_single_node_branches(s, keys, mesh_vertices, 3);
        let mut after = vec![-1i64; old2new.len()];
        for (new, &before) in kept.iter().enumerate() {
            after[before] = new as i64;
        }
        for o in &mut old2new {
            if *o >= 0 {
                *o = after[*o as usize];
            }
        }
    }
    Pruned {
        old2new,
        soma_verts,
        n_merged,
    }
}

/// Each node's parent in a breadth-first tree from `root` over the
/// undirected `edges`, a node's neighbours in the order the edges name
/// them; −1 for the root and for every node it does not reach. `edges`
/// must name nodes below `n`, and `root` must be one.
pub fn bfs_parents(edges: &[[usize; 2]], n: usize, root: usize) -> Vec<i64> {
    let mut adj = vec![Vec::new(); n];
    for &[a, b] in edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut parent = vec![-1i64; n];
    let mut queue = VecDeque::from([root]);
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if v != root && parent[v] == -1 {
                parent[v] = u as i64;
                queue.push_back(v);
            }
        }
    }
    parent
}

/// The order igraph's `igraph_vector_sort_ind` puts `w` in: its vendored
/// BSD quicksort (Bentley and McIlroy's) over pointers to the values,
/// which compares them with `<` and `>` only, so it is not stable and a
/// NaN ties with everything.
fn igraph_sort_order(w: &[f64]) -> Vec<usize> {
    let mut v: Vec<usize> = (0..w.len()).collect();
    bsd_qsort(&mut v, 0, w.len(), w);
    v
}

fn cmp(v: &[usize], w: &[f64], i: usize, j: usize) -> i32 {
    let (a, b) = (w[v[i]], w[v[j]]);
    if a < b {
        -1
    } else if a > b {
        1
    } else {
        0
    }
}

fn med3(v: &[usize], w: &[f64], a: usize, b: usize, c: usize) -> usize {
    if cmp(v, w, a, b) < 0 {
        if cmp(v, w, b, c) < 0 {
            b
        } else if cmp(v, w, a, c) < 0 {
            c
        } else {
            a
        }
    } else if cmp(v, w, b, c) > 0 {
        b
    } else if cmp(v, w, a, c) < 0 {
        a
    } else {
        c
    }
}

fn insertion(v: &mut [usize], w: &[f64], a: usize, n: usize) {
    for pm in a + 1..a + n {
        let mut pl = pm;
        while pl > a && cmp(v, w, pl - 1, pl) > 0 {
            v.swap(pl, pl - 1);
            pl -= 1;
        }
    }
}

/// `v[a..a + n]` sorted by `w` as the BSD `qsort` sorts it, step for step.
fn bsd_qsort(v: &mut [usize], mut a: usize, mut n: usize, w: &[f64]) {
    loop {
        if n < 2 {
            return;
        }
        if n < 7 {
            insertion(v, w, a, n);
            return;
        }
        let mut pm = a + n / 2;
        if n > 7 {
            let (mut pl, mut pn) = (a, a + n - 1);
            if n > 40 {
                let d = n / 8;
                pl = med3(v, w, pl, pl + d, pl + 2 * d);
                pm = med3(v, w, pm - d, pm, pm + d);
                pn = med3(v, w, pn - 2 * d, pn - d, pn);
            }
            pm = med3(v, w, pl, pm, pn);
        }
        v.swap(a, pm);
        let (mut pa, mut pb) = (a + 1, a + 1);
        // `pc` and `pd` run down to `a`, one below the first unsorted slot
        let (mut pc, mut pd) = (a + n - 1, a + n - 1);
        let mut swapped = false;
        loop {
            while pb <= pc {
                let r = cmp(v, w, pb, a);
                if r > 0 {
                    break;
                }
                if r == 0 {
                    swapped = true;
                    v.swap(pa, pb);
                    pa += 1;
                }
                pb += 1;
            }
            while pb <= pc {
                let r = cmp(v, w, pc, a);
                if r < 0 {
                    break;
                }
                if r == 0 {
                    swapped = true;
                    v.swap(pc, pd);
                    pd -= 1;
                }
                pc -= 1;
            }
            if pb > pc {
                break;
            }
            v.swap(pb, pc);
            swapped = true;
            pb += 1;
            pc -= 1;
        }
        if !swapped {
            insertion(v, w, a, n);
            return;
        }
        let pn = a + n;
        let d1 = (pa - a).min(pb - pa);
        for k in 0..d1 {
            v.swap(a + k, pb - d1 + k);
        }
        let d1 = (pd - pc).min(pn - pd - 1);
        for k in 0..d1 {
            v.swap(pb + k, pn - d1 + k);
        }
        let (d1, d2) = (pb - pa, pd - pc);
        if d1 <= d2 {
            if d1 > 1 {
                bsd_qsort(v, a, d1, w);
            }
            if d2 <= 1 {
                return;
            }
            a = pn - d2;
            n = d2;
        } else {
            if d2 > 1 {
                bsd_qsort(v, pn - d2, d2, w);
            }
            if d1 <= 1 {
                return;
            }
            n = d1;
        }
    }
}

/// Why a spanning tree could not be built.
#[derive(Debug, PartialEq)]
pub enum MstError {
    /// An edge's length is NaN (a node at NaN).
    NanLength,
}

/// The minimum spanning forest of `edges` weighted by their lengths, as
/// igraph's `spanning_tree` picks it: Kruskal's algorithm over the edges
/// in the order igraph's sort leaves them, ties included
/// (`igraph_sort_order`); each length `‖a − b‖` summed left to right, as
/// `np.linalg.norm` sums a 3-vector. The forest's edges as sorted pairs,
/// in order. `edges` must name rows of `nodes`.
pub fn build_mst(nodes: &[[f64; 3]], edges: &[[usize; 2]]) -> Result<Vec<[usize; 2]>, MstError> {
    let w: Vec<f64> = edges
        .iter()
        .map(|&[a, b]| {
            let d = [0, 1, 2].map(|k| nodes[a][k] - nodes[b][k]);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .collect();
    if w.iter().any(|x| x.is_nan()) {
        return Err(MstError::NanLength);
    }
    let mut comp: Vec<usize> = (0..nodes.len()).collect();
    fn root(comp: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while comp[r] != r {
            r = comp[r];
        }
        comp[x] = r;
        r
    }
    let mut tree = Vec::new();
    for i in igraph_sort_order(&w) {
        let [a, b] = edges[i];
        let (ra, rb) = (root(&mut comp, a), root(&mut comp, b));
        if ra != rb {
            comp[ra] = rb;
            tree.push([a.min(b), a.max(b)]);
        }
    }
    tree.sort_unstable();
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfs_parents_follow_the_edges_in_order() {
        // 0 - 1, 0 - 2, 2 - 3, and 4 unreached
        let p = bfs_parents(&[[2, 3], [0, 2], [1, 0]], 5, 0);
        assert_eq!(p, vec![-1, 0, 0, 2, -1]);
        assert_eq!(bfs_parents(&[[0, 1]], 2, 1), vec![1, -1]);
    }

    #[test]
    fn the_sort_is_a_permutation_into_order() {
        // which of the tied weights goes first is igraph's, checked against
        // igraph itself in `tests/test_core.py`
        let w = [2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        let order = igraph_sort_order(&w);
        let sorted: Vec<f64> = order.iter().map(|&i| w[i]).collect();
        assert!(sorted.windows(2).all(|p| p[0] <= p[1]));
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..w.len()).collect::<Vec<_>>());
    }

    #[test]
    fn the_spanning_forest_keeps_the_short_edges() {
        let nodes = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [9.0, 9.0, 9.0],
            [9.0, 9.0, 10.0],
        ];
        // a unit square with one diagonal, a self-loop, and a second tree
        let edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [2, 2], [5, 4]];
        let t = build_mst(&nodes, &edges).expect("finite");
        assert_eq!(t.len(), 4);
        assert!(t.contains(&[4, 5]) && !t.contains(&[0, 2]));
        let nan = [[f64::NAN; 3], [0.0; 3]];
        assert_eq!(build_mst(&nan, &[[0, 1]]), Err(MstError::NanLength));
        assert_eq!(build_mst(&nodes, &[]), Ok(vec![]));
    }
}
