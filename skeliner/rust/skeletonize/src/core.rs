//! `skeliner._core`'s stages, run by both tracks of `skeletonize` and by
//! `post`'s wrappers.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use mesh::geometry::{numpy_argmin, numpy_order, numpy_sum, row_times_matrix};
use mesh::spatial::{KdTree, Sum};
use rayon::prelude::*;

use crate::skeleton::{estimate_radius_sorted, quantile_sorted};

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
        let mut sorted = d.to_vec();
        sorted.sort_by(numpy_order);
        for (key, r) in keys.iter().zip(&mut self.radii) {
            if let Some(v) = estimate_radius_sorted(d, &sorted, key) {
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

/// `dataclass.Soma`'s ellipsoid: `world = R @ body + center`, its semi-axes
/// largest first.
#[derive(Debug, Clone, PartialEq)]
pub struct Ellipsoid {
    pub center: [f64; 3],
    pub axes: [f64; 3],
    /// The rows of `R`, whose columns are the principal axes.
    pub r: [[f64; 3]; 3],
}

impl Ellipsoid {
    /// `Soma.spherical_radius`: Python's `max` over the axes.
    pub fn spherical_radius(&self) -> f64 {
        let mut best = self.axes[0];
        for &a in &self.axes[1..] {
            if a > best {
                best = a;
            }
        }
        best
    }

    /// `Soma.distance_to_center`: `‖x − center‖`, summed left to right.
    pub fn distance_to_center(&self, x: &[[f64; 3]]) -> Vec<f64> {
        x.iter()
            .map(|p| {
                let d = [0, 1, 2].map(|k| p[k] - self.center[k]);
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect()
    }

    /// `Soma.distance_to_surface` (`atol` 1e-9, 64 steps at most): the
    /// signed distance to the surface, negative inside, as numpy computes
    /// it — the body coordinates through `dgemm`'s FMA chain, every sum left
    /// to right, `** 3` as `pow`, and one Newton loop for all the points
    /// outside, stopped once every step is below `atol`, so a point's result
    /// depends on the others in the same call.
    pub fn distance_to_surface(&self, x: &[[f64; 3]]) -> Vec<f64> {
        const ATOL: f64 = 1e-9;
        let a = self.axes;
        let a2 = [a[0] * a[0], a[1] * a[1], a[2] * a[2]];
        let p: Vec<[f64; 3]> = x
            .iter()
            .map(|q| row_times_matrix(&[0, 1, 2].map(|k| q[k] - self.center[k]), &self.r))
            .collect();
        let r2: Vec<f64> = p
            .iter()
            .map(|q| (q[0] * q[0] / a2[0] + q[1] * q[1] / a2[1]) + q[2] * q[2] / a2[2])
            .collect();
        let out: Vec<usize> = (0..p.len()).filter(|&i| r2[i] > 1.0 + 1e-12).collect();
        let mut t = vec![0.0f64; out.len()];
        let step = |(&i, &ti): (&usize, &f64)| {
            let q = p[i];
            let den = [ti + a2[0], ti + a2[1], ti + a2[2]];
            let f = [0, 1, 2].map(|k| a2[k] * (q[k] * q[k]) / (den[k] * den[k]));
            let fp = [0, 1, 2].map(|k| (-2.0 * a2[k]) * (q[k] * q[k]) / den[k].powf(3.0));
            -((f[0] + f[1]) + f[2] - 1.0) / ((fp[0] + fp[1]) + fp[2])
        };
        for _ in 0..64 {
            // every point's step at once when there are enough of them
            let dts: Vec<f64> = if out.len() >= 1024 {
                threads::install(|| out.par_iter().zip(&t).map(step).collect())
            } else {
                out.iter().zip(&t).map(step).collect()
            };
            let mut converged = true;
            for (ti, dt) in t.iter_mut().zip(&dts) {
                *ti += dt;
                // a NaN step never converges, as with numpy's `all`
                converged &= dt.abs() < ATOL;
            }
            if converged {
                break;
            }
        }
        let norm = |e: [f64; 3]| (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        let mut dist = vec![0.0f64; p.len()];
        let mut outside = vec![false; p.len()];
        for (&i, &ti) in out.iter().zip(&t) {
            outside[i] = true;
            let q = p[i];
            dist[i] = norm([0, 1, 2].map(|k| a2[k] * q[k] / (ti + a2[k]) - q[k]));
        }
        // numpy's `min` of the axes
        let min_axis = a.iter().copied().fold(f64::INFINITY, |m, v| {
            if m.is_nan() || v.is_nan() {
                f64::NAN
            } else {
                m.min(v)
            }
        });
        for i in (0..p.len()).filter(|&i| !outside[i]) {
            let s = r2[i].sqrt();
            dist[i] = if s > ATOL {
                let q = p[i];
                -norm([0, 1, 2].map(|k| q[k] / s - q[k]))
            } else {
                -min_axis
            };
        }
        dist
    }
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

/// What `merge_near_soma_nodes` changed: per node before, its index after
/// (−1 for a node merged into the soma), the soma's vertices to refit it
/// from (`None` while it has none), and how many nodes went into it.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedNear {
    pub old2new: Vec<i64>,
    pub soma_verts: Option<Vec<i64>>,
    pub n_merged: usize,
}

/// `_merge_near_soma_nodes` but the soma's refit: every node but the root
/// inside the soma (`distance_to_surface < −inside_tol`), or near it
/// (`distance_to_center < near_factor × r_soma`) and fat (its radius in row
/// `radius_row` above `fat_factor × r_soma`), is merged into the root and
/// dropped; the kept nodes it touched are tied to the root. Of the merged
/// nodes' vertices, in order, those within `near_factor × r_soma` of the
/// surface (all of them without `mesh_vertices`) join the root's vertices
/// and the soma's, which end sorted and unique.
#[allow(clippy::too_many_arguments)]
pub fn merge_near_soma_nodes(
    s: &mut Arrays,
    radius_row: usize,
    soma: &Ellipsoid,
    soma_verts: Option<&[i64]>,
    mesh_vertices: Option<&[[f64; 3]]>,
    inside_tol: f64,
    near_factor: f64,
    fat_factor: f64,
) -> MergedNear {
    let r_soma = soma.spherical_radius();
    let d2s = soma.distance_to_surface(&s.nodes);
    let d2c = soma.distance_to_center(&s.nodes);
    let n = s.nodes.len();
    let mut keep: Vec<bool> = (0..n)
        .map(|i| {
            let inside = d2s[i] < -inside_tol;
            let near = d2c[i] < near_factor * r_soma;
            let fat = s.radii[radius_row][i] > fat_factor * r_soma;
            !(inside || (near && fat))
        })
        .collect();
    keep[0] = true;
    let merged: Vec<usize> = (0..n).filter(|&i| !keep[i]).collect();
    let edges = s.edges.clone();
    let merged_verts: Vec<Vec<usize>> = merged.iter().map(|&i| s.node2verts[i].clone()).collect();
    let old2new = s.compact(&keep);

    let mut leaves: Vec<usize> = edges
        .iter()
        .filter_map(|&[u, v]| match (keep[u], keep[v]) {
            (true, false) => Some(old2new[u] as usize),
            (false, true) => Some(old2new[v] as usize),
            _ => None,
        })
        .filter(|&l| l != 0)
        .collect();
    leaves.sort_unstable();
    leaves.dedup();
    if !leaves.is_empty() {
        s.edges.extend(leaves.iter().map(|&l| [0, l]));
        s.edges.sort_unstable();
        s.edges.dedup();
    }

    let mut verts: Option<Vec<i64>> = soma_verts.map(<[i64]>::to_vec);
    let limit = near_factor * r_soma;
    // each merged node's vertices near the soma, the nodes at once
    let contribs: Vec<Vec<usize>> = threads::install(|| {
        merged_verts
            .par_iter()
            .map(|vidx| match mesh_vertices {
                Some(mv) => {
                    let pts: Vec<[f64; 3]> = vidx.iter().map(|&v| mv[v]).collect();
                    let d = soma.distance_to_surface(&pts);
                    vidx.iter()
                        .zip(&d)
                        .filter(|&(_, &d)| d < limit)
                        .map(|(&v, _)| v)
                        .collect()
                }
                None => vidx.clone(),
            })
            .collect()
    });
    for contrib in contribs {
        if !contrib.is_empty() {
            verts
                .get_or_insert_with(Vec::new)
                .extend(contrib.iter().map(|&v| v as i64));
            s.node2verts[0].extend_from_slice(&contrib);
        }
    }
    if let Some(v) = &mut verts {
        v.sort_unstable();
        v.dedup();
    }
    MergedNear {
        old2new,
        soma_verts: verts,
        n_merged: merged.len(),
    }
}

/// What `find_soma` found: the soma's nodes, the node of largest radius,
/// the soma's centre and radius, and whether there are enough nodes for a
/// soma.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundSoma {
    pub soma_nodes: Vec<usize>,
    pub idx_max: usize,
    pub center: [f64; 3],
    pub r_max: f64,
    pub has_soma: bool,
}

/// numpy's `argmax`: the first largest value, or the first NaN.
fn numpy_argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `_find_soma`: the nodes whose `radius` is at least its `pct_large`-th
/// percentile (numpy's linear one; NaN, so no nodes, if any radius is
/// NaN) and which lie within `dist_factor` times the largest radius of the
/// node that has it. A soma with `min_keep` such nodes or more is centred
/// on their mean (summed row by row from the first); otherwise on that
/// node. `None` when no node reaches the percentile.
pub fn find_soma(
    nodes: &[[f64; 3]],
    radius: &[f64],
    pct_large: f64,
    dist_factor: f64,
    min_keep: usize,
) -> Option<FoundSoma> {
    let threshold = if radius.iter().any(|r| r.is_nan()) {
        f64::NAN
    } else {
        let mut sorted = radius.to_vec();
        sorted.sort_by(numpy_order);
        quantile_sorted(&sorted, pct_large / 100.0)
    };
    let candidates: Vec<usize> = (0..radius.len())
        .filter(|&i| radius[i] >= threshold)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let idx_max = numpy_argmax(radius);
    let r_max = radius[idx_max];
    let c = nodes[idx_max];
    let soma_nodes: Vec<usize> = candidates
        .into_iter()
        .filter(|&i| {
            let d = [0, 1, 2].map(|k| nodes[i][k] - c[k]);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() <= dist_factor * r_max
        })
        .collect();
    let has_soma = soma_nodes.len() >= min_keep;
    let center = match (has_soma, soma_nodes.split_first()) {
        (true, Some((&first, rest))) => {
            let mut sum = nodes[first];
            for &i in rest {
                for k in 0..3 {
                    sum[k] += nodes[i][k];
                }
            }
            sum.map(|s| s / soma_nodes.len() as f64)
        }
        _ => c,
    };
    Some(FoundSoma {
        soma_nodes,
        idx_max,
        center,
        r_max,
        has_soma,
    })
}

/// The soma's vertices, from its nodes at `points` with their vertex lists
/// `lists`: those of the nodes closer to `center` than twice `r_max`
/// (lists that are not empty), merged, sorted and unique; failing those,
/// the first node's list as it is.
pub fn soma_vertices(
    points: &[[f64; 3]],
    lists: &[Vec<usize>],
    center: [f64; 3],
    r_max: f64,
) -> Vec<usize> {
    let mut verts: Vec<usize> = points
        .iter()
        .zip(lists)
        .filter(|(p, l)| {
            let d = [0, 1, 2].map(|k| p[k] - center[k]);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < r_max * 2.0 && !l.is_empty()
        })
        .flat_map(|(_, l)| l.iter().copied())
        .collect();
    if verts.is_empty() {
        return lists.first().cloned().unwrap_or_default();
    }
    verts.sort_unstable();
    verts.dedup();
    verts
}

/// A component's gap to the tied part: its length, the component, the
/// index of its node in the component and of the tied node, ordered as
/// Python orders the tuple (the gap, then the rest).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Gap(f64, usize, usize, usize);

impl Eq for Gap {}

impl Ord for Gap {
    fn cmp(&self, other: &Self) -> Ordering {
        // reversed, so `BinaryHeap` pops the smallest
        other
            .0
            .total_cmp(&self.0)
            .then(other.1.cmp(&self.1))
            .then(other.2.cmp(&self.2))
            .then(other.3.cmp(&self.3))
    }
}

impl PartialOrd for Gap {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The nodes tied so far, for [`bridge_gaps`]' nearest searches: a tree
/// over those tied at the last rebuild and one over each component tied
/// since, rebuilt as one past eight.  Each tree's nearest is its least by
/// squared distance, then index, whatever its shape, so the least of theirs
/// is what one tree over every tied node finds.
struct Tied<'a> {
    nodes: &'a [[f64; 3]],
    /// The tied nodes, in the order they were tied.
    island: Vec<usize>,
    /// Each tree, and the island index of its first node.
    trees: Vec<(KdTree, u32)>,
}

impl<'a> Tied<'a> {
    fn new(nodes: &'a [[f64; 3]], island: Vec<usize>) -> Self {
        let mut tied = Tied {
            nodes,
            island,
            trees: Vec::new(),
        };
        tied.rebuild();
        tied
    }

    fn rebuild(&mut self) {
        let points = self.island.iter().map(|&v| self.nodes[v]).collect();
        self.trees = vec![(KdTree::new(points, Sum::Plain), 0)];
    }

    fn tie(&mut self, comp: &[usize]) {
        let at = self.island.len() as u32;
        self.island.extend_from_slice(comp);
        if self.trees.len() > 8 {
            self.rebuild();
        } else {
            let points = comp.iter().map(|&v| self.nodes[v]).collect();
            self.trees.push((KdTree::new(points, Sum::Plain), at));
        }
    }

    /// `KdTree::nearest` over the tied nodes: an island index and distance.
    fn nearest(&self, q: &[f64; 3]) -> (u32, f64) {
        let mut best = (u32::MAX, f64::INFINITY);
        for (tree, at) in &self.trees {
            match tree.nearest_sq(q) {
                // no point nearer than infinity, as with NaN distances
                Some((u32::MAX, _)) | None => {}
                Some((i, d)) => {
                    let i = i + at;
                    if d < best.1 || (d == best.1 && i < best.0) {
                        best = (i, d);
                    }
                }
            }
        }
        (best.0, best.1.sqrt())
    }
}

/// `_bridge_gaps`: every connected component of `edges` tied to the
/// component of node 0 by one new edge, nearest component first — the
/// closest pair of nodes between it and what is tied so far (a component's
/// nodes ascending, the first of equally close ones; the tied nodes in the
/// order they joined, ties to the earliest). A component farther than
/// `bridge_max_factor` mean edge lengths waits (by default the 55th
/// percentile of the first gaps over the mean edge length, clipped to 6 to
/// 12), and the limit grows by half after twice three times as many waits
/// as there were gaps; every gap is measured again after
/// `bridge_recalc_after` bridges (by default 2 to 32 by the number of
/// gaps). The edges with the bridges, unique and in order; `None` when
/// there is one component (the edges stay as they are).
pub fn bridge_gaps(
    nodes: &[[f64; 3]],
    edges: &[[usize; 2]],
    bridge_max_factor: Option<f64>,
    bridge_recalc_after: Option<usize>,
) -> Option<Vec<[usize; 2]>> {
    let n = nodes.len();
    // components numbered as igraph numbers them: by their lowest node
    let mut adj = vec![Vec::new(); n];
    for &[a, b] in edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut membership = vec![usize::MAX; n];
    let mut n_comp = 0;
    for start in 0..n {
        if membership[start] != usize::MAX {
            continue;
        }
        membership[start] = n_comp;
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if membership[v] == usize::MAX {
                    membership[v] = n_comp;
                    stack.push(v);
                }
            }
        }
        n_comp += 1;
    }
    if n_comp == 1 {
        return None;
    }
    let mut comps: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for (v, &c) in membership.iter().enumerate() {
        comps[c].push(v);
    }
    let soma_cid = membership[0];
    let lengths: Vec<f64> = edges
        .iter()
        .map(|&[a, b]| {
            let d = [0, 1, 2].map(|k| nodes[a][k] - nodes[b][k]);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .collect();
    let edge_len_mean = numpy_sum(&lengths) / lengths.len() as f64;

    let mut tree = Tied::new(nodes, comps[soma_cid].clone());
    let closest = |tree: &Tied, cid: usize| -> (f64, usize, usize) {
        let near = |&v: &usize| tree.nearest(&nodes[v]);
        let found: Vec<(u32, f64)> = if comps[cid].len() >= 1024 {
            threads::install(|| comps[cid].par_iter().map(near).collect())
        } else {
            comps[cid].iter().map(near).collect()
        };
        let dists: Vec<f64> = found.iter().map(|f| f.1).collect();
        let best = numpy_argmin(&dists);
        (dists[best], best, found[best].0 as usize)
    };
    let measured = |tree: &Tied, cids: Vec<usize>| {
        let gaps: Vec<Gap> = threads::install(|| {
            cids.into_par_iter()
                .map(|c| {
                    let (g, bc, bi) = closest(tree, c);
                    Gap(g, c, bc, bi)
                })
                .collect()
        });
        // the gaps are all different, so the heap pops them in one order
        BinaryHeap::from(gaps)
    };

    let mut pq = measured(&tree, (0..n_comp).filter(|&c| c != soma_cid).collect());
    let factor = bridge_max_factor.unwrap_or_else(|| {
        let mut gaps: Vec<f64> = pq.iter().map(|g| g.0).collect();
        let raw = if gaps.iter().any(|g| g.is_nan()) {
            f64::NAN
        } else {
            gaps.sort_by(numpy_order);
            quantile_sorted(&gaps, 55.0 / 100.0)
        } / edge_len_mean;
        raw.clamp(6.0, 12.0)
    });
    let n_gaps = n_comp - 1;
    let recalc_after = bridge_recalc_after.unwrap_or(match n_gaps {
        0..=10 => 2,
        11..=50 => 4,
        51..=200 => (n_gaps / 20).max(4),
        _ => 32,
    });

    let mut bridges: Vec<[usize; 2]> = Vec::new();
    let (mut since_recalc, mut stall) = (0usize, 0usize);
    let max_stall = 3 * pq.len();
    let mut current_max = factor * edge_len_mean;
    while let Some(Gap(_, cid, _, _)) = pq.pop() {
        let (gap, best_c, best_i) = closest(&tree, cid);
        if gap > current_max {
            pq.push(Gap(gap, cid, best_c, best_i));
            stall += 1;
            if stall >= 2 * max_stall {
                let cids: Vec<usize> = pq.iter().map(|g| g.1).collect();
                pq = measured(&tree, cids);
                current_max *= 1.5;
                stall = 0;
            }
            continue;
        }
        stall = 0;
        bridges.push([comps[cid][best_c], tree.island[best_i]]);
        tree.tie(&comps[cid]);
        since_recalc += 1;
        if since_recalc >= recalc_after {
            let cids: Vec<usize> = pq.iter().map(|g| g.1).collect();
            pq = measured(&tree, cids);
            since_recalc = 0;
        }
    }
    let mut out: Vec<[usize; 2]> = edges.to_vec();
    out.extend(bridges);
    out.sort_unstable();
    out.dedup();
    Some(out)
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
