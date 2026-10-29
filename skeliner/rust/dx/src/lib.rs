//! `skeliner.dx`'s kernels: distances from points to a skeleton, its
//! volume and cable, its graph's checks and walks (`graph`), and its bins
//! against the mesh (`bins`).

#![forbid(unsafe_code)]

pub mod bins;
pub mod graph;

use mesh::geometry::HAS_FMA;
use mesh::spatial::{KdTree, Sum};
use rayon::prelude::*;

/// Which parts of the skeleton a point is measured against.
#[derive(Debug, Clone, PartialEq)]
pub enum Candidates {
    /// The `k` nearest nodes (ties to the lowest index) and every edge at
    /// any of them.
    Nearest(usize),
    /// These nodes and these edges (node pairs, as given), for every point.
    Whitelist {
        centres: Vec<usize>,
        segments: Vec<(usize, usize)>,
    },
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Python's `max(a, b)`: `a` unless `b > a`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// numpy's `min`: NaN if any is.
fn numpy_min(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(f64::INFINITY, |m, x| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else {
            m.min(x)
        }
    })
}

/// `_point_segment_distance` (without radii) or
/// `_point_segment_capsule_distance` (with `radii`): from `p` to the
/// segment from node `a` to node `b`, less the radius interpolated along it
/// for a capsule.
fn to_segment(p: [f64; 3], nodes: &[[f64; 3]], radii: Option<&[f64]>, a: usize, b: usize) -> f64 {
    let (s, e) = (nodes[a], nodes[b]);
    let v = sub(e, s);
    let len2 = dot(v, v);
    if len2 <= 0.0 {
        let d = norm(sub(p, s));
        return match radii {
            Some(r) => d - py_max(r[a], r[b]),
            None => d,
        };
    }
    // Python's min(1.0, max(0.0, t)), which sends NaN to 0
    let t = dot(sub(p, s), v) / len2;
    let t = if t > 0.0 { t } else { 0.0 };
    let t = if t < 1.0 { t } else { 1.0 };
    let d = norm(sub(p, [s[0] + t * v[0], s[1] + t * v[1], s[2] + t * v[2]]));
    match radii {
        Some(r) => d - ((1.0 - t) * r[a] + t * r[b]),
        None => d,
    }
}

/// A skeleton's nodes, their neighbours and a KD-tree over them: what
/// `skeliner.dx.distance` measures against, built once per skeleton.
#[derive(Debug, Clone)]
pub struct SkeletonIndex {
    nodes: Vec<[f64; 3]>,
    adjacency: Vec<Vec<usize>>,
    tree: KdTree,
}

impl SkeletonIndex {
    /// `adjacency` lists each node's neighbours, which must be nodes. The
    /// tree sums squared distances as scipy's KD-tree does where it is
    /// built: fused on targets with multiply-adds, plainly elsewhere.
    pub fn new(nodes: Vec<[f64; 3]>, adjacency: Vec<Vec<usize>>) -> Self {
        let sum = if HAS_FMA { Sum::Fused } else { Sum::Plain };
        let tree = KdTree::new(nodes.clone(), sum);
        Self {
            nodes,
            adjacency,
            tree,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes(&self) -> &[[f64; 3]] {
        &self.nodes
    }

    pub fn adjacency(&self) -> &[Vec<usize>] {
        &self.adjacency
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// `skeliner.dx.distance`, every mode: per point (scaled by
    /// `scale_in`), the distance to the nearest of the `candidates`' nodes
    /// and edges — to their surface given `radii` (then never below 0), to
    /// the centreline otherwise — scaled by `scale_out`. Every id in the
    /// candidates indexes the nodes, and `radii` has one per node.
    pub fn distance(
        &self,
        points: &[[f64; 3]],
        radii: Option<&[f64]>,
        candidates: &Candidates,
        scale_in: f64,
        scale_out: f64,
    ) -> Vec<f64> {
        let nodes = &self.nodes;
        let less_radius = |i: usize, d: f64| radii.map_or(d, |r| d - r[i]);
        let one = |q: &[f64; 3]| -> f64 {
            let p = q.map(|x| x * scale_in);
            let (mut best, segments): (f64, Vec<(usize, usize)>) = match candidates {
                Candidates::Nearest(k) => {
                    let near = self.tree.k_nearest(&p, *k);
                    let best = numpy_min(near.iter().map(|&(i, d)| less_radius(i as usize, d)));
                    let mut segs: Vec<(usize, usize)> = near
                        .iter()
                        .flat_map(|&(i, _)| {
                            let i = i as usize;
                            self.adjacency[i]
                                .iter()
                                .map(move |&nb| (i.min(nb), i.max(nb)))
                        })
                        .collect();
                    segs.sort_unstable();
                    segs.dedup();
                    (best, segs)
                }
                Candidates::Whitelist { centres, segments } => {
                    let best = if centres.is_empty() {
                        f64::INFINITY
                    } else {
                        numpy_min(
                            centres
                                .iter()
                                .map(|&c| less_radius(c, norm(sub(nodes[c], p)))),
                        )
                    };
                    (best, segments.clone())
                }
            };
            for (a, b) in segments {
                let d = to_segment(p, nodes, radii, a, b);
                if d < best {
                    best = d;
                }
            }
            if radii.is_some() {
                best = py_max(best, 0.0);
            }
            best * scale_out
        };
        threads::install(|| points.par_iter().map(one).collect())
    }
}

/// The soma's ellipsoid for `voxel_union_count`: centre, semi-axes, `R`'s
/// transpose by rows, and its bounding box.
#[derive(Debug, Clone, PartialEq)]
pub struct SomaVoxels {
    pub center: [f64; 3],
    pub axes: [f64; 3],
    pub rt: [[f64; 3]; 3],
    pub lo: [f64; 3],
    pub hi: [f64; 3],
}

/// numpy's elementwise `minimum` and `maximum`: NaN if either is.
fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The voxels from `x0` to `x1` along an axis starting at `lo` in steps of
/// `h` with `n` voxels, as `_voxelize_union`'s `_range_x` takes them:
/// `int(max(0, floor(..)))` to `int(min(n - 1, floor(..)))`, Python's
/// `max` and `min` sending NaN to the other side.
fn voxel_range(x0: f64, x1: f64, lo: f64, h: f64, n: usize) -> (i64, i64) {
    let f = ((x0 - lo) / h).floor();
    let g = ((x1 - lo) / h).floor();
    let top = n as i64 - 1;
    let i0 = if f > 0.0 { f as i64 } else { 0 };
    let i1 = if g < top as f64 { g as i64 } else { top };
    (i0, i1)
}

/// One edge's frustum for the voxel union: its start, its vector and their
/// square, its end radii and its box of voxels.
struct Frustum {
    a: [f64; 3],
    v: [f64; 3],
    l2: f64,
    r0: f64,
    r1: f64,
    rmax: f64,
    vox: [(i64, i64); 3],
}

/// `dx.volume`'s voxel union: how many voxels of the grid from `lo` in
/// steps of `h`, `shape` voxels per axis, have their centre inside an
/// edge's frustum (the radius linear along it) or, with `include_soma`,
/// the soma's ellipsoid; without it the soma's voxels are carved out
/// after. Each box is clipped to `[lo, hi]`; every voxel is tested as
/// `_voxelize_union` tests it, operation for operation.
#[allow(clippy::too_many_arguments)]
pub fn voxel_union_count(
    nodes: &[[f64; 3]],
    radii: &[f64],
    edges: &[[usize; 2]],
    lo: [f64; 3],
    hi: [f64; 3],
    h: f64,
    shape: [usize; 3],
    soma: Option<&SomaVoxels>,
    include_soma: bool,
) -> u64 {
    let [nx, ny, nz] = shape;
    let centre = |d: usize, i: i64| lo[d] + (i as f64 + 0.5) * h;
    let range = |d: usize, x0: f64, x1: f64| voxel_range(x0, x1, lo[d], h, shape[d]);

    let mut frusta: Vec<Frustum> = Vec::new();
    for &[i, j] in edges {
        let (a, b) = (nodes[i], nodes[j]);
        let (r0, r1) = (radii[i], radii[j]);
        let rmax = if r1 > r0 { r1 } else { r0 };
        if !rmax.is_finite() || rmax < 0.0 {
            continue;
        }
        let lo_e = [0, 1, 2].map(|d| np_max(np_min(a[d], b[d]) - rmax, lo[d]));
        let hi_e = [0, 1, 2].map(|d| np_min(np_max(a[d], b[d]) + rmax, hi[d]));
        if (0..3).any(|d| lo_e[d] > hi_e[d]) {
            continue;
        }
        let vox = [0, 1, 2].map(|d| range(d, lo_e[d], hi_e[d]));
        if vox.iter().any(|&(v0, v1)| v1 < v0) {
            continue;
        }
        let v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        frusta.push(Frustum {
            a,
            v,
            l2: v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
            r0,
            r1,
            rmax,
            vox,
        });
    }
    // the soma's voxel box (clipped to [lo, hi]) and its test
    let soma_box = soma.and_then(|s| {
        let slo = [0, 1, 2].map(|d| np_max(s.lo[d], lo[d]));
        let shi = [0, 1, 2].map(|d| np_min(s.hi[d], hi[d]));
        let vox = [0, 1, 2].map(|d| range(d, slo[d], shi[d]));
        vox.iter().all(|&(v0, v1)| v1 >= v0).then_some((s, vox))
    });
    let in_soma = |s: &SomaVoxels, x: f64, y: f64, z: f64| {
        let (dx, dy, dz) = (x - s.center[0], y - s.center[1], z - s.center[2]);
        let u = s.rt.map(|r| r[0] * dx + r[1] * dy + r[2] * dz);
        let a2 = s.axes.map(|a| a * a);
        (u[0] * u[0]) / a2[0] + (u[1] * u[1]) / a2[1] + (u[2] * u[2]) / a2[2] <= 1.0
    };
    let inside = |f: &Frustum, x: f64, y: f64, z: f64| {
        let (dx, dy, dz) = (x - f.a[0], y - f.a[1], z - f.a[2]);
        if f.l2 <= 1e-24 {
            return dx * dx + dy * dy + dz * dz <= f.rmax * f.rmax;
        }
        let s = (dx * f.v[0] + dy * f.v[1] + dz * f.v[2]) / f.l2;
        // np.clip, which keeps a NaN
        let s = if s.is_nan() { s } else { s.clamp(0.0, 1.0) };
        let (rx, ry, rz) = (dx - s * f.v[0], dy - s * f.v[1], dz - s * f.v[2]);
        let r = f.r0 + s * (f.r1 - f.r0);
        rx * rx + ry * ry + rz * rz <= r * r
    };
    let plane = |i: i64| -> u64 {
        let x = centre(0, i);
        let mut occ = vec![false; ny * nz];
        let soma_here = soma_box.filter(|(_, v)| v[0].0 <= i && i <= v[0].1);
        if include_soma && let Some((s, v)) = soma_here {
            for j in v[1].0..=v[1].1 {
                for k in v[2].0..=v[2].1 {
                    if in_soma(s, x, centre(1, j), centre(2, k)) {
                        occ[j as usize * nz + k as usize] = true;
                    }
                }
            }
        }
        for f in frusta.iter().filter(|f| f.vox[0].0 <= i && i <= f.vox[0].1) {
            for j in f.vox[1].0..=f.vox[1].1 {
                for k in f.vox[2].0..=f.vox[2].1 {
                    let at = j as usize * nz + k as usize;
                    if !occ[at] && inside(f, x, centre(1, j), centre(2, k)) {
                        occ[at] = true;
                    }
                }
            }
        }
        if !include_soma && let Some((s, v)) = soma_here {
            for j in v[1].0..=v[1].1 {
                for k in v[2].0..=v[2].1 {
                    if in_soma(s, x, centre(1, j), centre(2, k)) {
                        occ[j as usize * nz + k as usize] = false;
                    }
                }
            }
        }
        occ.iter().filter(|&&o| o).count() as u64
    };
    threads::install(|| (0..nx as i64).into_par_iter().map(plane).sum())
}

/// Python's `min(a, b)`: `b` if `b < a`, else `a`.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `_segment_aabb_clip_length`: how much of the segment from `a` to `b`
/// lies inside the box `[lo, hi]` (Liang–Barsky clipping), 0 for a
/// degenerate segment; Python's `max` and `min` throughout.
fn clip_length(a: [f64; 3], b: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let v = sub(b, a);
    let len = norm(v);
    if len <= 0.0 {
        return 0.0;
    }
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let eps = 1e-12
        * [v[0].abs(), v[1].abs(), v[2].abs()]
            .iter()
            .fold(1.0, |m, &x| py_max(m, x));
    for d in 0..3 {
        let (vd, ad) = (v[d], a[d]);
        if vd.abs() < eps {
            if ad < lo[d] || ad > hi[d] {
                return 0.0;
            }
            continue;
        }
        let (mut t_enter, mut t_exit) = ((lo[d] - ad) / vd, (hi[d] - ad) / vd);
        if t_enter > t_exit {
            std::mem::swap(&mut t_enter, &mut t_exit);
        }
        t0 = py_max(t0, t_enter);
        t1 = py_min(t1, t_exit);
        if t0 > t1 {
            return 0.0;
        }
    }
    len * py_max(0.0, t1 - t0)
}

/// `total_path_length` within a box: the length of the skeleton `edges`
/// inside `[lo, hi]`, summed edge by edge in order, and how many edges
/// reach in.
pub fn path_length_in_box(
    nodes: &[[f64; 3]],
    edges: &[[usize; 2]],
    lo: [f64; 3],
    hi: [f64; 3],
) -> (f64, usize) {
    let (mut total, mut hit) = (0.0f64, 0usize);
    for &[u, v] in edges {
        let ell = clip_length(nodes[u], nodes[v], lo, hi);
        if ell > 0.0 {
            total += ell;
            hit += 1;
        }
    }
    (total, hit)
}

/// What `arm_cables` measures at each junction: the node, its degree, the
/// cable through its parent's side, and the cable of every other arm,
/// largest first.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArmCables {
    pub junctions: Vec<usize>,
    pub degrees: Vec<usize>,
    pub proximal: Vec<f64>,
    pub distal: Vec<Vec<f64>>,
}

/// `_arm_cables`: every node with `min_degree` edges or more that is not a
/// component's root, split into its arms. Each component is rooted at its
/// lowest node (node 0 for the soma's) and walked breadth first, a node's
/// neighbours in edge order; `sub[v]`, the cable strictly below `v`, sums
/// up in reverse of that order. An arm through child `c` carries `sub[c]`
/// plus the edge to it; the parent's side, the component's cable less
/// `sub[v]`.
pub fn arm_cables(nodes: &[[f64; 3]], edges: &[[usize; 2]], min_degree: usize) -> ArmCables {
    let n = nodes.len();
    if n == 0 || edges.is_empty() {
        return ArmCables::default();
    }
    let lengths: Vec<f64> = edges
        .iter()
        .map(|&[a, b]| norm(sub(nodes[a], nodes[b])))
        .collect();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (&[a, b], &len) in edges.iter().zip(&lengths) {
        adj[a].push((b, len));
        adj[b].push((a, len));
    }
    let mut parent = vec![usize::MAX; n];
    let mut parent_len = vec![0.0f64; n];
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut head = order.len();
        order.push(root);
        while head < order.len() {
            let v = order[head];
            head += 1;
            for &(w, len) in &adj[v] {
                if !seen[w] {
                    seen[w] = true;
                    parent[w] = v;
                    parent_len[w] = len;
                    order.push(w);
                }
            }
        }
    }
    let mut below = vec![0.0f64; n];
    for &v in order.iter().rev() {
        let p = parent[v];
        if p != usize::MAX {
            below[p] += below[v] + parent_len[v];
        }
    }
    let mut comp_of = vec![0usize; n];
    for &v in &order {
        comp_of[v] = if parent[v] == usize::MAX {
            v
        } else {
            comp_of[parent[v]]
        };
    }
    let mut out = ArmCables::default();
    for v in 0..n {
        let degree = adj[v].len();
        if degree < min_degree || parent[v] == usize::MAX {
            continue;
        }
        let mut distal: Vec<f64> = adj[v]
            .iter()
            .filter(|&&(w, _)| w != parent[v])
            .map(|&(w, len)| below[w] + len)
            .collect();
        // Python's sort(reverse=True): largest first, ties in order
        distal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        out.junctions.push(v);
        out.degrees.push(degree);
        out.proximal.push(below[comp_of[v]] - below[v]);
        out.distal.push(distal);
    }
    out
}

/// `skeliner.dx.distance` in centreline mode against a whitelist: per point,
/// the distance to the nearest of the `centres` (node ids) or of the
/// `segments` (node-id pairs), each computed as `_point_segment_distance`
/// computes it; infinity when there are neither. Every id indexes `nodes`.
pub fn segment_distances(
    points: &[[f64; 3]],
    nodes: &[[f64; 3]],
    centres: &[usize],
    segments: &[(usize, usize)],
) -> Vec<f64> {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    points
        .iter()
        .map(|&p| {
            let mut best = f64::INFINITY;
            for &c in centres {
                let d = norm(sub(nodes[c], p));
                if d < best {
                    best = d;
                }
            }
            for &(a, b) in segments {
                let (s, e) = (nodes[a], nodes[b]);
                let v = sub(e, s);
                let len2 = dot(v, v);
                let d = if len2 <= 0.0 {
                    norm(sub(p, s))
                } else {
                    // Python's min(1.0, max(0.0, t)), which sends NaN to 0
                    let t = dot(sub(p, s), v) / len2;
                    let t = if t > 0.0 { t } else { 0.0 };
                    let t = if t < 1.0 { t } else { 1.0 };
                    norm(sub(p, [s[0] + t * v[0], s[1] + t * v[1], s[2] + t * v[2]]))
                };
                if d < best {
                    best = d;
                }
            }
            best
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_of_centres_and_segments() {
        let nodes = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 5.0, 0.0]];
        let pts = [[1.0, 1.0, 0.0], [3.0, 0.0, 0.0], [5.0, 5.0, 1.0]];
        let d = segment_distances(&pts, &nodes, &[2], &[(0, 1)]);
        assert_eq!(d, vec![1.0, 1.0, 1.0]);
        assert_eq!(
            segment_distances(&pts, &nodes, &[], &[]),
            vec![f64::INFINITY; 3]
        );
    }
}
