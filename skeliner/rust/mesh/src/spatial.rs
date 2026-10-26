//! Nearest-neighbour and radius queries over 3D points.

use rayon::prelude::*;

/// How a squared distance is summed.  scipy's KD-tree accumulates
/// `s += d * d` per coordinate, which a compiler may or may not fuse into a
/// multiply-add; the two round differently in the last bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sum {
    /// `(dx * dx + dy * dy) + dz * dz`, each product rounded.
    Plain,
    /// `fma(dz, dz, fma(dy, dy, dx * dx))`.
    Fused,
}

impl Sum {
    #[inline]
    pub fn sq_dist(self, a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        match self {
            Sum::Plain => dx * dx + dy * dy + dz * dz,
            Sum::Fused => dz.mul_add(dz, dy.mul_add(dy, dx * dx)),
        }
    }
}

const LEAF: usize = 16;
const NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
struct Node {
    /// Range of `order` the node covers.
    lo: usize,
    hi: usize,
    bmin: [f64; 3],
    bmax: [f64; 3],
    left: u32,
    right: u32,
}

/// A static KD-tree over 3D points, median-split on the widest axis.
#[derive(Debug, Clone)]
pub struct KdTree {
    points: Vec<[f64; 3]>,
    order: Vec<u32>,
    nodes: Vec<Node>,
    sum: Sum,
}

/// Subtrees over at least this many points build their halves in parallel.
const PARALLEL: usize = 1 << 15;

/// How many nodes a subtree over `n` points has.
fn node_count(n: usize) -> usize {
    if n <= LEAF {
        1
    } else {
        1 + node_count(n / 2) + node_count(n - n / 2)
    }
}

/// Bounding box of the points `order` lists.
fn bounds(points: &[[f64; 3]], order: &[u32]) -> ([f64; 3], [f64; 3]) {
    let grow = |(mut bmin, mut bmax): ([f64; 3], [f64; 3]), p: &[f64; 3]| {
        for ((lo_d, hi_d), &x) in bmin.iter_mut().zip(bmax.iter_mut()).zip(p) {
            *lo_d = lo_d.min(x);
            *hi_d = hi_d.max(x);
        }
        (bmin, bmax)
    };
    let empty = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    if order.len() < PARALLEL {
        return order
            .iter()
            .fold(empty, |b, &i| grow(b, &points[i as usize]));
    }
    // min and max ignore NaN and take the same value in any order
    order
        .par_iter()
        .fold(|| empty, |b, &i| grow(b, &points[i as usize]))
        .reduce(
            || empty,
            |(amin, amax), (bmin, bmax)| {
                (
                    [0, 1, 2].map(|k| amin[k].min(bmin[k])),
                    [0, 1, 2].map(|k| amax[k].max(bmax[k])),
                )
            },
        )
}

/// Build the subtree over `order` — the slots `at..at + order.len()` of the
/// tree's order — into `nodes`, numbered from `id` as a depth-first build
/// numbers them: the node, its left subtree, then its right.  The halves of
/// a large subtree build in parallel; the tree is the same either way.
fn build(points: &[[f64; 3]], order: &mut [u32], at: usize, nodes: &mut [Node], id: u32) {
    let n = order.len();
    let (bmin, bmax) = bounds(points, order);
    nodes[0] = Node {
        lo: at,
        hi: at + n,
        bmin,
        bmax,
        left: NONE,
        right: NONE,
    };
    if n <= LEAF {
        return;
    }
    let dim = (0..3)
        .max_by(|&a, &b| (bmax[a] - bmin[a]).total_cmp(&(bmax[b] - bmin[b])))
        .expect("three axes");
    let mid = n / 2;
    order.select_nth_unstable_by(mid, |&a, &b| {
        points[a as usize][dim].total_cmp(&points[b as usize][dim])
    });
    let n_left = node_count(mid);
    let (left, right) = (id + 1, id + 1 + n_left as u32);
    nodes[0].left = left;
    nodes[0].right = right;
    let (left_nodes, right_nodes) = nodes[1..].split_at_mut(n_left);
    let (left_order, right_order) = order.split_at_mut(mid);
    let mut build_left = || build(points, left_order, at, left_nodes, left);
    let mut build_right = || build(points, right_order, at + mid, right_nodes, right);
    if n >= PARALLEL {
        rayon::join(build_left, build_right);
    } else {
        build_left();
        build_right();
    }
}

impl KdTree {
    /// Build a tree over `points`, measuring distances the `sum` way, in
    /// parallel on [`threads`]' pool.
    pub fn new(points: Vec<[f64; 3]>, sum: Sum) -> Self {
        let n = points.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let placeholder = Node {
            lo: 0,
            hi: 0,
            bmin: [0.0; 3],
            bmax: [0.0; 3],
            left: NONE,
            right: NONE,
        };
        let mut nodes = Vec::new();
        if n > 0 {
            nodes = vec![placeholder; node_count(n)];
            threads::install(|| build(&points, &mut order, 0, &mut nodes, 0));
        }
        Self {
            points,
            order,
            nodes,
            sum,
        }
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// The points, in the order given (the indices queries return).
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// Whether the tree holds no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Squared distance from `q` to the node's box, a lower bound that never
    /// exceeds the distance to any point inside.
    fn box_sq_dist(node: &Node, q: &[f64; 3]) -> f64 {
        let mut s = 0.0;
        for ((&x, &lo), &hi) in q.iter().zip(&node.bmin).zip(&node.bmax) {
            let g = if x < lo {
                lo - x
            } else if x > hi {
                x - hi
            } else {
                0.0
            };
            s += g * g;
        }
        s
    }

    /// Indices of the points within `r` of `q` (squared distance at most
    /// `r * r`), ascending — scipy's `query_ball_point(..., return_sorted=True)`.
    pub fn within(&self, q: &[f64; 3], r: f64) -> Vec<u32> {
        let mut out = Vec::new();
        self.within_into(q, r, &mut out);
        out.sort_unstable();
        out
    }

    /// As [`KdTree::within`], appending unsorted to `out`.
    pub fn within_into(&self, q: &[f64; 3], r: f64, out: &mut Vec<u32>) {
        if self.nodes.is_empty() {
            return;
        }
        let r2 = r * r;
        // The box bound is summed plainly; widen it a hair so it never
        // prunes a point the exact test would take.
        let prune = r2 * (1.0 + 1e-12);
        let mut stack = vec![0u32];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            if Self::box_sq_dist(node, q) > prune {
                continue;
            }
            if node.left == NONE {
                for &i in &self.order[node.lo..node.hi] {
                    if self.sum.sq_dist(&self.points[i as usize], q) <= r2 {
                        out.push(i);
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
    }

    /// The nearest point to `q` and its distance; among exact ties, the
    /// lowest index.  `None` for an empty tree.
    pub fn nearest(&self, q: &[f64; 3]) -> Option<(u32, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = (u32::MAX, f64::INFINITY);
        let mut stack = vec![0u32];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            if Self::box_sq_dist(node, q) > best.1 * (1.0 + 1e-12) {
                continue;
            }
            if node.left == NONE {
                for &i in &self.order[node.lo..node.hi] {
                    let d = self.sum.sq_dist(&self.points[i as usize], q);
                    if d < best.1 || (d == best.1 && i < best.0) {
                        best = (i, d);
                    }
                }
            } else {
                // nearer child last, so it is searched first
                let (l, r) = (node.left as usize, node.right as usize);
                let (dl, dr) = (
                    Self::box_sq_dist(&self.nodes[l], q),
                    Self::box_sq_dist(&self.nodes[r], q),
                );
                if dl <= dr {
                    stack.push(node.right);
                    stack.push(node.left);
                } else {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            }
        }
        Some((best.0, best.1.sqrt()))
    }
}

impl KdTree {
    /// The `k` nearest points to `q`, nearest first, among equally near
    /// points the lowest index first; each with its distance.
    pub fn k_nearest(&self, q: &[f64; 3], k: usize) -> Vec<(u32, f64)> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        // the best so far, ascending by (squared distance, index)
        let mut best: Vec<(f64, u32)> = Vec::with_capacity(k + 1);
        let mut stack = vec![0u32];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            let bound = if best.len() == k {
                best[k - 1].0 * (1.0 + 1e-12)
            } else {
                f64::INFINITY
            };
            if Self::box_sq_dist(node, q) > bound {
                continue;
            }
            if node.left == NONE {
                for &i in &self.order[node.lo..node.hi] {
                    let d = self.sum.sq_dist(&self.points[i as usize], q);
                    let full = best.len() == k;
                    if full && (d, i) >= (best[k - 1].0, best[k - 1].1) {
                        continue;
                    }
                    let at = best.partition_point(|&(bd, bi)| (bd, bi) < (d, i));
                    best.insert(at, (d, i));
                    best.truncate(k);
                }
            } else {
                let (l, r) = (node.left as usize, node.right as usize);
                let (dl, dr) = (
                    Self::box_sq_dist(&self.nodes[l], q),
                    Self::box_sq_dist(&self.nodes[r], q),
                );
                // nearer child last, so it is searched first
                if dl <= dr {
                    stack.push(node.right);
                    stack.push(node.left);
                } else {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            }
        }
        best.into_iter().map(|(d, i)| (i, d.sqrt())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tree as one depth-first pass builds it, a node at a time:
    /// `(order, [(lo, hi, bmin, bmax, left, right)])`.
    #[allow(clippy::type_complexity)]
    fn built_one_node_at_a_time(
        points: &[[f64; 3]],
    ) -> (Vec<u32>, Vec<(usize, usize, [f64; 3], [f64; 3], u32, u32)>) {
        fn go(
            points: &[[f64; 3]],
            order: &mut [u32],
            nodes: &mut Vec<(usize, usize, [f64; 3], [f64; 3], u32, u32)>,
            lo: usize,
            hi: usize,
        ) -> u32 {
            let mut bmin = [f64::INFINITY; 3];
            let mut bmax = [f64::NEG_INFINITY; 3];
            for &i in &order[lo..hi] {
                for k in 0..3 {
                    bmin[k] = bmin[k].min(points[i as usize][k]);
                    bmax[k] = bmax[k].max(points[i as usize][k]);
                }
            }
            let id = nodes.len() as u32;
            nodes.push((lo, hi, bmin, bmax, NONE, NONE));
            if hi - lo > LEAF {
                let dim = (0..3)
                    .max_by(|&a, &b| (bmax[a] - bmin[a]).total_cmp(&(bmax[b] - bmin[b])))
                    .unwrap();
                let mid = lo + (hi - lo) / 2;
                order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
                    points[a as usize][dim].total_cmp(&points[b as usize][dim])
                });
                let left = go(points, order, nodes, lo, mid);
                let right = go(points, order, nodes, mid, hi);
                (nodes[id as usize].4, nodes[id as usize].5) = (left, right);
            }
            id
        }
        let mut order: Vec<u32> = (0..points.len() as u32).collect();
        let mut nodes = Vec::new();
        go(points, &mut order, &mut nodes, 0, points.len());
        (order, nodes)
    }

    #[test]
    fn a_large_tree_is_the_tree_a_one_node_at_a_time_build_makes() {
        // 200k points on a coarse grid: ties on every axis, and a NaN
        let mut state = 0x853c_49e6_748f_ea9bu64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 97) as f64
        };
        let mut pts: Vec<[f64; 3]> = (0..200_000).map(|_| [next(), next(), next()]).collect();
        pts[7][1] = f64::NAN;
        let tree = KdTree::new(pts.clone(), Sum::Plain);
        let (order, nodes) = built_one_node_at_a_time(&pts);
        assert!(
            pts.len() >= 2 * PARALLEL,
            "the halves must build in parallel"
        );
        assert_eq!(tree.order, order);
        assert_eq!(tree.nodes.len(), nodes.len());
        for (a, b) in tree.nodes.iter().zip(&nodes) {
            let same = |x: &[f64; 3], y: &[f64; 3]| {
                x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
            };
            assert!(
                (a.lo, a.hi, a.left, a.right) == (b.0, b.1, b.4, b.5)
                    && same(&a.bmin, &b.2)
                    && same(&a.bmax, &b.3),
                "{a:?} != {b:?}"
            );
        }
    }

    #[test]
    fn k_nearest_matches_brute_force_ties_to_the_lowest_index() {
        // a grid: many points equally near
        let mut pts = Vec::new();
        for x in 0..6 {
            for y in 0..6 {
                for z in 0..4 {
                    pts.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        let tree = KdTree::new(pts.clone(), Sum::Plain);
        for q in [
            [2.5, 2.5, 1.5],
            [0.0, 0.0, 0.0],
            [9.0, -1.0, 2.0],
            [2.0, 3.0, 1.0],
        ] {
            for k in [1, 4, 7, 200] {
                let mut brute: Vec<(f64, u32)> = (0..pts.len() as u32)
                    .map(|i| (Sum::Plain.sq_dist(&pts[i as usize], &q), i))
                    .collect();
                brute.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let want: Vec<(u32, f64)> =
                    brute.iter().take(k).map(|&(d, i)| (i, d.sqrt())).collect();
                assert_eq!(tree.k_nearest(&q, k), want, "q {q:?} k {k}");
            }
        }
        assert!(
            KdTree::new(Vec::new(), Sum::Plain)
                .k_nearest(&[0.0; 3], 3)
                .is_empty()
        );
    }

    /// A deterministic scatter of points.
    fn scatter(n: usize) -> Vec<[f64; 3]> {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    fn brute_within(points: &[[f64; 3]], q: &[f64; 3], r: f64, sum: Sum) -> Vec<u32> {
        (0..points.len() as u32)
            .filter(|&i| sum.sq_dist(&points[i as usize], q) <= r * r)
            .collect()
    }

    #[test]
    fn radius_queries_match_brute_force() {
        let points = scatter(2000);
        for sum in [Sum::Plain, Sum::Fused] {
            let tree = KdTree::new(points.clone(), sum);
            for q in scatter(50) {
                for r in [0.0, 0.05, 0.2, 2.0] {
                    assert_eq!(tree.within(&q, r), brute_within(&points, &q, r, sum));
                }
            }
        }
    }

    #[test]
    fn a_point_exactly_at_the_radius_is_inside() {
        let tree = KdTree::new(vec![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], Sum::Plain);
        assert_eq!(tree.within(&[0.0, 0.0, 0.0], 5.0), vec![0, 1]);
    }

    #[test]
    fn nearest_matches_brute_force_and_breaks_ties_low() {
        let points = scatter(1000);
        let tree = KdTree::new(points.clone(), Sum::Plain);
        for q in scatter(100) {
            let (i, d) = tree.nearest(&q).unwrap();
            let best = points
                .iter()
                .map(|p| Sum::Plain.sq_dist(p, &q))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(d, best.sqrt());
            assert_eq!(Sum::Plain.sq_dist(&points[i as usize], &q), best);
        }
        let tied = KdTree::new(vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], Sum::Plain);
        assert_eq!(tied.nearest(&[0.0, 0.0, 0.0]), Some((0, 1.0)));
    }

    #[test]
    fn an_empty_tree_finds_nothing() {
        let tree = KdTree::new(Vec::new(), Sum::Plain);
        assert!(tree.within(&[0.0; 3], 1.0).is_empty());
        assert_eq!(tree.nearest(&[0.0; 3]), None);
    }
}
