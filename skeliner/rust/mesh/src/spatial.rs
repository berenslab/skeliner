//! Nearest-neighbour and radius queries over 3D points.

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

impl KdTree {
    /// Build a tree over `points`, measuring distances the `sum` way.
    pub fn new(points: Vec<[f64; 3]>, sum: Sum) -> Self {
        let mut tree = Self {
            order: (0..points.len() as u32).collect(),
            points,
            nodes: Vec::new(),
            sum,
        };
        if !tree.points.is_empty() {
            tree.build(0, tree.points.len());
        }
        tree
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

    fn build(&mut self, lo: usize, hi: usize) -> u32 {
        let mut bmin = [f64::INFINITY; 3];
        let mut bmax = [f64::NEG_INFINITY; 3];
        for &i in &self.order[lo..hi] {
            let p = self.points[i as usize];
            for ((lo_d, hi_d), &x) in bmin.iter_mut().zip(bmax.iter_mut()).zip(&p) {
                *lo_d = lo_d.min(x);
                *hi_d = hi_d.max(x);
            }
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            lo,
            hi,
            bmin,
            bmax,
            left: NONE,
            right: NONE,
        });
        if hi - lo > LEAF {
            let dim = (0..3)
                .max_by(|&a, &b| (bmax[a] - bmin[a]).total_cmp(&(bmax[b] - bmin[b])))
                .expect("three axes");
            let mid = lo + (hi - lo) / 2;
            let points = &self.points;
            self.order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
                points[a as usize][dim].total_cmp(&points[b as usize][dim])
            });
            let left = self.build(lo, mid);
            let right = self.build(mid, hi);
            let node = &mut self.nodes[id as usize];
            node.left = left;
            node.right = right;
        }
        id
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

#[cfg(test)]
mod tests {
    use super::*;

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
