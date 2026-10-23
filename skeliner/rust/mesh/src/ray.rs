//! Rays and nearest points against triangles.
//!
//! Rays use trimesh's `ray_triangle` hit test.  A triangle counts as hit
//! when the ray crosses its plane (the direction not within 1e-5 of
//! parallel to the normal), the crossing's barycentric coordinates
//! (Cramer's rule) all lie within 1e-13 of `[0, 1]`, and the crossing is no
//! more than 1e-6 behind the origin.  Every quantity is computed as trimesh
//! computes it — products rounded, then summed left to right — so the
//! answer is the same ray for ray.
//!
//! Nearest points use Ericson's region test (Real-Time Collision Detection,
//! 5.1.5), exact up to rounding.

type V3 = [f64; 3];

/// trimesh's `diagonal_dot` of two 3-vectors.
fn dd(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Does the ray from `o` along `d` hit triangle `t` with normal `n`?
pub fn hits_triangle(o: V3, d: V3, t: &[V3; 3], n: V3) -> bool {
    hit_distance(o, d, t, n).is_some()
}

/// Where the ray from `o` along `d` hits triangle `t` (normal `n`), as the
/// ray parameter — the distance along a unit `d` — or `None` when
/// [`hits_triangle`] says it misses.
pub fn hit_distance(o: V3, d: V3, t: &[V3; 3], n: V3) -> Option<f64> {
    let proj_ori = dd(sub(t[0], o), n);
    let proj_dir = dd(d, n);
    if proj_dir.abs() <= 1e-5 {
        return None;
    }
    let distance = proj_ori / proj_dir;
    let p = [0, 1, 2].map(|k| d[k] * distance + o[k]);
    let e0 = sub(t[1], t[0]);
    let e1 = sub(t[2], t[0]);
    let w = sub(p, t[0]);
    let (d00, d01, d02, d11, d12) = (dd(e0, e0), dd(e0, e1), dd(e0, w), dd(e1, e1), dd(e1, w));
    let inv = 1.0 / (d00 * d11 - d01 * d01);
    let b2 = (d00 * d12 - d01 * d02) * inv;
    let b1 = (d11 * d02 - d01 * d12) * inv;
    let b0 = 1.0 - b1 - b2;
    let zero = 1e-13;
    if ![b0, b1, b2].iter().all(|&b| b > -zero && b < 1.0 + zero) {
        return None;
    }
    (dd(sub(p, o), d) > -1e-6).then_some(distance)
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// The point of triangle `t` closest to `p`.
pub fn closest_point_on_triangle(p: V3, t: &[V3; 3]) -> V3 {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dd(ab, ap), dd(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dd(ab, bp), dd(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, scale(ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dd(ab, cp), dd(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = va + vb + vc;
    if denom == 0.0 || !denom.is_finite() {
        // a degenerate triangle: the nearest of its edges
        let edge = |u: V3, v: V3| {
            let e = sub(v, u);
            let l = dd(e, e);
            let s = if l > 0.0 {
                (dd(sub(p, u), e) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            add(u, scale(e, s))
        };
        return [edge(a, b), edge(b, c), edge(c, a)]
            .into_iter()
            .min_by(|x, y| dd(sub(p, *x), sub(p, *x)).total_cmp(&dd(sub(p, *y), sub(p, *y))))
            .expect("three edges");
    }
    add(a, add(scale(ab, vb / denom), scale(ac, vc / denom)))
}

/// Squared distance from `p` to the box `[lo, hi]`.
fn box_distance2(p: V3, lo: V3, hi: V3) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = if p[k] < lo[k] {
            lo[k] - p[k]
        } else if p[k] > hi[k] {
            p[k] - hi[k]
        } else {
            0.0
        };
        s += d * d;
    }
    s
}

const LEAF: usize = 8;

#[derive(Debug, Clone)]
struct Node {
    lo: V3,
    hi: V3,
    /// Range of `order` under this node.
    start: usize,
    end: usize,
    left: usize,
    right: usize,
}

/// A bounding-volume hierarchy over triangles, median-split on the widest
/// axis of the centroids.
pub struct TriangleBvh {
    tris: Vec<[V3; 3]>,
    normals: Vec<V3>,
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl TriangleBvh {
    pub fn new(tris: Vec<[V3; 3]>, normals: Vec<V3>) -> TriangleBvh {
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes = Vec::new();
        if !tris.is_empty() {
            build(&tris, &mut order, 0, tris.len(), &mut nodes);
        }
        TriangleBvh {
            tris,
            normals,
            order,
            nodes,
        }
    }

    /// Does the ray hit any triangle?
    pub fn any_hit(&self, o: V3, d: V3) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv = d.map(|c| 1.0 / c);
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i];
            if !slab(o, inv, n.lo, n.hi) {
                continue;
            }
            if n.left == usize::MAX {
                for &t in &self.order[n.start..n.end] {
                    let t = t as usize;
                    if hits_triangle(o, d, &self.tris[t], self.normals[t]) {
                        return true;
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }
        false
    }

    /// The first triangle the ray from `o` along `d` hits past `min_t`, as
    /// `(index, ray parameter)`, leaving out triangle `skip` (the one the
    /// ray starts on); ties to the lowest index.
    pub fn first_hit(&self, o: V3, d: V3, min_t: f64, skip: Option<usize>) -> Option<(usize, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = d.map(|c| 1.0 / c);
        let mut best: Option<(usize, f64)> = None;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i];
            let Some(entry) = slab_entry(o, inv, n.lo, n.hi) else {
                continue;
            };
            if best.is_some_and(|(_, bt)| entry > bt) {
                continue;
            }
            if n.left == usize::MAX {
                for &t in &self.order[n.start..n.end] {
                    let t = t as usize;
                    if Some(t) == skip {
                        continue;
                    }
                    let Some(h) = hit_distance(o, d, &self.tris[t], self.normals[t]) else {
                        continue;
                    };
                    if h <= min_t {
                        continue;
                    }
                    let better = match best {
                        None => true,
                        Some((bt, bh)) => h < bh || (h == bh && t < bt),
                    };
                    if better {
                        best = Some((t, h));
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }
        best
    }

    /// The triangle closest to `p` no farther than `r`, as `(index,
    /// squared distance)`; exact ties go to the lowest index.
    pub fn closest_within(&self, p: V3, r: f64) -> Option<(usize, f64)> {
        if self.nodes.is_empty() || r.is_nan() || r < 0.0 {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        let mut limit = r * r;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i];
            if box_distance2(p, n.lo, n.hi) > limit {
                continue;
            }
            if n.left == usize::MAX {
                for &t in &self.order[n.start..n.end] {
                    let t = t as usize;
                    let q = closest_point_on_triangle(p, &self.tris[t]);
                    let d2 = dd(sub(q, p), sub(q, p));
                    let better = match best {
                        None => d2 <= limit,
                        Some((bt, bd)) => d2 < bd || (d2 == bd && t < bt),
                    };
                    if better {
                        best = Some((t, d2));
                        limit = d2;
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }
        best
    }

    /// Every triangle no farther than `r` from `p`, as `(index, squared
    /// distance)` in ascending index order.
    pub fn within(&self, p: V3, r: f64) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        if self.nodes.is_empty() || r.is_nan() || r < 0.0 {
            return out;
        }
        let limit = r * r;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i];
            if box_distance2(p, n.lo, n.hi) > limit {
                continue;
            }
            if n.left == usize::MAX {
                for &t in &self.order[n.start..n.end] {
                    let t = t as usize;
                    let q = closest_point_on_triangle(p, &self.tris[t]);
                    let d2 = dd(sub(q, p), sub(q, p));
                    if d2 <= limit {
                        out.push((t, d2));
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }
        out.sort_unstable_by_key(|&(t, _)| t);
        out
    }

    /// The normal given for triangle `t`.
    pub fn normal(&self, t: usize) -> V3 {
        self.normals[t]
    }
}

fn bounds(tris: &[[V3; 3]], ids: &[u32]) -> (V3, V3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &t in ids {
        for p in &tris[t as usize] {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    (lo, hi)
}

fn build(
    tris: &[[V3; 3]],
    order: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> usize {
    let (lo, hi) = bounds(tris, &order[start..end]);
    let me = nodes.len();
    nodes.push(Node {
        lo,
        hi,
        start,
        end,
        left: usize::MAX,
        right: usize::MAX,
    });
    if end - start <= LEAF {
        return me;
    }
    let centroid = |t: u32, k: usize| {
        let p = &tris[t as usize];
        p[0][k] + p[1][k] + p[2][k]
    };
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .expect("three axes");
    let mid = start + (end - start) / 2;
    order[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
        centroid(a, axis).total_cmp(&centroid(b, axis))
    });
    let left = build(tris, order, start, mid, nodes);
    let right = build(tris, order, mid, end, nodes);
    nodes[me].left = left;
    nodes[me].right = right;
    me
}

/// Where the ray enters the (padded) box, if it reaches it — as [`slab`]
/// decides, so no triangle the hit test would accept is culled.
fn slab_entry(o: V3, inv: V3, lo: V3, hi: V3) -> Option<f64> {
    let pad = |v: f64| 1e-6 * v.abs().max(1.0);
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;
    for k in 0..3 {
        let (l, h) = (lo[k] - pad(lo[k]), hi[k] + pad(hi[k]));
        if inv[k].is_infinite() {
            if o[k] < l || o[k] > h {
                return None;
            }
            continue;
        }
        let (t1, t2) = ((l - o[k]) * inv[k], (h - o[k]) * inv[k]);
        let (a, b) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(a);
        tmax = tmax.min(b);
    }
    (tmax >= tmin && tmax >= -1e-3).then_some(tmin)
}

/// Does the ray reach the box, from slightly behind its origin on?  The
/// box is padded so a triangle the hit test would accept is never culled.
fn slab(o: V3, inv: V3, lo: V3, hi: V3) -> bool {
    let pad = |v: f64| 1e-6 * v.abs().max(1.0);
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;
    for k in 0..3 {
        let (l, h) = (lo[k] - pad(lo[k]), hi[k] + pad(hi[k]));
        if inv[k].is_infinite() {
            // parallel to this slab: inside it or never
            if o[k] < l || o[k] > h {
                return false;
            }
            continue;
        }
        let (t1, t2) = ((l - o[k]) * inv[k], (h - o[k]) * inv[k]);
        let (a, b) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(a);
        tmax = tmax.min(b);
    }
    tmax >= tmin && tmax >= -1e-3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> (Vec<[V3; 3]>, Vec<V3>) {
        let t = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ];
        (t, vec![[0.0, 0.0, 1.0]; 2])
    }

    #[test]
    fn rays_through_and_beside_a_square() {
        let (t, n) = square();
        let bvh = TriangleBvh::new(t, n);
        assert!(bvh.any_hit([0.5, 0.5, 1.0], [0.0, 0.0, -1.0]));
        assert!(
            !bvh.any_hit([0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            "behind the ray"
        );
        assert!(
            !bvh.any_hit([2.0, 0.5, 1.0], [0.0, 0.0, -1.0]),
            "beside the square"
        );
        assert!(!bvh.any_hit([0.5, 0.5, 1.0], [1.0, 0.0, 0.0]), "parallel");
        // on the shared diagonal counts (barycentric tolerance)
        assert!(bvh.any_hit([0.5, 0.5, 1.0], [0.0, 0.0, -1.0]));
    }

    #[test]
    fn the_first_hit_agrees_with_testing_every_triangle() {
        let mut state = 19u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 10.0 - 5.0
        };
        let mut tris = Vec::new();
        let mut normals = Vec::new();
        for _ in 0..300 {
            let a = [rnd(), rnd(), rnd()];
            let t = [
                a,
                [a[0] + rnd() * 0.4, a[1] + rnd() * 0.4, a[2]],
                [a[0], a[1] + rnd() * 0.4, a[2] + rnd() * 0.4],
            ];
            let (e0, e1) = (sub(t[1], t[0]), sub(t[2], t[0]));
            let c = [
                e0[1] * e1[2] - e0[2] * e1[1],
                e0[2] * e1[0] - e0[0] * e1[2],
                e0[0] * e1[1] - e0[1] * e1[0],
            ];
            let l = dd(c, c).sqrt();
            normals.push(c.map(|x| x / l));
            tris.push(t);
        }
        let bvh = TriangleBvh::new(tris.clone(), normals.clone());
        let mut found = 0;
        for k in 0..2000 {
            let o = [rnd(), rnd(), rnd()];
            let d = [rnd(), rnd(), rnd()];
            let skip = (k % 3 == 0).then_some(k % tris.len());
            let brute = (0..tris.len())
                .filter(|&t| Some(t) != skip)
                .filter_map(|t| hit_distance(o, d, &tris[t], normals[t]).map(|h| (t, h)))
                .filter(|&(_, h)| h > 1e-9)
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            assert_eq!(bvh.first_hit(o, d, 1e-9, skip), brute);
            found += brute.is_some() as usize;
        }
        assert!(found > 100, "the fixture should hit triangles: {found}");
    }

    #[test]
    fn closest_points_in_every_region() {
        let t = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        // above the face, beyond each vertex, beyond each edge
        assert_eq!(
            closest_point_on_triangle([0.5, 0.5, 3.0], &t),
            [0.5, 0.5, 0.0]
        );
        assert_eq!(closest_point_on_triangle([-1.0, -1.0, 0.0], &t), t[0]);
        assert_eq!(closest_point_on_triangle([3.0, -1.0, 0.0], &t), t[1]);
        assert_eq!(closest_point_on_triangle([-1.0, 3.0, 0.0], &t), t[2]);
        assert_eq!(
            closest_point_on_triangle([1.0, -1.0, 1.0], &t),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            closest_point_on_triangle([-1.0, 1.0, 1.0], &t),
            [0.0, 1.0, 0.0]
        );
        assert_eq!(
            closest_point_on_triangle([2.0, 2.0, 0.0], &t),
            [1.0, 1.0, 0.0]
        );
        // a triangle collapsed to a segment
        let flat = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert_eq!(
            closest_point_on_triangle([1.5, 1.0, 0.0], &flat),
            [1.5, 0.0, 0.0]
        );
    }

    #[test]
    fn the_nearest_triangle_agrees_with_testing_every_one() {
        let mut state = 11u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 10.0 - 5.0
        };
        let mut tris = Vec::new();
        for _ in 0..400 {
            let a = [rnd(), rnd(), rnd()];
            tris.push([
                a,
                [a[0] + rnd() * 0.3, a[1] + rnd() * 0.3, a[2]],
                [a[0], a[1] + rnd() * 0.3, a[2] + rnd() * 0.3],
            ]);
        }
        // an exact duplicate: ties must go to the lower index
        tris.push(tris[17]);
        let bvh = TriangleBvh::new(tris.clone(), vec![[0.0, 0.0, 1.0]; tris.len()]);
        let mut found = 0;
        for _ in 0..3000 {
            let p = [rnd(), rnd(), rnd()];
            let r = rnd().abs() * 0.4;
            let brute = tris
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let q = closest_point_on_triangle(p, t);
                    (i, dd(sub(q, p), sub(q, p)))
                })
                .filter(|&(_, d2)| d2 <= r * r)
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            assert_eq!(bvh.closest_within(p, r), brute);
            found += brute.is_some() as usize;
        }
        assert!(found > 100, "the fixture should find triangles: {found}");
        let on_dup = tris[17][0];
        assert_eq!(bvh.closest_within(on_dup, 0.1).map(|h| h.0), Some(17));
        // `within` lists exactly the triangles brute force finds
        for _ in 0..300 {
            let p = [rnd(), rnd(), rnd()];
            let r = rnd().abs() * 0.6;
            let brute: Vec<(usize, f64)> = tris
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let q = closest_point_on_triangle(p, t);
                    (i, dd(sub(q, p), sub(q, p)))
                })
                .filter(|&(_, d2)| d2 <= r * r)
                .collect();
            assert_eq!(bvh.within(p, r), brute);
        }
    }

    #[test]
    fn the_bvh_agrees_with_testing_every_triangle() {
        let mut state = 7u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 10.0 - 5.0
        };
        let mut tris = Vec::new();
        let mut normals = Vec::new();
        for _ in 0..300 {
            let a = [rnd(), rnd(), rnd()];
            let t = [
                a,
                [a[0] + rnd() * 0.2, a[1] + rnd() * 0.2, a[2]],
                [a[0], a[1] + rnd() * 0.2, a[2] + rnd() * 0.2],
            ];
            let (e0, e1) = (sub(t[1], t[0]), sub(t[2], t[0]));
            let c = [
                e0[1] * e1[2] - e0[2] * e1[1],
                e0[2] * e1[0] - e0[0] * e1[2],
                e0[0] * e1[1] - e0[1] * e1[0],
            ];
            let l = dd(c, c).sqrt();
            normals.push(c.map(|x| x / l));
            tris.push(t);
        }
        let bvh = TriangleBvh::new(tris.clone(), normals.clone());
        for _ in 0..2000 {
            let o = [rnd(), rnd(), rnd()];
            let d = [rnd(), rnd(), rnd()];
            let brute = tris
                .iter()
                .zip(&normals)
                .any(|(t, &n)| hits_triangle(o, d, t, n));
            assert_eq!(bvh.any_hit(o, d), brute);
        }
    }
}
