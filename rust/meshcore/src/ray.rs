//! Rays against triangles, with trimesh's `ray_triangle` hit test.
//!
//! A triangle counts as hit when the ray crosses its plane (the direction
//! not within 1e-5 of parallel to the normal), the crossing's barycentric
//! coordinates (Cramer's rule) all lie within 1e-13 of `[0, 1]`, and the
//! crossing is no more than 1e-6 behind the origin.  Every quantity is
//! computed as trimesh computes it — products rounded, then summed left to
//! right — so the answer is the same ray for ray.

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
    let proj_ori = dd(sub(t[0], o), n);
    let proj_dir = dd(d, n);
    if proj_dir.abs() <= 1e-5 {
        return false;
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
        return false;
    }
    dd(sub(p, o), d) > -1e-6
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
