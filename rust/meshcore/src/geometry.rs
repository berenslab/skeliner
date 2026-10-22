//! Small numerical routines: numpy-compatible sums, best-fit planes, areas.

/// Whether to fuse the multiply-adds that numpy and scipy fuse on aarch64
/// macOS (Accelerate's `dgemm`, clang-contracted C).  Fusing needs the
/// target's FMA — without it `mul_add` is a slow software routine — so
/// x86_64 builds without the `fma` feature sum plainly, matching numpy only
/// where numpy sums plainly too.
pub const HAS_FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `d @ m` for a row vector and a 3x3 matrix, rounded as numpy's matmul
/// rounds it through BLAS `dgemm`: each column a chain of fused
/// multiply-adds down the rows, where the target has them, started from
/// `+0.0` as BLAS starts it (so a column of zero products gives `+0.0`).
#[inline]
pub fn row_times_matrix(d: &[f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    [0, 1, 2].map(|j| {
        if HAS_FMA {
            d[2].mul_add(m[2][j], d[1].mul_add(m[1][j], d[0] * m[0][j] + 0.0))
        } else {
            0.0 + d[0] * m[0][j] + d[1] * m[1][j] + d[2] * m[2][j]
        }
    })
}

/// `numpy.sum` of a contiguous float64 array: pairwise summation in blocks
/// of eight, splitting arrays longer than 128, so the result rounds exactly
/// as numpy's does.
pub fn numpy_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let full = n - n % 8;
        let mut i = 8;
        while i < full {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[full..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_sum(&a[..n2]) + numpy_sum(&a[n2..])
    }
}

/// Orders floats as numpy's sort does: `-0.0` equal to `0.0`, and every NaN,
/// whatever its sign bit, after every number.  `f64::total_cmp` puts `-0.0`
/// first and a NaN with its sign bit set — what x86 makes of `0.0 / 0.0` —
/// before `-inf`.  With a stable sort this is `argsort(kind="stable")`.
pub fn numpy_order(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).expect("neither is NaN"),
        (a_nan, b_nan) => a_nan.cmp(&b_nan),
    }
}

/// `numpy.arange(start, stop, step)` for floats: `ceil((stop - start) /
/// step)` values, `start + i * delta` with `delta = (start + step) - start`.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil();
    if len.is_nan() || len <= 0.0 {
        return Vec::new();
    }
    let len = len as usize;
    let mut out = Vec::with_capacity(len);
    out.push(start);
    if len > 1 {
        out.push(start + step);
    }
    let delta = start + step - start;
    for i in 2..len {
        out.push(start + i as f64 * delta);
    }
    out
}

/// Eigenvalues and eigenvectors of a symmetric 3x3 matrix by cyclic Jacobi
/// rotations, sorted by eigenvalue, largest first; `vectors[k]` goes with
/// `values[k]`.
pub fn symmetric_eigen(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off <= f64::EPSILON * (a[0][0].abs() + a[1][1].abs() + a[2][2].abs()) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // A <- J^T A J for the rotation J in the (p, q) plane
            for row in &mut a {
                let (akp, akq) = (row[p], row[q]);
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (rp, rq) = (a[p], a[q]);
            a[p] = [0, 1, 2].map(|k| c * rp[k] - s * rq[k]);
            a[q] = [0, 1, 2].map(|k| s * rp[k] + c * rq[k]);
            for row in &mut v {
                let (vkp, vkq) = (row[p], row[q]);
                row[p] = c * vkp - s * vkq;
                row[q] = s * vkp + c * vkq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let values = order.map(|i| a[i][i]);
    let vectors = order.map(|i| [v[0][i], v[1][i], v[2][i]]);
    (values, vectors)
}

/// Mean of `pts` and an orthonormal basis `(u, w)` of their best-fit plane
/// (the two leading principal directions), as numpy's SVD of the centred
/// points gives them up to sign.
pub fn best_fit_plane(pts: &[[f64; 3]]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let n = pts.len() as f64;
    let mut mean = [0.0f64; 3];
    for p in pts {
        for k in 0..3 {
            mean[k] += p[k];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut cov = [[0.0f64; 3]; 3];
    for p in pts {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let (_, vecs) = symmetric_eigen(cov);
    (mean, vecs[0], vecs[1])
}

/// Coordinates of `pts` in the plane through `origin` spanned by `u`, `w`.
pub fn project(pts: &[[f64; 3]], origin: [f64; 3], u: [f64; 3], w: [f64; 3]) -> Vec<[f64; 2]> {
    pts.iter()
        .map(|p| {
            let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
            [
                d[0] * u[0] + d[1] * u[1] + d[2] * u[2],
                d[0] * w[0] + d[1] * w[1] + d[2] * w[2],
            ]
        })
        .collect()
}

/// Area enclosed by a closed polygon (vertices in order), by the shoelace
/// formula.
pub fn polygon_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * s.abs()
}

/// Area of the convex hull of 2D points (monotone chain); 0 for fewer than
/// three points or collinear ones.
pub fn convex_hull_area(points: &[[f64; 2]]) -> f64 {
    let mut p: Vec<[f64; 2]> = points.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return 0.0;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &q in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0
            {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    polygon_area(&hull)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_of_short_and_long_arrays() {
        let a: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();
        assert!((numpy_sum(&a) - 49950.0).abs() < 1e-9);
        assert_eq!(numpy_sum(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(numpy_sum(&[]), 0.0);
    }

    #[test]
    fn sorts_signed_zeros_and_nans_as_numpy_does() {
        // np.argsort([3, nan, -nan, -0.0, 0.0, -1], kind="stable")
        let x = [3.0, f64::NAN, -f64::NAN, -0.0, 0.0, -1.0];
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&i, &j| numpy_order(&x[i], &x[j]));
        assert_eq!(order, [5, 3, 4, 0, 1, 2]);
        // a NaN with its sign bit set, as x86 makes it
        let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000);
        assert!(neg_nan.is_nan() && neg_nan.is_sign_negative());
        let mut v = [neg_nan, 1.0, f64::NEG_INFINITY];
        v.sort_by(numpy_order);
        assert_eq!(v[..2], [f64::NEG_INFINITY, 1.0]);
        assert!(v[2].is_nan());
    }

    #[test]
    fn eigen_of_a_diagonal_and_a_rotated_matrix() {
        let (vals, vecs) = symmetric_eigen([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]);
        assert_eq!(vals, [3.0, 2.0, 1.0]);
        assert_eq!(vecs[0].map(f64::abs), [0.0, 1.0, 0.0]);
        let (vals, vecs) = symmetric_eigen([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.5]]);
        assert!((vals[0] - 3.0).abs() < 1e-12 && (vals[1] - 1.0).abs() < 1e-12);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((vecs[0][0].abs() - s).abs() < 1e-12 && (vecs[0][1].abs() - s).abs() < 1e-12);
    }

    #[test]
    fn a_tilted_square_has_its_area_in_its_plane() {
        // unit square in a plane tilted about x
        let (c, s) = (0.6, 0.8);
        let sq = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, c, s], [0.0, c, s]];
        let (o, u, w) = best_fit_plane(&sq);
        let area = polygon_area(&project(&sq, o, u, w));
        assert!((area - 1.0).abs() < 1e-12, "{area}");
        assert!((convex_hull_area(&project(&sq, o, u, w)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hull_ignores_interior_and_collinear_points() {
        let pts = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ];
        assert_eq!(convex_hull_area(&pts), 4.0);
        assert_eq!(convex_hull_area(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 0.0);
    }
}
