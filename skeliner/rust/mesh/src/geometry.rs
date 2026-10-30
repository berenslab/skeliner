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

/// Length of each edge, `sqrt((dx² + dy²) + dz²)`: trimesh's
/// `edges_unique_length`, whose BLAS dot of the squares with ones adds
/// them in that order (multiplying by one rounds nothing, fused or not).
pub fn edge_lengths(vertices: &[[f64; 3]], edges: &[[u32; 2]]) -> Vec<f64> {
    threads::map(edges, |&[a, b]| {
        let (p, q) = (&vertices[a as usize], &vertices[b as usize]);
        let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    })
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

/// numpy's `argmin`: the first smallest value, or the first NaN; 0 when
/// `v` is empty.
pub fn numpy_argmin(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x < v[best] {
            best = i;
        }
    }
    best
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

/// A bound on [`jacobi`]'s sweeps, which a 3 × 3 matrix never comes near.
const JACOBI_SWEEPS: usize = 64;

/// Jacobi's eigen-decomposition of the symmetric `a`: its eigenvalues, and
/// the rows of the matrix whose columns are their eigenvectors.  Sweeps
/// over the pairs (0, 1), (0, 2), (1, 2), each rotation zeroing its entry
/// (Rutishauser's formulas), until no entry is left off the diagonal; one
/// too small to move either diagonal entry it couples is zeroed without a
/// rotation.  Only + − × ÷ and √: the Python path's twin computes the same
/// bits on any platform.
pub fn jacobi(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..JACOBI_SWEEPS {
        if a[0][1] == 0.0 && a[0][2] == 0.0 && a[1][2] == 0.0 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq == 0.0 {
                continue;
            }
            let (app, aqq) = (a[p][p], a[q][q]);
            if app + apq == app && aqq + apq == aqq {
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                continue;
            }
            let theta = (aqq - app) / (2.0 * apq);
            let t = 1.0f64.copysign(theta) / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let tau = s / (1.0 + c);
            let h = t * apq;
            a[p][p] = app - h;
            a[q][q] = aqq + h;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            let r = 3 - p - q;
            let (arp, arq) = (a[r][p], a[r][q]);
            a[r][p] = arp - s * (arq + tau * arp);
            a[p][r] = a[r][p];
            a[r][q] = arq + s * (arp - tau * arq);
            a[q][r] = a[r][q];
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = vp - s * (vq + tau * vp);
                row[q] = vq + s * (vp - tau * vq);
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// numpy's `np.linalg.svd(x)[2]` up to the vectors' signs: the right
/// singular vectors of `x` (as given — centre it first for principal
/// directions) as rows, largest singular value first, ties in axis order.
/// They are the eigenvectors of `xᵀx` by [`jacobi`], its entries numpy's
/// pairwise sums of the columns' products ([`numpy_sum`]), so the Python
/// path's twin computes the same bits on any platform.
pub fn right_singular(x: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut gram = [[0.0f64; 3]; 3];
    let mut products = vec![0.0f64; x.len()];
    for j in 0..3 {
        for k in j..3 {
            for (p, r) in products.iter_mut().zip(x) {
                *p = r[j] * r[k];
            }
            let g = numpy_sum(&products);
            gram[j][k] = g;
            gram[k][j] = g;
        }
    }
    let (lam, v) = jacobi(gram);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        lam[j]
            .partial_cmp(&lam[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.map(|i| [v[0][i], v[1][i], v[2][i]])
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

/// The convex hull of 2D points, counter-clockwise, by the monotone chain:
/// the points sorted by x then y (`total_cmp`), repeats dropped, collinear
/// points on the hull left out. Fewer than three distinct points come back
/// as they are, sorted.
pub fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut p: Vec<[f64; 2]> = points.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return p;
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
    hull
}

/// Whether qhull, as scipy's `ConvexHull` runs it, refuses `points` (their
/// `hull` from [`convex_hull`]) as flat: fewer than three distinct points,
/// or the triangle it starts from — the hull's points of least and greatest
/// x and the one farthest from their line — with its centroid within
/// qhull's roundoff of that line (`DISTround`, from the largest coordinate).
pub fn qhull_flat(points: &[[f64; 2]], hull: &[[f64; 2]]) -> bool {
    if hull.len() < 3 {
        return true;
    }
    let (mut maxabs, mut maxsum) = (0.0f64, 0.0f64);
    for p in points {
        maxabs = maxabs.max(p[0].abs()).max(p[1].abs());
        maxsum = maxsum.max(p[0].abs() + p[1].abs());
    }
    let distsum = (2f64.sqrt() * maxabs).min(maxsum);
    let roundoff = f64::EPSILON * (2.0 * distsum * 1.01 + maxabs);
    let order = |p: &&[f64; 2], q: &&[f64; 2]| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1]));
    let a = *hull.iter().min_by(order).expect("three points");
    let b = *hull.iter().max_by(order).expect("three points");
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length = (dx * dx + dy * dy).sqrt();
    let height = hull
        .iter()
        .map(|q| ((q[0] - a[0]) * dy - (q[1] - a[1]) * dx).abs() / length)
        .fold(0.0f64, f64::max);
    height / 3.0 <= roundoff
}

/// `(area, perimeter)` of the convex hull of 2D points — scipy's
/// `ConvexHull` `volume` and `area` — by the shoelace formula about the
/// hull's first point (far from the origin, the products otherwise cancel)
/// and the edge lengths; both 0 where qhull refuses the points as flat.
pub fn hull_measures(points: &[[f64; 2]]) -> (f64, f64) {
    let hull = convex_hull(points);
    if qhull_flat(points, &hull) {
        return (0.0, 0.0);
    }
    let o = hull[0];
    let (mut s, mut perimeter) = (0.0, 0.0);
    for (i, &a) in hull.iter().enumerate() {
        let b = hull[(i + 1) % hull.len()];
        s += (a[0] - o[0]) * (b[1] - o[1]) - (b[0] - o[0]) * (a[1] - o[1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    (0.5 * s.abs(), perimeter)
}

/// The edges of the convex hull of 2D points as scipy's
/// `ConvexHull.equations` gives them: per edge `(n0, n1, offset)`, the unit
/// normal pointing out, `n . p + offset <= 0` inside, counter-clockwise from
/// the lowest point; None where qhull refuses the points as flat.
pub fn hull_equations(points: &[[f64; 2]]) -> Option<Vec<[f64; 3]>> {
    let hull = convex_hull(points);
    if qhull_flat(points, &hull) {
        return None;
    }
    Some(
        (0..hull.len())
            .map(|i| {
                let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let length = (dx * dx + dy * dy).sqrt();
                let n = [dy / length, -dx / length];
                [n[0], n[1], -(n[0] * a[0] + n[1] * a[1])]
            })
            .collect(),
    )
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
    fn edge_lengths_add_the_squares_in_axis_order() {
        let v = [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 12.0],
            [1.0, 1.2247e-8, 1.2247e-8],
        ];
        let l = edge_lengths(&v, &[[0, 1], [1, 0], [2, 2], [0, 2]]);
        assert_eq!(l[..3], [13.0, 13.0, 0.0]);
        // each small square rounds the sum up a step; added to each other
        // first they would round it up once, and the root would come out 1
        assert_eq!(l[3], 1.0000000000000002);
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
    fn jacobi_diagonalises_a_diagonal_and_a_rotated_matrix() {
        let (vals, vecs) = jacobi([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]);
        assert_eq!(vals, [1.0, 3.0, 2.0]);
        assert_eq!(vecs, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let (vals, vecs) = jacobi([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.5]]);
        let big = (0..3).max_by(|&i, &j| vals[i].total_cmp(&vals[j])).unwrap();
        assert!((vals[big] - 3.0).abs() < 1e-12);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((vecs[0][big].abs() - s).abs() < 1e-12 && (vecs[1][big].abs() - s).abs() < 1e-12);
    }

    #[test]
    fn a_tilted_square_has_its_area_in_its_plane() {
        // unit square in a plane tilted about x, centred
        let (c, s) = (0.6, 0.8);
        let sq = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, c, s], [0.0, c, s]];
        let mid = [0.5, c / 2.0, s / 2.0];
        let centred: Vec<[f64; 3]> = sq
            .iter()
            .map(|p| [0, 1, 2].map(|k| p[k] - mid[k]))
            .collect();
        let vt = right_singular(&centred);
        // the normal comes last, the plane's two directions first
        assert!(((vt[2][1] * -s + vt[2][2] * c).abs() - 1.0).abs() < 1e-12);
        let plane = project(&centred, [0.0; 3], vt[0], vt[1]);
        assert!((polygon_area(&plane) - 1.0).abs() < 1e-12);
        assert!((hull_measures(&plane).0 - 1.0).abs() < 1e-12);
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
        assert_eq!(hull_measures(&pts), (4.0, 8.0));
        assert_eq!(
            convex_hull(&pts),
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
        );
        // collinear, too few, and repeated points span no area
        assert_eq!(
            hull_measures(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            (0.0, 0.0)
        );
        assert_eq!(hull_measures(&[[0.0, 0.0], [1.0, 1.0]]), (0.0, 0.0));
        assert!(hull_equations(&[[1.0, 1.0]; 5]).is_none());
    }

    #[test]
    fn hull_edges_point_out() {
        let eq = hull_equations(&[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]])
            .expect("a square");
        assert_eq!(
            eq,
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, -2.0],
                [0.0, 1.0, -2.0],
                [-1.0, 0.0, 0.0]
            ]
        );
    }

    #[test]
    fn points_flat_to_rounding_are_flat_as_qhull_has_them() {
        // on a line through large coordinates, off it by a couple of ulps
        // there, far less than qhull's roundoff: refused, as qhull refuses
        let line: Vec<[f64; 2]> = (0..20)
            .map(|i| {
                let t = f64::from(i) * 37.0;
                [
                    500_000.0 + t,
                    300_000.0 + 0.5 * t + if i % 3 == 0 { 1e-10 } else { 0.0 },
                ]
            })
            .collect();
        assert!(convex_hull(&line).len() >= 3);
        assert_eq!(hull_measures(&line), (0.0, 0.0));
        assert!(hull_equations(&line).is_none());
        // a nanometre off the line is a (thin) hull
        let mut thin = line.clone();
        thin[7][1] += 1.0;
        assert!(hull_measures(&thin).0 > 0.0);
        assert!(hull_equations(&thin).is_some());
    }
}
