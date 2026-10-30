//! `skeliner.dataclass.Soma`: the soma's ellipsoid — fitted to points, and
//! measured as the Python path's numpy measures it
//! (`skeliner/python/dataclass.py`).

use mesh::geometry::{jacobi, numpy_sum, row_times_matrix};
use rayon::prelude::*;

/// `Soma.distance_to_surface`'s defaults: its Newton steps stop once every
/// one is below `ATOL`, after `MAX_ITER` at most.
pub const ATOL: f64 = 1e-9;
pub const MAX_ITER: usize = 64;

/// A Newton step for this many points or more is taken at once.
const NEWTON_PARALLEL: usize = 1024;
/// Points inside, each a bisection of tens of halvings: at once from this
/// many.
const INSIDE_PARALLEL: usize = 256;
/// Eberly's bound on the halvings: as many as a double has values between
/// two of its powers of two, down to the subnormals.
const MAX_BISECTIONS: usize = (f64::MANTISSA_DIGITS as i32 - f64::MIN_EXP) as usize;

/// Why [`fit`] made no ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitError {
    /// Fewer than two points: no spread.
    TooFew(usize),
    /// A coordinate, or the spread, is not finite.
    NotFinite,
    /// No spread along an axis: an eigenvalue at or below zero.
    Flat,
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFew(n) => write!(f, "a soma is fitted to two points or more, not {n}"),
            Self::NotFinite => write!(f, "the points' spread is not finite"),
            Self::Flat => write!(
                f,
                "the points have no spread along an axis: no ellipsoid fits them"
            ),
        }
    }
}

impl std::error::Error for FitError {}

/// `Soma`'s ellipsoid: `world = R @ body + center`, its semi-axes largest
/// first.
#[derive(Debug, Clone, PartialEq)]
pub struct Soma {
    pub center: [f64; 3],
    pub axes: [f64; 3],
    /// The rows of `R`, whose columns are the principal axes.
    pub r: [[f64; 3]; 3],
}

/// `Soma._W`, the map to the unit sphere: a point is inside when
/// `|(x - center) @ w|^2 <= 1`, where `w = R / axes`.
#[derive(Debug, Clone, Copy)]
pub struct Body {
    pub center: [f64; 3],
    pub w: [[f64; 3]; 3],
}

impl Body {
    /// `Soma._body_coords`: `(x - center) @ w`, rounded as numpy's matmul.
    pub fn coords(&self, x: &[f64; 3]) -> [f64; 3] {
        let d = [
            x[0] - self.center[0],
            x[1] - self.center[1],
            x[2] - self.center[2],
        ];
        row_times_matrix(&d, &self.w)
    }

    /// `|coords(x)|^2`, summed in numpy's order.
    pub fn rho2(&self, x: &[f64; 3]) -> f64 {
        let xi = self.coords(x);
        xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
    }

    pub fn contains(&self, x: &[f64; 3]) -> bool {
        self.rho2(x) <= 1.0
    }
}

impl Soma {
    /// The map to the unit sphere: `R` over the axes, by column.
    pub fn body(&self) -> Body {
        let a = self.axes;
        Body {
            center: self.center,
            w: self
                .r
                .map(|row| [row[0] / a[0], row[1] / a[1], row[2] / a[2]]),
        }
    }

    /// `Soma.contains`: which points are inside the ellipsoid scaled to
    /// `rho2 <= limit` (`inside_frac ** 2`).
    pub fn contains(&self, x: &[[f64; 3]], limit: f64) -> Vec<bool> {
        let body = self.body();
        threads::map(x, |p| body.rho2(p) <= limit)
    }

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

    /// `Soma.distance_to_center`.
    pub fn distance_to_center(&self, x: &[[f64; 3]]) -> Vec<f64> {
        distance_to_center(self.center, x)
    }

    /// [`Soma::distance_to_surface_with`] at `Soma.distance_to_surface`'s
    /// defaults.
    pub fn distance_to_surface(&self, x: &[[f64; 3]]) -> Vec<f64> {
        self.distance_to_surface_with(x, ATOL, MAX_ITER)
    }

    /// `Soma.distance_to_surface`: the signed Euclidean distance to the
    /// surface, negative inside, as numpy computes it — the body coordinates
    /// through `dgemm`'s FMA chain, every sum left to right, `** 3` as
    /// `pow`.  Outside (`|body / axes|^2 > 1 + 1e-12`), Newton's iteration
    /// on the Lagrange multiplier, one loop for all of them stopped once
    /// every step is below `atol` (so a point's result depends on the others
    /// in the same call), `max_iter` steps at most; inside, Eberly's
    /// bisection ([`inside_distance`]); within `atol` of the centre, in body
    /// units, the smallest semi-axis.
    pub fn distance_to_surface_with(&self, x: &[[f64; 3]], atol: f64, max_iter: usize) -> Vec<f64> {
        let a = self.axes;
        let a2 = [a[0] * a[0], a[1] * a[1], a[2] * a[2]];
        let p: Vec<[f64; 3]> = threads::map(x, |q| {
            row_times_matrix(&[0, 1, 2].map(|k| q[k] - self.center[k]), &self.r)
        });
        let r2: Vec<f64> = threads::map(&p, |q| {
            (q[0] * q[0] / a2[0] + q[1] * q[1] / a2[1]) + q[2] * q[2] / a2[2]
        });
        let out: Vec<usize> = (0..p.len()).filter(|&i| r2[i] > 1.0 + 1e-12).collect();
        let mut t = vec![0.0f64; out.len()];
        let step = |q: &[f64; 3], ti: f64| {
            let den = [ti + a2[0], ti + a2[1], ti + a2[2]];
            let f = [0, 1, 2].map(|k| a2[k] * (q[k] * q[k]) / (den[k] * den[k]));
            let fp = [0, 1, 2].map(|k| (-2.0 * a2[k]) * (q[k] * q[k]) / den[k].powf(3.0));
            -((f[0] + f[1]) + f[2] - 1.0) / ((fp[0] + fp[1]) + fp[2])
        };
        // a NaN step never converges, as with numpy's `all`
        let advance = |(ti, &i): (&mut f64, &usize)| {
            let dt = step(&p[i], *ti);
            *ti += dt;
            dt.abs() < atol
        };
        for _ in 0..max_iter {
            let converged = if out.len() >= NEWTON_PARALLEL {
                threads::install(|| {
                    t.par_iter_mut()
                        .zip(&out)
                        .map(advance)
                        .reduce(|| true, |a, b| a & b)
                })
            } else {
                t.iter_mut().zip(&out).map(advance).fold(true, |a, b| a & b)
            };
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
        let inside: Vec<usize> = (0..p.len()).filter(|&i| !outside[i]).collect();
        let depth = |&i: &usize| {
            if r2[i].sqrt() > atol {
                -inside_distance(a, p[i].map(f64::abs))
            } else {
                -min_axis
            }
        };
        let depths: Vec<f64> = if inside.len() >= INSIDE_PARALLEL {
            threads::install(|| inside.par_iter().map(depth).collect())
        } else {
            inside.iter().map(depth).collect()
        };
        for (&i, d) in inside.iter().zip(depths) {
            dist[i] = d;
        }
        dist
    }
}

/// `Soma.fit`: the ellipsoid of `pts`' spread — centred on their mean
/// (numpy's over the rows: a sum down them, then over the count), its axes
/// the covariance's eigenvectors, each semi-axis `√(5 λ)` of its
/// eigenvalue, largest first (ties in axis order), `R` a rotation (the
/// smallest axis turned when the columns come left-handed).  The covariance
/// is numpy's pairwise sums of the centred products over `n − 1`, its
/// eigen-decomposition [`jacobi`]'s (`mesh::geometry`): only + − × ÷ and
/// √, so the Python path computes the same bits on any platform.
pub fn fit(pts: &[[f64; 3]]) -> Result<Soma, FitError> {
    let n = pts.len();
    if n < 2 {
        return Err(FitError::TooFew(n));
    }
    let mut sum = pts[0];
    for p in &pts[1..] {
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    let center = sum.map(|s| s / n as f64);
    let d: Vec<[f64; 3]> = pts
        .iter()
        .map(|p| [p[0] - center[0], p[1] - center[1], p[2] - center[2]])
        .collect();
    let mut cov = [[0.0f64; 3]; 3];
    let mut products = vec![0.0f64; n];
    for j in 0..3 {
        for k in j..3 {
            for (x, p) in products.iter_mut().zip(&d) {
                *x = p[j] * p[k];
            }
            let c = numpy_sum(&products) / (n - 1) as f64;
            cov[j][k] = c;
            cov[k][j] = c;
        }
    }
    if !cov.iter().flatten().all(|c| c.is_finite()) {
        return Err(FitError::NotFinite);
    }
    let (lam, vec) = jacobi(cov);
    if !lam.iter().all(|l| l.is_finite()) {
        return Err(FitError::NotFinite);
    }
    if lam.iter().any(|&l| l <= 0.0) {
        return Err(FitError::Flat);
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        lam[j]
            .partial_cmp(&lam[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let axes = order.map(|i| (lam[i] * 5.0).sqrt());
    let mut r = vec.map(|row| order.map(|i| row[i]));
    let col = |c: usize| [r[0][c], r[1][c], r[2][c]];
    let (e0, e1, e2) = (col(0), col(1), col(2));
    let cross = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ];
    if cross[0] * e2[0] + cross[1] * e2[1] + cross[2] * e2[2] < 0.0 {
        for row in &mut r {
            row[2] = -row[2];
        }
    }
    Ok(Soma { center, axes, r })
}

/// `Soma.distance_to_center`: `‖x − center‖`, summed left to right.
pub fn distance_to_center(center: [f64; 3], x: &[[f64; 3]]) -> Vec<f64> {
    threads::map(x, |p| {
        let d = [0, 1, 2].map(|k| p[k] - center[k]);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    })
}

/// The distance from `y` to the ellipsoid of semi-axes `e`, largest first,
/// `y` in its frame and in its first octant (`y >= 0`): Eberly's robust
/// bisection ("Distance from a point to an ellipse, an ellipsoid, or a
/// hyperellipsoid", Geometric Tools), which needs no tolerance — it halves
/// until no double lies between its ends — on `u = s + 1` rather than his
/// `s`: near a principal plane the root lies a small coordinate's share of
/// its axis from the pole, where `s + 1` cancels and the foot `y / (s + 1)`
/// is lost.  A coordinate whose share of its axis (`y_i / e_i`) is below
/// the smallest normal double — where the bisection's steps are no longer
/// relative — counts as zero: it leaves the ellipse of the other two, or
/// the foot of a perpendicular off the plane.
pub fn inside_distance(e: [f64; 3], y: [f64; 3]) -> f64 {
    let [e0, e1, e2] = e;
    let [y0, y1, y2] = y;
    let [z0, z1, z2] = [y0 / e0, y1 / e1, y2 / e2];
    if z2 >= f64::MIN_POSITIVE {
        if z1 >= f64::MIN_POSITIVE {
            if z0 >= f64::MIN_POSITIVE {
                let g = z0 * z0 + z1 * z1 + z2 * z2 - 1.0;
                if g == 0.0 {
                    return 0.0;
                }
                let (q0, q1) = (e0 / e2, e1 / e2);
                let (r0, r1) = (q0 * q0, q1 * q1);
                let (gap0, gap1) = (pole_gap(e0, e2), pole_gap(e1, e2));
                let u = bisect(&[r0 * z0, r1 * z1], &[gap0, gap1], z2, g);
                let x = [r0 * y0 / (u + gap0), r1 * y1 / (u + gap1), y2 / u];
                let d = [x[0] - y0, x[1] - y1, x[2] - y2];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            } else {
                ellipse_distance(e1, e2, y1, y2)
            }
        } else if z0 >= f64::MIN_POSITIVE {
            ellipse_distance(e0, e2, y0, y2)
        } else {
            (y2 - e2).abs()
        }
    } else {
        let (denom0, denom1) = (squares_apart(e0, e2), squares_apart(e1, e2));
        let (numer0, numer1) = (e0 * y0, e1 * y1);
        if numer0 < denom0 && numer1 < denom1 {
            let (xde0, xde1) = (numer0 / denom0, numer1 / denom1);
            let discr = 1.0 - xde0 * xde0 - xde1 * xde1;
            if discr > 0.0 {
                let x = [e0 * xde0, e1 * xde1, e2 * discr.sqrt()];
                let d = [x[0] - y0, x[1] - y1];
                return (d[0] * d[0] + d[1] * d[1] + x[2] * x[2]).sqrt();
            }
        }
        ellipse_distance(e0, e1, y0, y1)
    }
}

/// [`inside_distance`] in two dimensions: `y` to the ellipse of semi-axes
/// `e0 >= e1`.
fn ellipse_distance(e0: f64, e1: f64, y0: f64, y1: f64) -> f64 {
    let (z0, z1) = (y0 / e0, y1 / e1);
    if z1 >= f64::MIN_POSITIVE {
        if z0 >= f64::MIN_POSITIVE {
            let g = z0 * z0 + z1 * z1 - 1.0;
            if g == 0.0 {
                return 0.0;
            }
            let q0 = e0 / e1;
            let r0 = q0 * q0;
            let gap0 = pole_gap(e0, e1);
            let u = bisect(&[r0 * z0], &[gap0], z1, g);
            let d = [r0 * y0 / (u + gap0) - y0, y1 / u - y1];
            (d[0] * d[0] + d[1] * d[1]).sqrt()
        } else {
            (y1 - e1).abs()
        }
    } else {
        let (numer0, denom0) = (e0 * y0, squares_apart(e0, e1));
        if numer0 < denom0 {
            let xde0 = numer0 / denom0;
            let x = [e0 * xde0, e1 * (1.0 - xde0 * xde0).sqrt()];
            let d0 = x[0] - y0;
            (d0 * d0 + x[1] * x[1]).sqrt()
        } else {
            (y0 - e0).abs()
        }
    }
}

/// `a² − b²` without the cancellation of squaring first.
fn squares_apart(a: f64, b: f64) -> f64 {
    (a - b) * (a + b)
}

/// How far below `u = 0` the pole of axis `a` lies against axis `b`:
/// `(a / b)² − 1`, without its cancellation when `a` is near `b`.
fn pole_gap(a: f64, b: f64) -> f64 {
    squares_apart(a, b) / (b * b)
}

/// `|v|`, scaled by its largest magnitude so no square overflows.
fn robust_length(v: &[f64]) -> f64 {
    let m = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if m > 0.0 {
        m * v.iter().fold(0.0, |s, x| s + (x / m) * (x / m)).sqrt()
    } else {
        0.0
    }
}

/// The root `u` of `sum_i (n_i / (u + gap_i))^2 + (z / u)^2 = 1` that
/// [`inside_distance`] bisects for: at least `z`, where the last term alone
/// is one, and at most 1 inside (`g < 0`), the length of `(n, z)` outside.
fn bisect(n: &[f64], gap: &[f64], z: f64, g: f64) -> f64 {
    let mut lo = z;
    let mut hi = if g < 0.0 {
        1.0
    } else {
        let mut v = n.to_vec();
        v.push(z);
        robust_length(&v)
    };
    let mut u = lo;
    for _ in 0..MAX_BISECTIONS {
        u = (lo + hi) / 2.0;
        if u == lo || u == hi {
            break;
        }
        let mut sum = 0.0;
        for (&ni, &gi) in n.iter().zip(gap) {
            let ratio = ni / (u + gi);
            sum += ratio * ratio;
        }
        let ratio = z / u;
        let g = sum + ratio * ratio - 1.0;
        if g > 0.0 {
            lo = u;
        } else if g < 0.0 {
            hi = u;
        } else {
            break;
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    fn soma(axes: [f64; 3]) -> Soma {
        Soma {
            center: [0.0; 3],
            axes,
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    #[test]
    fn a_sphere_measures_from_its_radius() {
        let s = soma([2.0; 3]);
        let d = s.distance_to_surface(&[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]);
        assert_eq!(d[..2], [-1.5, -2.0]);
        assert!((d[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inside_the_nearest_surface_is_not_along_the_ray() {
        // axes (2, 1, 1) at (1, 0.5, 0): the ray from the centre meets the
        // surface 0.463 away, the side of the ellipsoid is nearer
        let s = soma([2.0, 1.0, 1.0]);
        let d = -s.distance_to_surface(&[[1.0, 0.5, 0.0]])[0];
        let ray = (1.25f64).sqrt() * (1.0 / (0.5f64).sqrt() - 1.0);
        assert!(d < ray - 0.1, "{d} against the ray's {ray}");
        // no point of the surface is nearer than it
        let near = (0..20_000)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 20_000.0;
                let (x, y) = (2.0 * t.cos() - 1.0, t.sin() - 0.5);
                (x * x + y * y).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(near >= d - 1e-12 && near - d < 1e-6, "{near} against {d}");
    }

    #[test]
    fn points_on_the_axes_take_the_degenerate_branches() {
        let e = [3.0, 2.0, 1.0];
        assert_eq!(inside_distance(e, [0.0, 0.0, 0.5]), 0.5);
        assert_eq!(inside_distance(e, [0.0, 1.5, 0.0]), 0.5);
        // on the long axis the nearest point is off it, towards the short
        // one: from (2, 0, 0) the foot is (2.25, 0, √0.4375)
        let d = inside_distance(e, [0.5, 0.0, 0.0]);
        assert!(d < 1.0 && d > 0.9, "{d}");
        assert!((inside_distance(e, [2.0, 0.0, 0.0]) - 0.5f64.sqrt()).abs() < 1e-15);
        // until the axis' end is nearer
        assert!((inside_distance(e, [2.9, 0.0, 0.0]) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn near_a_principal_plane_the_distance_is_the_planes() {
        // a coordinate a rounding's share of its axis puts the root at the
        // pole, where Eberly's `s + 1` cancelled: at (1, 0.5, t) inside
        // (3, 2, 1) it gave inf at 1e-17, 0.50 at 1e-16, 0.92 at 1e-15 for
        // the plane's 0.89
        let e = [3.0, 2.0, 1.0];
        for flat in [[1.0, 0.5, 0.0], [0.0, 0.5, 0.25], [1.0, 0.0, 0.25]] {
            let on = inside_distance(e, flat);
            for t in [5e-324, 1e-310, 1e-300, 1e-17, 1e-16, 1e-15, 1e-12, 1e-9] {
                let near = flat.map(|v: f64| if v == 0.0 { t } else { v });
                let d = inside_distance(e, near);
                assert!(
                    d.is_finite() && (d - on).abs() < 1e-8,
                    "{near:?}: {d} against {on}"
                );
            }
        }
    }

    #[test]
    fn a_fit_recovers_the_ellipsoid_it_was_sampled_from() {
        // points on the axes of a rotated ellipsoid: their covariance has
        // the rotation's columns as eigenvectors
        let (c, s) = (0.6f64, 0.8f64);
        let rot = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let lengths = [5.0, 3.0, 1.0];
        let mut pts = Vec::new();
        for (k, &len) in lengths.iter().enumerate() {
            for sign in [-1.0, 1.0] {
                pts.push([0, 1, 2].map(|row| 10.0 + sign * len * rot[row][k]));
            }
        }
        let soma = fit(&pts).expect("a spread in every direction");
        assert!(soma.center.iter().all(|&x| (x - 10.0).abs() < 1e-12));
        // each axis' variance: two points at ±len over n − 1 = 5
        for (a, len) in soma.axes.iter().zip(lengths) {
            assert!((a - (5.0 * 2.0 * len * len / 5.0f64).sqrt()).abs() < 1e-12);
        }
        let columns = |m: [[f64; 3]; 3]| [0, 1, 2].map(|k| [m[0][k], m[1][k], m[2][k]]);
        for (k, (got, want)) in columns(soma.r).iter().zip(columns(rot)).enumerate() {
            let dot: f64 = got.iter().zip(want).map(|(a, b)| a * b).sum();
            assert!((dot.abs() - 1.0).abs() < 1e-12, "axis {k}: {dot}");
        }
        let det = {
            let r = soma.r;
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        };
        assert!(det > 0.0);
    }

    #[test]
    fn a_fit_refuses_too_few_flat_and_infinite_spreads() {
        assert_eq!(fit(&[[1.0, 2.0, 3.0]]), Err(FitError::TooFew(1)));
        let flat = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert_eq!(fit(&flat), Err(FitError::Flat));
        assert_eq!(
            fit(&[[0.0; 3], [f64::NAN, 0.0, 0.0]]),
            Err(FitError::NotFinite)
        );
    }

    #[test]
    fn the_map_to_the_unit_sphere_divides_by_column() {
        let s = Soma {
            center: [1.0, 2.0, 3.0],
            axes: [4.0, 2.0, 1.0],
            r: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        };
        let b = s.body();
        assert_eq!(b.w, [[0.0, 0.0, 1.0], [0.0, 0.5, 0.0], [-0.25, 0.0, 0.0]]);
        assert_eq!(
            s.contains(&[[1.0, 2.0, 3.0], [1.0, 2.0, 7.5]], 1.0),
            [true, false]
        );
        assert_eq!(s.spherical_radius(), 4.0);
    }
}
