//! What the ports leave to numpy.  Its SVD runs on the platform's LAPACK
//! (Accelerate here), which no Rust solver reproduces bit for bit — and a
//! plane or axis off by an ulp moves areas and signs across thresholds.

/// The numpy calls a stage needs, made by the caller.
pub trait Svd {
    /// `np.linalg.svd(x, full_matrices=full_matrices)[2]` for `x` of three
    /// or more rows: right-singular vectors as rows, largest first.
    fn right_singular(&self, x: &[[f64; 3]], full_matrices: bool) -> [[f64; 3]; 3];

    /// [`Svd::right_singular`] of each of `xs`, in order: one call each,
    /// unless the caller makes them at once.
    fn right_singular_each(&self, xs: &[&[[f64; 3]]], full_matrices: bool) -> Vec<[[f64; 3]; 3]> {
        xs.iter()
            .map(|x| self.right_singular(x, full_matrices))
            .collect()
    }

    /// `find_pocket_mouths`' opening when no rim loop closes: the convex-hull
    /// area of `pts` centred on their mean, on the plane of their first two
    /// right-singular vectors — 0 where qhull, which scipy's `ConvexHull`
    /// ran, refuses them as flat ([`mesh::geometry::hull_measures`]).
    fn hull_area(&self, pts: &[[f64; 3]]) -> f64 {
        let centred = centred(pts);
        hull_area_on(&centred, self.right_singular(&centred, true))
    }
}

/// `pts` less their mean, as [`Svd::hull_area`] takes them.
pub fn centred(pts: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let n = pts.len() as f64;
    let mut sum = [0.0f64; 3];
    for p in pts {
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    pts.iter()
        .map(|p| [0, 1, 2].map(|k| p[k] - sum[k] / n))
        .collect()
}

/// [`Svd::hull_area`] of [`centred`] points whose right-singular vectors
/// are `vt`.
pub fn hull_area_on(centred: &[[f64; 3]], vt: [[f64; 3]; 3]) -> f64 {
    let plane = mesh::geometry::project(centred, [0.0; 3], vt[0], vt[1]);
    mesh::geometry::hull_measures(&plane).0
}

/// A pure-Rust stand-in: Jacobi eigenvectors of the scatter matrix for the
/// SVD.  Agrees with numpy to an ulp or so, not to the bit — for callers
/// without numpy, and for tests.
pub struct JacobiSvd;

impl Svd for JacobiSvd {
    fn right_singular(&self, x: &[[f64; 3]], _full_matrices: bool) -> [[f64; 3]; 3] {
        let mut cov = [[0.0f64; 3]; 3];
        for d in x {
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += d[i] * d[j];
                }
            }
        }
        mesh::geometry::symmetric_eigen(cov).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stand_in_finds_a_plane_and_its_area() {
        // a unit square, centred, in a plane tilted about x
        let (c, s) = (0.6, 0.8);
        let sq = [
            [-0.5, -0.5 * c, -0.5 * s],
            [0.5, -0.5 * c, -0.5 * s],
            [0.5, 0.5 * c, 0.5 * s],
            [-0.5, 0.5 * c, 0.5 * s],
        ];
        let vt = JacobiSvd.right_singular(&sq, false);
        let normal = vt[2];
        let dot = normal[1] * -s + normal[2] * c;
        assert!((dot.abs() - 1.0).abs() < 1e-12, "{normal:?}");
        assert!((JacobiSvd.hull_area(&sq) - 1.0).abs() < 1e-12);
    }
}
