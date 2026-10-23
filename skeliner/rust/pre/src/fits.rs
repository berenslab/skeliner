//! What the ports leave to numpy.  Its SVD runs on the platform's LAPACK
//! (Accelerate here), which no Rust solver reproduces bit for bit — and a
//! plane or axis off by an ulp moves areas and signs across thresholds.

/// The numpy calls a stage needs, made by the caller.
pub trait Svd {
    /// `np.linalg.svd(x, full_matrices=full_matrices)[2]` for `x` of three
    /// or more rows: right-singular vectors as rows, largest first.
    fn right_singular(&self, x: &[[f64; 3]], full_matrices: bool) -> [[f64; 3]; 3];

    /// `find_pocket_mouths`' opening when no rim loop closes: scipy's convex
    /// hull area of `pts` projected onto their best-fit plane, 0 when qhull
    /// fails.
    fn hull_area(&self, pts: &[[f64; 3]]) -> f64;
}

/// A pure-Rust stand-in: Jacobi eigenvectors of the scatter matrix for the
/// SVD and a monotone-chain hull.  Agrees with numpy to an ulp or so, not to
/// the bit — for callers without numpy, and for tests.
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

    fn hull_area(&self, pts: &[[f64; 3]]) -> f64 {
        let (origin, u, w) = mesh::geometry::best_fit_plane(pts);
        mesh::geometry::convex_hull_area(&mesh::geometry::project(pts, origin, u, w))
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
