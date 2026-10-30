//! The fits `pre`'s stages make: a point set's principal directions
//! ([`mesh::geometry::right_singular`], numpy's SVD up to signs) and an
//! opening's area on their plane.

/// `pts` less their mean, as [`hull_area_on`] takes them.
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

/// `find_pocket_mouths`' opening when no rim loop closes: the convex-hull
/// area of [`centred`] points on the plane of their first two
/// right-singular vectors `vt` — 0 where qhull, which scipy's `ConvexHull`
/// ran, refuses them as flat ([`mesh::geometry::hull_measures`]).
pub fn hull_area_on(centred: &[[f64; 3]], vt: [[f64; 3]; 3]) -> f64 {
    let plane = mesh::geometry::project(centred, [0.0; 3], vt[0], vt[1]);
    mesh::geometry::hull_measures(&plane).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tilted_square_opens_its_own_area() {
        // a unit square, centred, in a plane tilted about x
        let (c, s) = (0.6, 0.8);
        let sq = [
            [-0.5, -0.5 * c, -0.5 * s],
            [0.5, -0.5 * c, -0.5 * s],
            [0.5, 0.5 * c, 0.5 * s],
            [-0.5, 0.5 * c, 0.5 * s],
        ];
        let centred = centred(&sq);
        let vt = mesh::geometry::right_singular(&centred);
        assert!((hull_area_on(&centred, vt) - 1.0).abs() < 1e-12);
    }
}
