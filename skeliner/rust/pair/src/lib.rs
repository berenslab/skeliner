//! `skeliner.pair`'s kernels: the faces of one mesh that touch another, and
//! how far rays travel before they first hit a mesh.

#![forbid(unsafe_code)]

use mesh::ray::TriangleBvh;
use rayon::prelude::*;

/// Which faces of one mesh touch another: per face, the other mesh's
/// triangle closest to its `center` — of several equally close (meeting at
/// the closest point), the one whose normal most opposes the face's, then
/// the lowest index — kept when it lies within `tol` and `dot <= max_dot`.
/// Per face, `(other face, distance)` when kept.
pub fn contact_faces(
    centers: &[[f64; 3]],
    normals: &[[f64; 3]],
    triangles: Vec<[[f64; 3]; 3]>,
    other_normals: Vec<[f64; 3]>,
    tol: f64,
    max_dot: f64,
) -> Vec<Option<(usize, f64)>> {
    let bvh = TriangleBvh::new(triangles, other_normals);
    threads::install(|| {
        centers
            .par_iter()
            .zip(normals.par_iter())
            .map(|(&c, &n)| {
                let (_, d2) = bvh.closest_within(c, tol)?;
                let dot = |t: usize| {
                    let m = bvh.normal(t);
                    n[0] * m[0] + n[1] * m[1] + n[2] * m[2]
                };
                // the triangles that share the closest distance, rounding aside
                let tied = d2 + 1e-8;
                let (best_dot, best) = bvh
                    .within(c, tied.sqrt())
                    .into_iter()
                    .filter(|&(_, e2)| e2 <= tied)
                    .map(|(t, _)| (dot(t), t))
                    .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))?;
                (best_dot <= max_dot).then_some((best, d2.sqrt()))
            })
            .collect()
    })
}

/// How far each ray travels before it first hits a triangle of `bvh`: per
/// ray from `origins[k]` along `directions[k]`, the distance (ray parameter)
/// to the first triangle past `min_t`, leaving out triangle `skip[k]` (the
/// one the ray starts on; negative for none); NaN when it hits nothing.
pub fn first_hits(
    bvh: &TriangleBvh,
    origins: &[[f64; 3]],
    directions: &[[f64; 3]],
    skip: &[i64],
    min_t: f64,
) -> Vec<f64> {
    threads::install(|| {
        (0..origins.len())
            .into_par_iter()
            .map(|k| {
                let s = usize::try_from(skip[k]).ok();
                bvh.first_hit(origins[k], directions[k], min_t, s)
                    .map_or(f64::NAN, |(_, h)| h)
            })
            .collect()
    })
}
