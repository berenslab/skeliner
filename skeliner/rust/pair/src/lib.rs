//! `skeliner.pair`'s kernels: the faces of one mesh that touch another, how
//! far rays travel before they first hit a mesh, and the seeded path's
//! nearest points on a mesh (trimesh's `ProximityQuery.on_surface`).

#![forbid(unsafe_code)]

use mesh::ray::TriangleBvh;
use rayon::prelude::*;

/// trimesh's `ProximityQuery.on_surface` over triangles `tris` with their
/// face `normals`, for each of `points`: `(closest point, distance,
/// triangle)` as [`TriangleBvh::nearest_trimesh`] finds it, `None` without
/// triangles. Points in parallel.
pub fn closest_points(
    tris: Vec<[[f64; 3]; 3]>,
    normals: Vec<[f64; 3]>,
    points: &[[f64; 3]],
) -> Vec<Option<([f64; 3], f64, usize)>> {
    let bvh = TriangleBvh::new(tris, normals);
    threads::install(|| points.par_iter().map(|&p| bvh.nearest_trimesh(p)).collect())
}

/// The farthest any face's corner lies from its centre (`(0 + a + b + c) /
/// 3`, as trimesh places it): a face within `r` of a point has its centre
/// within `r` plus this. 0 without faces, NaN when a coordinate is NaN.
pub fn face_reach(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> f64 {
    fn max(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    }
    threads::install(|| {
        faces
            .par_iter()
            .map(|f| {
                let [p, q, r] = f.map(|v| verts[v as usize]);
                let c: [f64; 3] = [0, 1, 2].map(|k| (0.0 + p[k] + q[k] + r[k]) / 3.0);
                [p, q, r]
                    .into_iter()
                    .map(|v| {
                        let d = [v[0] - c[0], v[1] - c[1], v[2] - c[2]];
                        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                    })
                    .fold(0.0, max)
            })
            .reduce(|| 0.0, max)
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_faces_reach_is_its_farthest_corner() {
        let s3 = 3f64.sqrt();
        let verts = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, 1.5 * s3, 0.0],
            [9.0, 0.0, 0.0],
        ];
        // equilateral, side 3: each corner √3 from the centre
        let reach = face_reach(&verts, &[[0, 1, 2]]);
        assert!((reach - s3).abs() < 1e-12, "{reach}");
        // a sliver along x: its far end 6 from its centre (3, 0, 0)
        assert_eq!(face_reach(&verts, &[[0, 1, 2], [0, 0, 3]]), 6.0);
        assert_eq!(face_reach(&verts, &[]), 0.0);
        let nan = [[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(face_reach(&nan, &[[0, 1, 2]]).is_nan());
    }
}
