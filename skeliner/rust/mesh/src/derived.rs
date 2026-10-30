//! The arrays trimesh derives from a mesh's vertices and faces, computed as
//! trimesh 4.7 computes them, bit for bit: per face the cross product,
//! normal, area and centre, and the pairs of faces that share an edge.

use rayon::prelude::*;

/// trimesh's zero for a cross product's norm: `np.finfo(float64).resolution`
/// times 100.
pub const TOL_ZERO: f64 = 1e-15 * 100.0;

/// trimesh's `triangles_cross`: `(t1 − t0) × (t2 − t1)` per face, each
/// component numpy's product less a product.
pub fn face_crosses(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<[f64; 3]> {
    threads::map(faces, |f| {
        let [p, q, r] = f.map(|v| verts[v as usize]);
        let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let w = [r[0] - q[0], r[1] - q[1], r[2] - q[2]];
        [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ]
    })
}

/// A cross product's norm, `sqrt((x² + y²) + z²)`: trimesh's BLAS dot of the
/// squares with ones.
fn norm(c: &[f64; 3]) -> f64 {
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

/// trimesh's `face_normals` from the `crosses`: each times the reciprocal of
/// its norm; zero where the norm is not above [`TOL_ZERO`].
pub fn face_normals(crosses: &[[f64; 3]]) -> Vec<[f64; 3]> {
    threads::map(crosses, |c| {
        let n = norm(c);
        if n > TOL_ZERO {
            let r = 1.0 / n;
            [c[0] * r, c[1] * r, c[2] * r]
        } else {
            [0.0; 3]
        }
    })
}

/// trimesh's `area_faces` from the `crosses`: half each one's norm.
pub fn face_areas(crosses: &[[f64; 3]]) -> Vec<f64> {
    threads::map(crosses, |c| norm(c) / 2.0)
}

/// trimesh's `triangles_center`: each face's vertices summed in order from
/// zero, as numpy's mean sums them, over three.
pub fn face_centers(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<[f64; 3]> {
    threads::map(faces, |f| {
        let [p, q, r] = f.map(|v| verts[v as usize]);
        [0, 1, 2].map(|k| (0.0 + p[k] + q[k] + r[k]) / 3.0)
    })
}

/// `sk.Mesh.triangles` into `out`, nine values a face: each face's three
/// vertices.  `out` holds `9 * faces.len()` values.
pub fn fill_triangles(verts: &[[f64; 3]], faces: &[[u32; 3]], out: &mut [f64]) {
    let put = |(slot, f): (&mut [f64], &[u32; 3])| {
        for (k, &v) in f.iter().enumerate() {
            slot[3 * k..3 * k + 3].copy_from_slice(&verts[v as usize]);
        }
    };
    if faces.len() < threads::MIN_PARALLEL {
        out.chunks_mut(9).zip(faces).for_each(put);
    } else {
        threads::install(|| out.par_chunks_mut(9).zip(faces.par_iter()).for_each(put));
    }
}

/// trimesh's `submesh(..., append=True)` (`sk.Mesh.submesh`) of `faces`: the
/// vertices they name, ascending, and the faces renumbered to them, in
/// order (a face naming one vertex twice is kept).
pub fn submesh(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut named = vec![false; verts.len()];
    for &v in faces.as_flattened() {
        named[v as usize] = true;
    }
    let mut renumber = vec![u32::MAX; verts.len()];
    let mut kept = Vec::new();
    for (v, _) in named.iter().enumerate().filter(|&(_, &n)| n) {
        renumber[v] = kept.len() as u32;
        kept.push(verts[v]);
    }
    let faces = threads::map(faces, |f| f.map(|v| renumber[v as usize]));
    (kept, faces)
}

/// trimesh's `face_adjacency`: for every edge two face edges share — by its
/// higher end, then its lower — the two faces, lower first; an edge three
/// or more face edges share has none, and one a face shares with itself
/// (a repeated vertex) is left out.
///
/// Every index in `faces` must be below `n_vertices`.
pub fn adjacent_faces(faces: &[[u32; 3]], n_vertices: usize) -> Vec<[u32; 2]> {
    debug_assert!(faces.iter().flatten().all(|&v| (v as usize) < n_vertices));
    threads::install(|| {
        // each face edge as (high, low, face), sorted: equal entries are
        // equal values, so the order is the same however it is sorted
        let mut entries: Vec<(u64, u32)> = faces
            .par_iter()
            .enumerate()
            .flat_map_iter(|(fi, f)| {
                [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])].map(|(a, b)| {
                    let key = (u64::from(a.max(b)) << 32) | u64::from(a.min(b));
                    (key, fi as u32)
                })
            })
            .collect();
        entries.par_sort_unstable();
        entries
            .par_chunk_by(|x, y| x.0 == y.0)
            .filter_map(|group| match group {
                [(_, a), (_, b)] if a != b => Some([*a, *b]),
                _ => None,
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_submesh_keeps_the_vertices_its_faces_name_in_order() {
        let verts: Vec<[f64; 3]> = (0..6).map(|i| [i as f64, 0.0, 0.0]).collect();
        let (v, f) = submesh(&verts, &[[5, 1, 3], [3, 3, 3], [1, 5, 5]]);
        assert_eq!(v, [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
        assert_eq!(f, [[2, 0, 1], [1, 1, 1], [0, 2, 2]]);
        let mut out = [0.0; 9];
        fill_triangles(&verts, &[[2, 0, 4]], &mut out);
        assert_eq!(out, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn a_square_of_two_triangles() {
        let v = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let f = [[0, 1, 2], [0, 2, 3], [1, 1, 3]];
        let c = face_crosses(&v, &f);
        assert_eq!(c[0], [0.0, 0.0, 4.0]);
        assert_eq!(
            face_normals(&c),
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0; 3]]
        );
        assert_eq!(face_areas(&c), [2.0, 2.0, 0.0]);
        assert_eq!(face_centers(&v, &f)[0], [4.0 / 3.0, 2.0 / 3.0, 0.0]);
        // the diagonal 0-2 joins the two triangles; 1-3 is the third face's
        // edge twice over, a face beside itself
        assert_eq!(adjacent_faces(&f, 4), [[0, 1]]);
    }

    #[test]
    fn many_adjacent_faces_are_what_bucketing_by_high_end_gives() {
        // random faces on 5,000 vertices: edges shared by one, two or more
        // faces, and faces with a repeated vertex
        let mut state = 0x510e_527f_ade6_82d1u64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % n) as u32
        };
        let n = 5000u32;
        let faces: Vec<[u32; 3]> = (0..60_000)
            .map(|_| {
                let a = next(u64::from(n));
                let b = next(u64::from(n));
                [a, if next(40) == 0 { a } else { b }, (a + 1 + next(3)) % n]
            })
            .collect();
        let mut buckets = vec![Vec::new(); n as usize];
        for (fi, f) in faces.iter().enumerate() {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                buckets[a.max(b) as usize].push((a.min(b), fi as u32));
            }
        }
        let mut bucketed = Vec::new();
        for bucket in &mut buckets {
            bucket.sort_unstable();
            for group in bucket.chunk_by(|x, y| x.0 == y.0) {
                if let [(_, a), (_, b)] = group
                    && a != b
                {
                    bucketed.push([*a, *b]);
                }
            }
        }
        assert_eq!(adjacent_faces(&faces, n as usize), bucketed);
        assert!(bucketed.len() > 1000);
    }
}
