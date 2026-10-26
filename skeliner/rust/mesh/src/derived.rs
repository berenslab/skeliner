//! The arrays trimesh derives from a mesh's vertices and faces, computed as
//! trimesh 4.7 computes them, bit for bit: per face the cross product,
//! normal, area and centre, and the pairs of faces that share an edge.

/// trimesh's zero for a cross product's norm: `np.finfo(float64).resolution`
/// times 100.
pub const TOL_ZERO: f64 = 1e-15 * 100.0;

/// trimesh's `triangles_cross`: `(t1 − t0) × (t2 − t1)` per face, each
/// component numpy's product less a product.
pub fn face_crosses(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<[f64; 3]> {
    faces
        .iter()
        .map(|f| {
            let [p, q, r] = f.map(|v| verts[v as usize]);
            let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            let w = [r[0] - q[0], r[1] - q[1], r[2] - q[2]];
            [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ]
        })
        .collect()
}

/// A cross product's norm, `sqrt((x² + y²) + z²)`: trimesh's BLAS dot of the
/// squares with ones.
fn norm(c: &[f64; 3]) -> f64 {
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

/// trimesh's `face_normals` from the `crosses`: each times the reciprocal of
/// its norm; zero where the norm is not above [`TOL_ZERO`].
pub fn face_normals(crosses: &[[f64; 3]]) -> Vec<[f64; 3]> {
    crosses
        .iter()
        .map(|c| {
            let n = norm(c);
            if n > TOL_ZERO {
                let r = 1.0 / n;
                [c[0] * r, c[1] * r, c[2] * r]
            } else {
                [0.0; 3]
            }
        })
        .collect()
}

/// trimesh's `area_faces` from the `crosses`: half each one's norm.
pub fn face_areas(crosses: &[[f64; 3]]) -> Vec<f64> {
    crosses.iter().map(|c| norm(c) / 2.0).collect()
}

/// trimesh's `triangles_center`: each face's vertices summed in order from
/// zero, as numpy's mean sums them, over three.
pub fn face_centers(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<[f64; 3]> {
    faces
        .iter()
        .map(|f| {
            let [p, q, r] = f.map(|v| verts[v as usize]);
            [0, 1, 2].map(|k| (0.0 + p[k] + q[k] + r[k]) / 3.0)
        })
        .collect()
}

/// trimesh's `face_adjacency`: for every edge two face edges share — by its
/// higher end, then its lower — the two faces, lower first; an edge three
/// or more face edges share has none, and one a face shares with itself
/// (a repeated vertex) is left out.
///
/// Every index in `faces` must be below `n_vertices`.
pub fn adjacent_faces(faces: &[[u32; 3]], n_vertices: usize) -> Vec<[u32; 2]> {
    let ends = |f: &[u32; 3]| [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])];
    let mut start = vec![0usize; n_vertices + 1];
    for f in faces {
        for (a, b) in ends(f) {
            start[a.max(b) as usize + 1] += 1;
        }
    }
    for v in 0..n_vertices {
        start[v + 1] += start[v];
    }
    let mut fill = start.clone();
    let mut entries = vec![(0u32, 0u32); start[n_vertices]];
    for (fi, f) in faces.iter().enumerate() {
        for (a, b) in ends(f) {
            let hi = a.max(b) as usize;
            entries[fill[hi]] = (a.min(b), fi as u32);
            fill[hi] += 1;
        }
    }
    let mut out = Vec::new();
    for hi in 0..n_vertices {
        let bucket = &mut entries[start[hi]..start[hi + 1]];
        bucket.sort_unstable();
        let mut i = 0;
        while i < bucket.len() {
            let mut j = i + 1;
            while j < bucket.len() && bucket[j].0 == bucket[i].0 {
                j += 1;
            }
            if j - i == 2 && bucket[i].1 != bucket[i + 1].1 {
                out.push([bucket[i].1, bucket[i + 1].1]);
            }
            i = j;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
