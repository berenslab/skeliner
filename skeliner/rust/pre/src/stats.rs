//! Per-face statistics the organelle and gap steps share.

use mesh::geometry::HAS_FMA;
use mesh::spatial::{KdTree, Sum};
use rayon::prelude::*;

/// Outward score of one face against the vertices `near` (ascending): the
/// face normal dotted with the unit direction from their centre of mass to
/// the face centre; 0 when fewer than four vertices are near or the
/// direction vanishes.
///
/// Rounded as the reference rounds it: the centre of mass accumulated in
/// ascending vertex order (scipy's sorted CSR rows), the dot summed left to
/// right.
fn face_dot(center: &[f64; 3], normal: &[f64; 3], verts: &[[f64; 3]], near: &[u32]) -> f64 {
    if near.len() < 4 {
        return 0.0;
    }
    let w = 1.0 / near.len() as f64;
    let mut com = [0.0f64; 3];
    for &j in near {
        let v = &verts[j as usize];
        for k in 0..3 {
            // scipy's CSR product compiles to fused multiply-adds where the
            // hardware has them
            com[k] = if HAS_FMA {
                w.mul_add(v[k], com[k])
            } else {
                com[k] + w * v[k]
            };
        }
    }
    let mut dir = [center[0] - com[0], center[1] - com[1], center[2] - com[2]];
    let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if norm <= 1e-10 {
        return 0.0;
    }
    for d in &mut dir {
        *d /= norm;
    }
    normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2]
}

/// `skeliner.pre._outward_dot`: per face, the outward score against the
/// vertices of its own component within `radius` of the face centre.
///
/// `vert_comp` labels vertices; a face belongs to the component of its first
/// vertex, and components with fewer than four vertices are skipped (their
/// faces score 0).  Without labels all vertices form one component.
pub fn outward_dots(
    verts: &[[f64; 3]],
    faces: &[[u32; 3]],
    centers: &[[f64; 3]],
    normals: &[[f64; 3]],
    vert_comp: Option<&[i64]>,
    radius: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; faces.len()];
    let one_component;
    let labels: &[i64] = match vert_comp {
        Some(l) => l,
        None => {
            one_component = vec![0i64; verts.len()];
            &one_component
        }
    };
    // Vertices and faces grouped by label (offset by the lowest, as labels
    // go negative), each group in ascending order.
    let lowest = labels.iter().copied().min().unwrap_or(0);
    let span = labels
        .iter()
        .copied()
        .max()
        .map_or(0, |m| (m - lowest + 1) as usize);
    let mut verts_of: Vec<Vec<usize>> = vec![Vec::new(); span];
    for (v, &c) in labels.iter().enumerate() {
        verts_of[(c - lowest) as usize].push(v);
    }
    let mut faces_of: Vec<Vec<usize>> = vec![Vec::new(); span];
    for (fi, f) in faces.iter().enumerate() {
        faces_of[(labels[f[0] as usize] - lowest) as usize].push(fi);
    }
    for (members, face_ids) in verts_of.iter().zip(&faces_of) {
        if face_ids.is_empty() || members.len() < 4 {
            continue;
        }
        let tree = KdTree::new(members.iter().map(|&v| verts[v]).collect(), Sum::Plain);
        let dots: Vec<f64> = threads::install(|| {
            face_ids
                .par_iter()
                .map_init(Vec::new, |near, &fi| {
                    near.clear();
                    tree.within_into(&centers[fi], radius, near);
                    near.sort_unstable();
                    face_dot(&centers[fi], &normals[fi], tree.points(), near)
                })
                .collect()
        });
        for (&fi, d) in face_ids.iter().zip(dots) {
            out[fi] = d;
        }
    }
    out
}

/// Vertex labels from face labels, as `skeliner.pre.compute_mesh_stats`
/// writes them: each vertex takes the label of the last labelled face (in
/// face order) that uses it, -1 if none.
pub fn vertex_components(faces: &[[u32; 3]], face_comp: &[i64], n_vertices: usize) -> Vec<i64> {
    let mut out = vec![-1i64; n_vertices];
    for (f, &c) in faces.iter().zip(face_comp) {
        if c >= 0 {
            for &v in f {
                out[v as usize] = c;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_face_on_the_outside_of_a_blob_points_away_from_its_centre() {
        // five vertices around the origin, a face out at +x facing +x
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ];
        let dot = face_dot(&[2.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &verts, &[0, 1, 2, 3, 4]);
        assert!(dot > 0.9);
        let inward = face_dot(
            &[2.0, 0.0, 0.0],
            &[-1.0, 0.0, 0.0],
            &verts,
            &[0, 1, 2, 3, 4],
        );
        assert_eq!(inward, -dot);
    }

    #[test]
    fn fewer_than_four_neighbours_score_zero() {
        let verts = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let d = face_dot(&[0.3, 0.3, 0.0], &[0.0, 0.0, 1.0], &verts, &[0, 1, 2]);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn vertex_labels_take_the_last_face() {
        let faces = [[0, 1, 2], [2, 3, 4], [4, 5, 6]];
        assert_eq!(
            vertex_components(&faces, &[0, 1, -2], 8),
            vec![0, 0, 1, 1, 1, -1, -1, -1]
        );
    }
}
