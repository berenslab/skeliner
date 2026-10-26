//! `skeliner.pre.compact_mesh`'s arrays: the mesh without its collapsed
//! faces and without the vertices no face left names.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

/// What [`compact`] made of a mesh.
pub struct Compacted {
    /// The faces kept: all but the collapsed ones, `[v, v, v]`.
    pub good: Vec<bool>,
    /// Each vertex's new id, −1 for one no kept face names.
    pub vert_map: Vec<i64>,
    /// Each face's new id, −1 for a collapsed one.
    pub face_map: Vec<i64>,
    /// The kept vertices, in their order.
    pub vertices: Vec<[f64; 3]>,
    /// The kept faces, renumbered.
    pub faces: Vec<[u32; 3]>,
}

/// Every index in `faces` must name one of `vertices`.
pub fn compact(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Compacted {
    threads::install(|| {
        let good: Vec<bool> = faces
            .par_iter()
            .map(|f| !(f[1] == f[0] && f[2] == f[0]))
            .collect();
        let named: Vec<AtomicBool> = (0..vertices.len())
            .into_par_iter()
            .map(|_| AtomicBool::new(false))
            .collect();
        faces.par_iter().zip(&good).for_each(|(f, &g)| {
            if g {
                for &v in f {
                    named[v as usize].store(true, Ordering::Relaxed);
                }
            }
        });
        let named: Vec<bool> = named.into_par_iter().map(AtomicBool::into_inner).collect();
        let vert_map = numbered(&named);
        let face_map = numbered(&good);
        let vertices = vertices
            .par_iter()
            .zip(&named)
            .filter(|&(_, &n)| n)
            .map(|(v, _)| *v)
            .collect();
        let faces = faces
            .par_iter()
            .zip(&good)
            .filter(|&(_, &g)| g)
            .map(|(f, _)| f.map(|v| vert_map[v as usize] as u32))
            .collect();
        Compacted {
            good,
            vert_map,
            face_map,
            vertices,
            faces,
        }
    })
}

/// Each kept item's place among the kept ones, −1 for the rest.
fn numbered(kept: &[bool]) -> Vec<i64> {
    let mut next = 0;
    kept.iter()
        .map(|&k| {
            if k {
                next += 1;
                next - 1
            } else {
                -1
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapsed_faces_and_their_lone_vertices_go() {
        let vertices: Vec<[f64; 3]> = (0..6).map(|i| [f64::from(i), 0.0, 0.0]).collect();
        // face 1 is collapsed, so vertex 4 is named by none left; vertex 3
        // by none at all; face 2 has a repeated vertex but stays
        let faces = [[0, 1, 2], [4, 4, 4], [2, 2, 5], [5, 1, 0]];
        let c = compact(&vertices, &faces);
        assert_eq!(c.good, [true, false, true, true]);
        assert_eq!(c.vert_map, [0, 1, 2, -1, -1, 3]);
        assert_eq!(c.face_map, [0, -1, 1, 2]);
        assert_eq!(
            c.vertices,
            [[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
        );
        assert_eq!(c.faces, [[0, 1, 2], [2, 2, 3], [3, 1, 0]]);
        let none = compact(&vertices, &[]);
        assert!(none.vertices.is_empty() && none.faces.is_empty());
        assert_eq!(none.vert_map, [-1; 6]);
    }
}
