//! Triangle meshes that need not be closed or manifold.
//!
//! A mesh is an indexed face set: vertex positions and vertex-index
//! triples.  Nothing here assumes an edge is shared by exactly two faces —
//! boundary edges (one face), manifold edges (two) and non-manifold edges
//! (three or more) are all ordinary.

#![forbid(unsafe_code)]

pub mod components;
pub mod derived;
pub mod geometry;
pub mod graph;
pub mod loops;
pub mod raster;
pub mod ray;
pub mod spatial;
pub mod topology;

use std::fmt;

pub use topology::{Neighbors, Topology};

/// Why a mesh was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// A face names a vertex that does not exist.
    VertexOutOfRange {
        face: usize,
        vertex: i64,
        n_vertices: usize,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexOutOfRange {
                face,
                vertex,
                n_vertices,
            } => write!(
                f,
                "face {face} names vertex {vertex}, but the mesh has {n_vertices} vertices"
            ),
        }
    }
}

impl std::error::Error for MeshError {}

/// An indexed triangle mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    vertices: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
}

/// Narrow face indices to `u32`, checking each names one of `n_vertices`.
pub fn checked_faces(faces: &[[i64; 3]], n_vertices: usize) -> Result<Vec<[u32; 3]>, MeshError> {
    let out_of_range = |v: i64| {
        u32::try_from(v)
            .ok()
            .filter(|&v| (v as usize) < n_vertices)
            .is_none()
    };
    let bad = threads::map(faces, |f| f.iter().any(|&v| out_of_range(v)));
    if let Some(face) = bad.iter().position(|&b| b) {
        let vertex = faces[face]
            .into_iter()
            .find(|&v| out_of_range(v))
            .expect("the face has one");
        return Err(MeshError::VertexOutOfRange {
            face,
            vertex,
            n_vertices,
        });
    }
    // every index fits a u32
    Ok(threads::map(faces, |f| f.map(|v| v as u32)))
}

impl Mesh {
    /// Build a mesh, checking that every face index names a vertex.
    pub fn new(vertices: Vec<[f64; 3]>, faces: &[[i64; 3]]) -> Result<Self, MeshError> {
        let faces = checked_faces(faces, vertices.len())?;
        Ok(Self { vertices, faces })
    }

    /// Vertex positions.
    pub fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }

    /// Vertex-index triples, one per face.
    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn keeps_what_it_is_given() {
        let mesh = Mesh::new(square(), &[[0, 1, 2], [0, 2, 3]]).unwrap();
        assert_eq!(mesh.vertices(), square().as_slice());
        assert_eq!(mesh.faces(), &[[0, 1, 2], [0, 2, 3]]);
    }

    #[test]
    fn refuses_an_index_past_the_last_vertex() {
        let err = Mesh::new(square(), &[[0, 1, 4]]).unwrap_err();
        assert_eq!(
            err,
            MeshError::VertexOutOfRange {
                face: 0,
                vertex: 4,
                n_vertices: 4
            }
        );
    }

    #[test]
    fn refuses_a_negative_index() {
        let err = Mesh::new(square(), &[[0, 1, 2], [-1, 1, 2]]).unwrap_err();
        assert!(matches!(
            err,
            MeshError::VertexOutOfRange {
                face: 1,
                vertex: -1,
                ..
            }
        ));
    }
}
