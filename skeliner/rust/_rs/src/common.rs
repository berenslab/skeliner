//! Conversions between numpy arrays and the crates' types, shared by every
//! binding module.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Rows of an `(N, 3)` array, refusing any other shape.
pub(crate) fn rows3<T: Copy + numpy::Element>(
    name: &str,
    a: &PyReadonlyArray2<'_, T>,
) -> PyResult<Vec<[T; 3]>> {
    let v = a.as_array();
    if v.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "expected ({name}, 3) array, got shape {:?}",
            v.shape()
        )));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// Faces narrowed to `u32` and checked against the vertex count, which
/// defaults to one past the highest index.
pub(crate) fn checked_faces(
    faces: &PyReadonlyArray2<'_, i64>,
    n_vertices: Option<usize>,
) -> PyResult<(Vec<[u32; 3]>, usize)> {
    let rows = rows3("F", faces)?;
    let n = match n_vertices {
        Some(n) => n,
        None => rows
            .iter()
            .flatten()
            .max()
            .map_or(0, |&m| usize::try_from(m).map_or(0, |m| m + 1)),
    };
    let checked =
        ::mesh::checked_faces(&rows, n).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((checked, n))
}

/// Faces narrowed to `u32`, without a vertex count to check against.
pub(crate) fn faces_u32(faces: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<[u32; 3]>> {
    rows3("F", faces)?
        .iter()
        .map(|f| {
            let mut g = [0u32; 3];
            for (slot, &v) in g.iter_mut().zip(f) {
                *slot = u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("vertex index {v} is out of range"))
                })?;
            }
            Ok(g)
        })
        .collect()
}

/// Face ids checked against the face count.
pub(crate) fn face_ids(
    ids: &numpy::PyReadonlyArray1<'_, i64>,
    n_faces: usize,
) -> PyResult<Vec<u32>> {
    ids.as_array()
        .iter()
        .map(|&i| {
            u32::try_from(i)
                .ok()
                .filter(|&i| (i as usize) < n_faces)
                .ok_or_else(|| PyValueError::new_err(format!("face index {i} is out of range")))
        })
        .collect()
}

/// Face ids from any iterable of ints; negative ids name no face.
pub(crate) fn selection(sel: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let mut out = Vec::new();
    for item in sel.try_iter()? {
        let f: i64 = item?.extract()?;
        if let Ok(f) = u32::try_from(f) {
            out.push(f);
        }
    }
    Ok(out)
}

/// A mesh handed back to Python: `(vertices, faces)`.
pub(crate) type MeshArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

pub(crate) fn rows_to_array<T: Copy>(rows: &[[T; 3]]) -> Array2<T> {
    let flat: Vec<T> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), 3), flat).expect("three columns per row")
}

pub(crate) fn points3(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    rows3("N", a)
}

/// Triangles given as `(F, 3, 3)` flattened to `(F, 9)`.
pub(crate) fn triangles9(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[[f64; 3]; 3]>> {
    let t = a.as_array();
    if t.ncols() != 9 {
        return Err(PyValueError::new_err("triangles must be (F, 9)"));
    }
    Ok(t.rows()
        .into_iter()
        .map(|r| [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]])
        .collect())
}

pub(crate) fn vertex_ids(ids: &numpy::PyReadonlyArray1<'_, i64>, n: usize) -> PyResult<Vec<u32>> {
    ids.as_array()
        .iter()
        .map(|&v| {
            u32::try_from(v)
                .ok()
                .filter(|&v| (v as usize) < n)
                .ok_or_else(|| PyValueError::new_err(format!("vertex index {v} is out of range")))
        })
        .collect()
}

/// Skeleton edges `(E, 2)` checked against the node count `n`.
pub(crate) fn node_pairs(edges: &PyReadonlyArray2<'_, i64>, n: usize) -> PyResult<Vec<[usize; 2]>> {
    let e = edges.as_array();
    if e.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "expected (E, 2) edges, got shape {:?}",
            e.shape()
        )));
    }
    let node = |v: i64| {
        usize::try_from(v)
            .ok()
            .filter(|&v| v < n)
            .ok_or_else(|| PyValueError::new_err(format!("edge names node {v}, not one of {n}")))
    };
    e.rows()
        .into_iter()
        .map(|r| Ok([node(r[0])?, node(r[1])?]))
        .collect()
}
