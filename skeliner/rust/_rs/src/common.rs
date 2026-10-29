//! Conversions between numpy arrays and the crates' types, shared by every
//! binding module.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Rows of an `(N, 3)` array, refusing any other shape; one copy when the
/// array is C-contiguous.
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
    Ok(match v.as_slice() {
        Some(flat) => flat.as_chunks::<3>().0.to_vec(),
        None => v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect(),
    })
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
        None => threads::map(&rows, |r| r[0].max(r[1]).max(r[2]))
            .into_iter()
            .max()
            .map_or(0, |m| usize::try_from(m).map_or(0, |m| m + 1)),
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
    let flat: Vec<T> = rows.as_flattened().to_vec();
    Array2::from_shape_vec((rows.len(), 3), flat).expect("three columns per row")
}

pub(crate) fn points3(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    rows3("N", a)
}

/// `points`, refused with `what must be finite` when a coordinate is NaN or
/// infinite (scipy's trees refuse such queries).
pub(crate) fn finite(what: &str, points: Vec<[f64; 3]>) -> PyResult<Vec<[f64; 3]>> {
    if points.iter().flatten().all(|x| x.is_finite()) {
        Ok(points)
    } else {
        Err(PyValueError::new_err(format!("{what} must be finite")))
    }
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
    match e.as_slice() {
        // the error naming the first bad id in row order
        Some(flat) => {
            let ok = threads::map(flat, |&v| node(v).is_ok());
            if let Some(i) = ok.iter().position(|&ok| !ok) {
                return Err(node(flat[i]).expect_err("a bad id"));
            }
            Ok(threads::map(flat.as_chunks::<2>().0, |r| {
                [r[0] as usize, r[1] as usize]
            }))
        }
        None => e
            .rows()
            .into_iter()
            .map(|r| Ok([node(r[0])?, node(r[1])?]))
            .collect(),
    }
}

/// A CSR (`ptr` `(K + 1,)` from 0, never decreasing, `idx`) as `K` lists of
/// ids, each one `0 ≤ id < bound` when there is a bound, else `0 ≤ id`;
/// `what` names the lists in the errors.
pub(crate) fn csr_lists(
    ptr: &numpy::PyReadonlyArray1<'_, i64>,
    idx: &numpy::PyReadonlyArray1<'_, i64>,
    bound: Option<usize>,
    what: &str,
) -> PyResult<Vec<Vec<usize>>> {
    let (ptr, idx) = csr_flat(ptr, idx, bound, what)?;
    Ok(ptr.windows(2).map(|w| idx[w[0]..w[1]].to_vec()).collect())
}

/// [`csr_lists`] left as one CSR, `ptr` and the checked `idx`.
pub(crate) fn csr_flat(
    ptr: &numpy::PyReadonlyArray1<'_, i64>,
    idx: &numpy::PyReadonlyArray1<'_, i64>,
    bound: Option<usize>,
    what: &str,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let (ptr, idx) = (ptr.as_array(), idx.as_array());
    let well_formed = !ptr.is_empty()
        && ptr[0] == 0
        && ptr.iter().zip(ptr.iter().skip(1)).all(|(a, b)| a <= b)
        && usize::try_from(ptr[ptr.len() - 1]).ok() == Some(idx.len());
    if !well_formed {
        return Err(PyValueError::new_err(format!(
            "{what} must be a CSR (ptr, idx)"
        )));
    }
    let id = |v: i64| {
        usize::try_from(v)
            .ok()
            .filter(|&v| bound.is_none_or(|n| v < n))
            .ok_or_else(|| match bound {
                Some(n) => PyValueError::new_err(format!(
                    "{what} name vertex {v}, but there are {n} vertices"
                )),
                None => PyValueError::new_err("vertex ids must not be negative"),
            })
    };
    let ptr: Vec<usize> = ptr.iter().map(|&p| p as usize).collect();
    let idx: Vec<i64> = idx.to_vec();
    // the error naming the first bad id
    let ok = threads::map(&idx, |&v| id(v).is_ok());
    if let Some(i) = ok.iter().position(|&ok| !ok) {
        return Err(id(idx[i]).expect_err("a bad id"));
    }
    Ok((ptr, threads::map(&idx, |&v| v as usize)))
}
