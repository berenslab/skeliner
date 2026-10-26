//! `skeliner._rs.dataclass`: the `dataclass` crate's kernels for
//! `skeliner._state`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::common::*;

/// `vert2node` from the bins (a CSR `ptr`, `idx`, nodes in order): `(keys,
/// node)` — each vertex the first time a bin lists it, and by vertex the
/// node of the last bin listing it (−1 for none), `n_vertices` long (one
/// past the highest listed when not given).
#[pyfunction]
#[pyo3(signature = (ptr, idx, n_vertices=None))]
#[allow(clippy::type_complexity)]
fn vertex_map<'py>(
    py: Python<'py>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    n_vertices: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let bins = csr_lists(&ptr, &idx, n_vertices, "the bins")?;
    let n = n_vertices.unwrap_or_else(|| bins.iter().flatten().max().map_or(0, |&v| v + 1));
    let bins: Vec<&[usize]> = bins.iter().map(Vec::as_slice).collect();
    let (keys, node) = py.detach(|| ::dataclass::state::vertex_map(&bins, n));
    let keys = Array1::from_iter(keys.into_iter().map(|v| v as i64));
    Ok((keys.into_pyarray(py), Array1::from(node).into_pyarray(py)))
}

/// Add this module's functions to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vertex_map, m)?)?;
    Ok(())
}
