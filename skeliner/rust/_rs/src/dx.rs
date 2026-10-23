//! `skeliner._rs.dx`: the `dx` crate's kernels for `skeliner.dx`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::*;

/// `skeliner.dx.distance` in centreline mode against a whitelist: per point,
/// the distance to the nearest of the `centres` (node ids) or of the
/// `segments` (node-id pairs), each computed as `_point_segment_distance`
/// computes it; infinity when there are neither.
#[pyfunction]
fn segment_distances<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    nodes: PyReadonlyArray2<'py, f64>,
    centres: Vec<usize>,
    segments: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pts = rows3("P", &points)?;
    let nodes = rows3("N", &nodes)?;
    if centres
        .iter()
        .chain(segments.iter().flat_map(|(a, b)| [a, b]))
        .any(|&i| i >= nodes.len())
    {
        return Err(PyValueError::new_err("node index out of range"));
    }
    let out = py.detach(|| ::dx::segment_distances(&pts, &nodes, &centres, &segments));
    Ok(Array1::from(out).into_pyarray(py))
}

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(segment_distances, m)?)?;
    Ok(())
}
