//! `skeliner._rs.dataclass`: the `dataclass` crate's kernels — the soma's
//! ellipsoid (`skeliner.dataclass.Soma`) and `skeliner._state`.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::*;

/// The soma's ellipsoid from its centre, semi-axes and rotation `(3, 3)`.
fn soma_of(
    center: [f64; 3],
    axes: [f64; 3],
    rotation: &PyReadonlyArray2<'_, f64>,
) -> PyResult<::dataclass::soma::Soma> {
    let r = rows3("R", rotation)?;
    let [r0, r1, r2] = r.as_slice() else {
        return Err(PyValueError::new_err("the rotation must be (3, 3)"));
    };
    Ok(::dataclass::soma::Soma {
        center,
        axes,
        r: [*r0, *r1, *r2],
    })
}

/// `Soma.contains`: which of `points` `(N, 3)` lie inside the ellipsoid
/// (`center`, `axes`, `rotation`), their body coordinates' squared length
/// at most `limit`.
#[pyfunction]
fn soma_contains<'py>(
    py: Python<'py>,
    center: [f64; 3],
    axes: [f64; 3],
    rotation: PyReadonlyArray2<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    limit: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let soma = soma_of(center, axes, &rotation)?;
    let points = points3(&points)?;
    let inside = py.detach(|| soma.contains(&points, limit));
    Ok(Array1::from(inside).into_pyarray(py))
}

/// `Soma.distance_to_center`: each of `points` `(N, 3)`'s distance to
/// `center`.
#[pyfunction]
fn soma_distance_to_center<'py>(
    py: Python<'py>,
    center: [f64; 3],
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let points = points3(&points)?;
    let d = py.detach(|| ::dataclass::soma::distance_to_center(center, &points));
    Ok(Array1::from(d).into_pyarray(py))
}

/// `Soma.distance_to_surface`: each of `points` `(N, 3)`'s signed distance
/// to the ellipsoid's surface, negative inside — Newton's steps outside, all
/// stopped together once every one is below `atol`, `max_iter` at most;
/// Eberly's bisection inside.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn soma_distance_to_surface<'py>(
    py: Python<'py>,
    center: [f64; 3],
    axes: [f64; 3],
    rotation: PyReadonlyArray2<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    atol: f64,
    max_iter: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let soma = soma_of(center, axes, &rotation)?;
    let points = points3(&points)?;
    let steps = usize::try_from(max_iter).unwrap_or(0);
    let d = py.detach(|| soma.distance_to_surface_with(&points, atol, steps));
    Ok(Array1::from(d).into_pyarray(py))
}

/// Two int64 arrays: a CSR's `(ptr, idx)`, or a map's keys and values.
type IdPair<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// `Skeleton`'s adjacency: each of `n_nodes` nodes' neighbours as a CSR
/// `(ptr, idx)`, in the order `edges` `(E, 2)` name them, an edge's first
/// end before its second.
#[pyfunction]
fn node_neighbors<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
) -> PyResult<IdPair<'py>> {
    let edges = node_pairs(&edges, n_nodes)?;
    let (ptr, idx) = py.detach(|| ::dataclass::skeleton::node_neighbors(&edges, n_nodes));
    let as_i64 = |v: Vec<usize>| Array1::from_iter(v.into_iter().map(|x| x as i64));
    Ok((as_i64(ptr).into_pyarray(py), as_i64(idx).into_pyarray(py)))
}

/// `_state.remap_edges`: the edges `(E, 2)` between the nodes `old2new`
/// keeps (`>= 0`), renumbered by it, each sorted, no self-loops, unique and
/// in order.
#[pyfunction]
fn remap_edges<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    old2new: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let old2new = old2new.as_array().to_vec();
    let edges = node_pairs(&edges, old2new.len())?;
    let out = py.detach(|| ::dataclass::state::remap_edges(&edges, &old2new));
    let flat: Vec<i64> = out.as_flattened().iter().map(|&v| v as i64).collect();
    let out = Array2::from_shape_vec((out.len(), 2), flat).expect("two columns per row");
    Ok(out.into_pyarray(py))
}

/// `vert2node` from the bins (a CSR `ptr`, `idx`, nodes in order): `(keys,
/// node)` — each vertex the first time a bin lists it, and by vertex the
/// node of the last bin listing it (−1 for none), `n_vertices` long (one
/// past the highest listed when not given).
#[pyfunction]
#[pyo3(signature = (ptr, idx, n_vertices=None))]
fn vertex_map<'py>(
    py: Python<'py>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    n_vertices: Option<usize>,
) -> PyResult<IdPair<'py>> {
    let (ptr, idx) = csr_flat(&ptr, &idx, n_vertices, "the bins")?;
    let n = n_vertices.unwrap_or_else(|| idx.iter().max().map_or(0, |&v| v + 1));
    let bins: Vec<&[usize]> = ptr.windows(2).map(|w| &idx[w[0]..w[1]]).collect();
    let (keys, node) = py.detach(|| ::dataclass::state::vertex_map(&bins, n));
    let keys = Array1::from_iter(keys.into_iter().map(|v| v as i64));
    Ok((keys.into_pyarray(py), Array1::from(node).into_pyarray(py)))
}

/// Add this module's functions to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soma_contains, m)?)?;
    m.add_function(wrap_pyfunction!(soma_distance_to_center, m)?)?;
    m.add_function(wrap_pyfunction!(soma_distance_to_surface, m)?)?;
    m.add_function(wrap_pyfunction!(node_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(remap_edges, m)?)?;
    m.add_function(wrap_pyfunction!(vertex_map, m)?)?;
    Ok(())
}
