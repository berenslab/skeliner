//! `skeliner._rs.pair`: the `pair` crate's kernels for `skeliner.pair`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::*;

/// A BVH over triangles (`(F, 3, 3)` flattened to `(F, 9)`, `normals`
/// their face normals), built once to cast many rays at, with trimesh's
/// `ray_triangle` test.
#[pyclass(name = "TriangleBvh", module = "skeliner._rs.pair", frozen)]
struct PyTriangleBvh {
    inner: ::mesh::ray::TriangleBvh,
}

#[pymethods]
impl PyTriangleBvh {
    #[new]
    fn new(
        py: Python<'_>,
        triangles: PyReadonlyArray2<'_, f64>,
        normals: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let tris = triangles9(&triangles)?;
        let normals = rows3("N", &normals)?;
        if normals.len() != tris.len() {
            return Err(PyValueError::new_err("one normal per triangle"));
        }
        Ok(Self {
            inner: py.detach(|| ::mesh::ray::TriangleBvh::new(tris, normals)),
        })
    }

    /// How far each ray travels before it first hits a triangle: per ray
    /// from `origins` along `directions`, the distance (ray parameter) to
    /// the first triangle past `min_t`, leaving out triangle `skip` (the one
    /// the ray starts on; -1 for none); NaN when it hits nothing.
    fn first_hits<'py>(
        &self,
        py: Python<'py>,
        origins: PyReadonlyArray2<'py, f64>,
        directions: PyReadonlyArray2<'py, f64>,
        skip: numpy::PyReadonlyArray1<'py, i64>,
        min_t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (o, d) = (rows3("O", &origins)?, rows3("D", &directions)?);
        let skip: Vec<i64> = skip.as_array().to_vec();
        if o.len() != d.len() || o.len() != skip.len() {
            return Err(PyValueError::new_err(
                "one direction and one skip per origin",
            ));
        }
        let dist = py.detach(|| ::pair::first_hits(&self.inner, &o, &d, &skip, min_t));
        Ok(Array1::from(dist).into_pyarray(py))
    }
}

/// Which faces of one mesh touch another: per face, the other mesh's
/// triangle closest to its `center` — of several equally close (meeting at
/// the closest point), the one whose normal most opposes the face's, then
/// the lowest index — kept when it lies within `tol` and `dot <= max_dot`.
/// Returns `(faces, other_faces, distances)` for the faces kept.
/// `triangles` are the other mesh's `(G, 3, 3)` flattened to `(G, 9)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn contact_faces<'py>(
    py: Python<'py>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, f64>,
    other_normals: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_dot: f64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let centers = rows3("C", &centers)?;
    let normals = rows3("N", &normals)?;
    if normals.len() != centers.len() {
        return Err(PyValueError::new_err("one normal per face"));
    }
    let tris = triangles9(&triangles)?;
    let other_normals = rows3("M", &other_normals)?;
    if other_normals.len() != tris.len() {
        return Err(PyValueError::new_err("one normal per triangle"));
    }
    let found =
        py.detach(|| ::pair::contact_faces(&centers, &normals, tris, other_normals, tol, max_dot));
    let (mut faces, mut others, mut dist) = (Vec::new(), Vec::new(), Vec::new());
    for (f, hit) in found.into_iter().enumerate() {
        if let Some((t, d)) = hit {
            faces.push(f as i64);
            others.push(t as i64);
            dist.push(d);
        }
    }
    Ok((
        Array1::from(faces).into_pyarray(py),
        Array1::from(others).into_pyarray(py),
        Array1::from(dist).into_pyarray(py),
    ))
}

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTriangleBvh>()?;
    m.add_function(wrap_pyfunction!(contact_faces, m)?)?;
    Ok(())
}
