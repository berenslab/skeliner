//! `skeliner._rs.post`: the `post` crate's kernels for `skeliner.post`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::*;

/// Which rays hit any of the triangles, with trimesh's `ray_triangle` test
/// (`triangles` `(F, 3, 3)` flattened to `(F, 9)`, `normals` their face
/// normals).
#[pyfunction]
fn ray_hits<'py>(
    py: Python<'py>,
    triangles: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    origins: PyReadonlyArray2<'py, f64>,
    directions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let tris = triangles9(&triangles)?;
    let normals = rows3("N", &normals)?;
    if normals.len() != tris.len() {
        return Err(PyValueError::new_err("one normal per triangle"));
    }
    let (o, d) = (rows3("O", &origins)?, rows3("D", &directions)?);
    if o.len() != d.len() {
        return Err(PyValueError::new_err("one direction per origin"));
    }
    let hits = py.detach(|| ::post::ray_hits(tris, normals, &o, &d));
    Ok(Array1::from(hits).into_pyarray(py))
}

/// Faces by vertex, for cutting a mesh down to the faces of a vertex set.
#[pyclass(name = "VertexFaces", module = "skeliner._rs.post", frozen)]
struct PyVertexFaces {
    inner: ::post::VertexFaces,
}

#[pymethods]
impl PyVertexFaces {
    #[new]
    fn new(faces: PyReadonlyArray2<'_, i64>, n_vertices: usize) -> PyResult<Self> {
        let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
        Ok(Self {
            inner: ::post::VertexFaces::new(tris, n),
        })
    }

    /// Faces with all three vertices among `vids`, ascending — the faces
    /// `skeliner.post.submesh_by_vertices` keeps.
    fn faces_within<'py>(
        &self,
        py: Python<'py>,
        vids: numpy::PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ids = vertex_ids(&vids, self.inner.n_vertices())?;
        let out = self.inner.faces_within(&ids);
        Ok(Array1::from_iter(out.into_iter().map(i64::from)).into_pyarray(py))
    }
}

/// Each node's radius and how it was found (0 fallback, 1 full centreline,
/// 2 outer centreline).
type CalibratedOut<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>);

/// `skeliner.post.calibrate_radii` on arrays, every node in one call:
/// `(radii, kinds)`.  `centers` and `normals` are the mesh's face centres
/// and normals, `directions` the rays to cast, `rays_threshold` the hits
/// that make a face inner.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn calibrate_radii<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, i64>,
    node2verts: Vec<numpy::PyReadonlyArray1<'py, i64>>,
    skip: numpy::PyReadonlyArray1<'py, bool>,
    base: numpy::PyReadonlyArray1<'py, f64>,
    directions: PyReadonlyArray2<'py, f64>,
    min_n_verts_bulb: i64,
    check_outer: bool,
    min_n_outer: i64,
    min_frac_outer: f64,
    rays_threshold: i64,
    aggregate: String,
) -> PyResult<CalibratedOut<'py>> {
    let verts = rows3("V", &vertices)?;
    let (tris, n_vertices) = checked_faces(&faces, Some(verts.len()))?;
    let centers = rows3("F", &centers)?;
    let normals = rows3("F", &normals)?;
    if centers.len() != tris.len() || normals.len() != tris.len() {
        return Err(PyValueError::new_err(
            "centers and normals need one row per face",
        ));
    }
    let nodes = rows3("N", &nodes)?;
    let n = nodes.len();
    let skip = skip.as_array().to_vec();
    let base = base.as_array().to_vec();
    if base.len() != n || skip.len() != n || node2verts.len() != n {
        return Err(PyValueError::new_err(
            "one radius, skip flag and vertex list per node",
        ));
    }
    let e = edges.as_array();
    if e.ncols() != 2 {
        return Err(PyValueError::new_err("edges must be (E, 2)"));
    }
    let node = |v: i64| {
        u32::try_from(v)
            .ok()
            .filter(|&v| (v as usize) < n)
            .ok_or_else(|| PyValueError::new_err("edges name a node out of range"))
    };
    let edges: Vec<[u32; 2]> = e
        .rows()
        .into_iter()
        .map(|r| Ok([node(r[0])?, node(r[1])?]))
        .collect::<PyResult<_>>()?;
    let node2verts: Vec<Vec<u32>> = node2verts
        .iter()
        .map(|a| vertex_ids(a, n_vertices))
        .collect::<PyResult<_>>()?;
    let directions = rows3("R", &directions)?;
    let found = py.detach(|| {
        let vertex_faces = ::post::VertexFaces::new(tris.clone(), n_vertices);
        let m = ::post::CalibrateMesh {
            vertices: &verts,
            faces: &tris,
            centers: &centers,
            normals: &normals,
            vertex_faces: &vertex_faces,
        };
        let p = ::post::CalibrateParams {
            directions: &directions,
            min_n_verts_bulb,
            check_outer,
            min_n_outer,
            min_frac_outer,
            rays_threshold,
            aggregate: &aggregate,
        };
        ::post::calibrate_radii(&m, &nodes, &edges, &node2verts, &skip, &base, &p)
    });
    let (radii, kinds) = found
        .ok_or_else(|| PyValueError::new_err(format!("Unknown radius estimator '{aggregate}'.")))?;
    Ok((
        Array1::from(radii).into_pyarray(py),
        Array1::from_iter(kinds.into_iter().map(|k| k as u8)).into_pyarray(py),
    ))
}

/// Add this module's functions and classes to `m`.
/// Node types after the nodes are renumbered by `old2new` (−1 for a node
/// dropped) into `new_len` nodes: each new node the most common non-zero
/// label of the nodes merged into it (ties by soma, root mark, neurite
/// types, then the smallest), the root marked `root_label`, and with
/// `fill_gaps` zero labels filled from neighbours over `edges`.
#[pyfunction]
fn remap_ntype<'py>(
    py: Python<'py>,
    ntype: numpy::PyReadonlyArray1<'py, i64>,
    old2new: numpy::PyReadonlyArray1<'py, i64>,
    new_len: usize,
    root_label: i64,
    edges: PyReadonlyArray2<'py, i64>,
    fill_gaps: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let edges = node_pairs(&edges, new_len)?;
    let (ntype, old2new) = (ntype.as_array().to_vec(), old2new.as_array().to_vec());
    let mapped =
        ::post::ntype::remap_ntype(&ntype, &old2new, new_len, root_label, &edges, fill_gaps)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Array1::from_vec(mapped).into_pyarray(py))
}

/// Zero node types filled from their neighbours over `edges`: sweeps over
/// the zeros in order, each taking the most common neighbouring type other
/// than 0, 1 and -1, until a sweep fills nothing.
#[pyfunction]
fn fill_ntype_gaps<'py>(
    py: Python<'py>,
    ntype: numpy::PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ntype = ntype.as_array().to_vec();
    let edges = node_pairs(&edges, ntype.len())?;
    let filled = ::post::ntype::fill_ntype_gaps(&ntype, &edges);
    Ok(Array1::from_vec(filled).into_pyarray(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(remap_ntype, m)?)?;
    m.add_function(wrap_pyfunction!(fill_ntype_gaps, m)?)?;
    m.add_class::<PyVertexFaces>()?;
    m.add_function(wrap_pyfunction!(ray_hits, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_radii, m)?)?;
    Ok(())
}
