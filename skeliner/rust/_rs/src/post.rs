//! `skeliner._rs.post`: the `post` crate's kernels for `skeliner.post`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
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

/// `downsample`'s result: nodes, radii `(K, M)`, edges in the order they
/// were made, `old2new`, and the bins as a CSR (`None`, `None` without).
type DownsampledOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Option<Bound<'py, PyArray1<i64>>>,
    Option<Bound<'py, PyArray1<i64>>>,
);

/// `post.downsample` on arrays: `nodes` `(N, 3)`, `radii` `(K, N)` (row
/// `key` decides), `edges` `(E, 2)`, the bins as a CSR (`ptr`, `idx`) or
/// `None`: `(nodes, radii, edges, old2new, ptr, idx)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn downsample<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray2<'py, f64>,
    key: usize,
    edges: PyReadonlyArray2<'py, i64>,
    ptr: Option<numpy::PyReadonlyArray1<'py, i64>>,
    idx: Option<numpy::PyReadonlyArray1<'py, i64>>,
    rtol: f64,
    atol: f64,
    aggregate: &str,
    merge_endpoints: bool,
    slide_branchpoints: bool,
    max_anchor_shift: Option<f64>,
) -> PyResult<DownsampledOut<'py>> {
    use ::post::downsample::{Aggregate, Settings};

    let nodes = rows3("N", &nodes)?;
    let n = nodes.len();
    let edges = node_pairs(&edges, n)?;
    let radii = radii.as_array();
    if radii.ncols() != n {
        return Err(PyValueError::new_err("one radius per node"));
    }
    let radii: Vec<Vec<f64>> = radii.rows().into_iter().map(|r| r.to_vec()).collect();
    if key >= radii.len() {
        return Err(PyValueError::new_err("no such radius"));
    }
    let aggregate = Aggregate::parse(aggregate)
        .ok_or_else(|| PyValueError::new_err("aggregate must be 'median', 'mean', or 'area'"))?;
    let bins = match (ptr, idx) {
        (Some(ptr), Some(idx)) => {
            let bins = csr_lists(&ptr, &idx, None, "the bins")?;
            if bins.len() != n {
                return Err(PyValueError::new_err(
                    "the bins must be a CSR (ptr, idx) over the nodes",
                ));
            }
            Some(bins)
        }
        _ => None,
    };
    let s = Settings {
        rtol,
        atol,
        aggregate,
        merge_endpoints,
        slide_branchpoints,
        max_anchor_shift,
    };
    let d = py
        .detach(|| ::post::downsample::downsample(&nodes, &radii, key, &edges, bins.as_deref(), s));
    let m = d.nodes.len();
    let flat: Vec<f64> = d.radii.iter().flatten().copied().collect();
    let radii = numpy::ndarray::Array2::from_shape_vec((d.radii.len(), m), flat)
        .expect("one radius per node");
    let pairs: Vec<i64> = d.edges.iter().flatten().map(|&v| v as i64).collect();
    let edges = numpy::ndarray::Array2::from_shape_vec((d.edges.len(), 2), pairs)
        .expect("two ends per edge");
    let (ptr, idx) = match d.node2verts {
        Some(bins) => {
            let mut ptr = vec![0i64];
            let mut idx = Vec::new();
            for b in &bins {
                idx.extend(b.iter().map(|&v| v as i64));
                ptr.push(idx.len() as i64);
            }
            (
                Some(Array1::from_vec(ptr).into_pyarray(py)),
                Some(Array1::from_vec(idx).into_pyarray(py)),
            )
        }
        None => (None, None),
    };
    Ok((
        rows_to_array(&d.nodes).into_pyarray(py),
        radii.into_pyarray(py),
        edges.into_pyarray(py),
        Array1::from_vec(d.old2new).into_pyarray(py),
        ptr,
        idx,
    ))
}

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(downsample, m)?)?;
    m.add_function(wrap_pyfunction!(remap_ntype, m)?)?;
    m.add_function(wrap_pyfunction!(fill_ntype_gaps, m)?)?;
    m.add_class::<PyVertexFaces>()?;
    m.add_function(wrap_pyfunction!(ray_hits, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_radii, m)?)?;
    Ok(())
}
