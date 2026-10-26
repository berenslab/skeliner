//! `skeliner._rs.mesh`: the `mesh` crate's mesh structures, and the
//! array conversion checks.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PyTuple};
use std::collections::BTreeSet;

use crate::common::*;

/// Pass a mesh through Rust and back, unchanged.
///
/// Exercises the conversions every binding relies on: dtypes, shapes,
/// non-contiguous input, and refusing faces that name missing vertices.
#[pyfunction]
fn mesh_roundtrip<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
) -> PyResult<MeshArrays<'py>> {
    let verts = rows3("V", &vertices)?;
    let tris = rows3("F", &faces)?;
    let mesh = ::mesh::Mesh::new(verts, &tris).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let faces: Vec<[i64; 3]> = mesh.faces().iter().map(|f| f.map(i64::from)).collect();
    Ok((
        rows_to_array(mesh.vertices()).into_pyarray(py),
        rows_to_array(&faces).into_pyarray(py),
    ))
}

/// Edge → face incidence of a mesh, readable like the dict
/// `skeliner.pre._edge_to_faces` builds.
///
/// Keys are `(low, high)` vertex pairs, values the faces sharing that edge,
/// and iteration follows the dict's insertion order.  Faces with a repeated
/// vertex have no edges.
#[pyclass(name = "Topology", module = "skeliner._rs.mesh", frozen)]
pub(crate) struct PyTopology {
    pub(crate) inner: ::mesh::Topology,
}

impl PyTopology {
    fn faces_of(&self, (a, b): (i64, i64)) -> Option<&[u32]> {
        let (a, b) = (u32::try_from(a).ok()?, u32::try_from(b).ok()?);
        self.inner.edge_id(a, b).map(|e| self.inner.edge_faces(e))
    }

    fn key<'py>(&self, py: Python<'py>, e: usize) -> PyResult<Bound<'py, PyTuple>> {
        let [a, b] = self.inner.edges()[e];
        PyTuple::new(py, [a, b])
    }
}

#[pymethods]
impl PyTopology {
    #[new]
    #[pyo3(signature = (faces, n_vertices=None))]
    fn new(
        py: Python<'_>,
        faces: PyReadonlyArray2<'_, i64>,
        n_vertices: Option<usize>,
    ) -> PyResult<Self> {
        let (tris, n) = checked_faces(&faces, n_vertices)?;
        let inner = py.detach(|| ::mesh::Topology::new(&tris, n));
        Ok(Self { inner })
    }

    #[getter]
    fn n_faces(&self) -> usize {
        self.inner.n_faces()
    }

    #[getter]
    fn n_vertices(&self) -> usize {
        self.inner.n_vertices()
    }

    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.n_edges()
    }

    fn __len__(&self) -> usize {
        self.inner.n_edges()
    }

    fn __contains__(&self, key: (i64, i64)) -> bool {
        self.faces_of(key).is_some()
    }

    fn __getitem__(&self, key: (i64, i64)) -> PyResult<Vec<u32>> {
        self.faces_of(key)
            .map(<[u32]>::to_vec)
            .ok_or_else(|| PyKeyError::new_err(key))
    }

    /// Faces of edge `key`, or `default` if no face has that edge.
    #[pyo3(signature = (key, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: (i64, i64),
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self.faces_of(key) {
            Some(fs) => Ok(PyList::new(py, fs)?.into_any()),
            None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
        }
    }

    /// Edge keys, in the dict's insertion order.
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: PyResult<Vec<_>> = self
            .inner
            .first_seen_order()
            .into_iter()
            .map(|e| self.key(py, e))
            .collect();
        PyList::new(py, keys?)
    }

    /// Face lists, in the dict's insertion order.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let values: PyResult<Vec<_>> = self
            .inner
            .first_seen_order()
            .into_iter()
            .map(|e| PyList::new(py, self.inner.edge_faces(e)))
            .collect();
        PyList::new(py, values?)
    }

    /// `(key, faces)` pairs, in the dict's insertion order.
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: PyResult<Vec<_>> = self
            .inner
            .first_seen_order()
            .into_iter()
            .map(|e| {
                let faces = PyList::new(py, self.inner.edge_faces(e))?;
                PyTuple::new(py, [self.key(py, e)?.into_any(), faces.into_any()])
            })
            .collect();
        PyList::new(py, items?)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        PyIterator::from_object(self.keys(py)?.as_any())
    }

    /// The same mapping as a plain dict.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for e in self.inner.first_seen_order() {
            d.set_item(self.key(py, e)?, PyList::new(py, self.inner.edge_faces(e))?)?;
        }
        Ok(d)
    }

    /// Distinct edges as an `(E, 2)` array, in ascending order.
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let flat: Vec<i64> = self
            .inner
            .edges()
            .iter()
            .flatten()
            .map(|&v| i64::from(v))
            .collect();
        Array2::from_shape_vec((self.inner.n_edges(), 2), flat)
            .expect("two columns per edge")
            .into_pyarray(py)
    }

    /// Edges between a face of `sel` and a face outside it, ascending.
    fn border_edges(&self, sel: &Bound<'_, PyAny>) -> PyResult<Vec<(u32, u32)>> {
        let sel = selection(sel)?;
        Ok(::mesh::loops::border_edges(&self.inner, &sel)
            .into_iter()
            .map(|[a, b]| (a, b))
            .collect())
    }

    /// Closed loops along the border of the faces in `sel`, as
    /// `skeliner.pre._trace_border_loops` returns them.
    fn border_loops(&self, py: Python<'_>, sel: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
        let sel = selection(sel)?;
        Ok(py.detach(|| ::mesh::loops::border_loops(&self.inner, &sel)))
    }

    /// `skeliner.pre._expand_tip_to_good_rim` on this mesh: `(selection,
    /// longest loop or None)`.
    #[pyo3(signature = (tip, target_verts=6, max_iters=20, max_eat_factor=30))]
    fn expand_tip(
        &self,
        py: Python<'_>,
        tip: &Bound<'_, PyAny>,
        target_verts: usize,
        max_iters: usize,
        max_eat_factor: usize,
    ) -> PyResult<(BTreeSet<u32>, Option<Vec<u32>>)> {
        let tip = selection(tip)?;
        Ok(py.detach(|| {
            ::pre::gaps::expand_tip_to_good_rim(
                &self.inner,
                &tip,
                target_verts,
                max_iters,
                max_eat_factor,
            )
        }))
    }

    /// `skeliner.pre._expand_selection_per_group` on this mesh:
    /// `(selection, loops)`.
    #[pyo3(signature = (sel, target_verts=6, max_iters=20, max_eat_factor=30))]
    fn expand_selection_per_group(
        &self,
        py: Python<'_>,
        sel: &Bound<'_, PyAny>,
        target_verts: usize,
        max_iters: usize,
        max_eat_factor: usize,
    ) -> PyResult<(BTreeSet<u32>, Vec<Vec<u32>>)> {
        let sel = selection(sel)?;
        Ok(py.detach(|| {
            ::pre::gaps::expand_selection_per_group(
                &self.inner,
                &sel,
                target_verts,
                max_iters,
                max_eat_factor,
            )
        }))
    }

    /// `skeliner.pre._sever_cost` on this mesh.
    fn sever_cost(&self, py: Python<'_>, sel: &Bound<'_, PyAny>, budget: usize) -> PyResult<usize> {
        let sel: BTreeSet<u32> = selection(sel)?.into_iter().collect();
        Ok(py.detach(|| ::pre::gaps::sever_cost(&self.inner, &sel, budget)))
    }

    /// Edge → faces as CSR arrays `(offsets, faces)`, edges in ascending
    /// order.
    fn edge_faces_csr<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
        let (ptr, faces) = self.inner.edge_faces_csr();
        let ptr: Array1<i64> = ptr.iter().map(|&p| p as i64).collect();
        let faces: Array1<i64> = faces.iter().map(|&f| i64::from(f)).collect();
        (ptr.into_pyarray(py), faces.into_pyarray(py))
    }
}

/// Distances and indices of nearest points.
type Nearest<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>);

/// A KD-tree over `(N, 3)` points, queried like scipy's `KDTree`.
///
/// `fused` picks how squared distances are summed (see
/// `::mesh::spatial::Sum`).
#[pyclass(name = "KdTree", module = "skeliner._rs.mesh", frozen)]
struct PyKdTree {
    inner: ::mesh::spatial::KdTree,
}

#[pymethods]
impl PyKdTree {
    /// `fused`: whether squared distances are summed with fused
    /// multiply-adds — by default as scipy's tree sums them here.
    #[new]
    #[pyo3(signature = (points, fused=None))]
    fn new(
        py: Python<'_>,
        points: PyReadonlyArray2<'_, f64>,
        fused: Option<bool>,
    ) -> PyResult<Self> {
        let pts = points3(&points)?;
        let sum = if fused.unwrap_or(::mesh::geometry::HAS_FMA) {
            ::mesh::spatial::Sum::Fused
        } else {
            ::mesh::spatial::Sum::Plain
        };
        Ok(Self {
            inner: py.detach(|| ::mesh::spatial::KdTree::new(pts, sum)),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Indices within `r` of each query point, ascending, one array per
    /// point; `r` one radius, or one per point.
    fn query_ball_point<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        r: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<i64>>>> {
        let qs = finite("query points", points3(&x)?)?;
        let radii = match r.extract::<f64>() {
            Ok(r) => vec![r; qs.len()],
            Err(_) => r
                .extract::<PyReadonlyArray1<'py, f64>>()
                .ok()
                .map(|r| r.as_array().to_vec())
                .filter(|r| r.len() == qs.len())
                .ok_or_else(|| PyValueError::new_err("r must be a radius or one per point"))?,
        };
        let hits: Vec<Vec<u32>> = py.detach(|| {
            qs.iter()
                .zip(&radii)
                .map(|(q, &r)| self.inner.within(q, r))
                .collect()
        });
        Ok(hits
            .into_iter()
            .map(|h| {
                let a: Array1<i64> = h.into_iter().map(i64::from).collect();
                a.into_pyarray(py)
            })
            .collect())
    }

    /// `(distances, indices)` of each query point's nearest point.
    fn query<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<Nearest<'py>> {
        let qs = finite("query points", points3(&x)?)?;
        if self.inner.is_empty() {
            return Err(PyValueError::new_err("the tree is empty"));
        }
        let found: Vec<(u32, f64)> = py.detach(|| {
            qs.iter()
                .map(|q| self.inner.nearest(q).expect("non-empty tree"))
                .collect()
        });
        let d: Array1<f64> = found.iter().map(|&(_, d)| d).collect();
        let i: Array1<i64> = found.iter().map(|&(i, _)| i64::from(i)).collect();
        Ok((d.into_pyarray(py), i.into_pyarray(py)))
    }
}

/// A mesh's vertices and faces as the crates take them.
type MeshIn = (Vec<[f64; 3]>, Vec<[u32; 3]>);

/// The vertices, and the faces checked against them.
fn mesh_in(
    vertices: &PyReadonlyArray2<'_, f64>,
    faces: &PyReadonlyArray2<'_, i64>,
) -> PyResult<MeshIn> {
    let verts = rows3("V", vertices)?;
    let (tris, _) = checked_faces(faces, Some(verts.len()))?;
    Ok((verts, tris))
}

fn pairs_out<'py>(py: Python<'py>, pairs: &[[u32; 2]]) -> Bound<'py, PyArray2<i64>> {
    let flat: Vec<i64> = threads::map(pairs.as_flattened(), |&v| i64::from(v));
    Array2::from_shape_vec((pairs.len(), 2), flat)
        .expect("two per row")
        .into_pyarray(py)
}

/// trimesh's `triangles_cross`: `(t1 − t0) × (t2 − t1)` per face.
#[pyfunction]
fn face_crosses<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (verts, tris) = mesh_in(&vertices, &faces)?;
    let out = py.detach(|| ::mesh::derived::face_crosses(&verts, &tris));
    Ok(rows_to_array(&out).into_pyarray(py))
}

/// trimesh's `face_normals` from `crosses` `(F, 3)`: each cross product
/// over its norm, zero where the norm is not above trimesh's zero.
#[pyfunction]
fn face_normals<'py>(
    py: Python<'py>,
    crosses: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let c = rows3("F", &crosses)?;
    let out = py.detach(|| ::mesh::derived::face_normals(&c));
    Ok(rows_to_array(&out).into_pyarray(py))
}

/// trimesh's `area_faces` from `crosses` `(F, 3)`: half each one's norm.
#[pyfunction]
fn face_areas<'py>(
    py: Python<'py>,
    crosses: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = rows3("F", &crosses)?;
    Ok(Array1::from(::mesh::derived::face_areas(&c)).into_pyarray(py))
}

/// trimesh's `triangles_center`: each face's vertices, summed, over three.
#[pyfunction]
fn face_centers<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (verts, tris) = mesh_in(&vertices, &faces)?;
    let out = py.detach(|| ::mesh::derived::face_centers(&verts, &tris));
    Ok(rows_to_array(&out).into_pyarray(py))
}

/// trimesh's `face_adjacency` of `faces` `(F, 3)` over `n_vertices`: the
/// two faces of each edge exactly two face edges share, by the edge's
/// higher end then its lower, lower face first.
#[pyfunction]
fn adjacent_faces<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    n_vertices: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
    let out = py.detach(|| ::mesh::derived::adjacent_faces(&tris, n));
    Ok(pairs_out(py, &out))
}

/// trimesh's `edges_unique` of `faces` `(F, 3)` over `n_vertices`: every
/// distinct edge as `[low, high]`, by `high` then `low`.
#[pyfunction]
fn unique_edges<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    n_vertices: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
    let out = py.detach(|| ::mesh::topology::unique_edges(&tris, n));
    Ok(pairs_out(py, &out))
}

/// trimesh's `edges_unique_length`: each edge's length,
/// `sqrt((dx² + dy²) + dz²)`.
#[pyfunction]
fn edge_lengths<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let verts = rows3("V", &vertices)?;
    let pairs = node_pairs(&edges, verts.len())?;
    let pairs: Vec<[u32; 2]> = threads::map(&pairs, |&[a, b]| [a as u32, b as u32]);
    let out = py.detach(|| ::mesh::geometry::edge_lengths(&verts, &pairs));
    Ok(Array1::from(out).into_pyarray(py))
}

/// Rows of a 2D `points` array `(N, 2)`.
fn points2(points: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 2]>> {
    let a = points.as_array();
    if a.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "expected (N, 2) points, got shape {:?}",
            a.shape()
        )));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

/// `(area, perimeter)` of the convex hull of 2D `points` `(N, 2)` — scipy's
/// `ConvexHull` `volume` and `area` — both 0 where qhull refuses the points
/// as flat.
#[pyfunction]
fn hull_measures(points: PyReadonlyArray2<'_, f64>) -> PyResult<(f64, f64)> {
    Ok(::mesh::geometry::hull_measures(&points2(&points)?))
}

/// The edges of the convex hull of 2D `points` `(N, 2)` as scipy's
/// `ConvexHull.equations` gives them — per edge `(n0, n1, offset)`, the unit
/// normal pointing out, `n . p + offset <= 0` inside — or None where qhull
/// refuses the points as flat.
#[pyfunction]
fn hull_equations<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    let found = ::mesh::geometry::hull_equations(&points2(&points)?);
    Ok(found.map(|eq| rows_to_array(&eq).into_pyarray(py)))
}

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mesh_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(face_crosses, m)?)?;
    m.add_function(wrap_pyfunction!(face_normals, m)?)?;
    m.add_function(wrap_pyfunction!(face_areas, m)?)?;
    m.add_function(wrap_pyfunction!(face_centers, m)?)?;
    m.add_function(wrap_pyfunction!(adjacent_faces, m)?)?;
    m.add_function(wrap_pyfunction!(unique_edges, m)?)?;
    m.add_function(wrap_pyfunction!(edge_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(hull_measures, m)?)?;
    m.add_function(wrap_pyfunction!(hull_equations, m)?)?;
    m.add_class::<PyTopology>()?;
    m.add_class::<PyKdTree>()?;
    Ok(())
}
