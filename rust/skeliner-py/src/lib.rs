//! The `skeliner._rs` extension module.

use std::collections::BTreeSet;

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PyTuple};

/// Rows of an `(N, 3)` array, refusing any other shape.
fn rows3<T: Copy + numpy::Element>(
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
fn checked_faces(
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
        meshcore::checked_faces(&rows, n).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((checked, n))
}

/// Faces narrowed to `u32`, without a vertex count to check against.
fn faces_u32(faces: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<[u32; 3]>> {
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
fn face_ids(ids: &numpy::PyReadonlyArray1<'_, i64>, n_faces: usize) -> PyResult<Vec<u32>> {
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
fn selection(sel: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
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
type MeshArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

fn rows_to_array<T: Copy>(rows: &[[T; 3]]) -> Array2<T> {
    let flat: Vec<T> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), 3), flat).expect("three columns per row")
}

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
    let mesh =
        meshcore::Mesh::new(verts, &tris).map_err(|e| PyValueError::new_err(e.to_string()))?;
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
#[pyclass(name = "Topology", module = "skeliner._rs", frozen)]
struct PyTopology {
    inner: meshcore::Topology,
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
        let inner = py.detach(|| meshcore::Topology::new(&tris, n));
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
        Ok(meshcore::loops::border_edges(&self.inner, &sel)
            .into_iter()
            .map(|[a, b]| (a, b))
            .collect())
    }

    /// Closed loops along the border of the faces in `sel`, as
    /// `skeliner.pre._trace_border_loops` returns them.
    fn border_loops(&self, py: Python<'_>, sel: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
        let sel = selection(sel)?;
        Ok(py.detach(|| meshcore::loops::border_loops(&self.inner, &sel)))
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
            skeliner_pre::gaps::expand_tip_to_good_rim(
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
            skeliner_pre::gaps::expand_selection_per_group(
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
        Ok(py.detach(|| skeliner_pre::gaps::sever_cost(&self.inner, &sel, budget)))
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

/// Edge-connected components of the non-degenerate faces.
///
/// Returns `(labels, main)` exactly as `skeliner.pre._face_edge_components`
/// does: components numbered by their lowest face, `-2` for faces with a
/// repeated vertex, `main` the largest with ties to the lowest id.
#[pyfunction]
#[pyo3(signature = (faces, n_vertices=None))]
fn face_edge_components<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    n_vertices: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, usize)> {
    let (tris, n) = checked_faces(&faces, n_vertices)?;
    let (labels, main) = py.detach(|| {
        let topo = meshcore::Topology::new(&tris, n);
        meshcore::components::face_edge_components(&topo)
    });
    Ok((labels.into_pyarray(py), main))
}

/// Components of the listed faces, as `skeliner.pre._face_components_fast`
/// returns them: largest first, ties to the lower lowest face, each an int
/// array of face ids in the order listed.
#[pyfunction]
fn face_components<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    face_indices: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Vec<Bound<'py, PyArray1<i64>>>> {
    let tris = faces_u32(&faces)?;
    let fi = face_ids(&face_indices, tris.len())?;
    let comps = py.detach(|| meshcore::components::listed_face_components(&tris, &fi));
    Ok(comps
        .into_iter()
        .map(|c| {
            let c: Array1<i64> = c.into_iter().map(i64::from).collect();
            c.into_pyarray(py)
        })
        .collect())
}

/// `skeliner.pre._refine_components_with_fusions` on arrays.
///
/// `fusion` lists the fusion-cluster faces; ids past the last face are
/// ignored.  Returns `(labels, main)`.
#[pyfunction]
fn refine_components_with_fusions<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    labels: numpy::PyReadonlyArray1<'py, i64>,
    main: i64,
    fusion: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)> {
    let tris = faces_u32(&faces)?;
    let labels = labels.as_array().to_vec();
    if labels.len() != tris.len() {
        return Err(PyValueError::new_err(format!(
            "{} labels for {} faces",
            labels.len(),
            tris.len()
        )));
    }
    let fusion: Vec<u32> = fusion
        .as_array()
        .iter()
        .filter_map(|&f| u32::try_from(f).ok())
        .collect();
    let (new, main) = py.detach(|| {
        skeliner_pre::components::refine_components_with_fusions(&tris, &labels, main, &fusion)
    });
    Ok((new.into_pyarray(py), main))
}

/// `skeliner.pre._find_nonmanifold_fusions` with the outward dots given;
/// with `sort`, largest first as `skeliner.pre.find_fusions` returns them.
#[pyfunction]
#[pyo3(signature = (
    faces, n_vertices, areas, outward_dots, grow_rings=20, min_branch_size=5, sort=false,
))]
#[allow(clippy::too_many_arguments)]
fn find_nonmanifold_fusions<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    n_vertices: usize,
    areas: numpy::PyReadonlyArray1<'py, f64>,
    outward_dots: numpy::PyReadonlyArray1<'py, f64>,
    grow_rings: usize,
    min_branch_size: usize,
    sort: bool,
) -> PyResult<Vec<Vec<u32>>> {
    let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
    let areas = areas.as_array().to_vec();
    let dots = outward_dots.as_array().to_vec();
    if areas.len() != tris.len() || dots.len() != tris.len() {
        return Err(PyValueError::new_err(
            "areas and outward_dots need one value per face",
        ));
    }
    Ok(py.detach(|| {
        let topo = meshcore::Topology::new(&tris, n);
        let find = if sort {
            skeliner_pre::fusions::find_fusions
        } else {
            skeliner_pre::fusions::find_nonmanifold_fusions
        };
        find(&topo, &tris, &areas, &dots, grow_rings, min_branch_size)
    }))
}

/// Relabelled faces, and the vertex each new vertex copies.
type FacesAndSources<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>);

/// `skeliner.pre._split_fan_vertices` on arrays: `(faces, sources)`, where
/// `sources[i]` is the vertex that new vertex `n_vertices + i` copies.
/// `sources` is empty when no fan is split.
#[pyfunction]
#[pyo3(signature = (faces, n_vertices, candidates=None))]
fn split_fan_vertices<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    n_vertices: usize,
    candidates: Option<numpy::PyReadonlyArray1<'py, i64>>,
) -> PyResult<FacesAndSources<'py>> {
    let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
    let candidates: Option<Vec<u32>> = candidates
        .map(|c| {
            c.as_array()
                .iter()
                .map(|&v| {
                    u32::try_from(v)
                        .ok()
                        .filter(|&v| (v as usize) < n)
                        .ok_or_else(|| PyValueError::new_err(format!("vertex {v} is out of range")))
                })
                .collect::<PyResult<Vec<u32>>>()
        })
        .transpose()?;
    let split = py.detach(|| {
        let topo = meshcore::Topology::new(&tris, n);
        skeliner_pre::fusions::split_fan_vertices(&topo, &tris, candidates.as_deref())
    });
    let (out, sources) = split.unwrap_or((tris, Vec::new()));
    let out: Vec<[i64; 3]> = out.iter().map(|f| f.map(i64::from)).collect();
    let sources: Array1<i64> = sources.into_iter().map(i64::from).collect();
    Ok((
        rows_to_array(&out).into_pyarray(py),
        sources.into_pyarray(py),
    ))
}

/// Distances and indices of nearest points.
type Nearest<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>);

/// A KD-tree over `(N, 3)` points, queried like scipy's `KDTree`.
///
/// `fused` picks how squared distances are summed (see
/// `meshcore::spatial::Sum`).
#[pyclass(name = "KdTree", module = "skeliner._rs", frozen)]
struct PyKdTree {
    inner: meshcore::spatial::KdTree,
}

fn points3(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    rows3("N", a)
}

#[pymethods]
impl PyKdTree {
    #[new]
    #[pyo3(signature = (points, fused=false))]
    fn new(py: Python<'_>, points: PyReadonlyArray2<'_, f64>, fused: bool) -> PyResult<Self> {
        let pts = points3(&points)?;
        let sum = if fused {
            meshcore::spatial::Sum::Fused
        } else {
            meshcore::spatial::Sum::Plain
        };
        Ok(Self {
            inner: py.detach(|| meshcore::spatial::KdTree::new(pts, sum)),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Indices within `r` of each query point, ascending, one array per point.
    fn query_ball_point<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        r: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray1<i64>>>> {
        let qs = points3(&x)?;
        let hits: Vec<Vec<u32>> =
            py.detach(|| qs.iter().map(|q| self.inner.within(q, r)).collect());
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
        let qs = points3(&x)?;
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

/// `skeliner.pre._outward_dot` on arrays.
#[pyfunction]
#[pyo3(signature = (vertices, faces, centers, normals, radius, vert_comp=None))]
fn outward_dots<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    radius: f64,
    vert_comp: Option<numpy::PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let centers = rows3("F", &centers)?;
    let normals = rows3("F", &normals)?;
    if centers.len() != tris.len() || normals.len() != tris.len() {
        return Err(PyValueError::new_err(
            "centers and normals need one row per face",
        ));
    }
    let labels: Option<Vec<i64>> = vert_comp.map(|c| c.as_array().to_vec());
    if labels.as_ref().is_some_and(|l| l.len() != verts.len()) {
        return Err(PyValueError::new_err(
            "vert_comp needs one label per vertex",
        ));
    }
    let dots = py.detach(|| {
        skeliner_pre::stats::outward_dots(
            &verts,
            &tris,
            &centers,
            &normals,
            labels.as_deref(),
            radius,
        )
    });
    Ok(Array1::from(dots).into_pyarray(py))
}

/// Outward dots, face components and the main component.
type MeshStatsArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, usize);

/// `skeliner.pre.compute_mesh_stats` on arrays, with the radius given:
/// `(outward_dots, face_comp, main_ci)`.
#[pyfunction]
fn mesh_stats<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    radius: f64,
) -> PyResult<MeshStatsArrays<'py>> {
    let verts = rows3("V", &vertices)?;
    let (tris, n) = checked_faces(&faces, Some(verts.len()))?;
    let centers = rows3("F", &centers)?;
    let normals = rows3("F", &normals)?;
    if centers.len() != tris.len() || normals.len() != tris.len() {
        return Err(PyValueError::new_err(
            "centers and normals need one row per face",
        ));
    }
    let (dots, labels, main) = py.detach(|| {
        let topo = meshcore::Topology::new(&tris, n);
        let (labels, main) = meshcore::components::face_edge_components(&topo);
        let vert_comp = skeliner_pre::stats::vertex_components(&tris, &labels, n);
        let dots = skeliner_pre::stats::outward_dots(
            &verts,
            &tris,
            &centers,
            &normals,
            Some(&vert_comp),
            radius,
        );
        (dots, labels, main)
    });
    Ok((
        Array1::from(dots).into_pyarray(py),
        Array1::from(labels).into_pyarray(py),
        main,
    ))
}

/// Kept components and the counts of those set aside.
type DisconnectedOut = (Vec<Vec<u32>>, usize, usize, usize, usize);

/// `skeliner.pre.find_disconnected`'s component filter on arrays:
/// `(components, n_non_main, n_soma, n_organelle, n_enclosed)`.
///
/// `soma_w` is `R / axes`, the map that takes the soma to the unit sphere.
#[pyfunction]
#[pyo3(signature = (
    faces, vertices, centers, normals, median_edge, labels, main,
    organelle=None, soma_center=None, soma_w=None,
))]
#[allow(clippy::too_many_arguments)]
fn find_disconnected<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    median_edge: f64,
    labels: numpy::PyReadonlyArray1<'py, i64>,
    main: i64,
    organelle: Option<numpy::PyReadonlyArray1<'py, bool>>,
    soma_center: Option<numpy::PyReadonlyArray1<'py, f64>>,
    soma_w: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<DisconnectedOut> {
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let centers = rows3("F", &centers)?;
    let normals = rows3("F", &normals)?;
    let labels = labels.as_array().to_vec();
    if centers.len() != tris.len() || normals.len() != tris.len() || labels.len() != tris.len() {
        return Err(PyValueError::new_err(
            "centers, normals and labels need one row per face",
        ));
    }
    let organelle: Option<Vec<bool>> = organelle.map(|m| m.as_array().to_vec());
    if organelle.as_ref().is_some_and(|m| m.len() != tris.len()) {
        return Err(PyValueError::new_err("organelle needs one value per face"));
    }
    let soma = match (soma_center, soma_w) {
        (Some(c), Some(w)) => {
            let c = c.as_array();
            let w = rows3("3", &w)?;
            if c.len() != 3 || w.len() != 3 {
                return Err(PyValueError::new_err("soma_center is (3,), soma_w (3, 3)"));
            }
            Some(skeliner_pre::disconnected::Ellipsoid {
                center: [c[0], c[1], c[2]],
                w: [w[0], w[1], w[2]],
            })
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "give both soma_center and soma_w, or neither",
            ));
        }
    };
    let found = py.detach(|| {
        let surface = skeliner_pre::disconnected::Surface {
            faces: &tris,
            verts: &verts,
            centers: &centers,
            normals: &normals,
            median_edge,
        };
        skeliner_pre::disconnected::find_disconnected(
            &surface,
            &labels,
            main,
            organelle.as_deref(),
            soma.as_ref(),
        )
    });
    Ok((
        found.components,
        found.n_non_main,
        found.n_soma,
        found.n_organelle,
        found.n_enclosed,
    ))
}

/// Pocket and isolated masks, and the report.
/// numpy's SVD and scipy's hull, called back from the Rust core
/// (`skeliner_pre::fits::Svd`).
struct PySvd {
    rows: Py<PyAny>,
    hull: Option<Py<PyAny>>,
}

impl PySvd {
    fn call_rows(&self, x: &[[f64; 3]], full_matrices: bool) -> PyResult<[[f64; 3]; 3]> {
        Python::attach(|py| {
            let out = self
                .rows
                .call1(py, (rows_to_array(x).into_pyarray(py), full_matrices))?;
            let m: PyReadonlyArray2<'_, f64> = out.extract(py)?;
            let rows = rows3("Vt", &m)?;
            if rows.len() != 3 {
                return Err(PyValueError::new_err("svd callback must return (3, 3)"));
            }
            Ok([rows[0], rows[1], rows[2]])
        })
    }

    fn call_hull(&self, pts: &[[f64; 3]]) -> PyResult<f64> {
        let hull = self
            .hull
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no hull callback given"))?;
        Python::attach(|py| {
            hull.call1(py, (rows_to_array(pts).into_pyarray(py),))?
                .extract(py)
        })
    }
}

impl skeliner_pre::fits::Svd for PySvd {
    fn right_singular(&self, x: &[[f64; 3]], full_matrices: bool) -> [[f64; 3]; 3] {
        self.call_rows(x, full_matrices)
            .unwrap_or_else(|e| panic!("svd callback failed: {e}"))
    }

    fn hull_area(&self, pts: &[[f64; 3]]) -> f64 {
        self.call_hull(pts)
            .unwrap_or_else(|e| panic!("hull callback failed: {e}"))
    }
}

type OrganelleOut<'py> = (
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyDict>,
);

/// `skeliner.pre.find_organelles` given the mesh stats:
/// `(pocket, isolated, report)`, the report holding the counts its verbose
/// output prints.
#[pyfunction]
#[pyo3(signature = (
    vertices, faces, centers, normals, areas, outward_dots, face_comp, main, adjacency,
    svd_rows, hull_area, min_cluster_size=5,
))]
#[allow(clippy::too_many_arguments)]
fn find_organelles<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    areas: numpy::PyReadonlyArray1<'py, f64>,
    outward_dots: numpy::PyReadonlyArray1<'py, f64>,
    face_comp: numpy::PyReadonlyArray1<'py, i64>,
    main: i64,
    adjacency: PyReadonlyArray2<'py, i64>,
    svd_rows: Bound<'py, PyAny>,
    hull_area: Bound<'py, PyAny>,
    min_cluster_size: usize,
) -> PyResult<OrganelleOut<'py>> {
    let svd = PySvd {
        rows: svd_rows.unbind(),
        hull: Some(hull_area.unbind()),
    };
    let verts = rows3("V", &vertices)?;
    let (tris, n) = checked_faces(&faces, Some(verts.len()))?;
    let centers = rows3("F", &centers)?;
    let normals = rows3("F", &normals)?;
    let areas = areas.as_array().to_vec();
    let dots = outward_dots.as_array().to_vec();
    let face_comp = face_comp.as_array().to_vec();
    let nf = tris.len();
    if [
        centers.len(),
        normals.len(),
        areas.len(),
        dots.len(),
        face_comp.len(),
    ]
    .iter()
    .any(|&k| k != nf)
    {
        return Err(PyValueError::new_err(
            "per-face arrays need one row per face",
        ));
    }
    let pairs = adjacency.as_array();
    let adjacency: Vec<[u32; 2]> = pairs
        .rows()
        .into_iter()
        .map(|r| {
            let ok = |v: i64| u32::try_from(v).ok().filter(|&v| (v as usize) < nf);
            match (ok(r[0]), ok(r[1])) {
                (Some(a), Some(b)) => Ok([a, b]),
                _ => Err(PyValueError::new_err("adjacency names a face out of range")),
            }
        })
        .collect::<PyResult<_>>()?;
    let found = py.detach(|| {
        let topo = meshcore::Topology::new(&tris, n);
        let surface = skeliner_pre::organelles::Surface {
            topo: &topo,
            faces: &tris,
            verts: &verts,
            centers: &centers,
            normals: &normals,
            areas: &areas,
            dots: &dots,
        };
        skeliner_pre::organelles::find_organelles(
            &surface,
            &face_comp,
            main,
            &adjacency,
            min_cluster_size,
            &svd,
        )
    });
    let report = PyDict::new(py);
    let iso = &found.isolated;
    let pk = &found.pockets;
    for (k, v) in [
        ("n_structural", found.n_structural),
        ("n_enclosed", found.n_enclosed),
        ("n_internal", iso.n_internal),
        ("n_internal_faces", iso.n_internal_faces),
        ("n_kept", iso.n_kept),
        ("n_via_dots", iso.n_via_dots),
        ("n_via_geometry", iso.n_via_geometry),
        ("n_mouths", pk.n_mouths),
        ("n_initial", pk.n_initial),
        ("n_bridged", pk.n_bridged),
        ("n_holes", pk.n_holes),
        ("n_rejected", pk.n_rejected),
    ] {
        report.set_item(k, v)?;
    }
    Ok((
        Array1::from(found.pocket).into_pyarray(py),
        Array1::from(found.isolated.mask).into_pyarray(py),
        report,
    ))
}

/// One gap as `skeliner.pre.find_gaps` returns it.
type GapTuple = (Vec<u32>, Vec<u32>, f64, i64, i64);

/// `skeliner.pre.find_gaps` once the components are known:
/// `(gaps, n_capped, n_repaired)`.  The thresholds come resolved, in mesh
/// units.
#[pyfunction]
#[pyo3(signature = (
    faces, vertices, centers, labels, labels_pre_fusion, main, disc, fusion_faces,
    tip_rings, fusion_clear, fusion_fallback, max_bridge_ratio, bridge_floor,
    bridge_abs_cap, kiss_r, kiss_penalty, relay_min_faces, repair_stranded, svd_rows,
))]
#[allow(clippy::too_many_arguments)]
fn find_gaps<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    labels: numpy::PyReadonlyArray1<'py, i64>,
    labels_pre_fusion: numpy::PyReadonlyArray1<'py, i64>,
    main: i64,
    disc: Vec<Vec<i64>>,
    fusion_faces: numpy::PyReadonlyArray1<'py, i64>,
    tip_rings: usize,
    fusion_clear: f64,
    fusion_fallback: f64,
    max_bridge_ratio: f64,
    bridge_floor: f64,
    bridge_abs_cap: f64,
    kiss_r: f64,
    kiss_penalty: f64,
    relay_min_faces: usize,
    repair_stranded: bool,
    svd_rows: Bound<'py, PyAny>,
) -> PyResult<(Vec<GapTuple>, usize, usize)> {
    let svd = PySvd {
        rows: svd_rows.unbind(),
        hull: None,
    };
    let verts = rows3("V", &vertices)?;
    let (tris, n) = checked_faces(&faces, Some(verts.len()))?;
    let centers = rows3("F", &centers)?;
    let labels = labels.as_array().to_vec();
    let pre = labels_pre_fusion.as_array().to_vec();
    let nf = tris.len();
    if centers.len() != nf || labels.len() != nf || pre.len() != nf {
        return Err(PyValueError::new_err(
            "per-face arrays need one row per face",
        ));
    }
    let face_id = |f: i64| {
        u32::try_from(f)
            .ok()
            .filter(|&f| (f as usize) < nf)
            .ok_or_else(|| PyValueError::new_err(format!("face index {f} is out of range")))
    };
    let disc: Vec<Vec<u32>> = disc
        .into_iter()
        .map(|c| {
            if c.is_empty() {
                return Err(PyValueError::new_err(
                    "a disconnected component has no faces",
                ));
            }
            c.into_iter().map(face_id).collect()
        })
        .collect::<PyResult<_>>()?;
    let fusion: Vec<u32> = fusion_faces
        .as_array()
        .iter()
        .map(|&f| face_id(f))
        .collect::<PyResult<_>>()?;
    let params = skeliner_pre::find_gaps::GapParams {
        tip_rings,
        fusion_clear,
        fusion_fallback,
        max_bridge_ratio,
        bridge_floor,
        bridge_abs_cap,
        kiss_r,
        kiss_penalty,
        relay_min_faces,
        repair_stranded,
    };
    let found = py.detach(|| {
        let topo = meshcore::Topology::new(&tris, n);
        let mesh = skeliner_pre::find_gaps::GapMesh {
            topo: &topo,
            faces: &tris,
            verts: &verts,
            centers: &centers,
        };
        skeliner_pre::find_gaps::find_gaps(
            &mesh, &labels, &pre, main, &disc, &fusion, &params, &svd,
        )
    });
    let gaps = found
        .gaps
        .into_iter()
        .map(|g| (g.faces_a, g.faces_b, g.dist, g.disc_a, g.disc_b))
        .collect();
    Ok((gaps, found.n_capped, found.n_repaired))
}

/// `skeliner.pre._filter_small_clusters` on arrays.
#[pyfunction]
fn filter_small_clusters<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    face_mask: numpy::PyReadonlyArray1<'py, bool>,
    min_cluster_size: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let tris = faces_u32(&faces)?;
    let mask = face_mask.as_array().to_vec();
    if mask.len() != tris.len() {
        return Err(PyValueError::new_err("face_mask needs one value per face"));
    }
    let keep =
        py.detach(|| meshcore::components::keep_large_clusters(&tris, &mask, min_cluster_size));
    Ok(Array1::from(keep).into_pyarray(py))
}

/// Triangles given as `(F, 3, 3)` flattened to `(F, 9)`.
fn triangles9(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[[f64; 3]; 3]>> {
    let t = a.as_array();
    if t.ncols() != 9 {
        return Err(PyValueError::new_err("triangles must be (F, 9)"));
    }
    Ok(t.rows()
        .into_iter()
        .map(|r| [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]])
        .collect())
}

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
    let hits: Vec<bool> = py.detach(|| {
        use rayon::prelude::*;
        let bvh = meshcore::ray::TriangleBvh::new(tris, normals);
        o.par_iter()
            .zip(d.par_iter())
            .map(|(&o, &d)| bvh.any_hit(o, d))
            .collect()
    });
    Ok(Array1::from(hits).into_pyarray(py))
}

/// A BVH over triangles (`(F, 3, 3)` flattened to `(F, 9)`, `normals`
/// their face normals), built once to cast many rays at, with trimesh's
/// `ray_triangle` test.
#[pyclass(name = "TriangleBvh", module = "skeliner._rs", frozen)]
struct PyTriangleBvh {
    inner: meshcore::ray::TriangleBvh,
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
            inner: py.detach(|| meshcore::ray::TriangleBvh::new(tris, normals)),
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
        let dist: Vec<f64> = py.detach(|| {
            use rayon::prelude::*;
            (0..o.len())
                .into_par_iter()
                .map(|k| {
                    let s = usize::try_from(skip[k]).ok();
                    self.inner
                        .first_hit(o[k], d[k], min_t, s)
                        .map_or(f64::NAN, |(_, h)| h)
                })
                .collect()
        });
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
    let found: Vec<Option<(usize, f64)>> = py.detach(|| {
        use rayon::prelude::*;
        let bvh = meshcore::ray::TriangleBvh::new(tris, other_normals);
        centers
            .par_iter()
            .zip(normals.par_iter())
            .map(|(&c, &n)| {
                let (_, d2) = bvh.closest_within(c, tol)?;
                let dot = |t: usize| {
                    let m = bvh.normal(t);
                    n[0] * m[0] + n[1] * m[1] + n[2] * m[2]
                };
                // the triangles that share the closest distance, rounding aside
                let tied = d2 + 1e-8;
                let (best_dot, best) = bvh
                    .within(c, tied.sqrt())
                    .into_iter()
                    .filter(|&(_, e2)| e2 <= tied)
                    .map(|(t, _)| (dot(t), t))
                    .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))?;
                (best_dot <= max_dot).then_some((best, d2.sqrt()))
            })
            .collect()
    });
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
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    let out: Vec<f64> = py.detach(|| {
        pts.iter()
            .map(|&p| {
                let mut best = f64::INFINITY;
                for &c in &centres {
                    let d = norm(sub(nodes[c], p));
                    if d < best {
                        best = d;
                    }
                }
                for &(a, b) in &segments {
                    let (s, e) = (nodes[a], nodes[b]);
                    let v = sub(e, s);
                    let len2 = dot(v, v);
                    let d = if len2 <= 0.0 {
                        norm(sub(p, s))
                    } else {
                        // Python's min(1.0, max(0.0, t)), which sends NaN to 0
                        let t = dot(sub(p, s), v) / len2;
                        let t = if t > 0.0 { t } else { 0.0 };
                        let t = if t < 1.0 { t } else { 1.0 };
                        norm(sub(p, [s[0] + t * v[0], s[1] + t * v[1], s[2] + t * v[2]]))
                    };
                    if d < best {
                        best = d;
                    }
                }
                best
            })
            .collect()
    });
    Ok(Array1::from(out).into_pyarray(py))
}

/// Faces by vertex, for cutting a mesh down to the faces of a vertex set.
#[pyclass(name = "VertexFaces", module = "skeliner._rs", frozen)]
struct PyVertexFaces {
    start: Vec<usize>,
    faces_of: Vec<u32>,
    faces: Vec<[u32; 3]>,
}

#[pymethods]
impl PyVertexFaces {
    #[new]
    fn new(faces: PyReadonlyArray2<'_, i64>, n_vertices: usize) -> PyResult<Self> {
        let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
        let mut deg = vec![0usize; n + 1];
        for f in &tris {
            for &v in f {
                deg[v as usize + 1] += 1;
            }
        }
        for v in 0..n {
            deg[v + 1] += deg[v];
        }
        let mut fill = deg.clone();
        let mut faces_of = vec![0u32; deg[n]];
        for (fi, f) in tris.iter().enumerate() {
            for &v in f {
                faces_of[fill[v as usize]] = fi as u32;
                fill[v as usize] += 1;
            }
        }
        Ok(Self {
            start: deg,
            faces_of,
            faces: tris,
        })
    }

    /// Faces with all three vertices among `vids`, ascending — the faces
    /// `skeliner.post.submesh_by_vertices` keeps.
    fn faces_within<'py>(
        &self,
        py: Python<'py>,
        vids: numpy::PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let n = self.start.len() - 1;
        let ids = vertex_ids(&vids, n)?;
        let inside: std::collections::HashSet<u32> = ids.iter().copied().collect();
        let mut out: Vec<u32> = ids
            .iter()
            .flat_map(|&v| {
                self.faces_of[self.start[v as usize]..self.start[v as usize + 1]]
                    .iter()
                    .copied()
            })
            .filter(|&f| self.faces[f as usize].iter().all(|v| inside.contains(v)))
            .collect();
        out.sort_unstable();
        out.dedup();
        Ok(Array1::from_iter(out.into_iter().map(i64::from)).into_pyarray(py))
    }
}

/// Shells as Python holds them: per shell, its bins as int64 arrays.
type ShellsOut<'py> = Vec<Vec<Bound<'py, PyArray1<i64>>>>;

fn shells_to_py<'py>(py: Python<'py>, shells: Vec<Vec<Vec<u32>>>) -> ShellsOut<'py> {
    shells
        .into_iter()
        .map(|band| {
            band.into_iter()
                .map(|bin| Array1::from_iter(bin.into_iter().map(i64::from)).into_pyarray(py))
                .collect()
        })
        .collect()
}

fn shells_from_py(shells: Vec<Vec<Vec<i64>>>, n: usize) -> PyResult<Vec<Vec<Vec<u32>>>> {
    shells
        .into_iter()
        .map(|band| {
            band.into_iter()
                .map(|bin| {
                    bin.into_iter()
                        .map(|v| {
                            u32::try_from(v)
                                .ok()
                                .filter(|&v| (v as usize) < n)
                                .ok_or_else(|| {
                                    PyValueError::new_err(format!(
                                        "vertex index {v} is out of range"
                                    ))
                                })
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn vertex_ids(ids: &numpy::PyReadonlyArray1<'_, i64>, n: usize) -> PyResult<Vec<u32>> {
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

/// `skeliner.skeletonize`'s surface graph: one vertex per mesh vertex, one
/// edge per unique mesh edge weighted by its length.
#[pyclass(name = "SurfaceGraph", module = "skeliner._rs", frozen)]
struct PySurfaceGraph {
    inner: meshcore::graph::WeightedGraph,
}

#[pymethods]
impl PySurfaceGraph {
    #[new]
    fn new(
        py: Python<'_>,
        n_vertices: usize,
        edges: PyReadonlyArray2<'_, i64>,
        weights: numpy::PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let e = edges.as_array();
        if e.ncols() != 2 {
            return Err(PyValueError::new_err("edges must be (E, 2)"));
        }
        let w = weights.as_array().to_vec();
        if w.len() != e.nrows() {
            return Err(PyValueError::new_err("one weight per edge"));
        }
        let pairs: Vec<[u32; 2]> = e
            .rows()
            .into_iter()
            .map(|r| {
                let id = |v: i64| {
                    u32::try_from(v)
                        .ok()
                        .filter(|&v| (v as usize) < n_vertices)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!("vertex index {v} is out of range"))
                        })
                };
                Ok([id(r[0])?, id(r[1])?])
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            inner: py.detach(|| meshcore::graph::WeightedGraph::new(n_vertices, &pairs, &w)),
        })
    }

    /// `_bin_one_component` on the preprocessing track: shells of bins
    /// over `verts` (ascending) from the nearest of `seeds`.
    #[pyo3(signature = (verts, seeds, mean_edge, step_size, target_shell_count, min_shell_vertices, max_shell_width_factor))]
    #[allow(clippy::too_many_arguments)]
    fn bin_component<'py>(
        &self,
        py: Python<'py>,
        verts: numpy::PyReadonlyArray1<'py, i64>,
        seeds: numpy::PyReadonlyArray1<'py, i64>,
        mean_edge: f64,
        step_size: Option<f64>,
        target_shell_count: usize,
        min_shell_vertices: usize,
        max_shell_width_factor: f64,
    ) -> PyResult<ShellsOut<'py>> {
        let n = self.inner.n_vertices();
        let (verts, seeds) = (vertex_ids(&verts, n)?, vertex_ids(&seeds, n)?);
        let p = skeliner_pre::skeleton::BinParams {
            step_size,
            target_shell_count,
            min_shell_vertices,
            max_shell_width_factor,
        };
        let g = &self.inner;
        let shells = py
            .detach(|| skeliner_pre::skeleton::bin_one_component(g, &verts, &seeds, mean_edge, &p))
            .map_err(|e| match e {
                skeliner_pre::skeleton::BinError::NoSeed => {
                    PyValueError::new_err("no seed vertex lies in this component")
                }
                skeliner_pre::skeleton::BinError::Unreachable => {
                    PyValueError::new_err("arange: cannot compute length")
                }
            })?;
        Ok(shells_to_py(py, shells))
    }

    /// Keep the re-binned rings over `verts`, fragments merged into them.
    fn recheck_rings<'py>(
        &self,
        py: Python<'py>,
        rebinned: Vec<Vec<Vec<i64>>>,
        verts: numpy::PyReadonlyArray1<'py, i64>,
    ) -> PyResult<ShellsOut<'py>> {
        let n = self.inner.n_vertices();
        let (rebinned, verts) = (shells_from_py(rebinned, n)?, vertex_ids(&verts, n)?);
        let g = &self.inner;
        let shells = py.detach(|| skeliner_pre::skeleton::recheck_rings(g, rebinned, &verts));
        Ok(shells_to_py(py, shells))
    }

    /// Make every bin one connected piece.
    fn reunite_bins<'py>(
        &self,
        py: Python<'py>,
        shells: Vec<Vec<Vec<i64>>>,
    ) -> PyResult<ShellsOut<'py>> {
        let shells = shells_from_py(shells, self.inner.n_vertices())?;
        let g = &self.inner;
        Ok(shells_to_py(
            py,
            py.detach(|| skeliner_pre::skeleton::reunite_bins(g, shells)),
        ))
    }

    /// Split bins that wrap a branch point.
    fn split_branch_bins<'py>(
        &self,
        py: Python<'py>,
        shells: Vec<Vec<Vec<i64>>>,
    ) -> PyResult<ShellsOut<'py>> {
        let shells = shells_from_py(shells, self.inner.n_vertices())?;
        let g = &self.inner;
        Ok(shells_to_py(
            py,
            py.detach(|| skeliner_pre::skeleton::split_branch_bins(g, shells)),
        ))
    }
}

/// What [`rebin_chains`] hands back: each node's new vertices, then the
/// assigned vertices and their centreline distances.
type RebinOut<'py> = (
    Vec<Bound<'py, PyArray1<i64>>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// `_perpendicular_rebin` after the branch donations, which numpy decides:
/// `donated` lists `(branch node, neighbour, vertices)`.
#[pyfunction]
fn rebin_chains<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, i64>,
    node2verts: Vec<Vec<i64>>,
    vertices: PyReadonlyArray2<'py, f64>,
    r_ref: numpy::PyReadonlyArray1<'py, f64>,
    donated: Vec<(i64, i64, Vec<i64>)>,
) -> PyResult<RebinOut<'py>> {
    let nodes = rows3("N", &nodes)?;
    let verts = rows3("V", &vertices)?;
    let r_ref = r_ref.as_array().to_vec();
    let (n, nv) = (nodes.len(), verts.len());
    if r_ref.len() != n || node2verts.len() != n {
        return Err(PyValueError::new_err(
            "one radius and one vertex list per node",
        ));
    }
    let node_id = |v: i64| {
        u32::try_from(v)
            .ok()
            .filter(|&v| (v as usize) < n)
            .ok_or_else(|| PyValueError::new_err(format!("node {v} is out of range")))
    };
    let vert_id = |v: i64| {
        u32::try_from(v)
            .ok()
            .filter(|&v| (v as usize) < nv)
            .ok_or_else(|| PyValueError::new_err(format!("vertex {v} is out of range")))
    };
    let e = edges.as_array();
    if e.ncols() != 2 {
        return Err(PyValueError::new_err("edges must be (E, 2)"));
    }
    let edges: Vec<[u32; 2]> = e
        .rows()
        .into_iter()
        .map(|r| Ok([node_id(r[0])?, node_id(r[1])?]))
        .collect::<PyResult<_>>()?;
    let node2verts: Vec<Vec<u32>> = node2verts
        .into_iter()
        .map(|vs| vs.into_iter().map(vert_id).collect())
        .collect::<PyResult<_>>()?;
    let mut donations = std::collections::HashMap::new();
    for (b, nb, vs) in donated {
        let vs: Vec<u32> = vs.into_iter().map(vert_id).collect::<PyResult<_>>()?;
        donations.insert((node_id(b)?, node_id(nb)?), vs);
    }
    let (out, dists) = py.detach(|| {
        skeliner_pre::skeleton::rebin_chains(
            &nodes,
            &edges,
            &node2verts,
            &verts,
            &r_ref,
            &donations,
        )
    });
    let out = out
        .into_iter()
        .map(|vs| Array1::from_iter(vs.into_iter().map(i64::from)).into_pyarray(py))
        .collect();
    let vids = Array1::from_iter(dists.iter().map(|&(v, _)| i64::from(v))).into_pyarray(py);
    let ds = Array1::from_iter(dists.iter().map(|&(_, d)| d)).into_pyarray(py);
    Ok((out, vids, ds))
}

/// What [`skeleton_nodes`] hands back: centroids and, per estimator, radii.
type NodesOut<'py> = (Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray1<f64>>>);

/// `_make_nodes` without the nested merge: each bin's centroid and its
/// radius under each of `estimators`.
#[pyfunction]
fn skeleton_nodes<'py>(
    py: Python<'py>,
    shells: Vec<Vec<Vec<i64>>>,
    vertices: PyReadonlyArray2<'py, f64>,
    estimators: Vec<String>,
) -> PyResult<NodesOut<'py>> {
    let verts = rows3("V", &vertices)?;
    let shells = shells_from_py(shells, verts.len())?;
    if shells.iter().flatten().any(|b| b.is_empty()) {
        return Err(PyValueError::new_err("a bin has no vertices"));
    }
    let (nodes, radii) = py
        .detach(|| skeliner_pre::skeleton::make_nodes(&shells, &verts, &estimators))
        .ok_or_else(|| {
            PyValueError::new_err(format!("Unknown radius estimator in {estimators:?}."))
        })?;
    let flat: Vec<f64> = nodes.iter().flatten().copied().collect();
    let nodes = Array2::from_shape_vec((nodes.len(), 3), flat)
        .expect("three columns per row")
        .into_pyarray(py);
    Ok((
        nodes,
        radii
            .into_iter()
            .map(|r| Array1::from(r).into_pyarray(py))
            .collect(),
    ))
}

/// A parallel patch crossing the boundary: `(axis, bval, faces, up, down)`.
type PatchTuple = (usize, i64, Vec<i64>, Vec<i64>, Vec<i64>);

/// Faces, vertices and face normals, checked against each other.
type PatchInputs = (Vec<[u32; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>);

fn patch_mesh_inputs(
    faces: &PyReadonlyArray2<'_, i64>,
    vertices: &PyReadonlyArray2<'_, f64>,
    normals: &PyReadonlyArray2<'_, f64>,
) -> PyResult<PatchInputs> {
    let verts = rows3("V", vertices)?;
    let (tris, _) = checked_faces(faces, Some(verts.len()))?;
    let normals = rows3("N", normals)?;
    if normals.len() != tris.len() {
        return Err(PyValueError::new_err("normals needs one row per face"));
    }
    Ok((tris, verts, normals))
}

fn patch_from_tuple(t: PatchTuple, nf: usize) -> PyResult<skeliner_pre::patches::Patch> {
    let ids = |v: Vec<i64>| -> PyResult<Vec<u32>> {
        v.into_iter()
            .map(|f| {
                u32::try_from(f)
                    .ok()
                    .filter(|&f| (f as usize) < nf)
                    .ok_or_else(|| PyValueError::new_err(format!("face index {f} is out of range")))
            })
            .collect()
    };
    if t.0 > 2 {
        return Err(PyValueError::new_err(format!("no axis {}", t.0)));
    }
    Ok(skeliner_pre::patches::Patch {
        axis: t.0,
        bval: t.1,
        faces: ids(t.2)?,
        up: ids(t.3)?,
        down: ids(t.4)?,
    })
}

fn patch_to_tuple(p: skeliner_pre::patches::Patch) -> PatchTuple {
    let ids = |v: Vec<u32>| v.into_iter().map(i64::from).collect();
    (p.axis, p.bval, ids(p.faces), ids(p.up), ids(p.down))
}

/// `skeliner.pre.find_parallel_patches` before its sandwich-overlap filter
/// and final sort.  `boundaries` are `(axis, planes)` as
/// `find_chunk_boundaries` lists them.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn find_parallel_patches<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    boundaries: Vec<(usize, Vec<f64>)>,
    normal_thresh: f64,
    boundary_dist: f64,
    expand_tilted: bool,
    loose_normal_thresh: f64,
) -> PyResult<Vec<PatchTuple>> {
    let (tris, verts, normals) = patch_mesh_inputs(&faces, &vertices, &normals)?;
    if boundaries.iter().any(|(a, _)| *a > 2) {
        return Err(PyValueError::new_err("boundary axes are 0, 1 or 2"));
    }
    let p = skeliner_pre::patches::FindParams {
        normal_thresh,
        boundary_dist,
        expand_tilted,
        loose_normal_thresh,
    };
    let found = py.detach(|| {
        let m = skeliner_pre::patches::PatchMesh {
            faces: &tris,
            verts: &verts,
            normals: &normals,
        };
        skeliner_pre::patches::find_parallel_patches(&m, &boundaries, &p)
    });
    Ok(found.into_iter().map(patch_to_tuple).collect())
}

/// What [`remove_parallel_patches`] hands back: the faces, then the
/// removed, fold, parallel, orphan and stitched counts.
type PatchesRemovedOut<'py> = (Bound<'py, PyArray2<i64>>, usize, usize, usize, usize, usize);

/// `skeliner.pre.remove_parallel_patches` on already trimmed patches.
#[pyfunction]
fn remove_parallel_patches<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    patches: Vec<PatchTuple>,
) -> PyResult<PatchesRemovedOut<'py>> {
    let (tris, verts, normals) = patch_mesh_inputs(&faces, &vertices, &normals)?;
    let patches: Vec<skeliner_pre::patches::Patch> = patches
        .into_iter()
        .map(|t| patch_from_tuple(t, tris.len()))
        .collect::<PyResult<_>>()?;
    let done = py.detach(|| {
        let m = skeliner_pre::patches::PatchMesh {
            faces: &tris,
            verts: &verts,
            normals: &normals,
        };
        skeliner_pre::patches::remove_parallel_patches(&m, &patches)
    });
    let out: Vec<[i64; 3]> = done.faces.iter().map(|f| f.map(i64::from)).collect();
    Ok((
        rows_to_array(&out).into_pyarray(py),
        done.n_removed,
        done.n_fold,
        done.n_parallel,
        done.n_orphan,
        done.n_stitched,
    ))
}

/// What [`remove_gaps`] hands back: new vertices, the rebuilt faces (None
/// when nothing was stitched), and the bridged, removed and skipped counts.
type GapsOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<i64>>>,
    usize,
    usize,
    usize,
);

/// `skeliner.pre.remove_gaps` once the gaps are known.
///
/// `gaps` holds each gap's two tip-face lists; `labels` are the raw
/// face-edge components (computed when a sever check needs them if None);
/// `fit(ids)` is `_fit_loop_circle` of the loop vertices `ids`, returning
/// `(center, radius, normal)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn remove_gaps<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    face_normals: PyReadonlyArray2<'py, f64>,
    median_edge: f64,
    labels: Option<numpy::PyReadonlyArray1<'py, i64>>,
    gaps: Vec<(Vec<i64>, Vec<i64>)>,
    fit: Bound<'py, PyAny>,
) -> PyResult<GapsOut<'py>> {
    use skeliner_pre::stitch::{GapTips, LoopFit, StitchMesh};
    let verts = rows3("V", &vertices)?;
    let (tris, n) = checked_faces(&faces, Some(verts.len()))?;
    let normals = rows3("N", &face_normals)?;
    if normals.len() != tris.len() {
        return Err(PyValueError::new_err("face_normals needs one row per face"));
    }
    let labels = labels.map(|l| l.as_array().to_vec());
    if labels.as_ref().is_some_and(|l| l.len() != tris.len()) {
        return Err(PyValueError::new_err("labels needs one value per face"));
    }
    let nf = tris.len();
    let tip = |c: Vec<i64>| -> PyResult<Vec<u32>> {
        if c.is_empty() {
            return Err(PyValueError::new_err("a gap side has no faces"));
        }
        c.into_iter()
            .map(|f| {
                u32::try_from(f)
                    .ok()
                    .filter(|&f| (f as usize) < nf)
                    .ok_or_else(|| PyValueError::new_err(format!("face index {f} is out of range")))
            })
            .collect()
    };
    let gaps: Vec<GapTips> = gaps
        .into_iter()
        .map(|(a, b)| {
            Ok(GapTips {
                faces_a: tip(a)?,
                faces_b: tip(b)?,
            })
        })
        .collect::<PyResult<_>>()?;
    let mesh = StitchMesh {
        faces: &tris,
        verts: &verts,
        face_normals: &normals,
        median_edge,
    };
    let mut call = |ids: &[u32]| -> PyResult<LoopFit> {
        let arr = Array1::from_iter(ids.iter().map(|&v| i64::from(v))).into_pyarray(py);
        let (center, radius, normal): (Vec<f64>, f64, Vec<f64>) = fit.call1((arr,))?.extract()?;
        if center.len() != 3 || normal.len() != 3 {
            return Err(PyValueError::new_err("fit must return (3,), float, (3,)"));
        }
        Ok(LoopFit {
            center: [center[0], center[1], center[2]],
            radius,
            normal: [normal[0], normal[1], normal[2]],
        })
    };
    let topo = meshcore::Topology::new(&tris, n);
    let done =
        skeliner_pre::stitch::remove_gaps(&topo, &mesh, labels.as_deref(), &gaps, &mut call)?;
    let flat: Vec<f64> = done.new_verts.iter().flatten().copied().collect();
    let new_verts = Array2::from_shape_vec((done.new_verts.len(), 3), flat)
        .expect("three columns per row")
        .into_pyarray(py);
    let faces = done.faces.map(|f| {
        let f: Vec<[i64; 3]> = f.iter().map(|t| t.map(i64::from)).collect();
        rows_to_array(&f).into_pyarray(py)
    });
    Ok((
        new_verts,
        faces,
        done.n_bridged,
        done.n_removed,
        done.n_skipped,
    ))
}

/// `skeliner.pre.remove_fusions` on arrays: fusion faces degenerate to
/// `[0, 0, 0]` and the fans they held together split.  Returns the new
/// faces and, per appended vertex, the vertex it copies.
#[pyfunction]
fn remove_fusions<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    n_vertices: usize,
    fusions: Vec<Vec<i64>>,
) -> PyResult<FacesAndSources<'py>> {
    let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
    let nf = tris.len();
    let fusions: Vec<Vec<u32>> = fusions
        .into_iter()
        .map(|c| {
            c.into_iter()
                .map(|f| {
                    u32::try_from(f)
                        .ok()
                        .filter(|&f| (f as usize) < nf)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!("face index {f} is out of range"))
                        })
                })
                .collect()
        })
        .collect::<PyResult<_>>()?;
    let (out, sources) = py.detach(|| skeliner_pre::fusions::remove_fusions(&tris, n, &fusions));
    let out: Vec<[i64; 3]> = out.iter().map(|f| f.map(i64::from)).collect();
    let sources: Array1<i64> = sources.into_iter().map(i64::from).collect();
    Ok((
        rows_to_array(&out).into_pyarray(py),
        sources.into_pyarray(py),
    ))
}

/// Pieces from [`break_up_classify`] and their kinds.
type PiecesOut<'py> = (Vec<Bound<'py, PyArray1<i64>>>, Bound<'py, PyArray1<u8>>);

/// `skeliner.pre.break_up_mesh`'s pieces: the usable faces that are
/// neither soma nor organelle, by component, largest first, each with its
/// kind (0 candidate, 1 trapped behind organelles, 2 soma).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn break_up_classify<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    n_vertices: usize,
    usable: numpy::PyReadonlyArray1<'py, bool>,
    organelle: numpy::PyReadonlyArray1<'py, bool>,
    soma_face: numpy::PyReadonlyArray1<'py, bool>,
    claimed: numpy::PyReadonlyArray1<'py, bool>,
    has_soma: bool,
) -> PyResult<PiecesOut<'py>> {
    let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
    let flags = |name: &str, a: &numpy::PyReadonlyArray1<'py, bool>| -> PyResult<Vec<bool>> {
        let v = a.as_array().to_vec();
        if v.len() != tris.len() {
            return Err(PyValueError::new_err(format!(
                "{name} needs one value per face"
            )));
        }
        Ok(v)
    };
    let usable = flags("usable", &usable)?;
    let organelle = flags("organelle", &organelle)?;
    let soma_face = flags("soma_face", &soma_face)?;
    let claimed = flags("claimed", &claimed)?;
    let (comps, kinds) = py.detach(|| {
        let m = skeliner_pre::breakup::BreakUpMesh {
            faces: &tris,
            n_vertices: n,
            usable: &usable,
            organelle: &organelle,
            soma_face: &soma_face,
            claimed: &claimed,
        };
        skeliner_pre::breakup::classify(&m, has_soma)
    });
    let comps = comps
        .into_iter()
        .map(|c| Array1::from_iter(c.into_iter().map(i64::from)).into_pyarray(py))
        .collect();
    let kinds = Array1::from_iter(kinds.into_iter().map(|k| k as u8)).into_pyarray(py);
    Ok((comps, kinds))
}

/// What [`find_soma_ring_cutoff`] hands back: the index of the fit that
/// stands, the soma's vertices, the cutoff ring and the last ring.
type SomaOut<'py> = (usize, Bound<'py, PyArray1<i64>>, usize, usize);

/// `skeliner.pre.find_soma_via_ring_cutoff` from the nucleus on.
///
/// `fit(ids, refit)` fits the soma to vertices `ids` (ascending int64) and
/// returns `(center, W, smallest axis)`, or None when a refit fails; the
/// result's first element counts the fits that returned something.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn find_soma_ring_cutoff<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    labels: numpy::PyReadonlyArray1<'py, i64>,
    main: i64,
    seed_vi: numpy::PyReadonlyArray1<'py, i64>,
    center: numpy::PyReadonlyArray1<'py, f64>,
    z_span: f64,
    avg_edge: f64,
    organelle: numpy::PyReadonlyArray1<'py, bool>,
    fit: Bound<'py, PyAny>,
) -> PyResult<SomaOut<'py>> {
    use skeliner_pre::soma::{Fit, FitKind, Nucleus, SomaError, SomaMesh};
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let labels = labels.as_array().to_vec();
    let organelle = organelle.as_array().to_vec();
    if labels.len() != tris.len() || organelle.len() != tris.len() {
        return Err(PyValueError::new_err(
            "labels and organelle need one value per face",
        ));
    }
    let seeds: Vec<u32> = seed_vi
        .as_array()
        .iter()
        .map(|&v| {
            u32::try_from(v)
                .ok()
                .filter(|&v| (v as usize) < verts.len())
                .ok_or_else(|| PyValueError::new_err(format!("vertex index {v} is out of range")))
        })
        .collect::<PyResult<_>>()?;
    let c = center.as_array();
    if c.len() != 3 {
        return Err(PyValueError::new_err("center needs three coordinates"));
    }
    let mesh = SomaMesh {
        faces: &tris,
        verts: &verts,
        labels: &labels,
        main,
        organelle: &organelle,
        avg_edge,
    };
    let nuc = Nucleus {
        center: [c[0], c[1], c[2]],
        z_span,
        seed_vi: &seeds,
    };
    let call = |ids: &[u32], kind: FitKind| -> PyResult<Option<Fit>> {
        let arr = Array1::from_iter(ids.iter().map(|&v| i64::from(v))).into_pyarray(py);
        let out = fit.call1((arr, kind == FitKind::Refit))?;
        if out.is_none() {
            return Ok(None);
        }
        let (centre, w, axes_min): (Vec<f64>, Vec<Vec<f64>>, f64) = out.extract()?;
        if centre.len() != 3 || w.len() != 3 || w.iter().any(|r| r.len() != 3) {
            return Err(PyValueError::new_err("fit must return (3,), (3, 3), float"));
        }
        Ok(Some(Fit {
            ellipsoid: skeliner_pre::soma::Ellipsoid {
                center: [centre[0], centre[1], centre[2]],
                w: [0, 1, 2].map(|i| [w[i][0], w[i][1], w[i][2]]),
            },
            axes_min,
        }))
    };
    let found = skeliner_pre::soma::ring_cutoff(&mesh, &nuc, call).map_err(|e| match e {
        SomaError::Fit(err) => err,
        SomaError::NoRings => PyValueError::new_err("attempt to get argmax of an empty sequence"),
        SomaError::NoExtents => pyo3::exceptions::PyIndexError::new_err(
            "index -1 is out of bounds for axis 0 with size 0",
        ),
    })?;
    let verts_out = Array1::from_iter(found.verts.iter().map(|&v| i64::from(v))).into_pyarray(py);
    Ok((found.fit, verts_out, found.cutoff, found.max_ring))
}

/// Hull equations `(F, 3)` from scipy's `ConvexHull.equations`.
fn hull_rows(eq: Option<PyReadonlyArray2<'_, f64>>) -> PyResult<Option<Vec<[f64; 3]>>> {
    eq.map(|e| rows3("H", &e)).transpose()
}

/// `skeliner.pre._z_scan`'s Z-level scan of a mesh's vertices.
///
/// Construction runs the first pass and finds the nucleus chain;
/// `retry` takes the convex hulls of the chain's two end levels' soma
/// clusters for the second pass.
#[pyclass(name = "ZScan", module = "skeliner._rs")]
struct PyZScan {
    inner: skeliner_pre::nucleus::ZScan,
}

#[pymethods]
impl PyZScan {
    #[new]
    fn new(
        py: Python<'_>,
        vertices: PyReadonlyArray2<'_, f64>,
        grid_res: f64,
        z_tol: f64,
        min_void_r: f64,
        max_shift: f64,
    ) -> PyResult<Self> {
        let verts = rows3("V", &vertices)?;
        let p = skeliner_pre::nucleus::ZParams {
            grid_res,
            z_tol,
            min_void_r,
            max_shift,
        };
        Ok(Self {
            inner: py.detach(|| skeliner_pre::nucleus::ZScan::new(&verts, p)),
        })
    }

    /// Indices of the nucleus chain's levels, or None.
    #[getter]
    fn best(&self) -> Option<Vec<usize>> {
        self.inner.best.clone()
    }

    /// The chain's first and last level, or None.
    fn chain_ends(&self) -> Option<(usize, usize)> {
        self.inner.chain_ends()
    }

    /// Vertex ids of level `i`'s soma cluster, or None if it has none.
    fn soma_vertices<'py>(
        &self,
        py: Python<'py>,
        i: usize,
    ) -> PyResult<Option<Bound<'py, PyArray1<i64>>>> {
        if i >= self.inner.levels.len() {
            return Err(PyValueError::new_err(format!("no level {i}")));
        }
        Ok(self
            .inner
            .soma_vertices(i)
            .map(|v| Array1::from_iter(v.into_iter().map(i64::from)).into_pyarray(py)))
    }

    /// The hull-guided second pass; a missing hull skips that side.
    #[pyo3(signature = (hull_lo=None, hull_hi=None))]
    fn retry(
        &mut self,
        py: Python<'_>,
        hull_lo: Option<PyReadonlyArray2<'_, f64>>,
        hull_hi: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<()> {
        let (lo, hi) = (hull_rows(hull_lo)?, hull_rows(hull_hi)?);
        let scan = &mut self.inner;
        py.detach(|| scan.retry(lo.as_ref(), hi.as_ref()));
        Ok(())
    }

    /// Mean void centre and Z over the chain, or None.
    #[getter]
    fn center<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .center()
            .map(|c| Array1::from(c.to_vec()).into_pyarray(py))
    }

    /// `_z_scan`'s `raw`: per level `(z, cx, cy, void_r, void_xy, vi,
    /// soma_mask)`.
    fn raw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for l in &self.inner.levels {
            let void_xy = l.void_xy.as_ref().map(|xy| {
                let flat: Vec<f64> = xy.iter().flatten().copied().collect();
                Array2::from_shape_vec((xy.len(), 2), flat)
                    .expect("two columns per row")
                    .into_pyarray(py)
            });
            let vi = Array1::from_iter(l.vi.iter().map(|&v| i64::from(v))).into_pyarray(py);
            let mask = l
                .soma_mask
                .as_ref()
                .map(|m| Array1::from(m.clone()).into_pyarray(py));
            out.append((l.z, l.cx, l.cy, l.void_r, void_xy, vi, mask))?;
        }
        Ok(out)
    }
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTopology>()?;
    m.add_class::<PyKdTree>()?;
    m.add_class::<PyZScan>()?;
    m.add_class::<PySurfaceGraph>()?;
    m.add_class::<PyVertexFaces>()?;
    m.add_class::<PyTriangleBvh>()?;
    m.add_function(wrap_pyfunction!(mesh_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(face_edge_components, m)?)?;
    m.add_function(wrap_pyfunction!(face_components, m)?)?;
    m.add_function(wrap_pyfunction!(refine_components_with_fusions, m)?)?;
    m.add_function(wrap_pyfunction!(find_nonmanifold_fusions, m)?)?;
    m.add_function(wrap_pyfunction!(split_fan_vertices, m)?)?;
    m.add_function(wrap_pyfunction!(filter_small_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(outward_dots, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_stats, m)?)?;
    m.add_function(wrap_pyfunction!(find_disconnected, m)?)?;
    m.add_function(wrap_pyfunction!(find_organelles, m)?)?;
    m.add_function(wrap_pyfunction!(find_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(find_soma_ring_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(break_up_classify, m)?)?;
    m.add_function(wrap_pyfunction!(remove_fusions, m)?)?;
    m.add_function(wrap_pyfunction!(remove_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(find_parallel_patches, m)?)?;
    m.add_function(wrap_pyfunction!(remove_parallel_patches, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(rebin_chains, m)?)?;
    m.add_function(wrap_pyfunction!(ray_hits, m)?)?;
    m.add_function(wrap_pyfunction!(contact_faces, m)?)?;
    m.add_function(wrap_pyfunction!(segment_distances, m)?)?;
    Ok(())
}
