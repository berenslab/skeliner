//! `skeliner._rs.skeletonize`: the `skeletonize` crate's binning for
//! `skeliner.skeletonize`.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::*;

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

/// Shells from Python: per shell its bins, each an int64 array (read in
/// place) or any sequence of ints, every vertex checked against `n`.
fn shells_from_py(shells: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<Vec<Vec<u32>>>> {
    let id = |v: i64| {
        u32::try_from(v)
            .ok()
            .filter(|&v| (v as usize) < n)
            .ok_or_else(|| PyValueError::new_err(format!("vertex index {v} is out of range")))
    };
    shells
        .try_iter()?
        .map(|band| {
            band?
                .try_iter()?
                .map(|bin| {
                    let bin = bin?;
                    match bin.extract::<numpy::PyReadonlyArray1<'_, i64>>() {
                        Ok(a) => a.as_array().iter().map(|&v| id(v)).collect(),
                        Err(_) => bin.extract::<Vec<i64>>()?.into_iter().map(id).collect(),
                    }
                })
                .collect()
        })
        .collect()
}

/// `skeliner.skeletonize`'s surface graph: one vertex per mesh vertex, one
/// edge per unique mesh edge weighted by its length.
#[pyclass(name = "SurfaceGraph", module = "skeliner._rs.skeletonize", frozen)]
struct PySurfaceGraph {
    inner: ::mesh::graph::WeightedGraph,
    edges: Py<PyArray2<i64>>,
    lengths: Py<PyArray1<f64>>,
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
        let inner = py.detach(|| ::mesh::graph::WeightedGraph::new(n_vertices, &pairs, &w));
        Ok(Self {
            inner,
            edges: e.to_owned().into_pyarray(py).unbind(),
            lengths: Array1::from(w).into_pyarray(py).unbind(),
        })
    }

    /// The surface graph of a mesh, from its faces: `edges` and `lengths`
    /// are trimesh's `edges_unique` and `edges_unique_length`, the same
    /// values in the same order, without trimesh's sort of every face edge.
    #[staticmethod]
    fn from_mesh(
        py: Python<'_>,
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Self> {
        let verts = points3(&vertices)?;
        let (tris, n) = checked_faces(&faces, Some(verts.len()))?;
        let (inner, edges, lengths) = py.detach(|| {
            let edges = ::mesh::topology::unique_edges(&tris, n);
            let lengths = ::mesh::geometry::edge_lengths(&verts, &edges);
            let graph = ::mesh::graph::WeightedGraph::new(n, &edges, &lengths);
            (graph, edges, lengths)
        });
        let edges = Array2::from_shape_fn((edges.len(), 2), |(i, j)| i64::from(edges[i][j]));
        Ok(Self {
            inner,
            edges: edges.into_pyarray(py).unbind(),
            lengths: Array1::from(lengths).into_pyarray(py).unbind(),
        })
    }

    /// The edges, `(E, 2)`.
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        self.edges.bind(py).clone()
    }

    /// The edge weights, `(E,)`: their lengths.
    #[getter]
    fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.lengths.bind(py).clone()
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
        let p = ::skeletonize::skeleton::BinParams {
            step_size,
            target_shell_count,
            min_shell_vertices,
            max_shell_width_factor,
        };
        let g = &self.inner;
        let shells = py
            .detach(|| ::skeletonize::skeleton::bin_one_component(g, &verts, &seeds, mean_edge, &p))
            .map_err(|e| match e {
                ::skeletonize::skeleton::BinError::NoSeed => {
                    PyValueError::new_err("no seed vertex lies in this component")
                }
                ::skeletonize::skeleton::BinError::Unreachable => {
                    PyValueError::new_err("arange: cannot compute length")
                }
            })?;
        Ok(shells_to_py(py, shells))
    }

    /// Connected components, each ascending, in order of their smallest
    /// vertex — igraph's `components()`.
    fn components<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<i64>>> {
        let comps = py.detach(|| self.inner.components());
        comps
            .into_iter()
            .map(|c| Array1::from_iter(c.into_iter().map(i64::from)).into_pyarray(py))
            .collect()
    }

    /// `_bin_geodesic_shells` on the direct track's settings: every
    /// connected component binned on its own, in `components()` order, from
    /// `seed` where it lies and otherwise from the component's vertex at its
    /// index modulo its size; the shells of all components in that order.
    #[pyo3(signature = (seed, mean_edge, step_size, target_shell_count, min_shell_vertices, max_shell_width_factor))]
    #[allow(clippy::too_many_arguments)]
    fn bin_mesh<'py>(
        &self,
        py: Python<'py>,
        seed: i64,
        mean_edge: f64,
        step_size: Option<f64>,
        target_shell_count: usize,
        min_shell_vertices: usize,
        max_shell_width_factor: f64,
    ) -> PyResult<ShellsOut<'py>> {
        let n = self.inner.n_vertices();
        let seed = u32::try_from(seed)
            .ok()
            .filter(|&v| (v as usize) < n)
            .ok_or_else(|| PyValueError::new_err(format!("vertex index {seed} is out of range")))?;
        let p = ::skeletonize::skeleton::BinParams {
            step_size,
            target_shell_count,
            min_shell_vertices,
            max_shell_width_factor,
        };
        let g = &self.inner;
        let shells = py
            .detach(|| ::skeletonize::skeleton::bin_mesh(g, seed, mean_edge, &p))
            .map_err(|e| match e {
                ::skeletonize::skeleton::BinError::NoSeed => {
                    PyValueError::new_err("no seed vertex lies in this component")
                }
                ::skeletonize::skeleton::BinError::Unreachable => {
                    PyValueError::new_err("arange: cannot compute length")
                }
            })?;
        Ok(shells_to_py(py, shells))
    }

    /// Keep the re-binned rings over `verts`, fragments merged into them.
    fn recheck_rings<'py>(
        &self,
        py: Python<'py>,
        rebinned: Bound<'py, PyAny>,
        verts: numpy::PyReadonlyArray1<'py, i64>,
    ) -> PyResult<ShellsOut<'py>> {
        let n = self.inner.n_vertices();
        let (rebinned, verts) = (shells_from_py(&rebinned, n)?, vertex_ids(&verts, n)?);
        let g = &self.inner;
        let shells = py.detach(|| ::skeletonize::skeleton::recheck_rings(g, rebinned, &verts));
        Ok(shells_to_py(py, shells))
    }

    /// Make every bin one connected piece.
    fn reunite_bins<'py>(
        &self,
        py: Python<'py>,
        shells: Bound<'py, PyAny>,
    ) -> PyResult<ShellsOut<'py>> {
        let shells = shells_from_py(&shells, self.inner.n_vertices())?;
        let g = &self.inner;
        Ok(shells_to_py(
            py,
            py.detach(|| ::skeletonize::skeleton::reunite_bins(g, shells)),
        ))
    }

    /// Split bins that wrap a branch point.
    fn split_branch_bins<'py>(
        &self,
        py: Python<'py>,
        shells: Bound<'py, PyAny>,
    ) -> PyResult<ShellsOut<'py>> {
        let shells = shells_from_py(&shells, self.inner.n_vertices())?;
        let g = &self.inner;
        Ok(shells_to_py(
            py,
            py.detach(|| ::skeletonize::skeleton::split_branch_bins(g, shells)),
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
        ::skeletonize::skeleton::rebin_chains(
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
    shells: Bound<'py, PyAny>,
    vertices: PyReadonlyArray2<'py, f64>,
    estimators: Vec<String>,
) -> PyResult<NodesOut<'py>> {
    let verts = rows3("V", &vertices)?;
    let shells = shells_from_py(&shells, verts.len())?;
    if shells.iter().flatten().any(|b| b.is_empty()) {
        return Err(PyValueError::new_err("a bin has no vertices"));
    }
    let (nodes, radii) = py
        .detach(|| ::skeletonize::skeleton::make_nodes(&shells, &verts, &estimators))
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

/// Kept mask, kept nodes' vertices and old → new ids from
/// `merge_nested_nodes`.
type MergedOut<'py> = (
    Bound<'py, PyArray1<bool>>,
    Vec<Bound<'py, PyArray1<i64>>>,
    Bound<'py, PyArray1<i64>>,
);

/// `_merge_nested_nodes`: node `j` folds into node `i` when
/// `|c_i - c_j| + inside_frac * r_j <= r_i + tol`; nodes visited biggest
/// first (ties by index), node 0 never absorbing when `keep_root`, each
/// one's candidates within `r_i + max(r)` by index.  Returns the kept mask,
/// the kept nodes' vertex ids and each node's new id.
#[pyfunction]
#[pyo3(signature = (nodes, radii, node2verts, inside_frac=0.9, keep_root=true, tol=1e-6))]
fn merge_nested_nodes<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    radii: numpy::PyReadonlyArray1<'py, f64>,
    node2verts: Vec<numpy::PyReadonlyArray1<'py, i64>>,
    inside_frac: f64,
    keep_root: bool,
    tol: f64,
) -> PyResult<MergedOut<'py>> {
    if !(inside_frac > 0.0 && inside_frac <= 1.0) {
        return Err(PyValueError::new_err("inside_frac must be in (0, 1]."));
    }
    let pts = rows3("N", &nodes)?;
    let r = radii.as_array().to_vec();
    if r.len() != pts.len() || node2verts.len() != pts.len() {
        return Err(PyValueError::new_err(
            "one radius and one vertex list per node",
        ));
    }
    let lists: Vec<Vec<u32>> = node2verts
        .iter()
        .map(|a| vertex_ids(a, u32::MAX as usize))
        .collect::<PyResult<_>>()?;
    let (keep, kept, old2new) = py.detach(|| {
        ::skeletonize::skeleton::merge_nested_nodes(&pts, &r, lists, inside_frac, keep_root, tol)
    });
    Ok((
        Array1::from(keep).into_pyarray(py),
        kept.into_iter()
            .map(|v| Array1::from_iter(v.into_iter().map(i64::from)).into_pyarray(py))
            .collect(),
        Array1::from(old2new).into_pyarray(py),
    ))
}

/// Each node's parent in a breadth-first tree from `root` over the
/// undirected `edges` `(E, 2)`, neighbours in the order the edges name
/// them; −1 for the root and every node it does not reach.
#[pyfunction]
#[pyo3(signature = (edges, n_nodes, root = 0))]
fn bfs_parents<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    root: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let edges = node_pairs(&edges, n_nodes)?;
    let root = usize::try_from(root)
        .ok()
        .filter(|&r| r < n_nodes)
        .ok_or_else(|| {
            PyValueError::new_err(format!("root {root} is not one of {n_nodes} nodes"))
        })?;
    let parent = ::skeletonize::core::bfs_parents(&edges, n_nodes, root);
    Ok(Array1::from_vec(parent).into_pyarray(py))
}

/// The minimum spanning forest of the skeleton `edges` `(E, 2)` weighted by
/// their lengths between `nodes` `(N, 3)`, as igraph's `spanning_tree` picks
/// it (Kruskal's algorithm, ties in igraph's sort order): its edges as
/// sorted pairs, in order, `(K, 2)`.
#[pyfunction]
fn build_mst<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let nodes = rows3("N", &nodes)?;
    let edges = node_pairs(&edges, nodes.len())?;
    let tree = py
        .detach(|| ::skeletonize::core::build_mst(&nodes, &edges))
        .map_err(|_| PyValueError::new_err("edge lengths must not be NaN"))?;
    let out = Array2::from_shape_fn((tree.len(), 2), |(i, j)| tree[i][j] as i64);
    Ok(out.into_pyarray(py))
}

/// A skeleton's arrays as the stages below give them back: nodes `(N, 3)`,
/// radii `(K, N)` (one row per key), the nodes' vertices as CSR (`ptr`
/// `(N + 1,)`, `idx`) and edges `(E, 2)`.
type ArraysOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<i64>>,
);

/// The arrays `_core`'s stages work on, checked: `radii` one row per key and
/// one column per node, `ptr` and `idx` a CSR of each node's vertices,
/// vertex ids within `mesh_vertices` when given.
fn arrays_in(
    nodes: &PyReadonlyArray2<'_, f64>,
    keys: &[String],
    radii: &PyReadonlyArray2<'_, f64>,
    ptr: &numpy::PyReadonlyArray1<'_, i64>,
    idx: &numpy::PyReadonlyArray1<'_, i64>,
    edges: &PyReadonlyArray2<'_, i64>,
    n_mesh: Option<usize>,
) -> PyResult<::skeletonize::core::Arrays> {
    let nodes = rows3("N", nodes)?;
    let n = nodes.len();
    let r = radii.as_array();
    if r.nrows() != keys.len() || r.ncols() != n {
        return Err(PyValueError::new_err(
            "radii must be one row per key, one column per node",
        ));
    }
    let radii = r.rows().into_iter().map(|row| row.to_vec()).collect();
    let (ptr, idx) = (ptr.as_array(), idx.as_array());
    let well_formed = ptr.len() == n + 1
        && ptr[0] == 0
        && ptr.iter().zip(ptr.iter().skip(1)).all(|(a, b)| a <= b)
        && usize::try_from(ptr[n]).ok() == Some(idx.len());
    if !well_formed {
        return Err(PyValueError::new_err(
            "node2verts must be a CSR over the nodes",
        ));
    }
    let vertex = |v: i64| {
        usize::try_from(v)
            .ok()
            .filter(|&v| n_mesh.is_none_or(|m| v < m))
            .ok_or_else(|| PyValueError::new_err(format!("vertex index {v} is out of range")))
    };
    let node2verts = (0..n)
        .map(|i| {
            idx.slice(numpy::ndarray::s![ptr[i] as usize..ptr[i + 1] as usize])
                .iter()
                .map(|&v| vertex(v))
                .collect::<PyResult<Vec<usize>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let edges = node_pairs(edges, n)?;
    Ok(::skeletonize::core::Arrays {
        nodes,
        radii,
        node2verts,
        edges,
    })
}

fn arrays_out<'py>(py: Python<'py>, s: ::skeletonize::core::Arrays) -> ArraysOut<'py> {
    let n = s.nodes.len();
    let nodes = rows_to_array(&s.nodes).into_pyarray(py);
    let radii = Array2::from_shape_fn((s.radii.len(), n), |(k, i)| s.radii[k][i]).into_pyarray(py);
    let mut ptr = Vec::with_capacity(n + 1);
    ptr.push(0i64);
    let mut idx = Vec::new();
    for v in &s.node2verts {
        idx.extend(v.iter().map(|&x| x as i64));
        ptr.push(idx.len() as i64);
    }
    let edges = Array2::from_shape_fn((s.edges.len(), 2), |(i, j)| s.edges[i][j] as i64);
    (
        nodes,
        radii,
        Array1::from_vec(ptr).into_pyarray(py),
        Array1::from_vec(idx).into_pyarray(py),
        edges.into_pyarray(py),
    )
}

/// `_merge_single_node_branches` on a skeleton's arrays (see `arrays_in`):
/// every leaf whose parent from the root has `min_parent_degree` edges or
/// more merged into it, again until none is left; the arrays after, and
/// each node's index before.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn merge_single_node_branches<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    keys: Vec<String>,
    radii: PyReadonlyArray2<'py, f64>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray2<'py, i64>,
    mesh_vertices: Option<PyReadonlyArray2<'py, f64>>,
    min_parent_degree: usize,
) -> PyResult<(ArraysOut<'py>, Bound<'py, PyArray1<i64>>)> {
    let mv = mesh_vertices.as_ref().map(|m| rows3("V", m)).transpose()?;
    let n_mesh = mv.as_ref().map(Vec::len);
    let mut s = arrays_in(&nodes, &keys, &radii, &ptr, &idx, &edges, n_mesh)?;
    let kept = py.detach(|| {
        ::skeletonize::core::merge_single_node_branches(
            &mut s,
            &keys,
            mv.as_deref(),
            min_parent_degree,
        )
    });
    let kept = Array1::from_iter(kept.into_iter().map(|i| i as i64)).into_pyarray(py);
    Ok((arrays_out(py, s), kept))
}

/// `prune_neurites`' result: the arrays, `old2new`, the soma's vertices and
/// how many nodes went into the soma.
type PrunedOut<'py> = (
    ArraysOut<'py>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    usize,
);

/// `_prune_neurites` but the soma's refit, on a skeleton's arrays (see
/// `arrays_in`): the neurites at the root within `stem_extent_factor ×
/// r_soma` of the soma's `center` merged into it, then with
/// `drop_single_node_branches` the single-node branches. The arrays after,
/// `old2new`, the soma's vertices to refit it from, and how many nodes went
/// into the soma.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prune_neurites<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    keys: Vec<String>,
    radii: PyReadonlyArray2<'py, f64>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray2<'py, i64>,
    center: [f64; 3],
    r_soma: f64,
    mesh_vertices: Option<PyReadonlyArray2<'py, f64>>,
    tip_extent_factor: f64,
    stem_extent_factor: f64,
    drop_single_node_branches: bool,
) -> PyResult<PrunedOut<'py>> {
    let mv = mesh_vertices.as_ref().map(|m| rows3("V", m)).transpose()?;
    let n_mesh = mv.as_ref().map(Vec::len);
    let mut s = arrays_in(&nodes, &keys, &radii, &ptr, &idx, &edges, n_mesh)?;
    if s.nodes.is_empty() {
        return Err(PyValueError::new_err("the skeleton has no nodes"));
    }
    let pruned = py.detach(|| {
        ::skeletonize::core::prune_neurites(
            &mut s,
            &keys,
            center,
            r_soma,
            mv.as_deref(),
            tip_extent_factor,
            stem_extent_factor,
            drop_single_node_branches,
        )
    });
    let soma_verts = Array1::from_iter(pruned.soma_verts.iter().map(|&v| v as i64));
    Ok((
        arrays_out(py, s),
        Array1::from_vec(pruned.old2new).into_pyarray(py),
        soma_verts.into_pyarray(py),
        pruned.n_merged,
    ))
}

/// `merge_near_soma_nodes`' result: the arrays, `old2new`, the soma's
/// vertices (`None` while it has none) and how many nodes went into it.
type MergedNearOut<'py> = (
    ArraysOut<'py>,
    Bound<'py, PyArray1<i64>>,
    Option<Bound<'py, PyArray1<i64>>>,
    usize,
);

/// `_merge_near_soma_nodes` but the soma's refit, on a skeleton's arrays
/// (see `arrays_in`) and the soma's ellipsoid (`center`, `axes`, `rotation`
/// `(3, 3)`): the nodes inside the soma, or near it and fat by radius row
/// `radius_row`, merged into the root; their vertices near the surface to
/// the root's and the soma's `soma_verts`. The arrays after, `old2new`, the
/// soma's vertices and how many nodes went into it.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn merge_near_soma_nodes<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    keys: Vec<String>,
    radii: PyReadonlyArray2<'py, f64>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray2<'py, i64>,
    radius_row: usize,
    center: [f64; 3],
    axes: [f64; 3],
    rotation: PyReadonlyArray2<'py, f64>,
    soma_verts: Option<numpy::PyReadonlyArray1<'py, i64>>,
    mesh_vertices: Option<PyReadonlyArray2<'py, f64>>,
    inside_tol: f64,
    near_factor: f64,
    fat_factor: f64,
) -> PyResult<MergedNearOut<'py>> {
    let mv = mesh_vertices.as_ref().map(|m| rows3("V", m)).transpose()?;
    let n_mesh = mv.as_ref().map(Vec::len);
    let mut s = arrays_in(&nodes, &keys, &radii, &ptr, &idx, &edges, n_mesh)?;
    if radius_row >= keys.len() {
        return Err(PyValueError::new_err(format!(
            "radius row {radius_row} is not one of {}",
            keys.len()
        )));
    }
    if s.nodes.is_empty() {
        return Err(PyValueError::new_err("the skeleton has no nodes"));
    }
    let r = rows3("R", &rotation)?;
    let [r0, r1, r2] = r.as_slice() else {
        return Err(PyValueError::new_err("the rotation must be (3, 3)"));
    };
    let soma = ::dataclass::soma::Soma {
        center,
        axes,
        r: [*r0, *r1, *r2],
    };
    let verts = soma_verts.map(|v| v.as_array().to_vec());
    let merged = py.detach(|| {
        ::skeletonize::core::merge_near_soma_nodes(
            &mut s,
            radius_row,
            &soma,
            verts.as_deref(),
            mv.as_deref(),
            inside_tol,
            near_factor,
            fat_factor,
        )
    });
    Ok((
        arrays_out(py, s),
        Array1::from_vec(merged.old2new).into_pyarray(py),
        merged
            .soma_verts
            .map(|v| Array1::from_vec(v).into_pyarray(py)),
        merged.n_merged,
    ))
}

/// `find_soma`'s result: the soma's nodes, the node of largest radius, the
/// soma's centre and radius, and whether there are enough nodes for one.
type FoundSomaOut<'py> = (Bound<'py, PyArray1<i64>>, usize, [f64; 3], f64, bool);

/// `_find_soma` on nodes `(N, 3)` and one radius per node: the nodes at or
/// above the `pct_large`-th percentile of `radius` within `dist_factor`
/// times the largest radius of its node, and a soma centred on their mean
/// when there are `min_keep` or more. None when no node reaches the
/// percentile.
#[pyfunction]
fn find_soma<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    radius: numpy::PyReadonlyArray1<'py, f64>,
    pct_large: f64,
    dist_factor: f64,
    min_keep: usize,
) -> PyResult<Option<FoundSomaOut<'py>>> {
    let nodes = rows3("N", &nodes)?;
    let radius = radius.as_array().to_vec();
    if radius.len() != nodes.len() {
        return Err(PyValueError::new_err("one radius per node"));
    }
    if nodes.is_empty() {
        return Err(PyValueError::new_err("empty skeleton"));
    }
    if !(0.0..=100.0).contains(&pct_large) {
        return Err(PyValueError::new_err(
            "Percentiles must be in the range [0, 100]",
        ));
    }
    let found = ::skeletonize::core::find_soma(&nodes, &radius, pct_large, dist_factor, min_keep);
    Ok(found.map(|f| {
        let soma_nodes = Array1::from_iter(f.soma_nodes.iter().map(|&i| i as i64));
        (
            soma_nodes.into_pyarray(py),
            f.idx_max,
            f.center,
            f.r_max,
            f.has_soma,
        )
    }))
}

/// The soma's vertices from its nodes at `points` `(K, 3)` and their vertex
/// lists (a CSR `ptr` `(K + 1,)`, `idx`): those of the nodes closer to
/// `center` than twice `r_max`, merged, sorted and unique; failing those,
/// the first node's list as it is.
#[pyfunction]
fn soma_vertices<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    center: [f64; 3],
    r_max: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let points = rows3("P", &points)?;
    let (ptr, idx) = (ptr.as_array(), idx.as_array());
    let k = points.len();
    let well_formed = ptr.len() == k + 1
        && ptr[0] == 0
        && ptr.iter().zip(ptr.iter().skip(1)).all(|(a, b)| a <= b)
        && usize::try_from(ptr[k]).ok() == Some(idx.len())
        && idx.iter().all(|&v| v >= 0);
    if !well_formed {
        return Err(PyValueError::new_err(
            "the vertex lists must be a CSR over the points",
        ));
    }
    let lists: Vec<Vec<usize>> = (0..k)
        .map(|i| {
            idx.slice(numpy::ndarray::s![ptr[i] as usize..ptr[i + 1] as usize])
                .iter()
                .map(|&v| v as usize)
                .collect()
        })
        .collect();
    let verts = ::skeletonize::core::soma_vertices(&points, &lists, center, r_max);
    Ok(Array1::from_iter(verts.into_iter().map(|v| v as i64)).into_pyarray(py))
}

/// `_bridge_gaps` on nodes `(N, 3)` and edges `(E, 2)`: every component tied
/// to node 0's by one new edge, nearest first, waiting past
/// `bridge_max_factor` mean edge lengths and measuring every gap again
/// after `bridge_recalc_after` bridges (both chosen from the gaps when
/// None). The edges with the bridges, unique and in order; the edges as
/// they are when there is one component.
#[pyfunction]
fn bridge_gaps<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, i64>,
    bridge_max_factor: Option<f64>,
    bridge_recalc_after: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let nodes = rows3("N", &nodes)?;
    let pairs = node_pairs(&edges, nodes.len())?;
    if nodes.is_empty() {
        return Err(PyValueError::new_err("the skeleton has no nodes"));
    }
    let out = py
        .detach(|| {
            ::skeletonize::core::bridge_gaps(&nodes, &pairs, bridge_max_factor, bridge_recalc_after)
        })
        .unwrap_or(pairs);
    let arr = Array2::from_shape_fn((out.len(), 2), |(i, j)| out[i][j] as i64);
    Ok(arr.into_pyarray(py))
}

/// `_edges_from_mesh` from the mesh `edges` `(E, 2)` and each vertex's node
/// `owner` `(V,)` (−1 for none): the node pairs a mesh edge joins, each
/// once, smaller first, in order, `(K, 2)`.
#[pyfunction]
fn edges_between<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    owner: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let owner = owner.as_array().to_vec();
    let pairs = node_pairs(&edges, owner.len())?;
    let out = py.detach(|| ::skeletonize::skeleton::edges_between(&pairs, &owner));
    let arr = Array2::from_shape_fn((out.len(), 2), |(i, j)| out[i][j] as i64);
    Ok(arr.into_pyarray(py))
}

/// Each node's `centerline` radius: the trimmed mean of its vertices'
/// distances to the centreline — `dists[k]` for vertex `vids[k]`, the last
/// for a vertex listed twice, 0 for one not listed — and 0 for a node
/// without vertices; the nodes' vertices as a CSR (`ptr`, `idx`) into
/// `n_vertices` vertices.
#[pyfunction]
fn centerline_radii<'py>(
    py: Python<'py>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    vids: numpy::PyReadonlyArray1<'py, i64>,
    dists: numpy::PyReadonlyArray1<'py, f64>,
    n_vertices: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (ptr, idx) = (ptr.as_array(), idx.as_array());
    let n = ptr.len().saturating_sub(1);
    let vertex = |v: i64| {
        usize::try_from(v)
            .ok()
            .filter(|&v| v < n_vertices)
            .ok_or_else(|| PyValueError::new_err(format!("vertex index {v} is out of range")))
    };
    let well_formed = !ptr.is_empty()
        && ptr[0] == 0
        && ptr.iter().zip(ptr.iter().skip(1)).all(|(a, b)| a <= b)
        && usize::try_from(ptr[n]).ok() == Some(idx.len());
    if !well_formed {
        return Err(PyValueError::new_err(
            "node2verts must be a CSR over the nodes",
        ));
    }
    let node2verts = (0..n)
        .map(|i| {
            idx.slice(numpy::ndarray::s![ptr[i] as usize..ptr[i + 1] as usize])
                .iter()
                .map(|&v| vertex(v))
                .collect::<PyResult<Vec<usize>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let (vids, dists) = (vids.as_array(), dists.as_array());
    if vids.len() != dists.len() {
        return Err(PyValueError::new_err("one distance per vertex"));
    }
    let mut cl = vec![0.0f64; n_vertices];
    for (&v, &d) in vids.iter().zip(dists.iter()) {
        cl[vertex(v)?] = d;
    }
    let radii = py.detach(|| ::skeletonize::skeleton::centerline_radii(&node2verts, &cl));
    Ok(Array1::from_vec(radii).into_pyarray(py))
}

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edges_between, m)?)?;
    m.add_function(wrap_pyfunction!(centerline_radii, m)?)?;
    m.add_function(wrap_pyfunction!(bridge_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(find_soma, m)?)?;
    m.add_function(wrap_pyfunction!(soma_vertices, m)?)?;
    m.add_function(wrap_pyfunction!(merge_near_soma_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(merge_single_node_branches, m)?)?;
    m.add_function(wrap_pyfunction!(prune_neurites, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_parents, m)?)?;
    m.add_function(wrap_pyfunction!(build_mst, m)?)?;
    m.add_class::<PySurfaceGraph>()?;
    m.add_function(wrap_pyfunction!(rebin_chains, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(merge_nested_nodes, m)?)?;
    Ok(())
}
