//! `skeliner._rs.dx`: the `dx` crate's kernels for `skeliner.dx`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::*;
use ::dx::graph::Graph;

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

/// A skeleton's nodes `(N, 3)`, their neighbours (a CSR `ptr`, `idx`) and a
/// KD-tree over them: what `skeliner.dx.distance` measures against, built
/// once per skeleton.
#[pyclass(name = "SkeletonIndex", module = "skeliner._rs.dx", frozen)]
struct PySkeletonIndex {
    inner: ::dx::SkeletonIndex,
}

#[pymethods]
impl PySkeletonIndex {
    #[new]
    fn new(
        py: Python<'_>,
        nodes: PyReadonlyArray2<'_, f64>,
        ptr: numpy::PyReadonlyArray1<'_, i64>,
        idx: numpy::PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Self> {
        let nodes = rows3("N", &nodes)?;
        let n = nodes.len();
        if n == 0 {
            return Err(PyValueError::new_err(
                "Skeleton has no nodes; cannot compute distances.",
            ));
        }
        let (ptr, idx) = (ptr.as_array(), idx.as_array());
        let well_formed = ptr.len() == n + 1
            && ptr[0] == 0
            && ptr.iter().zip(ptr.iter().skip(1)).all(|(a, b)| a <= b)
            && usize::try_from(ptr[n]).ok() == Some(idx.len())
            && idx.iter().all(|&v| v >= 0 && (v as usize) < n);
        if !well_formed {
            return Err(PyValueError::new_err(
                "the neighbours must be a CSR over the nodes",
            ));
        }
        let adjacency: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                idx.slice(numpy::ndarray::s![ptr[i] as usize..ptr[i + 1] as usize])
                    .iter()
                    .map(|&v| v as usize)
                    .collect()
            })
            .collect();
        let inner = py.detach(|| ::dx::SkeletonIndex::new(nodes, adjacency));
        Ok(Self { inner })
    }

    /// Pickled (and copied) as the arguments that build it again.
    #[allow(clippy::type_complexity)]
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> (
        Bound<'py, PyAny>,
        (
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray1<i64>>,
            Bound<'py, PyArray1<i64>>,
        ),
    ) {
        let py = slf.py();
        let inner = &slf.get().inner;
        let mut ptr = vec![0i64];
        let mut idx = Vec::new();
        for nbrs in inner.adjacency() {
            idx.extend(nbrs.iter().map(|&v| v as i64));
            ptr.push(idx.len() as i64);
        }
        (
            slf.get_type().into_any(),
            (
                rows_to_array(inner.nodes()).into_pyarray(py),
                Array1::from_vec(ptr).into_pyarray(py),
                Array1::from_vec(idx).into_pyarray(py),
            ),
        )
    }

    /// Per point of `points` `(M, 3)`, scaled by `scale_in`, the distance to
    /// the nearest node or edge — to the surface given `radii` (then never
    /// below 0), to the centreline otherwise — scaled by `scale_out`:
    /// against the `k` nearest nodes (ties to the lowest index) and every
    /// edge at one of them, or, given `centres` and `segments` (node ids,
    /// node-id pairs), against those for every point.
    #[allow(clippy::too_many_arguments)]
    fn distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        radii: Option<numpy::PyReadonlyArray1<'py, f64>>,
        k: usize,
        centres: Option<Vec<usize>>,
        segments: Option<Vec<(usize, usize)>>,
        scale_in: f64,
        scale_out: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pts = rows3("P", &points)?;
        let n = self.inner.len();
        let radii = radii.map(|r| r.as_array().to_vec());
        if radii.as_ref().is_some_and(|r| r.len() != n) {
            return Err(PyValueError::new_err("one radius per node"));
        }
        let candidates = match (centres, segments) {
            (None, None) => {
                if k == 0 {
                    return Err(PyValueError::new_err("k_nearest must be at least 1."));
                }
                ::dx::Candidates::Nearest(k.min(n))
            }
            (centres, segments) => {
                let (centres, segments) =
                    (centres.unwrap_or_default(), segments.unwrap_or_default());
                if centres
                    .iter()
                    .chain(segments.iter().flat_map(|(a, b)| [a, b]))
                    .any(|&i| i >= n)
                {
                    return Err(PyValueError::new_err("node index out of range"));
                }
                ::dx::Candidates::Whitelist { centres, segments }
            }
        };
        let out = py.detach(|| {
            self.inner
                .distance(&pts, radii.as_deref(), &candidates, scale_in, scale_out)
        });
        Ok(Array1::from(out).into_pyarray(py))
    }
}

/// The soma's ellipsoid for `voxel_union_count`: centre, semi-axes, `R`'s
/// transpose and its bounding box (`lo`, `hi`).
type SomaIn = ([f64; 3], [f64; 3], [[f64; 3]; 3], [f64; 3], [f64; 3]);

/// `dx.volume`'s voxel union: how many voxels of the grid from `lo` in steps
/// of `h`, `shape` voxels per axis, have their centre in an edge's frustum
/// (nodes `(N, 3)`, one radius per node, edges `(E, 2)`) or, with
/// `include_soma`, in the `soma`'s ellipsoid; without it the soma's voxels
/// are carved out after. Boxes are clipped to `[lo, hi]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn voxel_union_count(
    py: Python<'_>,
    nodes: PyReadonlyArray2<'_, f64>,
    radii: numpy::PyReadonlyArray1<'_, f64>,
    edges: PyReadonlyArray2<'_, i64>,
    lo: [f64; 3],
    hi: [f64; 3],
    h: f64,
    shape: [usize; 3],
    soma: Option<SomaIn>,
    include_soma: bool,
) -> PyResult<u64> {
    let nodes = rows3("N", &nodes)?;
    let radii = radii.as_array().to_vec();
    if radii.len() != nodes.len() {
        return Err(PyValueError::new_err("one radius per node"));
    }
    let edges = node_pairs(&edges, nodes.len())?;
    if h.is_nan() || h <= 0.0 {
        return Err(PyValueError::new_err("voxel_size must be positive"));
    }
    let soma = soma.map(|(center, axes, rt, lo, hi)| ::dx::SomaVoxels {
        center,
        axes,
        rt,
        lo,
        hi,
    });
    Ok(py.detach(|| {
        ::dx::voxel_union_count(
            &nodes,
            &radii,
            &edges,
            lo,
            hi,
            h,
            shape,
            soma.as_ref(),
            include_soma,
        )
    }))
}

/// `total_path_length` within a box: the length of the edges `(E, 2)`
/// between `nodes` `(N, 3)` inside `[lo, hi]`, summed edge by edge in
/// order, and how many edges reach in.
#[pyfunction]
fn path_length_in_box(
    nodes: PyReadonlyArray2<'_, f64>,
    edges: PyReadonlyArray2<'_, i64>,
    lo: [f64; 3],
    hi: [f64; 3],
) -> PyResult<(f64, usize)> {
    let nodes = rows3("N", &nodes)?;
    let edges = node_pairs(&edges, nodes.len())?;
    Ok(::dx::path_length_in_box(&nodes, &edges, lo, hi))
}

/// `arm_cables`' result: junctions, their degrees, their proximal cables,
/// and their distal cables as a CSR (`ptr`, values).
type ArmCablesOut<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// `_arm_cables` on nodes `(N, 3)` and edges `(E, 2)`: every node with
/// `min_degree` edges or more, not a component's root, split into its
/// arms — `(junctions, degrees, proximal, distal_ptr, distal)`, each
/// junction's distal cables largest first.
#[pyfunction]
fn arm_cables<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, i64>,
    min_degree: usize,
) -> PyResult<ArmCablesOut<'py>> {
    let nodes = rows3("N", &nodes)?;
    let edges = node_pairs(&edges, nodes.len())?;
    let arms = py.detach(|| ::dx::arm_cables(&nodes, &edges, min_degree));
    let mut ptr = vec![0i64];
    let mut values = Vec::new();
    for d in &arms.distal {
        values.extend_from_slice(d);
        ptr.push(values.len() as i64);
    }
    let ids = |v: &[usize]| Array1::from_iter(v.iter().map(|&x| x as i64)).into_pyarray(py);
    Ok((
        ids(&arms.junctions),
        ids(&arms.degrees),
        Array1::from_vec(arms.proximal).into_pyarray(py),
        Array1::from_vec(ptr).into_pyarray(py),
        Array1::from_vec(values).into_pyarray(py),
    ))
}

/// A skeleton's edges `(E, 2)` on `n` nodes as a graph.
fn graph(edges: &PyReadonlyArray2<'_, i64>, n: usize) -> PyResult<Graph> {
    Ok(Graph::new(n, &node_pairs(edges, n)?))
}

/// `v` as one of `n` nodes; `what` names it in the error.
fn node_id(v: i64, n: usize, what: &str) -> PyResult<usize> {
    usize::try_from(v)
        .ok()
        .filter(|&v| v < n)
        .ok_or_else(|| PyValueError::new_err(format!("{what} {v} is not one of {n} nodes")))
}

fn ids_out<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyArray1<i64>> {
    Array1::from_iter(v.iter().map(|&x| x as i64)).into_pyarray(py)
}

/// Each of the `n` nodes' component over the `edges` `(E, 2)`, numbered in
/// the order of their lowest node, as igraph numbers them.
#[pyfunction]
fn components<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let g = graph(&edges, n)?;
    Ok(ids_out(py, &g.components()))
}

/// Hops from `source` to each of the `n` nodes over the `edges` `(E, 2)`,
/// infinite where it does not reach.
#[pyfunction]
fn hops<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n: usize,
    source: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = graph(&edges, n)?;
    let source = node_id(source, n, "source")?;
    Ok(Array1::from(g.hops(source)).into_pyarray(py))
}

/// igraph's `minimum_cycle_basis` of the `edges` `(E, 2)` on `n` nodes:
/// each cycle's edge ids in the order they run round it, shortest first.
#[pyfunction]
fn cycle_basis<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n: usize,
) -> PyResult<Vec<Bound<'py, PyArray1<i64>>>> {
    let g = graph(&edges, n)?;
    let basis = py.detach(|| g.minimum_cycle_basis());
    Ok(basis.iter().map(|c| ids_out(py, c)).collect())
}

/// `branches_of_length` on the `edges` `(E, 2)` of `n` nodes: the paths of
/// `k` nodes between nodes of degree other than 2 (node 0 always one), in
/// the order their lower end and its edges come.
#[pyfunction]
fn branches(
    edges: PyReadonlyArray2<'_, i64>,
    n: usize,
    k: i64,
    include_endpoints: bool,
) -> PyResult<Vec<Vec<usize>>> {
    let g = graph(&edges, n)?;
    Ok(usize::try_from(k).map_or_else(|_| Vec::new(), |k| g.branches(k, include_endpoints)))
}

/// `twigs_of_length` on the `edges` `(E, 2)` of `n` nodes: the terminal
/// chains of `k` nodes from each leaf up to its branching node, proximal
/// first, by leaf.
#[pyfunction]
fn twigs(
    edges: PyReadonlyArray2<'_, i64>,
    n: usize,
    k: i64,
    include_branching_node: bool,
) -> PyResult<Vec<Vec<usize>>> {
    let g = graph(&edges, n)?;
    Ok(usize::try_from(k).map_or_else(|_| Vec::new(), |k| g.twigs(k, include_branching_node)))
}

/// `extract_neurites` on the `edges` `(E, 2)` of `n` nodes: `root` and the
/// nodes distal to it from node 0, ascending.
#[pyfunction]
fn neurite<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n: usize,
    root: i64,
    include_root: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let g = graph(&edges, n)?;
    let root = node_id(root, n, "root")?;
    Ok(ids_out(py, &g.neurite(root, include_root)))
}

/// `neurites_out_of_bounds` on `nodes` `(N, 3)` and `edges` `(E, 2)`: the
/// nodes at and beyond each first node outside `[lo, hi]` on the way out
/// from node 0, ascending.
#[pyfunction]
fn neurites_out_of_bounds<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<'py, f64>,
    edges: PyReadonlyArray2<'py, i64>,
    lo: [f64; 3],
    hi: [f64; 3],
    include_root: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let nodes = rows3("N", &nodes)?;
    let g = graph(&edges, nodes.len())?;
    Ok(ids_out(
        py,
        &g.neurites_out_of_bounds(&nodes, lo, hi, include_root),
    ))
}

/// The shortest path from each of `sources` to `target` over the `edges`
/// `(E, 2)` of `n` nodes weighted by `weights` (one per edge, none NaN or
/// negative), summed from the source; infinite where there is none.
#[pyfunction]
fn path_lengths<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n: usize,
    weights: numpy::PyReadonlyArray1<'py, f64>,
    sources: numpy::PyReadonlyArray1<'py, i64>,
    target: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = graph(&edges, n)?;
    let weights = weights.as_array().to_vec();
    if weights.len() != edges.as_array().nrows() {
        return Err(PyValueError::new_err("one weight per edge"));
    }
    if weights.iter().any(|w| w.is_nan()) {
        return Err(PyValueError::new_err("edge lengths must not be NaN"));
    }
    if weights.iter().any(|&w| w < 0.0) {
        return Err(PyValueError::new_err("edge lengths must not be negative"));
    }
    let sources = sources
        .as_array()
        .iter()
        .map(|&v| node_id(v, n, "source"))
        .collect::<PyResult<Vec<_>>>()?;
    let target = node_id(target, n, "target")?;
    let out = py.detach(|| g.path_lengths(&weights, &sources, target));
    Ok(Array1::from(out).into_pyarray(py))
}

/// `check_bins`' result: vertices owned, duplicated, mismatched, the empty
/// bins and the soma's overlap.
type BinCheckOut<'py> = (
    usize,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Vec<usize>,
    usize,
);

/// `check_bins` on the bins as a CSR (`ptr`, `idx`; the first is the soma's
/// `verts`) and `vert2node` as `keys` → `vals`: `(n_owned, duplicated,
/// mismatched, empty_bins, soma_overlap)`.
#[pyfunction]
fn check_bins<'py>(
    py: Python<'py>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    keys: numpy::PyReadonlyArray1<'py, i64>,
    vals: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<BinCheckOut<'py>> {
    let bins = csr_lists(&ptr, &idx, None, "the bins")?;
    let keys = keys
        .as_array()
        .iter()
        .map(|&k| usize::try_from(k))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PyValueError::new_err("vertex ids must not be negative"))?;
    let vals = vals.as_array().to_vec();
    if vals.len() != keys.len() {
        return Err(PyValueError::new_err("one node per vertex"));
    }
    let bins: Vec<&[usize]> = bins.iter().map(Vec::as_slice).collect();
    let c = ::dx::bins::check_bins(&bins, &keys, &vals);
    Ok((
        c.n_owned,
        ids_out(py, &c.duplicated),
        ids_out(py, &c.mismatched),
        c.empty_bins,
        c.soma_overlap,
    ))
}

/// The bins (a CSR `ptr`, `idx` over a mesh of `n_vertices`; the first is
/// the soma's `verts`) in more than one patch of surface over the mesh
/// `edges` `(M, 2)`: `(nodes, patches)`, ascending.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn fragmented_bins<'py>(
    py: Python<'py>,
    ptr: numpy::PyReadonlyArray1<'py, i64>,
    idx: numpy::PyReadonlyArray1<'py, i64>,
    edges: PyReadonlyArray2<'py, i64>,
    n_vertices: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let bins = csr_lists(&ptr, &idx, Some(n_vertices), "the bins")?;
    let edges = node_pairs(&edges, n_vertices)?;
    let bins: Vec<&[usize]> = bins.iter().map(Vec::as_slice).collect();
    let out = py.detach(|| ::dx::bins::fragmented_bins(&bins, &edges, n_vertices));
    let nodes: Vec<usize> = out.iter().map(|&(v, _)| v).collect();
    let pieces: Vec<usize> = out.iter().map(|&(_, p)| p).collect();
    Ok((ids_out(py, &nodes), ids_out(py, &pieces)))
}

/// `face_owner`: per face of `faces` `(F, 3)`, the node `owner` (one per
/// vertex, −1 for none) gives at least two of its vertices, or −1.
#[pyfunction]
fn face_owner<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    owner: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let owner = owner.as_array().to_vec();
    let (faces, _) = checked_faces(&faces, Some(owner.len()))?;
    let faces: Vec<[usize; 3]> = faces.iter().map(|f| f.map(|v| v as usize)).collect();
    let out = py.detach(|| ::dx::bins::face_owner(&faces, &owner));
    Ok(Array1::from(out).into_pyarray(py))
}

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(components, m)?)?;
    m.add_function(wrap_pyfunction!(hops, m)?)?;
    m.add_function(wrap_pyfunction!(cycle_basis, m)?)?;
    m.add_function(wrap_pyfunction!(branches, m)?)?;
    m.add_function(wrap_pyfunction!(twigs, m)?)?;
    m.add_function(wrap_pyfunction!(neurite, m)?)?;
    m.add_function(wrap_pyfunction!(neurites_out_of_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(path_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(check_bins, m)?)?;
    m.add_function(wrap_pyfunction!(fragmented_bins, m)?)?;
    m.add_function(wrap_pyfunction!(face_owner, m)?)?;
    m.add_function(wrap_pyfunction!(arm_cables, m)?)?;
    m.add_function(wrap_pyfunction!(path_length_in_box, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_union_count, m)?)?;
    m.add_class::<PySkeletonIndex>()?;
    m.add_function(wrap_pyfunction!(segment_distances, m)?)?;
    Ok(())
}
