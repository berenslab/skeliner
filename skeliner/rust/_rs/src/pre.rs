//! `skeliner._rs.pre`: the `pre` crate's stages for `skeliner.pre`.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::*;

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
        let topo = ::mesh::Topology::new(&tris, n);
        ::mesh::components::face_edge_components(&topo)
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
    let comps = py.detach(|| ::mesh::components::listed_face_components(&tris, &fi));
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
        ::pre::components::refine_components_with_fusions(&tris, &labels, main, &fusion)
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
        let topo = ::mesh::Topology::new(&tris, n);
        let find = if sort {
            ::pre::fusions::find_fusions
        } else {
            ::pre::fusions::find_nonmanifold_fusions
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
        let topo = ::mesh::Topology::new(&tris, n);
        ::pre::fusions::split_fan_vertices(&topo, &tris, candidates.as_deref())
    });
    let (out, sources) = split.unwrap_or((tris, Vec::new()));
    let out: Vec<[i64; 3]> = out.iter().map(|f| f.map(i64::from)).collect();
    let sources: Array1<i64> = sources.into_iter().map(i64::from).collect();
    Ok((
        rows_to_array(&out).into_pyarray(py),
        sources.into_pyarray(py),
    ))
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
        ::pre::stats::outward_dots(&verts, &tris, &centers, &normals, labels.as_deref(), radius)
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
        let topo = ::mesh::Topology::new(&tris, n);
        let (labels, main) = ::mesh::components::face_edge_components(&topo);
        let vert_comp = ::pre::stats::vertex_components(&tris, &labels, n);
        let dots =
            ::pre::stats::outward_dots(&verts, &tris, &centers, &normals, Some(&vert_comp), radius);
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
            Some(::pre::disconnected::Ellipsoid {
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
        let surface = ::pre::disconnected::Surface {
            faces: &tris,
            verts: &verts,
            centers: &centers,
            normals: &normals,
            median_edge,
        };
        ::pre::disconnected::find_disconnected(
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

/// A mask over the faces and the counts behind it.
type MaskCounts<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyDict>);

fn counts<'py>(py: Python<'py>, items: &[(&str, usize)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for &(k, v) in items {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// One value per face, or an error naming what is short.
fn per_face<T: Copy + numpy::Element>(
    a: numpy::PyReadonlyArray1<'_, T>,
    nf: usize,
    what: &str,
) -> PyResult<Vec<T>> {
    let v = a.as_array().to_vec();
    if v.len() != nf {
        return Err(PyValueError::new_err(format!(
            "{what} needs one value per face"
        )));
    }
    Ok(v)
}

/// `skeliner.pre.find_isolated_organelles` on arrays: `(isolated, counts)`,
/// the counts its verbose report prints.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn isolated_organelles<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    outward_dots: numpy::PyReadonlyArray1<'py, f64>,
    face_comp: numpy::PyReadonlyArray1<'py, i64>,
    main: i64,
) -> PyResult<MaskCounts<'py>> {
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let nf = tris.len();
    let centers = rows3("F", &centers)?;
    let normals = rows3("F", &normals)?;
    if centers.len() != nf || normals.len() != nf {
        return Err(PyValueError::new_err(
            "per-face arrays need one row per face",
        ));
    }
    let dots = per_face(outward_dots, nf, "outward_dots")?;
    let face_comp = per_face(face_comp, nf, "face_comp")?;
    let fragments = ::pre::organelles::Fragments {
        faces: &tris,
        verts: &verts,
        centers: &centers,
        normals: &normals,
        dots: &dots,
    };
    let found = py.detach(|| ::pre::organelles::isolated_organelles(&fragments, &face_comp, main));
    let report = counts(
        py,
        &[
            ("n_internal", found.n_internal),
            ("n_internal_faces", found.n_internal_faces),
            ("n_kept", found.n_kept),
            ("n_via_dots", found.n_via_dots),
            ("n_via_geometry", found.n_via_geometry),
        ],
    )?;
    Ok((Array1::from(found.mask).into_pyarray(py), report))
}

/// The arrays the pocket kernels read, checked: triangles, vertices, areas,
/// outward dots, the main mask and the optional non-main mask.
type PocketArrays = (
    Vec<[u32; 3]>,
    Vec<[f64; 3]>,
    Vec<f64>,
    Vec<f64>,
    Vec<bool>,
    Option<Vec<bool>>,
);

fn pocket_arrays(
    faces: &PyReadonlyArray2<'_, i64>,
    vertices: &PyReadonlyArray2<'_, f64>,
    areas: numpy::PyReadonlyArray1<'_, f64>,
    outward_dots: numpy::PyReadonlyArray1<'_, f64>,
    main_face_mask: numpy::PyReadonlyArray1<'_, bool>,
    non_main_mask: Option<numpy::PyReadonlyArray1<'_, bool>>,
    topology: &::mesh::Topology,
) -> PyResult<PocketArrays> {
    let verts = rows3("V", vertices)?;
    let (tris, _) = checked_faces(faces, Some(verts.len()))?;
    let nf = tris.len();
    if topology.n_faces() != nf {
        return Err(PyValueError::new_err("the topology is not of these faces"));
    }
    let areas = per_face(areas, nf, "areas")?;
    let dots = per_face(outward_dots, nf, "outward_dots")?;
    let main = per_face(main_face_mask, nf, "main_face_mask")?;
    let non_main = non_main_mask
        .map(|m| per_face(m, nf, "non_main_mask"))
        .transpose()?;
    Ok((tris, verts, areas, dots, main, non_main))
}

/// `skeliner.pre.find_pocket_mouths` on arrays and the mesh's `topology`:
/// each pocket's mouth edges, pockets in the order of their lowest face.
#[pyfunction]
#[pyo3(signature = (
    faces, vertices, areas, outward_dots, main_face_mask, topology,
    *, min_pocket_size=5, min_fold_ratio=3.0, non_main_mask=None,
))]
#[allow(clippy::too_many_arguments)]
fn pocket_mouths<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    areas: numpy::PyReadonlyArray1<'py, f64>,
    outward_dots: numpy::PyReadonlyArray1<'py, f64>,
    main_face_mask: numpy::PyReadonlyArray1<'py, bool>,
    topology: PyRef<'py, crate::mesh::PyTopology>,
    min_pocket_size: usize,
    min_fold_ratio: f64,
    non_main_mask: Option<numpy::PyReadonlyArray1<'py, bool>>,
) -> PyResult<Vec<Vec<(u32, u32)>>> {
    let topo = &topology.inner;
    let (tris, verts, areas, dots, main, non_main) = pocket_arrays(
        &faces,
        &vertices,
        areas,
        outward_dots,
        main_face_mask,
        non_main_mask,
        topo,
    )?;
    let params = ::pre::organelles::PocketParams {
        min_pocket_size,
        min_fold_ratio,
        ..Default::default()
    };
    // the pocket kernels read no centres or normals
    let surface = ::pre::organelles::Surface {
        topo,
        faces: &tris,
        verts: &verts,
        centers: &[],
        normals: &[],
        areas: &areas,
        dots: &dots,
    };
    let mouths = py
        .detach(|| ::pre::organelles::pocket_mouths(&surface, &main, non_main.as_deref(), &params));
    Ok(mouths
        .into_iter()
        .map(|m| m.into_iter().map(|[a, b]| (a, b)).collect())
        .collect())
}

/// `skeliner.pre.find_pocket_organelles` on arrays and the mesh's
/// `topology`: `(pocket, counts)`, the counts its verbose report prints.
#[pyfunction]
#[pyo3(signature = (
    faces, vertices, areas, outward_dots, main_face_mask, topology,
    *, centers, normals, grow_threshold=0.1, bridge_threshold=0.3, max_hole_size=500,
    hole_enclosure_ratio=0.5, min_pocket_size=5, min_fold_ratio=3.0, min_cluster_size=5,
    non_main_mask=None,
))]
#[allow(clippy::too_many_arguments)]
fn pocket_organelles<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    areas: numpy::PyReadonlyArray1<'py, f64>,
    outward_dots: numpy::PyReadonlyArray1<'py, f64>,
    main_face_mask: numpy::PyReadonlyArray1<'py, bool>,
    topology: PyRef<'py, crate::mesh::PyTopology>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    grow_threshold: f64,
    bridge_threshold: f64,
    max_hole_size: usize,
    hole_enclosure_ratio: f64,
    min_pocket_size: usize,
    min_fold_ratio: f64,
    min_cluster_size: usize,
    non_main_mask: Option<numpy::PyReadonlyArray1<'py, bool>>,
) -> PyResult<MaskCounts<'py>> {
    let topo = &topology.inner;
    let (tris, verts, areas, dots, main, non_main) = pocket_arrays(
        &faces,
        &vertices,
        areas,
        outward_dots,
        main_face_mask,
        non_main_mask,
        topo,
    )?;
    let params = ::pre::organelles::PocketParams {
        grow_threshold,
        bridge_threshold,
        max_hole_size,
        hole_enclosure_ratio,
        min_pocket_size,
        min_fold_ratio,
        min_cluster_size,
    };
    let (centers, normals) = (rows3("F", &centers)?, rows3("F", &normals)?);
    if centers.len() != tris.len() || normals.len() != tris.len() {
        return Err(PyValueError::new_err(
            "centers and normals need one row per face",
        ));
    }
    let surface = ::pre::organelles::Surface {
        topo,
        faces: &tris,
        verts: &verts,
        centers: &centers,
        normals: &normals,
        areas: &areas,
        dots: &dots,
    };
    let inside = ::pre::organelles::Inside::new(&verts, &tris, &centers, &normals);
    let found = py.detach(|| {
        ::pre::organelles::pocket_organelles(&surface, &main, non_main.as_deref(), &params, &inside)
    });
    let report = counts(
        py,
        &[
            ("n_mouths", found.n_mouths),
            ("n_initial", found.n_initial),
            ("n_bridged", found.n_bridged),
            ("n_holes", found.n_holes),
            ("n_rejected", found.n_rejected),
        ],
    )?;
    Ok((Array1::from(found.mask).into_pyarray(py), report))
}

/// One axis' chunk boundaries: the axis, the coordinates, the vertex
/// count at each and its spike.
type BoundariesOut<'py> = (
    usize,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// `skeliner.pre.find_chunk_boundaries` on `vertices` `(N, 3)`: per axis
/// with any, `(axis, values, counts, spikes)`.
#[pyfunction]
fn find_chunk_boundaries<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
) -> PyResult<Vec<BoundariesOut<'py>>> {
    let verts = rows3("V", &vertices)?;
    let found = py.detach(|| ::pre::patches::find_chunk_boundaries(&verts));
    Ok(found
        .into_iter()
        .map(|b| {
            (
                b.axis,
                Array1::from(b.values).into_pyarray(py),
                Array1::from(b.counts).into_pyarray(py),
                Array1::from(b.spikes).into_pyarray(py),
            )
        })
        .collect())
}

/// `break_up_mesh`'s usable faces: three distinct vertices and a non-zero
/// cross product.
#[pyfunction]
fn usable_faces<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let mask = py.detach(|| ::pre::breakup::usable_faces(&verts, &tris));
    Ok(Array1::from(mask).into_pyarray(py))
}

/// `compact_mesh`'s arrays: the kept faces, the vertex and face maps (−1
/// for what goes), and the compacted vertices and faces.
type CompactOut<'py> = (
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
);

/// `skeliner.pre.compact_mesh`'s arrays: the mesh without its collapsed
/// faces (`[v, v, v]`) and the vertices no face left names, ids renumbered
/// in order.
#[pyfunction]
fn compact_mesh<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
) -> PyResult<CompactOut<'py>> {
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let c = py.detach(|| ::pre::compact::compact(&verts, &tris));
    let faces: Vec<[i64; 3]> = threads::map(&c.faces, |f| f.map(i64::from));
    Ok((
        Array1::from(c.good).into_pyarray(py),
        Array1::from(c.vert_map).into_pyarray(py),
        Array1::from(c.face_map).into_pyarray(py),
        rows_to_array(&c.vertices).into_pyarray(py),
        rows_to_array(&faces).into_pyarray(py),
    ))
}

/// Pocket and isolated masks, and the report.
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
    min_cluster_size=5,
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
    min_cluster_size: usize,
) -> PyResult<OrganelleOut<'py>> {
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
    let ok = |v: i64| u32::try_from(v).ok().filter(|&v| (v as usize) < nf);
    let pair = |a: i64, b: i64| match (ok(a), ok(b)) {
        (Some(a), Some(b)) => Ok([a, b]),
        _ => Err(PyValueError::new_err("adjacency names a face out of range")),
    };
    let adjacency: Vec<[u32; 2]> = match pairs.as_slice() {
        Some(flat) if pairs.ncols() == 2 => {
            threads::map(flat.as_chunks::<2>().0, |r| pair(r[0], r[1]))
                .into_iter()
                .collect::<PyResult<_>>()?
        }
        _ => pairs
            .rows()
            .into_iter()
            .map(|r| pair(r[0], r[1]))
            .collect::<PyResult<_>>()?,
    };
    let found = py.detach(|| {
        let topo = ::mesh::Topology::new(&tris, n);
        let surface = ::pre::organelles::Surface {
            topo: &topo,
            faces: &tris,
            verts: &verts,
            centers: &centers,
            normals: &normals,
            areas: &areas,
            dots: &dots,
        };
        ::pre::organelles::find_organelles(&surface, &face_comp, main, &adjacency, min_cluster_size)
    });
    let report = PyDict::new(py);
    let iso = &found.isolated;
    let pk = &found.pockets;
    for (k, v) in [
        ("n_structural", found.n_structural),
        ("n_enclosed", found.n_enclosed),
        ("n_grown", found.n_grown),
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

/// `find_organelles`' enclosure pass on arrays: `(pocket, n_added)`.
///
/// `adjacency` is trimesh's `face_adjacency`, in its order; `vertices`,
/// `faces`, `centers` and `normals` tell whether a subtree lies inside the
/// cell.
#[pyfunction]
#[pyo3(signature = (
    adjacency, np_mask, pocket, min_cluster_size, *, vertices, faces, centers, normals,
))]
#[allow(clippy::too_many_arguments)]
fn enclosure_pass<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<'py, i64>,
    np_mask: numpy::PyReadonlyArray1<'py, bool>,
    pocket: numpy::PyReadonlyArray1<'py, bool>,
    min_cluster_size: usize,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<bool>>, usize)> {
    let np_mask = np_mask.as_array().to_vec();
    let pocket = pocket.as_array().to_vec();
    if np_mask.len() != pocket.len() {
        return Err(PyValueError::new_err("np_mask and pocket differ in length"));
    }
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let (centers, normals) = (rows3("F", &centers)?, rows3("F", &normals)?);
    if [tris.len(), centers.len(), normals.len()] != [np_mask.len(); 3] {
        return Err(PyValueError::new_err(
            "faces, centers and normals need one row per face of the masks",
        ));
    }
    let pairs = adjacency.as_array();
    if pairs.ncols() != 2 {
        return Err(PyValueError::new_err("adjacency must be (m, 2)"));
    }
    let n = np_mask.len();
    let adjacency: Vec<[u32; 2]> = pairs
        .rows()
        .into_iter()
        .map(|r| {
            let ok = |v: i64| u32::try_from(v).ok().filter(|&v| (v as usize) < n);
            match (ok(r[0]), ok(r[1])) {
                (Some(a), Some(b)) => Ok([a, b]),
                _ => Err(PyValueError::new_err("adjacency names a face out of range")),
            }
        })
        .collect::<PyResult<_>>()?;
    let inside = ::pre::organelles::Inside::new(&verts, &tris, &centers, &normals);
    let holds = |piece: &[u32]| inside.holds(piece);
    let (new, added) = py.detach(|| {
        ::pre::organelles::enclosure_pass(&np_mask, &adjacency, &pocket, min_cluster_size, &holds)
    });
    Ok((Array1::from(new).into_pyarray(py), added))
}

/// `find_organelles`' last step on arrays: the faces reached from `seeds`
/// across faces that look into the cell (a face's normal ray meets another
/// face), over the faces' edge adjacency.
#[pyfunction]
fn grow_into_cell<'py>(
    py: Python<'py>,
    faces: PyReadonlyArray2<'py, i64>,
    vertices: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    seeds: numpy::PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let verts = rows3("V", &vertices)?;
    let (tris, _) = checked_faces(&faces, Some(verts.len()))?;
    let (centers, normals) = (rows3("F", &centers)?, rows3("F", &normals)?);
    let seeds = seeds.as_array().to_vec();
    if [centers.len(), normals.len(), seeds.len()] != [tris.len(); 3] {
        return Err(PyValueError::new_err(
            "centers, normals and seeds need one row per face",
        ));
    }
    let grown = py.detach(|| {
        let topo = ::mesh::Topology::new(&tris, verts.len());
        let inside = ::pre::organelles::Inside::new(&verts, &tris, &centers, &normals);
        ::pre::organelles::grow_into_cell(&topo, &seeds, &inside)
    });
    Ok(Array1::from(grown).into_pyarray(py))
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
    bridge_abs_cap, kiss_r, kiss_penalty, relay_min_faces, repair_stranded,
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
) -> PyResult<(Vec<GapTuple>, usize, usize)> {
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
    let params = ::pre::find_gaps::GapParams {
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
        let topo = ::mesh::Topology::new(&tris, n);
        let mesh = ::pre::find_gaps::GapMesh {
            topo: &topo,
            faces: &tris,
            verts: &verts,
            centers: &centers,
        };
        ::pre::find_gaps::find_gaps(&mesh, &labels, &pre, main, &disc, &fusion, &params)
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
        py.detach(|| ::mesh::components::keep_large_clusters(&tris, &mask, min_cluster_size));
    Ok(Array1::from(keep).into_pyarray(py))
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

fn patch_from_tuple(t: PatchTuple, nf: usize) -> PyResult<::pre::patches::Patch> {
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
    Ok(::pre::patches::Patch {
        axis: t.0,
        bval: t.1,
        faces: ids(t.2)?,
        up: ids(t.3)?,
        down: ids(t.4)?,
    })
}

fn patch_to_tuple(p: ::pre::patches::Patch) -> PatchTuple {
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
    let p = ::pre::patches::FindParams {
        normal_thresh,
        boundary_dist,
        expand_tilted,
        loose_normal_thresh,
    };
    let found = py.detach(|| {
        let m = ::pre::patches::PatchMesh {
            faces: &tris,
            verts: &verts,
            normals: &normals,
        };
        ::pre::patches::find_parallel_patches(&m, &boundaries, &p)
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
    let patches: Vec<::pre::patches::Patch> = patches
        .into_iter()
        .map(|t| patch_from_tuple(t, tris.len()))
        .collect::<PyResult<_>>()?;
    let done = py.detach(|| {
        let m = ::pre::patches::PatchMesh {
            faces: &tris,
            verts: &verts,
            normals: &normals,
        };
        ::pre::patches::remove_parallel_patches(&m, &patches)
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
    use ::pre::stitch::{GapTips, LoopFit, StitchMesh};
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
    let topo = ::mesh::Topology::new(&tris, n);
    let done = ::pre::stitch::remove_gaps(&topo, &mesh, labels.as_deref(), &gaps, &mut call)?;
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
    let (out, sources) = py.detach(|| ::pre::fusions::remove_fusions(&tris, n, &fusions));
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
/// kind (0 candidate, 1 trapped behind organelles, 2 soma).  `vertices`,
/// `centers` and `normals` cast the rays that tell a piece walled in by
/// organelles inside the cell from a neurite cut off by them.
#[pyfunction]
#[pyo3(signature = (
    faces, n_vertices, usable, organelle, soma_face, claimed, has_soma,
    *, vertices, centers, normals,
))]
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
    vertices: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
) -> PyResult<PiecesOut<'py>> {
    let (tris, n) = checked_faces(&faces, Some(n_vertices))?;
    let verts = rows3("V", &vertices)?;
    if verts.len() != n {
        return Err(PyValueError::new_err(
            "vertices need one row per vertex, n_vertices of them",
        ));
    }
    let (centers, normals) = (rows3("F", &centers)?, rows3("F", &normals)?);
    if centers.len() != tris.len() || normals.len() != tris.len() {
        return Err(PyValueError::new_err(
            "centers and normals need one row per face",
        ));
    }
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
        let m = ::pre::breakup::BreakUpMesh {
            faces: &tris,
            n_vertices: n,
            usable: &usable,
            organelle: &organelle,
            soma_face: &soma_face,
            claimed: &claimed,
            verts: &verts,
            centers: &centers,
            normals: &normals,
        };
        ::pre::breakup::classify(&m, has_soma)
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

/// `skeliner.pre.find_soma` from the nucleus on.
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
    use ::pre::soma::{Fit, FitKind, Nucleus, SomaError, SomaMesh};
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
            ellipsoid: ::pre::soma::Ellipsoid {
                center: [centre[0], centre[1], centre[2]],
                w: [0, 1, 2].map(|i| [w[i][0], w[i][1], w[i][2]]),
            },
            axes_min,
        }))
    };
    let found = ::pre::soma::ring_cutoff(&mesh, &nuc, call).map_err(|e| match e {
        SomaError::Fit(err) => err,
        SomaError::NoRings => PyValueError::new_err("attempt to get argmax of an empty sequence"),
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
/// clusters for the second pass (`_rs.mesh.hull_equations`).
#[pyclass(name = "ZScan", module = "skeliner._rs.pre")]
struct PyZScan {
    inner: ::pre::nucleus::ZScan,
}

#[pymethods]
impl PyZScan {
    #[new]
    #[pyo3(signature = (vertices, grid_res, z_tol, min_void_r, max_shift))]
    fn new(
        py: Python<'_>,
        vertices: PyReadonlyArray2<'_, f64>,
        grid_res: f64,
        z_tol: f64,
        min_void_r: Option<f64>,
        max_shift: f64,
    ) -> PyResult<Self> {
        let verts = rows3("V", &vertices)?;
        let p = ::pre::nucleus::ZParams {
            grid_res,
            z_tol,
            min_void_r,
            max_shift,
        };
        Ok(Self {
            inner: py.detach(|| ::pre::nucleus::ZScan::new(&verts, p)),
        })
    }

    /// Indices of the nucleus chain's levels, or None.
    #[getter]
    fn best(&self) -> Option<Vec<usize>> {
        self.inner.best.clone()
    }

    /// The void a level needs to join the chain, or None: no nucleus.
    #[getter]
    fn min_void_r(&self) -> Option<f64> {
        self.inner.min_void_r
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

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(face_edge_components, m)?)?;
    m.add_function(wrap_pyfunction!(face_components, m)?)?;
    m.add_function(wrap_pyfunction!(refine_components_with_fusions, m)?)?;
    m.add_function(wrap_pyfunction!(find_nonmanifold_fusions, m)?)?;
    m.add_function(wrap_pyfunction!(split_fan_vertices, m)?)?;
    m.add_function(wrap_pyfunction!(outward_dots, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_stats, m)?)?;
    m.add_function(wrap_pyfunction!(find_disconnected, m)?)?;
    m.add_function(wrap_pyfunction!(find_organelles, m)?)?;
    m.add_function(wrap_pyfunction!(find_chunk_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(usable_faces, m)?)?;
    m.add_function(wrap_pyfunction!(compact_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(isolated_organelles, m)?)?;
    m.add_function(wrap_pyfunction!(pocket_mouths, m)?)?;
    m.add_function(wrap_pyfunction!(pocket_organelles, m)?)?;
    m.add_function(wrap_pyfunction!(enclosure_pass, m)?)?;
    m.add_function(wrap_pyfunction!(grow_into_cell, m)?)?;
    m.add_function(wrap_pyfunction!(find_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(filter_small_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(find_parallel_patches, m)?)?;
    m.add_function(wrap_pyfunction!(remove_parallel_patches, m)?)?;
    m.add_function(wrap_pyfunction!(remove_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(remove_fusions, m)?)?;
    m.add_function(wrap_pyfunction!(break_up_classify, m)?)?;
    m.add_function(wrap_pyfunction!(find_soma_ring_cutoff, m)?)?;
    m.add_class::<PyZScan>()?;
    Ok(())
}
