//! `skeliner._rs.io`: the `io` crate's mesh readers for `skeliner.io`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict};

/// A mesh's vertices `(V, 3)` and faces `(F, 3)`.
type MeshOut<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

fn mesh_out<'py>(py: Python<'py>, m: ::io::Mesh) -> MeshOut<'py> {
    let flat: Vec<f64> = m.vertices.iter().flatten().copied().collect();
    let vertices = Array2::from_shape_vec((m.vertices.len(), 3), flat).expect("three columns");
    let faces = Array2::from_shape_fn((m.faces.len(), 3), |(i, j)| i64::from(m.faces[i][j]));
    (vertices.into_pyarray(py), faces.into_pyarray(py))
}

/// `e` as Python's `open()` raises it: `OSError(errno, strerror, path)`,
/// which Python makes the matching subclass (`FileNotFoundError`, ...).
fn os_error(path: &str, e: std::io::Error) -> PyErr {
    let Some(errno) = e.raw_os_error() else {
        return PyOSError::new_err(format!("{path}: {e}"));
    };
    let text = e.to_string();
    let suffix = format!(" (os error {errno})");
    let strerror = text.strip_suffix(&suffix).unwrap_or(&text).to_string();
    PyOSError::new_err((errno, strerror, path.to_string()))
}

/// The OBJ at `path` as trimesh reads it: `(vertices, faces)`, or None when
/// it holds more than vertices and triangles (trimesh reads those). Read and
/// parsed with the GIL released.
#[pyfunction]
fn read_obj<'py>(py: Python<'py>, path: &str) -> PyResult<Option<MeshOut<'py>>> {
    let found = py
        .detach(|| std::fs::read(path).map(|data| ::io::read_obj(&data)))
        .map_err(|e| os_error(path, e))?;
    Ok(found.map(|m| mesh_out(py, m)))
}

/// The vertices and triangles of the OpenCTM v5 file at `path`, only for
/// reading EyeWire 1's meshes
/// (<https://drive.google.com/file/d/1-qmE1sEL3AIGpNfi0w50MEOgm2WIF4_1/view>):
/// `(vertices, faces)`, or None for a file that is not OpenCTM v5 or uses
/// another compression than RAW, MG1 or MG2. The LZMA blocks are unpacked
/// by Python's `lzma`.
#[pyfunction]
fn read_ctm<'py>(py: Python<'py>, path: &str) -> PyResult<Option<MeshOut<'py>>> {
    let data = py
        .detach(|| std::fs::read(path))
        .map_err(|e| os_error(path, e))?;
    let lzma = py.import("lzma")?;
    let lzma_error = lzma.getattr("LZMAError")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("format", lzma.getattr("FORMAT_ALONE")?)?;
    // an LZMA "alone" stream: the five property bytes, the unpacked size
    // (all ones for unknown), then the packed data
    let decompress = |props: &[u8], size: [u8; 8], packed: &[u8]| -> PyResult<Vec<u8>> {
        let mut alone = Vec::with_capacity(13 + packed.len());
        alone.extend_from_slice(props);
        alone.extend_from_slice(&size);
        alone.extend_from_slice(packed);
        let out = lzma.call_method("decompress", (PyBytes::new(py, &alone),), Some(&kwargs))?;
        Ok(out.extract::<PyBackedBytes>()?.to_vec())
    };
    // read as the reference library reads a block, up to its size whatever
    // follows: with the size known, and where the data ends in an end
    // marker, which liblzma before 5.4 refuses then, with it unknown
    let mut unlzma = |props: &[u8], packed: &[u8], size: usize| -> PyResult<Vec<u8>> {
        match decompress(props, (size as u64).to_le_bytes(), packed) {
            Err(e) if e.is_instance(py, &lzma_error) => decompress(props, [0xff; 8], packed),
            found => found,
        }
    };
    match ::io::read_ctm(&data, &mut unlzma) {
        Ok(found) => Ok(found.map(|m| mesh_out(py, m))),
        Err(::io::CtmError::Malformed(why)) => Err(PyValueError::new_err(why)),
        Err(::io::CtmError::Unpack(e)) => Err(e),
    }
}

/// Add this module's functions to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_obj, m)?)?;
    m.add_function(wrap_pyfunction!(read_ctm, m)?)?;
    Ok(())
}
