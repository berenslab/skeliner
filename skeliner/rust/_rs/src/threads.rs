//! `skeliner._rs.threads`: the `threads` crate's pool choice for
//! `sk.threads`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Run skeliner's parallel loops on a pool of `n` threads from now on, or
/// on rayon's global pool again with `None` (all cores, or
/// `RAYON_NUM_THREADS`).
#[pyfunction]
#[pyo3(signature = (n=None))]
fn set_threads(n: Option<usize>) -> PyResult<()> {
    let n = match n {
        Some(0) => return Err(PyValueError::new_err("threads must be at least 1")),
        n => n.and_then(std::num::NonZeroUsize::new),
    };
    ::threads::set(n).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// The pool size `set_threads` chose, or None for rayon's global pool.
#[pyfunction]
fn thread_override() -> Option<usize> {
    ::threads::chosen()
}

/// How many threads skeliner's next parallel loop runs on.
#[pyfunction]
fn current_threads() -> usize {
    ::threads::current()
}

/// Add this module's functions and classes to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(thread_override, m)?)?;
    m.add_function(wrap_pyfunction!(current_threads, m)?)?;
    Ok(())
}
