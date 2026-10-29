//! The `skeliner._rs` extension module: one submodule per crate of the
//! workspace, each named after the Python module it serves —
//! `_rs.mesh`, `_rs.pre`, `_rs.skeletonize`, `_rs.post`, `_rs.dx`,
//! `_rs.pair` and `_rs.threads`.

use pyo3::prelude::*;

mod common;
mod dataclass;
mod dx;
mod io;
mod mesh;
mod pair;
mod post;
mod pre;
mod skeletonize;
mod threads;

/// Add `skeliner._rs.<name>` to `parent`, built by `register`, and make it
/// importable under that dotted name too.
fn submodule(
    parent: &Bound<'_, PyModule>,
    name: &str,
    register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, name)?;
    register(&m)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(format!("skeliner._rs.{name}"), &m)?;
    Ok(())
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    submodule(m, "mesh", mesh::register)?;
    submodule(m, "dataclass", dataclass::register)?;
    submodule(m, "pre", pre::register)?;
    submodule(m, "skeletonize", skeletonize::register)?;
    submodule(m, "post", post::register)?;
    submodule(m, "dx", dx::register)?;
    submodule(m, "pair", pair::register)?;
    submodule(m, "io", io::register)?;
    submodule(m, "threads", threads::register)?;
    Ok(())
}
