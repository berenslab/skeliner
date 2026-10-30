//! `skeliner.dataclass`'s kernels: the soma's ellipsoid (`soma`), a
//! skeleton's adjacency (`skeleton`), and the arrays a skeleton keeps in
//! step (`state`, for `skeliner._state`).

#![forbid(unsafe_code)]

pub mod skeleton;
pub mod soma;
pub mod state;
