//! Components that have come apart from the main surface.

use std::collections::BTreeMap;

use mesh::geometry::numpy_order;
use mesh::spatial::{KdTree, Sum};

pub use crate::soma::Ellipsoid;

/// What [`find_disconnected`] found and why it set components aside.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Disconnected {
    /// Kept components, largest first, each its faces ascending.
    pub components: Vec<Vec<u32>>,
    pub n_non_main: usize,
    pub n_soma: usize,
    pub n_organelle: usize,
    pub n_enclosed: usize,
}

/// The inputs [`find_disconnected`] reads about the mesh.
pub struct Surface<'a> {
    pub faces: &'a [[u32; 3]],
    pub verts: &'a [[f64; 3]],
    pub centers: &'a [[f64; 3]],
    pub normals: &'a [[f64; 3]],
    pub median_edge: f64,
}

/// `skeliner.pre.find_disconnected`'s component filter.
///
/// Every component other than `main` (labels below 0 skipped) with at least
/// seven faces is kept unless it is all organelle, its vertex centroid lies
/// in the soma, or it sits within five median edges of the main surface
/// with most of its vertices behind the nearest main face.
pub fn find_disconnected(
    s: &Surface<'_>,
    labels: &[i64],
    main: i64,
    organelle: Option<&[bool]>,
    soma: Option<&Ellipsoid>,
) -> Disconnected {
    let mut comps: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
    let mut first_seen: Vec<i64> = Vec::new();
    for (fi, &c) in labels.iter().enumerate() {
        if c == main || c < 0 {
            continue;
        }
        let entry = comps.entry(c).or_default();
        if entry.is_empty() {
            first_seen.push(c);
        }
        entry.push(fi as u32);
    }
    let main_faces: Vec<usize> = (0..labels.len()).filter(|&f| labels[f] == main).collect();
    let tree = KdTree::new(
        main_faces.iter().map(|&f| s.centers[f]).collect(),
        Sum::Plain,
    );

    let mut out = Disconnected {
        n_non_main: comps.len(),
        ..Default::default()
    };
    let mut kept: Vec<Vec<u32>> = Vec::new();
    for c in first_seen {
        let fis = comps.remove(&c).expect("seen");
        if fis.len() < 7 {
            continue;
        }
        if let Some(mask) = organelle
            && fis.iter().all(|&f| mask[f as usize])
        {
            out.n_organelle += 1;
            continue;
        }
        let mut vids: Vec<u32> = fis.iter().flat_map(|&f| s.faces[f as usize]).collect();
        vids.sort_unstable();
        vids.dedup();
        let coords: Vec<[f64; 3]> = vids.iter().map(|&v| s.verts[v as usize]).collect();
        let mut sum = [0.0f64; 3];
        for p in &coords {
            for k in 0..3 {
                sum[k] += p[k];
            }
        }
        let n = coords.len() as f64;
        let centroid = [sum[0] / n, sum[1] / n, sum[2] / n];
        if soma.is_some_and(|e| e.contains(&centroid)) {
            out.n_soma += 1;
            continue;
        }
        if !tree.is_empty() && is_enclosed(s, &tree, &main_faces, &coords) {
            out.n_enclosed += 1;
            continue;
        }
        kept.push(fis);
    }
    kept.sort_by_key(|c| std::cmp::Reverse(c.len()));
    out.components = kept;
    out
}

/// Whether a component near the main surface (median vertex distance at
/// most five median edges) has a strict majority of vertices behind their
/// nearest main face.
fn is_enclosed(s: &Surface<'_>, tree: &KdTree, main_faces: &[usize], coords: &[[f64; 3]]) -> bool {
    let mut dists = Vec::with_capacity(coords.len());
    let mut dots = Vec::with_capacity(coords.len());
    for p in coords {
        let (j, _) = tree.nearest(p).expect("non-empty");
        let f = main_faces[j as usize];
        let c = s.centers[f];
        let v = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        dists.push((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
        let nrm = s.normals[f];
        dots.push(v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2]);
    }
    if median(&mut dists) > 5.0 * s.median_edge {
        return false;
    }
    let behind = dots.iter().filter(|&&d| d < 0.0).count();
    behind as f64 > dots.len() as f64 / 2.0
}

/// numpy's median: the middle value, or the mean of the middle two.
fn median(x: &mut [f64]) -> f64 {
    x.sort_unstable_by(numpy_order);
    let n = x.len();
    if n % 2 == 1 {
        x[n / 2]
    } else {
        (x[n / 2 - 1] + x[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unit_sphere_contains_its_inside() {
        let e = Ellipsoid {
            center: [0.0; 3],
            w: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        assert!(e.contains(&[0.5, 0.5, 0.5]));
        assert!(e.contains(&[1.0, 0.0, 0.0]));
        assert!(!e.contains(&[1.0, 0.1, 0.0]));
    }

    #[test]
    fn numpy_median() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
