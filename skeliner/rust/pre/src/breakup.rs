//! `skeliner.pre.break_up_mesh`: the pieces left once the soma and the
//! organelles are cut out, and what each one is.

use mesh::Topology;
use mesh::components::listed_face_components;
use rayon::prelude::*;

use crate::organelles::Inside;

/// What [`classify`] makes of a piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    /// A neurite, or a discard if it is too small.
    Candidate = 0,
    /// Reachable from its component's body only across organelle faces,
    /// and an organelle's inside: no bigger than the organelles walling it,
    /// or looking into the cell.  A neurite cut off by a ring of folds at
    /// its base is bigger and looks out of it.
    Trapped = 1,
    /// More than half its outer edges are shared with soma faces, and it
    /// has no more faces than the soma.
    Soma = 2,
}

/// The inputs [`classify`] reads, one flag per face.
pub struct BreakUpMesh<'a> {
    pub faces: &'a [[u32; 3]],
    pub n_vertices: usize,
    /// Three distinct vertices and a non-zero area.
    pub usable: &'a [bool],
    pub organelle: &'a [bool],
    /// At least two soma vertices.
    pub soma_face: &'a [bool],
    /// Named by the user as arbor.
    pub claimed: &'a [bool],
    /// The corners, and per face its centre and normal, for the rays that
    /// tell a piece inside the cell from one looking out of it.
    pub verts: &'a [[f64; 3]],
    pub centers: &'a [[f64; 3]],
    pub normals: &'a [[f64; 3]],
}

/// The pieces of the usable faces that are neither soma nor organelle,
/// largest first (ties by lowest face), each with its kind.  The soma test
/// is skipped without a soma, and claimed pieces are never trapped or soma.
/// Nor is a piece with more faces than the soma: a piece `find_soma` missed
/// is a patch of the soma's surface, while a neurite leaving the soma
/// reads as soma by its boundary at any length (its far end is capped), so
/// an organelle-free arbor all one piece would be taken whole.
pub fn classify(m: &BreakUpMesh<'_>, has_soma: bool) -> (Vec<Vec<u32>>, Vec<Piece>) {
    let nf = m.faces.len();
    let pick = |keep: &dyn Fn(usize) -> bool| -> Vec<u32> {
        (0..nf).filter(|&f| keep(f)).map(|f| f as u32).collect()
    };

    // each mesh component's body: its largest piece once organelles are
    // removed, the lowest label among equals
    let non_org = listed_face_components(m.faces, &pick(&|f| m.usable[f] && !m.organelle[f]));
    let mut label = vec![u32::MAX; nf];
    for (ci, comp) in non_org.iter().enumerate() {
        for &f in comp {
            label[f as usize] = ci as u32;
        }
    }
    let mut is_body = vec![false; non_org.len()];
    let mut count = vec![0usize; non_org.len()];
    for mc in listed_face_components(m.faces, &pick(&|f| m.usable[f])) {
        let mut present: Vec<u32> = mc
            .iter()
            .map(|&f| label[f as usize])
            .filter(|&l| l != u32::MAX)
            .collect();
        if present.is_empty() {
            continue;
        }
        for &l in &present {
            count[l as usize] += 1;
        }
        present.sort_unstable();
        present.dedup();
        let mut best = present[0];
        for &l in &present[1..] {
            if count[l as usize] > count[best as usize] {
                best = l;
            }
        }
        is_body[best as usize] = true;
        for &l in &present {
            count[l as usize] = 0;
        }
    }
    let reachable: Vec<bool> = label
        .iter()
        .map(|&l| l != u32::MAX && is_body[l as usize])
        .collect();

    let remain = pick(&|f| m.usable[f] && !m.soma_face[f] && !m.organelle[f]);
    let components = listed_face_components(m.faces, &remain);
    if components.is_empty() {
        return (components, Vec::new());
    }
    let topo = Topology::new(m.faces, m.n_vertices);
    let soma_size = (0..nf).filter(|&f| m.usable[f] && m.soma_face[f]).count();
    // the organelles' pieces, which a cut-off piece is weighed against
    let org_comps = listed_face_components(m.faces, &pick(&|f| m.usable[f] && m.organelle[f]));
    let mut org_of = vec![u32::MAX; nf];
    for (k, oc) in org_comps.iter().enumerate() {
        for &f in oc {
            org_of[f as usize] = k as u32;
        }
    }
    // a piece cut off from the cell's body is trapped when no bigger than
    // the organelles walling it, or lying inside the cell: the rays of the
    // bigger ones cast across the pool
    let wraps: Vec<Option<usize>> = components
        .iter()
        .map(|comp| {
            let cut_off = !comp.iter().any(|&f| m.claimed[f as usize])
                && !comp.iter().any(|&f| reachable[f as usize]);
            cut_off.then(|| {
                let mut around: Vec<u32> = comp
                    .iter()
                    .flat_map(|&f| topo.face_neighbors(f as usize))
                    .map(|g| org_of[g as usize])
                    .filter(|&k| k != u32::MAX)
                    .collect();
                around.sort_unstable();
                around.dedup();
                around.iter().map(|&k| org_comps[k as usize].len()).sum()
            })
        })
        .collect();
    let mut trapped: Vec<bool> = components
        .iter()
        .zip(&wraps)
        .map(|(comp, wrap)| wrap.is_some_and(|w| comp.len() <= w))
        .collect();
    let cast: Vec<usize> = (0..components.len())
        .filter(|&i| wraps[i].is_some() && !trapped[i])
        .collect();
    if !cast.is_empty() {
        let inside = Inside::new(m.verts, m.faces, m.centers, m.normals);
        inside.prepare();
        let held: Vec<bool> = threads::install(|| {
            cast.par_iter()
                .map(|&i| inside.holds(&components[i]))
                .collect()
        });
        for (&i, h) in cast.iter().zip(held) {
            trapped[i] = h;
        }
    }
    let mut in_comp = vec![false; nf];
    let kinds = components
        .iter()
        .zip(&trapped)
        .map(|(comp, &trapped)| {
            if trapped {
                return Piece::Trapped;
            }
            let claimed = comp.iter().any(|&f| m.claimed[f as usize]);
            if !has_soma || claimed || comp.len() > soma_size {
                return Piece::Candidate;
            }
            for &f in comp {
                in_comp[f as usize] = true;
            }
            // (face, edge, outside neighbour) incidences across the
            // piece's boundary, and how many reach soma faces
            let (mut n_soma, mut n_total) = (0usize, 0usize);
            for &f in comp {
                for e in topo
                    .face_edges(f as usize)
                    .expect("usable faces have edges")
                {
                    for &nb in topo.edge_faces(e as usize) {
                        let nb = nb as usize;
                        if m.usable[nb] && !in_comp[nb] {
                            n_total += 1;
                            n_soma += usize::from(m.soma_face[nb]);
                        }
                    }
                }
            }
            for &f in comp {
                in_comp[f as usize] = false;
            }
            let soma_frac = if n_total > 0 {
                n_soma as f64 / n_total as f64
            } else {
                0.0
            };
            if soma_frac > 0.5 {
                Piece::Soma
            } else {
                Piece::Candidate
            }
        })
        .collect();
    (components, kinds)
}

/// `_usable_face_mask`: faces with three distinct vertices whose cross
/// product `(v1 − v0) × (v2 − v0)`, computed as numpy computes it, has a
/// norm above zero.
pub fn usable_faces(verts: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<bool> {
    threads::map(faces, |&[a, b, c]| {
        if a == b || b == c || a == c {
            return false;
        }
        let (p, q, r) = (verts[a as usize], verts[b as usize], verts[c as usize]);
        let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let w = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
        let x = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt() > 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A strip of quads: faces 2k, 2k+1 make quad k.
    fn strip(n: usize) -> Vec<[u32; 3]> {
        let mut f = Vec::new();
        for k in 0..n as u32 {
            let (a, b, c, d) = (2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3);
            f.push([a, b, d]);
            f.push([a, d, c]);
        }
        f
    }

    /// The strip's corners laid flat, or rolled into a tube whose faces
    /// look at its axis (the inside of a cavity).
    fn corners(n: usize, rolled: bool) -> Vec<[f64; 3]> {
        (0..=n)
            .flat_map(|k| {
                let (x, y) = if rolled {
                    let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                    (a.cos(), a.sin())
                } else {
                    (k as f64, 0.0)
                };
                [[x, y, 0.0], [x, y + f64::from(!rolled), f64::from(rolled)]]
            })
            .collect()
    }

    fn centres_and_normals(v: &[[f64; 3]], f: &[[u32; 3]]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        f.iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| v[i as usize]);
                let centre = [0, 1, 2].map(|i| (a[i] + b[i] + c[i]) / 3.0);
                let (u, w) = (
                    [0, 1, 2].map(|i| b[i] - a[i]),
                    [0, 1, 2].map(|i| c[i] - a[i]),
                );
                let x = [
                    u[1] * w[2] - u[2] * w[1],
                    u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0],
                ];
                let len = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
                (centre, x.map(|c| c / len))
            })
            .unzip()
    }

    fn kinds_on(
        rolled: bool,
        soma: std::ops::Range<usize>,
        organelle: std::ops::Range<usize>,
        claimed: Option<usize>,
        has_soma: bool,
    ) -> (Vec<Vec<u32>>, Vec<Piece>) {
        let faces = strip(7);
        let verts = corners(7, rolled);
        let (centers, normals) = centres_and_normals(&verts, &faces);
        let nf = faces.len();
        let flags =
            |r: &std::ops::Range<usize>| -> Vec<bool> { (0..nf).map(|f| r.contains(&f)).collect() };
        let claimed: Vec<bool> = (0..nf).map(|f| Some(f) == claimed).collect();
        let (soma_face, organelle, usable) = (flags(&soma), flags(&organelle), vec![true; nf]);
        let m = BreakUpMesh {
            faces: &faces,
            n_vertices: verts.len(),
            usable: &usable,
            organelle: &organelle,
            soma_face: &soma_face,
            claimed: &claimed,
            verts: &verts,
            centers: &centers,
            normals: &normals,
        };
        classify(&m, has_soma)
    }

    fn kinds(
        soma: std::ops::Range<usize>,
        organelle: std::ops::Range<usize>,
        claimed: Option<usize>,
        has_soma: bool,
    ) -> (Vec<Vec<u32>>, Vec<Piece>) {
        kinds_on(false, soma, organelle, claimed, has_soma)
    }

    #[test]
    fn pieces_behind_organelles_are_trapped_and_soma_bound_ones_are_soma() {
        // quads 0-1 soma; quads 3-4 organelle split quad 2 from quads 5-6,
        // which are no bigger than the organelle.  The near piece touches
        // one soma face and one organelle face: half, not more.
        let (comps, k) = kinds(0..4, 6..10, None, true);
        assert_eq!(comps, vec![vec![10, 11, 12, 13], vec![4, 5]]);
        assert_eq!(k, vec![Piece::Trapped, Piece::Candidate]);
        // quads 0-3 soma, no organelle: the one piece, quads 4-6, touches
        // only soma
        let (comps, k) = kinds(0..8, 0..0, None, true);
        assert_eq!(comps.len(), 1);
        assert_eq!(k, vec![Piece::Soma]);
        // a claim or no soma keeps it a candidate
        assert_eq!(kinds(0..8, 0..0, Some(9), true).1, vec![Piece::Candidate]);
        assert_eq!(kinds(0..8, 0..0, None, false).1, vec![Piece::Candidate]);
        // and so does more faces than the soma: quads 2-6 on quads 0-1
        assert_eq!(kinds(0..4, 0..0, None, true).1, vec![Piece::Candidate]);
        // a claim also keeps a trapped piece
        assert_eq!(
            kinds(0..4, 6..10, Some(11), true).1,
            vec![Piece::Candidate, Piece::Candidate]
        );
    }

    #[test]
    fn a_piece_bigger_than_its_walls_is_trapped_only_inside_the_cell() {
        // quad 4 alone organelle: quads 5-6 behind it are bigger.  Laid
        // flat they look out, a neurite cut off; rolled into a cavity they
        // look in, the organelle's inside
        let (comps, k) = kinds_on(false, 0..4, 8..10, None, true);
        assert_eq!(comps, vec![vec![4, 5, 6, 7], vec![10, 11, 12, 13]]);
        assert_eq!(k, vec![Piece::Candidate, Piece::Candidate]);
        let (_, k) = kinds_on(true, 0..4, 8..10, None, true);
        assert_eq!(k[1], Piece::Trapped);
    }
}
