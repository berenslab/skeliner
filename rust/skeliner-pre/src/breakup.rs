//! `skeliner.pre.break_up_mesh`: the pieces left once the soma and the
//! organelles are cut out, and what each one is.

use meshcore::Topology;
use meshcore::components::listed_face_components;

/// What [`classify`] makes of a piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    /// A neurite, or a discard if it is too small.
    Candidate = 0,
    /// Reachable from its component's body only across organelle faces.
    Trapped = 1,
    /// More than half its outer edges are shared with soma faces.
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
}

/// The pieces of the usable faces that are neither soma nor organelle,
/// largest first (ties by lowest face), each with its kind.  The soma test
/// is skipped without a soma, and claimed pieces are never trapped or soma.
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
    let mut in_comp = vec![false; nf];
    let kinds = components
        .iter()
        .map(|comp| {
            let claimed = comp.iter().any(|&f| m.claimed[f as usize]);
            if !claimed && !comp.iter().any(|&f| reachable[f as usize]) {
                return Piece::Trapped;
            }
            if !has_soma || claimed {
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

    fn kinds(
        soma: std::ops::Range<usize>,
        organelle: std::ops::Range<usize>,
        claimed: Option<usize>,
        has_soma: bool,
    ) -> (Vec<Vec<u32>>, Vec<Piece>) {
        let faces = strip(7);
        let nf = faces.len();
        let flags =
            |r: &std::ops::Range<usize>| -> Vec<bool> { (0..nf).map(|f| r.contains(&f)).collect() };
        let claimed: Vec<bool> = (0..nf).map(|f| Some(f) == claimed).collect();
        let (soma_face, organelle, usable) = (flags(&soma), flags(&organelle), vec![true; nf]);
        let m = BreakUpMesh {
            faces: &faces,
            n_vertices: 16,
            usable: &usable,
            organelle: &organelle,
            soma_face: &soma_face,
            claimed: &claimed,
        };
        classify(&m, has_soma)
    }

    #[test]
    fn pieces_behind_organelles_are_trapped_and_soma_bound_ones_are_soma() {
        // quads 0-1 soma; quad 4 organelle splits quads 2-3 from 5-6.  The
        // near piece touches one soma face and one organelle face: half,
        // not more.
        let (comps, k) = kinds(0..4, 8..10, None, true);
        assert_eq!(comps, vec![vec![4, 5, 6, 7], vec![10, 11, 12, 13]]);
        assert_eq!(k, vec![Piece::Candidate, Piece::Trapped]);
        // without the organelle the one piece touches only soma
        let (comps, k) = kinds(0..4, 0..0, None, true);
        assert_eq!(comps.len(), 1);
        assert_eq!(k, vec![Piece::Soma]);
        // a claim or no soma keeps it a candidate
        assert_eq!(kinds(0..4, 0..0, Some(5), true).1, vec![Piece::Candidate]);
        assert_eq!(kinds(0..4, 0..0, None, false).1, vec![Piece::Candidate]);
        // a claim also keeps a trapped piece
        assert_eq!(
            kinds(0..4, 8..10, Some(11), true).1,
            vec![Piece::Candidate, Piece::Candidate]
        );
    }
}
