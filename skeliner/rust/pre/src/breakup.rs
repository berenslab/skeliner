//! `skeliner.pre.break_up_mesh`: the pieces left once the soma and the
//! organelles are cut out, and what each one is.

use std::collections::{HashMap, HashSet, VecDeque};

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
    let reachable = bodies(m.faces, m.usable, m.organelle);

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

/// Per face, whether it lies on its mesh component's body: the component's
/// largest piece once the organelles are out (counted in faces, the lowest
/// label among equals), soma faces and all.
fn bodies(faces: &[[u32; 3]], usable: &[bool], organelle: &[bool]) -> Vec<bool> {
    let nf = faces.len();
    let pick = |keep: &dyn Fn(usize) -> bool| -> Vec<u32> {
        (0..nf).filter(|&f| keep(f)).map(|f| f as u32).collect()
    };
    let non_org = listed_face_components(faces, &pick(&|f| usable[f] && !organelle[f]));
    let mut label = vec![u32::MAX; nf];
    for (ci, comp) in non_org.iter().enumerate() {
        for &f in comp {
            label[f as usize] = ci as u32;
        }
    }
    let mut is_body = vec![false; non_org.len()];
    let mut count = vec![0usize; non_org.len()];
    for mc in listed_face_components(faces, &pick(&|f| usable[f])) {
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
    label
        .iter()
        .map(|&l| l != u32::MAX && is_body[l as usize])
        .collect()
}

/// The organelle faces that join severed arbor back to the cell
/// (`break_up_mesh`): a piece [`classify`] found cut off from its mesh
/// component's body yet not trapped — bigger than the organelles walling
/// it and looking out of the cell, arbor a ring of folds severed — is
/// joined back across each wall touching both it and the body (an
/// organelle piece next to both, the trapped pieces in it counted as its
/// own: an organelle's inside) by the wall's [`corridor`] between them.
/// A wall holding a face `fixed` (an organelle drawn by hand) stays, and a
/// `claimed` piece is the user's to place.  `pieces` and `kinds` are
/// [`classify`]'s on the same faces.  Returns the organelle faces of the
/// corridors, ascending.
#[allow(clippy::too_many_arguments)]
pub fn severed_walls(
    faces: &[[u32; 3]],
    n_vertices: usize,
    usable: &[bool],
    organelle: &[bool],
    claimed: &[bool],
    fixed: &[bool],
    pieces: &[Vec<u32>],
    kinds: &[Piece],
) -> Vec<u32> {
    let nf = faces.len();
    let reachable = bodies(faces, usable, organelle);
    let topo = Topology::new(faces, n_vertices);
    // a wall is organelle faces and the trapped pieces among them — an
    // organelle's inside is the organelle's: a piece can be severed from
    // the body by two walls with an inside between
    let mut wall_face: Vec<bool> = (0..nf).map(|f| usable[f] && organelle[f]).collect();
    for (p, kind) in pieces.iter().zip(kinds) {
        if *kind == Piece::Trapped {
            for &f in p {
                wall_face[f as usize] = true;
            }
        }
    }
    let walls = listed_face_components(
        faces,
        &(0..nf as u32)
            .filter(|&f| wall_face[f as usize])
            .collect::<Vec<_>>(),
    );
    let mut wall_of = vec![u32::MAX; nf];
    for (k, w) in walls.iter().enumerate() {
        for &f in w {
            wall_of[f as usize] = k as u32;
        }
    }
    // per wall, once asked: may it go (no face fixed, and it touches the
    // body)
    let mut may: Vec<Option<bool>> = vec![None; walls.len()];
    let mut give = vec![false; nf];
    for (p, kind) in pieces.iter().zip(kinds) {
        let cut_off = *kind == Piece::Candidate
            && !p
                .iter()
                .any(|&f| reachable[f as usize] || claimed[f as usize]);
        if !cut_off {
            continue;
        }
        let mut around: Vec<u32> = p
            .iter()
            .flat_map(|&f| topo.face_neighbors(f as usize))
            .map(|g| wall_of[g as usize])
            .filter(|&k| k != u32::MAX)
            .collect();
        around.sort_unstable();
        around.dedup();
        let in_piece: HashSet<u32> = p.iter().copied().collect();
        for k in around {
            let k = k as usize;
            let ok = *may[k].get_or_insert_with(|| {
                let w = &walls[k];
                !w.iter().any(|&f| fixed[f as usize])
                    && w.iter().any(|&f| {
                        topo.face_neighbors(f as usize)
                            .into_iter()
                            .any(|g| reachable[g as usize])
                    })
            });
            if ok {
                corridor(&topo, &walls[k], &in_piece, &reachable, &mut give);
            }
        }
    }
    (0..nf as u32)
        .filter(|&f| give[f as usize] && organelle[f as usize])
        .collect()
}

/// Marks in `give` the faces of `wall` on a shortest path of faces sharing
/// edges, inside the wall, from a face next to the piece (`in_piece`) to
/// one next to the body (`reachable`): where the two come closest across
/// it — round a fold's mouth rather than down into it.
fn corridor(
    topo: &Topology,
    wall: &[u32],
    in_piece: &HashSet<u32>,
    reachable: &[bool],
    give: &mut [bool],
) {
    let at: HashMap<u32, usize> = wall.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let hops = |next_to: &dyn Fn(u32) -> bool| -> Vec<usize> {
        let mut d = vec![usize::MAX; wall.len()];
        let mut queue = VecDeque::new();
        for (i, &f) in wall.iter().enumerate() {
            if topo.face_neighbors(f as usize).into_iter().any(next_to) {
                d[i] = 0;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            for g in topo.face_neighbors(wall[i] as usize) {
                if let Some(&j) = at.get(&g)
                    && d[j] == usize::MAX
                {
                    d[j] = d[i] + 1;
                    queue.push_back(j);
                }
            }
        }
        d
    };
    let from_piece = hops(&|g| in_piece.contains(&g));
    let from_body = hops(&|g| reachable[g as usize]);
    let length = |i: usize| from_piece[i].saturating_add(from_body[i]);
    let shortest = (0..wall.len()).map(length).min().unwrap_or(usize::MAX);
    if shortest == usize::MAX {
        return;
    }
    for (i, &f) in wall.iter().enumerate() {
        if length(i) == shortest {
            give[f as usize] = true;
        }
    }
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
    use std::collections::BTreeSet;

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

    /// [`severed_walls`] on the strip, quads 0-1 soma: the walls it gives
    /// back with organelle faces `organelle` and those of them `fixed`.
    fn walls_on(
        rolled: bool,
        organelle: std::ops::Range<usize>,
        claimed: Option<usize>,
        fixed: std::ops::Range<usize>,
    ) -> Vec<u32> {
        let faces = strip(7);
        let verts = corners(7, rolled);
        let (centers, normals) = centres_and_normals(&verts, &faces);
        let nf = faces.len();
        let flags =
            |r: &std::ops::Range<usize>| -> Vec<bool> { (0..nf).map(|f| r.contains(&f)).collect() };
        let claimed: Vec<bool> = (0..nf).map(|f| Some(f) == claimed).collect();
        let (soma_face, organelle, usable) = (flags(&(0..4)), flags(&organelle), vec![true; nf]);
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
        let (pieces, kinds) = classify(&m, true);
        severed_walls(
            &faces,
            verts.len(),
            &usable,
            &organelle,
            &claimed,
            &flags(&fixed),
            &pieces,
            &kinds,
        )
    }

    #[test]
    fn the_walls_severing_a_piece_of_arbor_are_given_back() {
        // quad 4 alone organelle severs quads 5-6, bigger and looking out:
        // arbor, joined back across it
        assert_eq!(walls_on(false, 8..10, None, 0..0), vec![8, 9]);
        // an organelle drawn by hand stays, and a claimed piece is the
        // user's to place
        assert!(walls_on(false, 8..10, None, 8..9).is_empty());
        assert!(walls_on(false, 8..10, Some(11), 0..0).is_empty());
        // rolled into a cavity the piece is an organelle's inside, and
        // behind quads 3-4 it is no bigger than its walls: trapped both ways
        assert!(walls_on(true, 8..10, None, 0..0).is_empty());
        assert!(walls_on(false, 6..10, None, 0..0).is_empty());
    }

    /// A flat grid of `cols` x `rows` unit quads, quad `(c, r)` faces
    /// `2 q` (on its left edge) and `2 q + 1` (on its right), `q = c rows + r`.
    fn grid(cols: u32, rows: u32) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let v = |c: u32, r: u32| c * (rows + 1) + r;
        let verts = (0..=cols)
            .flat_map(|c| (0..=rows).map(move |r| [c as f64, r as f64, 0.0]))
            .collect();
        let mut faces = Vec::new();
        for c in 0..cols {
            for r in 0..rows {
                let (a, b, d, e) = (v(c, r), v(c, r + 1), v(c + 1, r), v(c + 1, r + 1));
                faces.push([a, d, b]);
                faces.push([b, d, e]);
            }
        }
        (verts, faces)
    }

    /// Only the wall's corridor goes: a wall column (6) between the body
    /// (columns 0-5) and a piece (7-9), with a spur of wall into the body
    /// (column 5, rows 2-3).  The rows of the column where the body and the
    /// piece meet across it go; its other rows and the spur stay.
    #[test]
    fn only_the_corridor_across_a_wall_is_given_back() {
        let (cols, rows) = (10, 4);
        let (verts, faces) = grid(cols, rows);
        let (centers, normals) = centres_and_normals(&verts, &faces);
        let nf = faces.len();
        let quad = |f: usize| ((f / 2) as u32 / rows, (f / 2) as u32 % rows);
        let organelle: Vec<bool> = (0..nf)
            .map(|f| {
                let (c, r) = quad(f);
                c == 6 || (c == 5 && r >= 2)
            })
            .collect();
        let none = vec![false; nf];
        let usable = vec![true; nf];
        let m = BreakUpMesh {
            faces: &faces,
            n_vertices: verts.len(),
            usable: &usable,
            organelle: &organelle,
            soma_face: &none,
            claimed: &none,
            verts: &verts,
            centers: &centers,
            normals: &normals,
        };
        let (pieces, kinds) = classify(&m, false);
        let walls = severed_walls(
            &faces,
            verts.len(),
            &usable,
            &organelle,
            &none,
            &none,
            &pieces,
            &kinds,
        );
        let want: Vec<u32> = (0..nf as u32)
            .filter(|&f| {
                let (c, r) = quad(f as usize);
                c == 6 && r < 2
            })
            .collect();
        assert_eq!(walls, want);
        // the spur and the rest of the column stay organelle
        assert!(organelle.iter().filter(|&&o| o).count() > want.len());
    }

    /// Two walls with an organelle's inside between: the body (columns
    /// 0-3), a wall (4), a piece no bigger than its walls, trapped (5),
    /// a wall (6) and a piece of arbor (7-12).  Neither wall touches both
    /// the arbor and the body; with the trapped piece they are one wall
    /// that does, and its corridor joins the arbor back.
    #[test]
    fn a_wall_holds_the_organelle_inside_it() {
        let (cols, rows) = (13, 3);
        let (verts, faces) = grid(cols, rows);
        let (centers, normals) = centres_and_normals(&verts, &faces);
        let nf = faces.len();
        let column = |f: usize| (f / 2) as u32 / rows;
        let organelle: Vec<bool> = (0..nf).map(|f| [4, 6].contains(&column(f))).collect();
        let none = vec![false; nf];
        let usable = vec![true; nf];
        let m = BreakUpMesh {
            faces: &faces,
            n_vertices: verts.len(),
            usable: &usable,
            organelle: &organelle,
            soma_face: &none,
            claimed: &none,
            verts: &verts,
            centers: &centers,
            normals: &normals,
        };
        let (pieces, kinds) = classify(&m, false);
        let kind_of = |col: u32| {
            let i = pieces
                .iter()
                .position(|p| column(p[0] as usize) == col)
                .expect("a piece");
            kinds[i]
        };
        assert_eq!(kind_of(5), Piece::Trapped);
        assert_eq!(kind_of(7), Piece::Candidate);
        let walls = severed_walls(
            &faces,
            verts.len(),
            &usable,
            &organelle,
            &none,
            &none,
            &pieces,
            &kinds,
        );
        // organelle faces of both walls, none of the trapped piece's
        assert!(walls.iter().all(|&f| organelle[f as usize]));
        let cols: BTreeSet<u32> = walls.iter().map(|&f| column(f as usize)).collect();
        assert_eq!(cols, BTreeSet::from([4, 6]));
        // released, the arbor and the inside are the body's again
        let mut left = organelle.clone();
        for &f in &walls {
            left[f as usize] = false;
        }
        let m2 = BreakUpMesh {
            organelle: &left,
            ..m
        };
        let (pieces2, _) = classify(&m2, false);
        let arbor = pieces2
            .iter()
            .find(|p| p.iter().any(|&f| column(f as usize) == 12))
            .expect("the arbor");
        assert!(arbor.iter().any(|&f| column(f as usize) == 0));
    }
}
