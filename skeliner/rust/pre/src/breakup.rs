//! `skeliner.pre.break_up_mesh`: the pieces left once the soma and the
//! organelles are cut out, and what each one is.

use std::collections::{HashMap, HashSet, VecDeque};

use mesh::Topology;
use mesh::components::{LowestRootSets, listed_face_components};
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
///
/// Also, per face, whether it lies on its mesh component's body — what a
/// piece is cut off from, which [`severed_walls`] reads on the same faces.
pub fn classify(m: &BreakUpMesh<'_>, has_soma: bool) -> (Vec<Vec<u32>>, Vec<Piece>, Vec<bool>) {
    let nf = m.faces.len();
    let pick = |keep: &dyn Fn(usize) -> bool| -> Vec<u32> {
        (0..nf).filter(|&f| keep(f)).map(|f| f as u32).collect()
    };
    let reachable = bodies(m.faces, m.usable, m.organelle);

    let remain = pick(&|f| m.usable[f] && !m.soma_face[f] && !m.organelle[f]);
    let components = listed_face_components(m.faces, &remain);
    if components.is_empty() {
        return (components, Vec::new(), reachable);
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
    (components, kinds, reachable)
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
/// `claimed` piece is the user's to place.  `pieces`, `kinds` and `body`
/// are [`classify`]'s on the same faces and organelles.  Returns the
/// organelle faces of the corridors, ascending.
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
    body: &[bool],
) -> Vec<u32> {
    let nf = faces.len();
    // the pieces cut off from their body and not trapped; with none there
    // is no wall to look at
    let cut: Vec<&Vec<u32>> = pieces
        .iter()
        .zip(kinds)
        .filter(|(p, kind)| {
            **kind == Piece::Candidate
                && !p.iter().any(|&f| body[f as usize] || claimed[f as usize])
        })
        .map(|(p, _)| p)
        .collect();
    if cut.is_empty() {
        return Vec::new();
    }
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
    for p in cut {
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
                            .any(|g| body[g as usize])
                    })
            });
            if ok {
                corridor(&topo, &walls[k], &in_piece, body, &mut give);
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
/// Which pieces hold an organelle. `label` names the piece each face is in
/// (`u32::MAX` for none: a soma face, an organelle's), and an organelle is a
/// set of `organelle` faces that span an area, joined at their vertices. A
/// piece holds an organelle it surrounds: of the six rays from the
/// organelle's centre along the axes, most meet that piece first, among the
/// faces of no organelle that span an area, and the organelle lies within
/// the piece's bounds (the box of its vertices) on every axis. A flat piece
/// — a sliver on an organelle's wall, a fragment beside it — meets one ray
/// of each opposite pair at most, half of them; a piece closed round the
/// organelle meets all six, the surface round a pocket all but those
/// leaving through its mouth. Where most meet the soma's faces, or none
/// does, no piece holds it. Returns a flag per piece, `n_pieces` of them.
pub fn organelle_hosts(
    verts: &[[f64; 3]],
    faces: &[[u32; 3]],
    label: &[u32],
    organelle: &[bool],
    n_pieces: usize,
) -> Vec<bool> {
    let mut held = vec![false; n_pieces];
    let in_organelle = |f: usize| organelle.get(f).copied().unwrap_or(false);
    if n_pieces == 0 || !(0..faces.len()).any(in_organelle) {
        return held;
    }
    let unit_normal = |f: &[u32; 3]| -> Option<[f64; 3]> {
        let t = f.map(|v| verts[v as usize]);
        let u = [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]];
        let w = [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]];
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        (len > 0.0).then(|| n.map(|x| x / len))
    };
    // the organelles: their faces' vertices joined
    let sets = LowestRootSets::new(verts.len());
    let mut on = vec![false; verts.len()];
    for (fi, f) in faces.iter().enumerate() {
        if in_organelle(fi) && unit_normal(f).is_some() {
            sets.join(f[0], f[1]);
            sets.join(f[0], f[2]);
            for &v in f {
                on[v as usize] = true;
            }
        }
    }
    let mut members: HashMap<u32, Vec<u32>> = HashMap::new();
    for v in (0..verts.len()).filter(|&v| on[v]) {
        members
            .entry(sets.root(v as u32))
            .or_default()
            .push(v as u32);
    }
    // each piece's bounds, and the faces of no organelle that span an area
    let mut lo = vec![[f64::INFINITY; 3]; n_pieces];
    let mut hi = vec![[f64::NEG_INFINITY; 3]; n_pieces];
    let mut tris: Vec<[[f64; 3]; 3]> = Vec::new();
    let mut normals: Vec<[f64; 3]> = Vec::new();
    let mut whose: Vec<u32> = Vec::new();
    for (fi, f) in faces.iter().enumerate() {
        if in_organelle(fi) {
            continue;
        }
        let l = label
            .get(fi)
            .copied()
            .filter(|&l| (l as usize) < n_pieces)
            .unwrap_or(u32::MAX);
        let t = f.map(|v| verts[v as usize]);
        if l != u32::MAX {
            for p in t {
                for k in 0..3 {
                    lo[l as usize][k] = lo[l as usize][k].min(p[k]);
                    hi[l as usize][k] = hi[l as usize][k].max(p[k]);
                }
            }
        }
        if let Some(n) = unit_normal(f) {
            tris.push(t);
            normals.push(n);
            whose.push(l);
        }
    }
    if tris.is_empty() {
        return held;
    }
    let bvh = mesh::ray::TriangleBvh::new_parallel(tris, normals);
    const AXES: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for vs in members.values() {
        let mut c = [0.0f64; 3];
        for &v in vs {
            for k in 0..3 {
                c[k] += verts[v as usize][k];
            }
        }
        let c = c.map(|x| x / vs.len() as f64);
        let mut votes: HashMap<u32, usize> = HashMap::new();
        for d in AXES {
            if let Some((ti, _)) = bvh.first_hit(c, d, 0.0, None) {
                *votes.entry(whose[ti]).or_default() += 1;
            }
        }
        let Some((&l, _)) = votes.iter().find(|&(_, &k)| 2 * k > AXES.len()) else {
            continue;
        };
        if l == u32::MAX {
            continue;
        }
        let (lo, hi) = (lo[l as usize], hi[l as usize]);
        if vs.iter().all(|&v| {
            let p = verts[v as usize];
            (0..3).all(|k| lo[k] <= p[k] && p[k] <= hi[k])
        }) {
            held[l as usize] = true;
        }
    }
    held
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// A cube of half-width 1 (faces 0–11) round a small closed organelle
    /// (faces 12–15) near its centre.
    fn cube_round_organelle() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let v = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.2, 0.03, 0.01],
            [-0.1, 0.23, 0.02],
            [-0.1, -0.17, 0.03],
            [0.01, 0.02, 0.24],
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
            [8, 9, 10],
            [8, 9, 11],
            [9, 10, 11],
            [10, 8, 11],
        ];
        (v, f)
    }

    #[test]
    fn a_closed_organelle_is_held_by_the_piece_round_it() {
        let (v, f) = cube_round_organelle();
        let organelle: Vec<bool> = (0..16).map(|i| i >= 12).collect();
        let mut label = vec![u32::MAX; 16];
        label[..12].fill(1);
        assert_eq!(
            organelle_hosts(&v, &f, &label, &organelle, 2),
            vec![false, true]
        );
        // round it the soma's faces: no piece holds it
        label[..12].fill(u32::MAX);
        assert_eq!(
            organelle_hosts(&v, &f, &label, &organelle, 2),
            vec![false, false]
        );
    }

    /// The cube of [`cube_round_organelle`] (vertices 1–8, faces 0–11) with
    /// a pocket folding in from one of its edges (faces 12 and 14, its floor
    /// vertices 0 and 11) and a face touching the floor at 0 (face 13, out
    /// to vertices 9 and 10 at `reach`).
    fn pocket_in_cube(reach: [[f64; 3]; 2]) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let (cube_v, cube_f) = cube_round_organelle();
        let mut v = vec![[-0.5, 0.0, 0.5]];
        v.extend_from_slice(&cube_v[..8]);
        v.extend_from_slice(&reach);
        v.push([-0.6, 0.4, 0.6]);
        let mut f: Vec<[u32; 3]> = cube_f[..12].iter().map(|t| t.map(|i| i + 1)).collect();
        f.extend([[8, 0, 5], [0, 9, 10], [8, 11, 0]]);
        (v, f)
    }

    #[test]
    fn a_pocket_is_held_by_the_piece_round_its_mouth() {
        // piece 0 round the pocket, a sliver of piece 1 on its floor
        let (v, f) = pocket_in_cube([[-0.4, 0.1, 0.45], [-0.45, -0.1, 0.4]]);
        let mut label = vec![0; 15];
        label[12] = u32::MAX;
        label[13] = 1;
        label[14] = u32::MAX;
        let organelle: Vec<bool> = (0..15).map(|i| i == 12 || i == 14).collect();
        assert_eq!(
            organelle_hosts(&v, &f, &label, &organelle, 2),
            vec![true, false]
        );
        // the soma's faces round it, and a piece on its floor spanning it:
        // the soma's
        let (v, f) = pocket_in_cube([[-1.2, -1.2, 1.2], [-1.2, 1.2, 0.4]]);
        let mut label = vec![u32::MAX; 15];
        label[13] = 0;
        assert_eq!(organelle_hosts(&v, &f, &label, &organelle, 1), vec![false]);
    }

    #[test]
    fn organelle_faces_of_no_area_are_no_organelle() {
        // in the cube, faces of no area marked organelle at one point and
        // along one line: nothing to hold
        let (v, mut f) = cube_round_organelle();
        f.truncate(12);
        f.extend([[8, 8, 8], [8, 9, 9]]);
        let organelle: Vec<bool> = (0..14).map(|i| i >= 12).collect();
        let mut label = vec![0; 14];
        label[12..].fill(u32::MAX);
        assert_eq!(organelle_hosts(&v, &f, &label, &organelle, 1), vec![false]);
        // fixture: one face there with an area is held
        f.push([8, 9, 10]);
        label.push(u32::MAX);
        let organelle: Vec<bool> = (0..15).map(|i| i >= 12).collect();
        assert_eq!(organelle_hosts(&v, &f, &label, &organelle, 1), vec![true]);
    }

    #[test]
    fn a_fragment_beside_an_organelle_does_not_hold_it() {
        let (mut v, mut f) = cube_round_organelle();
        let mut organelle: Vec<bool> = (0..16).map(|i| i >= 12).collect();
        organelle.push(false);
        let mut label = vec![u32::MAX; 17];
        label[16] = 0;
        // touching it: the only piece it shares a vertex with, the soma
        // round both
        let mut touching = v.clone();
        touching.extend([[0.3, 0.05, 0.0], [0.25, -0.05, 0.05]]);
        let mut with_sliver = f.clone();
        with_sliver.push([8, 12, 13]);
        assert_eq!(
            organelle_hosts(&touching, &with_sliver, &label, &organelle, 1),
            vec![false]
        );
        // a sheet across it, spanning it, meeting three of the rays from
        // its centre: with the soma round both, which meets the other three
        let c = [0.0025, 0.0275, 0.075];
        for d in [[0.9, -0.2, -0.2], [-0.2, 0.9, -0.2], [-0.2, -0.2, 0.9]] {
            v.push([c[0] + d[0], c[1] + d[1], c[2] + d[2]]);
        }
        f.push([12, 13, 14]);
        assert_eq!(organelle_hosts(&v, &f, &label, &organelle, 1), vec![false]);
        // and with nothing round them, the other three meeting nothing
        let alone: Vec<[u32; 3]> = f[12..].to_vec();
        assert_eq!(
            organelle_hosts(&v, &alone, &label[12..], &organelle[12..], 1),
            vec![false]
        );
        // the sheet round it as the soma is: one piece
        label[..12].fill(0);
        assert_eq!(organelle_hosts(&v, &f, &label, &organelle, 1), vec![true]);
    }

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
        let (pieces, kinds, _) = classify(&m, has_soma);
        (pieces, kinds)
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
        let (pieces, kinds, body) = classify(&m, true);
        severed_walls(
            &faces,
            verts.len(),
            &usable,
            &organelle,
            &claimed,
            &flags(&fixed),
            &pieces,
            &kinds,
            &body,
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
        let (pieces, kinds, body) = classify(&m, false);
        let walls = severed_walls(
            &faces,
            verts.len(),
            &usable,
            &organelle,
            &none,
            &none,
            &pieces,
            &kinds,
            &body,
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
    /// 0-7), a wall (8), a piece no bigger than its walls, trapped (9),
    /// a wall (10) and a piece of arbor, smaller than the body (11-16).
    /// Neither wall touches both the arbor and the body; with the trapped
    /// piece they are one wall that does, and its corridor joins the arbor
    /// back.
    #[test]
    fn a_wall_holds_the_organelle_inside_it() {
        let (cols, rows) = (17, 3);
        let (verts, faces) = grid(cols, rows);
        let (centers, normals) = centres_and_normals(&verts, &faces);
        let nf = faces.len();
        let column = |f: usize| (f / 2) as u32 / rows;
        let organelle: Vec<bool> = (0..nf).map(|f| [8, 10].contains(&column(f))).collect();
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
        let (pieces, kinds, body) = classify(&m, false);
        let kind_of = |col: u32| {
            let i = pieces
                .iter()
                .position(|p| column(p[0] as usize) == col)
                .expect("a piece");
            kinds[i]
        };
        assert_eq!(kind_of(9), Piece::Trapped);
        assert_eq!(kind_of(11), Piece::Candidate);
        // the body is the columns before the first wall: neither the
        // inside nor the arbor, nor the walls
        assert!((0..nf).all(|f| body[f] == (column(f) < 8)));
        let walls = severed_walls(
            &faces,
            verts.len(),
            &usable,
            &organelle,
            &none,
            &none,
            &pieces,
            &kinds,
            &body,
        );
        // organelle faces of both walls, none of the trapped piece's
        assert!(walls.iter().all(|&f| organelle[f as usize]));
        let cols: BTreeSet<u32> = walls.iter().map(|&f| column(f as usize)).collect();
        assert_eq!(cols, BTreeSet::from([8, 10]));
        // released, the arbor and the inside are the body's again
        let mut left = organelle.clone();
        for &f in &walls {
            left[f as usize] = false;
        }
        let m2 = BreakUpMesh {
            organelle: &left,
            ..m
        };
        let (pieces2, _, _) = classify(&m2, false);
        let arbor = pieces2
            .iter()
            .find(|p| p.iter().any(|&f| column(f as usize) == 16))
            .expect("the arbor");
        assert!(arbor.iter().any(|&f| column(f as usize) == 0));
    }

    /// With no piece cut off there is no wall to give back: an organelle
    /// quad at the strip's far end (6) cuts nothing off, and with no
    /// organelle at all every piece lies on the body.
    #[test]
    fn nothing_cut_off_gives_nothing_back() {
        assert!(walls_on(false, 12..14, None, 0..0).is_empty());
        assert!(walls_on(false, 0..0, None, 0..0).is_empty());
    }
}
