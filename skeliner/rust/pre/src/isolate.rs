//! The organelles cut free of the rest of the mesh: every vertex an
//! organelle face shares with another face is given, for each fan of the
//! faces round it, a copy of its own.

use mesh::Topology;
use mesh::components::LowestRootSets;

use crate::fusions::{Split, copy_point_at, split_fan_vertices};

type V3 = [f64; 3];

/// Twice the area of face `f`.
fn doubled_area(at: &dyn Fn(u32) -> V3, f: [u32; 3]) -> f64 {
    let [p, q, r] = f.map(at);
    let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let w = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
    let x = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
}

/// The fans of `fan` (faces at `v`, ascending) on one side: those of
/// `side`, each set of them joined across edges at `v`, in the order of
/// their lowest faces.
fn fans_of(
    topo: &Topology,
    faces: &[[u32; 3]],
    v: u32,
    fan: &[u32],
    side: &dyn Fn(u32) -> bool,
) -> Vec<Vec<u32>> {
    let mine: Vec<u32> = fan.iter().copied().filter(|&f| side(f)).collect();
    let sets = LowestRootSets::new(mine.len());
    for (i, &f) in mine.iter().enumerate() {
        for w in faces[f as usize] {
            let Some(e) = topo.edge_id(v, w).filter(|_| w != v) else {
                continue;
            };
            for &g in topo.edge_faces(e) {
                if let Ok(j) = mine.binary_search(&g) {
                    sets.join(i as u32, j as u32);
                }
            }
        }
    }
    let root = sets.roots();
    let mut by_root: Vec<(u32, Vec<u32>)> = Vec::new();
    for (i, &r) in root.iter().enumerate() {
        match by_root.iter_mut().find(|(q, _)| *q == r) {
            Some((_, m)) => m.push(mine[i]),
            None => by_root.push((r, vec![mine[i]])),
        }
    }
    by_root.into_iter().map(|(_, m)| m).collect()
}

/// The mesh `verts`, `faces` with its organelles cut free: wherever an
/// `organelle` face (one flag per face, or fewer: the faces after them are
/// none) shares a vertex with a face that is none, the organelle faces
/// round it — each set of them joined across edges at the vertex — get a
/// copy of it, so no organelle shares a vertex, and with it an edge, with
/// anything else. A pocket's wall comes off the surface its mouth opened
/// in, and with the mouth patched, each mouth edge has two faces again
/// where it had three. Where the other faces at such a vertex then fall
/// into several fans — surface joined only across an organelle there —
/// each fan but the first gets a copy too, and a fan of faces with no area
/// is dropped (as [`split_fan_vertices`] treats a cut); a face so dropped
/// leaves the fans at its other vertices, which are split where they fall
/// apart, by [`split_fan_vertices`] (its copies numbered after these).
///
/// An organelle's copy goes where its vertex stood in `positions` (the
/// vertices below its length; the rest where they are) — before the
/// surface round a mouth was faired, which moved the wall's rim with it —
/// and every copy then a thousandth of its fan's reach into it
/// ([`copy_point_at`]), so it shares no point with the vertex it copies.
/// Faces naming a vertex twice belong to no fan and keep their vertices.
/// The copies are numbered on from the vertices, in the order of the
/// vertices they copy, the organelles' fans first, each set in the order
/// of its lowest faces.
pub fn isolate_organelles(
    verts: &[V3],
    faces: &[[u32; 3]],
    organelle: &[bool],
    positions: &[V3],
) -> Split {
    let nv = verts.len();
    let is_org = |f: u32| organelle.get(f as usize).copied().unwrap_or(false);
    let now = |w: u32| verts[w as usize];
    let before = |w: u32| {
        positions
            .get(w as usize)
            .copied()
            .unwrap_or(verts[w as usize])
    };
    let topo = Topology::new(faces, nv);
    if nv > 0 {
        let _ = topo.vertex_faces(0);
    }
    // per vertex shared by both: the organelles' fans, the others' fans
    type Fans = (u32, Vec<Vec<u32>>, Vec<Vec<u32>>);
    let shared: Vec<Fans> = threads::install(|| {
        use rayon::prelude::*;
        (0..nv as u32)
            .into_par_iter()
            .filter_map(|v| {
                let fan = topo.vertex_faces(v as usize);
                let n_org = fan.iter().filter(|&&f| is_org(f)).count();
                if n_org == 0 || n_org == fan.len() {
                    return None;
                }
                let org = fans_of(&topo, faces, v, fan, &|f| is_org(f));
                let rest = fans_of(&topo, faces, v, fan, &|f| !is_org(f));
                Some((v, org, rest))
            })
            .collect()
    });
    let mut out = faces.to_vec();
    let (mut points, mut sources) = (Vec::new(), Vec::new());
    let mut dropped = 0;
    // the other vertices of the faces dropped, as labelled then
    let mut parted: Vec<u32> = Vec::new();
    let relabel = |out: &mut Vec<[u32; 3]>, piece: &[u32], v: u32, id: u32| {
        for &f in piece {
            for slot in out[f as usize].iter_mut() {
                if *slot == v {
                    *slot = id;
                }
            }
        }
    };
    for (v, org, rest) in shared {
        for piece in &org {
            let id = (nv + sources.len()) as u32;
            points.push(copy_point_at(&before, faces, v, piece));
            sources.push(v);
            relabel(&mut out, piece, v, id);
        }
        let has_area: Vec<bool> = rest
            .iter()
            .map(|p| {
                p.iter()
                    .any(|&f| doubled_area(&now, faces[f as usize]) > 0.0)
            })
            .collect();
        let keeper = has_area.iter().position(|&a| a);
        for (i, piece) in rest.iter().enumerate() {
            if Some(i) == keeper {
                continue;
            }
            if !has_area[i] {
                for &f in piece {
                    parted.extend(out[f as usize].into_iter().filter(|&w| w != v));
                    out[f as usize] = [0, 0, 0];
                    dropped += 1;
                }
                continue;
            }
            let id = (nv + sources.len()) as u32;
            points.push(copy_point_at(&now, faces, v, piece));
            sources.push(v);
            relabel(&mut out, piece, v, id);
        }
    }
    parted.sort_unstable();
    parted.dedup();
    if !parted.is_empty() {
        let mut at = verts.to_vec();
        at.extend_from_slice(&points);
        let topo = Topology::new(&out, at.len());
        if let Some(s) = split_fan_vertices(&topo, &out, &at, Some(&parted)) {
            // a copy of a copy copies the first vertex
            let first: Vec<u32> = s
                .sources
                .iter()
                .map(|&w| {
                    if (w as usize) < nv {
                        w
                    } else {
                        sources[w as usize - nv]
                    }
                })
                .collect();
            out = s.faces;
            points.extend(s.points);
            sources.extend(first);
            dropped += s.dropped;
        }
    }
    Split {
        faces: out,
        points,
        sources,
        dropped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square sheet, 4 x 4 squares in z = 0 wound up (+z), with a pocket
    /// dropping from its middle square's hole: a square sac two squares
    /// deep, wound with the sheet, and the mouth patched by two faces as
    /// the fill would. Returns the vertices, faces and the pocket's faces.
    fn pocketed() -> (Vec<V3>, Vec<[u32; 3]>, Vec<u32>) {
        let id = |i: u32, j: u32| j * 5 + i;
        let mut v: Vec<V3> = (0..25)
            .map(|k| [f64::from(k % 5), f64::from(k / 5), 0.0])
            .collect();
        let mut f: Vec<[u32; 3]> = Vec::new();
        for j in 0..4 {
            for i in 0..4 {
                if (i, j) == (1, 1) || (i, j) == (2, 1) || (i, j) == (1, 2) || (i, j) == (2, 2) {
                    continue;
                }
                let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                f.push([a, b, c]);
                f.push([a, c, d]);
            }
        }
        // the mouth: the loop round the middle 2 x 2 squares
        let mouth = [
            id(1, 1),
            id(2, 1),
            id(3, 1),
            id(3, 2),
            id(3, 3),
            id(2, 3),
            id(1, 3),
            id(1, 2),
        ];
        // a sac below it: the mouth's ring at z = -1 and a bottom at z = -2
        let ring1: Vec<u32> = mouth
            .iter()
            .map(|&m| {
                let p = v[m as usize];
                v.push([p[0], p[1], -1.0]);
                (v.len() - 1) as u32
            })
            .collect();
        let bottom = v.len() as u32;
        v.push([2.0, 2.0, -2.0]);
        let mut pocket = Vec::new();
        for k in 0..8 {
            let (a, b) = (mouth[k], mouth[(k + 1) % 8]);
            let (c, d) = (ring1[k], ring1[(k + 1) % 8]);
            // the wall continues the sheet's winding into the sac
            pocket.push(f.len() as u32);
            f.push([a, d, b]);
            pocket.push(f.len() as u32);
            f.push([a, c, d]);
            pocket.push(f.len() as u32);
            f.push([c, bottom, d]);
        }
        // the patch over the mouth, wound with the sheet, its middle vertex
        // id(2, 2) lifted a little as the fairing would
        v[id(2, 2) as usize][2] = 0.1;
        for k in 0..8 {
            f.push([mouth[k], mouth[(k + 1) % 8], id(2, 2)]);
        }
        (v, f, pocket)
    }

    #[test]
    fn a_patched_pocket_comes_off_the_surface() {
        let (v, f, pocket) = pocketed();
        let before = mesh::validity::check(&v, &f);
        assert_eq!(before.crowded.len(), 8, "every mouth edge on three faces");
        let mut org = vec![false; f.len()];
        for &p in &pocket {
            org[p as usize] = true;
        }
        let s = isolate_organelles(&v, &f, &org, &v);
        assert_eq!(s.sources.len(), 8);
        let all: Vec<V3> = v.iter().chain(&s.points).copied().collect();
        let after = mesh::validity::check(&all, &s.faces);
        assert!(after.is_valid(), "{after:?}");
        // no vertex named by both an organelle face and another
        let mut on_org = vec![false; all.len()];
        let mut on_other = vec![false; all.len()];
        for (fi, t) in s.faces.iter().enumerate() {
            for &x in t {
                if org[fi] {
                    on_org[x as usize] = true;
                } else {
                    on_other[x as usize] = true;
                }
            }
        }
        assert!((0..all.len()).all(|x| !(on_org[x] && on_other[x])));
        // each copy a thousandth of its fan's reach off its vertex, into
        // the sac (below the sheet)
        for (&src, c) in s.sources.iter().zip(&s.points) {
            let p = v[src as usize];
            assert!(c[2] < p[2], "{c:?} from {p:?}");
            let d = ((c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2)).sqrt();
            assert!(d > 0.0 && d < 0.01, "{d}");
        }
    }

    #[test]
    fn surface_joined_only_across_an_organelle_is_split_too() {
        // four triangles round vertex 0, alternating surface and organelle:
        // without the organelle the surface is two fans touching at 0
        let v: Vec<V3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let f = vec![[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]];
        let org = [false, true, false, true];
        let s = isolate_organelles(&v, &f, &org, &v);
        // at vertex 0 the two organelle fans and the second surface fan
        // get copies; at each rim vertex, shared too, its organelle fan
        assert_eq!(s.sources, vec![0, 0, 0, 1, 2, 3, 4]);
        let all: Vec<V3> = v.iter().chain(&s.points).copied().collect();
        let c = mesh::validity::check(&all, &s.faces);
        assert!(c.is_valid(), "{c:?}");
        assert_eq!(s.faces[0], [0, 1, 2]);
    }

    #[test]
    fn a_flat_fan_dropped_at_one_vertex_splits_the_fan_it_held_at_another() {
        // at vertex 1, shared with the organelle face [1, 8, 9], two faces
        // of no area on the line y = z = 0 — a fan of their own beside
        // [1, 6, 7] — that alone join, at vertex 0, [0, 4, 2] and [0, 3, 5]
        let v: Vec<V3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.5, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, 2.0],
        ];
        let faces = vec![
            [0, 4, 2],
            [0, 3, 5],
            [0, 2, 1],
            [0, 1, 3],
            [1, 6, 7],
            [1, 8, 9],
        ];
        let org = [false, false, false, false, false, true];
        let s = isolate_organelles(&v, &faces, &org, &[]);
        assert_eq!(
            s.faces,
            vec![
                [0, 4, 2],
                [11, 3, 5],
                [0, 0, 0],
                [0, 0, 0],
                [1, 6, 7],
                [10, 8, 9]
            ]
        );
        assert_eq!(s.sources, vec![1, 0]);
        assert_eq!(s.dropped, 2);
        let mut all = v.clone();
        all.extend_from_slice(&s.points);
        assert!(mesh::validity::check(&all, &s.faces).split_fans.is_empty());
    }

    #[test]
    fn a_copy_goes_where_its_vertex_stood_before() {
        let (v, f, pocket) = pocketed();
        let mut org = vec![false; f.len()];
        for &p in &pocket {
            org[p as usize] = true;
        }
        // the mouth's first vertex had been moved 0.25 up by the fairing
        let mut moved = v.clone();
        moved[6][2] = 0.25;
        let s = isolate_organelles(&moved, &f, &org, &v);
        let k = s.sources.iter().position(|&x| x == 6).unwrap();
        assert!(
            s.points[k][2] < 0.0 && s.points[k][2] > -0.01,
            "{:?}",
            s.points[k]
        );
        // a mesh with no organelle, or all organelle, is left as it is
        let none = isolate_organelles(&v, &f, &[], &v);
        assert!(none.sources.is_empty() && none.faces == f);
        let all = isolate_organelles(&v, &f, &vec![true; f.len()], &v);
        assert!(all.sources.is_empty());
    }
}
