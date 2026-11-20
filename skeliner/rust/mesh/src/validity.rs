//! Whether a triangle mesh is a valid surface, rule by rule.
//!
//! A surface is its faces: each names three distinct vertices that span
//! some area, and no two name the same three. An edge lies on one face (the
//! surface's border) or on two that run it opposite ways, so the faces on
//! either side agree on which way is out. The faces round a vertex are one
//! fan, each joined to the next across an edge, and no two vertices sit at
//! one point. [`check`] lists every element that breaks a rule.
//!
//! Edges and fans are those of the faces of three distinct vertices, a face
//! of no area among them: it is listed as flat, and what it holds together
//! is judged as it is stored. A face naming one vertex three times,
//! `[v, v, v]` — the mark a removal leaves so that no index moves — is no
//! face, and a vertex no face names is no vertex: compaction drops both, so
//! they are counted apart and break no rule.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::Topology;
use crate::components::LowestRootSets;

/// The elements of a mesh that break a rule of a valid surface, each list
/// ascending.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Validity {
    /// Faces naming a vertex twice.
    pub repeated: Vec<u32>,
    /// Faces of three distinct vertices that span no area.
    pub flat: Vec<u32>,
    /// Faces naming the three vertices of a lower one, in any order.
    pub duplicate: Vec<u32>,
    /// Edges, `[low, high]`, on three faces or more.
    pub crowded: Vec<[u32; 2]>,
    /// Edges on two faces that run them the same way.
    pub misoriented: Vec<[u32; 2]>,
    /// Vertices whose faces fall into two fans or more.
    pub split_fans: Vec<u32>,
    /// Vertices at the point of a lower vertex, both named by a face.
    pub shared_points: Vec<u32>,
    /// Faces `[v, v, v]`: removals' marks.
    pub collapsed: usize,
    /// Vertices no face names.
    pub unnamed: usize,
}

impl Validity {
    /// Whether no element breaks a rule.
    pub fn is_valid(&self) -> bool {
        self.repeated.is_empty()
            && self.flat.is_empty()
            && self.duplicate.is_empty()
            && self.crowded.is_empty()
            && self.misoriented.is_empty()
            && self.split_fans.is_empty()
            && self.shared_points.is_empty()
    }
}

fn collapsed(f: &[u32; 3]) -> bool {
    f[0] == f[1] && f[1] == f[2]
}

fn repeats(f: &[u32; 3]) -> bool {
    f[0] == f[1] || f[1] == f[2] || f[0] == f[2]
}

/// Twice the area of the triangle `a`, `b`, `c`.
fn doubled_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let x = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
}

/// Whether face `f` runs from `a` to `b`.
fn runs(f: [u32; 3], a: u32, b: u32) -> bool {
    (f[0] == a && f[1] == b) || (f[1] == a && f[2] == b) || (f[2] == a && f[0] == b)
}

/// A position's bits, −0.0 read as 0.0; None for one with a NaN, which
/// equals nothing.
fn point(v: &[f64; 3]) -> Option<[u64; 3]> {
    (!v.iter().any(|x| x.is_nan())).then(|| v.map(|x| (x + 0.0).to_bits()))
}

/// Every element of the mesh `vertices`, `faces` that breaks a rule of a
/// valid surface ([`Validity`]). Every index in `faces` must name one of
/// `vertices`.
pub fn check(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Validity {
    threads::install(|| validity_of(vertices, faces))
}

fn validity_of(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Validity {
    let nv = vertices.len();
    let n_collapsed = faces.par_iter().filter(|f| collapsed(f)).count();
    let repeated: Vec<u32> = (0..faces.len() as u32)
        .into_par_iter()
        .filter(|&f| {
            let f = &faces[f as usize];
            repeats(f) && !collapsed(f)
        })
        .collect();
    let flat: Vec<u32> = (0..faces.len() as u32)
        .into_par_iter()
        .filter(|&fi| {
            let f = faces[fi as usize];
            !repeats(&f)
                && doubled_area(
                    vertices[f[0] as usize],
                    vertices[f[1] as usize],
                    vertices[f[2] as usize],
                ) == 0.0
        })
        .collect();

    // faces with the vertices of a lower one
    let mut keyed: Vec<([u32; 3], u32)> = faces
        .par_iter()
        .enumerate()
        .filter(|(_, f)| !collapsed(f))
        .map(|(fi, f)| {
            let mut k = *f;
            k.sort_unstable();
            (k, fi as u32)
        })
        .collect();
    keyed.par_sort_unstable();
    let mut duplicate: Vec<u32> = keyed
        .chunk_by(|x, y| x.0 == y.0)
        .flat_map(|run| run[1..].iter().map(|&(_, fi)| fi))
        .collect();
    duplicate.par_sort_unstable();

    // the faces of three distinct vertices make the edges and the fans
    let topo = Topology::new(faces, nv);
    // per edge: 0 fine, 1 on three faces or more, 2 two faces one way
    let kind: Vec<u8> = (0..topo.n_edges())
        .into_par_iter()
        .map(|e| {
            let on = topo.edge_faces(e);
            let [a, b] = topo.edges()[e];
            match on.len() {
                0..=1 => 0,
                2 => {
                    let (f, g) = (faces[on[0] as usize], faces[on[1] as usize]);
                    if runs(f, a, b) == runs(g, a, b) { 2 } else { 0 }
                }
                _ => 1,
            }
        })
        .collect();
    let of_kind = |k: u8| -> Vec<[u32; 2]> {
        (0..topo.n_edges())
            .filter(|&e| kind[e] == k)
            .map(|e| topo.edges()[e])
            .collect()
    };
    let crowded = of_kind(1);
    let misoriented = of_kind(2);

    let mut named = vec![false; nv];
    for f in faces.iter().filter(|f| !collapsed(f)) {
        for &v in f {
            named[v as usize] = true;
        }
    }
    let unnamed = named.iter().filter(|&&n| !n).count();
    let split_fans: Vec<u32> = (0..nv as u32)
        .into_par_iter()
        .filter(|&v| fans_at(&topo, faces, v) > 1)
        .collect();

    // vertices at the point of a lower one, among those faces name
    let mut first: HashMap<[u64; 3], u32> = HashMap::new();
    let mut shared_points = Vec::new();
    for (v, p) in vertices.iter().enumerate() {
        if !named[v] {
            continue;
        }
        let Some(key) = point(p) else { continue };
        if first.entry(key).or_insert(v as u32) != &(v as u32) {
            shared_points.push(v as u32);
        }
    }

    Validity {
        repeated,
        flat,
        duplicate,
        crowded,
        misoriented,
        split_fans,
        shared_points,
        collapsed: n_collapsed,
        unnamed,
    }
}

/// How many fans the faces at `v` make: sets of them joined, face to face,
/// across edges at `v`.
fn fans_at(topo: &Topology, faces: &[[u32; 3]], v: u32) -> usize {
    let fan = topo.vertex_faces(v as usize);
    if fan.len() < 2 {
        return fan.len();
    }
    let sets = LowestRootSets::new(fan.len());
    for (i, &f) in fan.iter().enumerate() {
        for w in faces[f as usize] {
            if w == v {
                continue;
            }
            let Some(e) = topo.edge_id(v, w) else {
                continue;
            };
            for &g in topo.edge_faces(e) {
                if let Ok(j) = fan.binary_search(&g) {
                    sets.join(i as u32, j as u32);
                }
            }
        }
    }
    let mut roots = sets.roots();
    roots.sort_unstable();
    roots.dedup();
    roots.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square of two faces, wound the same way, in z = 0.
    fn square() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        (v, vec![[0, 1, 2], [0, 2, 3]])
    }

    #[test]
    fn a_wound_square_is_valid() {
        let (v, f) = square();
        let c = check(&v, &f);
        assert!(c.is_valid(), "{c:?}");
        assert_eq!((c.collapsed, c.unnamed), (0, 0));
    }

    #[test]
    fn removal_marks_and_lone_vertices_break_no_rule() {
        let (mut v, mut f) = square();
        v.push([5.0, 5.0, 5.0]);
        f.push([0, 0, 0]);
        let c = check(&v, &f);
        assert!(c.is_valid(), "{c:?}");
        assert_eq!((c.collapsed, c.unnamed), (1, 1));
    }

    #[test]
    fn each_broken_rule_is_listed() {
        let (mut v, mut f) = square();
        // a face naming a vertex twice; one spanning no area (vertex 4 on
        // the edge 0-1); the first face again, wound the other way
        v.push([0.5, 0.0, 0.0]);
        f.extend([[1, 1, 2], [0, 4, 1], [0, 2, 1]]);
        let c = check(&v, &f);
        assert_eq!(c.repeated, [2]);
        assert_eq!(c.flat, [3]);
        assert_eq!(c.duplicate, [4]);
        // edges 0-1 and 0-2 have three faces now (the flat face 3 on
        // 0-1 among them); 1-2 has two, face 4 running it 2-1 against
        // face 0's 1-2
        assert_eq!(c.crowded, [[0, 1], [0, 2]]);
        assert!(c.misoriented.is_empty() && c.split_fans.is_empty());
    }

    #[test]
    fn a_flat_face_holding_a_t_junction_together_is_listed_alone() {
        // vertex 4 on the square's edge 0-1, closed by a face of no area
        // against a fan of two faces below it: without that face vertices
        // 0 and 1 would each be two fans, with it every edge has two faces
        let (mut v, mut f) = square();
        v.extend([[0.5, 0.0, 0.0], [0.5, -1.0, 0.0]]);
        f.extend([[0, 4, 1], [0, 5, 4], [4, 5, 1]]);
        let c = check(&v, &f);
        assert_eq!(c.flat, [2]);
        assert!(c.crowded.is_empty() && c.misoriented.is_empty());
        assert!(c.split_fans.is_empty() && c.duplicate.is_empty());
    }

    #[test]
    fn faces_running_an_edge_the_same_way_are_misoriented() {
        let (v, mut f) = square();
        f[1] = [0, 3, 2];
        let c = check(&v, &f);
        assert_eq!(c.misoriented, [[0, 2]]);
        assert!(c.crowded.is_empty() && c.split_fans.is_empty());
    }

    #[test]
    fn a_bowtie_vertex_splits_its_fan() {
        // two triangles meeting only at vertex 2
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ];
        let f = vec![[0, 1, 2], [2, 3, 4]];
        let c = check(&v, &f);
        assert_eq!(c.split_fans, [2]);
        assert!(c.crowded.is_empty() && c.misoriented.is_empty());
    }

    #[test]
    fn two_vertices_at_one_point_are_shared() {
        // the bow-tie cut apart at vertex 2, its copy 5 at the same point;
        // −0.0 is 0.0, and a vertex no face names is none
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 1.0, -0.0],
            [1.0, 0.0, 0.0],
        ];
        let f = vec![[0, 1, 2], [5, 3, 4]];
        let c = check(&v, &f);
        assert_eq!(c.shared_points, [5]);
        assert!(c.split_fans.is_empty());
        assert_eq!(c.unnamed, 1);
    }
}
