//! `skeliner.post`'s kernels: `calibrate_radii` (its ray cast and the faces
//! of a vertex set on their own too), node types through edits (`ntype`),
//! and `downsample`.

#![forbid(unsafe_code)]

pub mod downsample;
pub mod ntype;

use std::collections::HashMap;

use mesh::ray::TriangleBvh;
use rayon::prelude::*;
use skeletonize::skeleton::estimate_radius;

/// Which rays hit any of the triangles, with trimesh's `ray_triangle` test:
/// per ray from `origins[k]` along `directions[k]`; `normals` are the
/// triangles' face normals.
pub fn ray_hits(
    triangles: Vec<[[f64; 3]; 3]>,
    normals: Vec<[f64; 3]>,
    origins: &[[f64; 3]],
    directions: &[[f64; 3]],
) -> Vec<bool> {
    let bvh = TriangleBvh::new(triangles, normals);
    threads::install(|| {
        origins
            .par_iter()
            .zip(directions.par_iter())
            .map(|(&o, &d)| bvh.any_hit(o, d))
            .collect()
    })
}

/// Faces by vertex, for cutting a mesh down to the faces of a vertex set.
pub struct VertexFaces {
    start: Vec<usize>,
    faces_of: Vec<u32>,
    faces: Vec<[u32; 3]>,
}

impl VertexFaces {
    /// Index the `faces` of a mesh with `n_vertices` vertices (every vertex
    /// id below it).
    pub fn new(faces: Vec<[u32; 3]>, n_vertices: usize) -> Self {
        let n = n_vertices;
        let mut deg = vec![0usize; n + 1];
        for f in &faces {
            for &v in f {
                deg[v as usize + 1] += 1;
            }
        }
        for v in 0..n {
            deg[v + 1] += deg[v];
        }
        let mut fill = deg.clone();
        let mut faces_of = vec![0u32; deg[n]];
        for (fi, f) in faces.iter().enumerate() {
            for &v in f {
                faces_of[fill[v as usize]] = fi as u32;
                fill[v as usize] += 1;
            }
        }
        Self {
            start: deg,
            faces_of,
            faces,
        }
    }

    /// The number of vertices indexed.
    pub fn n_vertices(&self) -> usize {
        self.start.len() - 1
    }

    /// Faces with all three vertices among `ids` (each below
    /// [`Self::n_vertices`]), ascending — the faces
    /// `skeliner.post.submesh_by_vertices` keeps.
    pub fn faces_within(&self, ids: &[u32]) -> Vec<u32> {
        let inside: std::collections::HashSet<u32> = ids.iter().copied().collect();
        let mut out: Vec<u32> = ids
            .iter()
            .flat_map(|&v| {
                self.faces_of[self.start[v as usize]..self.start[v as usize + 1]]
                    .iter()
                    .copied()
            })
            .filter(|&f| self.faces[f as usize].iter().all(|v| inside.contains(v)))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// What [`calibrate_radii`] made of a node's radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// The base radius, kept.
    Fallback = 0,
    /// The aggregate over all the node's vertices, smaller than the base.
    FullCenterline = 1,
    /// The aggregate over the vertices on the outer surface.
    OuterCenterline = 2,
}

/// The mesh [`calibrate_radii`] measures: trimesh's face centres and normals,
/// the same per face as a submesh's, and its faces by vertex.
pub struct CalibrateMesh<'a> {
    pub vertices: &'a [[f64; 3]],
    pub faces: &'a [[u32; 3]],
    pub centers: &'a [[f64; 3]],
    pub normals: &'a [[f64; 3]],
    pub vertex_faces: &'a VertexFaces,
}

/// [`calibrate_radii`]'s settings, `skeliner.post.calibrate_radii`'s
/// resolved: the rays to cast, the vertex count from which a node is tested
/// for internal membranes (with `check_outer`), what an outer surface needs,
/// how many hits make a face inner, and the estimator.
pub struct CalibrateParams<'a> {
    pub directions: &'a [[f64; 3]],
    pub min_n_verts_bulb: i64,
    pub check_outer: bool,
    pub min_n_outer: i64,
    pub min_frac_outer: f64,
    pub rays_threshold: i64,
    pub aggregate: &'a str,
}

/// The vertices (positions in `vids`, ascending) of the faces `sub` fewer
/// than `rays_threshold` rays hit: from each face's centre, nudged along its
/// normal, every direction pointing out of the face is cast against all the
/// faces of `sub`; with no such ray at all, every face is outer.
fn outer_vertices(
    m: &CalibrateMesh<'_>,
    sub: &[u32],
    vids: &[u32],
    p: &CalibrateParams<'_>,
) -> Vec<usize> {
    // a vertex's id in the submesh: its last position in `vids`
    let local: HashMap<u32, usize> = vids.iter().enumerate().map(|(k, &v)| (v, k)).collect();
    let local_face = |f: u32| m.faces[f as usize].map(|v| local[&v]);
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut origins = Vec::new();
    let mut rays = Vec::new();
    let mut owner = Vec::new();
    for (k, &f) in sub.iter().enumerate() {
        let (c, n) = (m.centers[f as usize], m.normals[f as usize]);
        let o = [
            c[0] + 0.001 * n[0],
            c[1] + 0.001 * n[1],
            c[2] + 0.001 * n[2],
        ];
        for &d in p.directions {
            if dot(d, n) > 0.0 {
                origins.push(o);
                rays.push(d);
                owner.push(k);
            }
        }
    }
    let mut outer: Vec<usize> = if origins.is_empty() {
        sub.iter().flat_map(|&f| local_face(f)).collect()
    } else {
        let tris = sub
            .iter()
            .map(|&f| m.faces[f as usize].map(|v| m.vertices[v as usize]))
            .collect();
        let normals = sub.iter().map(|&f| m.normals[f as usize]).collect();
        let bvh = TriangleBvh::new(tris, normals);
        // a big node's rays at once, so it does not hold the others up
        let hit = |(&o, &d): (&[f64; 3], &[f64; 3])| bvh.any_hit(o, d);
        let hit: Vec<bool> = if origins.len() >= 1 << 14 {
            threads::install(|| origins.par_iter().zip(&rays).map(hit).collect())
        } else {
            origins.iter().zip(&rays).map(hit).collect()
        };
        let mut hits = vec![0i64; sub.len()];
        for (&h, &k) in hit.iter().zip(&owner) {
            hits[k] += i64::from(h);
        }
        sub.iter()
            .zip(&hits)
            .filter(|&(_, &h)| h < p.rays_threshold)
            .flat_map(|(&f, _)| local_face(f))
            .collect()
    };
    outer.sort_unstable();
    outer.dedup();
    outer
}

/// `skeliner.post.calibrate_radii` on arrays, nodes in parallel: each node's
/// radius from the distances of its vertices to its centreline (the node
/// and its edges), and how it was found.
///
/// With `check_outer`, a node of `min_n_verts_bulb` vertices or more is cut
/// out of the mesh (the faces with all three vertices among its own) and
/// ray cast to tell its outer surface from internal membranes; when at
/// least `min_n_outer`, and `min_frac_outer` of them all, but not all of its
/// vertices are outer, its radius is the aggregate over them. Otherwise the
/// aggregate over all its vertices replaces the base radius if smaller.
/// Nodes in `skip` or without vertices keep the base. `None` when the
/// estimator is unknown and a node needed it.
pub fn calibrate_radii(
    m: &CalibrateMesh<'_>,
    nodes: &[[f64; 3]],
    edges: &[[u32; 2]],
    node2verts: &[Vec<u32>],
    skip: &[bool],
    base: &[f64],
    p: &CalibrateParams<'_>,
) -> Option<(Vec<f64>, Vec<Kind>)> {
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for &[a, b] in edges {
        neighbours[a as usize].push(b as usize);
        neighbours[b as usize].push(a as usize);
    }
    let one = |i: usize| -> Option<(f64, Kind)> {
        let vids = &node2verts[i];
        if skip[i] || vids.is_empty() {
            return Some((base[i], Kind::Fallback));
        }
        let mut segments: Vec<(usize, usize)> = neighbours[i]
            .iter()
            .map(|&nb| (i.min(nb), i.max(nb)))
            .collect();
        segments.sort_unstable();
        segments.dedup();
        let radius = |ids: &mut dyn Iterator<Item = u32>| -> Option<f64> {
            let pts: Vec<[f64; 3]> = ids.map(|v| m.vertices[v as usize]).collect();
            let d = dx::segment_distances(&pts, nodes, &[i], &segments);
            estimate_radius(&d, p.aggregate)
        };
        let n_inner = vids.len();
        if p.check_outer && n_inner as i64 >= p.min_n_verts_bulb {
            let sub = m.vertex_faces.faces_within(vids);
            let outer = outer_vertices(m, &sub, vids, p);
            let n_outer = outer.len();
            if n_outer as i64 >= p.min_n_outer
                && n_outer as f64 >= p.min_frac_outer * n_inner as f64
                && n_outer < n_inner
            {
                let r = radius(&mut outer.iter().map(|&k| vids[k]))?;
                return Some((r, Kind::OuterCenterline));
            }
        }
        let r = radius(&mut vids.iter().copied())?;
        Some(if r < base[i] {
            (r, Kind::FullCenterline)
        } else {
            (base[i], Kind::Fallback)
        })
    };
    let out: Option<Vec<(f64, Kind)>> =
        threads::install(|| (0..nodes.len()).into_par_iter().map(one).collect());
    Some(out?.into_iter().unzip())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faces_within_keeps_only_faces_whose_three_vertices_are_listed() {
        // two triangles sharing the edge 1-2
        let vf = VertexFaces::new(vec![[0, 1, 2], [1, 3, 2]], 4);
        assert_eq!(vf.n_vertices(), 4);
        assert_eq!(vf.faces_within(&[2, 1, 0]), vec![0]);
        assert_eq!(vf.faces_within(&[0, 1, 2, 3]), vec![0, 1]);
        assert!(vf.faces_within(&[1, 2]).is_empty());
    }

    /// An axis-aligned box of half-size `h` about the origin, its faces
    /// wound outward, vertex ids from `first`.
    fn cube(h: f64, first: u32) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let verts = (0..8)
            .map(|k| {
                let s = |bit: u32| if k & bit == 0 { -h } else { h };
                [s(1), s(2), s(4)]
            })
            .collect();
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        let faces = quads
            .iter()
            .flat_map(|q: &[u32; 4]| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .map(|f| f.map(|v| v + first))
            .collect();
        (verts, faces)
    }

    #[test]
    fn an_inner_membrane_is_left_out_of_the_radius() {
        // one node at the origin owning a box of half-size 2 and one of
        // half-size 1 inside it
        let (mut verts, mut faces) = cube(2.0, 0);
        let (inner_v, inner_f) = cube(1.0, 8);
        verts.extend(inner_v);
        faces.extend(inner_f);
        let tri = |f: &[u32; 3]| f.map(|v| verts[v as usize]);
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let normals: Vec<[f64; 3]> = faces
            .iter()
            .map(|f| {
                let [a, b, c] = tri(f);
                let (u, v) = (sub(b, a), sub(c, a));
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                n.map(|x| x / l)
            })
            .collect();
        let centers: Vec<[f64; 3]> = faces
            .iter()
            .map(|f| {
                let [a, b, c] = tri(f);
                [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0)
            })
            .collect();
        let vertex_faces = VertexFaces::new(faces.clone(), verts.len());
        let m = CalibrateMesh {
            vertices: &verts,
            faces: &faces,
            centers: &centers,
            normals: &normals,
            vertex_faces: &vertex_faces,
        };
        let directions: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let mut p = CalibrateParams {
            directions: &directions,
            min_n_verts_bulb: 0,
            check_outer: true,
            min_n_outer: 4,
            min_frac_outer: 0.3,
            rays_threshold: 1,
            aggregate: "median",
        };
        let node2verts = vec![(0..16).collect::<Vec<u32>>()];
        let run = |p: &CalibrateParams<'_>| {
            calibrate_radii(&m, &[[0.0; 3]], &[], &node2verts, &[false], &[100.0], p)
                .expect("a known estimator")
        };
        // the outer box's corners only
        let corner = 2.0 * 3f64.sqrt();
        assert_eq!(run(&p), (vec![corner], vec![Kind::OuterCenterline]));
        // without the check, all sixteen corners: the median between boxes
        p.check_outer = false;
        let both = (corner + 3f64.sqrt()) / 2.0;
        assert_eq!(run(&p), (vec![both], vec![Kind::FullCenterline]));
        // a skipped node keeps its base radius, as does one under it
        let skipped = calibrate_radii(&m, &[[0.0; 3]], &[], &node2verts, &[true], &[1.0], &p);
        assert_eq!(skipped, Some((vec![1.0], vec![Kind::Fallback])));
        p.aggregate = "nope";
        assert_eq!(
            calibrate_radii(&m, &[[0.0; 3]], &[], &node2verts, &[false], &[1.0], &p),
            None
        );
    }
}
