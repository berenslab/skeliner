//! Non-manifold fusions: where two branches share edges or vertices.

use std::collections::{BTreeSet, VecDeque};

use mesh::Topology;
use rayon::prelude::*;

/// Faces with area below this count as zero-area.
const ZERO_AREA: f64 = 1e-6;

/// Edge-connected components of `region`, two faces joined only through an
/// edge with exactly two faces in the whole mesh; largest first, equal sizes
/// in order of their lowest face.
fn manifold_components(t: &Topology, region: &BTreeSet<u32>) -> Vec<BTreeSet<u32>> {
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut comps: Vec<BTreeSet<u32>> = Vec::new();
    for &start in region {
        if seen.contains(&start) {
            continue;
        }
        let mut comp = BTreeSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            if !seen.insert(f) {
                continue;
            }
            comp.insert(f);
            let Some(edges) = t.face_edges(f as usize) else {
                continue;
            };
            for e in edges {
                let fs = t.edge_faces(e as usize);
                if fs.len() != 2 {
                    continue;
                }
                for &g in fs {
                    if g != f && region.contains(&g) && !seen.contains(&g) {
                        queue.push_back(g);
                    }
                }
            }
        }
        comps.push(comp);
    }
    comps.sort_by_key(|c| std::cmp::Reverse(c.len()));
    comps
}

/// Whether the faces in `fan` (distinct, ascending) around one vertex fall
/// into more than one edge-connected piece.
fn fan_is_split(t: &Topology, fan: &[u32]) -> bool {
    let mut seen = vec![false; fan.len()];
    let mut n_seen = 0;
    let mut queue = VecDeque::from([0usize]);
    while let Some(i) = queue.pop_front() {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        n_seen += 1;
        for g in t.face_neighbors(fan[i] as usize) {
            if let Ok(j) = fan.binary_search(&g)
                && !seen[j]
            {
                queue.push_back(j);
            }
        }
    }
    n_seen < fan.len()
}

/// `skeliner.pre._find_nonmanifold_fusions` with the outward dots given.
///
/// Seeds are faces with a negative outward dot on an edge of three or more
/// faces, and duplicated faces with more than three neighbours.  Each
/// edge-connected seed cluster grows by rings of neighbours (zero-area faces
/// dropped); from the third ring on, once two manifold components of at
/// least `min_branch_size` faces appear, the faces where the two largest
/// meet across an edge are one fusion — none when no edge joins them: two
/// pieces sharing only a vertex and a face that bridges them can be one
/// branch's own neck as well as two branches touching.  Fans that fall apart
/// around a vertex are fusions too, after the grown ones.
pub fn find_nonmanifold_fusions(
    t: &Topology,
    faces: &[[u32; 3]],
    areas: &[f64],
    outward_dots: &[f64],
    grow_rings: usize,
    min_branch_size: usize,
) -> Vec<Vec<u32>> {
    threads::install(|| fusions_of(t, faces, areas, outward_dots, grow_rings, min_branch_size))
}

fn fusions_of(
    t: &Topology,
    faces: &[[u32; 3]],
    areas: &[f64],
    outward_dots: &[f64],
    grow_rings: usize,
    min_branch_size: usize,
) -> Vec<Vec<u32>> {
    let zero: Vec<bool> = areas.par_iter().map(|&a| a < ZERO_AREA).collect();

    // Signal 1: negative-dot faces on non-manifold edges.
    let mut seeds: BTreeSet<u32> = BTreeSet::new();
    for e in 0..t.n_edges() {
        let fs = t.edge_faces(e);
        if fs.len() > 2 {
            seeds.extend(
                fs.iter()
                    .copied()
                    .filter(|&f| outward_dots[f as usize] < 0.0),
            );
        }
    }

    // Signal 2: duplicated faces (the same three vertices) with more than
    // three neighbours.
    let mut keyed: Vec<([u32; 3], u32)> = faces
        .par_iter()
        .enumerate()
        .map(|(fi, f)| {
            let mut k = *f;
            k.sort_unstable();
            (k, fi as u32)
        })
        .collect();
    keyed.par_sort_unstable();
    let mut copied = vec![false; faces.len()];
    for run in keyed.chunk_by(|x, y| x.0 == y.0) {
        if run.len() > 1 {
            for &(_, fi) in run {
                copied[fi as usize] = true;
            }
        }
    }
    let duplicated: Vec<u32> = (0..faces.len())
        .into_par_iter()
        .filter(|&fi| copied[fi] && !zero[fi] && t.face_neighbors(fi).len() > 3)
        .map(|fi| fi as u32)
        .collect();
    seeds.extend(duplicated);

    // Signal 3: vertices whose fan — its faces that have edges and area, in
    // face order — falls apart.
    let fan_clusters: Vec<Vec<u32>> = (0..t.n_vertices())
        .into_par_iter()
        .filter_map(|v| {
            let fan: Vec<u32> = t
                .vertex_faces(v)
                .iter()
                .copied()
                .filter(|&f| !zero[f as usize])
                .collect();
            (fan.len() >= 2 && fan_is_split(t, &fan)).then_some(fan)
        })
        .collect();

    if seeds.is_empty() && fan_clusters.is_empty() {
        return Vec::new();
    }

    // Edge-connected clusters of seeds, in order of their lowest face.
    let mut clustered: BTreeSet<u32> = BTreeSet::new();
    let mut clusters: Vec<BTreeSet<u32>> = Vec::new();
    for &s in &seeds {
        if clustered.contains(&s) {
            continue;
        }
        let mut cluster = BTreeSet::new();
        let mut queue = VecDeque::from([s]);
        while let Some(f) = queue.pop_front() {
            if !clustered.insert(f) {
                continue;
            }
            cluster.insert(f);
            for g in t.face_neighbors(f as usize) {
                if seeds.contains(&g) && !clustered.contains(&g) {
                    queue.push_back(g);
                }
            }
        }
        clusters.push(cluster);
    }

    let mut result = Vec::new();
    for cluster in clusters {
        let mut region = cluster;
        for ring in 1..=grow_rings {
            let mut grown = region.clone();
            for &f in &region {
                grown.extend(t.face_neighbors(f as usize));
            }
            grown.retain(|&f| !zero[f as usize]);
            region = grown;
            if ring < 3 {
                continue;
            }
            let comps = manifold_components(t, &region);
            let big: Vec<&BTreeSet<u32>> = comps
                .iter()
                .filter(|c| c.len() >= min_branch_size)
                .collect();
            if big.len() >= 2 {
                let mut boundary: BTreeSet<u32> = BTreeSet::new();
                for &f in big[0] {
                    for g in t.face_neighbors(f as usize) {
                        if big[1].contains(&g) {
                            boundary.insert(f);
                            boundary.insert(g);
                        }
                    }
                }
                if !boundary.is_empty() {
                    result.push(boundary.into_iter().collect());
                }
                break;
            }
        }
    }
    result.extend(fan_clusters);
    result
}

/// `skeliner.pre.find_fusions`: [`find_nonmanifold_fusions`], largest first
/// (a stable sort, so equal sizes keep their order).
pub fn find_fusions(
    t: &Topology,
    faces: &[[u32; 3]],
    areas: &[f64],
    outward_dots: &[f64],
    grow_rings: usize,
    min_branch_size: usize,
) -> Vec<Vec<u32>> {
    let mut clusters =
        find_nonmanifold_fusions(t, faces, areas, outward_dots, grow_rings, min_branch_size);
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    clusters
}

type V3 = [f64; 3];

/// Twice the area of face `f`, as [`crate::breakup::usable_faces`] reads it.
fn doubled_area(verts: &[V3], f: [u32; 3]) -> f64 {
    let (p, q, r) = (
        verts[f[0] as usize],
        verts[f[1] as usize],
        verts[f[2] as usize],
    );
    let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let w = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
    let x = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
}

/// Where the copy of vertex `v` that the faces `piece` take goes: off `v`'s
/// point by a thousandth of their mean distance from it, towards the mean
/// of their other vertices — into the piece, so that the copies of a cut
/// vertex never sit at one point (two vertices at one point are no valid
/// surface, and a reader merging copies along a seam would join the cut
/// again). A thousandth stays far above the eight decimals an OBJ keeps and
/// far below anything a mesh shows. Where that mean is the point itself,
/// the copy goes against the piece's normal (into the solid it bounds), and
/// where that is none, towards its first other vertex.
pub fn copy_point(verts: &[V3], faces: &[[u32; 3]], v: u32, piece: &[u32]) -> V3 {
    let p = verts[v as usize];
    let mut link: Vec<u32> = piece
        .iter()
        .flat_map(|&f| faces[f as usize])
        .filter(|&w| w != v)
        .collect();
    link.sort_unstable();
    link.dedup();
    let k = link.len() as f64;
    let (mut c, mut reach) = ([0.0f64; 3], 0.0f64);
    for &w in &link {
        let q = verts[w as usize];
        for i in 0..3 {
            c[i] += q[i];
        }
        let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        reach += (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    }
    let c = c.map(|x| x / k);
    let reach = reach / k;
    let length = |d: V3| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let mut d = [c[0] - p[0], c[1] - p[1], c[2] - p[2]];
    if length(d) == 0.0 {
        let mut ascending = piece.to_vec();
        ascending.sort_unstable();
        let mut n = [0.0f64; 3];
        for &f in &ascending {
            let [a, b, e] = faces[f as usize].map(|w| verts[w as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let w = [e[0] - a[0], e[1] - a[1], e[2] - a[2]];
            n[0] += u[1] * w[2] - u[2] * w[1];
            n[1] += u[2] * w[0] - u[0] * w[2];
            n[2] += u[0] * w[1] - u[1] * w[0];
        }
        d = n.map(|x| -x);
    }
    if length(d) == 0.0 {
        let q = verts[link[0] as usize];
        d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    }
    let step = reach / 1000.0 / length(d);
    [p[0] + d[0] * step, p[1] + d[1] * step, p[2] + d[2] * step]
}

/// What [`split_fan_vertices`] made of a mesh's faces.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// The faces relabelled, the dropped ones `[0, 0, 0]`.
    pub faces: Vec<[u32; 3]>,
    /// Per new vertex, numbered on from the mesh's: its point
    /// ([`copy_point`]) and the vertex it copies.
    pub points: Vec<V3>,
    pub sources: Vec<u32>,
    /// The faces dropped: pieces of a split fan with no area.
    pub dropped: usize,
}

/// `skeliner.pre._split_fan_vertices`: give each extra edge-connected piece
/// of a vertex's fan its own copy of the vertex, off its point.
///
/// `candidates` (ascending) are the vertices to check, all of them when
/// `None`. Pieces are found in face order, among the faces not dropped yet.
/// The first piece with an area keeps the vertex and each later one gets a
/// new vertex, numbered on from `n_vertices` in the order split, at
/// [`copy_point`]; a piece of faces with no area is no surface — debris
/// the split leaves hanging by a point or an edge — and is dropped, every
/// one of them where no piece has an area. The pieces are read on
/// the faces as given (a copy changes no face's neighbours) and the copies'
/// points from the vertices as given. None when no fan is split.
pub fn split_fan_vertices(
    t: &Topology,
    faces: &[[u32; 3]],
    verts: &[V3],
    candidates: Option<&[u32]>,
) -> Option<Split> {
    let n_vertices = t.n_vertices();
    let all: Vec<u32>;
    let check: &[u32] = match candidates {
        Some(c) => c,
        None => {
            all = (0..n_vertices as u32).collect();
            &all
        }
    };
    let gone = |f: &[u32; 3]| f[0] == f[1] && f[1] == f[2];
    let mut out = faces.to_vec();
    let (mut points, mut sources) = (Vec::new(), Vec::new());
    let mut dropped = 0;
    for &vid in check {
        let fan: Vec<u32> = t
            .vertex_faces(vid as usize)
            .iter()
            .copied()
            .filter(|&f| !gone(&out[f as usize]))
            .collect();
        if fan.len() < 2 {
            continue;
        }
        let members: BTreeSet<u32> = fan.iter().copied().collect();
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut pieces: Vec<Vec<u32>> = Vec::new();
        for &start in &fan {
            if seen.contains(&start) {
                continue;
            }
            let mut piece = Vec::new();
            let mut queue = VecDeque::from([start]);
            while let Some(f) = queue.pop_front() {
                if !seen.insert(f) {
                    continue;
                }
                piece.push(f);
                for g in t.face_neighbors(f as usize) {
                    if members.contains(&g) && !seen.contains(&g) {
                        queue.push_back(g);
                    }
                }
            }
            pieces.push(piece);
        }
        if pieces.len() < 2 {
            continue;
        }
        let has_area: Vec<bool> = pieces
            .iter()
            .map(|p| {
                p.iter()
                    .any(|&f| doubled_area(verts, faces[f as usize]) > 0.0)
            })
            .collect();
        let keeper = has_area.iter().position(|&a| a);
        for (i, piece) in pieces.iter().enumerate() {
            if Some(i) == keeper {
                continue;
            }
            if !has_area[i] {
                for &f in piece {
                    out[f as usize] = [0, 0, 0];
                    dropped += 1;
                }
                continue;
            }
            let new_vid = (n_vertices + sources.len()) as u32;
            points.push(copy_point(verts, faces, vid, piece));
            sources.push(vid);
            for &f in piece {
                for slot in out[f as usize].iter_mut() {
                    if *slot == vid {
                        *slot = new_vid;
                    }
                }
            }
        }
    }
    (!sources.is_empty() || dropped > 0).then_some(Split {
        faces: out,
        points,
        sources,
        dropped,
    })
}

/// `skeliner.pre.remove_fusions` on face arrays: the fusion faces become
/// `[0, 0, 0]`, so no index moves, then the fans of the vertices they
/// touched are split ([`split_fan_vertices`]).
pub fn remove_fusions(faces: &[[u32; 3]], verts: &[V3], fusions: &[Vec<u32>]) -> Split {
    let n_vertices = verts.len();
    let mut out = faces.to_vec();
    let mut touched = vec![false; n_vertices];
    for &f in fusions.iter().flatten() {
        for &v in &faces[f as usize] {
            touched[v as usize] = true;
        }
        out[f as usize] = [0, 0, 0];
    }
    let candidates: Vec<u32> = (0..n_vertices as u32)
        .filter(|&v| touched[v as usize])
        .collect();
    let topo = Topology::new(&out, n_vertices);
    split_fan_vertices(&topo, &out, verts, Some(&candidates)).unwrap_or(Split {
        faces: out,
        points: Vec::new(),
        sources: Vec::new(),
        dropped: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles meeting only at vertex 2 (a bow-tie), plus a third
    /// sharing an edge with the first.
    fn bowtie() -> Vec<[u32; 3]> {
        vec![[0, 1, 2], [2, 3, 4], [0, 2, 5]]
    }

    #[test]
    fn pieces_sharing_only_a_bridging_face_are_no_fusion() {
        // fans 0-2 and 3-5 share only vertex 0; face 6, a seed, bridges them
        // along the edges 0-2 and 0-6, each of three faces: no edge joins
        // the two pieces, so nothing to cut — and no empty cluster
        let faces = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 5, 6],
            [0, 6, 7],
            [0, 7, 8],
            [0, 2, 6],
        ];
        let t = Topology::new(&faces, 9);
        let mut dots = [1.0; 7];
        dots[6] = -1.0;
        let fusions = find_nonmanifold_fusions(&t, &faces, &[1.0; 7], &dots, 20, 2);
        assert!(fusions.is_empty());
    }

    #[test]
    fn a_bowtie_vertex_is_a_fan_fusion() {
        let faces = bowtie();
        let t = Topology::new(&faces, 6);
        let fusions = find_nonmanifold_fusions(&t, &faces, &[1.0; 3], &[1.0; 3], 20, 5);
        assert_eq!(fusions, vec![vec![0, 1, 2]]);
    }

    /// The bow-tie's vertices: the first triangle and the third below the
    /// x axis, the second above it.
    fn bowtie_verts() -> Vec<V3> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    }

    #[test]
    fn splitting_the_bowtie_gives_the_second_piece_a_new_vertex() {
        let faces = bowtie();
        let v = bowtie_verts();
        let t = Topology::new(&faces, 6);
        let s = split_fan_vertices(&t, &faces, &v, None).unwrap();
        // fan of vertex 2 is faces 0, 1, 2; faces 0 and 2 share edge (0, 2),
        // face 1 is the second piece and gets vertex 6
        assert_eq!(s.faces, vec![[0, 1, 2], [6, 3, 4], [0, 2, 5]]);
        assert_eq!(s.sources, vec![2]);
        assert_eq!(s.dropped, 0);
        // the copy sits off vertex 2, a thousandth of the way its piece's
        // other vertices lie, towards them
        let p = v[2];
        let c = s.points[0];
        let d = [c[0] - p[0], c[1] - p[1], c[2] - p[2]];
        let moved = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let reach = (2f64.sqrt() + 1.0) / 2.0;
        assert!((moved - reach / 1000.0).abs() < 1e-15, "{moved}");
        // towards (1.5, 1), the mean of vertices 3 and 4
        let toward = [0.5, 1.0, 0.0];
        let cos = (d[0] * toward[0] + d[1] * toward[1]) / (moved * 1.25f64.sqrt());
        assert!((cos - 1.0).abs() < 1e-12, "{cos}");
    }

    #[test]
    fn only_the_candidates_are_split() {
        let faces = bowtie();
        let t = Topology::new(&faces, 6);
        assert_eq!(
            split_fan_vertices(&t, &faces, &bowtie_verts(), Some(&[0, 1])),
            None
        );
    }

    #[test]
    fn a_piece_with_no_area_is_dropped_not_copied() {
        // the bow-tie's second triangle flattened onto a line through
        // vertex 2: it is no surface, so the split drops it
        let faces = bowtie();
        let mut v = bowtie_verts();
        v[3] = [2.0, 0.0, 0.0];
        v[4] = [3.0, 0.0, 0.0];
        let t = Topology::new(&faces, 6);
        let s = split_fan_vertices(&t, &faces, &v, None).unwrap();
        assert_eq!(s.faces, vec![[0, 1, 2], [0, 0, 0], [0, 2, 5]]);
        assert!(s.sources.is_empty() && s.points.is_empty());
        assert_eq!(s.dropped, 1);
        // flattened first, the area piece after it keeps the vertex
        let first = vec![[2, 3, 4], [0, 1, 2], [0, 2, 5]];
        let t = Topology::new(&first, 6);
        let s = split_fan_vertices(&t, &first, &v, None).unwrap();
        assert_eq!(s.faces, vec![[0, 0, 0], [0, 1, 2], [0, 2, 5]]);
        // two flat pieces and nothing else at the vertex: both go
        let mut line = v.clone();
        line[0] = [-1.0, 0.0, 0.0];
        line[1] = [0.0, 0.0, 0.0];
        let pair = vec![[0, 1, 2], [2, 3, 4]];
        let t = Topology::new(&pair, 6);
        let s = split_fan_vertices(&t, &pair, &line, None).unwrap();
        assert_eq!(s.faces, vec![[0, 0, 0], [0, 0, 0]]);
        assert_eq!(s.dropped, 2);
    }

    #[test]
    fn a_flat_full_fan_moves_its_copy_against_its_normal() {
        // two square fans round vertex 0, one in z = 0 and one in x = 0,
        // meeting only at vertex 0: each piece's other vertices lie round
        // the point, so the second's copy goes against its normal
        let v: Vec<V3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, -1.0],
        ];
        let faces = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [0, 5, 6],
            [0, 6, 7],
            [0, 7, 8],
            [0, 8, 5],
        ];
        let t = Topology::new(&faces, 9);
        let s = split_fan_vertices(&t, &faces, &v, None).unwrap();
        assert_eq!(s.sources, vec![0]);
        let c = s.points[0];
        // the second fan, in x = 0, is wound with its normal along -x: the
        // copy goes to +x
        assert!(c[0] > 0.0 && c[1] == 0.0 && c[2] == 0.0, "{c:?}");
    }

    #[test]
    fn copies_of_one_vertex_never_share_a_point() {
        // three triangles meeting only at vertex 0
        let v: Vec<V3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 3, 4], [0, 5, 6]];
        let t = Topology::new(&faces, 7);
        let s = split_fan_vertices(&t, &faces, &v, None).unwrap();
        assert_eq!(s.sources, vec![0, 0]);
        let mut all = vec![v[0]];
        all.extend(&s.points);
        for i in 0..all.len() {
            for j in 0..i {
                assert_ne!(all[i], all[j]);
            }
        }
    }

    #[test]
    fn a_clean_sheet_has_no_fusions() {
        let faces = vec![[0, 1, 2], [1, 3, 2], [2, 3, 4]];
        let t = Topology::new(&faces, 5);
        assert!(find_nonmanifold_fusions(&t, &faces, &[1.0; 3], &[-1.0; 3], 20, 5).is_empty());
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        assert_eq!(split_fan_vertices(&t, &faces, &v, None), None);
    }

    #[test]
    fn zero_area_faces_are_left_out_of_fans() {
        // face 1 would connect the bow-tie's halves but has zero area
        let faces = vec![[0, 1, 2], [1, 2, 3], [2, 3, 4]];
        let t = Topology::new(&faces, 5);
        let fusions = find_nonmanifold_fusions(&t, &faces, &[1.0, 0.0, 1.0], &[1.0; 3], 20, 5);
        assert!(fusions.contains(&vec![0, 2]));
    }

    #[test]
    fn removing_a_fusion_face_splits_the_fan_it_held_together() {
        // two triangles meeting at vertex 0, joined by fusion face 2
        let faces = [[0, 1, 2], [0, 3, 4], [0, 2, 3]];
        let v: Vec<V3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let s = remove_fusions(&faces, &v, &[vec![2]]);
        assert_eq!(s.faces[2], [0, 0, 0]);
        assert_eq!(s.sources, vec![0]);
        assert_eq!(s.faces[0], [0, 1, 2]);
        assert_eq!(s.faces[1], [5, 3, 4]);
        assert_ne!(s.points[0], v[0]);
        let same = remove_fusions(&faces, &v, &[]);
        assert_eq!(same.faces, faces.to_vec());
        assert!(same.sources.is_empty() && same.points.is_empty());
    }
}
