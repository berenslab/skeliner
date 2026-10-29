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

/// `skeliner.pre._split_fan_vertices`: give each extra edge-connected piece
/// of a vertex's fan its own copy of the vertex.
///
/// `candidates` (ascending) are the vertices to check, all of them when
/// `None`.  Pieces are found in face order; the first keeps the vertex and
/// each later one gets a new vertex, numbered on from `n_vertices` in the
/// order split.  Returns the relabelled faces and, per new vertex, the
/// vertex it copies; nothing when no fan is split.
pub fn split_fan_vertices(
    t: &Topology,
    faces: &[[u32; 3]],
    candidates: Option<&[u32]>,
) -> Option<(Vec<[u32; 3]>, Vec<u32>)> {
    let n_vertices = t.n_vertices();
    let all: Vec<u32>;
    let check: &[u32] = match candidates {
        Some(c) => c,
        None => {
            all = (0..n_vertices as u32).collect();
            &all
        }
    };
    let mut out = faces.to_vec();
    let mut sources = Vec::new();
    for &vid in check {
        let fan = t.vertex_faces(vid as usize);
        if fan.len() < 2 {
            continue;
        }
        let members: BTreeSet<u32> = fan.iter().copied().collect();
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut pieces: Vec<Vec<u32>> = Vec::new();
        for &start in fan {
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
        for piece in pieces.iter().skip(1) {
            let new_vid = (n_vertices + sources.len()) as u32;
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
    (!sources.is_empty()).then_some((out, sources))
}

/// `skeliner.pre.remove_fusions` on face arrays: the fusion faces become
/// `[0, 0, 0]`, so no index moves, then the fans of the vertices they
/// touched are split ([`split_fan_vertices`]).  Returns the new faces and
/// the vertex each appended vertex copies.
pub fn remove_fusions(
    faces: &[[u32; 3]],
    n_vertices: usize,
    fusions: &[Vec<u32>],
) -> (Vec<[u32; 3]>, Vec<u32>) {
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
    match split_fan_vertices(&topo, &out, Some(&candidates)) {
        Some(split) => split,
        None => (out, Vec::new()),
    }
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

    #[test]
    fn splitting_the_bowtie_gives_the_second_piece_a_new_vertex() {
        let faces = bowtie();
        let t = Topology::new(&faces, 6);
        let (out, sources) = split_fan_vertices(&t, &faces, None).unwrap();
        // fan of vertex 2 is faces 0, 1, 2; faces 0 and 2 share edge (0, 2),
        // face 1 is the second piece and gets vertex 6
        assert_eq!(out, vec![[0, 1, 2], [6, 3, 4], [0, 2, 5]]);
        assert_eq!(sources, vec![2]);
    }

    #[test]
    fn only_the_candidates_are_split() {
        let faces = bowtie();
        let t = Topology::new(&faces, 6);
        assert_eq!(split_fan_vertices(&t, &faces, Some(&[0, 1])), None);
    }

    #[test]
    fn a_clean_sheet_has_no_fusions() {
        let faces = vec![[0, 1, 2], [1, 3, 2], [2, 3, 4]];
        let t = Topology::new(&faces, 5);
        assert!(find_nonmanifold_fusions(&t, &faces, &[1.0; 3], &[-1.0; 3], 20, 5).is_empty());
        assert_eq!(split_fan_vertices(&t, &faces, None), None);
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
        let (out, sources) = remove_fusions(&faces, 5, &[vec![2]]);
        assert_eq!(out[2], [0, 0, 0]);
        assert_eq!(sources, vec![0]);
        assert_eq!(out[0], [0, 1, 2]);
        assert_eq!(out[1], [5, 3, 4]);
        let (same, none) = remove_fusions(&faces, 5, &[]);
        assert_eq!(same, faces.to_vec());
        assert!(none.is_empty());
    }
}
