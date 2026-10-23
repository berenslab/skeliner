//! Boundary loops of face selections.

use std::collections::{BTreeMap, BTreeSet};

use crate::topology::Topology;

/// Edges shared by a face in `sel` and a face outside it, as sorted
/// `[low, high]` pairs in ascending order.
///
/// Degenerate faces have no edges, so they neither make nor break a border.
pub fn border_edges(t: &Topology, sel: &[u32]) -> Vec<[u32; 2]> {
    let mut chosen: Vec<u32> = sel.to_vec();
    chosen.sort_unstable();
    chosen.dedup();
    let inside = |f: u32| chosen.binary_search(&f).is_ok();
    let mut found = BTreeSet::new();
    for &f in &chosen {
        let Some(edges) = t.face_edges(f as usize) else {
            continue;
        };
        for e in edges {
            if t.edge_faces(e as usize).iter().any(|&g| !inside(g)) {
                found.insert(t.edges()[e as usize]);
            }
        }
    }
    found.into_iter().collect()
}

/// Closed loops along the border of `sel`, as vertex sequences.
///
/// Each loop starts at its smallest vertex and, wherever the border offers
/// a choice, steps to the smallest unvisited neighbour; a walk that cannot
/// close is dropped.  Fewer than three border edges give no loops.
pub fn border_loops(t: &Topology, sel: &[u32]) -> Vec<Vec<u32>> {
    let border = border_edges(t, sel);
    if border.len() < 3 {
        return Vec::new();
    }
    let mut adj: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &[a, b] in &border {
        adj.entry(a).or_default().insert(b);
        adj.entry(b).or_default().insert(a);
    }

    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut loops = Vec::new();
    for &start in adj.keys() {
        if visited.contains(&start) {
            continue;
        }
        let mut walk = vec![start];
        visited.insert(start);
        let (mut current, mut prev) = (start, None);
        loop {
            let nbs: Vec<u32> = adj[&current]
                .iter()
                .copied()
                .filter(|&n| Some(n) != prev)
                .collect();
            if nbs.is_empty() {
                break;
            }
            if walk.len() > 2 && nbs.contains(&start) {
                loops.push(walk);
                break;
            }
            let Some(next) = nbs.iter().copied().find(|n| !visited.contains(n)) else {
                break;
            };
            visited.insert(next);
            walk.push(next);
            (prev, current) = (Some(current), next);
        }
    }
    loops
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x4 grid of vertices, 18 triangles, faces numbered row by row.
    fn grid() -> (Vec<[u32; 3]>, usize) {
        let mut faces = Vec::new();
        for r in 0..3u32 {
            for c in 0..3u32 {
                let v = r * 4 + c;
                faces.push([v, v + 1, v + 5]);
                faces.push([v, v + 5, v + 4]);
            }
        }
        (faces, 16)
    }

    #[test]
    fn the_middle_square_is_bounded_by_one_loop() {
        let (faces, n) = grid();
        let t = Topology::new(&faces, n);
        // faces 8 and 9 form the square on vertices 5, 6, 9, 10
        let loops = border_loops(&t, &[8, 9]);
        assert_eq!(loops, vec![vec![5, 6, 10, 9]]);
    }

    #[test]
    fn an_annulus_reaching_the_mesh_boundary_has_one_border_loop() {
        let (faces, n) = grid();
        let t = Topology::new(&faces, n);
        // everything except the middle square: an annulus
        let sel: Vec<u32> = (0..18).filter(|f| *f != 8 && *f != 9).collect();
        let loops = border_loops(&t, &sel);
        assert_eq!(
            loops.len(),
            1,
            "the outer rim is a mesh boundary, not a border"
        );
        assert_eq!(loops[0], vec![5, 6, 10, 9]);
    }

    #[test]
    fn selecting_everything_leaves_no_border() {
        let (faces, n) = grid();
        let t = Topology::new(&faces, n);
        let sel: Vec<u32> = (0..18).collect();
        assert!(border_edges(&t, &sel).is_empty());
        assert!(border_loops(&t, &sel).is_empty());
    }

    #[test]
    fn a_single_face_is_too_small_for_a_loop_only_at_the_edge() {
        let (faces, n) = grid();
        let t = Topology::new(&faces, n);
        // the corner face 0 has two edges on the mesh boundary: two border
        // edges, fewer than three
        assert!(border_loops(&t, &[0]).is_empty());
        // an interior face has three border edges and closes
        assert_eq!(border_loops(&t, &[8]), vec![vec![5, 6, 10]]);
    }

    #[test]
    fn repeated_and_unsorted_selections_are_the_same_selection() {
        let (faces, n) = grid();
        let t = Topology::new(&faces, n);
        assert_eq!(border_loops(&t, &[9, 8, 9]), border_loops(&t, &[8, 9]));
    }
}
