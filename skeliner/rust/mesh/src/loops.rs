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
    loops_of_edges(&border_edges(t, sel))
}

/// Closed loops through `edges` (`[low, high]` pairs), as vertex sequences,
/// traced as [`border_loops`] traces a border: each from its smallest
/// vertex, at a choice to the smallest unvisited neighbour, a walk that
/// cannot close dropped.  Fewer than three edges give no loops.
pub fn loops_of_edges(edges: &[[u32; 2]]) -> Vec<Vec<u32>> {
    let border = edges;
    if border.len() < 3 {
        return Vec::new();
    }
    let mut adj: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &[a, b] in border {
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

/// The simple cycles through `edges` (`[low, high]` pairs), every edge in
/// at most one.  Dangling paths are pruned first (no cycle runs through an
/// end with one edge); then each walk starts at the smallest vertex with an
/// edge left, steps along its smallest unused edge, and cuts a cycle off
/// whenever it comes back to a vertex already on it; a walk that runs out
/// of edges leaves its open end unlooped, and what dangles is pruned again.
/// Where every vertex has two edges these are the loops [`loops_of_edges`]
/// traces; at a pinch — a vertex where two loops touch — both come out,
/// where [`loops_of_edges`] can trace neither.
pub fn cycles_of_edges(edges: &[[u32; 2]]) -> Vec<Vec<u32>> {
    let mut adj: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &[a, b] in edges {
        adj.entry(a).or_default().insert(b);
        adj.entry(b).or_default().insert(a);
    }
    let mut loops = Vec::new();
    loop {
        prune_dangling(&mut adj);
        let Some(start) = adj.iter().find(|(_, n)| !n.is_empty()).map(|(&v, _)| v) else {
            break;
        };
        let mut walk = vec![start];
        // each vertex on the walk, at its place in it
        let mut on: BTreeMap<u32, usize> = BTreeMap::from([(start, 0)]);
        let mut cur = start;
        while let Some(next) = adj.get(&cur).and_then(|n| n.first().copied()) {
            adj.get_mut(&cur).expect("on an edge").remove(&next);
            adj.get_mut(&next).expect("on an edge").remove(&cur);
            if let Some(&at) = on.get(&next) {
                let mut cycle = vec![next];
                cycle.extend(walk.split_off(at + 1));
                for v in &cycle[1..] {
                    on.remove(v);
                }
                loops.push(cycle);
                if next == start && walk.len() == 1 {
                    break;
                }
            } else {
                on.insert(next, walk.len());
                walk.push(next);
            }
            cur = next;
        }
    }
    loops
}

/// Remove, end by end, every path that dangles from `adj`: a vertex with
/// one edge left loses it, which may leave its neighbour with one.
fn prune_dangling(adj: &mut BTreeMap<u32, BTreeSet<u32>>) {
    let mut ends: Vec<u32> = adj
        .iter()
        .filter(|(_, n)| n.len() == 1)
        .map(|(&v, _)| v)
        .collect();
    while let Some(v) = ends.pop() {
        let Some(w) = adj
            .get(&v)
            .filter(|n| n.len() == 1)
            .and_then(|n| n.first().copied())
        else {
            continue;
        };
        adj.get_mut(&v).expect("an end").clear();
        let nw = adj.get_mut(&w).expect("on an edge");
        nw.remove(&v);
        if nw.len() == 1 {
            ends.push(w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycles_are_the_traced_loops_where_every_vertex_has_two_edges() {
        let (faces, n) = grid();
        let t = Topology::new(&faces, n);
        for sel in [
            vec![8, 9],
            vec![8],
            (0..18).filter(|f| *f != 8 && *f != 9).collect(),
        ] {
            let edges = border_edges(&t, &sel);
            assert_eq!(cycles_of_edges(&edges), loops_of_edges(&edges));
        }
    }

    #[test]
    fn a_pinch_gives_both_loops() {
        // two triangles touching at vertex 2: 0-1-2 and 2-3-4
        let edges = [[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4]];
        assert!(loops_of_edges(&edges).len() < 2);
        assert_eq!(cycles_of_edges(&edges), [vec![0, 1, 2], vec![2, 3, 4]]);
        // the walk meets the pinch mid-loop: 0-1-2-3 and 1-4-5
        let edges = [[0, 1], [1, 2], [2, 3], [0, 3], [1, 4], [4, 5], [1, 5]];
        let mut lobes: Vec<Vec<u32>> = cycles_of_edges(&edges);
        for lp in &mut lobes {
            lp.sort_unstable();
        }
        lobes.sort();
        assert_eq!(lobes, [vec![0, 1, 2, 3], vec![1, 4, 5]]);
    }

    #[test]
    fn an_open_path_has_no_cycle() {
        assert!(cycles_of_edges(&[[0, 1], [1, 2], [2, 3]]).is_empty());
        // a loop with a tail: the loop alone
        let edges = [[0, 1], [1, 2], [0, 2], [2, 3]];
        assert_eq!(cycles_of_edges(&edges), [vec![0, 1, 2]]);
    }

    #[test]
    fn a_walk_never_spends_a_loop_on_a_tail() {
        // The loop 2-5-6-7 with tails 0-2 and 5-1.  Unpruned, the walk from
        // 0 comes in at 2, steps to 5, and at 5 the smaller step is the
        // tail to 1: it dead-ends there having spent the loop's edge 2-5.
        let edges = [[0, 2], [2, 5], [5, 6], [6, 7], [2, 7], [1, 5]];
        assert_eq!(cycles_of_edges(&edges), [vec![2, 5, 6, 7]]);
    }

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
