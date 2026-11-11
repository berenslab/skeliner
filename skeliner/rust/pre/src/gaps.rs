//! Choosing the patches a gap stitch removes, and what removing them costs.

use std::collections::BTreeSet;

use mesh::Topology;
use mesh::loops::{border_edges, cycles_of_edges};

/// Whether two of `loops` share a vertex: the rim pinches there.
fn pinched(loops: &[Vec<u32>]) -> bool {
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    loops.iter().flatten().any(|&v| !seen.insert(v))
}

/// The first longest of `loops`, as Python's `max(loops, key=len)` picks.
fn first_longest(loops: &[Vec<u32>]) -> Option<&Vec<u32>> {
    loops
        .iter()
        .reduce(|a, b| if b.len() > a.len() { b } else { a })
}

/// The rim of `sel`: the closed loops along the border of what removing it
/// leaves.  That border is the edges between `sel` and the faces outside
/// it and, where `sel` takes faces off a hole in the mesh (an edge of one
/// face), the rest of that hole's edges.  A tip on an open end — a neurite
/// the segmentation cut, gaping at the piece across the gap — has its rim
/// round the end: by the edges to outside faces alone its border runs open
/// from hole to hole and closes no loop.  A selection that takes no face
/// off a hole has the border loops of its own faces; one touching a hole
/// only at a vertex leaves the hole whole.
///
/// The loops are the border's simple cycles ([`cycles_of_edges`]): where
/// the border pinches — touches itself at a vertex, as when a patch meets a
/// hole at a vertex besides the edges it takes off it — each lobe is a
/// loop, so a patch that cuts a flap off at the pinch has two, as one that
/// wraps a tube does.  Traced as one walk, a pinched border gave no loop or
/// one lobe, and a patch that severed passed for one that does not.
pub fn rim_loops(t: &Topology, sel: &BTreeSet<u32>) -> Vec<Vec<u32>> {
    let faces: Vec<u32> = sel.iter().copied().collect();
    let mut edges: BTreeSet<[u32; 2]> = border_edges(t, &faces).into_iter().collect();
    let open = |e: usize| t.edge_faces(e).len() == 1;
    // the selection's own edges on a hole, and the hole edges joined to
    // them through shared vertices
    let mut hole: BTreeSet<usize> = sel
        .iter()
        .filter_map(|&f| t.face_edges(f as usize))
        .flatten()
        .map(|e| e as usize)
        .filter(|&e| open(e))
        .collect();
    let mut walk: Vec<u32> = hole.iter().flat_map(|&e| t.edges()[e]).collect();
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    while let Some(v) = walk.pop() {
        if !seen.insert(v) {
            continue;
        }
        for &g in t.vertex_faces(v as usize) {
            for e in t.face_edges(g as usize).into_iter().flatten() {
                let e = e as usize;
                let [a, b] = t.edges()[e];
                if (a == v || b == v) && open(e) && hole.insert(e) {
                    walk.push(if a == v { b } else { a });
                }
            }
        }
    }
    for e in hole {
        if !sel.contains(&t.edge_faces(e)[0]) {
            edges.insert(t.edges()[e]);
        }
    }
    let edges: Vec<[u32; 2]> = edges.into_iter().collect();
    cycles_of_edges(&edges)
}

/// Grow `tip` by rings of edge neighbours until its longest rim loop
/// ([`rim_loops`]) has `target_verts` vertices and the rim is clean — no
/// two of its loops share a vertex (`skeliner.pre._expand_tip_to_good_rim`).
/// A pinched rim, two lobes touching at a vertex, would leave the lobe the
/// tube does not take hanging by that vertex: a sliver stranded, which the
/// next ring takes in.
///
/// Stops after `max_iters` rings, when a ring adds nothing, or before the
/// selection passes `max(len(tip) * max_eat_factor, target_verts * 4)`
/// faces.  Returns a selection and the longest loop of its rim: the first
/// whose rim is good, else the last whose rim had a loop — the loop is
/// always the rim of the selection that comes with it — else the last
/// selection and no loop.  (Grown past its last loop — a speck swallowed
/// whole — the selection would take away the faces the loop's tube was to
/// meet.)
pub fn expand_tip_to_good_rim(
    t: &Topology,
    tip: &[u32],
    target_verts: usize,
    max_iters: usize,
    max_eat_factor: usize,
) -> (BTreeSet<u32>, Option<Vec<u32>>) {
    let mut sel: BTreeSet<u32> = tip.iter().copied().collect();
    if sel.is_empty() {
        return (sel, None);
    }
    let max_size = (sel.len() * max_eat_factor).max(target_verts * 4);
    let mut best: Option<(BTreeSet<u32>, Vec<u32>)> = None;
    for _ in 0..max_iters {
        let loops = rim_loops(t, &sel);
        if let Some(biggest) = first_longest(&loops) {
            if biggest.len() >= target_verts && !pinched(&loops) {
                let lp = biggest.clone();
                return (sel, Some(lp));
            }
            best = Some((sel.clone(), biggest.clone()));
        }
        let mut next = sel.clone();
        for &f in &sel {
            next.extend(t.face_neighbors(f as usize));
        }
        if next == sel || next.len() > max_size {
            break;
        }
        sel = next;
    }
    match best {
        Some((at, lp)) => (at, Some(lp)),
        None => (sel, None),
    }
}

/// Expand each edge-connected group of `sel` on its own, then trace the
/// merged selection's rim (`skeliner.pre._expand_selection_per_group`).
pub fn expand_selection_per_group(
    t: &Topology,
    sel: &[u32],
    target_verts: usize,
    max_iters: usize,
    max_eat_factor: usize,
) -> (BTreeSet<u32>, Vec<Vec<u32>>) {
    let chosen: BTreeSet<u32> = sel.iter().copied().collect();
    if chosen.is_empty() {
        return (chosen, Vec::new());
    }
    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut expanded: BTreeSet<u32> = BTreeSet::new();
    for &seed in &chosen {
        if visited.contains(&seed) {
            continue;
        }
        let mut group = Vec::new();
        let mut stack = vec![seed];
        while let Some(f) = stack.pop() {
            if !visited.insert(f) {
                continue;
            }
            group.push(f);
            for nb in t.face_neighbors(f as usize) {
                if chosen.contains(&nb) && !visited.contains(&nb) {
                    stack.push(nb);
                }
            }
        }
        let (grown, _) = expand_tip_to_good_rim(t, &group, target_verts, max_iters, max_eat_factor);
        expanded.extend(grown);
    }
    let loops = rim_loops(t, &expanded);
    (expanded, loops)
}

/// How many faces removing `sel` would strand, capped at `budget + 1`
/// (`skeliner.pre._sever_cost`).
///
/// The pieces left are walked whole from the faces bordering `sel`, lowest
/// first.  Everything is stranded but the largest piece, which stays, and
/// the piece holding a face of `rejoined` — the bridge's other tip, which
/// the bridge joins back: two parts glued only by fusion faces lie in one
/// face-edge component, and peeling a tip off one can cut the other off
/// across the glue.
pub fn sever_cost(t: &Topology, sel: &BTreeSet<u32>, budget: usize, rejoined: &[u32]) -> usize {
    let mut seeds: BTreeSet<u32> = BTreeSet::new();
    for &f in sel {
        seeds.extend(
            t.face_neighbors(f as usize)
                .into_iter()
                .filter(|g| !sel.contains(g)),
        );
    }
    if seeds.is_empty() {
        return 0;
    }
    let rejoined: BTreeSet<u32> = rejoined.iter().copied().collect();
    let mut seen = vec![false; t.n_faces()];
    for &f in sel {
        seen[f as usize] = true;
    }
    // each piece's size, and whether it holds the other tip
    let mut pieces: Vec<(usize, bool)> = Vec::new();
    for &seed in &seeds {
        if seen[seed as usize] {
            continue;
        }
        seen[seed as usize] = true;
        let mut stack = vec![seed];
        let (mut size, mut holds) = (0usize, false);
        while let Some(u) = stack.pop() {
            size += 1;
            holds |= rejoined.contains(&u);
            for v in t.face_neighbors(u as usize) {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    stack.push(v);
                }
            }
        }
        pieces.push((size, holds));
    }
    let total: usize = pieces.iter().map(|p| p.0).sum();
    let stay = (0..pieces.len())
        .max_by_key(|&i| (pieces[i].0, std::cmp::Reverse(i)))
        .expect("a piece per seed");
    let other = pieces
        .iter()
        .enumerate()
        .find(|&(i, p)| p.1 && i != stay)
        .map_or(0, |(_, p)| p.0);
    (total - pieces[stay].0 - other).min(budget + 1)
}

/// Faces a stitch between faces `fa` and `fb` wins back
/// (`skeliner.pre._rescue_size`): the smaller of their components, or
/// nothing when both sit in one component (a fusion join).
pub fn rescue_size(labels: &[i64], sizes: &dyn Fn(i64) -> usize, fa: usize, fb: usize) -> usize {
    let (la, lb) = (labels[fa], labels[fb]);
    if la == lb {
        return 0;
    }
    sizes(la).min(sizes(lb))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An open-ended tube of `rings` x `around` squares.
    fn tube(rings: u32, around: u32) -> Topology {
        let mut faces = Vec::new();
        for r in 0..rings {
            for i in 0..around {
                let j = (i + 1) % around;
                let (a, b) = (r * around + i, r * around + j);
                let (c, d) = (a + around, b + around);
                faces.push([a, b, d]);
                faces.push([a, d, c]);
            }
        }
        Topology::new(&faces, ((rings + 1) * around) as usize)
    }

    /// Faces of square `i` in ring `r` of a tube `around` squares wide.
    fn square(r: u32, i: u32, around: u32) -> [u32; 2] {
        let f = 2 * (r * around + i);
        [f, f + 1]
    }

    #[test]
    fn a_patch_with_a_big_enough_rim_is_returned_as_is() {
        let t = tube(10, 8);
        let mut patch = Vec::new();
        for (r, i) in [(4, 2), (4, 3), (5, 2), (5, 3)] {
            patch.extend(square(r, i, 8));
        }
        let (sel, lp) = expand_tip_to_good_rim(&t, &patch, 6, 20, 30);
        assert_eq!(sel, patch.iter().copied().collect());
        // the 2 x 2 block is bounded by its 8 perimeter vertices
        assert_eq!(lp.unwrap().len(), 8);
    }

    #[test]
    fn a_small_tip_grows_until_its_rim_is_long_enough() {
        let t = tube(10, 8);
        let tip = [square(4, 2, 8)[0]];
        let (sel, lp) = expand_tip_to_good_rim(&t, &tip, 6, 20, 30);
        assert!(sel.len() > 1 && sel.contains(&tip[0]));
        assert!(lp.unwrap().len() >= 6);
    }

    #[test]
    fn growth_stops_at_the_size_cap() {
        // On a tube four squares round, a growing patch soon wraps it and is
        // then bounded by two 4-vertex loops, so a 20-vertex rim is never
        // reached; the cap max(2 * 2, 20 * 4) = 80 stops the growth long
        // before the 240 faces run out.
        let t = tube(30, 4);
        let tip = square(15, 1, 4);
        let (sel, lp) = expand_tip_to_good_rim(&t, &tip, 20, 50, 2);
        assert!(sel.len() > 2 && sel.len() <= 80, "{}", sel.len());
        assert!(lp.unwrap().len() < 20);
    }

    #[test]
    fn a_pinched_rim_does_not_end_the_growth() {
        // Square (0,2) takes the open end's edge 2-3, and face 11
        // ([5, 14, 13]) touches the end at vertex 5: the rim is the end round
        // to 5 and the triangle round face 11, two loops sharing vertex 5.
        let t = tube(10, 8);
        let mut tip = square(0, 2, 8).to_vec();
        tip.push(11);
        let first: BTreeSet<u32> = tip.iter().copied().collect();
        let rims = rim_loops(&t, &first);
        assert!(pinched(&rims));
        assert!(first_longest(&rims).unwrap().len() >= 6);
        let (sel, lp) = expand_tip_to_good_rim(&t, &tip, 6, 20, 30);
        assert!(sel.len() > tip.len());
        let rims = rim_loops(&t, &sel);
        assert!(!pinched(&rims));
        assert_eq!(first_longest(&rims), lp.as_ref());
    }

    #[test]
    fn a_speck_swallowed_whole_keeps_the_patch_its_loop_rims() {
        // A tetrahedron from one face: its rim is a triangle, short of six;
        // the next ring takes the whole speck, which has no rim.  The loop
        // comes back with the face it rims, not with the whole speck.
        let faces = [[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]];
        let t = Topology::new(&faces, 4);
        let (sel, lp) = expand_tip_to_good_rim(&t, &[0], 6, 20, 30);
        assert_eq!(sel, BTreeSet::from([0]));
        assert_eq!(rim_loops(&t, &sel), [lp.unwrap()]);
    }

    fn squares(at: &[(u32, u32)], around: u32) -> Vec<u32> {
        at.iter().flat_map(|&(r, i)| square(r, i, around)).collect()
    }

    fn sorted(lp: &[u32]) -> Vec<u32> {
        let mut s = lp.to_vec();
        s.sort_unstable();
        s
    }

    #[test]
    fn a_tip_on_an_open_end_has_the_end_in_its_rim() {
        // Two squares on the tube's open end (ring 0's vertices 0..8): their
        // border with the other faces runs from 2 to 4, open, and closes no
        // loop; with the end's six other edges it is one rim.
        let t = tube(10, 8);
        let tip = squares(&[(0, 2), (0, 3)], 8);
        let faces: Vec<u32> = tip.clone();
        assert!(mesh::loops::border_loops(&t, &faces).is_empty());
        let (sel, lp) = expand_tip_to_good_rim(&t, &tip, 6, 20, 30);
        assert_eq!(sel, tip.iter().copied().collect());
        assert_eq!(sorted(&lp.unwrap()), [0, 1, 2, 4, 5, 6, 7, 10, 11, 12]);
    }

    #[test]
    fn a_tip_touching_an_open_end_at_a_vertex_leaves_the_end_whole() {
        // face 5, [2, 11, 10], has vertex 2 on the open end but no edge
        let t = tube(10, 8);
        let sel = BTreeSet::from([5]);
        assert_eq!(rim_loops(&t, &sel), [[2, 10, 11]]);
        assert_eq!(rim_loops(&t, &sel), mesh::loops::border_loops(&t, &[5]));
    }

    #[test]
    fn a_band_cutting_a_flap_off_an_open_end_has_two_rims() {
        // Up column 2 from the end, across ring 3, down column 5: the band
        // cuts columns 3 and 4 of rings 0..3 off, a flap of 12 faces round
        // the end's edges 3-4-5. Its border is two open paths, end to end;
        // the rim is the flap's loop and the rest's.
        let t = tube(10, 8);
        let mut at: Vec<(u32, u32)> = (0..4).flat_map(|r| [(r, 2), (r, 5)]).collect();
        at.extend([(3, 3), (3, 4)]);
        let sel: BTreeSet<u32> = squares(&at, 8).into_iter().collect();
        let faces: Vec<u32> = sel.iter().copied().collect();
        assert!(mesh::loops::border_loops(&t, &faces).is_empty());
        let mut rims: Vec<Vec<u32>> = rim_loops(&t, &sel).iter().map(|l| sorted(l)).collect();
        rims.sort();
        assert_eq!(
            rims,
            [
                vec![0, 1, 2, 6, 7, 10, 14, 18, 22, 26, 30, 34, 35, 36, 37, 38],
                vec![3, 4, 5, 11, 13, 19, 21, 27, 28, 29],
            ]
        );
        assert_eq!(sever_cost(&t, &sel, 1000, &[]), 12);
    }

    #[test]
    fn an_annulus_strands_the_far_side() {
        let t = tube(10, 6);
        // ring 4 of the tube
        let sel: BTreeSet<u32> = (48..60).collect();
        // 48 faces below, 60 above: the smaller side is stranded
        assert_eq!(sever_cost(&t, &sel, 1000, &[]), 48);
        // a budget too small for either side reports budget + 1
        assert_eq!(sever_cost(&t, &sel, 10, &[]), 11);
        // the side below holding the bridge's other tip is joined back
        assert_eq!(sever_cost(&t, &sel, 1000, &[3]), 0);
        // the other tip on the side that stays strands the side below
        assert_eq!(sever_cost(&t, &sel, 1000, &[100]), 48);
    }

    #[test]
    fn a_piece_crossing_the_budget_is_walked_whole() {
        // A path of seven faces; removing the two ends leaves one piece of
        // five, larger than the budget and the only piece: nothing strands.
        let faces: Vec<[u32; 3]> = (0..7).map(|k| [k, k + 1, k + 2]).collect();
        let t = Topology::new(&faces, 9);
        assert_eq!(sever_cost(&t, &BTreeSet::from([0, 6]), 2, &[]), 0);
    }

    #[test]
    fn a_cap_strands_nothing() {
        let t = tube(10, 6);
        let sel: BTreeSet<u32> = (0..12).collect();
        assert_eq!(sever_cost(&t, &sel, 1000, &[]), 0);
    }

    #[test]
    fn groups_expand_separately() {
        let t = tube(10, 8);
        let (a, b) = (square(2, 1, 8), square(8, 5, 8));
        let sel = [a[0], a[1], b[0], b[1]];
        let (expanded, loops) = expand_selection_per_group(&t, &sel, 6, 20, 30);
        assert!(expanded.is_superset(&sel.iter().copied().collect()));
        assert_eq!(loops.len(), 2);
    }

    #[test]
    fn rescue_is_the_smaller_side_and_nothing_inside_one_component() {
        let labels = [0, 0, 1, 2];
        let sizes = |l: i64| [10, 4, 7][l as usize];
        assert_eq!(rescue_size(&labels, &sizes, 0, 2), 4);
        assert_eq!(rescue_size(&labels, &sizes, 0, 1), 0);
        assert_eq!(rescue_size(&labels, &sizes, 2, 3), 4);
    }
}
