//! Choosing the patches a gap stitch removes, and what removing them costs.

use std::collections::BTreeSet;

use meshcore::Topology;
use meshcore::loops::border_loops;

/// The first longest of `loops`, as Python's `max(loops, key=len)` picks.
fn first_longest(loops: &[Vec<u32>]) -> Option<&Vec<u32>> {
    loops
        .iter()
        .reduce(|a, b| if b.len() > a.len() { b } else { a })
}

fn loops_of(t: &Topology, sel: &BTreeSet<u32>) -> Vec<Vec<u32>> {
    let faces: Vec<u32> = sel.iter().copied().collect();
    border_loops(t, &faces)
        .into_iter()
        .filter(|lp| lp.len() >= 3)
        .collect()
}

/// Grow `tip` by rings of edge neighbours until its longest border loop
/// has `target_verts` vertices (`skeliner.pre._expand_tip_to_good_rim`).
///
/// Stops after `max_iters` rings, when a ring adds nothing, or before the
/// selection passes `max(len(tip) * max_eat_factor, target_verts * 4)`
/// faces.  Returns the selection and the longest loop of the last traced
/// border, if any border had one.
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
    let mut best: Option<Vec<u32>> = None;
    for _ in 0..max_iters {
        let loops = loops_of(t, &sel);
        if let Some(biggest) = first_longest(&loops) {
            best = Some(biggest.clone());
            if biggest.len() >= target_verts {
                return (sel, best);
            }
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
    (sel, best)
}

/// Expand each edge-connected group of `sel` on its own, then trace the
/// merged selection's border (`skeliner.pre._expand_selection_per_group`).
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
    let loops = loops_of(t, &expanded);
    (expanded, loops)
}

/// How many faces removing `sel` would strand, capped at `budget`
/// (`skeliner.pre._sever_cost`).
///
/// Everything but the largest piece left over is stranded.  Each piece is
/// grown from the faces bordering `sel`, lowest first, and abandoned once it
/// passes `budget`; a second such piece means the cost exceeds the budget,
/// reported as `budget + 1`.
pub fn sever_cost(t: &Topology, sel: &BTreeSet<u32>, budget: usize) -> usize {
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
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut closed: Vec<usize> = Vec::new();
    let mut n_over = 0;
    for &seed in &seeds {
        if seen.contains(&seed) {
            continue;
        }
        let mut stack = vec![seed];
        seen.insert(seed);
        let mut size = 0;
        let mut over = false;
        while let Some(u) = stack.pop() {
            size += 1;
            if size > budget {
                over = true;
                // u is popped but not yet walked; the drain below walks it
                stack.push(u);
                break;
            }
            for v in t.face_neighbors(u as usize) {
                if !sel.contains(&v) && seen.insert(v) {
                    stack.push(v);
                }
            }
        }
        if over {
            n_over += 1;
            if n_over >= 2 {
                return budget + 1;
            }
            // drain the rest of this piece so its faces are not revisited
            while let Some(u) = stack.pop() {
                for v in t.face_neighbors(u as usize) {
                    if !sel.contains(&v) && seen.insert(v) {
                        stack.push(v);
                    }
                }
            }
        } else {
            closed.push(size);
        }
    }
    let total: usize = closed.iter().sum();
    if n_over > 0 {
        return total;
    }
    total - closed.iter().copied().max().unwrap_or(0)
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
    fn an_annulus_strands_the_far_side() {
        let t = tube(10, 6);
        // ring 4 of the tube
        let sel: BTreeSet<u32> = (48..60).collect();
        // 48 faces below, 60 above: the smaller side is stranded
        assert_eq!(sever_cost(&t, &sel, 1000), 48);
        // a budget too small for either side reports budget + 1
        assert_eq!(sever_cost(&t, &sel, 10), 11);
    }

    #[test]
    fn a_piece_crossing_the_budget_is_walked_whole() {
        // A path of seven faces; removing the two ends leaves one piece of
        // five, larger than the budget and the only piece: nothing strands.
        let faces: Vec<[u32; 3]> = (0..7).map(|k| [k, k + 1, k + 2]).collect();
        let t = Topology::new(&faces, 9);
        assert_eq!(sever_cost(&t, &BTreeSet::from([0, 6]), 2), 0);
    }

    #[test]
    fn a_cap_strands_nothing() {
        let t = tube(10, 6);
        let sel: BTreeSet<u32> = (0..12).collect();
        assert_eq!(sever_cost(&t, &sel, 1000), 0);
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
