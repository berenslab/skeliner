//! Component labelling specific to the pipeline.

use mesh::components::LowestRootSets;
use rayon::prelude::*;

/// Label a fusion-cluster face gets in [`refine_components_with_fusions`].
pub const FUSION: i64 = -3;

/// Re-label every component with fusion-cluster faces as walls and edges
/// shared by more than two labelled faces severed.
///
/// `labels` are per-face component labels, negative for faces outside every
/// component; `fusion` lists the fusion-cluster faces.  Each sub-component
/// found gets a new label above `labels`' maximum, in order of its lowest
/// face; fusion faces inside a component get [`FUSION`]; other faces keep
/// their label.  Returns the labels and the largest sub-component, ties to
/// the lowest label — or `main` unchanged when there is none.
pub fn refine_components_with_fusions(
    faces: &[[u32; 3]],
    labels: &[i64],
    main: i64,
    fusion: &[u32],
) -> (Vec<i64>, i64) {
    threads::install(|| refine(faces, labels, main, fusion))
}

fn refine(faces: &[[u32; 3]], labels: &[i64], main: i64, fusion: &[u32]) -> (Vec<i64>, i64) {
    let n = faces.len();
    let mut is_fusion = vec![false; n];
    for &f in fusion {
        if let Some(slot) = is_fusion.get_mut(f as usize) {
            *slot = true;
        }
    }
    let members: Vec<usize> = (0..n)
        .into_par_iter()
        .filter(|&fi| labels[fi] >= 0)
        .collect();

    // Each (edge, face) use among the labelled faces; an edge with more than
    // two uses is severed, and over every other one its non-fusion faces
    // join.  Equal uses (a degenerate face's repeated edge) are the same
    // value, so the order is the same however it is sorted.
    let mut uses: Vec<(u64, u32)> = members
        .par_iter()
        .flat_map_iter(|&fi| {
            let f = faces[fi];
            (0..3).map(move |i| {
                let (a, b) = (f[i], f[(i + 1) % 3]);
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                ((u64::from(lo) << 32) | u64::from(hi), fi as u32)
            })
        })
        .collect();
    uses.par_sort_unstable();
    let sets = LowestRootSets::new(n);
    uses.par_chunk_by(|x, y| x.0 == y.0).for_each(|group| {
        if group.len() > 2 {
            return;
        }
        let mut walk = group
            .iter()
            .map(|&(_, fi)| fi)
            .filter(|&fi| !is_fusion[fi as usize]);
        if let (Some(a), Some(b)) = (walk.next(), walk.next()) {
            sets.join(a, b);
        }
    });
    let roots = sets.roots();

    // a sub-component per root, the lowest non-fusion face, in face order
    let mut new_labels = labels.to_vec();
    let mut id = vec![-1i64; n];
    let mut sizes: Vec<usize> = Vec::new();
    let first = labels.iter().copied().max().map_or(0, |m| m + 1);
    for &fi in &members {
        if is_fusion[fi] {
            new_labels[fi] = FUSION;
            continue;
        }
        let r = roots[fi] as usize;
        if r == fi {
            id[fi] = first + sizes.len() as i64;
            sizes.push(0);
        }
        new_labels[fi] = id[r];
        sizes[(id[r] - first) as usize] += 1;
    }
    let mut best: Option<(usize, i64)> = None;
    for (k, &size) in sizes.iter().enumerate() {
        if best.is_none_or(|(s, _)| size > s) {
            best = Some((size, first + k as i64));
        }
    }
    (new_labels, best.map_or(main, |(_, cid)| cid))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refinement as a depth-first walk from each lowest face labels it.
    fn walked(faces: &[[u32; 3]], labels: &[i64], main: i64, fusion: &[u32]) -> (Vec<i64>, i64) {
        let n = faces.len();
        let mut is_fusion = vec![false; n];
        for &f in fusion {
            if let Some(slot) = is_fusion.get_mut(f as usize) {
                *slot = true;
            }
        }
        let members: Vec<usize> = (0..n).filter(|&fi| labels[fi] >= 0).collect();

        // Each (edge, face) use among the labelled faces: the use count decides
        // which edges are severed; only non-fusion faces can be walked to.
        let mut uses: Vec<(u64, u32)> = Vec::with_capacity(3 * members.len());
        for &fi in &members {
            let f = faces[fi];
            for i in 0..3 {
                let (a, b) = (f[i], f[(i + 1) % 3]);
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                uses.push(((u64::from(lo) << 32) | u64::from(hi), fi as u32));
            }
        }
        uses.sort_unstable();

        // Per face, the edge groups it belongs to; per group, its use count and
        // its walkable faces.
        let mut group_of_use = vec![0u32; uses.len()];
        let mut severed: Vec<bool> = Vec::new();
        let mut walk_ptr: Vec<usize> = vec![0];
        let mut walkable: Vec<u32> = Vec::new();
        let mut offset = 0;
        for group in uses.chunk_by(|x, y| x.0 == y.0) {
            let g = severed.len() as u32;
            severed.push(group.len() > 2);
            walkable.extend(
                group
                    .iter()
                    .map(|&(_, fi)| fi)
                    .filter(|&fi| !is_fusion[fi as usize]),
            );
            walk_ptr.push(walkable.len());
            group_of_use[offset..offset + group.len()].fill(g);
            offset += group.len();
        }
        let mut face_groups: Vec<[u32; 3]> = vec![[u32::MAX; 3]; n];
        let mut filled = vec![0u8; n];
        for (k, &(_, fi)) in uses.iter().enumerate() {
            let fi = fi as usize;
            face_groups[fi][filled[fi] as usize] = group_of_use[k];
            filled[fi] += 1;
        }

        let mut new_labels = labels.to_vec();
        let mut visited = vec![false; n];
        let mut next_id = labels.iter().copied().max().map_or(0, |m| m + 1);
        let mut best: Option<(usize, i64)> = None;
        let mut stack = Vec::new();
        for &seed in &members {
            if visited[seed] || is_fusion[seed] {
                continue;
            }
            let cid = next_id;
            next_id += 1;
            let mut size = 0usize;
            stack.push(seed);
            while let Some(fi) = stack.pop() {
                if visited[fi] {
                    continue;
                }
                visited[fi] = true;
                new_labels[fi] = cid;
                size += 1;
                for &g in &face_groups[fi] {
                    let g = g as usize;
                    if severed[g] {
                        continue;
                    }
                    for &nb in &walkable[walk_ptr[g]..walk_ptr[g + 1]] {
                        let nb = nb as usize;
                        if !visited[nb] && !is_fusion[nb] {
                            stack.push(nb);
                        }
                    }
                }
            }
            if best.is_none_or(|(s, _)| size > s) {
                best = Some((size, cid));
            }
        }
        for &fi in &members {
            if is_fusion[fi] {
                new_labels[fi] = FUSION;
            }
        }
        (new_labels, best.map_or(main, |(_, cid)| cid))
    }

    #[test]
    fn many_pieces_refine_as_a_walk_refines_them() {
        // clusters of 12 vertices, 30 random faces each: repeated vertices,
        // fused edges; a few labels negative and one face in eight fusion
        let mut state = 0x6a09_e667_f3bc_c908u64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % n) as u32
        };
        let faces: Vec<[u32; 3]> = (0..30_000)
            .map(|_| {
                let base = 12 * next(1000);
                [base + next(12), base + next(12), base + next(12)]
            })
            .collect();
        let labels: Vec<i64> = faces
            .iter()
            .map(|f| {
                if next(20) == 0 {
                    -2
                } else {
                    i64::from(f[0] / 1200)
                }
            })
            .collect();
        let fusion: Vec<u32> = (0..faces.len() as u32).filter(|_| next(8) == 0).collect();
        let (new, main) = refine_components_with_fusions(&faces, &labels, 7, &fusion);
        assert_eq!((new.clone(), main), walked(&faces, &labels, 7, &fusion));
        let pieces = new
            .iter()
            .filter(|&&l| l > 10)
            .collect::<std::collections::HashSet<_>>();
        assert!(pieces.len() > 1000 && new.contains(&FUSION) && new.contains(&-2));
        assert_eq!(
            refine_components_with_fusions(&faces, &labels, 7, &[]),
            walked(&faces, &labels, 7, &[])
        );
    }

    /// Two triangle strips glued along edge (0, 1) by a third face, which is
    /// what makes the edge non-manifold.
    fn glued() -> Vec<[u32; 3]> {
        vec![
            [0, 1, 2], // strip A
            [1, 2, 3], // strip A
            [1, 0, 4], // strip B
            [0, 4, 5], // strip B
            [0, 1, 6], // third sheet on (0, 1)
            [9, 9, 8], // degenerate
        ]
    }

    #[test]
    fn a_non_manifold_edge_splits_the_component() {
        let labels = vec![0, 0, 0, 0, 0, -2];
        let (new, main) = refine_components_with_fusions(&glued(), &labels, 0, &[]);
        // strips and the third sheet come apart; numbering starts above 0
        assert_eq!(new, vec![1, 1, 2, 2, 3, -2]);
        assert_eq!(main, 1);
    }

    #[test]
    fn fusion_faces_are_walls_and_marked() {
        let faces = vec![[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]];
        let labels = vec![0, 0, 0, 0];
        let (new, main) = refine_components_with_fusions(&faces, &labels, 0, &[1]);
        assert_eq!(new, vec![1, FUSION, 2, 2]);
        assert_eq!(main, 2);
    }

    #[test]
    fn severed_edges_still_count_fusion_faces() {
        // (0, 1) has three uses, one of them a fusion face: still severed.
        let labels = vec![0, 0, 0, 0, 0, -2];
        let (new, _) = refine_components_with_fusions(&glued(), &labels, 0, &[4]);
        assert_eq!(new, vec![1, 1, 2, 2, FUSION, -2]);
    }

    #[test]
    fn nothing_to_relabel_keeps_main() {
        let labels = vec![-2, -2];
        let faces = vec![[0, 0, 1], [1, 1, 2]];
        assert_eq!(
            refine_components_with_fusions(&faces, &labels, 7, &[]),
            (labels.clone(), 7)
        );
    }
}
