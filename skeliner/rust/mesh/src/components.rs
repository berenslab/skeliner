//! Connected components of faces.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::topology::Topology;

/// Label [`face_edge_components`] gives a degenerate face.
pub const DEGENERATE: i64 = -2;

/// Components of the non-degenerate faces, two faces being connected when
/// they share an edge — any edge, non-manifold ones included.
///
/// Components are numbered in order of their lowest face.  Returns the
/// per-face labels, [`DEGENERATE`] for faces with a repeated vertex, and the
/// id of the largest component, ties going to the lowest id (0 when there
/// are no components).  Runs in parallel on [`threads`]' pool.
pub fn face_edge_components(t: &Topology) -> (Vec<i64>, usize) {
    components_of(t, |_| false)
}

/// [`face_edge_components`] of the mesh without the faces `gone` marks:
/// labelled [`DEGENERATE`] and connecting nothing, as if their vertices
/// had been collapsed.
pub fn face_edge_components_without(t: &Topology, gone: &[bool]) -> (Vec<i64>, usize) {
    components_of(t, |f| gone[f])
}

fn components_of(t: &Topology, gone: impl Fn(usize) -> bool + Sync) -> (Vec<i64>, usize) {
    threads::install(|| {
        let n = t.n_faces();
        let out: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|f| t.is_degenerate(f) || gone(f))
            .collect();
        // the faces left on an edge join its first
        let sets = LowestRootSets::new(n);
        (0..t.n_edges()).into_par_iter().for_each(|e| {
            let mut left = t
                .edge_faces(e)
                .iter()
                .copied()
                .filter(|&f| !out[f as usize]);
            if let Some(first) = left.next() {
                for f in left {
                    sets.join(first, f);
                }
            }
        });
        let roots = sets.roots();

        // numbered in order of their lowest face, which is the root
        let mut id = vec![-1i64; n];
        let mut n_comps = 0i64;
        for f in 0..n {
            if !out[f] && roots[f] as usize == f {
                id[f] = n_comps;
                n_comps += 1;
            }
        }
        let labels: Vec<i64> = (0..n)
            .into_par_iter()
            .map(|f| {
                if out[f] {
                    DEGENERATE
                } else {
                    id[roots[f] as usize]
                }
            })
            .collect();
        let mut sizes = vec![0usize; n_comps as usize];
        for &l in &labels {
            if l >= 0 {
                sizes[l as usize] += 1;
            }
        }
        let (mut best, mut best_size) = (0usize, 0usize);
        for (c, &size) in sizes.iter().enumerate() {
            if size > best_size {
                best = c;
                best_size = size;
            }
        }
        (labels, best)
    })
}

/// Disjoint sets over `0..n` that threads may join at once.  Parents only
/// ever point to lower members, so each set's root is its lowest member,
/// whatever order the joins came in.
pub struct LowestRootSets {
    parent: Vec<AtomicU32>,
}

impl LowestRootSets {
    /// `n` sets of one member each.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).into_par_iter().map(AtomicU32::new).collect(),
        }
    }

    /// The root of `x`'s set, halving the path on the way.  Losing a race
    /// to another thread only leaves a path longer.
    pub fn root(&self, mut x: u32) -> u32 {
        let parent = &self.parent;
        loop {
            let p = parent[x as usize].load(Ordering::Relaxed);
            if p == x {
                return x;
            }
            let g = parent[p as usize].load(Ordering::Relaxed);
            let _ = parent[x as usize].compare_exchange_weak(
                p,
                g,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            x = g;
        }
    }

    /// Join the sets of `a` and `b`, the higher root under the lower.
    pub fn join(&self, mut a: u32, mut b: u32) {
        loop {
            a = self.root(a);
            b = self.root(b);
            if a == b {
                return;
            }
            let (lo, hi) = (a.min(b), a.max(b));
            match self.parent[hi as usize].compare_exchange(
                hi,
                lo,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                // another thread hung `hi` meanwhile: go on from where it points
                Err(now) => (a, b) = (now, lo),
            }
        }
    }

    /// Every member's root, once the joins are done.
    pub fn roots(&self) -> Vec<u32> {
        (0..self.parent.len() as u32)
            .into_par_iter()
            .map(|x| self.root(x))
            .collect()
    }
}

/// Components of the faces listed in `fi`, two faces being connected when
/// they have an edge in common, largest component first.
///
/// A face's edges here are its three sorted vertex pairs, so a face with a
/// repeated vertex still has some.  Components with as many faces come in
/// order of their lowest face, and each lists its faces in the order of
/// `fi`.  A face listed twice is connected only through its last listing;
/// the earlier one stands alone.
pub fn listed_face_components(faces: &[[u32; 3]], fi: &[u32]) -> Vec<Vec<u32>> {
    let n = fi.len();
    if n == 0 {
        return Vec::new();
    }
    threads::install(|| {
        // Only a face's last listing carries its edges.
        let mut last = vec![usize::MAX; fi.iter().max().map_or(0, |&m| m as usize + 1)];
        for (i, &f) in fi.iter().enumerate() {
            last[f as usize] = i;
        }
        // Equal entries (a repeated vertex's edge, twice) are the same
        // value, so the order is the same however it is sorted.
        let mut entries: Vec<(u64, usize)> = (0..n)
            .into_par_iter()
            .filter(|&i| last[fi[i] as usize] == i)
            .flat_map_iter(|i| {
                let t = faces[fi[i] as usize];
                crate::topology::EDGE_COLS.iter().map(move |&(a, b)| {
                    let (lo, hi) = if t[a] <= t[b] {
                        (t[a], t[b])
                    } else {
                        (t[b], t[a])
                    };
                    ((u64::from(lo) << 32) | u64::from(hi), i)
                })
            })
            .collect();
        entries.par_sort_unstable();

        let sets = LowestRootSets::new(n);
        entries.par_chunk_by(|x, y| x.0 == y.0).for_each(|group| {
            for w in group.windows(2) {
                sets.join(w[0].1 as u32, w[1].1 as u32);
            }
        });
        let roots = sets.roots();

        // Number components by their first listing, as a graph traversal
        // from the lowest node would.
        let mut label = vec![usize::MAX; n];
        let mut members: Vec<Vec<u32>> = Vec::new();
        for (i, &f) in fi.iter().enumerate() {
            let root = roots[i] as usize;
            if label[root] == usize::MAX {
                label[root] = members.len();
                members.push(Vec::new());
            }
            members[label[root]].push(f);
        }
        members.sort_by_key(|c| {
            (
                std::cmp::Reverse(c.len()),
                *c.iter().min().expect("non-empty"),
            )
        });
        members
    })
}

/// The flagged faces left after dropping each edge-connected group of
/// flagged faces smaller than `min_size` (edges as in
/// [`listed_face_components`]).
pub fn keep_large_clusters(faces: &[[u32; 3]], flagged: &[bool], min_size: usize) -> Vec<bool> {
    let listed: Vec<u32> = (0..faces.len() as u32)
        .filter(|&f| flagged[f as usize])
        .collect();
    let mut keep = vec![false; faces.len()];
    for comp in listed_face_components(faces, &listed) {
        if comp.len() >= min_size {
            for f in comp {
                keep[f as usize] = true;
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The components as a sequential union-find lists them.
    fn listed_one_at_a_time(faces: &[[u32; 3]], fi: &[u32]) -> Vec<Vec<u32>> {
        let n = fi.len();
        if n == 0 {
            return Vec::new();
        }
        // Only a face's last listing carries its edges.
        let mut last = vec![usize::MAX; fi.iter().max().map_or(0, |&m| m as usize + 1)];
        for (i, &f) in fi.iter().enumerate() {
            last[f as usize] = i;
        }
        let mut entries: Vec<(u64, usize)> = Vec::with_capacity(3 * n);
        for (i, &f) in fi.iter().enumerate() {
            if last[f as usize] != i {
                continue;
            }
            let t = faces[f as usize];
            for &(a, b) in &crate::topology::EDGE_COLS {
                let (lo, hi) = if t[a] <= t[b] {
                    (t[a], t[b])
                } else {
                    (t[b], t[a])
                };
                entries.push(((u64::from(lo) << 32) | u64::from(hi), i));
            }
        }
        entries.sort_unstable();

        let mut sets = UnionFind::new(n);
        for group in entries.chunk_by(|x, y| x.0 == y.0) {
            for w in group.windows(2) {
                sets.union(w[0].1, w[1].1);
            }
        }

        // Number components by their first listing, as a graph traversal from
        // the lowest node would.
        let mut label = vec![usize::MAX; n];
        let mut members: Vec<Vec<u32>> = Vec::new();
        for (i, &f) in fi.iter().enumerate() {
            let root = sets.find(i);
            if label[root] == usize::MAX {
                label[root] = members.len();
                members.push(Vec::new());
            }
            members[label[root]].push(f);
        }
        members.sort_by_key(|c| {
            (
                std::cmp::Reverse(c.len()),
                *c.iter().min().expect("non-empty"),
            )
        });
        members
    }

    /// Disjoint sets over `0..n`, by size with path halving.
    struct UnionFind {
        parent: Vec<usize>,
        size: Vec<usize>,
    }

    impl UnionFind {
        fn new(n: usize) -> Self {
            Self {
                parent: (0..n).collect(),
                size: vec![1; n],
            }
        }

        fn find(&mut self, mut x: usize) -> usize {
            while self.parent[x] != x {
                self.parent[x] = self.parent[self.parent[x]];
                x = self.parent[x];
            }
            x
        }

        fn union(&mut self, a: usize, b: usize) {
            let (mut a, mut b) = (self.find(a), self.find(b));
            if a == b {
                return;
            }
            if self.size[a] < self.size[b] {
                std::mem::swap(&mut a, &mut b);
            }
            self.parent[b] = a;
            self.size[a] += self.size[b];
        }
    }

    #[test]
    fn many_listed_faces_group_as_a_sequential_union_finds() {
        // 40k faces over 9,000 vertices in clusters, some repeated vertices,
        // listed out of order with some faces twice
        let mut state = 0xbb67_ae85_84ca_a73bu64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % n) as u32
        };
        let faces: Vec<[u32; 3]> = (0..40_000)
            .map(|_| {
                let base = 9 * next(1000);
                [base + next(9), base + next(9), base + next(9)]
            })
            .collect();
        let listed: Vec<u32> = (0..60_000).map(|_| next(40_000)).collect();
        let comps = listed_face_components(&faces, &listed);
        assert_eq!(comps, listed_one_at_a_time(&faces, &listed));
        assert!(comps.len() > 500, "{}", comps.len());
    }

    #[test]
    fn listed_components_keep_listing_order_and_sort_by_size() {
        let faces = [[0, 1, 2], [1, 0, 3], [5, 6, 7], [2, 2, 9]];
        let comps = listed_face_components(&faces, &[1, 0, 2, 3]);
        assert_eq!(comps, vec![vec![1, 0], vec![2], vec![3]]);
    }

    #[test]
    fn a_repeated_listing_stands_alone() {
        let faces = [[0, 1, 2], [1, 0, 3]];
        assert_eq!(
            listed_face_components(&faces, &[0, 1, 0]),
            vec![vec![1, 0], vec![0]]
        );
    }

    #[test]
    fn a_degenerate_face_connects_through_its_real_edge() {
        // [2, 2, 1] has the edge (1, 2), which face 0 also has.
        let faces = [[0, 1, 2], [2, 2, 1], [7, 8, 9]];
        assert_eq!(
            listed_face_components(&faces, &[2, 1, 0]),
            vec![vec![1, 0], vec![2]]
        );
    }

    #[test]
    fn small_flagged_groups_are_dropped() {
        // faces 0-1 share an edge; face 2 is alone; face 3 is not flagged
        let faces = [[0, 1, 2], [1, 0, 3], [7, 8, 9], [2, 1, 4]];
        let flagged = [true, true, true, false];
        assert_eq!(
            keep_large_clusters(&faces, &flagged, 2),
            vec![true, true, false, false]
        );
        assert_eq!(keep_large_clusters(&faces, &flagged, 1), flagged.to_vec());
    }

    #[test]
    fn nothing_listed_means_no_components() {
        assert!(listed_face_components(&[[0, 1, 2]], &[]).is_empty());
    }

    fn labels_of(faces: &[[u32; 3]], n_vertices: usize) -> (Vec<i64>, usize) {
        face_edge_components(&Topology::new(faces, n_vertices))
    }

    #[test]
    fn a_shared_vertex_does_not_connect_faces() {
        // Two triangles touching at vertex 2 only.
        let (labels, main) = labels_of(&[[0, 1, 2], [2, 3, 4]], 5);
        assert_eq!(labels, vec![0, 1]);
        assert_eq!(main, 0);
    }

    #[test]
    fn a_non_manifold_edge_connects_all_its_faces() {
        // Three sheets fused along edge (0, 1), plus a separate triangle.
        let faces = [[5, 6, 7], [0, 1, 2], [1, 0, 3], [0, 1, 4]];
        let (labels, main) = labels_of(&faces, 8);
        assert_eq!(labels, vec![0, 1, 1, 1]);
        assert_eq!(main, 1);
    }

    #[test]
    fn components_are_numbered_by_their_lowest_face() {
        // Face 0 and 3 connect; faces 1 and 2 are each alone.
        let faces = [[0, 1, 2], [10, 11, 12], [20, 21, 22], [1, 0, 3]];
        let (labels, _) = labels_of(&faces, 23);
        assert_eq!(labels, vec![0, 1, 2, 0]);
    }

    #[test]
    fn equal_sizes_go_to_the_lowest_id() {
        let faces = [[0, 1, 2], [10, 11, 12], [1, 0, 3], [11, 10, 13]];
        let (labels, main) = labels_of(&faces, 14);
        assert_eq!(labels, vec![0, 1, 0, 1]);
        assert_eq!(main, 0);
    }

    #[test]
    fn degenerate_faces_are_labelled_apart_and_connect_nothing() {
        // Face 1 is degenerate but repeats vertices of faces 0 and 2.
        let faces = [[0, 1, 2], [2, 2, 3], [2, 3, 4]];
        let (labels, main) = labels_of(&faces, 5);
        assert_eq!(labels, vec![0, DEGENERATE, 1]);
        assert_eq!(main, 0);
    }

    #[test]
    fn faces_taken_out_label_as_if_collapsed() {
        // a strip 0-1-2-3 whose second face links the two ends
        let faces = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [9, 9, 8]];
        let gone = [false, true, false, false];
        let mut collapsed = faces;
        collapsed[1] = [0, 0, 0];
        let without = face_edge_components_without(&Topology::new(&faces, 10), &gone);
        assert_eq!(without, labels_of(&collapsed, 10));
        assert_eq!(without.0, vec![0, DEGENERATE, 1, DEGENERATE]);
    }

    /// Labels by a search from each lowest unlabelled face, one at a time.
    fn searched(faces: &[[u32; 3]], n_vertices: usize, gone: &[bool]) -> (Vec<i64>, usize) {
        let t = Topology::new(faces, n_vertices);
        let out = |f: usize| t.is_degenerate(f) || gone[f];
        let mut labels = vec![-1i64; faces.len()];
        let (mut n_comps, mut best, mut best_size) = (0usize, 0usize, 0usize);
        for seed in 0..faces.len() {
            if out(seed) {
                labels[seed] = DEGENERATE;
                continue;
            }
            if labels[seed] >= 0 {
                continue;
            }
            labels[seed] = n_comps as i64;
            let (mut stack, mut size) = (vec![seed], 0);
            while let Some(f) = stack.pop() {
                size += 1;
                for e in t.face_edges(f).unwrap() {
                    for &g in t.edge_faces(e as usize) {
                        let g = g as usize;
                        if !out(g) && labels[g] == -1 {
                            labels[g] = n_comps as i64;
                            stack.push(g);
                        }
                    }
                }
            }
            if size > best_size {
                (best, best_size) = (n_comps, size);
            }
            n_comps += 1;
        }
        (labels, best)
    }

    #[test]
    fn many_components_label_as_a_search_labels_them() {
        // 2,000 clusters of 12 vertices, 30 random faces each, interleaved
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % n) as u32
        };
        let faces: Vec<[u32; 3]> = (0..60_000)
            .map(|_| {
                let base = 12 * next(2000);
                [base + next(12), base + next(12), base + next(12)]
            })
            .collect();
        let gone: Vec<bool> = (0..faces.len()).map(|_| next(10) == 0).collect();
        let n = 24_000;
        let (all, main) = face_edge_components(&Topology::new(&faces, n));
        assert_eq!(
            (all.clone(), main),
            searched(&faces, n, &vec![false; faces.len()])
        );
        let without = face_edge_components_without(&Topology::new(&faces, n), &gone);
        assert_eq!(without, searched(&faces, n, &gone));
        let n_comps = all.iter().max().unwrap() + 1;
        assert!(n_comps > 2000 && all.contains(&DEGENERATE), "{n_comps}");
    }

    #[test]
    fn no_faces_means_main_zero() {
        assert_eq!(labels_of(&[], 0), (vec![], 0));
        assert_eq!(labels_of(&[[1, 1, 2]], 3), (vec![DEGENERATE], 0));
    }
}
