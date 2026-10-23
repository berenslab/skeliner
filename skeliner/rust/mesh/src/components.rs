//! Connected components of faces.

use crate::topology::Topology;

/// Label [`face_edge_components`] gives a degenerate face.
pub const DEGENERATE: i64 = -2;

/// Components of the non-degenerate faces, two faces being connected when
/// they share an edge — any edge, non-manifold ones included.
///
/// Components are numbered in order of their lowest face.  Returns the
/// per-face labels, [`DEGENERATE`] for faces with a repeated vertex, and the
/// id of the largest component, ties going to the lowest id (0 when there
/// are no components).
pub fn face_edge_components(t: &Topology) -> (Vec<i64>, usize) {
    let n = t.n_faces();
    let mut labels = vec![-1i64; n];
    let mut stack = Vec::new();
    let (mut n_comps, mut best, mut best_size) = (0usize, 0usize, 0usize);
    for seed in 0..n {
        if t.is_degenerate(seed) {
            labels[seed] = DEGENERATE;
            continue;
        }
        if labels[seed] >= 0 {
            continue;
        }
        let id = n_comps as i64;
        labels[seed] = id;
        stack.push(seed);
        let mut size = 0usize;
        while let Some(fi) = stack.pop() {
            size += 1;
            let edges = t
                .face_edges(fi)
                .expect("only faces with edges are labelled");
            for e in edges {
                for &nb in t.edge_faces(e as usize) {
                    let nb = nb as usize;
                    if labels[nb] == -1 {
                        labels[nb] = id;
                        stack.push(nb);
                    }
                }
            }
        }
        if size > best_size {
            best = n_comps;
            best_size = size;
        }
        n_comps += 1;
    }
    (labels, best)
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn no_faces_means_main_zero() {
        assert_eq!(labels_of(&[], 0), (vec![], 0));
        assert_eq!(labels_of(&[[1, 1, 2]], 3), (vec![DEGENERATE], 0));
    }
}
