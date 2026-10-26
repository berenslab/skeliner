//! An undirected, edge-weighted vertex graph and shortest paths over it.

use rayon::prelude::*;

use crate::components::LowestRootSets;

/// Adjacency as compressed rows: each vertex's neighbours ascending by id,
/// the order igraph's `incident` lists them in, with the edge weights.
#[derive(Debug, Clone)]
pub struct WeightedGraph {
    start: Vec<usize>,
    nbr: Vec<u32>,
    weight: Vec<f64>,
}

impl WeightedGraph {
    /// From `edges` (each pair once) with their `weights`, over `n` vertices.
    pub fn new(n: usize, edges: &[[u32; 2]], weights: &[f64]) -> WeightedGraph {
        let mut deg = vec![0usize; n];
        for e in edges {
            deg[e[0] as usize] += 1;
            deg[e[1] as usize] += 1;
        }
        let mut start = vec![0usize; n + 1];
        for v in 0..n {
            start[v + 1] = start[v] + deg[v];
        }
        let mut fill = start.clone();
        let mut pairs = vec![(0u32, 0.0f64); start[n]];
        for (e, &w) in edges.iter().zip(weights) {
            for (a, b) in [(e[0], e[1]), (e[1], e[0])] {
                pairs[fill[a as usize]] = (b, w);
                fill[a as usize] += 1;
            }
        }
        for v in 0..n {
            pairs[start[v]..start[v + 1]].sort_by_key(|&(b, _)| b);
        }
        let (nbr, weight) = pairs.into_iter().unzip();
        WeightedGraph { start, nbr, weight }
    }

    pub fn n_vertices(&self) -> usize {
        self.start.len() - 1
    }

    /// Neighbours of `v`, ascending.
    pub fn neighbors(&self, v: u32) -> &[u32] {
        &self.nbr[self.start[v as usize]..self.start[v as usize + 1]]
    }

    /// Neighbours of `v` with the weights of the edges to them.
    pub fn weighted(&self, v: u32) -> impl Iterator<Item = (u32, f64)> + '_ {
        let r = self.start[v as usize]..self.start[v as usize + 1];
        self.nbr[r.clone()]
            .iter()
            .copied()
            .zip(self.weight[r].iter().copied())
    }

    /// Connected components, each ascending, in order of their smallest
    /// vertex — igraph's `components()`.
    pub fn components(&self) -> Vec<Vec<u32>> {
        threads::install(|| {
            let n = self.n_vertices();
            // each set's root is its lowest vertex, so the order holds
            let sets = LowestRootSets::new(n);
            (0..n as u32).into_par_iter().for_each(|v| {
                for &u in self.neighbors(v) {
                    if u > v {
                        sets.join(v, u);
                    }
                }
            });
            let roots = sets.roots();
            let mut slot = vec![usize::MAX; n];
            let mut out: Vec<Vec<u32>> = Vec::new();
            for (v, &r) in roots.iter().enumerate() {
                let r = r as usize;
                if slot[r] == usize::MAX {
                    slot[r] = out.len();
                    out.push(Vec::new());
                }
                out[slot[r]].push(v as u32);
            }
            out
        })
    }

    /// Shortest-path distance to each vertex marked `inside` from the
    /// nearest of `seeds`, travelling only through `inside` vertices;
    /// infinity where none reaches.  Each distance is the path's weights
    /// added in order from the seed, as igraph's Dijkstra adds them.
    pub fn distances_within(&self, inside: &[bool], seeds: &[u32]) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.n_vertices()];
        self.distances_within_into(inside, seeds, &mut dist);
        dist
    }

    /// [`Self::distances_within`] into `dist`, which must hold infinity at
    /// every `inside` vertex; only those entries are written.
    ///
    /// The weights must not be negative.  Which of equally near vertices is
    /// settled first changes no distance: each is the least, over the paths
    /// to its vertex, of the sum along it (adding a weight never lowers a
    /// sum), so any order of ties ends the same.
    pub fn distances_within_into(&self, inside: &[bool], seeds: &[u32], dist: &mut [f64]) {
        let mut heap = RadixHeap::new();
        for &s in seeds {
            if inside[s as usize] && dist[s as usize] > 0.0 {
                dist[s as usize] = 0.0;
                heap.push(0.0, s);
            }
        }
        while let Some((d, v)) = heap.pop() {
            if d > dist[v as usize] {
                continue;
            }
            for (u, w) in self.weighted(v) {
                if !inside[u as usize] {
                    continue;
                }
                let nd = d + w;
                if nd < dist[u as usize] {
                    dist[u as usize] = nd;
                    heap.push(nd, u);
                }
            }
        }
    }
}

/// A queue of distances that never fall below the last one taken, as
/// Dijkstra's do, nearest first: each non-negative distance kept by its
/// bits (which order as the distances do), in the bucket of the highest
/// bit where it differs from the last taken.
struct RadixHeap {
    last: u64,
    len: usize,
    buckets: [Vec<(u64, u32)>; 65],
    spare: Vec<(u64, u32)>,
}

impl RadixHeap {
    fn new() -> Self {
        RadixHeap {
            last: 0,
            len: 0,
            buckets: std::array::from_fn(|_| Vec::new()),
            spare: Vec::new(),
        }
    }

    fn bucket(&self, key: u64) -> usize {
        (64 - (key ^ self.last).leading_zeros()) as usize
    }

    fn push(&mut self, d: f64, v: u32) {
        let key = d.to_bits();
        debug_assert!(d >= 0.0 && key >= self.last, "a distance below the last");
        let b = self.bucket(key);
        self.buckets[b].push((key, v));
        self.len += 1;
    }

    fn pop(&mut self) -> Option<(f64, u32)> {
        if self.len == 0 {
            return None;
        }
        if self.buckets[0].is_empty() {
            // the lowest filled bucket's least key is the next to go; the
            // rest of that bucket all move to lower ones
            let i = (1..65)
                .find(|&i| !self.buckets[i].is_empty())
                .expect("a filled bucket");
            std::mem::swap(&mut self.spare, &mut self.buckets[i]);
            self.last = self.spare.iter().map(|e| e.0).min().expect("filled");
            for k in 0..self.spare.len() {
                let e = self.spare[k];
                let b = self.bucket(e.0);
                self.buckets[b].push(e);
            }
            self.spare.clear();
        }
        self.len -= 1;
        let (key, v) = self.buckets[0].pop().expect("bucket 0 filled");
        Some((f64::from_bits(key), v))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    use super::*;

    #[test]
    fn neighbours_are_ascending_and_distances_take_the_short_way() {
        // a square 0-1-2-3 with a long diagonal 0-2
        let g = WeightedGraph::new(
            4,
            &[[0, 1], [1, 2], [2, 3], [3, 0], [2, 0]],
            &[1.0, 1.0, 1.0, 1.0, 5.0],
        );
        assert_eq!(g.neighbors(0), &[1, 2, 3]);
        let all = vec![true; 4];
        assert_eq!(g.distances_within(&all, &[0]), vec![0.0, 1.0, 2.0, 1.0]);
        // with 1 and 3 left out, only the diagonal remains
        let some = vec![true, false, true, false];
        let d = g.distances_within(&some, &[0]);
        assert_eq!((d[0], d[2]), (0.0, 5.0));
        assert!(d[1].is_infinite());
        // two seeds: the nearer wins
        assert_eq!(g.distances_within(&all, &[0, 2]), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn components_come_by_their_smallest_vertex_each_ascending() {
        // 5-0-3 and 4-1, with 2 and 6 alone
        let g = WeightedGraph::new(7, &[[5, 0], [0, 3], [4, 1]], &[1.0, 1.0, 1.0]);
        let comps = g.components();
        assert_eq!(comps, vec![vec![0, 3, 5], vec![1, 4], vec![2], vec![6]]);
    }

    #[test]
    fn distances_into_a_buffer_touch_only_the_inside_vertices() {
        let g = WeightedGraph::new(
            4,
            &[[0, 1], [1, 2], [2, 3], [3, 0], [2, 0]],
            &[1.0, 1.0, 1.0, 1.0, 5.0],
        );
        let some = vec![true, false, true, false];
        let mut dist = vec![f64::INFINITY; 4];
        dist[1] = -7.0; // outside: left alone
        g.distances_within_into(&some, &[0], &mut dist);
        assert_eq!((dist[0], dist[1], dist[2]), (0.0, -7.0, 5.0));
        assert!(dist[3].is_infinite());
    }

    /// Dijkstra on a binary heap ordered by (distance, vertex).
    fn by_binary_heap(g: &WeightedGraph, inside: &[bool], seeds: &[u32]) -> Vec<f64> {
        #[derive(PartialEq)]
        struct Entry(f64, u32);
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .0
                    .total_cmp(&self.0)
                    .then_with(|| other.1.cmp(&self.1))
            }
        }
        let mut dist = vec![f64::INFINITY; g.n_vertices()];
        let mut heap = BinaryHeap::new();
        for &s in seeds {
            if inside[s as usize] && dist[s as usize] > 0.0 {
                dist[s as usize] = 0.0;
                heap.push(Entry(0.0, s));
            }
        }
        while let Some(Entry(d, v)) = heap.pop() {
            if d > dist[v as usize] {
                continue;
            }
            for (u, w) in g.weighted(v) {
                let nd = d + w;
                if inside[u as usize] && nd < dist[u as usize] {
                    dist[u as usize] = nd;
                    heap.push(Entry(nd, u));
                }
            }
        }
        dist
    }

    #[test]
    fn distances_are_what_a_binary_heap_finds() {
        // random edges, some of zero length, some of equal ones, over 20,000
        // vertices; a few seeds; a tenth of the vertices left out
        let mut state = 0x9b05_688c_2b3e_6c1fu64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n
        };
        let n = 20_000u32;
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for v in 1..n {
            for _ in 0..3 {
                let u = (v + n - 1 - next(40) as u32) % n;
                if u != v {
                    edges.push([u.min(v), u.max(v)]);
                    weights.push(match next(10) {
                        0 => 0.0,
                        1 => 1.0,
                        _ => next(1000) as f64 * 0.37 + 0.01,
                    });
                }
            }
        }
        let g = WeightedGraph::new(n as usize, &edges, &weights);
        let inside: Vec<bool> = (0..n).map(|_| next(10) > 0).collect();
        let seeds = [0, 7, 9_999, 19_000];
        let got = g.distances_within(&inside, &seeds);
        assert_eq!(got, by_binary_heap(&g, &inside, &seeds));
        assert!(got.iter().filter(|d| d.is_finite()).count() > 15_000);
    }
}
