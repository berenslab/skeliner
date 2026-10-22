//! An undirected, edge-weighted vertex graph and shortest paths over it.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

    /// Shortest-path distance to each vertex marked `inside` from the
    /// nearest of `seeds`, travelling only through `inside` vertices;
    /// infinity where none reaches.  Each distance is the path's weights
    /// added in order from the seed, as igraph's Dijkstra adds them.
    pub fn distances_within(&self, inside: &[bool], seeds: &[u32]) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.n_vertices()];
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
            for (u, w) in self.weighted(v) {
                if !inside[u as usize] {
                    continue;
                }
                let nd = d + w;
                if nd < dist[u as usize] {
                    dist[u as usize] = nd;
                    heap.push(Entry(nd, u));
                }
            }
        }
        dist
    }
}

/// A heap entry ordered so the smallest distance pops first.
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[cfg(test)]
mod tests {
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
}
