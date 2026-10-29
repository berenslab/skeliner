//! A skeleton's edges as an undirected multigraph, walked the way igraph
//! walks one: `sk.dx`'s graph checks and walks, in igraph's orders, so they
//! give what igraph gives on cycles, forests, loops and repeated edges too.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use rayon::prelude::*;

/// An undirected multigraph on `n` nodes. Each node's edges come in igraph's
/// incidence order: by the node at the other end, ascending, edges between
/// the same two nodes by descending id, a self-loop once.
pub struct Graph {
    /// Each edge's ends, lower first: igraph's `TO` and `FROM`.
    ends: Vec<[usize; 2]>,
    ptr: Vec<usize>,
    inc: Vec<usize>,
}

impl Graph {
    pub fn new(n: usize, edges: &[[usize; 2]]) -> Self {
        let ends: Vec<[usize; 2]> = edges.iter().map(|&[a, b]| [a.min(b), a.max(b)]).collect();
        let mut ptr = vec![0usize; n + 1];
        for &[a, b] in &ends {
            ptr[a + 1] += 1;
            if a != b {
                ptr[b + 1] += 1;
            }
        }
        for v in 0..n {
            ptr[v + 1] += ptr[v];
        }
        let mut fill = ptr[..n].to_vec();
        let mut inc = vec![0usize; ptr[n]];
        for (e, &[a, b]) in ends.iter().enumerate() {
            inc[fill[a]] = e;
            fill[a] += 1;
            if a != b {
                inc[fill[b]] = e;
                fill[b] += 1;
            }
        }
        for v in 0..n {
            inc[ptr[v]..ptr[v + 1]].sort_unstable_by(|&e, &f| {
                other(ends[e], v).cmp(&other(ends[f], v)).then(f.cmp(&e))
            });
        }
        Self { ends, ptr, inc }
    }

    pub fn len(&self) -> usize {
        self.ptr.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn incident(&self, v: usize) -> &[usize] {
        &self.inc[self.ptr[v]..self.ptr[v + 1]]
    }

    fn other(&self, e: usize, v: usize) -> usize {
        other(self.ends[e], v)
    }

    /// Each node's degree, a self-loop counted twice.
    pub fn degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.len()];
        for &[a, b] in &self.ends {
            deg[a] += 1;
            deg[b] += 1;
        }
        deg
    }

    /// igraph's `bfs` from `root`: each node's parent, −1 for `root` and −2
    /// for a node it does not reach.
    pub fn bfs_parents(&self, root: usize) -> Vec<i64> {
        let mut parent = vec![-2i64; self.len()];
        parent[root] = -1;
        let mut queue = VecDeque::from([root]);
        while let Some(v) = queue.pop_front() {
            for &e in self.incident(v) {
                let u = self.other(e, v);
                if parent[u] == -2 {
                    parent[u] = v as i64;
                    queue.push_back(u);
                }
            }
        }
        parent
    }

    /// Hops from `source` to every node, infinite where it does not reach.
    pub fn hops(&self, source: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.len()];
        dist[source] = 0.0;
        let mut queue = VecDeque::from([source]);
        while let Some(v) = queue.pop_front() {
            for &e in self.incident(v) {
                let u = self.other(e, v);
                if dist[u] == f64::INFINITY {
                    dist[u] = dist[v] + 1.0;
                    queue.push_back(u);
                }
            }
        }
        dist
    }

    /// Each node's component, numbered as igraph numbers them: in the order
    /// of their lowest node.
    pub fn components(&self) -> Vec<usize> {
        let mut label = vec![usize::MAX; self.len()];
        let mut next = 0;
        for start in 0..self.len() {
            if label[start] != usize::MAX {
                continue;
            }
            label[start] = next;
            let mut queue = VecDeque::from([start]);
            while let Some(v) = queue.pop_front() {
                for &e in self.incident(v) {
                    let u = self.other(e, v);
                    if label[u] == usize::MAX {
                        label[u] = next;
                        queue.push_back(u);
                    }
                }
            }
            next += 1;
        }
        label
    }

    /// The length of the shortest path from each of `sources` to `target`
    /// over the edges' `weights`, summed from the source as igraph's
    /// Dijkstra sums it (an infinite edge never taken); infinite where there
    /// is none. Sources in parallel.
    pub fn path_lengths(&self, weights: &[f64], sources: &[usize], target: usize) -> Vec<f64> {
        let one = |&source: &usize| -> f64 {
            let mut dist = vec![f64::INFINITY; self.len()];
            let mut done = vec![false; self.len()];
            let mut heap = BinaryHeap::from([Reached(0.0, source)]);
            dist[source] = 0.0;
            while let Some(Reached(d, v)) = heap.pop() {
                if done[v] {
                    continue;
                }
                if v == target {
                    return d;
                }
                done[v] = true;
                for &e in self.incident(v) {
                    let w = weights[e];
                    if w == f64::INFINITY {
                        continue;
                    }
                    let u = self.other(e, v);
                    let alt = d + w;
                    if !done[u] && alt < dist[u] {
                        dist[u] = alt;
                        heap.push(Reached(alt, u));
                    }
                }
            }
            f64::INFINITY
        };
        threads::install(|| sources.par_iter().map(one).collect())
    }

    /// igraph's `minimum_cycle_basis` (Horton's candidates, no cutoff): each
    /// cycle's edge ids in the order they run round it, shortest cycles
    /// first.
    pub fn minimum_cycle_basis(&self) -> Vec<Vec<usize>> {
        let n = self.len();
        let deg = self.degrees();
        let n_comp = self.components().iter().map(|&c| c + 1).max().unwrap_or(0);
        let rank = self.ends.len() + n_comp - n;

        let mut candidates: Vec<Vec<usize>> = Vec::new();
        let mut visited = vec![0usize; n];
        let mut mark = 0;
        for i in 0..n {
            let seen = !visited[i].is_multiple_of(3);
            if deg[i] <= 1 || (seen && deg[i] < 3) {
                continue;
            }
            self.fundamental_cycles(i, &mut visited, mark, &mut candidates);
            mark += 3;
        }
        for c in &mut candidates {
            c.sort_unstable();
        }
        candidates.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        candidates.dedup();

        let mut basis = Vec::new();
        let mut reduced: Vec<Vec<usize>> = Vec::new();
        for cycle in candidates {
            if reduced.len() == rank {
                break;
            }
            if let Some(row) = eliminate(&reduced, &cycle) {
                let at = reduced.partition_point(|r| r[0] < row[0]);
                reduced.insert(at, row);
                basis.push(cycle);
            }
        }
        basis.iter().map(|c| self.order_cycle(c)).collect()
    }

    /// `igraph_i_fundamental_cycles_bfs`: the cycles a breadth-first tree
    /// from `start` closes, appended to `out`; `visited` marks a node seen
    /// with `mark + 1` and done with `mark + 2`.
    fn fundamental_cycles(
        &self,
        start: usize,
        visited: &mut [usize],
        mark: usize,
        out: &mut Vec<Vec<usize>>,
    ) {
        let mut pred = vec![usize::MAX; self.len()];
        let mut queue = VecDeque::from([start]);
        visited[start] = mark + 1;
        while let Some(v) = queue.pop_front() {
            for &e in self.incident(v) {
                if e == pred[v] {
                    continue;
                }
                let u = self.other(e, v);
                if visited[u] == mark + 2 {
                    continue;
                }
                if visited[u] == mark + 1 {
                    // up the tree from both ends to where they meet
                    let (mut up, mut vp) = (u, v);
                    let mut u_back = Vec::new();
                    let mut v_back = vec![e];
                    while up != vp {
                        u_back.push(pred[up]);
                        up = self.other(pred[up], up);
                        if up == vp {
                            break;
                        }
                        v_back.push(pred[vp]);
                        vp = self.other(pred[vp], vp);
                    }
                    v_back.extend(u_back.iter().rev());
                    out.push(v_back);
                } else {
                    queue.push_back(u);
                    visited[u] = mark + 1;
                    pred[u] = e;
                }
            }
            visited[v] = mark + 2;
        }
    }

    /// `igraph_i_order_cycle`: a cycle's edges (ascending ids) in the order
    /// they run round it, from the first edge's higher end.
    fn order_cycle(&self, cycle: &[usize]) -> Vec<usize> {
        let mut at: HashMap<usize, [Option<usize>; 2]> = HashMap::new();
        for &e in cycle {
            let [to, from] = self.ends[e];
            for v in [from, to] {
                let p = at.entry(v).or_insert([None, None]);
                if p[0].is_none() {
                    p[0] = Some(e);
                } else {
                    p[1] = Some(e);
                }
            }
        }
        let mut out = Vec::with_capacity(cycle.len());
        let mut e = cycle[0];
        let mut v = self.ends[e][1];
        for _ in 0..cycle.len() {
            out.push(e);
            let p = at[&v];
            e = if p[0] == Some(e) { p[1] } else { p[0] }.expect("a cycle");
            let [to, from] = self.ends[e];
            v = if from == v { to } else { from };
        }
        out
    }

    /// `branches_of_length`: every maximal path whose ends have a degree
    /// other than 2 (node 0 always an end) and whose inner nodes have
    /// degree 2, kept when it has `k` nodes (`k` inner nodes without
    /// `include_endpoints`); walked from the ends in ascending order, each
    /// end's edges in turn, a pair of nodes walked once.
    pub fn branches(&self, k: usize, include_endpoints: bool) -> Vec<Vec<usize>> {
        let deg = self.degrees();
        let is_end = |v: usize| v == 0 || deg[v] != 2;
        let mut walked = HashSet::new();
        let mut out = Vec::new();
        for end in (0..self.len()).filter(|&v| is_end(v)) {
            for &e in self.incident(end) {
                let first = self.other(e, end);
                if walked.contains(&pair(end, first)) {
                    continue;
                }
                let mut path = vec![end];
                let (mut prev, mut cur) = (end, first);
                loop {
                    path.push(cur);
                    walked.insert(pair(prev, cur));
                    if is_end(cur) {
                        break;
                    }
                    let next = self
                        .incident(cur)
                        .iter()
                        .map(|&f| self.other(f, cur))
                        .find(|&w| w != prev);
                    match next {
                        Some(w) => (prev, cur) = (cur, w),
                        None => break,
                    }
                }
                let size = if include_endpoints {
                    Some(path.len())
                } else {
                    path.len().checked_sub(2)
                };
                if size == Some(k) {
                    out.push(path);
                }
            }
        }
        out
    }

    /// `twigs_of_length`: from every leaf (degree 1, node 0 aside),
    /// ascending, the chain up the breadth-first tree from node 0 through
    /// nodes of degree 2, up to the first node that is not (node 0 always
    /// ends it); kept when it has `k` nodes, proximal first, that node in
    /// front with `include_branching_node`. A leaf node 0 does not reach
    /// has none.
    pub fn twigs(&self, k: usize, include_branching_node: bool) -> Vec<Vec<usize>> {
        if self.is_empty() {
            return Vec::new();
        }
        let deg = self.degrees();
        let parent = self.bfs_parents(0);
        let mut out = Vec::new();
        for leaf in (1..self.len()).filter(|&v| deg[v] == 1) {
            let mut chain = vec![leaf];
            let mut cur = leaf;
            while parent[cur] >= 0 {
                let par = parent[cur] as usize;
                if deg[par] == 2 && par != 0 {
                    chain.push(par);
                    cur = par;
                    continue;
                }
                if chain.len() == k {
                    if include_branching_node {
                        chain.push(par);
                    }
                    chain.reverse();
                    out.push(chain);
                }
                break;
            }
        }
        out
    }

    /// Each node's children with every edge pointing away from node 0, in
    /// the order the edges come: an edge between two nodes as many hops
    /// from node 0, or two it does not reach, points nowhere.
    fn distal_children(&self) -> Vec<Vec<usize>> {
        let hops = self.hops(0);
        let mut children = vec![Vec::new(); self.len()];
        for &[a, b] in &self.ends {
            let (da, db) = (hops[a], hops[b]);
            if da == db {
                continue;
            }
            let (parent, child) = if da < db { (a, b) } else { (b, a) };
            children[parent].push(child);
        }
        children
    }

    /// `extract_neurites`: `root` and everything distal to it, from node 0,
    /// ascending (a node reached twice, on a cycle, listed twice).
    pub fn neurite(&self, root: usize, include_root: bool) -> Vec<usize> {
        let mut out = subtree(&self.distal_children(), root, include_root);
        out.sort_unstable();
        out
    }

    /// `neurites_out_of_bounds`: the nodes distal to (and with
    /// `include_root`, at) each first node outside the box `[lo, hi]` on
    /// the way out from node 0 — a node whose breadth-first parent is
    /// inside, or that has none — ascending, once each.
    pub fn neurites_out_of_bounds(
        &self,
        nodes: &[[f64; 3]],
        lo: [f64; 3],
        hi: [f64; 3],
        include_root: bool,
    ) -> Vec<usize> {
        let outside: Vec<bool> = nodes
            .iter()
            .map(|p| (0..3).any(|k| p[k] < lo[k] || p[k] > hi[k]))
            .collect();
        if !outside.contains(&true) {
            return Vec::new();
        }
        let parent = self.bfs_parents(0);
        let children = self.distal_children();
        let mut target = vec![false; self.len()];
        for v in (0..self.len()).filter(|&v| outside[v]) {
            if parent[v] >= 0 && outside[parent[v] as usize] {
                continue;
            }
            for w in subtree(&children, v, include_root) {
                target[w] = true;
            }
        }
        (0..self.len()).filter(|&v| target[v]).collect()
    }
}

fn other([lo, hi]: [usize; 2], v: usize) -> usize {
    if lo == v { hi } else { lo }
}

fn pair(a: usize, b: usize) -> [usize; 2] {
    [a.min(b), a.max(b)]
}

/// `root` and every node below it in `children`, depth first.
fn subtree(children: &[Vec<usize>], root: usize, include_root: bool) -> Vec<usize> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        if v != root || include_root {
            out.push(v);
        }
        stack.extend_from_slice(&children[v]);
    }
    out
}

/// `gaussian_elimination` over GF(2) on sorted edge-id vectors, the rows of
/// `reduced` in order of their first id: `cycle` reduced against them, or
/// `None` when they span it.
fn eliminate(reduced: &[Vec<usize>], cycle: &[usize]) -> Option<Vec<usize>> {
    let mut work = cycle.to_vec();
    for row in reduced {
        match row[0].cmp(&work[0]) {
            Ordering::Less => continue,
            Ordering::Greater => break,
            Ordering::Equal => {
                work = symmetric_difference(row, &work);
                if work.is_empty() {
                    return None;
                }
            }
        }
    }
    Some(work)
}

fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// A node reached at a distance, ordered for a min-heap.
#[derive(PartialEq)]
struct Reached(f64, usize);

impl Eq for Reached {}

impl PartialOrd for Reached {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Reached {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incidence_follows_igraph() {
        // igraph 1.0, checked: node 1's incident edges [8, 3, 6, 6, 4, 1, 0,
        // 5, 7] with loops twice, so [8, 3, 6, 4, 1, 0, 5, 7] with loops once
        let edges = [
            [2, 1],
            [1, 2],
            [3, 3],
            [0, 1],
            [1, 2],
            [3, 1],
            [1, 1],
            [4, 1],
            [0, 1],
        ];
        let g = Graph::new(5, &edges);
        assert_eq!(g.incident(0), [8, 3]);
        assert_eq!(g.incident(1), [8, 3, 6, 4, 1, 0, 5, 7]);
        assert_eq!(g.incident(2), [4, 1, 0]);
        assert_eq!(g.incident(3), [5, 2]);
        assert_eq!(g.degrees(), [2, 9, 3, 3, 1]);
        assert_eq!(g.bfs_parents(0), [-1, 0, 1, 1, 1]);
    }

    #[test]
    fn a_square_with_a_tail() {
        // 0-1-2-3-0 and 3-4: one cycle; 4's twig of one node
        let g = Graph::new(5, &[[0, 1], [1, 2], [2, 3], [3, 0], [3, 4]]);
        assert_eq!(g.minimum_cycle_basis(), vec![vec![0, 1, 2, 3]]);
        assert_eq!(g.twigs(1, true), vec![vec![3, 4]]);
        assert_eq!(g.hops(0), [0.0, 1.0, 2.0, 1.0, 2.0]);
        let forest = Graph::new(4, &[[0, 1], [2, 3]]);
        assert_eq!(forest.components(), [0, 0, 1, 1]);
        assert_eq!(forest.bfs_parents(0), [-1, 0, -2, -2]);
        assert!(forest.twigs(1, false).iter().all(|t| t == &[1]));
        let lengths = forest.path_lengths(&[1.5, 2.0], &[1, 3], 0);
        assert_eq!(lengths, [1.5, f64::INFINITY]);
    }
}
