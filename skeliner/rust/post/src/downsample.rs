//! `post.downsample`: runs of degree-2 nodes whose radii agree merged into
//! single nodes, the anchors between them kept.

use std::cmp::Ordering;
use std::collections::HashSet;

use mesh::geometry::{numpy_order, numpy_sum};

/// How a merged node's radii are aggregated from its nodes'.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregate {
    Median,
    Mean,
    /// The root mean square: the mean cross-section kept.
    Area,
}

impl Aggregate {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "median" => Some(Self::Median),
            "mean" => Some(Self::Mean),
            "area" => Some(Self::Area),
            _ => None,
        }
    }

    /// As numpy computes `median`, `mean` and `sqrt(mean(v * v))`.
    fn of(self, vals: &[f64]) -> f64 {
        let n = vals.len() as f64;
        match self {
            Self::Median => {
                let mut s = vals.to_vec();
                s.sort_by(numpy_order);
                let m = s.len();
                if m == 0 || s[m - 1].is_nan() {
                    f64::NAN
                } else if m % 2 == 1 {
                    (0.0 + s[m / 2]) / 1.0
                } else {
                    (0.0 + s[m / 2 - 1] + s[m / 2]) / 2.0
                }
            }
            Self::Mean => numpy_sum(vals) / n,
            Self::Area => {
                let sq: Vec<f64> = vals.iter().map(|x| x * x).collect();
                (numpy_sum(&sq) / n).sqrt()
            }
        }
    }
}

/// `downsample`'s settings.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub rtol: f64,
    pub atol: f64,
    pub aggregate: Aggregate,
    pub merge_endpoints: bool,
    pub slide_branchpoints: bool,
    pub max_anchor_shift: Option<f64>,
}

/// A downsampled skeleton's arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct Downsampled {
    pub nodes: Vec<[f64; 3]>,
    /// One array per radius estimator, one radius per new node.
    pub radii: Vec<Vec<f64>>,
    /// The new nodes' bins: an anchor's as it was, a merged node's sorted.
    pub node2verts: Option<Vec<Vec<usize>>>,
    /// The new edges, lower node first, in the order they were made.
    pub edges: Vec<[usize; 2]>,
    /// Each old node's new one, −1 for none.
    pub old2new: Vec<i64>,
}

/// Python's `max(a, b)`: `a` unless `b > a`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// The length-weighted centroid of the polylines `lines`: segment midpoints
/// weighted by segment length, summed as numpy sums them; the mean of the
/// `members` when a length is not finite or every one is zero; the one
/// member's position when there is one.
fn centroid(nodes: &[[f64; 3]], lines: &[Vec<usize>], members: &[usize]) -> [f64; 3] {
    if members.len() <= 1 {
        return nodes[members[0]];
    }
    let mut lengths = Vec::new();
    let mut mids = Vec::new();
    for line in lines {
        for w in line.windows(2) {
            let (p, q) = (nodes[w[0]], nodes[w[1]]);
            let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            lengths.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            mids.push([
                0.5 * (q[0] + p[0]),
                0.5 * (q[1] + p[1]),
                0.5 * (q[2] + p[2]),
            ]);
        }
    }
    let mut s = [0.0f64; 3];
    if !lengths.iter().all(|l| l.is_finite()) || lengths.iter().all(|&l| l == 0.0) {
        for &v in members {
            for k in 0..3 {
                s[k] += nodes[v][k];
            }
        }
        let m = members.len() as f64;
        return s.map(|x| x / m);
    }
    for (mid, &l) in mids.iter().zip(&lengths) {
        for k in 0..3 {
            s[k] += mid[k] * l;
        }
    }
    let total = numpy_sum(&lengths);
    s.map(|x| x / total)
}

/// Greedy 1D segmentation along a path: a new group starts when the next
/// radius deviates from the group's running mean beyond tolerance.
fn partition_by_radius(ids: &[usize], r: &[f64], rtol: f64, atol: f64) -> Vec<Vec<usize>> {
    let Some((&first, rest)) = ids.split_first() else {
        return Vec::new();
    };
    let mut groups = Vec::new();
    let mut cur = vec![first];
    let mut r_ref = r[first];
    for &nid in rest {
        let ri = r[nid];
        let tol = atol + rtol * py_max(r_ref.abs(), ri.abs());
        if (ri - r_ref).abs() <= tol {
            cur.push(nid);
            r_ref += (ri - r_ref) / cur.len() as f64;
        } else {
            groups.push(std::mem::replace(&mut cur, vec![nid]));
            r_ref = ri;
        }
    }
    groups.push(cur);
    groups
}

struct Builder<'a> {
    nodes: &'a [[f64; 3]],
    radii: &'a [Vec<f64>],
    bins: Option<&'a [Vec<usize>]>,
    key: usize,
    s: Settings,
    deg: Vec<usize>,
    out: Downsampled,
    /// An anchor's old nodes, in aggregate order, once it has absorbed.
    held: Vec<Option<Vec<usize>>>,
    /// The runs it absorbed.
    runs: Vec<Vec<Vec<usize>>>,
}

/// The union of the bins of `ids`, ascending.
fn merged_bins(bins: &[Vec<usize>], ids: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = ids.iter().flat_map(|&j| bins[j].iter().copied()).collect();
    v.sort_unstable();
    v.dedup();
    v
}

impl Builder<'_> {
    fn add_anchor(&mut self, oid: usize) -> usize {
        if self.out.old2new[oid] >= 0 {
            return self.out.old2new[oid] as usize;
        }
        let nid = self.out.nodes.len();
        self.out.nodes.push(self.nodes[oid]);
        for (k, r) in self.radii.iter().enumerate() {
            self.out.radii[k].push(r[oid]);
        }
        if let (Some(out), Some(bins)) = (self.out.node2verts.as_mut(), self.bins) {
            out.push(bins[oid].clone());
        }
        self.held.push(None);
        self.runs.push(Vec::new());
        self.out.old2new[oid] = nid as i64;
        nid
    }

    fn add_group(&mut self, gids: &[usize]) -> usize {
        let nid = self.out.nodes.len();
        let pos = centroid(self.nodes, &[gids.to_vec()], gids);
        self.out.nodes.push(pos);
        for (k, r) in self.radii.iter().enumerate() {
            let vals: Vec<f64> = gids.iter().map(|&j| r[j]).collect();
            self.out.radii[k].push(self.s.aggregate.of(&vals));
        }
        if let (Some(bins), Some(out)) = (self.bins, self.out.node2verts.as_mut()) {
            out.push(merged_bins(bins, gids));
        }
        self.held.push(None);
        self.runs.push(Vec::new());
        for &j in gids {
            self.out.old2new[j] = nid as i64;
        }
        nid
    }

    fn within_tol(&self, ra: f64, rb: f64) -> bool {
        let tol = self.s.atol + self.s.rtol * py_max(ra.abs(), rb.abs());
        (ra - rb).abs() <= tol
    }

    fn absorbs(&self, oid: usize) -> bool {
        oid != 0
            && ((self.s.merge_endpoints && self.deg[oid] == 1)
                || (self.s.slide_branchpoints && self.deg[oid] >= 3))
    }

    /// `group` into the anchor `nid` (old node `oid`), unless its radius is
    /// out of tolerance or that moves the anchor more than
    /// `max_anchor_shift`.
    fn absorb(&mut self, nid: usize, oid: usize, group: &[usize], anchor_first: bool) -> bool {
        let r = &self.radii[self.key];
        let rg: Vec<f64> = group.iter().map(|&j| r[j]).collect();
        if !self.within_tol(r[oid], self.s.aggregate.of(&rg)) {
            return false;
        }
        let line: Vec<usize> = if anchor_first {
            std::iter::once(oid).chain(group.iter().copied()).collect()
        } else {
            group.iter().copied().chain(std::iter::once(oid)).collect()
        };
        let members = match &self.held[nid] {
            None => line.clone(),
            Some(held) => held.iter().chain(group).copied().collect(),
        };
        let mut lines = self.runs[nid].clone();
        lines.push(line);
        let pos = centroid(self.nodes, &lines, &members);
        if let Some(shift) = self.s.max_anchor_shift {
            let p = self.nodes[oid];
            let d = [pos[0] - p[0], pos[1] - p[1], pos[2] - p[2]];
            let dist = (0.0 + d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            // not within the shift, or either is NaN
            if !matches!(
                dist.partial_cmp(&shift),
                Some(Ordering::Less | Ordering::Equal)
            ) {
                return false;
            }
        }
        self.out.nodes[nid] = pos;
        for (k, r) in self.radii.iter().enumerate() {
            let vals: Vec<f64> = members.iter().map(|&j| r[j]).collect();
            self.out.radii[k][nid] = self.s.aggregate.of(&vals);
        }
        if let (Some(bins), Some(out)) = (self.bins, self.out.node2verts.as_mut()) {
            out[nid] = merged_bins(bins, &members);
        }
        for &j in group {
            self.out.old2new[j] = nid as i64;
        }
        self.held[nid] = Some(members);
        self.runs[nid] = lines;
        true
    }
}

/// `downsample` on a skeleton's `nodes`, `radii` (one array per estimator;
/// `key` decides), `edges` and bins (`node2verts`, when it has them).
///
/// Every run of degree-2 nodes between anchors — node 0 and the nodes of
/// another degree, ascending, each one's neighbours ascending; then the
/// lowest node of each ring with none — splits greedily into groups whose
/// radii stay within `atol + rtol · max(|r|)` of their running mean, each
/// group one node at its length-weighted centroid with aggregated radii and
/// the union of its bins. A leaf (with `merge_endpoints`) or a branch point
/// (with `slide_branchpoints`), node 0 never, absorbs the group beside it
/// when that group's radius is within tolerance of its own and it moves no
/// more than `max_anchor_shift`; an anchor absorbing from several runs
/// takes them all — the centroid of every absorbed run, the radii of all
/// its nodes, their bins. A node with no other node to join stays as it is.
pub fn downsample(
    nodes: &[[f64; 3]],
    radii: &[Vec<f64>],
    key: usize,
    edges: &[[usize; 2]],
    node2verts: Option<&[Vec<usize>]>,
    s: Settings,
) -> Downsampled {
    let n = nodes.len();
    let mut deg = vec![0usize; n];
    let mut adj = vec![Vec::new(); n];
    for &[a, b] in edges {
        deg[a] += 1;
        deg[b] += 1;
        if a != b {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    for l in &mut adj {
        l.sort_unstable();
        l.dedup();
    }
    let mut b = Builder {
        nodes,
        radii,
        bins: node2verts,
        key,
        s,
        deg,
        out: Downsampled {
            nodes: Vec::new(),
            radii: vec![Vec::new(); radii.len()],
            node2verts: node2verts.map(|_| Vec::new()),
            edges: Vec::new(),
            old2new: vec![-1; n],
        },
        held: Vec::new(),
        runs: Vec::new(),
    };
    let mut visited: HashSet<[usize; 2]> = HashSet::new();
    let pair = |a: usize, c: usize| [a.min(c), a.max(c)];

    let mut walk_from = |b: &mut Builder<'_>, a: usize| {
        for &first in &adj[a] {
            if !visited.insert(pair(a, first)) {
                continue;
            }
            let mut path = vec![a];
            let (mut prev, mut cur) = (a, first);
            while b.deg[cur] == 2 && cur != a {
                path.push(cur);
                let mut nxts = adj[cur].iter().copied().filter(|&x| x != prev);
                let (Some(nxt), None) = (nxts.next(), nxts.next()) else {
                    break;
                };
                visited.insert(pair(cur, nxt));
                (prev, cur) = (cur, nxt);
            }
            if path.last() != Some(&cur) {
                path.push(cur);
            }
            let (a_id, z_id) = (path[0], path[path.len() - 1]);
            let left = b.add_anchor(a_id);
            let right = b.add_anchor(z_id);
            let r = &b.radii[b.key];
            let mut groups = partition_by_radius(&path[1..path.len() - 1], r, s.rtol, s.atol);
            if !groups.is_empty() && b.absorbs(a_id) && b.absorb(left, a_id, &groups[0], true) {
                groups.remove(0);
            }
            if let Some(last) = groups.last()
                && b.absorbs(z_id)
                && b.absorb(right, z_id, last, false)
            {
                groups.pop();
            }
            let mut l = left;
            for grp in &groups {
                let g = b.add_group(grp);
                b.out.edges.push(pair(l, g));
                l = g;
            }
            if l != right {
                b.out.edges.push(pair(l, right));
            }
        }
    };

    let anchors: Vec<usize> = (0..n).filter(|&v| v == 0 || b.deg[v] != 2).collect();
    for a in anchors {
        walk_from(&mut b, a);
    }
    for (v, nbrs) in adj.iter().enumerate() {
        if b.out.old2new[v] < 0 && b.deg[v] == 2 && !nbrs.is_empty() {
            walk_from(&mut b, v);
        }
    }
    for (v, nbrs) in adj.iter().enumerate() {
        if b.out.old2new[v] < 0 && nbrs.is_empty() {
            b.add_anchor(v);
        }
    }
    b.out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize) -> Vec<[f64; 3]> {
        (0..n).map(|i| [i as f64, 0.0, 0.0]).collect()
    }

    fn settings() -> Settings {
        Settings {
            rtol: 0.05,
            atol: 0.0,
            aggregate: Aggregate::Area,
            merge_endpoints: true,
            slide_branchpoints: true,
            max_anchor_shift: None,
        }
    }

    #[test]
    fn a_run_of_equal_radii_merges_into_the_leaf() {
        // 0-1-2-3-4: one group 1..3, absorbed by leaf 4
        let edges = [[0, 1], [1, 2], [2, 3], [3, 4]];
        let d = downsample(&line(5), &[vec![1.0; 5]], 0, &edges, None, settings());
        assert_eq!(d.nodes.len(), 2);
        assert_eq!(d.old2new, [0, 1, 1, 1, 1]);
        assert_eq!(d.edges, [[0, 1]]);
        assert_eq!(d.nodes[1], [2.5, 0.0, 0.0]);
    }

    #[test]
    fn rings_and_loops_end() {
        // a ring through node 0, a ring with no anchor, a loop through a
        // branch point whose run collapses into it, a node with a self-loop
        let edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]];
        let d = downsample(&line(6), &[vec![1.0; 6]], 0, &edges, None, settings());
        assert!(d.old2new.iter().all(|&v| v >= 0));
        let edges = [[0, 1], [1, 2], [2, 3], [3, 1], [4, 4]];
        let d = downsample(&line(5), &[vec![1.0; 5]], 0, &edges, None, settings());
        assert!(d.old2new.iter().all(|&v| v >= 0));
        assert!(d.edges.iter().all(|&[a, c]| a != c));
    }
}
