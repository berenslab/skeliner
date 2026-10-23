//! `skeliner.skeletonize`'s mesh side: geodesic shells over one neurite
//! (the preprocessing track) or over every component of a raw mesh (the
//! direct track), the ring tests that keep cross-sections apart from
//! fragments, the second pass's clean-up of the re-binned shells, the nodes
//! (centroid and radii) each bin becomes, and the direct track's merge of
//! nodes nested inside bigger ones.
//!
//! The surface is the reference's igraph graph: one vertex per mesh
//! vertex, one edge per unique mesh edge weighted by its length, every
//! neighbour list ascending — the order igraph's `incident` walks.

use std::collections::{HashMap, HashSet, VecDeque};

use meshcore::geometry::{arange, numpy_order, numpy_sum};
use meshcore::graph::WeightedGraph;
use meshcore::spatial::{KdTree, Sum};

/// A bin: vertex ids in the reference's order.
pub type Bin = Vec<u32>;

/// Where a bin sits: shell index, then index within the shell.
type Key = (usize, usize);

/// Insertion-ordered vote counts; the winner is the first key with the
/// most votes, as Python's `max(votes, key=votes.get)` picks it.
#[derive(Default)]
struct Votes(Vec<(Key, usize)>);

impl Votes {
    fn add(&mut self, k: Key) {
        match self.0.iter_mut().find(|(key, _)| *key == k) {
            Some((_, n)) => *n += 1,
            None => self.0.push((k, 1)),
        }
    }

    fn winner(&self) -> Option<Key> {
        self.0
            .iter()
            .copied()
            .reduce(|best, x| if x.1 > best.1 { x } else { best })
            .map(|(k, _)| k)
    }
}

/// `_geodesic_bins`: vertices into right-open shells of width `step` from
/// distance 0, the last shell closed.
fn geodesic_bins(vids: &[u32], dists: &[f64], step: f64) -> Vec<Vec<u32>> {
    if vids.is_empty() {
        return Vec::new();
    }
    let maxd = dists.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let mut edges = arange(0.0, maxd + step, step);
    if let Some(&last) = edges.last()
        && last <= maxd
    {
        edges.push(last + step);
    }
    let n = edges.len();
    let mut bins = vec![Vec::new(); n.saturating_sub(1)];
    for (&v, &d) in vids.iter().zip(dists) {
        // np.digitize(d, edges) - 1, NaN sorting past every edge
        let pos = if d.is_nan() {
            n
        } else {
            edges.partition_point(|&e| e <= d)
        };
        let mut idx = pos as i64 - 1;
        if idx == n as i64 - 1 {
            idx -= 1;
        }
        let idx = if idx < 0 {
            idx + bins.len() as i64
        } else {
            idx
        };
        bins[idx as usize].push(v);
    }
    bins
}

/// Connected pieces of `members` (ascending) in the graph, each ascending,
/// in order of their smallest vertex — igraph's `components()` of the
/// induced subgraph.
fn induced_components(
    g: &WeightedGraph,
    members: &[u32],
    mark: &mut [bool],
    seen: &mut [bool],
) -> Vec<Vec<u32>> {
    for &v in members {
        mark[v as usize] = true;
    }
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for &s in members {
        if seen[s as usize] {
            continue;
        }
        let mut comp = Vec::new();
        seen[s as usize] = true;
        stack.push(s);
        while let Some(v) = stack.pop() {
            comp.push(v);
            for &u in g.neighbors(v) {
                if mark[u as usize] && !seen[u as usize] {
                    seen[u as usize] = true;
                    stack.push(u);
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    for &v in members {
        mark[v as usize] = false;
        seen[v as usize] = false;
    }
    out
}

/// `_is_ring`: do the outside neighbours of `comp` (vertices marked in
/// `neurite` but not in it) fall into two or more connected groups?
fn is_ring(g: &WeightedGraph, comp: &[u32], neurite: &[bool]) -> bool {
    let inside: HashSet<u32> = comp.iter().copied().collect();
    let mut ext: HashSet<u32> = HashSet::new();
    for &v in comp {
        for &u in g.neighbors(v) {
            if !inside.contains(&u) && neurite[u as usize] {
                ext.insert(u);
            }
        }
    }
    if ext.len() < 2 {
        return false;
    }
    let mut visited: HashSet<u32> = HashSet::new();
    let mut groups = 0;
    let mut queue = VecDeque::new();
    for &start in &ext {
        if visited.contains(&start) {
            continue;
        }
        groups += 1;
        if groups >= 2 {
            return true;
        }
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            if !visited.insert(u) {
                continue;
            }
            for &w in g.neighbors(u) {
                if ext.contains(&w) && !visited.contains(&w) {
                    queue.push_back(w);
                }
            }
        }
    }
    groups >= 2
}

/// Merge each fragment into the ring its vertices touch most (first among
/// equals), in order; fragments touching no ring are dropped.
fn merge_fragments(g: &WeightedGraph, shells: &mut [Vec<Bin>], pending: Vec<Bin>) {
    if pending.is_empty() {
        return;
    }
    let mut owner: HashMap<u32, Key> = HashMap::new();
    for (bi, band) in shells.iter().enumerate() {
        for (ci, ring) in band.iter().enumerate() {
            for &v in ring {
                owner.insert(v, (bi, ci));
            }
        }
    }
    for frag in pending {
        let mut votes = Votes::default();
        for &v in &frag {
            for &u in g.neighbors(v) {
                if let Some(&k) = owner.get(&u) {
                    votes.add(k);
                }
            }
        }
        if let Some((bi, ci)) = votes.winner() {
            shells[bi][ci].extend_from_slice(&frag);
            for &v in &frag {
                owner.insert(v, (bi, ci));
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinParams {
    pub step_size: Option<f64>,
    pub target_shell_count: usize,
    pub min_shell_vertices: usize,
    pub max_shell_width_factor: f64,
}

/// Why a component cannot be binned (the reference raises `ValueError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinError {
    /// No seed lies in the component.
    NoSeed,
    /// A vertex no seed reaches, with a fixed shell width: numpy's
    /// `arange` up to an infinite distance.
    Unreachable,
}

/// Per-vertex work arrays over the whole graph, reused from one component to
/// the next; each use leaves them as it found them (all `false`, all
/// infinite).
pub struct Scratch {
    inside: Vec<bool>,
    mark: Vec<bool>,
    seen: Vec<bool>,
    dist: Vec<f64>,
}

impl Scratch {
    pub fn new(n: usize) -> Scratch {
        Scratch {
            inside: vec![false; n],
            mark: vec![false; n],
            seen: vec![false; n],
            dist: vec![f64::INFINITY; n],
        }
    }
}

/// `_bin_one_component` without soma exclusion or elongated-shell
/// splitting: geodesic shells from the nearest of `seeds` over `verts`
/// (ascending), each split into connected bins; rings stay, fragments and
/// bins under `min_shell_vertices` join the ring they touch most.
pub fn bin_one_component(
    g: &WeightedGraph,
    verts: &[u32],
    seeds: &[u32],
    mean_edge: f64,
    p: &BinParams,
) -> Result<Vec<Vec<Bin>>, BinError> {
    bin_one_component_in(
        g,
        verts,
        seeds,
        mean_edge,
        p,
        &mut Scratch::new(g.n_vertices()),
    )
}

/// [`bin_one_component`] with `scratch` sized to the graph.
fn bin_one_component_in(
    g: &WeightedGraph,
    verts: &[u32],
    seeds: &[u32],
    mean_edge: f64,
    p: &BinParams,
    scratch: &mut Scratch,
) -> Result<Vec<Vec<Bin>>, BinError> {
    for &v in verts {
        scratch.inside[v as usize] = true;
    }
    let out = bin_marked(g, verts, seeds, mean_edge, p, scratch);
    for &v in verts {
        scratch.inside[v as usize] = false;
        scratch.dist[v as usize] = f64::INFINITY;
    }
    out
}

/// The body of [`bin_one_component`], `verts` marked in `scratch.inside`.
fn bin_marked(
    g: &WeightedGraph,
    verts: &[u32],
    seeds: &[u32],
    mean_edge: f64,
    p: &BinParams,
    scratch: &mut Scratch,
) -> Result<Vec<Vec<Bin>>, BinError> {
    let inside = &scratch.inside;
    let roots: Vec<u32> = seeds
        .iter()
        .copied()
        .filter(|&s| inside[s as usize])
        .collect();
    if roots.is_empty() {
        return Err(BinError::NoSeed);
    }
    if verts.is_empty() {
        return Ok(Vec::new());
    }
    g.distances_within_into(inside, &roots, &mut scratch.dist);
    let dists: Vec<f64> = verts.iter().map(|&v| scratch.dist[v as usize]).collect();
    let mut step = match p.step_size {
        Some(s) => s,
        None => {
            let arc = dists.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (mean_edge * 2.0).max(arc / p.target_shell_count as f64)
        }
    };
    let mut shells: Vec<Vec<u32>> = Vec::new();
    while !shells.iter().any(|s| !s.is_empty()) && step < mean_edge * p.max_shell_width_factor {
        if dists.iter().any(|d| !d.is_finite()) {
            return Err(BinError::Unreachable);
        }
        shells = geodesic_bins(verts, &dists, step);
        step *= 1.5;
    }

    let mut out: Vec<Vec<Bin>> = Vec::new();
    let mut pending: Vec<Bin> = Vec::new();
    for shell in shells {
        if shell.is_empty() {
            continue;
        }
        let mut comps = Vec::new();
        for comp in induced_components(g, &shell, &mut scratch.mark, &mut scratch.seen) {
            if comp.len() < p.min_shell_vertices {
                pending.push(comp);
            } else {
                comps.push(comp);
            }
        }
        let (rings, frags): (Vec<Bin>, Vec<Bin>) = comps
            .into_iter()
            .partition(|c| is_ring(g, c, &scratch.inside));
        pending.extend(frags);
        if !rings.is_empty() {
            out.push(rings);
        }
    }
    merge_fragments(g, &mut out, pending);
    Ok(out)
}

/// `_bin_geodesic_shells` on the direct track's settings (no soma
/// exclusion, no elongated-shell split): every connected component of the
/// graph binned on its own, in igraph's component order, from `seed` in
/// the component that holds it and from the vertex at the component's
/// index modulo its size in the others (the reference's `hash(cid) %
/// len(verts)`); the shells of all components, in that order.
pub fn bin_mesh(
    g: &WeightedGraph,
    seed: u32,
    mean_edge: f64,
    p: &BinParams,
) -> Result<Vec<Vec<Bin>>, BinError> {
    let mut scratch = Scratch::new(g.n_vertices());
    let mut out = Vec::new();
    for (cid, verts) in g.components().into_iter().enumerate() {
        let start = if verts.binary_search(&seed).is_ok() {
            seed
        } else {
            verts[cid % verts.len()]
        };
        out.extend(bin_one_component_in(
            g,
            &verts,
            &[start],
            mean_edge,
            p,
            &mut scratch,
        )?);
    }
    Ok(out)
}

/// `_merge_nested_nodes`: node `j` collapses into node `i` when
/// `|c_i - c_j| + inside_frac * r_j <= r_i + tol`, and `i` takes `j`'s
/// vertices.  Nodes are visited biggest first, ties by index, node 0 never
/// when `keep_root`; each one's candidates are the nodes within
/// `r_i + max(r)`, by index.  Returns the kept mask, the kept nodes'
/// vertices in order, and each old node's new id (dropped ones take their
/// keeper's).
pub fn merge_nested_nodes(
    nodes: &[[f64; 3]],
    radii: &[f64],
    mut node2verts: Vec<Vec<u32>>,
    inside_frac: f64,
    keep_root: bool,
    tol: f64,
) -> (Vec<bool>, Vec<Vec<u32>>, Vec<i64>) {
    let n = nodes.len();
    let neg: Vec<f64> = radii.iter().map(|r| -r).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| numpy_order(&neg[a], &neg[b]));
    let biggest = radii.iter().copied().fold(f64::NEG_INFINITY, |m, r| {
        if m.is_nan() || r.is_nan() {
            f64::NAN
        } else {
            m.max(r)
        }
    });
    let tree = KdTree::new(nodes.to_vec(), Sum::Plain);
    let mut keep = vec![true; n];
    let mut old2new: Vec<i64> = (0..n as i64).collect();
    for i in order {
        if (keep_root && i == 0) || !keep[i] {
            continue;
        }
        for j in tree.within(&nodes[i], radii[i] + biggest) {
            let j = j as usize;
            if j == i || !keep[j] || radii[j] > radii[i] {
                continue;
            }
            let d = [
                nodes[i][0] - nodes[j][0],
                nodes[i][1] - nodes[j][1],
                nodes[i][2] - nodes[j][2],
            ];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if dist + inside_frac * radii[j] <= radii[i] + tol {
                let taken = std::mem::take(&mut node2verts[j]);
                node2verts[i].extend(taken);
                keep[j] = false;
                old2new[j] = old2new[i];
            }
        }
    }
    let kept = node2verts
        .into_iter()
        .zip(&keep)
        .filter_map(|(v, &k)| k.then_some(v))
        .collect();
    (keep, kept, old2new)
}

/// `_pieces`: connected pieces of `comp`, each in breadth-first order from
/// its lowest vertex, largest first (stable).
fn pieces(g: &WeightedGraph, comp: &[u32]) -> Vec<Vec<u32>> {
    let vset: HashSet<u32> = comp.iter().copied().collect();
    let mut starts: Vec<u32> = vset.iter().copied().collect();
    starts.sort_unstable();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut out: Vec<Vec<u32>> = Vec::new();
    let mut queue = VecDeque::new();
    for s in starts {
        if !seen.insert(s) {
            continue;
        }
        let mut grp = vec![s];
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            for &w in g.neighbors(u) {
                if vset.contains(&w) && seen.insert(w) {
                    queue.push_back(w);
                    grp.push(w);
                }
            }
        }
        out.push(grp);
    }
    out.sort_by_key(|p| std::cmp::Reverse(p.len()));
    out
}

/// `_neighbour_groups`: the vertices of other bins touching `comp`,
/// grouped by connectivity, groups touching a common bin merged.
fn neighbour_groups(
    g: &WeightedGraph,
    comp: &[u32],
    owner: &HashMap<u32, Key>,
    own: Key,
) -> Vec<Vec<u32>> {
    let mut ext: HashSet<u32> = HashSet::new();
    for &v in comp {
        for &u in g.neighbors(v) {
            if owner.get(&u).is_some_and(|&k| k != own) {
                ext.insert(u);
            }
        }
    }
    let mut starts: Vec<u32> = ext.iter().copied().collect();
    starts.sort_unstable();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut patches: Vec<Vec<u32>> = Vec::new();
    let mut queue = VecDeque::new();
    for s in starts {
        if !seen.insert(s) {
            continue;
        }
        let mut grp = vec![s];
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            for &w in g.neighbors(u) {
                if ext.contains(&w) && seen.insert(w) {
                    queue.push_back(w);
                    grp.push(w);
                }
            }
        }
        patches.push(grp);
    }
    let mut parent: Vec<usize> = (0..patches.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut first_seen: HashMap<Key, usize> = HashMap::new();
    for (pi, grp) in patches.iter().enumerate() {
        for v in grp {
            let other = *first_seen.entry(owner[v]).or_insert(pi);
            let (ra, rb) = (find(&mut parent, pi), find(&mut parent, other));
            if ra != rb {
                parent[rb] = ra;
            }
        }
    }
    let mut merged: Vec<(usize, Vec<u32>)> = Vec::new();
    for (pi, grp) in patches.iter().enumerate() {
        let r = find(&mut parent, pi);
        match merged.iter_mut().find(|(k, _)| *k == r) {
            Some((_, v)) => v.extend_from_slice(grp),
            None => merged.push((r, grp.clone())),
        }
    }
    merged.sort_by_key(|(k, _)| *k);
    merged.into_iter().map(|(_, v)| v).collect()
}

/// `_split_branch_band`: one piece per neighbouring group, grown from all
/// groups at once; vertices no group reaches join the largest piece.
fn split_branch_band(g: &WeightedGraph, comp: &[u32], groups: &[Vec<u32>]) -> Vec<Bin> {
    let vset: HashSet<u32> = comp.iter().copied().collect();
    let mut label: HashMap<u32, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    for (gi, grp) in groups.iter().enumerate() {
        let mut sorted = grp.clone();
        sorted.sort_unstable();
        for u in sorted {
            for &w in g.neighbors(u) {
                if vset.contains(&w) && !label.contains_key(&w) {
                    label.insert(w, gi);
                    queue.push_back(w);
                }
            }
        }
    }
    while let Some(u) = queue.pop_front() {
        let gi = label[&u];
        for &w in g.neighbors(u) {
            if vset.contains(&w) && !label.contains_key(&w) {
                label.insert(w, gi);
                queue.push_back(w);
            }
        }
    }
    let mut parts: Vec<Vec<u32>> = vec![Vec::new(); groups.len()];
    let mut orphans = Vec::new();
    for &v in comp {
        match label.get(&v) {
            Some(&gi) => parts[gi].push(v),
            None => orphans.push(v),
        }
    }
    let mut out: Vec<Bin> = parts.into_iter().filter(|p| !p.is_empty()).collect();
    if out.is_empty() {
        return vec![comp.to_vec()];
    }
    if !orphans.is_empty() {
        let mut big = 0;
        for i in 1..out.len() {
            if out[i].len() > out[big].len() {
                big = i;
            }
        }
        out[big].extend(orphans);
    }
    out
}

/// Each vertex's bin.
fn owners(shells: &[Vec<Bin>]) -> HashMap<u32, Key> {
    let mut owner = HashMap::new();
    for (bi, band) in shells.iter().enumerate() {
        for (ci, comp) in band.iter().enumerate() {
            for &v in comp {
                owner.insert(v, (bi, ci));
            }
        }
    }
    owner
}

/// The second pass's first clean-up of `_perpendicular_rebin`'s shells:
/// keep the rings (over the neurite's `verts`) and merge the fragments
/// into them.
pub fn recheck_rings(g: &WeightedGraph, rebinned: Vec<Vec<Bin>>, verts: &[u32]) -> Vec<Vec<Bin>> {
    let mut inside = vec![false; g.n_vertices()];
    for &v in verts {
        inside[v as usize] = true;
    }
    let mut shells: Vec<Vec<Bin>> = Vec::new();
    let mut pending: Vec<Bin> = Vec::new();
    for band in rebinned {
        let (rings, frags): (Vec<Bin>, Vec<Bin>) =
            band.into_iter().partition(|c| is_ring(g, c, &inside));
        pending.extend(frags);
        if !rings.is_empty() {
            shells.push(rings);
        }
    }
    merge_fragments(g, &mut shells, pending);
    shells
}

/// Make every bin one connected piece: its largest stays, strays go to the
/// bin they touch most (or stay if they touch none), up to eight rounds.
pub fn reunite_bins(g: &WeightedGraph, mut shells: Vec<Vec<Bin>>) -> Vec<Vec<Bin>> {
    let mut owner = owners(&shells);
    for _ in 0..8 {
        let mut changed = false;
        for bi in 0..shells.len() {
            for ci in 0..shells[bi].len() {
                if shells[bi][ci].len() < 2 {
                    continue;
                }
                let parts = pieces(g, &shells[bi][ci]);
                if parts.len() < 2 {
                    continue;
                }
                let mut parts = parts.into_iter();
                let first = parts.next().expect("two pieces");
                for &v in &first {
                    owner.insert(v, (bi, ci));
                }
                shells[bi][ci] = first;
                for stray in parts {
                    let mut votes = Votes::default();
                    for &v in &stray {
                        for &u in g.neighbors(v) {
                            if let Some(&k) = owner.get(&u)
                                && k != (bi, ci)
                            {
                                votes.add(k);
                            }
                        }
                    }
                    match votes.winner() {
                        None => shells[bi][ci].extend_from_slice(&stray),
                        Some((tb, tc)) => {
                            shells[tb][tc].extend_from_slice(&stray);
                            for &v in &stray {
                                owner.insert(v, (tb, tc));
                            }
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    shells
}

/// Split each bin touching three or more neighbourhoods into one piece per
/// tube, every decision against the ownership before any cut.
pub fn split_branch_bins(g: &WeightedGraph, mut shells: Vec<Vec<Bin>>) -> Vec<Vec<Bin>> {
    let snapshot = owners(&shells);
    let mut planned: Vec<(usize, usize, Vec<Bin>)> = Vec::new();
    for (bi, band) in shells.iter().enumerate() {
        for (ci, comp) in band.iter().enumerate() {
            if comp.len() < 3 {
                continue;
            }
            let groups = neighbour_groups(g, comp, &snapshot, (bi, ci));
            if groups.len() < 3 {
                continue;
            }
            let parts = split_branch_band(g, comp, &groups);
            if parts.len() > 1 {
                planned.push((bi, ci, parts));
            }
        }
    }
    for (bi, ci, parts) in planned {
        let mut parts = parts.into_iter();
        shells[bi][ci] = parts.next().expect("two parts");
        for extra in parts {
            shells[bi].push(extra);
        }
    }
    shells
}

/// `np.quantile(x, q)` (linear) of `x` sorted ascending.
fn quantile_sorted(x: &[f64], q: f64) -> f64 {
    let n = x.len();
    let v = (n - 1) as f64 * q;
    if v >= (n - 1) as f64 {
        return x[n - 1];
    }
    let prev = v.floor();
    let (a, b) = (x[prev as usize], x[prev as usize + 1]);
    let t = v - prev;
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

/// The trim fraction every `trim` radius uses (`skeliner._core.TRIM_FRACTION`).
pub const TRIM_FRACTION: f64 = 0.05;

/// `skeliner._core._estimate_radius` for one node's vertex distances.
pub fn estimate_radius(d: &[f64], method: &str) -> Option<f64> {
    let mean = |v: &[f64]| numpy_sum(v) / v.len() as f64;
    let mut sorted = d.to_vec();
    sorted.sort_by(numpy_order);
    let nan = d.iter().any(|v| v.is_nan());
    let n = d.len();
    let median = || {
        if nan {
            f64::NAN
        } else if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    };
    let quant = |q: f64| {
        if nan {
            f64::NAN
        } else {
            quantile_sorted(&sorted, q)
        }
    };
    let kept_mean = |keep: &dyn Fn(f64) -> bool| {
        let kept: Vec<f64> = d.iter().copied().filter(|&v| keep(v)).collect();
        if kept.is_empty() {
            mean(d)
        } else {
            mean(&kept)
        }
    };
    Some(match method {
        "median" => median(),
        "mean" => mean(d),
        "max" => sorted[n - 1],
        "min" => sorted[0],
        "percentile" => quant(0.8),
        "trim" => {
            let (lo, hi) = (quant(TRIM_FRACTION), quant(1.0 - TRIM_FRACTION));
            kept_mean(&|v| v >= lo && v <= hi)
        }
        "trimlow" => {
            let lo = quant(TRIM_FRACTION);
            kept_mean(&|v| v >= lo)
        }
        "trimhigh" => {
            let hi = quant(1.0 - TRIM_FRACTION);
            kept_mean(&|v| v <= hi)
        }
        _ => return None,
    })
}

/// Node centroids and, per estimator, one radius per node.
pub type Nodes = (Vec<[f64; 3]>, Vec<Vec<f64>>);

/// `_make_nodes` without the nested merge: each bin's centroid and, per
/// estimator, the radius of its vertices' distances to it; `None` for an
/// estimator name it does not know.
pub fn make_nodes(shells: &[Vec<Bin>], verts: &[[f64; 3]], estimators: &[String]) -> Option<Nodes> {
    let mut nodes = Vec::new();
    let mut radii: Vec<Vec<f64>> = vec![Vec::new(); estimators.len()];
    for bin in shells.iter().flatten() {
        let mut s = [0.0f64; 3];
        for &v in bin {
            let p = verts[v as usize];
            for k in 0..3 {
                s[k] += p[k];
            }
        }
        let n = bin.len() as f64;
        let c = s.map(|x| x / n);
        let d: Vec<f64> = bin
            .iter()
            .map(|&v| {
                let p = verts[v as usize];
                let e = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
            })
            .collect();
        for (est, r) in estimators.iter().zip(radii.iter_mut()) {
            r.push(estimate_radius(&d, est)?);
        }
        nodes.push(c);
    }
    Some((nodes, radii))
}

/// `_perpendicular_rebin` after the branch donations: re-assign each
/// chain's vertices (degree-2 runs between anchors) to the node nearest by
/// radius-weighted arc length along the chain.  `donated` gives, per
/// (branch node, neighbour), the branch vertices that face that neighbour
/// (numpy's projection decides them).  Returns each node's new vertices
/// and every assigned vertex's distance to the centreline.
pub fn rebin_chains(
    nodes: &[[f64; 3]],
    edges: &[[u32; 2]],
    node2verts: &[Vec<u32>],
    verts: &[[f64; 3]],
    r_ref: &[f64],
    donated: &HashMap<(u32, u32), Vec<u32>>,
) -> (Vec<Vec<u32>>, Vec<(u32, f64)>) {
    let n = nodes.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for e in edges {
        let (a, b) = (e[0], e[1]);
        if !adj[a as usize].contains(&b) {
            adj[a as usize].push(b);
        }
        if !adj[b as usize].contains(&a) {
            adj[b as usize].push(a);
        }
    }
    for list in &mut adj {
        list.sort_unstable();
    }
    let degree: Vec<usize> = adj.iter().map(Vec::len).collect();
    let is_anchor: Vec<bool> = degree.iter().map(|&d| d != 2).collect();

    // degree-2 paths between anchors
    let mut chains: Vec<Vec<u32>> = Vec::new();
    let mut visited: HashSet<(u32, u32)> = HashSet::new();
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    for anchor in (0..n as u32).filter(|&a| is_anchor[a as usize]) {
        for &nbr in &adj[anchor as usize] {
            if !visited.insert(key(anchor, nbr)) {
                continue;
            }
            let mut chain = vec![anchor, nbr];
            let (mut cur, mut prev) = (nbr, anchor);
            while !is_anchor[cur as usize] {
                let nexts: Vec<u32> = adj[cur as usize]
                    .iter()
                    .copied()
                    .filter(|&x| x != prev)
                    .collect();
                if nexts.len() != 1 {
                    break;
                }
                prev = cur;
                cur = nexts[0];
                visited.insert(key(prev, cur));
                chain.push(cur);
            }
            chains.push(chain);
        }
    }

    let positive: Vec<f64> = r_ref.iter().copied().filter(|&r| r > 0.0).collect();
    let r_floor = {
        let mut v = positive.clone();
        v.sort_by(numpy_order);
        let m = v.len();
        let median = if m == 0 {
            f64::NAN
        } else if m % 2 == 1 {
            v[m / 2]
        } else {
            (v[m / 2 - 1] + v[m / 2]) / 2.0
        };
        median * 0.01
    };
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut cl: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut assigned = vec![false; verts.len()];
    let norm = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    for chain in &chains {
        let last = chain.len() - 1;
        let mut ids: Vec<u32> = Vec::new();
        for (pos, &ni) in chain.iter().enumerate() {
            let k = if degree[ni as usize] >= 3 && (pos == 0 || pos == last) {
                Some((ni, if pos == 0 { chain[1] } else { chain[last - 1] }))
            } else {
                None
            };
            match k.and_then(|k| donated.get(&k)) {
                Some(d) => ids.extend_from_slice(d),
                None => ids.extend_from_slice(&node2verts[ni as usize]),
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids.retain(|&v| !assigned[v as usize]);
        if ids.is_empty() {
            continue;
        }
        let path: Vec<[f64; 3]> = chain.iter().map(|&c| nodes[c as usize]).collect();
        let seg_lens: Vec<f64> = path.windows(2).map(|w| norm(sub(w[1], w[0]))).collect();
        let mut arc_cum = vec![0.0f64];
        let mut acc = 0.0;
        for &l in &seg_lens {
            acc += l;
            arc_cum.push(acc);
        }
        let mut best_dist = vec![f64::INFINITY; ids.len()];
        let mut best_t = vec![0.0f64; ids.len()];
        for si in 0..chain.len() - 1 {
            let (p0, p1, s) = (path[si], path[si + 1], seg_lens[si]);
            if s < 1e-10 {
                continue;
            }
            let sd = sub(p1, p0).map(|c| c / s);
            for (k, &v) in ids.iter().enumerate() {
                let p = verts[v as usize];
                let df = sub(p, p0);
                let t = df[0] * sd[0] + df[1] * sd[1] + df[2] * sd[2];
                // np.clip(t, 0, s): maximum, then minimum, NaN kept
                let t = if t.is_nan() { t } else { t.max(0.0).min(s) };
                let q = [p0[0] + t * sd[0], p0[1] + t * sd[1], p0[2] + t * sd[2]];
                let d = norm(sub(p, q));
                if d < best_dist[k] {
                    best_dist[k] = d;
                    best_t[k] = arc_cum[si] + t;
                }
            }
        }
        let node_radii: Vec<f64> = chain
            .iter()
            .map(|&c| {
                let r = r_ref[c as usize];
                if r_floor > r { r_floor } else { r }
            })
            .collect();
        for (k, &v) in ids.iter().enumerate() {
            let mut best = 0;
            let mut best_w = f64::NAN;
            for (ci, (&arc, &rad)) in arc_cum.iter().zip(&node_radii).enumerate() {
                let w = (best_t[k] - arc).abs() / rad;
                if ci == 0 || w < best_w || (w.is_nan() && !best_w.is_nan()) {
                    best = ci;
                    best_w = w;
                }
            }
            let ni = chain[best] as usize;
            out[ni].push(v);
            cl[ni].push(best_dist[k]);
            assigned[v as usize] = true;
        }
    }
    for ni in (0..n).filter(|&i| is_anchor[i]) {
        for &v in &node2verts[ni] {
            if !assigned[v as usize] {
                out[ni].push(v);
                assigned[v as usize] = true;
            }
        }
    }
    let mut dists: Vec<(u32, f64)> = Vec::new();
    for ni in 0..n {
        for (&v, &d) in out[ni].iter().zip(&cl[ni]) {
            dists.push((v, d));
        }
    }
    (out, dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_are_right_open_with_the_last_closed() {
        let bins = geodesic_bins(&[0, 1, 2, 3], &[0.0, 0.5, 1.0, 2.0], 1.0);
        assert_eq!(bins, vec![vec![0, 1], vec![2], vec![3]]);
    }

    #[test]
    fn radii_follow_numpy() {
        let d = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(estimate_radius(&d, "median"), Some(3.0));
        assert_eq!(estimate_radius(&d, "mean"), Some(22.0));
        // quantiles 1.2 and 80.8: 2, 3, 4 kept
        assert_eq!(estimate_radius(&d, "trim"), Some(3.0));
        assert_eq!(estimate_radius(&d, "nope"), None);
    }

    /// A tube of `rings` rings of `around` vertices, each joined to the
    /// next by a ladder with diagonals.
    fn tube(rings: u32, around: u32) -> WeightedGraph {
        let id = |r: u32, k: u32| r * around + (k % around);
        let mut edges = Vec::new();
        for r in 0..rings {
            for k in 0..around {
                edges.push([id(r, k), id(r, k + 1)]);
                if r + 1 < rings {
                    edges.push([id(r, k), id(r + 1, k)]);
                    edges.push([id(r, k + 1), id(r + 1, k)]);
                }
            }
        }
        let edges: Vec<[u32; 2]> = edges
            .into_iter()
            .map(|[a, b]| [a.min(b), a.max(b)])
            .collect();
        let w = vec![1.0; edges.len()];
        WeightedGraph::new((rings * around) as usize, &edges, &w)
    }

    #[test]
    fn a_tube_bins_into_rings_from_its_end() {
        let g = tube(20, 8);
        let verts: Vec<u32> = (0..160).collect();
        let seeds: Vec<u32> = (0..8).collect();
        let p = BinParams {
            step_size: Some(2.0),
            target_shell_count: 500,
            min_shell_vertices: 6,
            max_shell_width_factor: 50.0,
        };
        let shells = bin_one_component(&g, &verts, &seeds, 1.0, &p).unwrap();
        assert!(shells.len() >= 8, "{}", shells.len());
        assert!(shells.iter().all(|s| s.len() == 1));
        let half: Vec<u32> = (0..80).collect();
        assert_eq!(
            bin_one_component(&g, &half, &[150], 1.0, &p).err(),
            Some(BinError::NoSeed)
        );
    }

    #[test]
    fn a_mesh_bins_each_component_from_its_own_seed() {
        // two tubes (0..160, 160..320), a lone vertex (320), a triangle
        let one = tube(20, 8);
        let mut edges = Vec::new();
        for v in 0..160u32 {
            for &u in one.neighbors(v) {
                if v < u {
                    edges.push([v, u]);
                    edges.push([v + 160, u + 160]);
                }
            }
        }
        edges.extend([[321, 322], [322, 323], [321, 323]]);
        let w = vec![1.0; edges.len()];
        let g = WeightedGraph::new(324, &edges, &w);
        let p = BinParams {
            step_size: Some(2.0),
            target_shell_count: 500,
            min_shell_vertices: 6,
            max_shell_width_factor: 50.0,
        };
        // the seed is in the second tube; the others start at the vertex
        // at their component index modulo their size
        let got = bin_mesh(&g, 170, 1.0, &p).unwrap();
        let tube_a: Vec<u32> = (0..160).collect();
        let tube_b: Vec<u32> = (160..320).collect();
        let mut want = bin_one_component(&g, &tube_a, &[0], 1.0, &p).unwrap();
        want.extend(bin_one_component(&g, &tube_b, &[170], 1.0, &p).unwrap());
        want.extend(bin_one_component(&g, &[320], &[320], 1.0, &p).unwrap());
        want.extend(bin_one_component(&g, &[321, 322, 323], &[321], 1.0, &p).unwrap());
        assert!(want.len() >= 16, "{}", want.len());
        assert_eq!(got, want);
    }

    #[test]
    fn nested_nodes_fold_into_the_bigger_one_biggest_first() {
        let nodes = [
            [0.0, 0.0, 0.0],   // 0: the root
            [100.0, 0.0, 0.0], // 1
            [102.0, 0.0, 0.0], // 2: inside 1
            [200.0, 0.0, 0.0], // 3
            [200.0, 0.0, 0.0], // 4: the same as 3; 3 comes first
            [0.5, 0.0, 0.0],   // 5: near the root, which never absorbs
            [0.0, 0.0, 0.0],   // 6: bigger than the root, so it takes it
        ];
        let radii = [1.0, 10.0, 1.0, 5.0, 5.0, 0.5, 3.0];
        let verts: Vec<Vec<u32>> = (0..7).map(|k| vec![10 * k, 10 * k + 1]).collect();
        let (keep, kept, old2new) = merge_nested_nodes(&nodes, &radii, verts, 0.8, true, 1e-6);
        assert_eq!(keep, vec![false, true, false, true, false, false, true]);
        assert_eq!(
            kept,
            vec![
                vec![10, 11, 20, 21],
                vec![30, 31, 40, 41],
                vec![60, 61, 0, 1, 50, 51],
            ]
        );
        assert_eq!(old2new, vec![6, 1, 1, 3, 3, 6, 6]);
    }
}
