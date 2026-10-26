//! Organelle detection steps.

/// `skeliner.pre.find_organelles`' enclosure pass.
///
/// `np_mask` marks the candidate faces; `adjacency` is trimesh's manifold
/// face adjacency, in its order, which fixes the depth-first order.  A
/// subtree hanging off an articulation face is claimed when its boundary
/// has some edge, at least 90% of it pocket, and its size is between
/// `min_cluster_size` and the pocket's.  Returns the pocket with the
/// claimed faces added, and how many were added.
pub fn enclosure_pass(
    np_mask: &[bool],
    adjacency: &[[u32; 2]],
    pocket: &[bool],
    min_cluster_size: usize,
) -> (Vec<bool>, usize) {
    let n = np_mask.len();
    let mut new_pocket = pocket.to_vec();
    if !np_mask.iter().any(|&b| b) {
        return (new_pocket, 0);
    }

    // Boundary edges per candidate face, total and towards pocket.
    let mut tbnd = vec![0i64; n];
    let mut pbnd = vec![0i64; n];
    let mut pairs: Vec<[u32; 2]> = Vec::new();
    for &[a, b] in adjacency {
        let (ia, ib) = (a as usize, b as usize);
        match (np_mask[ia], np_mask[ib]) {
            (true, true) => pairs.push([a, b]),
            (true, false) => {
                tbnd[ia] += 1;
                pbnd[ia] += i64::from(pocket[ib]);
            }
            (false, true) => {
                tbnd[ib] += 1;
                pbnd[ib] += i64::from(pocket[ia]);
            }
            (false, false) => {}
        }
    }

    // CSR over both directions: each face's targets in the order of the
    // edge list (pairs as given, then reversed).
    let mut off = vec![0usize; n + 1];
    for &[a, b] in &pairs {
        off[a as usize + 1] += 1;
        off[b as usize + 1] += 1;
    }
    for i in 0..n {
        off[i + 1] += off[i];
    }
    let mut fill = off.clone();
    let mut dst = vec![0u32; off[n]];
    let forward = pairs.iter().map(|&[a, b]| (a, b));
    let backward = pairs.iter().map(|&[a, b]| (b, a));
    for (s, t) in forward.chain(backward) {
        dst[fill[s as usize]] = t;
        fill[s as usize] += 1;
    }

    // Iterative Tarjan with subtree accumulation, in 32-bit words (a face's
    // discovery time, subtree size and boundary counts are under 3n) to
    // keep the walk's arrays small.
    let pocket_n = pocket.iter().filter(|&&p| p).count() as i64;
    let narrow = |x: Vec<i64>| -> Vec<i32> { x.into_iter().map(|v| v as i32).collect() };
    let mut disc = vec![-1i32; n];
    let mut low = vec![-1i32; n];
    let mut sub_sz = vec![1i32; n];
    let mut sub_pb = narrow(pbnd);
    let mut sub_tb = narrow(tbnd);
    let mut children = vec![0u32; n];
    let mut tmr = 0i32;
    let mut claims: Vec<(usize, i32)> = Vec::new();
    const NO_PARENT: u32 = u32::MAX;
    for root in (0..n).filter(|&f| np_mask[f]) {
        if disc[root] >= 0 {
            continue;
        }
        disc[root] = tmr;
        low[root] = tmr;
        tmr += 1;
        // (node, parent, next edge)
        let mut stack: Vec<(u32, u32, u32)> = vec![(root as u32, NO_PARENT, off[root] as u32)];
        while let Some(&(u, pu, ei)) = stack.last() {
            let (u, ei) = (u as usize, ei as usize);
            if ei < off[u + 1] {
                stack.last_mut().expect("non-empty").2 = ei as u32 + 1;
                let v = dst[ei] as usize;
                if disc[v] < 0 {
                    disc[v] = tmr;
                    low[v] = tmr;
                    tmr += 1;
                    stack.push((v as u32, u as u32, off[v] as u32));
                } else if v as u32 != pu && low[u] > disc[v] {
                    low[u] = disc[v];
                }
                continue;
            }
            stack.pop();
            let Some(&(p, pp, _)) = stack.last() else {
                break;
            };
            let p = p as usize;
            low[p] = low[p].min(low[u]);
            sub_sz[p] += sub_sz[u];
            sub_pb[p] += sub_pb[u];
            sub_tb[p] += sub_tb[u];
            children[p] += 1;
            let is_root = pp == NO_PARENT;
            if low[u] >= disc[p] && (!is_root || children[p] > 1) {
                let (sz, tb, pb) = (sub_sz[u], sub_tb[u], sub_pb[u]);
                if tb > 0
                    && f64::from(pb) >= f64::from(tb) * 0.9
                    && sz as usize >= min_cluster_size
                    && i64::from(sz) <= pocket_n
                {
                    claims.push((u, disc[u]));
                }
            }
        }
    }

    // Each claim is the subtree discovered from its face onwards.
    let mut added = 0usize;
    let mut stamp = vec![usize::MAX; n];
    let mut queue = std::collections::VecDeque::new();
    for (k, &(u, du)) in claims.iter().enumerate() {
        queue.push_back(u);
        while let Some(c) = queue.pop_front() {
            if stamp[c] == k {
                continue;
            }
            stamp[c] = k;
            if !new_pocket[c] {
                new_pocket[c] = true;
                added += 1;
            }
            for &nb in &dst[off[c]..off[c + 1]] {
                let nb = nb as usize;
                if stamp[nb] != k && disc[nb] >= du {
                    queue.push_back(nb);
                }
            }
        }
    }
    (new_pocket, added)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Faces on a path 0-1-2-3-4 hang a bag {5, 6} off face 2 through the
    /// single link 2-5, and the bag's only other neighbours are pocket.
    fn bag() -> (Vec<bool>, Vec<[u32; 2]>, Vec<bool>) {
        let np: Vec<bool> = (0..9).map(|f| f < 7).collect();
        let adjacency = vec![
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [2, 5],
            [5, 6],
            [5, 7], // pocket
            [6, 8], // pocket
        ];
        let pocket: Vec<bool> = (0..9).map(|f| f >= 7).collect();
        (np, adjacency, pocket)
    }

    #[test]
    fn a_pocket_walled_bag_behind_a_neck_is_claimed() {
        let (np, adjacency, pocket) = bag();
        let (new, added) = enclosure_pass(&np, &adjacency, &pocket, 2);
        assert_eq!(added, 2);
        assert!(new[5] && new[6]);
        assert!(!new[0] && !new[2]);
    }

    #[test]
    fn a_bag_smaller_than_min_cluster_size_is_left() {
        let (np, adjacency, pocket) = bag();
        let (new, added) = enclosure_pass(&np, &adjacency, &pocket, 3);
        assert_eq!(added, 0);
        assert_eq!(new, pocket);
    }

    #[test]
    fn a_bag_mostly_walled_by_non_pocket_is_left() {
        let (np, adjacency, mut pocket) = bag();
        pocket[8] = false; // half the bag's boundary is now ordinary surface
        let (_, added) = enclosure_pass(&np, &adjacency, &pocket, 1);
        assert_eq!(added, 0);
    }

    #[test]
    fn a_bag_bigger_than_the_pocket_is_left() {
        // Both boundary edges reach the one pocket face 7: walled, but the
        // bag of two outweighs a pocket of one.  (At size 1, face 6 alone
        // would still qualify, so the minimum is 2.)
        let (np, mut adjacency, mut pocket) = bag();
        pocket[8] = false;
        *adjacency.last_mut().unwrap() = [6, 7];
        let (_, added) = enclosure_pass(&np, &adjacency, &pocket, 2);
        assert_eq!(added, 0);
        pocket[8] = true;
        *adjacency.last_mut().unwrap() = [6, 8];
        assert_eq!(enclosure_pass(&np, &adjacency, &pocket, 2).1, 2);
    }

    #[test]
    fn no_candidates_changes_nothing() {
        let (_, adjacency, pocket) = bag();
        let (new, added) = enclosure_pass(&[false; 9], &adjacency, &pocket, 1);
        assert_eq!((new, added), (pocket, 0));
    }
}

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};

use mesh::Topology;
use mesh::components::LowestRootSets;
use mesh::geometry::numpy_sum;
use mesh::spatial::{KdTree, Sum};
use rayon::prelude::*;

use crate::fits::{Svd, centred, hull_area_on};

/// The arrays organelle detection reads about a mesh.
pub struct Surface<'a> {
    pub topo: &'a Topology,
    pub faces: &'a [[u32; 3]],
    pub verts: &'a [[f64; 3]],
    pub centers: &'a [[f64; 3]],
    pub normals: &'a [[f64; 3]],
    pub areas: &'a [f64],
    pub dots: &'a [f64],
}

/// The sorted vertex pair of each of a face's edges, in the order
/// `(0, 1), (1, 2), (2, 0)`.
fn face_edge_keys(f: &[u32; 3]) -> [[u32; 2]; 3] {
    let key = |a: u32, b: u32| if a < b { [a, b] } else { [b, a] };
    [key(f[0], f[1]), key(f[1], f[2]), key(f[2], f[0])]
}

/// Faces on edge `e`, or none if no face has it.
fn faces_on<'a>(s: &Surface<'a>, e: [u32; 2]) -> &'a [u32] {
    s.topo
        .edge_id(e[0], e[1])
        .map_or(&[], |id| s.topo.edge_faces(id))
}

/// Area sum the way the reference takes it: numpy's sum over the faces in
/// ascending order.
fn area_of(s: &Surface<'_>, faces: &BTreeSet<u32>) -> f64 {
    let a: Vec<f64> = faces.iter().map(|&f| s.areas[f as usize]).collect();
    numpy_sum(&a)
}

/// Edge-connected components of the faces where `inside`, by lowest face,
/// each listing its faces in the order a breadth-first search from its
/// lowest face meets them; with each face's component (its lowest face,
/// `u32::MAX` outside).  The components are searched at once.
fn components_where(
    s: &Surface<'_>,
    inside: impl Fn(usize) -> bool + Sync,
) -> (Vec<Vec<u32>>, Vec<u32>) {
    threads::install(|| {
        let n = s.faces.len();
        let is_in: Vec<bool> = (0..n).into_par_iter().map(&inside).collect();
        let roots = lowest_faces(s, &is_in);
        let lowest: Vec<u32> = (0..n as u32).filter(|&f| roots[f as usize] == f).collect();
        // a face lies in one component, so only that one's search marks it
        let seen: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
        let comps = lowest
            .par_iter()
            .map(|&seed| {
                let mut comp = Vec::new();
                let mut queue = VecDeque::from([seed]);
                seen[seed as usize].store(true, Ordering::Relaxed);
                while let Some(f) = queue.pop_front() {
                    comp.push(f);
                    for g in s.topo.face_neighbors(f as usize) {
                        if is_in[g as usize] && !seen[g as usize].load(Ordering::Relaxed) {
                            seen[g as usize].store(true, Ordering::Relaxed);
                            queue.push_back(g);
                        }
                    }
                }
                comp
            })
            .collect();
        (comps, roots)
    })
}

/// Each face's edge-connected component among those `is_in` marks, as the
/// component's lowest face; `u32::MAX` outside.
fn lowest_faces(s: &Surface<'_>, is_in: &[bool]) -> Vec<u32> {
    let n = s.faces.len();
    let sets = LowestRootSets::new(n);
    (0..s.topo.n_edges()).into_par_iter().for_each(|e| {
        let mut on = s
            .topo
            .edge_faces(e)
            .iter()
            .copied()
            .filter(|&f| is_in[f as usize]);
        if let Some(first) = on.next() {
            for f in on {
                sets.join(first, f);
            }
        }
    });
    (0..n as u32)
        .into_par_iter()
        .map(|f| {
            if is_in[f as usize] {
                sets.root(f)
            } else {
                u32::MAX
            }
        })
        .collect()
}

/// [`components_where`]'s components with each one's faces ascending: the
/// same faces, the components in the same order (by lowest face), for the
/// passes that read no order inside one.
fn components_where_unordered(
    s: &Surface<'_>,
    inside: impl Fn(usize) -> bool + Sync,
) -> Vec<Vec<u32>> {
    let n = s.faces.len();
    let is_in: Vec<bool> = (0..n).into_par_iter().map(&inside).collect();
    let roots = lowest_faces(s, &is_in);
    let mut slot = vec![usize::MAX; n];
    let mut comps: Vec<Vec<u32>> = Vec::new();
    for f in (0..n).filter(|&f| is_in[f]) {
        let r = roots[f] as usize;
        if slot[r] == usize::MAX {
            slot[r] = comps.len();
            comps.push(Vec::new());
        }
        comps[slot[r]].push(f as u32);
    }
    comps
}

/// What [`isolated_organelles`] found.
#[derive(Debug, Clone, Default)]
pub struct Isolated {
    pub mask: Vec<bool>,
    pub n_internal: usize,
    pub n_internal_faces: usize,
    pub n_kept: usize,
    pub n_via_dots: usize,
    pub n_via_geometry: usize,
}

/// A mesh's faces and vertices with each face's centre, normal and outward
/// dot: what [`isolated_organelles`] reads.
pub struct Fragments<'a> {
    pub faces: &'a [[u32; 3]],
    pub verts: &'a [[f64; 3]],
    pub centers: &'a [[f64; 3]],
    pub normals: &'a [[f64; 3]],
    pub dots: &'a [f64],
}

/// `skeliner.pre.find_isolated_organelles`: components other than `main`
/// that point inward — by a majority of their non-zero outward dots, or,
/// when all are zero, by most vertices lying behind the nearest main face.
pub fn isolated_organelles(s: &Fragments<'_>, face_comp: &[i64], main: i64) -> Isolated {
    let n = s.faces.len();
    let n_comps = face_comp
        .iter()
        .copied()
        .max()
        .map_or(0, |m| (m + 1).max(0) as usize);
    // the other components' faces; main's are only searched
    let mut by_comp: Vec<Vec<usize>> = vec![Vec::new(); n_comps];
    for (fi, &c) in face_comp.iter().enumerate() {
        if c >= 0 && c != main {
            by_comp[c as usize].push(fi);
        }
    }
    let main_faces: Vec<usize> = threads::install(|| {
        (0..n)
            .into_par_iter()
            .filter(|&f| face_comp[f] == main)
            .collect()
    });
    let others: Vec<&Vec<usize>> = by_comp
        .iter()
        .enumerate()
        .filter(|&(ci, fis)| ci as i64 != main && !fis.is_empty())
        .map(|(_, fis)| fis)
        .collect();
    let by_dots = |fis: &[usize]| fis.iter().any(|&f| s.dots[f] != 0.0);
    // the main faces' tree, for components whose outward dots are all zero
    let tree = others.iter().any(|fis| !by_dots(fis)).then(|| {
        let centers = threads::install(|| main_faces.par_iter().map(|&f| s.centers[f]).collect());
        KdTree::new(centers, Sum::Plain)
    });
    // each component: whether it points inward, and whether by its dots
    let decided: Vec<(bool, bool)> = threads::install(|| {
        others
            .par_iter()
            .map(|fis| {
                let n_valid = fis.iter().filter(|&&f| s.dots[f] != 0.0).count();
                if n_valid > 0 {
                    let neg = fis.iter().filter(|&&f| s.dots[f] < 0.0).count();
                    return (neg as f64 > n_valid as f64 / 2.0, true);
                }
                let tree = tree.as_ref().expect("built for these");
                let mut vids: Vec<u32> = fis.iter().flat_map(|&f| s.faces[f]).collect();
                vids.sort_unstable();
                vids.dedup();
                let mut neg = 0usize;
                for &v in &vids {
                    let p = s.verts[v as usize];
                    let (j, _) = tree.nearest(&p).expect("main has faces");
                    let f = main_faces[j as usize];
                    let (c, nr) = (s.centers[f], s.normals[f]);
                    let d = (p[0] - c[0]) * nr[0] + (p[1] - c[1]) * nr[1] + (p[2] - c[2]) * nr[2];
                    neg += usize::from(d < 0.0);
                }
                (neg as f64 > vids.len() as f64 / 2.0, false)
            })
            .collect()
    });
    let mut out = Isolated {
        mask: vec![false; n],
        ..Default::default()
    };
    for (fis, &(internal, via_dots)) in others.iter().zip(&decided) {
        if via_dots {
            out.n_via_dots += 1;
        } else {
            out.n_via_geometry += 1;
        }
        if internal {
            out.n_internal += 1;
            out.n_internal_faces += fis.len();
            for &f in fis.iter() {
                out.mask[f] = true;
            }
        } else {
            out.n_kept += 1;
        }
    }
    out
}

/// Faces of the components with at least `min_cluster_size` faces outside
/// `isolated` (every label counts, negative ones included), and among them
/// those not in `main`.
pub fn structural_masks(
    face_comp: &[i64],
    main: i64,
    isolated: &[bool],
    min_cluster_size: usize,
) -> (Vec<bool>, Vec<bool>, usize) {
    // per label, offset by the lowest one, as labels go negative
    let lowest = face_comp.iter().copied().min().unwrap_or(0);
    let highest = face_comp.iter().copied().max().unwrap_or(0);
    let span = (highest - lowest + 1).max(0) as usize;
    let (mut non_iso, mut present) = (vec![0usize; span], vec![false; span]);
    for (fi, &c) in face_comp.iter().enumerate() {
        non_iso[(c - lowest) as usize] += usize::from(!isolated[fi]);
        present[(c - lowest) as usize] = true;
    }
    let structural: Vec<bool> = non_iso
        .iter()
        .zip(&present)
        .map(|(&n, &p)| p && n >= min_cluster_size)
        .collect();
    let n_structural = structural.iter().filter(|&&s| s).count();
    let mask: Vec<bool> = face_comp
        .par_iter()
        .map(|&c| structural[(c - lowest) as usize])
        .collect();
    let non_main: Vec<bool> = face_comp
        .par_iter()
        .map(|&c| c != main && structural[(c - lowest) as usize])
        .collect();
    (mask, non_main, n_structural)
}

/// The simple closed loops among `edges`, each loop's points in order
/// round it and centred on their mean: what `skeliner.pre._rim_enclosed_area`
/// fits a plane to.
pub fn rim_loops(edges: &[[u32; 2]], verts: &[[f64; 3]]) -> Vec<Vec<[f64; 3]>> {
    let mut edges_at: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        edges_at.entry(e[0]).or_default().push(i);
        edges_at.entry(e[1]).or_default().push(i);
    }
    let mut visited = vec![false; edges.len()];
    let mut loops = Vec::new();
    for start in 0..edges.len() {
        if visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut queue = VecDeque::from([start]);
        while let Some(ei) = queue.pop_front() {
            if visited[ei] {
                continue;
            }
            visited[ei] = true;
            comp.push(ei);
            for v in edges[ei] {
                for &nei in &edges_at[&v] {
                    if !visited[nei] {
                        queue.push_back(nei);
                    }
                }
            }
        }
        let mut degree: HashMap<u32, usize> = HashMap::new();
        for &ei in &comp {
            for v in edges[ei] {
                *degree.entry(v).or_default() += 1;
            }
        }
        if degree.values().any(|&d| d != 2) {
            continue;
        }
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        for &ei in &comp {
            let [a, b] = edges[ei];
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        let mut ordered = vec![edges[comp[0]][0]];
        while ordered.len() < comp.len() {
            let curr = ordered[ordered.len() - 1];
            let prev = (ordered.len() > 1).then(|| ordered[ordered.len() - 2]);
            match adj[&curr].iter().find(|&&nb| Some(nb) != prev) {
                Some(&nb) => ordered.push(nb),
                None => break,
            }
        }
        if ordered.len() < 3 {
            continue;
        }
        let pts: Vec<[f64; 3]> = ordered.iter().map(|&v| verts[v as usize]).collect();
        let n_pts = pts.len() as f64;
        let mut mean = [0.0f64; 3];
        for p in &pts {
            for k in 0..3 {
                mean[k] += p[k];
            }
        }
        let mean = mean.map(|m| m / n_pts);
        loops.push(
            pts.iter()
                .map(|p| [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]])
                .collect(),
        );
    }
    loops
}

/// `skeliner.pre._rim_enclosed_area`: the planar area enclosed by the
/// [`rim_loops`], each projected onto the plane of the first two of its
/// right-singular vectors `vts`.
pub fn rim_area(loops: &[Vec<[f64; 3]>], vts: &[[[f64; 3]; 3]]) -> f64 {
    let mut total = 0.0;
    for (centred, vh) in loops.iter().zip(vts) {
        let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let p2: Vec<[f64; 2]> = centred
            .iter()
            .map(|c| [dot(c, &vh[0]), dot(c, &vh[1])])
            .collect();
        let n = p2.len();
        let terms: Vec<f64> = (0..n - 1)
            .map(|i| p2[i][0] * p2[i + 1][1] - p2[i + 1][0] * p2[i][1])
            .collect();
        let s = numpy_sum(&terms) + p2[n - 1][0] * p2[0][1] - p2[0][0] * p2[n - 1][1];
        total += 0.5 * s.abs();
    }
    total
}

/// Knobs of pocket detection, as `skeliner.pre.find_pocket_organelles`
/// names them.
#[derive(Debug, Clone, Copy)]
pub struct PocketParams {
    pub grow_threshold: f64,
    pub bridge_threshold: f64,
    pub max_hole_size: usize,
    pub hole_enclosure_ratio: f64,
    pub min_pocket_size: usize,
    pub min_fold_ratio: f64,
    pub min_cluster_size: usize,
}

impl Default for PocketParams {
    fn default() -> Self {
        Self {
            grow_threshold: 0.1,
            bridge_threshold: 0.3,
            max_hole_size: 500,
            hole_enclosure_ratio: 0.5,
            min_pocket_size: 5,
            min_fold_ratio: 3.0,
            min_cluster_size: 5,
        }
    }
}

/// `skeliner.pre.find_pocket_mouths`: the boundary edges of each cluster
/// of negative-dot faces on `main_mask` that folds inward through a small
/// opening (pocket area over opening area at least `min_fold_ratio`).
pub fn pocket_mouths(
    s: &Surface<'_>,
    main_mask: &[bool],
    non_main: Option<&[bool]>,
    params: &PocketParams,
    svd: &dyn Svd,
) -> Vec<Vec<[u32; 2]>> {
    let neg = |f: usize| s.dots[f] < 0.0 && main_mask[f];
    let (clusters, of) = components_where(s, neg);
    // each cluster big enough: its boundary edges (each where first met in
    // the cluster's order), its area and its rim's loops, all at once
    struct Rim {
        first: u32,
        boundary: Vec<[u32; 2]>,
        area: f64,
        loops: Vec<Vec<[f64; 3]>>,
    }
    let rims: Vec<Rim> = threads::install(|| {
        clusters
            .par_iter()
            .filter(|c| c.len() >= params.min_pocket_size)
            .filter_map(|cluster| {
                let first = cluster[0];
                let member = |f: u32| of[f as usize] == first;
                let mut boundary: Vec<[u32; 2]> = Vec::new();
                for &f in cluster {
                    let ids = s.topo.face_edges(f as usize);
                    for (k, e) in face_edge_keys(&s.faces[f as usize]).into_iter().enumerate() {
                        let on = match ids {
                            // the keys' column pairs are the topology's
                            Some(ids) => s.topo.edge_faces(ids[k] as usize),
                            None => faces_on(s, e),
                        };
                        if on.len() == 2 {
                            let other = if member(on[0]) { on[1] } else { on[0] };
                            if !member(other) {
                                boundary.push(e);
                            }
                        }
                    }
                }
                // an edge met twice is a face's repeated vertex pair
                let mut met = HashSet::with_capacity(boundary.len());
                boundary.retain(|e| met.insert(*e));
                if boundary.is_empty() {
                    return None;
                }
                let mut members = cluster.clone();
                members.sort_unstable();
                let areas: Vec<f64> = members.iter().map(|&f| s.areas[f as usize]).collect();
                Some(Rim {
                    first,
                    area: numpy_sum(&areas),
                    loops: rim_loops(&boundary, s.verts),
                    boundary,
                })
            })
            .collect()
    });

    // the rims' planes, every loop's at once
    let loops: Vec<&[[f64; 3]]> = rims
        .iter()
        .flat_map(|r| r.loops.iter().map(Vec::as_slice))
        .collect();
    let vts = svd.right_singular_each(&loops, true);
    let mut at = 0;
    let mut openings: Vec<f64> = rims
        .iter()
        .map(|r| {
            at += r.loops.len();
            rim_area(&r.loops, &vts[at - r.loops.len()..at])
        })
        .collect();
    // no rim encloses anything: the boundary's hull instead
    let hulls: Vec<(usize, Vec<[f64; 3]>)> = rims
        .iter()
        .enumerate()
        .filter(|&(i, r)| openings[i] <= 0.0 && !non_main.is_some_and(|m| m[r.first as usize]))
        .filter_map(|(i, r)| {
            let bverts: BTreeSet<u32> = r.boundary.iter().flatten().copied().collect();
            (bverts.len() >= 3).then(|| {
                let pts: Vec<[f64; 3]> = bverts.iter().map(|&v| s.verts[v as usize]).collect();
                (i, centred(&pts))
            })
        })
        .collect();
    let pts: Vec<&[[f64; 3]]> = hulls.iter().map(|(_, c)| c.as_slice()).collect();
    for ((i, c), vt) in hulls.iter().zip(svd.right_singular_each(&pts, true)) {
        openings[*i] = hull_area_on(c, vt);
    }
    rims.into_iter()
        .zip(openings)
        .filter(|(r, opening)| !(*opening <= 0.0 || r.area / opening < params.min_fold_ratio))
        .map(|(r, _)| r.boundary)
        .collect()
}

/// What [`pocket_organelles`] found, with the tallies the verbose report
/// prints.
#[derive(Debug, Clone, Default)]
pub struct Pockets {
    pub mask: Vec<bool>,
    pub n_mouths: usize,
    pub n_initial: usize,
    pub n_bridged: usize,
    pub n_holes: usize,
    pub n_rejected: usize,
}

/// `skeliner.pre.find_pocket_organelles` with the mouths, main mask and
/// non-main mask given.
pub fn pocket_organelles(
    s: &Surface<'_>,
    main_mask: &[bool],
    non_main: Option<&[bool]>,
    params: &PocketParams,
    svd: &dyn Svd,
) -> Pockets {
    let n = s.faces.len();
    let mouths = pocket_mouths(s, main_mask, non_main, params, svd);
    let mut out = Pockets {
        mask: vec![false; n],
        n_mouths: mouths.len(),
        ..Default::default()
    };
    if mouths.is_empty() {
        return out;
    }
    // the faces on the mouths' edges: seeds where they dip inward, else
    // walls
    let mut mouth_edges: Vec<[u32; 2]> = mouths.concat();
    let on_mouths: Vec<u32> = threads::install(|| {
        mouth_edges.par_sort_unstable();
        mouth_edges.dedup();
        mouth_edges
            .par_iter()
            .flat_map_iter(|&e| faces_on(s, e).iter().copied())
            .collect()
    });
    let seed = |f: u32| main_mask[f as usize] && s.dots[f as usize] < 0.0;
    let mut mouth_face = vec![false; n];
    for &f in &on_mouths {
        if !seed(f) {
            mouth_face[f as usize] = true;
        }
    }
    let pocket = &mut out.mask;

    // Phase 0: flood from the seeds, stopped by mouth faces and the grow
    // threshold — the pieces of the faces under it that hold a seed.
    // a NaN dot is not over the threshold
    let grows = |f: usize, threshold: f64| {
        !mouth_face[f]
            && main_mask[f]
            && s.dots[f].partial_cmp(&threshold) != Some(std::cmp::Ordering::Greater)
    };
    threads::install(|| {
        let ok: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|f| grows(f, params.grow_threshold))
            .collect();
        let roots = lowest_faces(s, &ok);
        let mut held = vec![false; n];
        for &f in on_mouths.iter().filter(|&&f| seed(f) && ok[f as usize]) {
            held[roots[f as usize] as usize] = true;
        }
        pocket
            .par_iter_mut()
            .enumerate()
            .for_each(|(f, p)| *p = ok[f] && held[roots[f] as usize]);
    });
    out.n_initial = pocket.iter().filter(|&&p| p).count();

    // Phase 1: bridge out from the pocket under the relaxed threshold —
    // the pieces of the other faces under it that touch the pocket.
    threads::install(|| {
        let ok: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|f| !pocket[f] && grows(f, params.bridge_threshold))
            .collect();
        let roots = lowest_faces(s, &ok);
        let touched: Vec<u32> = (0..n)
            .into_par_iter()
            .filter(|&f| pocket[f])
            .flat_map_iter(|f| s.topo.face_neighbors(f))
            .filter(|&g| ok[g as usize])
            .map(|g| roots[g as usize])
            .collect();
        let mut held = vec![false; n];
        for r in touched {
            held[r as usize] = true;
        }
        pocket
            .par_iter_mut()
            .enumerate()
            .for_each(|(f, p)| *p = *p || (ok[f] && held[roots[f] as usize]));
    });
    out.n_bridged = pocket.iter().filter(|&&p| p).count() - out.n_initial;

    // Phase 2: fill holes — non-pocket clusters mostly enclosed by pocket.
    // A hole's neighbours are pocket or off the main mask, never another
    // hole, so filling one leaves the others as they were.
    let open: Vec<bool> = (0..n).map(|f| main_mask[f] && !pocket[f]).collect();
    let holes = components_where_unordered(s, |f| open[f]);
    let walled_in: &[bool] = pocket;
    let filled: Vec<bool> = threads::install(|| {
        holes
            .par_iter()
            .map(|cluster| {
                // neither small nor under 5% of the mesh: never filled
                if cluster.len() > params.max_hole_size && cluster.len() as f64 >= n as f64 * 0.05 {
                    return false;
                }
                let (mut total, mut walled) = (0usize, 0usize);
                let mut walls: Vec<u32> = Vec::new();
                for &f in cluster {
                    for g in s.topo.face_neighbors(f as usize) {
                        if !open[g as usize] {
                            total += 1;
                            if walled_in[g as usize] {
                                walled += 1;
                                walls.push(g);
                            }
                        }
                    }
                }
                walls.sort_unstable();
                walls.dedup();
                let enclosure = if total > 0 {
                    walled as f64 / total as f64
                } else {
                    0.0
                };
                let small = cluster.len() <= params.max_hole_size;
                let entrapped = enclosure >= 0.99 && (cluster.len() as f64) < n as f64 * 0.05;
                if non_main.is_some_and(|m| m[cluster[0] as usize]) && cluster.len() > walls.len() {
                    return false;
                }
                (small && enclosure >= params.hole_enclosure_ratio) || entrapped
            })
            .collect()
    });
    for (cluster, _) in holes.iter().zip(&filled).filter(|(_, f)| **f) {
        for &f in cluster {
            pocket[f as usize] = true;
        }
        out.n_holes += cluster.len();
    }

    // Drop small clusters, then reject shallow dips: a pocket must have
    // much more area than the faces capping it.
    let kept = mesh::components::keep_large_clusters(s.faces, pocket, params.min_cluster_size);
    *pocket = kept;
    let flagged = pocket.clone();
    let comps = components_where_unordered(s, |f| flagged[f]);
    let shallow: Vec<bool> = threads::install(|| {
        comps
            .par_iter()
            .map(|comp| {
                let members: BTreeSet<u32> = comp.iter().copied().collect();
                let cap: BTreeSet<u32> = comp
                    .iter()
                    .flat_map(|&f| s.topo.face_neighbors(f as usize))
                    .filter(|g| !members.contains(g))
                    .collect();
                let pocket_area = area_of(s, &members);
                let cap_area = if cap.is_empty() {
                    0.0
                } else {
                    area_of(s, &cap)
                };
                cap_area <= 0.0 || pocket_area / cap_area < params.min_fold_ratio
            })
            .collect()
    });
    for (comp, &drop) in comps.iter().zip(&shallow) {
        if drop {
            for &f in comp {
                pocket[f as usize] = false;
            }
            out.n_rejected += 1;
        }
    }
    out
}

/// What [`find_organelles`] found.
#[derive(Debug, Clone, Default)]
pub struct Organelles {
    pub pocket: Vec<bool>,
    pub isolated: Isolated,
    pub pockets: Pockets,
    pub n_structural: usize,
    pub n_enclosed: usize,
}

/// `skeliner.pre.find_organelles` given the mesh stats: isolated organelles,
/// then pockets on every structural component at once, then the enclosure
/// pass over trimesh's manifold `adjacency`.
pub fn find_organelles(
    s: &Surface<'_>,
    face_comp: &[i64],
    main: i64,
    adjacency: &[[u32; 2]],
    min_cluster_size: usize,
    svd: &dyn Svd,
) -> Organelles {
    let fragments = Fragments {
        faces: s.faces,
        verts: s.verts,
        centers: s.centers,
        normals: s.normals,
        dots: s.dots,
    };
    let isolated = isolated_organelles(&fragments, face_comp, main);
    let (structural, non_main, n_structural) =
        structural_masks(face_comp, main, &isolated.mask, min_cluster_size);
    let params = PocketParams {
        min_cluster_size,
        ..PocketParams::default()
    };
    let pockets = pocket_organelles(s, &structural, Some(&non_main), &params, svd);
    let candidates: Vec<bool> = (0..s.faces.len())
        .map(|f| structural[f] && !pockets.mask[f] && !isolated.mask[f])
        .collect();
    let (pocket, n_enclosed) =
        enclosure_pass(&candidates, adjacency, &pockets.mask, min_cluster_size);
    Organelles {
        pocket,
        isolated,
        pockets,
        n_structural,
        n_enclosed,
    }
}
