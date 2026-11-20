//! `skeliner.pre.find_parallel_patches` and `remove_parallel_patches`
//! without their shapely steps: axis-aligned face patches at chunk
//! boundaries that face an opposing patch, their removal, and the strips
//! that close what the removal opens.
//!
//! The reference's edge maps are rebuilt as it builds them — keyed by each
//! face's three column pairs, a degenerate face listed once per pair —
//! because neighbour counts read those lists directly.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use mesh::Topology;
use mesh::components::{face_edge_components, face_edge_components_without};
use mesh::geometry::{numpy_order, numpy_sum};
use rayon::prelude::*;

type V3 = [f64; 3];
type Edge = (u32, u32);

/// A patch as the reference's dicts hold it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patch {
    pub axis: usize,
    pub bval: i64,
    pub faces: Vec<u32>,
    pub up: Vec<u32>,
    pub down: Vec<u32>,
}

/// The mesh the patch steps read; `normals` are trimesh's face normals.
pub struct PatchMesh<'a> {
    pub faces: &'a [[u32; 3]],
    pub verts: &'a [V3],
    pub normals: &'a [V3],
}

#[derive(Debug, Clone, Copy)]
pub struct FindParams {
    pub normal_thresh: f64,
    pub boundary_dist: f64,
    pub expand_tilted: bool,
    pub loose_normal_thresh: f64,
}

fn edge(a: u32, b: u32) -> Edge {
    (a.min(b), a.max(b))
}

/// The three `(f[i], f[(i + 1) % 3])` edges of a face.
fn face_edges(f: [u32; 3]) -> [Edge; 3] {
    [edge(f[0], f[1]), edge(f[1], f[2]), edge(f[2], f[0])]
}

fn perp(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

/// `verts[faces].mean(axis=1)`.
fn centroids(m: &PatchMesh<'_>) -> Vec<V3> {
    m.faces
        .par_iter()
        .map(|f| {
            let (a, b, c) = (
                m.verts[f[0] as usize],
                m.verts[f[1] as usize],
                m.verts[f[2] as usize],
            );
            [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0)
        })
        .collect()
}

/// Edge -> faces over `fis`, in their order.
fn edge_map(m: &PatchMesh<'_>, fis: impl Iterator<Item = u32>) -> HashMap<Edge, Vec<u32>> {
    let mut out: HashMap<Edge, Vec<u32>> = HashMap::new();
    for fi in fis {
        for e in face_edges(m.faces[fi as usize]) {
            out.entry(e).or_default().push(fi);
        }
    }
    out
}

/// numpy's `round` to an integer: half to even.
fn round_i64(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `_any_face_overlaps`: does a corner of one set's triangles fall
/// strictly inside a triangle of the other, in the plane across `axis`?
/// Corners on vertices both sets share do not count.
fn any_face_overlaps(m: &PatchMesh<'_>, fa: &[u32], fb: &[u32], axis: usize) -> bool {
    let [p, q] = perp(axis);
    let tri2 =
        |f: u32| m.faces[f as usize].map(|v| [m.verts[v as usize][p], m.verts[v as usize][q]]);
    let ta: Vec<[[f64; 2]; 3]> = fa.iter().map(|&f| tri2(f)).collect();
    let tb: Vec<[[f64; 2]; 3]> = fb.iter().map(|&f| tri2(f)).collect();
    let bbox = |t: &[[[f64; 2]; 3]]| {
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for c in t.iter().flatten() {
            for k in 0..2 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        (lo, hi)
    };
    let ((a_min, a_max), (b_min, b_max)) = (bbox(&ta), bbox(&tb));
    if !((0..2).all(|k| a_min[k] <= b_max[k]) && (0..2).all(|k| b_min[k] <= a_max[k])) {
        return false;
    }
    let verts_of =
        |fs: &[u32]| -> HashSet<u32> { fs.iter().flat_map(|&f| m.faces[f as usize]).collect() };
    let (va, vb) = (verts_of(fa), verts_of(fb));
    let inside = |pt: [f64; 2], t: &[[f64; 2]; 3]| {
        let [[x0, y0], [x1, y1], [x2, y2]] = *t;
        let (px, py) = (pt[0], pt[1]);
        let d00 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0);
        let d01 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
        let d02 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2);
        (d00 > 0.0 && d01 > 0.0 && d02 > 0.0) || (d00 < 0.0 && d01 < 0.0 && d02 < 0.0)
    };
    let corners = |fs: &[u32], tris: &[[[f64; 2]; 3]], other: &HashSet<u32>| -> Vec<[f64; 2]> {
        let mut out = Vec::new();
        for (f, t) in fs.iter().zip(tris) {
            for (k, &v) in m.faces[*f as usize].iter().enumerate() {
                if !other.contains(&v) {
                    out.push(t[k]);
                }
            }
        }
        out
    };
    let pa = corners(fa, &ta, &vb);
    if !pa.is_empty() && tb.iter().any(|t| pa.iter().any(|&c| inside(c, t))) {
        return true;
    }
    let pb = corners(fb, &tb, &va);
    !pb.is_empty() && ta.iter().any(|t| pb.iter().any(|&c| inside(c, t)))
}

/// `find_parallel_patches` up to (not including) the sandwich-overlap
/// filter and the final sort.  `boundaries` are `find_chunk_boundaries`'
/// planes per axis, in its order.  The loops over every face run on
/// [`threads`]' pool.
pub fn find_parallel_patches(
    m: &PatchMesh<'_>,
    boundaries: &[(usize, Vec<f64>)],
    p: &FindParams,
) -> Vec<Patch> {
    threads::install(|| find_patches(m, boundaries, p))
}

fn find_patches(m: &PatchMesh<'_>, boundaries: &[(usize, Vec<f64>)], p: &FindParams) -> Vec<Patch> {
    let nf = m.faces.len();
    let cent = centroids(m);
    let mut results: Vec<Patch> = Vec::new();

    for (axis, bvals) in boundaries {
        let axis = *axis;
        let n_ax = |f: u32| m.normals[f as usize][axis];
        // the last plane within `boundary_dist` of each face's centroid
        let face_bval: Vec<Option<i64>> = cent
            .par_iter()
            .map(|c| {
                bvals
                    .iter()
                    .rev()
                    .find(|&&bv| (c[axis] - bv).abs() < p.boundary_dist)
                    .map(|&bv| round_i64(bv))
            })
            .collect();
        let target: Vec<u32> = (0..nf as u32)
            .into_par_iter()
            .filter(|&f| n_ax(f).abs() > p.normal_thresh && face_bval[f as usize].is_some())
            .collect();
        if target.len() < 2 {
            continue;
        }
        let mut in_target = vec![false; nf];
        for &f in &target {
            in_target[f as usize] = true;
        }
        let e2f = edge_map(m, target.iter().copied());

        // connected pieces of the target, one boundary value each
        let mut visited = vec![false; nf];
        let mut pieces: Vec<(Vec<u32>, i64)> = Vec::new();
        for &seed in &target {
            if visited[seed as usize] {
                continue;
            }
            let sbv = face_bval[seed as usize];
            let mut comp = Vec::new();
            let mut queue = VecDeque::from([seed]);
            visited[seed as usize] = true;
            while let Some(fi) = queue.pop_front() {
                comp.push(fi);
                for e in face_edges(m.faces[fi as usize]) {
                    for &nb in &e2f[&e] {
                        let u = nb as usize;
                        if !visited[u] && in_target[u] && face_bval[u] == sbv {
                            visited[u] = true;
                            queue.push_back(nb);
                        }
                    }
                }
            }
            pieces.push((comp, sbv.expect("a target face is near a plane")));
        }

        for (comp, bv) in pieces {
            let pos: Vec<u32> = comp
                .iter()
                .copied()
                .filter(|&f| n_ax(f) > p.normal_thresh)
                .collect();
            let neg: Vec<u32> = comp
                .iter()
                .copied()
                .filter(|&f| n_ax(f) < -p.normal_thresh)
                .collect();
            let mut overlap =
                !pos.is_empty() && !neg.is_empty() && any_face_overlaps(m, &pos, &neg, axis);
            if !overlap {
                let in_comp: HashSet<u32> = comp.iter().copied().collect();
                let others: Vec<u32> = target
                    .iter()
                    .copied()
                    .filter(|f| face_bval[*f as usize] == Some(bv) && !in_comp.contains(f))
                    .collect();
                if !others.is_empty() {
                    let dirs: Vec<f64> = comp.iter().map(|&f| n_ax(f)).collect();
                    let mean_dir = numpy_sum(&dirs) / dirs.len() as f64;
                    let opp: Vec<u32> = if mean_dir > 0.0 {
                        others
                            .into_iter()
                            .filter(|&f| n_ax(f) < -p.normal_thresh)
                            .collect()
                    } else {
                        others
                            .into_iter()
                            .filter(|&f| n_ax(f) > p.normal_thresh)
                            .collect()
                    };
                    overlap = !opp.is_empty() && any_face_overlaps(m, &comp, &opp, axis);
                }
            }
            if overlap {
                results.push(Patch {
                    axis,
                    bval: bv,
                    faces: comp,
                    up: pos,
                    down: neg,
                });
            }
        }
    }

    // faces bordering two or more patch faces join, among faces near any
    // boundary: a face's count is, over its edges (a degenerate face's
    // repeated one twice), how often the patch faces have that edge
    let mut in_patch = vec![false; nf];
    let mut patch_edges: HashMap<Edge, usize> = HashMap::new();
    for r in &results {
        for &f in &r.faces {
            in_patch[f as usize] = true;
        }
    }
    for f in (0..nf).filter(|&f| in_patch[f]) {
        for e in face_edges(m.faces[f]) {
            *patch_edges.entry(e).or_insert(0) += 1;
        }
    }
    let joining: Vec<usize> = (0..nf)
        .into_par_iter()
        .filter(|&f| {
            !in_patch[f]
                && face_edges(m.faces[f])
                    .iter()
                    .map(|e| patch_edges.get(e).copied().unwrap_or(0))
                    .sum::<usize>()
                    >= 2
                && boundaries
                    .iter()
                    .any(|(ax, bv)| bv.iter().any(|&b| (cent[f][*ax] - b).abs() < 30.0))
        })
        .collect();
    for f in joining {
        in_patch[f] = true;
    }

    // merge into connected patches, each sorted, by lowest face
    let all: Vec<u32> = (0..nf as u32).filter(|&f| in_patch[f as usize]).collect();
    let merge_e2f = edge_map(m, all.iter().copied());
    let mut seen = vec![false; nf];
    let mut merged: Vec<Vec<u32>> = Vec::new();
    for &seed in &all {
        if seen[seed as usize] {
            continue;
        }
        let mut comp = Vec::new();
        let mut queue = VecDeque::from([seed]);
        seen[seed as usize] = true;
        while let Some(fi) = queue.pop_front() {
            comp.push(fi);
            for e in face_edges(m.faces[fi as usize]) {
                for &nb in &merge_e2f[&e] {
                    if !seen[nb as usize] {
                        seen[nb as usize] = true;
                        queue.push_back(nb);
                    }
                }
            }
        }
        comp.sort_unstable();
        merged.push(comp);
    }

    let mut results: Vec<Patch> = merged
        .into_iter()
        .map(|comp| {
            let mut best_axis = 0;
            let mut best_count = 0;
            for ax in [0, 1, 2] {
                let cnt = comp
                    .iter()
                    .filter(|&&f| m.normals[f as usize][ax].abs() > p.normal_thresh)
                    .count();
                if cnt > best_count {
                    best_count = cnt;
                    best_axis = ax;
                }
            }
            let mut c: Vec<f64> = comp.iter().map(|&f| cent[f as usize][best_axis]).collect();
            c.sort_by(numpy_order);
            let n = c.len();
            let median = if n % 2 == 1 {
                c[n / 2]
            } else {
                (c[n / 2 - 1] + c[n / 2]) / 2.0
            };
            let mut bv = round_i64(median);
            if let Some((_, bvals)) = boundaries.iter().find(|(ax, _)| *ax == best_axis)
                && !bvals.is_empty()
            {
                let mut best = 0;
                for (i, &b) in bvals.iter().enumerate() {
                    if (b - bv as f64).abs() < (bvals[best] - bv as f64).abs() {
                        best = i;
                    }
                }
                bv = round_i64(bvals[best]);
            }
            let up = comp
                .iter()
                .copied()
                .filter(|&f| m.normals[f as usize][best_axis] > 0.5)
                .collect();
            let down = comp
                .iter()
                .copied()
                .filter(|&f| m.normals[f as usize][best_axis] < -0.5)
                .collect();
            Patch {
                axis: best_axis,
                bval: bv,
                faces: comp,
                up,
                down,
            }
        })
        .collect();

    if p.expand_tilted {
        results =
            expand_parallel_patches(m, &results, &cent, p.loose_normal_thresh, p.boundary_dist);
    }
    results
}

fn patch_sign(p: &Patch) -> i32 {
    if !p.up.is_empty() {
        1
    } else if !p.down.is_empty() {
        -1
    } else {
        0
    }
}

/// `(axis, bval)` groups of patch indices, in first-seen order.
fn groups(patches: &[Patch]) -> Vec<((usize, i64), Vec<usize>)> {
    let mut out: Vec<((usize, i64), Vec<usize>)> = Vec::new();
    for (i, p) in patches.iter().enumerate() {
        let key = (p.axis, p.bval);
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(i),
            None => out.push((key, vec![i])),
        }
    }
    out
}

/// `_expand_parallel_patches`: absorb same-sign tilted faces at each seed
/// patch's rim that lie near its plane and over the bbox of an opposing
/// patch, then merge patches whose expansions meet.
fn expand_parallel_patches(
    m: &PatchMesh<'_>,
    patches: &[Patch],
    cent: &[V3],
    loose: f64,
    boundary_dist: f64,
) -> Vec<Patch> {
    if patches.is_empty() {
        return Vec::new();
    }
    let nf = m.faces.len();
    let expand_dist = 2.0 * boundary_dist;
    // a face joins only along a group's axis and near its plane (the tests
    // below; NaN fails neither there, so it is no reason to leave a face
    // out here); the edge map need hold no other face
    let planes: Vec<(usize, f64)> = groups(patches)
        .iter()
        .map(|&((axis, bval), _)| (axis, bval as f64))
        .collect();
    let candidates: Vec<u32> = (0..nf as u32)
        .into_par_iter()
        .filter(|&f| {
            let (t, n, c) = (m.faces[f as usize], m.normals[f as usize], cent[f as usize]);
            t[0] != t[1]
                && t[1] != t[2]
                && t[0] != t[2]
                && planes.iter().any(|&(axis, bval)| {
                    n[axis].abs().partial_cmp(&loose) != Some(Ordering::Less)
                        && (c[axis] - bval).abs().partial_cmp(&expand_dist)
                            != Some(Ordering::Greater)
                })
        })
        .collect();
    let full_ef = edge_map(m, candidates.into_iter());
    let mut expansions: Vec<HashSet<u32>> = patches
        .iter()
        .map(|p| p.faces.iter().copied().collect())
        .collect();

    for ((axis, bval), members) in groups(patches) {
        let [pa, pb] = perp(axis);
        let mut opp_bbox: HashMap<i32, ([f64; 2], [f64; 2])> = HashMap::new();
        for sign in [1, -1] {
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            let mut any = false;
            for &pj in &members {
                if patch_sign(&patches[pj]) != -sign {
                    continue;
                }
                for &f in &patches[pj].faces {
                    for &v in &m.faces[f as usize] {
                        let x = m.verts[v as usize];
                        for (k, c) in [x[pa], x[pb]].into_iter().enumerate() {
                            lo[k] = lo[k].min(c);
                            hi[k] = hi[k].max(c);
                        }
                        any = true;
                    }
                }
            }
            if any {
                opp_bbox.insert(sign, (lo, hi));
            }
        }
        for &pi in &members {
            let sign = patch_sign(&patches[pi]);
            let Some(&(xy_min, xy_max)) = (sign != 0).then(|| opp_bbox.get(&sign)).flatten() else {
                continue;
            };
            let mut queue: VecDeque<u32> = patches[pi].faces.iter().copied().collect();
            while let Some(fi) = queue.pop_front() {
                for e in face_edges(m.faces[fi as usize]) {
                    for &nb in full_ef.get(&e).map_or(&[][..], Vec::as_slice) {
                        if expansions[pi].contains(&nb) {
                            continue;
                        }
                        let n_ax = m.normals[nb as usize][axis];
                        if n_ax * f64::from(sign) <= 0.0 || n_ax.abs() < loose {
                            continue;
                        }
                        if (cent[nb as usize][axis] - bval as f64).abs() > expand_dist {
                            continue;
                        }
                        let corners = m.faces[nb as usize].map(|v| {
                            let x = m.verts[v as usize];
                            [x[pa], x[pb]]
                        });
                        let fits = (0..2).all(|k| {
                            let lo = corners.iter().map(|c| c[k]).fold(f64::INFINITY, f64::min);
                            let hi = corners
                                .iter()
                                .map(|c| c[k])
                                .fold(f64::NEG_INFINITY, f64::max);
                            lo < xy_max[k] && hi > xy_min[k]
                        });
                        if !fits {
                            continue;
                        }
                        expansions[pi].insert(nb);
                        queue.push_back(nb);
                    }
                }
            }
        }
    }

    // union-find over same-sign patches at one boundary that share faces
    let n = patches.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&patches[i], &patches[j]);
            if a.axis != b.axis || a.bval != b.bval || patch_sign(a) != patch_sign(b) {
                continue;
            }
            if !expansions[i].is_disjoint(&expansions[j]) {
                let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut merged: Vec<(usize, Vec<usize>)> = Vec::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        match merged.iter_mut().find(|(k, _)| *k == r) {
            Some((_, v)) => v.push(i),
            None => merged.push((r, vec![i])),
        }
    }
    merged
        .into_iter()
        .map(|(root, members)| {
            let axis = patches[root].axis;
            let mut faces: HashSet<u32> = HashSet::new();
            let mut up: HashSet<u32> = HashSet::new();
            let mut down: HashSet<u32> = HashSet::new();
            let mut orig: HashSet<u32> = HashSet::new();
            for &mi in &members {
                faces.extend(&expansions[mi]);
                up.extend(&patches[mi].up);
                down.extend(&patches[mi].down);
                orig.extend(&patches[mi].faces);
            }
            for &f in faces.difference(&orig) {
                if m.normals[f as usize][axis] > 0.0 {
                    up.insert(f);
                } else {
                    down.insert(f);
                }
            }
            let sorted = |s: HashSet<u32>| {
                let mut v: Vec<u32> = s.into_iter().collect();
                v.sort_unstable();
                v
            };
            Patch {
                axis,
                bval: patches[root].bval,
                faces: sorted(faces),
                up: sorted(up),
                down: sorted(down),
            }
        })
        .collect()
}

/// `_find_fold_faces`: faces of the patch on an edge where one of its up
/// faces meets one of its down faces.
fn fold_faces(m: &PatchMesh<'_>, p: &Patch) -> HashSet<u32> {
    let up: HashSet<u32> = p.up.iter().copied().collect();
    let down: HashSet<u32> = p.down.iter().copied().collect();
    let mut fold = HashSet::new();
    for fis in edge_map(m, p.faces.iter().copied()).values() {
        if fis.iter().any(|f| up.contains(f)) && fis.iter().any(|f| down.contains(f)) {
            fold.extend(fis.iter().copied());
        }
    }
    fold
}

/// A traced rim split where it passes a vertex twice: the vertices
/// between the two passes close a loop of their own (a lobe touching the
/// rest at that vertex), and what is left runs on as before. Each lobe
/// runs round the other way from the rest, so a strip that went round
/// both in one could be wound right on one only.
fn simple_loops(lp: &[u32], closed: bool) -> Vec<(Vec<u32>, bool)> {
    let mut out = Vec::new();
    let mut rest: Vec<u32> = Vec::with_capacity(lp.len());
    let mut at: HashMap<u32, usize> = HashMap::new();
    for &v in lp {
        if let Some(&i) = at.get(&v) {
            let lobe: Vec<u32> = rest.drain(i + 1..).collect();
            for w in &lobe {
                at.remove(w);
            }
            let mut sub = vec![v];
            sub.extend(lobe);
            out.push((sub, true));
        } else {
            at.insert(v, rest.len());
            rest.push(v);
        }
    }
    out.push((rest, closed));
    out
}

/// `_trace_edge_loops`: undirected edges as vertex sequences — open
/// chains from their degree-1 ends first, then closed loops — each with
/// whether it closes.
pub fn trace_edge_loops(edges: &[Edge]) -> Vec<(Vec<u32>, bool)> {
    let mut keys: Vec<Edge> = edges.iter().map(|&(a, b)| edge(a, b)).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut order: Vec<u32> = Vec::new();
    let mut ve: HashMap<u32, Vec<Edge>> = HashMap::new();
    for &k in &keys {
        for v in [k.0, k.1] {
            let list = ve.entry(v).or_default();
            if list.is_empty() {
                order.push(v);
            }
            list.push(k);
        }
    }
    let mut used: HashSet<Edge> = HashSet::new();
    let walk = |start: Edge, used: &mut HashSet<Edge>| -> (Vec<u32>, bool) {
        let mut chain = vec![start.0, start.1];
        used.insert(edge(start.0, start.1));
        let mut curr = start.1;
        let mut closed = false;
        while let Some(&next_e) = ve[&curr].iter().find(|e| !used.contains(e)) {
            used.insert(next_e);
            let nxt = if next_e.0 == curr { next_e.1 } else { next_e.0 };
            if nxt == chain[0] {
                closed = true;
                break;
            }
            chain.push(nxt);
            curr = nxt;
        }
        (chain, closed)
    };
    let mut out = Vec::new();
    let endpoints: Vec<u32> = order.iter().copied().filter(|v| ve[v].len() == 1).collect();
    for v in endpoints {
        let e = ve[&v][0];
        if used.contains(&e) {
            continue;
        }
        let oriented = if e.0 == v { e } else { (e.1, e.0) };
        out.push(walk(oriented, &mut used));
    }
    for &e in &keys {
        if !used.contains(&e) {
            out.push(walk(e, &mut used));
        }
    }
    out
}

fn dist(a: V3, b: V3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Whether the triangle `t` names a vertex twice.
fn names_twice(t: [u32; 3]) -> bool {
    t[0] == t[1] || t[1] == t[2] || t[0] == t[2]
}

/// Whether the triangle `t`, of three distinct vertices, spans no area.
fn flat(verts: &[V3], t: [u32; 3]) -> bool {
    let [a, b, c] = t.map(|v| verts[v as usize]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let x = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    x == [0.0; 3]
}

/// `_zip_vertex_loops`: for each way round `lb`, the cheapest triangle
/// strip between two vertex sequences over every start offset of `lb` (the
/// shorter bridge first at each step), with its cost — the cheaper way
/// first.  Each runs `la` forward; the caller takes one and winds it: a
/// loop whose direction was misjudged would twist the strip, its rungs
/// crossing at length, but the cheaper way can also be the one no winding
/// joins to the faces on both rims.
///
/// The rims of a sandwich lie in one plane and touch where its two
/// outlines meet. Where they share a vertex the strip pinches through it:
/// the step whose triangle would name the shared vertex twice goes first,
/// so that both walks pass the vertex together, and lays nothing — the
/// triangle before it already has the other rim's edge into the vertex as
/// its rung. Otherwise a step whose rung is `taken` (an edge faces already
/// have, which the strip's two triangles on it would crowd) gives way to
/// the other, and then a step whose triangle would span no area to the
/// other where that one spans some.
fn zip_vertex_loops(
    la: &[u32],
    lb: &[u32],
    verts: &[V3],
    closed_a: bool,
    closed_b: bool,
    taken: &dyn Fn(u32, u32) -> bool,
) -> Vec<(f64, Vec<[u32; 3]>)> {
    let (m, n) = (la.len(), lb.len());
    if m < 2 || n < 2 {
        return Vec::new();
    }
    let at = |v: u32| verts[v as usize];
    let walk = |sb: &[u32]| -> (f64, Vec<[u32; 3]>) {
        let mut tris = Vec::new();
        let mut cost = 0.0;
        let (mut i, mut j) = (0, 0);
        let steps_a = if closed_a { m } else { m - 1 };
        let steps_b = if closed_b { n } else { n - 1 };
        while i < steps_a || j < steps_b {
            let (a_curr, a_next) = (la[i % m], la[(i + 1) % m]);
            let (t_curr, t_next) = (sb[j % n], sb[(j + 1) % n]);
            let (by_a, by_b) = ([a_curr, a_next, t_curr], [a_curr, t_next, t_curr]);
            let take_a = if i >= steps_a {
                false
            } else if j >= steps_b {
                true
            } else if names_twice(by_a) != names_twice(by_b) {
                names_twice(by_a)
            } else if names_twice(by_a) {
                dist(at(a_next), at(t_curr)) <= dist(at(a_curr), at(t_next))
            } else if taken(a_next, t_curr) != taken(a_curr, t_next) {
                taken(a_curr, t_next)
            } else if flat(verts, by_a) != flat(verts, by_b) {
                flat(verts, by_b)
            } else {
                dist(at(a_next), at(t_curr)) <= dist(at(a_curr), at(t_next))
            };
            if take_a {
                cost += dist(at(a_next), at(t_curr));
                tris.push(by_a);
                i += 1;
            } else {
                cost += dist(at(a_curr), at(t_next));
                tris.push(by_b);
                j += 1;
            }
        }
        tris.retain(|&t| !names_twice(t));
        (cost, tris)
    };
    let shifts = if closed_b { n } else { 1 };
    let back: Vec<u32> = lb.iter().rev().copied().collect();
    let mut ways: Vec<(f64, Vec<[u32; 3]>)> = Vec::new();
    for way in [lb, &back[..]] {
        let mut best: Option<(f64, Vec<[u32; 3]>)> = None;
        for shift in 0..shifts {
            let shifted: Vec<u32> = way[shift..].iter().chain(&way[..shift]).copied().collect();
            let (cost, tris) = walk(&shifted);
            if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                best = Some((cost, tris));
            }
        }
        ways.extend(best);
    }
    // stable: on a tie the way `lb` runs comes first
    ways.sort_by(|x, y| x.0.total_cmp(&y.0));
    ways
}

/// Whether the faces left along `lp`'s edges run it forward (`lp[i]` to
/// `lp[i + 1]`) more often than backward.
fn runs_forward(
    lp: &[u32],
    closed: bool,
    topo: &Topology,
    gone: &[bool],
    orig: &[[u32; 3]],
) -> bool {
    let n = lp.len();
    let steps = if closed { n } else { n - 1 };
    let (mut forward, mut backward) = (0usize, 0usize);
    for i in 0..steps {
        let (a, b) = (lp[i], lp[(i + 1) % n]);
        let Some(e) = topo.edge_id(a, b) else {
            continue;
        };
        for &f in topo.edge_faces(e).iter().filter(|&&f| !gone[f as usize]) {
            let t = orig[f as usize];
            if (0..3).any(|k| t[k] == a && t[(k + 1) % 3] == b) {
                forward += 1;
            } else {
                backward += 1;
            }
        }
    }
    forward > backward
}

/// One `(axis, bval)` group's rims as [`stitch_mode_b`] pairs them: the
/// loops (oriented as the faces left on them want), per loop its centre
/// and its box across the axis, the signed area it encloses there (none
/// for an open one), and the pieces of the faces left along it.
struct Rims {
    up: Vec<(Vec<u32>, bool)>,
    dn: Vec<(Vec<u32>, bool)>,
    up_info: Vec<(V3, [f64; 2], [f64; 2])>,
    dn_info: Vec<(V3, [f64; 2], [f64; 2])>,
    up_turn: Vec<f64>,
    dn_turn: Vec<f64>,
    up_pieces: Vec<Vec<usize>>,
    dn_pieces: Vec<Vec<usize>>,
    dn_used: Vec<bool>,
    /// The up rims to pair, largest first; after the first pass, those it
    /// left over.
    order: Vec<usize>,
}

/// The pieces the removal leaves, joined as the strips are laid.
struct Joined(Vec<usize>);

impl Joined {
    fn root(&mut self, mut a: usize) -> usize {
        while self.0[a] != a {
            self.0[a] = self.0[self.0[a]];
            a = self.0[a];
        }
        a
    }

    fn join(&mut self, pieces: &[usize]) {
        for w in pieces.windows(2) {
            let (a, b) = (self.root(w[0]), self.root(w[1]));
            self.0[a.max(b)] = a.min(b);
        }
    }

    fn apart(&mut self, a: &[usize], b: &[usize]) -> bool {
        let ra: Vec<usize> = a.iter().map(|&p| self.root(p)).collect();
        b.iter().all(|&p| {
            let r = self.root(p);
            !ra.contains(&r)
        })
    }
}

/// What [`stitch_mode_b`] has laid so far.
struct Laid {
    tris: Vec<[u32; 3]>,
    keys: HashSet<[u32; 3]>,
    edges: HashSet<Edge>,
    n_turned: usize,
    joined: Joined,
}

/// `_stitch_mode_b`: per `(axis, bval)` group of patches with Mode B faces,
/// trace the rims each removed side leaves, pair up-side with down-side
/// rims (largest first, nearest centre with overlapping bboxes) and zip
/// each pair: in every group first the strip the cheaper way round, wound
/// as the faces on both rims want; then an up rim left over the dearer way
/// round to a closed down rim turning alike, only where the strip joins
/// two pieces no strip has joined.  `topo` is the mesh's before the removal
/// and `gone` the faces removed.  Returns the new triangles and how many
/// were laid the dearer way round.
fn stitch_mode_b(
    verts: &[V3],
    topo: &Topology,
    gone: &[bool],
    orig: &[[u32; 3]],
    mode_b: &HashSet<u32>,
    patches: &[Patch],
    open: &[OpenRim],
) -> (Vec<[u32; 3]>, usize) {
    let rim_edges = |side: &HashSet<u32>| -> Vec<Edge> {
        let side: Vec<u32> = side.iter().copied().collect();
        rim_left(topo, gone, orig, &side)
    };
    let with_b: Vec<Patch> = patches
        .iter()
        .filter(|p| p.faces.iter().any(|f| mode_b.contains(f)))
        .cloned()
        .collect();
    let (piece, _) = face_edge_components_without(topo, gone);
    let n_pieces = piece.iter().max().map_or(0, |&p| (p + 1).max(0) as usize);
    // the pieces of the faces left along a loop's edges
    let pieces_along = |(lp, closed): &(Vec<u32>, bool)| -> Vec<usize> {
        let steps = if *closed { lp.len() } else { lp.len() - 1 };
        let mut out: Vec<usize> = (0..steps)
            .filter_map(|i| topo.edge_id(lp[i], lp[(i + 1) % lp.len()]))
            .flat_map(|e| topo.edge_faces(e).iter().copied())
            .filter(|&f| !gone[f as usize] && piece[f as usize] >= 0)
            .map(|f| piece[f as usize] as usize)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    };
    let mut rims: Vec<Rims> = Vec::new();
    for ((axis, bval), members) in groups(&with_b) {
        let mut up_removed = HashSet::new();
        let mut dn_removed = HashSet::new();
        for &pi in &members {
            up_removed.extend(with_b[pi].up.iter().copied().filter(|f| mode_b.contains(f)));
            dn_removed.extend(
                with_b[pi]
                    .down
                    .iter()
                    .copied()
                    .filter(|f| mode_b.contains(f)),
            );
        }
        let (up_rim, dn_rim) = (rim_edges(&up_removed), rim_edges(&dn_removed));
        let keep = |(lp, c): &(Vec<u32>, bool)| (*c && lp.len() >= 3) || (!*c && lp.len() >= 2);
        let traced = |rim: &[Edge]| -> Vec<(Vec<u32>, bool)> {
            trace_edge_loops(rim)
                .into_iter()
                .flat_map(|(lp, closed)| simple_loops(&lp, closed))
                .filter(keep)
                .collect()
        };
        let (mut up, mut dn) = (traced(&up_rim), traced(&dn_rim));
        // an open end below the plane is the rim an up cap would leave
        for r in open.iter().filter(|r| r.axis == axis && r.bval == bval) {
            let side = if r.below { &mut up } else { &mut dn };
            side.push((r.lp.clone(), true));
        }
        if up.is_empty() || dn.is_empty() {
            continue;
        }
        // the strip runs an up rim forward and a down rim backward, and
        // must wind as the faces it joins: those run an up rim backward and
        // a down rim forward
        let orient = |(mut lp, closed): (Vec<u32>, bool), forward: bool| {
            if runs_forward(&lp, closed, topo, gone, orig) != forward {
                lp.reverse();
            }
            (lp, closed)
        };
        let up: Vec<(Vec<u32>, bool)> = up.into_iter().map(|l| orient(l, false)).collect();
        let dn: Vec<(Vec<u32>, bool)> = dn.into_iter().map(|l| orient(l, true)).collect();
        let [pa, pb] = perp(axis);
        let info = |(lp, _): &(Vec<u32>, bool)| {
            let mut s = [0.0f64; 3];
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for &v in lp {
                let x = verts[v as usize];
                for k in 0..3 {
                    s[k] += x[k];
                }
                for (k, c) in [x[pa], x[pb]].into_iter().enumerate() {
                    lo[k] = lo[k].min(c);
                    hi[k] = hi[k].max(c);
                }
            }
            (s.map(|c| c / lp.len() as f64), lo, hi)
        };
        // the area a closed loop, run as the faces left on it want, encloses
        // across the axis, signed by the way it turns (an open one none):
        // two sides of one surface turn alike, an outline and the wall of
        // an organelle beside it, facing the other way, do not
        let turning = |(lp, closed): &(Vec<u32>, bool)| -> f64 {
            let mut area = 0.0;
            if *closed {
                let o = verts[lp[0] as usize];
                for k in 0..lp.len() {
                    let p = verts[lp[k] as usize];
                    let q = verts[lp[(k + 1) % lp.len()] as usize];
                    area += (p[pa] - o[pa]) * (q[pb] - o[pb]) - (q[pa] - o[pa]) * (p[pb] - o[pb]);
                }
            }
            area
        };
        let mut order: Vec<usize> = (0..up.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(up[i].0.len()));
        rims.push(Rims {
            up_info: up.iter().map(info).collect(),
            dn_info: dn.iter().map(info).collect(),
            up_turn: up.iter().map(turning).collect(),
            dn_turn: dn.iter().map(turning).collect(),
            up_pieces: up.iter().map(pieces_along).collect(),
            dn_pieces: dn.iter().map(pieces_along).collect(),
            dn_used: vec![false; dn.len()],
            order,
            up,
            dn,
        });
    }
    let mut laid = Laid {
        tris: Vec::new(),
        keys: HashSet::new(),
        edges: HashSet::new(),
        n_turned: 0,
        joined: Joined((0..n_pieces).collect()),
    };
    for second in [false, true] {
        for g in &mut rims {
            let order = std::mem::take(&mut g.order);
            for ui in order {
                if !pair_rim(verts, topo, gone, orig, g, ui, second, &mut laid) && !second {
                    g.order.push(ui);
                }
            }
        }
    }
    (laid.tris, laid.n_turned)
}

/// [`stitch_mode_b`]'s pairing of up rim `ui` of group `g`, in the first
/// pass or the `second`; whether a strip was laid.
#[allow(clippy::too_many_arguments)]
fn pair_rim(
    verts: &[V3],
    topo: &Topology,
    gone: &[bool],
    orig: &[[u32; 3]],
    g: &mut Rims,
    ui: usize,
    second: bool,
    laid: &mut Laid,
) -> bool {
    let (uc, ulo, uhi) = g.up_info[ui];
    // the down loops overlapping it across the axis, nearest first
    let mut near: Vec<(usize, f64)> = g
        .dn_info
        .iter()
        .enumerate()
        .filter(|&(di, &(_, dlo, dhi))| {
            !g.dn_used[di] && (0..2).all(|k| uhi[k].min(dhi[k]) > ulo[k].max(dlo[k]))
        })
        .map(|(di, &(dc, _, _))| (di, dist(uc, dc)))
        .collect();
    near.sort_by(|x, y| x.1.total_cmp(&y.1));
    // a rung on an edge a face left there or a strip laid before already
    // has would give it a third face
    let edges = &laid.edges;
    let taken = |u: u32, v: u32| {
        edges.contains(&edge(u, v))
            || topo
                .edge_id(u, v)
                .is_some_and(|e| topo.edge_faces(e).iter().any(|&f| !gone[f as usize]))
    };
    // the first whose strip the cheaper way round the down rim one winding
    // joins to the faces left on both rims: two rims whose faces want it
    // wound both ways are no two sides of one surface (an outline beside an
    // organelle's wall, facing the other way). Two outlines far apart for
    // their size can make the twisted way the cheaper, so an up rim left
    // over then takes the dearer way to a closed down rim turning alike —
    // only to join two pieces no strip has joined: within one piece the
    // hole is the fill's.
    let mut chosen = None;
    'near: for (di, _) in near {
        if second {
            let (u, d) = (g.up_turn[ui], g.dn_turn[di]);
            let alike = (u > 0.0 && d > 0.0) || (u < 0.0 && d < 0.0);
            if !alike || !laid.joined.apart(&g.up_pieces[ui], &g.dn_pieces[di]) {
                continue;
            }
        }
        let (on_up, on_dn) = (&g.up[ui].0, &g.dn[di].0);
        let ways = zip_vertex_loops(on_up, on_dn, verts, g.up[ui].1, g.dn[di].1, &taken);
        let tried = if second { ways.len() } else { 1 };
        for (way, (_, strip)) in ways.into_iter().enumerate().take(tried) {
            // the faces left on its edges along each rim: [as the strip runs
            // the edge, against it], the up rim's and the down's
            let mut runs = [[0usize; 2]; 2];
            for t in &strip {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let side = if on_up.contains(&a) && on_up.contains(&b) {
                        0
                    } else if on_dn.contains(&a) && on_dn.contains(&b) {
                        1
                    } else {
                        continue;
                    };
                    let Some(e) = topo.edge_id(a, b) else {
                        continue;
                    };
                    for &f in topo.edge_faces(e).iter().filter(|&&f| !gone[f as usize]) {
                        let t = orig[f as usize];
                        let with = (0..3).any(|k| t[k] == a && t[(k + 1) % 3] == b);
                        runs[side][usize::from(!with)] += 1;
                    }
                }
            }
            let wants =
                |[with, against]: [usize; 2]| (with + against > 0).then_some(with > against);
            match (wants(runs[0]), wants(runs[1])) {
                (Some(u), Some(d)) if u != d => continue,
                (u, d) => {
                    chosen = Some((di, strip, u.or(d).unwrap_or(false), way > 0));
                    break 'near;
                }
            }
        }
    }
    let Some((di, strip, flip, turned)) = chosen else {
        return false;
    };
    g.dn_used[di] = true;
    laid.n_turned += usize::from(turned);
    let both: Vec<usize> = g.up_pieces[ui]
        .iter()
        .chain(&g.dn_pieces[di])
        .copied()
        .collect();
    laid.joined.join(&both);
    // a triangle a face left there already is, or one laid before, is not
    // laid again
    for t in strip {
        let t = if flip { [t[0], t[2], t[1]] } else { t };
        let mut key = t;
        key.sort_unstable();
        let left = topo.edge_id(t[0], t[1]).is_some_and(|e| {
            topo.edge_faces(e)
                .iter()
                .any(|&f| !gone[f as usize] && orig[f as usize].contains(&t[2]))
        });
        if !left && laid.keys.insert(key) {
            laid.edges.extend(face_edges(t));
            laid.tris.push(t);
        }
    }
    true
}

/// What [`remove_parallel_patches`] did.
pub struct Removed {
    /// The full face list: removed faces `[0, 0, 0]`, stitch faces after.
    pub faces: Vec<[u32; 3]>,
    pub n_removed: usize,
    pub n_fold: usize,
    pub n_parallel: usize,
    pub n_orphan: usize,
    pub n_stitched: usize,
    /// Strips laid round the dearer way, the cheaper one wound against
    /// the faces on a rim.
    pub n_turned: usize,
    /// Caps facing an open end across a seam, taken out and their rims
    /// zipped to it.
    pub n_open: usize,
}

/// The chunk planes, and what makes a face a cap on one, as
/// [`find_parallel_patches`] reads them: its centroid within
/// `boundary_dist` of the plane, its normal's component along the axis
/// over `normal_thresh`.
#[derive(Debug, Clone, Copy)]
pub struct Seam<'a> {
    pub boundaries: &'a [(usize, Vec<f64>)],
    pub boundary_dist: f64,
    pub normal_thresh: f64,
}

/// `remove_parallel_patches` on already trimmed patches: classify each
/// (fold or parallel), remove them all, drop pieces the removal cut off
/// main that border only fold patches, then stitch the parallel sides —
/// and the caps facing an open end across a seam ([`open_ends_on_seams`])
/// with them.  The loops over every face run on [`threads`]' pool.
pub fn remove_parallel_patches(m: &PatchMesh<'_>, patches: &[Patch], seam: &Seam<'_>) -> Removed {
    threads::install(|| remove_patches(m, patches, seam))
}

/// An open end of the mesh on a seam, to be zipped like the rim a removed
/// cap leaves; `bval` is the plane's rounded as a patch's is.
pub struct OpenRim {
    pub axis: usize,
    pub bval: i64,
    pub lp: Vec<u32>,
    /// Its faces lie below the plane: the side whose caps face up.
    pub below: bool,
}

/// The open ends on the seams that a cap of another piece faces across
/// the plane, and those caps as patches of one side.
///
/// Marching cubes run chunk by chunk closes a piece crossing a chunk's
/// face with a cap there, or leaves it open; the other chunk's piece
/// meets it across the plane. Where a cap faces a cap the two are a
/// parallel patch; where it faces an open end, the open end is the other
/// side, and nothing pairs them. An open end is on a plane when most of
/// its loop's vertices lie within `boundary_dist` of it (a ragged end may
/// leave the plane along part of its rim) — the first such plane, as
/// `seam.boundaries` lists them — and is on the side most of the faces
/// along its edges on the plane are. A cap facing it: faces of
/// another piece (`label`: face-edge components of the mesh before the
/// removal), not `removed`, whose centroids lie within `boundary_dist` of
/// the plane and inside the loop's outline across the axis, their normals
/// along the axis over `normal_thresh` toward the loop's side — only the
/// part of a flat sheet the open end covers, so a sheet that carries the
/// surface beyond it stays — of a piece across the plane: where the sheet
/// of its caps there turns off the plane, most faces lie on the far side.
/// (An organelle inside the open piece, a cavity wound into itself, can
/// have a face there facing the open end, its walls rising on the open
/// end's side.) Each piece's caps are a patch of their own.
pub fn open_ends_on_seams(
    m: &PatchMesh<'_>,
    topo: &Topology,
    label: &[i64],
    removed: &[bool],
    seam: &Seam<'_>,
) -> (Vec<Patch>, Vec<OpenRim>) {
    let planes: Vec<(usize, f64)> = seam
        .boundaries
        .iter()
        .flat_map(|(axis, values)| values.iter().map(move |&b| (*axis, b)))
        .collect();
    let mut found: (Vec<Patch>, Vec<OpenRim>) = (Vec::new(), Vec::new());
    let open: Vec<Edge> = (0..topo.n_edges())
        .filter(|&e| {
            let fs = topo.edge_faces(e);
            fs.len() == 1 && !removed[fs[0] as usize]
        })
        .map(|e| {
            let [a, b] = topo.edges()[e];
            (a, b)
        })
        .collect();
    if planes.is_empty() || open.is_empty() {
        return found;
    }
    let cent = centroids(m);
    // per plane, the faces that could cap it, facing up and facing down:
    // found for a plane when an open end first lies on it
    let mut caps_on: HashMap<usize, [Vec<u32>; 2]> = HashMap::new();
    let caps_at = |pi: usize| -> [Vec<u32>; 2] {
        let (axis, b) = planes[pi];
        let mut by = [Vec::new(), Vec::new()];
        for f in 0..m.faces.len() {
            let on = (cent[f][axis] - b).abs() < seam.boundary_dist;
            if removed[f] || label[f] < 0 || !on {
                continue;
            }
            let n = m.normals[f][axis];
            if n > seam.normal_thresh {
                by[0].push(f as u32);
            } else if n < -seam.normal_thresh {
                by[1].push(f as u32);
            }
        }
        by
    };
    let mut taken = vec![false; m.faces.len()];
    for (lp, closed) in trace_edge_loops(&open)
        .into_iter()
        .flat_map(|(lp, closed)| simple_loops(&lp, closed))
    {
        let n = lp.len();
        if !closed || n < 3 {
            continue;
        }
        // each edge's one face
        let along: Vec<(usize, u32)> = (0..n)
            .filter_map(|i| {
                topo.edge_id(lp[i], lp[(i + 1) % n])
                    .map(|e| (i, topo.edge_faces(e)[0]))
            })
            .collect();
        let Some(&(_, first)) = along.first() else {
            continue;
        };
        let own = label[first as usize];
        for (pi, &(axis, b)) in planes.iter().enumerate() {
            let near = |v: u32| (m.verts[v as usize][axis] - b).abs() < seam.boundary_dist;
            if 2 * lp.iter().filter(|&&v| near(v)).count() <= n {
                continue;
            }
            let (mut above, mut under) = (0usize, 0usize);
            for &(i, f) in &along {
                if near(lp[i]) && near(lp[(i + 1) % n]) {
                    let off = cent[f as usize][axis] - b;
                    above += usize::from(off > 0.0);
                    under += usize::from(off < 0.0);
                }
            }
            if above == under {
                continue;
            }
            let below = under > above;
            let [p, q] = perp(axis);
            let ring: Vec<[f64; 2]> = lp
                .iter()
                .map(|&v| [m.verts[v as usize][p], m.verts[v as usize][q]])
                .collect();
            // the caps facing it: up when it is above, down when below
            let mut by_piece: BTreeMap<i64, Vec<u32>> = BTreeMap::new();
            let dir = usize::from(below);
            let caps: &[Vec<u32>; 2] = caps_on.entry(pi).or_insert_with(|| caps_at(pi));
            for &f in &caps[dir] {
                let fu = f as usize;
                if !taken[fu] && label[fu] != own && inside_ring([cent[fu][p], cent[fu][q]], &ring)
                {
                    by_piece.entry(label[fu]).or_default().push(f);
                }
            }
            // a piece across the plane: where the sheet of its caps there
            // turns off the plane, its faces run away from the open end
            let across = |start: &[u32]| -> bool {
                let mut sheet: HashSet<u32> = start.iter().copied().collect();
                let mut walls: HashSet<u32> = HashSet::new();
                let mut stack = start.to_vec();
                let (mut far, mut near) = (0usize, 0usize);
                while let Some(f) = stack.pop() {
                    for e in topo.face_edges(f as usize).into_iter().flatten() {
                        for &g in topo.edge_faces(e as usize) {
                            if removed[g as usize] || sheet.contains(&g) {
                                continue;
                            }
                            if caps[dir].binary_search(&g).is_ok() {
                                sheet.insert(g);
                                stack.push(g);
                            } else if walls.insert(g) {
                                let off = cent[g as usize][axis] - b;
                                let (under, over) = (off < 0.0, off > 0.0);
                                far += usize::from(if below { over } else { under });
                                near += usize::from(if below { under } else { over });
                            }
                        }
                    }
                }
                far > near
            };
            by_piece.retain(|_, fs| across(fs));
            if by_piece.is_empty() {
                continue;
            }
            let bval = round_i64(b);
            for caps in by_piece.into_values() {
                for &f in &caps {
                    taken[f as usize] = true;
                }
                let (up, down) = if below {
                    (Vec::new(), caps.clone())
                } else {
                    (caps.clone(), Vec::new())
                };
                found.0.push(Patch {
                    axis,
                    bval,
                    faces: caps,
                    up,
                    down,
                });
            }
            found.1.push(OpenRim {
                axis,
                bval,
                lp: lp.clone(),
                below,
            });
            break;
        }
    }
    found
}

/// Whether `pt` lies inside the polygon `ring` (even-odd crossings).
fn inside_ring(pt: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = ring.len();
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if (a[1] > pt[1]) != (b[1] > pt[1]) {
            let x = a[0] + (pt[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if pt[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

fn remove_patches(m: &PatchMesh<'_>, patches: &[Patch], seam: &Seam) -> Removed {
    let nf = m.faces.len();
    let nv = m.verts.len();
    let topo = Topology::new(m.faces, nv);
    let (before, main_before) = face_edge_components(&topo);
    let mut in_patch = vec![false; nf];
    for p in patches {
        for &f in &p.faces {
            in_patch[f as usize] = true;
        }
    }
    let (caps, open) = open_ends_on_seams(m, &topo, &before, &in_patch, seam);
    let n_given = patches.len();
    let patches: Vec<Patch> = patches.iter().cloned().chain(caps).collect();
    let patches = &patches[..];
    let mut mode_a: HashSet<u32> = HashSet::new();
    let mut mode_b: HashSet<u32> = HashSet::new();
    let mut removed = vec![false; nf];
    let (mut n_a, mut n_b) = (0, 0);
    for p in patches {
        let fold = fold_faces(m, p);
        for &f in &p.faces {
            removed[f as usize] = true;
        }
        if !p.up.is_empty() && !p.down.is_empty() {
            mode_b.extend(p.faces.iter().copied());
            n_b += 1;
            mode_a.extend(fold);
        } else if !fold.is_empty() {
            mode_a.extend(p.faces.iter().copied());
            n_a += 1;
        } else {
            mode_b.extend(p.faces.iter().copied());
            n_b += 1;
        }
    }
    let rebuild = |drop: &[bool]| -> Vec<[u32; 3]> {
        m.faces
            .iter()
            .enumerate()
            .map(|(f, &t)| if drop[f] { [0, 0, 0] } else { t })
            .collect()
    };
    let mut faces = rebuild(&removed);

    // the removed faces are collapsed to [0, 0, 0], so the mesh after has
    // the topology before without them
    let (after, main_after) = face_edge_components_without(&topo, &removed);
    let main_before = main_before as i64;
    let main_after = main_after as i64;
    let newly: Vec<u32> = (0..nf as u32)
        .into_par_iter()
        .filter(|&f| {
            let f = f as usize;
            before[f] == main_before && !removed[f] && after[f] != main_after
        })
        .collect();
    let mut n_orphan = 0;
    let mut gone = removed.clone();
    if !newly.is_empty() {
        // an edge of a removed face is "b" if any removed face on it is
        let mut edge_b: HashMap<Edge, bool> = HashMap::new();
        for f in (0..nf as u32).filter(|&f| removed[f as usize]) {
            let is_b = !mode_a.contains(&f);
            for e in face_edges(m.faces[f as usize]) {
                let t = edge_b.entry(e).or_insert(is_b);
                *t |= is_b;
            }
        }
        // the faces of each component holding a newly cut off face
        let n_labels = after.iter().max().map_or(0, |&l| (l + 1).max(0) as usize);
        let mut slot = vec![usize::MAX; n_labels];
        let mut members: Vec<Vec<u32>> = Vec::new();
        for &f in &newly {
            let c = after[f as usize] as usize;
            if slot[c] == usize::MAX {
                slot[c] = members.len();
                members.push(Vec::new());
            }
        }
        for (f, &c) in after.iter().enumerate() {
            if c >= 0 && slot[c as usize] != usize::MAX {
                members[slot[c as usize]].push(f as u32);
            }
        }
        for comp in members {
            let borders_b = comp.iter().any(|&f| {
                face_edges(m.faces[f as usize])
                    .iter()
                    .any(|e| edge_b.get(e).copied().unwrap_or(false))
            });
            if !borders_b {
                for f in comp {
                    gone[f as usize] = true;
                    n_orphan += 1;
                }
            }
        }
        if n_orphan > 0 {
            faces = rebuild(&gone);
        }
    }
    let mut n_stitched = 0;
    let mut n_turned = 0;
    let mut n_open = 0;
    if !mode_b.is_empty() {
        let (new, turned) = stitch_mode_b(m.verts, &topo, &gone, m.faces, &mode_b, patches, &open);
        // a cap whose rim no strip joins is put back: the open end it
        // faces stays as it was, and so does the piece it closes
        let laid: HashSet<Edge> = new.iter().flat_map(|&t| face_edges(t)).collect();
        for p in &patches[n_given..] {
            let rim = rim_left(&topo, &gone, m.faces, &p.faces);
            if rim.iter().any(|e| laid.contains(e)) {
                n_open += 1;
            } else {
                for &f in &p.faces {
                    removed[f as usize] = false;
                    faces[f as usize] = m.faces[f as usize];
                }
                n_b -= 1;
            }
        }
        n_stitched = new.len();
        n_turned = turned;
        faces.extend(new);
    }
    let n_removed = removed.iter().filter(|&&r| r).count();
    Removed {
        faces,
        n_removed,
        n_fold: n_a,
        n_parallel: n_b,
        n_orphan,
        n_stitched,
        n_turned,
        n_open,
    }
}

/// The edges of `side`'s faces that exactly one face not `gone` has: the
/// rim their removal leaves.
fn rim_left(topo: &Topology, gone: &[bool], orig: &[[u32; 3]], side: &[u32]) -> Vec<Edge> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for &fi in side {
        let f = orig[fi as usize];
        for i in 0..3 {
            let (a, b) = (f[i], f[(i + 1) % 3]);
            if a == b || !seen.insert(edge(a, b)) {
                continue;
            }
            let left = topo.edge_id(a, b).map_or(0, |e| {
                topo.edge_faces(e)
                    .iter()
                    .filter(|&&f| !gone[f as usize])
                    .count()
            });
            if left == 1 {
                out.push(edge(a, b));
            }
        }
    }
    out
}

/// One axis' chunk boundaries: the coordinates, and at each the vertex
/// count and its spike over the local background.
#[derive(Debug, Clone, PartialEq)]
pub struct Boundaries {
    pub axis: usize,
    pub values: Vec<i64>,
    pub counts: Vec<i64>,
    pub spikes: Vec<f64>,
}

/// `find_chunk_boundaries`: per axis, the integer coordinates (the
/// vertices' rounded half to even) where the vertex count spikes — over
/// five times the mean count of the five coordinates on either side — less
/// the weaker of two spikes closer than a quarter of the median spacing.
/// An axis with fewer than three coordinates or no spike has none.
pub fn find_chunk_boundaries(verts: &[V3]) -> Vec<Boundaries> {
    // the axes at once, each one's coordinates sorted in parallel (equal
    // values are equal, so the order is the same however it is sorted)
    threads::install(|| {
        (0..3)
            .into_par_iter()
            .filter_map(|axis| axis_boundaries(verts, axis))
            .collect()
    })
}

/// [`find_chunk_boundaries`] on one axis.
fn axis_boundaries(verts: &[V3], axis: usize) -> Option<Boundaries> {
    let mut coords: Vec<i64> = verts
        .par_iter()
        .map(|v| v[axis].round_ties_even() as i64)
        .collect();
    coords.par_sort_unstable();
    let mut unique: Vec<i64> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    for c in coords {
        if unique.last() == Some(&c) {
            *counts.last_mut().expect("a count per value") += 1;
        } else {
            unique.push(c);
            counts.push(1);
        }
    }
    let n = unique.len();
    if n < 3 {
        return None;
    }
    let spike: Vec<f64> = (0..n)
        .map(|i| {
            let (lo, hi) = (i.saturating_sub(5), (i + 6).min(n));
            let around = (hi - lo - 1) as f64;
            let sum: i64 = counts[lo..hi].iter().sum::<i64>() - counts[i];
            let bg = if around > 0.0 {
                sum as f64 / around
            } else {
                1.0
            };
            counts[i] as f64 / bg.max(1.0)
        })
        .collect();
    let found: Vec<usize> = (0..n).filter(|&i| spike[i] > 5.0).collect();
    if found.is_empty() {
        return None;
    }
    let mut keep = vec![true; found.len()];
    if found.len() >= 3 {
        let spacings: Vec<i64> = found
            .windows(2)
            .map(|w| unique[w[1]] - unique[w[0]])
            .collect();
        let mut sorted = spacings.clone();
        sorted.sort_unstable();
        let m = sorted.len();
        let median = if m % 2 == 1 {
            sorted[m / 2] as f64
        } else {
            (sorted[m / 2 - 1] as f64 + sorted[m / 2] as f64) / 2.0
        };
        let close = median * 0.25;
        for i in 0..found.len() {
            let left = if i > 0 {
                spacings[i - 1] as f64
            } else {
                median
            };
            let right = if i < spacings.len() {
                spacings[i] as f64
            } else {
                median
            };
            if left.min(right) >= close {
                continue;
            }
            // of two spikes too close together, the weaker goes
            let this = spike[found[i]];
            if left < close && i > 0 && this < spike[found[i - 1]] {
                keep[i] = false;
            }
            if right < close && i < spacings.len() && this < spike[found[i + 1]] {
                keep[i] = false;
            }
        }
    }
    let kept: Vec<usize> = found
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(&i, _)| i)
        .collect();
    Some(Boundaries {
        axis,
        values: kept.iter().map(|&i| unique[i]).collect(),
        counts: kept.iter().map(|&i| counts[i]).collect(),
        spikes: kept.iter().map(|&i| spike[i]).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_chains_are_walked_from_either_end() {
        assert_eq!(
            trace_edge_loops(&[(0, 5), (0, 1)]),
            vec![(vec![1, 0, 5], false)]
        );
        assert_eq!(
            trace_edge_loops(&[(1, 3), (0, 1), (0, 2)]),
            vec![(vec![2, 0, 1, 3], false)]
        );
        assert_eq!(
            trace_edge_loops(&[(0, 1), (1, 2), (0, 2)]),
            vec![(vec![0, 1, 2], true)]
        );
    }

    #[test]
    fn a_figure_eight_is_two_closed_loops() {
        assert_eq!(
            trace_edge_loops(&[(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)]),
            vec![(vec![0, 1, 2], true), (vec![0, 3, 4], true)]
        );
    }

    #[test]
    fn a_rim_through_a_vertex_twice_is_split_into_its_lobes() {
        // a closed rim with a lobe 2-3-4 at vertex 2
        assert_eq!(
            simple_loops(&[1, 2, 3, 4, 2, 5], true),
            vec![(vec![2, 3, 4], true), (vec![1, 2, 5], true)]
        );
        // lobes within lobes: each its own loop
        assert_eq!(
            simple_loops(&[1, 2, 3, 6, 7, 3, 4, 2, 5], true),
            vec![
                (vec![3, 6, 7], true),
                (vec![2, 3, 4], true),
                (vec![1, 2, 5], true)
            ]
        );
        // an open chain keeps its ends
        assert_eq!(
            simple_loops(&[1, 2, 3, 4, 2, 5], false),
            vec![(vec![2, 3, 4], true), (vec![1, 2, 5], false)]
        );
        assert_eq!(simple_loops(&[1, 2, 3], true), vec![(vec![1, 2, 3], true)]);
    }

    #[test]
    fn zipping_two_squares_consumes_every_rim_edge_once() {
        let verts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let ways = zip_vertex_loops(&[0, 1, 2, 3], &[4, 5, 6, 7], &verts, true, true, &|_, _| {
            false
        });
        // both ways round, the cheaper (the loops alike, unturned) first
        assert_eq!(ways.len(), 2);
        assert!(ways[0].0 < ways[1].0);
        let tris = ways[0].1.clone();
        assert_eq!(tris.len(), 8);
        let mut rim: HashMap<Edge, usize> = HashMap::new();
        for t in &tris {
            for e in face_edges(*t) {
                *rim.entry(e).or_insert(0) += 1;
            }
        }
        for e in [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 3),
            (4, 5),
            (5, 6),
            (6, 7),
            (4, 7),
        ] {
            assert_eq!(rim[&e], 1, "{e:?}");
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    /// Two square tubes along x meeting at the chunk boundary x = 0, each
    /// closed there by a cap of its own: the left one's (up, facing +x)
    /// over its square of half-width 1, the right one's (down, facing -x)
    /// over `right` (its outline at x = 0, from y, z pairs). Every face
    /// faces out. The vertices of the right outline at a left corner are
    /// the left tube's. Returns the mesh's vertices and faces and the
    /// sandwich's patch.
    fn sandwich(right: &[[f64; 2]]) -> (Vec<V3>, Vec<[u32; 3]>, Patch) {
        let square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let mut v: Vec<V3> = Vec::new();
        for x in [-2.0, 0.0] {
            for [y, z] in square {
                v.push([x, y, z]);
            }
        }
        let shared = |p: [f64; 2]| square.iter().position(|q| *q == p).map(|i| 4 + i as u32);
        let mut ring_r: Vec<u32> = Vec::new();
        for &p in right {
            ring_r.push(shared(p).unwrap_or_else(|| {
                v.push([0.0, p[0], p[1]]);
                (v.len() - 1) as u32
            }));
        }
        let far_r: Vec<u32> = right
            .iter()
            .map(|&[y, z]| {
                v.push([2.0, y, z]);
                (v.len() - 1) as u32
            })
            .collect();
        let mut f: Vec<[u32; 3]> = Vec::new();
        // sides: rings run anticlockwise about +x, so [a, b, d] faces out
        let band = |f: &mut Vec<[u32; 3]>, lo: &[u32], hi: &[u32]| {
            let n = lo.len();
            for i in 0..n {
                let j = (i + 1) % n;
                f.push([lo[i], lo[j], hi[j]]);
                f.push([lo[i], hi[j], hi[i]]);
            }
        };
        band(&mut f, &[0, 1, 2, 3], &[4, 5, 6, 7]);
        band(&mut f, &ring_r, &far_r);
        // the far ends, capped facing out
        f.push([0, 2, 1]);
        f.push([0, 3, 2]);
        for i in 1..far_r.len() - 1 {
            f.push([far_r[0], far_r[i], far_r[i + 1]]);
        }
        // the sandwich: the left cap facing +x, the right one facing -x,
        // fanned from a vertex of its own (a cap sharing an edge with the
        // left one would be a fold, not a sandwich)
        let up_from = f.len() as u32;
        f.push([4, 5, 6]);
        f.push([4, 6, 7]);
        let down_from = f.len() as u32;
        let n = ring_r.len();
        let o = (0..n)
            .find(|&i| ring_r[i] > 7)
            .expect("a vertex of its own");
        for i in 1..n - 1 {
            let (a, b) = (ring_r[(o + i) % n], ring_r[(o + i + 1) % n]);
            f.push([ring_r[o], b, a]);
        }
        let up: Vec<u32> = (up_from..down_from).collect();
        let down: Vec<u32> = (down_from..f.len() as u32).collect();
        let patch = Patch {
            axis: 0,
            bval: 0,
            faces: up.iter().chain(&down).copied().collect(),
            up,
            down,
        };
        (v, f, patch)
    }

    fn removed_and_closed(v: &[V3], f: &[[u32; 3]], patch: Patch) -> (Vec<[u32; 3]>, usize) {
        let done = removed(v, f, patch);
        (done.faces, done.n_stitched)
    }

    fn removed(v: &[V3], f: &[[u32; 3]], patch: Patch) -> Removed {
        removed_on(v, f, &[patch], &[])
    }

    /// The removal with the chunk planes `boundaries`, a cap a face within
    /// 30 of one whose unit normal along it is over 0.99.
    fn removed_on(
        v: &[V3],
        f: &[[u32; 3]],
        patches: &[Patch],
        boundaries: &[(usize, Vec<f64>)],
    ) -> Removed {
        let normals: Vec<V3> = f
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| v[i as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    u[1] * w[2] - u[2] * w[1],
                    u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0],
                ];
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                n.map(|x| x / l)
            })
            .collect();
        let m = PatchMesh {
            faces: f,
            verts: v,
            normals: &normals,
        };
        let seam = Seam {
            boundaries,
            boundary_dist: 30.0,
            normal_thresh: 0.99,
        };
        remove_parallel_patches(&m, patches, &seam)
    }

    /// Square tubes along x either side of the chunk boundary x = 0, every
    /// face facing out, each closed at its far end: on the left one per
    /// `(centre, half-width)` of `left`, capped facing +x at x = -10; on
    /// the right one of half-width `right` round the axis, left open at
    /// x = 10.
    fn caps_facing_an_open_end(left: &[([f64; 2], f64)], right: f64) -> (Vec<V3>, Vec<[u32; 3]>) {
        let mut v: Vec<V3> = Vec::new();
        let mut f: Vec<[u32; 3]> = Vec::new();
        // a ring running anticlockwise about +x
        let ring = |v: &mut Vec<V3>, x: f64, c: [f64; 2], h: f64| -> [u32; 4] {
            let at = v.len() as u32;
            for [y, z] in [[-h, -h], [h, -h], [h, h], [-h, h]] {
                v.push([x, c[0] + y, c[1] + z]);
            }
            [at, at + 1, at + 2, at + 3]
        };
        let band = |f: &mut Vec<[u32; 3]>, lo: [u32; 4], hi: [u32; 4]| {
            for i in 0..4 {
                let j = (i + 1) % 4;
                f.push([lo[i], lo[j], hi[j]]);
                f.push([lo[i], hi[j], hi[i]]);
            }
        };
        for &(c, h) in left {
            let (far, near) = (ring(&mut v, -400.0, c, h), ring(&mut v, -10.0, c, h));
            band(&mut f, far, near);
            f.push([far[0], far[2], far[1]]);
            f.push([far[0], far[3], far[2]]);
            f.push([near[0], near[1], near[2]]);
            f.push([near[0], near[2], near[3]]);
        }
        let open = ring(&mut v, 10.0, [0.0, 0.0], right);
        let far = ring(&mut v, 400.0, [0.0, 0.0], right);
        band(&mut f, open, far);
        f.push([far[0], far[1], far[2]]);
        f.push([far[0], far[2], far[3]]);
        (v, f)
    }

    fn closed(v: &[V3], f: &[[u32; 3]]) -> bool {
        let t = Topology::new(f, v.len());
        (0..t.n_edges()).all(|e| t.edge_faces(e).len() == 2)
    }

    #[test]
    fn a_cap_facing_an_open_end_across_the_seam_is_zipped_to_it() {
        // the left tube capped at the boundary, the right one left open
        // there round it: no patch pairs them, the cap and the open end do
        let (v, f) = caps_facing_an_open_end(&[([0.0, 0.0], 100.0)], 150.0);
        assert!(!closed(&v, &f), "fixture: the right tube is open");
        let done = removed_on(&v, &f, &[], &[(0, vec![0.0])]);
        assert_eq!((done.n_open, done.n_removed, done.n_stitched), (1, 2, 8));
        assert_eq!(done.faces[10..12], [[0, 0, 0]; 2], "the cap is out");
        let c = mesh::validity::check(&v, &done.faces);
        assert!(c.is_valid(), "{c:?}");
        assert!(closed(&v, &done.faces));
        // without the plane there is no seam to zip across
        let none = removed_on(&v, &f, &[], &[(0, vec![500.0])]);
        assert_eq!((none.n_open, none.n_removed), (0, 0));
        assert_eq!(none.faces, f);
    }

    #[test]
    fn a_cavity_inside_the_open_piece_is_no_cap() {
        // an organelle in the right tube by its open end, wound into itself:
        // its face at the boundary faces the open end from inside the
        // outline, but its walls run on the open end's side of the plane
        let (mut v, mut f) = caps_facing_an_open_end(&[], 150.0);
        let at = v.len() as u32;
        for x in [5.0, 200.0] {
            for [y, z] in [[-50.0, -50.0], [50.0, -50.0], [50.0, 50.0], [-50.0, 50.0]] {
                v.push([x, y, z]);
            }
        }
        let (near, far) = (
            [at, at + 1, at + 2, at + 3],
            [at + 4, at + 5, at + 6, at + 7],
        );
        let mut out = Vec::new();
        for i in 0..4 {
            let j = (i + 1) % 4;
            out.push([near[i], near[j], far[j]]);
            out.push([near[i], far[j], far[i]]);
        }
        out.extend([[far[0], far[1], far[2]], [far[0], far[2], far[3]]]);
        out.extend([[near[0], near[2], near[1]], [near[0], near[3], near[2]]]);
        f.extend(out.iter().map(|t| [t[0], t[2], t[1]]));
        let done = removed_on(&v, &f, &[], &[(0, vec![0.0])]);
        assert_eq!((done.n_open, done.n_removed), (0, 0));
        assert_eq!(done.faces, f);
    }

    #[test]
    fn a_cap_whose_rim_no_strip_joins_is_put_back() {
        // two pieces capped inside the one open end: the larger one's rim
        // takes it, and the other's cap stays where it was
        let left = [([-150.0, 0.0], 100.0), ([170.0, 0.0], 60.0)];
        let (v, f) = caps_facing_an_open_end(&left, 300.0);
        let done = removed_on(&v, &f, &[], &[(0, vec![0.0])]);
        assert_eq!((done.n_open, done.n_removed, done.n_stitched), (1, 2, 8));
        assert_eq!(done.n_parallel, 1);
        assert_eq!(done.faces[10..12], [[0, 0, 0]; 2]);
        assert_eq!(done.faces[22..24], f[22..24], "the smaller cap is back");
        let c = mesh::validity::check(&v, &done.faces);
        assert!(c.is_valid(), "{c:?}");
        assert!(closed(&v, &done.faces));
    }

    #[test]
    fn the_two_sides_are_zipped_shut() {
        // the right tube wider: its outline runs round the left one's, and
        // the zip joins the two into one closed surface
        let right = [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]];
        let (v, f, patch) = sandwich(&right);
        let (out, n) = removed_and_closed(&v, &f, patch);
        assert_eq!(n, 8);
        let c = mesh::validity::check(&v, &out);
        assert!(c.is_valid(), "{c:?}");
        // closed: every edge has two faces
        let t = Topology::new(&out, v.len());
        assert!((0..t.n_edges()).all(|e| t.edge_faces(e).len() == 2));
    }

    #[test]
    fn a_strip_the_cheaper_way_round_against_a_rim_is_laid_the_other_way() {
        // the right outline long and thin, off to one side and turned: the
        // cheaper strip round it twists against the faces on one rim, the
        // other way round joins both, and the two sides close
        let right: Vec<[f64; 2]> = (0..4)
            .map(|k| {
                let a = std::f64::consts::FRAC_PI_2 * f64::from(k) + 1.3;
                [0.9 + 2.0 * a.cos(), 0.6 * a.sin()]
            })
            .collect();
        let (v, f, patch) = sandwich(&right);
        let done = removed(&v, &f, patch);
        assert_eq!(
            done.n_turned, 1,
            "fixture: the cheaper way is wound against a rim"
        );
        assert_eq!(done.n_stitched, 8);
        let c = mesh::validity::check(&v, &done.faces);
        assert!(c.is_valid(), "{c:?}");
        let t = Topology::new(&done.faces, v.len());
        assert!((0..t.n_edges()).all(|e| t.edge_faces(e).len() == 2));
    }

    #[test]
    fn the_dearer_way_round_joins_pieces_and_leaves_a_hole_in_one_alone() {
        // the same two sides, the tubes also joined at their far ends: one
        // piece, and the twisted way round is not laid to close a hole
        let right: Vec<[f64; 2]> = (0..4)
            .map(|k| {
                let a = std::f64::consts::FRAC_PI_2 * f64::from(k) + 1.3;
                [0.9 + 2.0 * a.cos(), 0.6 * a.sin()]
            })
            .collect();
        let (v, mut f, patch) = sandwich(&right);
        let far = (v.len() - 4) as u32;
        f.extend([[0, 1, far], [1, far + 1, far]]);
        let (label, _) = face_edge_components(&Topology::new(&f, v.len()));
        assert_eq!(label[0], label[f.len() - 7], "fixture: one piece");
        let done = removed(&v, &f, patch);
        assert_eq!((done.n_removed, done.n_turned, done.n_stitched), (4, 0, 0));
    }

    /// Square tubes along x either side of the chunk boundary x = 0, every
    /// face facing out: on the left one per `(centre, half-width)` of
    /// `left`, capped facing +x at x = -10 (12 faces each, the cap the last
    /// two); on the right one round `outer`, its end at x = 10 an annulus
    /// facing -x round the mouth of an organelle running into it, square
    /// round `organelle`, its walls facing into it — one surface, the
    /// marching-cubes cap of a piece holding an organelle across a chunk
    /// face. Returns the vertices, the faces, the patch of the caps across
    /// x = 0, and the first faces of the right tube's walls and of the
    /// organelle's.
    fn caps_over_an_organelle(
        left: &[([f64; 2], f64)],
        outer: ([f64; 2], f64),
        organelle: ([f64; 2], f64),
    ) -> (Vec<V3>, Vec<[u32; 3]>, Patch, usize, usize) {
        let mut v: Vec<V3> = Vec::new();
        let mut f: Vec<[u32; 3]> = Vec::new();
        // a ring running anticlockwise about +x
        let ring = |v: &mut Vec<V3>, x: f64, (c, h): ([f64; 2], f64)| -> [u32; 4] {
            let at = v.len() as u32;
            for [y, z] in [[-h, -h], [h, -h], [h, h], [-h, h]] {
                v.push([x, c[0] + y, c[1] + z]);
            }
            [at, at + 1, at + 2, at + 3]
        };
        // walls from ring `lo` to ring `hi` along +x, facing out
        let band = |f: &mut Vec<[u32; 3]>, lo: [u32; 4], hi: [u32; 4]| {
            for i in 0..4 {
                let j = (i + 1) % 4;
                f.push([lo[i], lo[j], hi[j]]);
                f.push([lo[i], hi[j], hi[i]]);
            }
        };
        let mut up = Vec::new();
        for &tube in left {
            let (far, near) = (ring(&mut v, -400.0, tube), ring(&mut v, -10.0, tube));
            band(&mut f, far, near);
            f.extend([[far[0], far[2], far[1]], [far[0], far[3], far[2]]]);
            up.extend([f.len() as u32, f.len() as u32 + 1]);
            f.extend([[near[0], near[1], near[2]], [near[0], near[2], near[3]]]);
        }
        let (rim, end) = (ring(&mut v, 10.0, outer), ring(&mut v, 400.0, outer));
        let walls = f.len();
        band(&mut f, rim, end);
        f.extend([[end[0], end[1], end[2]], [end[0], end[2], end[3]]]);
        let (mouth, deep) = (
            ring(&mut v, 10.0, organelle),
            ring(&mut v, 300.0, organelle),
        );
        let mut down = Vec::new();
        for i in 0..4 {
            let j = (i + 1) % 4;
            down.extend([f.len() as u32, f.len() as u32 + 1]);
            f.extend([[rim[i], mouth[j], rim[j]], [rim[i], mouth[i], mouth[j]]]);
        }
        let inside = f.len();
        let mut into = Vec::new();
        band(&mut into, mouth, deep);
        f.extend(into.iter().map(|t| [t[0], t[2], t[1]]));
        f.extend([[deep[0], deep[2], deep[1]], [deep[0], deep[3], deep[2]]]);
        let patch = Patch {
            axis: 0,
            bval: 0,
            faces: up.iter().chain(&down).copied().collect(),
            up,
            down,
        };
        (v, f, patch, walls, inside)
    }

    #[test]
    fn a_rim_turning_the_other_way_leaves_the_up_rim_to_one_turning_alike() {
        // the organelle's mouth, nearer the left rim than the right tube's
        // outline, turns the other way: the cheaper strip to it twists
        // against the faces on one rim, and the outline, the cheaper way
        // round, joins the tubes
        let (v, f, patch, walls, inside) = caps_over_an_organelle(
            &[([0.0, 0.0], 100.0)],
            ([40.0, 0.0], 150.0),
            ([0.0, 0.0], 30.0),
        );
        assert!(closed(&v, &f), "fixture: every piece closed");
        let done = removed(&v, &f, patch);
        assert_eq!((done.n_removed, done.n_turned, done.n_stitched), (10, 0, 8));
        let (label, _) = face_edge_components(&Topology::new(&done.faces, v.len()));
        assert_eq!(label[0], label[walls], "the tubes are one surface");
        assert_ne!(label[0], label[inside], "the organelle is apart");
    }

    #[test]
    fn a_rim_left_over_takes_no_rim_turning_the_other_way() {
        // a second left tube in the organelle's mouth, the outline taken by
        // the first: the dearer way round to the mouth winds as the faces
        // on both want, but the two turn apart, and no strip is laid
        let left = [([-150.0, 0.0], 100.0), ([60.0, 0.0], 20.0)];
        let (v, f, patch, walls, inside) =
            caps_over_an_organelle(&left, ([0.0, 0.0], 300.0), ([60.0, 0.0], 30.0));
        let done = removed(&v, &f, patch);
        assert_eq!((done.n_removed, done.n_turned, done.n_stitched), (12, 0, 8));
        let (label, _) = face_edge_components(&Topology::new(&done.faces, v.len()));
        assert_eq!(label[0], label[walls], "the first tube joined");
        assert_ne!(label[12], label[inside], "the second not to the organelle");
        assert_ne!(label[12], label[0]);
    }

    #[test]
    fn outlines_touching_at_corners_are_zipped_shut_through_them() {
        // the right outline runs through the left square's corners, bulging
        // out between: the strip pinches through each shared corner
        let right = [
            [-1.0, -1.0],
            [0.0, -1.5],
            [1.0, -1.0],
            [1.5, 0.0],
            [1.0, 1.0],
            [0.0, 1.5],
            [-1.0, 1.0],
            [-1.5, 0.0],
        ];
        let (v, f, patch) = sandwich(&right);
        let (out, n) = removed_and_closed(&v, &f, patch);
        assert_eq!(n, 4, "one triangle a bulge");
        let c = mesh::validity::check(&v, &out);
        assert!(c.is_valid(), "{c:?}");
        let t = Topology::new(&out, v.len());
        assert!((0..t.n_edges()).all(|e| t.edge_faces(e).len() == 2));
    }

    #[test]
    fn a_plane_of_vertices_is_a_boundary() {
        // a few vertices at each x, forty at x = 50 and x = 60
        let mut v = Vec::new();
        for x in 0..100 {
            let k = if x == 50 || x == 60 { 40 } else { 2 };
            for j in 0..k {
                v.push([x as f64, j as f64, (x * 7 % 13) as f64]);
            }
        }
        let b = find_chunk_boundaries(&v);
        let x = b.iter().find(|b| b.axis == 0).expect("an x boundary");
        assert_eq!(x.values, [50, 60]);
        assert_eq!(x.counts, [40, 40]);
    }
}
