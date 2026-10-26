//! `skeliner.pre.find_parallel_patches` and `remove_parallel_patches`
//! without their shapely steps: axis-aligned face patches at chunk
//! boundaries that face an opposing patch, their removal, and the strips
//! that close what the removal opens.
//!
//! The reference's edge maps are rebuilt as it builds them — keyed by each
//! face's three column pairs, a degenerate face listed once per pair —
//! because neighbour counts read those lists directly.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

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
            for ax in 0..3 {
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

/// `_zip_vertex_loops`: a triangle strip between two vertex sequences,
/// the shorter bridge first at each step, over every start offset and
/// both windings of `lb`; the cheapest strip wins.
fn zip_vertex_loops(
    la: &[u32],
    lb: &[u32],
    verts: &[V3],
    closed_a: bool,
    closed_b: bool,
) -> Vec<[u32; 3]> {
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
            if i >= steps_a {
                tris.push([a_curr, t_next, t_curr]);
                cost += dist(at(t_next), at(a_curr));
                j += 1;
            } else if j >= steps_b {
                tris.push([a_curr, a_next, t_curr]);
                cost += dist(at(a_next), at(t_curr));
                i += 1;
            } else {
                let d_a = dist(at(a_next), at(t_curr));
                let d_b = dist(at(a_curr), at(t_next));
                if d_a <= d_b {
                    tris.push([a_curr, a_next, t_curr]);
                    cost += d_a;
                    i += 1;
                } else {
                    tris.push([a_curr, t_next, t_curr]);
                    cost += d_b;
                    j += 1;
                }
            }
        }
        (cost, tris)
    };
    let mut best: Option<(f64, Vec<[u32; 3]>)> = None;
    for reverse in [false, true] {
        let base: Vec<u32> = if reverse {
            lb.iter().rev().copied().collect()
        } else {
            lb.to_vec()
        };
        let shifts = if closed_b { n } else { 1 };
        for shift in 0..shifts {
            let shifted: Vec<u32> = base[shift..]
                .iter()
                .chain(&base[..shift])
                .copied()
                .collect();
            let (cost, tris) = walk(&shifted);
            if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                best = Some((cost, tris));
            }
        }
    }
    best.map(|(_, t)| t).unwrap_or_default()
}

/// `_stitch_mode_b`: per `(axis, bval)` group of patches with Mode B faces,
/// trace the rims each removed side leaves, pair up-side with down-side
/// rims (largest first, nearest centre with overlapping bboxes) and zip
/// each pair.  `topo` is the mesh's before the removal and `gone` the faces
/// removed.  Returns the new triangles.
fn stitch_mode_b(
    verts: &[V3],
    topo: &Topology,
    gone: &[bool],
    orig: &[[u32; 3]],
    mode_b: &HashSet<u32>,
    patches: &[Patch],
) -> Vec<[u32; 3]> {
    // an edge is a rim edge when exactly one face that has it is left
    let one_left = |(a, b): Edge| {
        topo.edge_id(a, b).is_some_and(|e| {
            topo.edge_faces(e)
                .iter()
                .filter(|&&f| !gone[f as usize])
                .count()
                == 1
        })
    };
    let rim_edges = |side: &HashSet<u32>| -> Vec<Edge> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for &fi in side {
            let f = orig[fi as usize];
            for i in 0..3 {
                let (a, b) = (f[i], f[(i + 1) % 3]);
                if a == b {
                    continue;
                }
                let e = edge(a, b);
                if seen.contains(&e) {
                    continue;
                }
                if one_left(e) {
                    seen.insert(e);
                    out.push(e);
                }
            }
        }
        out
    };
    let with_b: Vec<Patch> = patches
        .iter()
        .filter(|p| p.faces.iter().any(|f| mode_b.contains(f)))
        .cloned()
        .collect();
    let mut tris_out = Vec::new();
    for ((axis, _), members) in groups(&with_b) {
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
        if up_rim.is_empty() || dn_rim.is_empty() {
            continue;
        }
        let keep = |(lp, c): &(Vec<u32>, bool)| (*c && lp.len() >= 3) || (!*c && lp.len() >= 2);
        let up: Vec<(Vec<u32>, bool)> =
            trace_edge_loops(&up_rim).into_iter().filter(keep).collect();
        let dn: Vec<(Vec<u32>, bool)> =
            trace_edge_loops(&dn_rim).into_iter().filter(keep).collect();
        if up.is_empty() || dn.is_empty() {
            continue;
        }
        let [pa, pb] = perp(axis);
        let info = |lp: &[u32]| {
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
        let up_info: Vec<_> = up.iter().map(|(lp, _)| info(lp)).collect();
        let dn_info: Vec<_> = dn.iter().map(|(lp, _)| info(lp)).collect();
        let mut up_order: Vec<usize> = (0..up.len()).collect();
        up_order.sort_by_key(|&i| std::cmp::Reverse(up[i].0.len()));
        let mut dn_used = vec![false; dn.len()];
        for ui in up_order {
            let (uc, ulo, uhi) = up_info[ui];
            let mut best: Option<(usize, f64)> = None;
            for (di, &(dc, dlo, dhi)) in dn_info.iter().enumerate() {
                if dn_used[di] {
                    continue;
                }
                let overlap = (0..2).all(|k| uhi[k].min(dhi[k]) > ulo[k].max(dlo[k]));
                if !overlap {
                    continue;
                }
                let d = dist(uc, dc);
                if best.is_none_or(|(_, b)| d < b) {
                    best = Some((di, d));
                }
            }
            let Some((di, _)) = best else { continue };
            dn_used[di] = true;
            tris_out.extend(zip_vertex_loops(
                &up[ui].0, &dn[di].0, verts, up[ui].1, dn[di].1,
            ));
        }
    }
    tris_out
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
}

/// `remove_parallel_patches` on already trimmed patches: classify each
/// (fold or parallel), remove them all, drop pieces the removal cut off
/// main that border only fold patches, then stitch the parallel sides.
/// The loops over every face run on [`threads`]' pool.
pub fn remove_parallel_patches(m: &PatchMesh<'_>, patches: &[Patch]) -> Removed {
    threads::install(|| remove_patches(m, patches))
}

fn remove_patches(m: &PatchMesh<'_>, patches: &[Patch]) -> Removed {
    let nf = m.faces.len();
    let nv = m.verts.len();
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
    let topo = Topology::new(m.faces, nv);
    let (before, main_before) = face_edge_components(&topo);
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
    let n_removed = removed.iter().filter(|&&r| r).count();
    let mut n_stitched = 0;
    if !mode_b.is_empty() {
        let new = stitch_mode_b(m.verts, &topo, &gone, m.faces, &mode_b, patches);
        n_stitched = new.len();
        faces.extend(new);
    }
    Removed {
        faces,
        n_removed,
        n_fold: n_a,
        n_parallel: n_b,
        n_orphan,
        n_stitched,
    }
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
        let tris = zip_vertex_loops(&[0, 1, 2, 3], &[4, 5, 6, 7], &verts, true, true);
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
