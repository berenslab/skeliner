//! Choosing the bridges that reconnect disconnected components.

use std::collections::{BTreeSet, HashMap, HashSet};

use meshcore::Topology;
use meshcore::geometry::numpy_order;
use meshcore::spatial::{KdTree, Sum};
use rayon::prelude::*;

use crate::fits::Svd;

/// Knobs of `skeliner.pre.find_gaps`, with every default resolved.
#[derive(Debug, Clone, Copy)]
pub struct GapParams {
    pub tip_rings: usize,
    /// Fusion clearance, already in mesh units.
    pub fusion_clear: f64,
    /// Gaps above this (6 median edges) with only near-fusion candidates
    /// keep their near-fusion contact instead.
    pub fusion_fallback: f64,
    pub max_bridge_ratio: f64,
    /// Gaps at or below this always bridge.
    pub bridge_floor: f64,
    /// No gap above this ever bridges.
    pub bridge_abs_cap: f64,
    /// Kiss neighbourhood radius, already in mesh units.
    pub kiss_r: f64,
    pub kiss_penalty: f64,
    pub relay_min_faces: usize,
    pub repair_stranded: bool,
}

/// One bridge: tip faces on each side, the gap distance, and the indices
/// of the two disconnected components (`-1` for main).
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    pub faces_a: Vec<u32>,
    pub faces_b: Vec<u32>,
    pub dist: f64,
    pub disc_a: i64,
    pub disc_b: i64,
}

/// Gaps and the tallies of the verbose report.
#[derive(Debug, Clone, Default)]
pub struct Gaps {
    pub gaps: Vec<Gap>,
    pub n_capped: usize,
    pub n_repaired: usize,
}

struct Component {
    faces: Vec<u32>,
    verts: Vec<u32>,
    coords: Vec<[f64; 3]>,
    tree: KdTree,
    fusion_dist: Option<Vec<f64>>,
    bbox: f64,
}

#[derive(Debug, Clone, Copy)]
struct EdgeInfo {
    mst_cost: f64,
    dist: f64,
    a: i64,
    ia: usize,
    b: i64,
    ib: usize,
    fusion_join: bool,
}

/// The mesh as `find_gaps` reads it.
pub struct GapMesh<'a> {
    pub topo: &'a Topology,
    pub faces: &'a [[u32; 3]],
    pub verts: &'a [[f64; 3]],
    pub centers: &'a [[f64; 3]],
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Local two-sidedness at a contact: 0 when the piece ends there (a tip or
/// cut), towards 1 when it carries on past it on both sides (a kiss).
fn flank(c: &Component, idx: usize, kiss_r: f64, svd: &dyn Svd) -> f64 {
    let p = c.coords[idx];
    let nb = c.tree.within(&p, kiss_r);
    if nb.len() < 6 {
        return 0.0;
    }
    let pts: Vec<[f64; 3]> = nb.iter().map(|&j| c.coords[j as usize]).collect();
    let n = pts.len() as f64;
    let mut mean = [0.0f64; 3];
    for q in &pts {
        for k in 0..3 {
            mean[k] += q[k];
        }
    }
    let mean = mean.map(|m| m / n);
    let centred: Vec<[f64; 3]> = pts.iter().map(|q| sub(*q, mean)).collect();
    let axis = svd.right_singular(&centred, false)[0];
    let (mut pos, mut neg) = (0usize, 0usize);
    for q in &pts {
        let d = sub(*q, p);
        let t = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
        pos += usize::from(t > 0.0);
        neg += usize::from(t < 0.0);
    }
    2.0 * (pos as f64 / n).min(neg as f64 / n)
}

/// Faces within `rings` edge-neighbour rings of the face of `comp_faces`
/// nearest to vertex `tip`, ascending.
fn tip_faces(
    m: &GapMesh<'_>,
    comp_faces: &[u32],
    comp_set: &HashSet<u32>,
    tip: u32,
    rings: usize,
) -> Vec<u32> {
    let p = m.verts[tip as usize];
    let mut best = (0usize, f64::INFINITY);
    for (k, &f) in comp_faces.iter().enumerate() {
        let t = m.faces[f as usize];
        let (a, b, c) = (
            m.verts[t[0] as usize],
            m.verts[t[1] as usize],
            m.verts[t[2] as usize],
        );
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        let d = norm(sub(centroid, p));
        if d < best.1 {
            best = (k, d);
        }
    }
    let seed = comp_faces[best.0];
    let mut visited: BTreeSet<u32> = BTreeSet::from([seed]);
    let mut frontier: BTreeSet<u32> = BTreeSet::from([seed]);
    for _ in 0..rings {
        let mut next: BTreeSet<u32> = BTreeSet::new();
        for &f in &frontier {
            for g in m.topo.face_neighbors(f as usize) {
                if comp_set.contains(&g) && visited.insert(g) {
                    next.insert(g);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    visited.into_iter().collect()
}

/// `skeliner.pre.find_gaps` once the components are known.
///
/// `labels` are the fusion-refined component labels and `labels_pre_fusion`
/// the ones before refinement (siblings share one); `disc` are the
/// disconnected components, each a face list; `fusion_faces` the distinct
/// fusion-cluster faces in first-seen order.
#[allow(clippy::too_many_arguments)]
pub fn find_gaps(
    m: &GapMesh<'_>,
    labels: &[i64],
    labels_pre_fusion: &[i64],
    main: i64,
    disc: &[Vec<u32>],
    fusion_faces: &[u32],
    params: &GapParams,
    svd: &dyn Svd,
) -> Gaps {
    let mut out = Gaps::default();
    if disc.is_empty() {
        return out;
    }
    let fusion_verts: HashSet<u32> = fusion_faces
        .iter()
        .flat_map(|&f| m.faces[f as usize])
        .collect();
    let fusion_tree = (!fusion_faces.is_empty()).then(|| {
        KdTree::new(
            fusion_faces
                .iter()
                .map(|&f| m.centers[f as usize])
                .collect(),
            Sum::Plain,
        )
    });

    let component = |faces: Vec<u32>| -> Component {
        let mut verts: Vec<u32> = faces.iter().flat_map(|&f| m.faces[f as usize]).collect();
        verts.sort_unstable();
        verts.dedup();
        if !fusion_verts.is_empty() {
            let kept: Vec<u32> = verts
                .iter()
                .copied()
                .filter(|v| !fusion_verts.contains(v))
                .collect();
            if !kept.is_empty() {
                verts = kept;
            }
        }
        let coords: Vec<[f64; 3]> = verts.iter().map(|&v| m.verts[v as usize]).collect();
        let fusion_dist = fusion_tree
            .as_ref()
            .filter(|_| !coords.is_empty())
            .map(|t| {
                coords
                    .par_iter()
                    .map(|p| t.nearest(p).expect("non-empty").1)
                    .collect()
            });
        let bbox = if coords.is_empty() {
            0.0
        } else {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for p in &coords {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            norm(sub(hi, lo))
        };
        Component {
            tree: KdTree::new(coords.clone(), Sum::Plain),
            faces,
            verts,
            coords,
            fusion_dist,
            bbox,
        }
    };

    let main_faces: Vec<u32> = (0..labels.len() as u32)
        .filter(|&f| labels[f as usize] == main)
        .collect();
    let mut comps: HashMap<i64, Component> = HashMap::new();
    comps.insert(main, component(main_faces));
    let disc_cids: Vec<i64> = disc.iter().map(|fis| labels[fis[0] as usize]).collect();
    for (fis, &cid) in disc.iter().zip(&disc_cids) {
        comps.insert(cid, component(fis.clone()));
    }
    let all_cids: Vec<i64> = std::iter::once(main)
        .chain(disc_cids.iter().copied())
        .collect();
    let orig_label = |cid: i64| labels_pre_fusion[comps[&cid].faces[0] as usize];

    // Pairwise candidate bridges.
    let mut edge_info: HashMap<(i64, i64), EdgeInfo> = HashMap::new();
    let mut pair_order: Vec<(i64, i64)> = Vec::new();
    for (i, &cid_a) in all_cids.iter().enumerate() {
        for &cid_b in &all_cids[i + 1..] {
            let (da, db) = (&comps[&cid_a], &comps[&cid_b]);
            let found: Vec<(u32, f64)> = da
                .coords
                .par_iter()
                .map(|p| db.tree.nearest(p).expect("non-empty"))
                .collect();
            let first_min = |valid: &dyn Fn(usize) -> bool| {
                let mut best: Option<(usize, f64)> = None;
                for (k, &(_, d)) in found.iter().enumerate() {
                    if valid(k) && best.is_none_or(|(_, b)| d < b) {
                        best = Some((k, d));
                    }
                }
                best
            };
            let (raw_i, raw_dist) = first_min(&|_| true).expect("non-empty component");
            let (min_i, cost) = match (&fusion_tree, &da.fusion_dist, &db.fusion_dist) {
                (Some(_), Some(fa), Some(fb)) => {
                    let valid = |k: usize| {
                        fa[k] > params.fusion_clear && fb[found[k].0 as usize] > params.fusion_clear
                    };
                    match first_min(&valid) {
                        None => (raw_i, raw_dist),
                        Some((fi, fd)) => {
                            if fd > params.fusion_fallback && raw_dist >= 1.0 {
                                (raw_i, raw_dist)
                            } else {
                                (fi, fd)
                            }
                        }
                    }
                }
                _ => (raw_i, raw_dist),
            };
            let ib = found[min_i].0 as usize;
            let mut mst_cost = cost;
            if cost >= 1.0 {
                let f = flank(da, min_i, params.kiss_r, svd).min(flank(db, ib, params.kiss_r, svd));
                mst_cost = cost * (1.0 + params.kiss_penalty * f);
            }
            let fusion_join = match (&da.fusion_dist, &db.fusion_dist) {
                (Some(fa), Some(fb)) if cost < 1.0 && orig_label(cid_a) == orig_label(cid_b) => {
                    fa[min_i].min(fb[ib]) <= params.fusion_clear
                }
                _ => false,
            };
            let key = (cid_a.min(cid_b), cid_a.max(cid_b));
            if !edge_info.contains_key(&key) {
                pair_order.push(key);
            }
            edge_info.insert(
                key,
                EdgeInfo {
                    mst_cost,
                    dist: cost,
                    a: cid_a,
                    ia: min_i,
                    b: cid_b,
                    ib,
                    fusion_join,
                },
            );
        }
    }

    // Union-find over components the bridges (or sub-1 nm contacts) join.
    let mut joined: HashMap<i64, i64> = all_cids.iter().map(|&c| (c, c)).collect();
    fn root(joined: &mut HashMap<i64, i64>, c: i64) -> i64 {
        let mut r = c;
        while joined[&r] != r {
            r = joined[&r];
        }
        let mut c = c;
        while joined[&c] != r {
            let next = joined[&c];
            joined.insert(c, r);
            c = next;
        }
        r
    }
    let join = |joined: &mut HashMap<i64, i64>, a: i64, b: i64| {
        let (ra, rb) = (root(joined, a), root(joined, b));
        if ra != rb {
            joined.insert(ra, rb);
        }
    };
    for key in &pair_order {
        let e = edge_info[key];
        if e.dist < 1.0 && !e.fusion_join {
            join(&mut joined, key.0, key.1);
        }
    }

    let bridge_cap = |a: i64, b: i64| {
        let smaller = comps[&a].bbox.min(comps[&b].bbox);
        params
            .bridge_abs_cap
            .min(params.bridge_floor.max(params.max_bridge_ratio * smaller))
    };
    let disc_index = |cid: i64| {
        disc_cids
            .iter()
            .position(|&c| c == cid)
            .map_or(-1, |i| i as i64)
    };
    let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
    let mut add_gap = |e: &EdgeInfo, out: &mut Gaps| -> bool {
        let key = (e.a.min(e.b), e.a.max(e.b));
        if !seen_pairs.insert(key) {
            return false;
        }
        if e.dist < 1.0 && !e.fusion_join {
            return false;
        }
        if e.dist > bridge_cap(e.a, e.b) {
            out.n_capped += 1;
            return false;
        }
        let (ca, cb) = (&comps[&e.a], &comps[&e.b]);
        let set_a: HashSet<u32> = ca.faces.iter().copied().collect();
        let set_b: HashSet<u32> = cb.faces.iter().copied().collect();
        let fa = tip_faces(m, &ca.faces, &set_a, ca.verts[e.ia], params.tip_rings);
        let fb = tip_faces(m, &cb.faces, &set_b, cb.verts[e.ib], params.tip_rings);
        if fa.is_empty() || fb.is_empty() {
            return false;
        }
        out.gaps.push(Gap {
            faces_a: fa,
            faces_b: fb,
            dist: e.dist,
            disc_a: disc_index(e.a),
            disc_b: disc_index(e.b),
        });
        true
    };

    // Prim's tree from main; equal costs go to the lowest (remaining,
    // connected) pair.  A small component can end a chain but not relay it.
    let relay_ok = |cid: i64| cid == main || comps[&cid].faces.len() >= params.relay_min_faces;
    let mut connected: BTreeSet<i64> = BTreeSet::from([main]);
    let mut remaining: BTreeSet<i64> = disc_cids.iter().copied().collect();
    while !remaining.is_empty() {
        let mut best: Option<(f64, i64, (i64, i64))> = None;
        for &r in &remaining {
            for &c in &connected {
                if !relay_ok(c) {
                    continue;
                }
                let key = (r.min(c), r.max(c));
                let cost = edge_info[&key].mst_cost;
                if best.is_none_or(|(b, _, _)| cost < b) {
                    best = Some((cost, r, key));
                }
            }
        }
        let Some((_, r, key)) = best else { break };
        connected.insert(r);
        remaining.remove(&r);
        let e = edge_info[&key];
        if add_gap(&e, &mut out) {
            join(&mut joined, e.a, e.b);
        }
    }

    // Retry each stranded cluster with its cheapest acceptable edge.
    if params.repair_stranded {
        loop {
            let main_root = root(&mut joined, main);
            let stranded: Vec<i64> = disc_cids
                .iter()
                .copied()
                .filter(|&c| root(&mut joined, c) != main_root)
                .collect();
            if stranded.is_empty() {
                break;
            }
            let mut cluster_order: Vec<i64> = Vec::new();
            let mut clusters: HashMap<i64, Vec<i64>> = HashMap::new();
            for &c in &stranded {
                let r = root(&mut joined, c);
                let entry = clusters.entry(r).or_default();
                if entry.is_empty() {
                    cluster_order.push(r);
                }
                entry.push(c);
            }
            let mut progress = false;
            for r in cluster_order {
                let members = &clusters[&r];
                let inside: HashSet<i64> = members.iter().copied().collect();
                let mut best: Option<(f64, (i64, i64))> = None;
                for &c_in in members {
                    for &c_out in &all_cids {
                        if inside.contains(&c_out) {
                            continue;
                        }
                        let key = (c_in.min(c_out), c_in.max(c_out));
                        let e = edge_info[&key];
                        if e.dist < 1.0 && !e.fusion_join {
                            continue;
                        }
                        if e.dist > bridge_cap(e.a, e.b) {
                            continue;
                        }
                        if best.is_none_or(|(b, _)| e.mst_cost < b) {
                            best = Some((e.mst_cost, key));
                        }
                    }
                }
                let Some((_, key)) = best else { continue };
                let e = edge_info[&key];
                if add_gap(&e, &mut out) {
                    join(&mut joined, e.a, e.b);
                    out.n_repaired += 1;
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
    }

    out.gaps.sort_by(|x, y| numpy_order(&x.dist, &y.dist));
    out
}
