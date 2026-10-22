//! `skeliner.pre.remove_gaps`: peel each gap's two tip patches back to
//! clean rims, bridge each pair of rims with a tube, and rebuild once.
//!
//! The tube is the reference's geometry operation for operation — numpy's
//! `linspace`, `interp`, `cross`, `norm` and `dot` as numpy rounds them,
//! libm's `pow`, `sin`, `cos` and `atan2` — so new vertices come out
//! bit-identical.  The loop fits (`np.cov` and `eigh`) are the caller's.

use std::collections::{BTreeSet, HashMap};

use meshcore::Topology;
use meshcore::components::face_edge_components;
use meshcore::geometry::{HAS_FMA, numpy_order};
use meshcore::loops::border_loops;

use crate::gaps::{expand_tip_to_good_rim, rescue_size, sever_cost};

type V3 = [f64; 3];

/// `_fit_loop_circle`'s result: centre, mean radius, plane normal.
#[derive(Debug, Clone, Copy)]
pub struct LoopFit {
    pub center: V3,
    pub radius: f64,
    pub normal: V3,
}

/// The mesh a stitch reads.
pub struct StitchMesh<'a> {
    pub faces: &'a [[u32; 3]],
    pub verts: &'a [V3],
    /// trimesh's face normals, for orienting the tube.
    pub face_normals: &'a [V3],
    /// Median unique-edge length, which sets the tube's ring count.
    pub median_edge: f64,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `np.dot` of two 3-vectors, or one row of `(N, 3) @ (3,)`: a plain
/// left-to-right sum started from `+0.0`, as BLAS starts it, so products
/// that are all zero give `+0.0`.  The sign matters to `atan2`: a degenerate
/// frame projects every point onto a zero, and `-0.0` turns `+pi` into `-pi`.
fn dot(a: V3, b: V3) -> f64 {
    0.0 + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `np.linalg.norm` of a 3-vector.
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// `np.cross` of two 3-vectors.
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `np.linspace(start, stop, n, endpoint)`: `i * step + start`, the last
/// value set to `stop` when it is an endpoint.
fn linspace(start: f64, stop: f64, n: usize, endpoint: bool) -> Vec<f64> {
    let div = if endpoint { n.saturating_sub(1) } else { n };
    let step = (stop - start) / div as f64;
    let mut y: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
    if endpoint && n > 1 {
        y[n - 1] = stop;
    }
    y
}

/// `np.interp` for `xp` non-decreasing: the largest `j` with
/// `xp[j] <= x`, then `slope * (x - xp[j]) + fp[j]`, fused where numpy's
/// compiler fuses it.
fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let n = xp.len();
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                return v;
            }
            if v < xp[0] {
                return fp[0];
            }
            if v > xp[n - 1] {
                return fp[n - 1];
            }
            let j = xp.partition_point(|&p| p <= v) - 1;
            if j == n - 1 || xp[j] == v {
                return fp[j];
            }
            let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
            let lerp = |a: f64, b: f64, c: f64| if HAS_FMA { a.mul_add(b, c) } else { a * b + c };
            let mut r = lerp(slope, v - xp[j], fp[j]);
            if r.is_nan() {
                r = lerp(slope, v - xp[j + 1], fp[j + 1]);
                if r.is_nan() && fp[j] == fp[j + 1] {
                    r = fp[j];
                }
            }
            r
        })
        .collect()
}

/// `_hermite_spline`: `n` stations of the cubic Hermite curve.
fn hermite(p0: V3, t0: V3, p1: V3, t1: V3, n: usize) -> Vec<V3> {
    linspace(0.0, 1.0, n, true)
        .into_iter()
        .map(|s| {
            let (s2, s3) = (s * s, s.powf(3.0));
            let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
            let h10 = s3 - 2.0 * s2 + s;
            let h01 = -2.0 * s3 + 3.0 * s2;
            let h11 = s3 - s2;
            [0, 1, 2].map(|k| h00 * p0[k] + h10 * t0[k] + h01 * p1[k] + h11 * t1[k])
        })
        .collect()
}

/// A loop's vertices as `(angle, radius)` around `center` in the frame
/// of `tangent` and `up`.
fn polar(pts: &[V3], center: V3, tangent: V3, up: V3) -> (Vec<f64>, Vec<f64>) {
    let right = cross(tangent, up);
    let d = norm(right) + 1e-10;
    let right = right.map(|c| c / d);
    let mut angles = Vec::with_capacity(pts.len());
    let mut radii = Vec::with_capacity(pts.len());
    for &p in pts {
        let rel = sub(p, center);
        let (u, r) = (dot(rel, up), dot(rel, right));
        angles.push(r.atan2(u));
        radii.push((u * u + r * r).sqrt());
    }
    (angles, radii)
}

/// A polar profile resampled at `n` evenly spaced angles from -pi.
fn resample(angles: &[f64], radii: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..angles.len()).collect();
    order.sort_by(|&a, &b| numpy_order(&angles[a], &angles[b]));
    let two_pi = 2.0 * std::f64::consts::PI;
    let a: Vec<f64> = order.iter().map(|&i| angles[i]).collect();
    let r: Vec<f64> = order.iter().map(|&i| radii[i]).collect();
    let mut a_closed: Vec<f64> = a.iter().map(|&v| v - two_pi).collect();
    a_closed.extend_from_slice(&a);
    a_closed.extend(a.iter().map(|&v| v + two_pi));
    let r_closed: Vec<f64> = [r.as_slice(), r.as_slice(), r.as_slice()].concat();
    let target = linspace(-std::f64::consts::PI, std::f64::consts::PI, n, false);
    let resampled = interp(&target, &a_closed, &r_closed);
    (target, resampled)
}

/// Vertex positions of the mesh plus the vertices stitched so far.
struct Positions<'a> {
    verts: &'a [V3],
    new: &'a [V3],
}

impl Positions<'_> {
    fn at(&self, v: u32) -> V3 {
        let v = v as usize;
        if v < self.verts.len() {
            self.verts[v]
        } else {
            self.new[v - self.verts.len()]
        }
    }
}

/// `_build_weld_map`: each ring vertex onto a loop vertex, walking the loop
/// forward at most one step per ring step so every seam edge is a rim
/// edge; the nearest loop vertex instead for loops longer than the ring.
fn weld_map(pos: &Positions<'_>, ring: &[u32], lp: &[u32]) -> Vec<(u32, u32)> {
    let (n, m) = (ring.len(), lp.len());
    let ring_pts: Vec<V3> = ring.iter().map(|&v| pos.at(v)).collect();
    let loop_pts: Vec<V3> = lp.iter().map(|&v| pos.at(v)).collect();
    if m < 2 || n < m {
        // nearest loop vertex, the lowest index among exact ties
        return ring
            .iter()
            .zip(&ring_pts)
            .map(|(&rv, &rp)| {
                let mut best = (f64::INFINITY, 0usize);
                for (i, &q) in loop_pts.iter().enumerate() {
                    let d = sub(q, rp);
                    let d2 = (d[0] * d[0] + d[1] * d[1]) + d[2] * d[2];
                    if d2 < best.0 {
                        best = (d2, i);
                    }
                }
                (rv, lp[best.1])
            })
            .collect();
    }
    let mut cur = 0;
    let mut best = f64::INFINITY;
    for (i, &q) in loop_pts.iter().enumerate() {
        let d = norm(sub(q, ring_pts[0]));
        if d < best {
            best = d;
            cur = i;
        }
    }
    let mut out = vec![(ring[0], lp[cur])];
    let mut adv = 0;
    for j in 0..n - 1 {
        let remaining_dec = (n - 1) - j;
        let remaining_adv = (m - 1) - adv;
        if remaining_adv > 0 {
            let nxt = (cur + 1) % m;
            let closer = norm(sub(loop_pts[nxt], ring_pts[j + 1]))
                < norm(sub(loop_pts[cur], ring_pts[j + 1]));
            if remaining_adv >= remaining_dec || closer {
                cur = nxt;
                adv += 1;
            }
        }
        out.push((ring[j + 1], lp[cur]));
    }
    out
}

/// `_zipper_stitch`: a tube from loop `la` to loop `lb`, its rings
/// appended to `new_verts`; returns the tube's triangles.
fn zipper_stitch<E>(
    m: &StitchMesh<'_>,
    la: &[u32],
    lb: &[u32],
    vert_to_faces: &[Vec<u32>],
    new_verts: &mut Vec<V3>,
    fit: &mut impl FnMut(&[u32]) -> Result<LoopFit, E>,
) -> Result<Vec<[u32; 3]>, E> {
    let a = fit(la)?;
    let b = fit(lb)?;
    let direction = sub(b.center, a.center);
    let gap_dist = norm(direction);
    let na = if dot(a.normal, direction) < 0.0 {
        a.normal.map(|c| -c)
    } else {
        a.normal
    };
    let nb = if dot(b.normal, direction) < 0.0 {
        b.normal.map(|c| -c)
    } else {
        b.normal
    };
    let n_rings = ((gap_dist / m.median_edge).round_ties_even() as i64 - 1).max(1) as usize;
    let n_stations = n_rings + 2;
    let centers = hermite(
        a.center,
        scale(na, gap_dist),
        b.center,
        scale(nb, gap_dist),
        n_stations,
    );
    let n_ring_pts = la.len().max(lb.len());
    let n_existing = m.verts.len();

    let mut tangents: Vec<V3> = (0..n_stations)
        .map(|i| {
            if i == 0 {
                sub(centers[1], centers[0])
            } else if i == n_stations - 1 {
                sub(centers[i], centers[i - 1])
            } else {
                sub(centers[i + 1], centers[i - 1])
            }
        })
        .collect();
    for t in &mut tangents {
        let d = norm(*t) + 1e-10;
        *t = t.map(|c| c / d);
    }
    let mut up: Vec<V3> = Vec::with_capacity(n_stations);
    let t0 = tangents[0];
    let mut seed_up = [0.0, 0.0, 1.0];
    if dot(seed_up, t0).abs() > 0.9 {
        seed_up = [0.0, 1.0, 0.0];
    }
    let u0 = sub(seed_up, scale(t0, dot(seed_up, t0)));
    let d0 = norm(u0) + 1e-10;
    up.push(u0.map(|c| c / d0));
    for i in 1..n_stations {
        let prev = up[i - 1];
        let u = sub(prev, scale(tangents[i], dot(prev, tangents[i])));
        let nrm = norm(u);
        up.push(if nrm > 1e-10 {
            u.map(|c| c / nrm)
        } else {
            prev
        });
    }

    let pts_a: Vec<V3> = la.iter().map(|&v| m.verts[v as usize]).collect();
    let pts_b: Vec<V3> = lb.iter().map(|&v| m.verts[v as usize]).collect();
    let (ang_a, rad_a) = polar(&pts_a, centers[0], tangents[0], up[0]);
    let last = n_stations - 1;
    let (ang_b, rad_b) = polar(&pts_b, centers[last], tangents[last], up[last]);
    let (target, profile_a) = resample(&ang_a, &rad_a, n_ring_pts);
    let (_, profile_b) = resample(&ang_b, &rad_b, n_ring_pts);

    let mut ring_ids: Vec<Vec<u32>> = Vec::with_capacity(n_stations);
    for si in 0..n_stations {
        let t = si as f64 / (n_stations - 1) as f64;
        let c = centers[si];
        let u_vec = up[si];
        let right = cross(tangents[si], u_vec);
        let dr = norm(right) + 1e-10;
        let right = right.map(|x| x / dr);
        let mut ids = Vec::with_capacity(n_ring_pts);
        for j in 0..n_ring_pts {
            let r = profile_a[j] * (1.0 - t) + profile_b[j] * t;
            let (cos, sin) = (target[j].cos(), target[j].sin());
            let pt = [0, 1, 2].map(|k| c[k] + r * (cos * u_vec[k] + sin * right[k]));
            ids.push((n_existing + new_verts.len()) as u32);
            new_verts.push(pt);
        }
        ring_ids.push(ids);
    }

    let pos = Positions {
        verts: m.verts,
        new: new_verts,
    };
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for w in ring_ids.windows(2) {
        let (ra, rb) = (&w[0], &w[1]);
        let (na_r, nb_r) = (ra.len(), rb.len());
        // start the next ring at its vertex nearest this ring's first
        let p0 = pos.at(ra[0]);
        let mut best_j = 0;
        let mut best_d = f64::INFINITY;
        for (j, &v) in rb.iter().enumerate() {
            let d = norm(sub(pos.at(v), p0));
            if d < best_d {
                best_d = d;
                best_j = j;
            }
        }
        let rb: Vec<u32> = rb[best_j..].iter().chain(&rb[..best_j]).copied().collect();
        if na_r == nb_r {
            for i in 0..na_r {
                let i2 = (i + 1) % na_r;
                tris.push([ra[i], ra[i2], rb[i]]);
                tris.push([ra[i2], rb[i2], rb[i]]);
            }
            continue;
        }
        let (mut ia, mut ib, mut sa, mut sb) = (0, 0, 0, 0);
        while sa < na_r || sb < nb_r {
            let ia_n = (ia + 1) % na_r;
            let ib_n = (ib + 1) % nb_r;
            let (can_a, can_b) = (sa < na_r, sb < nb_r);
            let adv_a = if can_a && can_b {
                let da = norm(sub(pos.at(ra[ia_n]), pos.at(rb[ib])));
                let db = norm(sub(pos.at(ra[ia]), pos.at(rb[ib_n])));
                da <= db
            } else {
                can_a
            };
            if adv_a {
                tris.push([ra[ia], ra[ia_n], rb[ib]]);
                ia = ia_n;
                sa += 1;
            } else {
                tris.push([ra[ia], rb[ib_n], rb[ib]]);
                ib = ib_n;
                sb += 1;
            }
        }
    }

    // weld the end rings onto the boundary loops
    let mut weld: HashMap<u32, u32> = HashMap::new();
    weld.extend(weld_map(&pos, &ring_ids[0], la));
    weld.extend(weld_map(&pos, &ring_ids[last], lb));
    for t in &mut tris {
        for v in t.iter_mut() {
            if let Some(&w) = weld.get(v) {
                *v = w;
            }
        }
    }

    // orient the tube like the faces around the first loop
    let mut ref_fis: Vec<u32> = Vec::new();
    for &vi in la.iter().take(5) {
        ref_fis.extend(vert_to_faces[vi as usize].iter().take(5));
    }
    if !ref_fis.is_empty() {
        let mean = |vs: &mut dyn Iterator<Item = V3>, k: usize| {
            let mut s = [0.0f64; 3];
            for v in vs {
                for c in 0..3 {
                    s[c] += v[c];
                }
            }
            s.map(|c| c / k as f64)
        };
        let ref_n = mean(
            &mut ref_fis.iter().map(|&f| m.face_normals[f as usize]),
            ref_fis.len(),
        );
        let tri_n = mean(
            &mut tris.iter().map(|t| {
                let p0 = pos.at(t[0]);
                cross(sub(pos.at(t[1]), p0), sub(pos.at(t[2]), p0))
            }),
            tris.len(),
        );
        if dot(tri_n, ref_n) < 0.0 {
            for t in &mut tris {
                t.swap(1, 2);
            }
        }
    }
    Ok(tris)
}

/// `_stitch_and_rebuild`: the faces of `sel` become `[0, 0, 0]`, each loop
/// pair's tube is appended, and each loop of `fills` is closed with a fan.
/// Returns the new vertices (to append) and the full face list.
pub fn stitch_and_rebuild<E>(
    m: &StitchMesh<'_>,
    sel: &BTreeSet<u32>,
    pairs: &[(Vec<u32>, Vec<u32>)],
    fills: &[Vec<u32>],
    fit: &mut impl FnMut(&[u32]) -> Result<LoopFit, E>,
) -> Result<(Vec<V3>, Vec<[u32; 3]>), E> {
    let mut vert_to_faces: Vec<Vec<u32>> = vec![Vec::new(); m.verts.len()];
    for (fi, f) in m.faces.iter().enumerate() {
        let degenerate = f[0] == f[1] || f[1] == f[2] || f[0] == f[2];
        if !degenerate && !sel.contains(&(fi as u32)) {
            for &v in f {
                vert_to_faces[v as usize].push(fi as u32);
            }
        }
    }
    let mut new_verts: Vec<V3> = Vec::new();
    let mut stitched: Vec<[u32; 3]> = Vec::new();
    for (la, lb) in pairs {
        stitched.extend(zipper_stitch(
            m,
            la,
            lb,
            &vert_to_faces,
            &mut new_verts,
            fit,
        )?);
    }
    for lp in fills {
        stitched.extend(fan_fill(m, lp, &vert_to_faces, &mut new_verts));
    }
    let mut faces = m.faces.to_vec();
    for &f in sel {
        faces[f as usize] = [0, 0, 0];
    }
    faces.extend(stitched);
    Ok((new_verts, faces))
}

/// Close the hole bounded by `lp` with a fan from its centroid, wound so
/// that every border edge runs against the kept face beside it.
/// `vert_to_faces` lists the kept faces at each vertex.
fn fan_fill(
    m: &StitchMesh<'_>,
    lp: &[u32],
    vert_to_faces: &[Vec<u32>],
    new_verts: &mut Vec<V3>,
) -> Vec<[u32; 3]> {
    let n = lp.len();
    let (mut along, mut against) = (0usize, 0usize);
    for i in 0..n {
        let (u, v) = (lp[i], lp[(i + 1) % n]);
        for &f in &vert_to_faces[u as usize] {
            let t = m.faces[f as usize];
            for k in 0..3 {
                let e = (t[k], t[(k + 1) % 3]);
                along += (e == (u, v)) as usize;
                against += (e == (v, u)) as usize;
            }
        }
    }
    let ring: Vec<u32> = if along >= against {
        lp.iter().rev().copied().collect()
    } else {
        lp.to_vec()
    };
    let mut c = [0.0; 3];
    for &v in lp {
        let q = m.verts[v as usize];
        c = [c[0] + q[0], c[1] + q[1], c[2] + q[2]];
    }
    let centre = (m.verts.len() + new_verts.len()) as u32;
    new_verts.push(c.map(|x| x / n as f64));
    (0..n)
        .map(|i| [ring[i], ring[(i + 1) % n], centre])
        .collect()
}

fn vertex_set(l: &[u32]) -> Vec<u32> {
    let mut s = l.to_vec();
    s.sort_unstable();
    s.dedup();
    s
}

/// One gap as `find_gaps` reports it: the tip faces on each side.
pub struct GapTips {
    pub faces_a: Vec<u32>,
    pub faces_b: Vec<u32>,
}

/// What [`remove_gaps`] did.
pub struct GapsRemoved {
    /// New vertices, to append; empty when nothing was stitched.
    pub new_verts: Vec<V3>,
    /// The full face list, or `None` when nothing was stitched.
    pub faces: Option<Vec<[u32; 3]>>,
    pub n_bridged: usize,
    pub n_removed: usize,
    pub n_skipped: usize,
}

/// `_validate_loop_pair`: both loops of four or more vertices, at most five
/// times apart in size, and not the same rim.
fn loop_pair_ok(a: &[u32], b: &[u32]) -> bool {
    let (na, nb) = (a.len(), b.len());
    if na < 4 || nb < 4 {
        return false;
    }
    if (na.max(nb) as f64 / na.min(nb) as f64) > 5.0 {
        return false;
    }
    vertex_set(a) != vertex_set(b)
}

/// `skeliner.pre.remove_gaps` once the gaps are known: per gap, peel both
/// tips to clean rims, skip pairs that fail validation or whose removal
/// would strand more than the stitch wins back, then stitch and rebuild.
///
/// Rims that share two or more vertices belong to tips lying side by side,
/// glued along those vertices (a fusion join); a tube from rim to rim would
/// weld both of its ends onto the glue and fold over it. Such a gap is
/// bridged by removing both tips and closing the one hole they leave with a
/// fan — or skipped when their union does not leave exactly one hole.
///
/// `labels` are the raw face-edge components the rescue sizes are read
/// from, computed here on first need when not given.
pub fn remove_gaps<E>(
    topo: &Topology,
    m: &StitchMesh<'_>,
    labels: Option<&[i64]>,
    gaps: &[GapTips],
    fit: &mut impl FnMut(&[u32]) -> Result<LoopFit, E>,
) -> Result<GapsRemoved, E> {
    let mut own_labels: Option<Vec<i64>> = None;
    let mut counts: Option<HashMap<i64, usize>> = None;
    let mut remove: BTreeSet<u32> = BTreeSet::new();
    let mut pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    let mut fills: Vec<Vec<u32>> = Vec::new();
    let mut n_skipped = 0;
    for g in gaps {
        let (sel_a, loop_a) = expand_tip_to_good_rim(topo, &g.faces_a, 6, 20, 30);
        let (sel_b, loop_b) = expand_tip_to_good_rim(topo, &g.faces_b, 6, 20, 30);
        let (Some(loop_a), Some(loop_b)) = (loop_a, loop_b) else {
            n_skipped += 1;
            continue;
        };
        let (set_a, set_b) = (vertex_set(&loop_a), vertex_set(&loop_b));
        let shared = set_a
            .iter()
            .filter(|v| set_b.binary_search(v).is_ok())
            .count();
        if shared >= 2 && set_a != set_b {
            let union: Vec<u32> = sel_a.union(&sel_b).copied().collect();
            let holes: Vec<Vec<u32>> = border_loops(topo, &union)
                .into_iter()
                .filter(|lp| lp.len() >= 3)
                .collect();
            if let [hole] = holes.as_slice() {
                fills.push(hole.clone());
                remove.extend(union);
            } else {
                n_skipped += 1;
            }
            continue;
        }
        if !loop_pair_ok(&loop_a, &loop_b) {
            n_skipped += 1;
            continue;
        }
        let severs = |sel: &BTreeSet<u32>| {
            let faces: Vec<u32> = sel.iter().copied().collect();
            border_loops(topo, &faces)
                .iter()
                .filter(|lp| lp.len() >= 3)
                .count()
                >= 2
        };
        let (sev_a, sev_b) = (severs(&sel_a), severs(&sel_b));
        if sev_a || sev_b {
            let labels: &[i64] = match labels {
                Some(l) => l,
                None => own_labels.get_or_insert_with(|| face_edge_components(topo).0),
            };
            let counts = counts.get_or_insert_with(|| {
                let mut c = HashMap::new();
                for &l in labels {
                    *c.entry(l).or_insert(0usize) += 1;
                }
                c
            });
            let sizes = |l: i64| counts.get(&l).copied().unwrap_or(0);
            let rescue = rescue_size(labels, &sizes, g.faces_a[0] as usize, g.faces_b[0] as usize);
            if (sev_a && sever_cost(topo, &sel_a, rescue) > rescue)
                || (sev_b && sever_cost(topo, &sel_b, rescue) > rescue)
            {
                n_skipped += 1;
                continue;
            }
        }
        pairs.push((loop_a, loop_b));
        remove.extend(sel_a);
        remove.extend(sel_b);
    }
    if pairs.is_empty() && fills.is_empty() {
        return Ok(GapsRemoved {
            new_verts: Vec::new(),
            faces: None,
            n_bridged: 0,
            n_removed: 0,
            n_skipped,
        });
    }
    let (new_verts, faces) = stitch_and_rebuild(m, &remove, &pairs, &fills, fit)?;
    Ok(GapsRemoved {
        new_verts,
        faces: Some(faces),
        n_bridged: pairs.len() + fills.len(),
        n_removed: remove.len(),
        n_skipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_and_interp_follow_numpy() {
        assert_eq!(linspace(0.0, 1.0, 5, true), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(0.0, 1.0, 4, false), vec![0.0, 0.25, 0.5, 0.75]);
        let xp = [0.0, 1.0, 1.0, 3.0];
        let fp = [0.0, 10.0, 20.0, 40.0];
        assert_eq!(
            interp(&[-1.0, 0.5, 1.0, 2.0, 3.0, 4.0], &xp, &fp),
            vec![0.0, 5.0, 20.0, 30.0, 40.0, 40.0]
        );
    }

    #[test]
    fn a_fan_closes_a_hole_against_the_faces_around_it() {
        // a 3 x 3 grid of squares in z = 0, wound counter-clockwise, its
        // middle square removed
        let vid = |i: u32, j: u32| j * 4 + i;
        let verts: Vec<V3> = (0..16)
            .map(|k| [(k % 4) as f64, (k / 4) as f64, 0.0])
            .collect();
        let mut faces = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                let (a, b, c, d) = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1));
                faces.push([a, b, c]);
                faces.push([a, c, d]);
            }
        }
        let normals = vec![[0.0, 0.0, 1.0]; faces.len()];
        let m = StitchMesh {
            faces: &faces,
            verts: &verts,
            face_normals: &normals,
            median_edge: 1.0,
        };
        let removed: BTreeSet<u32> = [8, 9].into_iter().collect();
        let mut vert_to_faces = vec![Vec::new(); verts.len()];
        for (fi, f) in faces.iter().enumerate() {
            if !removed.contains(&(fi as u32)) {
                for &v in f {
                    vert_to_faces[v as usize].push(fi as u32);
                }
            }
        }
        // the hole's border, walked either way round
        let hole = [vid(1, 1), vid(1, 2), vid(2, 2), vid(2, 1)];
        let mut reversed = hole;
        reversed.reverse();
        let mut other = Vec::new();
        let flipped = fan_fill(&m, &reversed, &vert_to_faces, &mut other);
        let mut new_verts = Vec::new();
        let fan = fan_fill(&m, &hole, &vert_to_faces, &mut new_verts);
        let directed = |f: &[[u32; 3]]| -> BTreeSet<(u32, u32)> {
            f.iter()
                .flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3])))
                .collect()
        };
        assert_eq!(
            directed(&flipped),
            directed(&fan),
            "the same winding either way"
        );
        assert_eq!(new_verts, vec![[1.5, 1.5, 0.0]]);
        assert_eq!(fan.len(), 4);
        let mut uses: HashMap<(u32, u32), usize> = HashMap::new();
        for (fi, f) in faces.iter().enumerate() {
            if removed.contains(&(fi as u32)) {
                continue;
            }
            for k in 0..3 {
                *uses.entry((f[k], f[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        for f in &fan {
            for k in 0..3 {
                *uses.entry((f[k], f[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        // every directed edge once: the patch is consistently wound, and
        // each border edge has its kept face and its fan face
        assert!(uses.values().all(|&n| n == 1), "{uses:?}");
        for i in 0..4 {
            let (u, v) = (hole[i], hole[(i + 1) % 4]);
            assert_eq!(
                uses.get(&(u, v)).copied().unwrap_or(0) + uses.get(&(v, u)).copied().unwrap_or(0),
                2
            );
        }
    }

    #[test]
    fn a_degenerate_frame_projects_onto_plus_zero() {
        // Two coincident loops leave a zero tangent and so no right vector;
        // numpy's dot of a point behind the centre with it is +0.0, and
        // arctan2 gives +pi, where a -0.0 would give -pi.
        let (angles, _) = polar(&[[-1.0, -2.0, -3.0]], [0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]);
        assert_eq!(angles, [std::f64::consts::PI]);
    }

    #[test]
    fn equal_angles_keep_their_order_however_their_zeros_are_signed() {
        // A loop around a zero-length frame projects to angles of +-0.0;
        // numpy's stable argsort keeps them in loop order, so the profile
        // at angle 0 is the second one's radius.
        let (target, profile) = resample(&[0.0, -0.0, 1.0], &[1.0, 2.0, 3.0], 4);
        assert_eq!(target[2], 0.0);
        assert_eq!(
            profile,
            [
                2.189279751107925,
                1.5946398755539626,
                2.0,
                2.7839196266618877
            ]
        );
    }

    #[test]
    fn the_hermite_curve_runs_from_end_to_end() {
        let c = hermite(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            5,
        );
        assert_eq!(c[0], [0.0; 3]);
        assert_eq!(c[4], [1.0, 0.0, 0.0]);
        assert_eq!(c[2], [0.5, 0.0, 0.0]);
    }

    #[test]
    fn small_mismatched_or_identical_loops_are_refused() {
        let loop_of = |start: u32, n: u32| (start..start + n).collect::<Vec<u32>>();
        assert!(!loop_pair_ok(&[0, 1, 2], &[0, 1, 2, 3]));
        assert!(loop_pair_ok(&loop_of(0, 4), &loop_of(10, 20)));
        assert!(!loop_pair_ok(&loop_of(0, 4), &loop_of(10, 21)));
        // a loop paired with itself, in any order, is refused; sharing a
        // few vertices (pieces glued at the rim) is not
        assert!(!loop_pair_ok(&[3, 4, 5, 6, 7], &[7, 6, 5, 4, 3]));
        assert!(loop_pair_ok(&[3, 4, 5, 6, 7], &[5, 6, 7, 8, 9]));
    }
}
