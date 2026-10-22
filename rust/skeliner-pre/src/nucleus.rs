//! `skeliner.pre._z_scan`: the nucleus void, found slice by slice in Z.
//!
//! Each Z-level's vertices are rasterised; the largest cluster is the soma
//! cross-section, and the biggest empty disk enclosed in it is that level's
//! void.  Consecutive levels whose voids stay put form chains, and the
//! longest chain is the nucleus.  Levels just past the chain whose soma ring
//! is broken get a second look inside the convex hull of the chain's end.

use meshcore::geometry::{arange, numpy_sum};
use meshcore::raster::{dilate3, enclosed, fill_holes, label, sq_distance_to};
use rayon::prelude::*;

/// Distance between Z-levels.
const Z_STEP: f64 = 210.0;

#[derive(Debug, Clone, Copy)]
pub struct ZParams {
    /// Grid cell size.
    pub grid_res: f64,
    /// Half-width of a level's slab.
    pub z_tol: f64,
    /// Smallest void radius a chain accepts.
    pub min_void_r: f64,
    /// Largest XY shift of the void between chained levels.
    pub max_shift: f64,
}

/// One Z-level, as an entry of `_z_scan`'s `raw` list.
#[derive(Debug, Clone, PartialEq)]
pub struct Level {
    pub z: f64,
    /// Void centre; NaN when the level has none.
    pub cx: f64,
    pub cy: f64,
    /// Void radius, 0 when there is none.
    pub void_r: f64,
    /// Cell centres of the void's deep part, row-major.
    pub void_xy: Option<Vec<[f64; 2]>>,
    /// Vertices within `z_tol` of `z`, ascending.
    pub vi: Vec<u32>,
    /// Which of `vi` fall in the soma cluster.
    pub soma_mask: Option<Vec<bool>>,
}

/// The rasterised slab of one level.
struct Slab {
    nx: usize,
    ny: usize,
    xy_min: [f64; 2],
    occ: Vec<bool>,
    /// Grid cell of each of the level's vertices.
    cell: Vec<usize>,
    /// Components of the dilated occupancy.
    labeled: Vec<u32>,
    n_labels: u32,
}

/// A void found in a slab.
struct Void {
    cx: f64,
    cy: f64,
    void_r: f64,
    void_xy: Option<Vec<[f64; 2]>>,
}

/// A 2D convex hull's facet equations `(n0, n1, offset)`, inside where
/// `n . p + offset <= 0`, as scipy's `ConvexHull.equations` gives them.
pub type Hull = Vec<[f64; 3]>;

/// `_z_slice_cluster`: rasterise the points (at least ten) and label the
/// dilated occupancy.  Returns the slab and the mask of the largest
/// cluster, the lowest label among equals.
fn rasterise(pts: &[[f64; 2]], grid_res: f64) -> Option<(Slab, Vec<bool>)> {
    if pts.len() < 10 {
        return None;
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let xy_min = [lo[0] - grid_res * 3.0, lo[1] - grid_res * 3.0];
    let nx = ((hi[0] - xy_min[0] + grid_res * 6.0) / grid_res) as usize + 1;
    let ny = ((hi[1] - xy_min[1] + grid_res * 6.0) / grid_res) as usize + 1;
    let index = |v: f64, origin: f64, n: usize| {
        (((v - origin) / grid_res) as i64).clamp(0, n as i64 - 1) as usize
    };
    let mut occ = vec![false; nx * ny];
    let cell: Vec<usize> = pts
        .iter()
        .map(|p| index(p[0], xy_min[0], nx) * ny + index(p[1], xy_min[1], ny))
        .collect();
    for &c in &cell {
        occ[c] = true;
    }
    let (labeled, n_labels) = label(&dilate3(&occ, nx, ny), nx, ny);
    if n_labels == 0 {
        return None;
    }
    let mut sizes = vec![0usize; n_labels as usize + 1];
    for &l in &labeled {
        sizes[l as usize] += 1;
    }
    let mut lid = 1;
    for l in 2..=n_labels as usize {
        if sizes[l] > sizes[lid] {
            lid = l;
        }
    }
    let region = labeled.iter().map(|&l| l as usize == lid).collect();
    Some((
        Slab {
            nx,
            ny,
            xy_min,
            occ,
            cell,
            labeled,
            n_labels,
        },
        region,
    ))
}

/// The void's peak and its deep part, given the occupancy `occ_s` the
/// distances are taken to and the candidate `void` cells (at least one).
fn locate_void(s: &Slab, occ_s: &[bool], void: &[bool], grid_res: f64) -> Void {
    let (nx, ny) = (s.nx, s.ny);
    let dt: Vec<f64> = sq_distance_to(occ_s, nx, ny)
        .into_iter()
        .map(|d| (d as f64).sqrt())
        .collect();
    // first maximum of dt * void in row-major order
    let mut pk = 0;
    let mut best = f64::NEG_INFINITY;
    for c in 0..nx * ny {
        let v = if void[c] { dt[c] } else { 0.0 };
        if v > best {
            best = v;
            pk = c;
        }
    }
    let (pi, pj) = (pk / ny, pk % ny);
    let centre = |i: usize, j: usize| {
        [
            s.xy_min[0] + i as f64 * grid_res + grid_res / 2.0,
            s.xy_min[1] + j as f64 * grid_res + grid_res / 2.0,
        ]
    };
    let [cx, cy] = centre(pi, pj);
    let void_r = best * grid_res;
    let deep: Vec<bool> = (0..nx * ny).map(|c| void[c] && dt[c] > 2.0).collect();
    if !deep.iter().any(|&d| d) {
        return Void {
            cx,
            cy,
            void_r,
            void_xy: None,
        };
    }
    let (vl, _) = label(&deep, nx, ny);
    let mut peak = vl[pk];
    if peak == 0 {
        // the deep cell nearest the peak in city-block distance, first in
        // row-major order among equals
        let mut near = usize::MAX;
        for c in (0..nx * ny).filter(|&c| deep[c]) {
            let d = (c / ny).abs_diff(pi) + (c % ny).abs_diff(pj);
            if d < near {
                near = d;
                peak = vl[c];
            }
        }
    }
    let void_xy = (0..nx * ny)
        .filter(|&c| vl[c] == peak)
        .map(|c| centre(c / ny, c % ny))
        .collect();
    Void {
        cx,
        cy,
        void_r,
        void_xy: Some(void_xy),
    }
}

/// `_z_slice_void`: the void enclosed by the soma cluster `region`.
fn slice_void(s: &Slab, region: &[bool], grid_res: f64) -> Option<Void> {
    let (nx, ny) = (s.nx, s.ny);
    let occ_s: Vec<bool> = s.occ.iter().zip(region).map(|(&o, &r)| o && r).collect();
    let filled = fill_holes(region, nx, ny);
    let enc = enclosed(&occ_s, nx, ny);
    let void: Vec<bool> = (0..nx * ny)
        .map(|c| enc[c] && filled[c] && !occ_s[c])
        .collect();
    if !void.iter().any(|&v| v) {
        return None;
    }
    Some(locate_void(s, &occ_s, &void, grid_res))
}

/// Chains of consecutive levels whose voids are big enough and shift less
/// than `max_shift`; the longest of at least two levels, the first among
/// equals.
fn longest_chain(levels: &[Level], p: &ZParams) -> Option<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for (i, l) in levels.iter().enumerate() {
        if l.void_r < p.min_void_r || l.cx.is_nan() {
            if !cur.is_empty() {
                runs.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if let Some(&last) = cur.last() {
            let (dx, dy) = (l.cx - levels[last].cx, l.cy - levels[last].cy);
            if (dx * dx + dy * dy).sqrt() >= p.max_shift {
                runs.push(std::mem::take(&mut cur));
            }
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    let mut best: Option<Vec<usize>> = None;
    for r in runs.into_iter().filter(|r| r.len() >= 2) {
        if best.as_ref().is_none_or(|b| r.len() > b.len()) {
            best = Some(r);
        }
    }
    best
}

/// The scan of one mesh.
pub struct ZScan {
    pub levels: Vec<Level>,
    /// Whether each level had enough vertices to rasterise.
    has_slab: Vec<bool>,
    /// Indices into `levels` of the nucleus chain.
    pub best: Option<Vec<usize>>,
    xy: Vec<[f64; 2]>,
    p: ZParams,
}

impl ZScan {
    /// Pass 1: every level on its own, then the longest chain.
    pub fn new(verts: &[[f64; 3]], p: ZParams) -> ZScan {
        let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in verts {
            zmin = zmin.min(v[2]);
            zmax = zmax.max(v[2]);
        }
        let zs = if verts.is_empty() {
            Vec::new()
        } else {
            arange(zmin + Z_STEP, zmax - Z_STEP, Z_STEP)
        };
        let xy: Vec<[f64; 2]> = verts.iter().map(|v| [v[0], v[1]]).collect();
        let (levels, has_slab): (Vec<Level>, Vec<bool>) = crate::threads::install(|| {
            zs.par_iter()
                .map(|&z| {
                    let vi: Vec<u32> = (0..verts.len() as u32)
                        .filter(|&v| (verts[v as usize][2] - z).abs() < p.z_tol)
                        .collect();
                    let pts: Vec<[f64; 2]> = vi.iter().map(|&v| xy[v as usize]).collect();
                    let mut level = Level {
                        z,
                        cx: f64::NAN,
                        cy: f64::NAN,
                        void_r: 0.0,
                        void_xy: None,
                        vi,
                        soma_mask: None,
                    };
                    let Some((slab, region)) = rasterise(&pts, p.grid_res) else {
                        return (level, false);
                    };
                    level.soma_mask = Some(slab.cell.iter().map(|&c| region[c]).collect());
                    if let Some(v) = slice_void(&slab, &region, p.grid_res) {
                        level.cx = v.cx;
                        level.cy = v.cy;
                        level.void_r = v.void_r;
                        level.void_xy = v.void_xy;
                    }
                    (level, true)
                })
                .unzip()
        });
        let best = longest_chain(&levels, &p);
        ZScan {
            levels,
            has_slab,
            best,
            xy,
            p,
        }
    }

    /// The chain's two end levels, whose soma clusters bound the retry.
    pub fn chain_ends(&self) -> Option<(usize, usize)> {
        let b = self.best.as_ref()?;
        Some((b[0], b[b.len() - 1]))
    }

    /// Vertices of level `i`'s soma cluster.
    pub fn soma_vertices(&self, i: usize) -> Option<Vec<u32>> {
        let l = &self.levels[i];
        let mask = l.soma_mask.as_ref()?;
        Some(
            l.vi.iter()
                .zip(mask)
                .filter(|&(_, &m)| m)
                .map(|(&v, _)| v)
                .collect(),
        )
    }

    /// Levels the retry visits: outward from each end of the chain until a
    /// level that could not be rasterised.
    fn retry_levels(&self) -> Vec<usize> {
        let Some((lo, hi)) = self.chain_ends() else {
            return Vec::new();
        };
        let mut out: Vec<usize> = (0..lo).rev().take_while(|&i| self.has_slab[i]).collect();
        out.extend((hi + 1..self.levels.len()).take_while(|&i| self.has_slab[i]));
        out
    }

    /// Pass 2: hull-guided void detection on the levels next to the chain
    /// that found none, inside `hull_lo` / `hull_hi` (the hulls of the end
    /// levels' soma clusters; `None` skips that side), then the chain again.
    pub fn retry(&mut self, hull_lo: Option<&Hull>, hull_hi: Option<&Hull>) {
        let Some((lo, _)) = self.chain_ends() else {
            return;
        };
        let p = self.p;
        let todo: Vec<usize> = self
            .retry_levels()
            .into_iter()
            .filter(|&i| {
                let l = &self.levels[i];
                !(l.void_r >= p.min_void_r && !l.cx.is_nan())
            })
            .collect();
        let (levels, xy) = (&self.levels, &self.xy);
        let found: Vec<(usize, Void, Vec<bool>)> = crate::threads::install(|| {
            todo.par_iter()
                .filter_map(|&i| {
                    let hull = if i < lo { hull_lo } else { hull_hi }?;
                    let pts: Vec<[f64; 2]> = levels[i].vi.iter().map(|&v| xy[v as usize]).collect();
                    let (slab, _) = rasterise(&pts, p.grid_res)?;
                    hull_guided_void(&slab, hull, p.grid_res).map(|(v, m)| (i, v, m))
                })
                .collect()
        });
        for (i, v, mask) in found {
            let l = &mut self.levels[i];
            l.cx = v.cx;
            l.cy = v.cy;
            l.void_r = v.void_r;
            l.void_xy = v.void_xy;
            l.soma_mask = Some(mask);
        }
        if let Some(b) = longest_chain(&self.levels, &p) {
            self.best = Some(b);
        }
    }

    /// Mean void centre and mean Z over the chain.
    pub fn center(&self) -> Option<[f64; 3]> {
        let b = self.best.as_ref()?;
        let col = |f: fn(&Level) -> f64| {
            let v: Vec<f64> = b.iter().map(|&i| f(&self.levels[i])).collect();
            numpy_sum(&v) / v.len() as f64
        };
        Some([col(|l| l.cx), col(|l| l.cy), col(|l| l.z)])
    }
}

/// The retry on one slab: clusters whose centroid lies inside the hull
/// (within one cell) form the soma, and the void is sought inside the hull.
/// Returns the void and the new soma mask over the level's vertices.
fn hull_guided_void(s: &Slab, hull: &Hull, grid_res: f64) -> Option<(Void, Vec<bool>)> {
    let (nx, ny) = (s.nx, s.ny);
    let centre = |i: f64, j: f64| {
        [
            s.xy_min[0] + i * grid_res + grid_res / 2.0,
            s.xy_min[1] + j * grid_res + grid_res / 2.0,
        ]
    };
    let inside =
        |p: [f64; 2], tol: f64| hull.iter().all(|e| p[0] * e[0] + p[1] * e[1] + e[2] <= tol);
    let hull_mask: Vec<bool> = (0..nx * ny)
        .map(|c| inside(centre((c / ny) as f64, (c % ny) as f64), 0.0))
        .collect();

    // centroid of each label's cells: exact integer sums, then one division
    let n = s.n_labels as usize;
    let mut sum = vec![[0u64; 2]; n + 1];
    let mut count = vec![0u64; n + 1];
    for (c, &l) in s.labeled.iter().enumerate() {
        sum[l as usize][0] += (c / ny) as u64;
        sum[l as usize][1] += (c % ny) as u64;
        count[l as usize] += 1;
    }
    let keep: Vec<bool> = (0..=n)
        .map(|l| {
            l > 0 && count[l] > 0 && {
                let k = count[l] as f64;
                inside(centre(sum[l][0] as f64 / k, sum[l][1] as f64 / k), grid_res)
            }
        })
        .collect();
    let merged: Vec<bool> = s.labeled.iter().map(|&l| keep[l as usize]).collect();
    if !merged.iter().any(|&m| m) {
        return None;
    }
    let mask = s.cell.iter().map(|&c| merged[c]).collect();

    let occ_h: Vec<bool> = s
        .occ
        .iter()
        .zip(&hull_mask)
        .map(|(&o, &h)| o && h)
        .collect();
    let enc = enclosed(&occ_h, nx, ny);
    let void: Vec<bool> = (0..nx * ny)
        .map(|c| enc[c] && hull_mask[c] && !occ_h[c])
        .collect();
    if !void.iter().any(|&v| v) {
        return None;
    }
    Some((locate_void(s, &occ_h, &void, grid_res), mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arange_matches_numpy() {
        assert_eq!(arange(0.0, 1.0, 0.25), vec![0.0, 0.25, 0.5, 0.75]);
        assert_eq!(arange(10.0, 10.0, 1.0), Vec::<f64>::new());
        assert_eq!(arange(0.0, 0.5, 1.0), vec![0.0]);
        assert_eq!(arange(1.0, 3.0, 0.7).len(), 3);
    }

    /// A ring of vertices around an empty disk, stacked in Z: the void sits
    /// at the ring's centre on every level and chains through all of them.
    #[test]
    fn a_hollow_cylinder_has_its_void_on_the_axis() {
        let mut verts = Vec::new();
        for zi in 0..20 {
            let z = zi as f64 * 100.0;
            for k in 0..400 {
                let a = k as f64 * std::f64::consts::TAU / 400.0;
                for r in [3000.0, 3100.0] {
                    verts.push([r * a.cos(), r * a.sin(), z]);
                }
            }
        }
        let p = ZParams {
            grid_res: 200.0,
            z_tol: 150.0,
            min_void_r: 500.0,
            max_shift: 3000.0,
        };
        let scan = ZScan::new(&verts, p);
        let best = scan.best.clone().expect("a chain");
        assert_eq!(best.len(), scan.levels.len());
        let c = scan.center().unwrap();
        assert!(c[0].abs() < 300.0 && c[1].abs() < 300.0, "{c:?}");
        assert!(scan.levels[best[0]].void_r > 2000.0);
    }
}
