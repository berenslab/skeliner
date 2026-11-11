//! `skeliner.pre._z_scan`: the nucleus void, found slice by slice in Z.
//!
//! Each Z-level's vertices are rasterised; the largest cluster is the soma
//! cross-section, and the biggest empty disk enclosed in it is that level's
//! void.  Consecutive levels whose voids stay put form chains, and the
//! longest chain past the void minimum is the nucleus — a minimum the mesh
//! sets itself unless given ([`void_floor`]).  Levels just past the chain
//! whose soma ring is broken get a second look inside the convex hull of the
//! chain's end.

use mesh::geometry::{arange, numpy_sum};
use mesh::raster::{dilate3, enclosed, fill_holes, label, sq_distance_to};
use rayon::prelude::*;

/// Distance between Z-levels.
const Z_STEP: f64 = 210.0;

#[derive(Debug, Clone, Copy)]
pub struct ZParams {
    /// Grid cell size.
    pub grid_res: f64,
    /// Half-width of a level's slab.
    pub z_tol: f64,
    /// Smallest void radius a chain accepts; `None` takes it from the mesh
    /// ([`void_floor`]).
    pub min_void_r: Option<f64>,
    /// Largest XY shift of the void between chained levels.
    pub max_shift: f64,
    /// A point on or in the soma, an annotation: the nucleus is the chain
    /// past the void minimum nearest it ([`nucleus_chain`]), not the
    /// longest.
    pub near: Option<[f64; 3]>,
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
    /// The section's radius at the void's centre — the distance to the
    /// nearest cell outside the filled cluster — 0 when there is no void.
    pub sec_r: f64,
    /// Cell centres of the void's deep part, row-major.
    pub void_xy: Option<Vec<[f64; 2]>>,
    /// Vertices within `z_tol` of `z`, ascending.
    pub vi: Vec<u32>,
    /// Which of `vi` fall in the soma cluster.
    pub soma_mask: Option<Vec<bool>>,
}

/// The rasterised slab of one level, or a window of it.
struct Slab {
    nx: usize,
    ny: usize,
    /// The level grid's origin.
    xy_min: [f64; 2],
    /// Where the window's first cell sits on the level grid; zero for the
    /// whole slab.  A cell's centre is taken from its place on the level
    /// grid, so a window gives the whole slab's coordinates to the bit.
    off: [usize; 2],
    occ: Vec<bool>,
    /// Grid cell of each of the level's vertices.
    cell: Vec<usize>,
    /// Components of the dilated occupancy.
    labeled: Vec<u32>,
    n_labels: u32,
}

/// A level's grid: its origin and size, and the cell each point falls in.
struct Grid {
    nx: usize,
    ny: usize,
    xy_min: [f64; 2],
    cell: Vec<usize>,
}

/// The grid [`rasterise`] lays over the points (at least ten): their box
/// with three cells of margin below, six above.
fn grid(pts: &[[f64; 2]], grid_res: f64) -> Option<Grid> {
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
    let cell = pts
        .iter()
        .map(|p| index(p[0], xy_min[0], nx) * ny + index(p[1], xy_min[1], ny))
        .collect();
    Some(Grid {
        nx,
        ny,
        xy_min,
        cell,
    })
}

/// A void found in a slab.
struct Void {
    cx: f64,
    cy: f64,
    void_r: f64,
    void_xy: Option<Vec<[f64; 2]>>,
    /// The peak's cell, row-major.
    pk: usize,
}

/// A 2D convex hull's facet equations `(n0, n1, offset)`, inside where
/// `n . p + offset <= 0`, as scipy's `ConvexHull.equations` gives them.
pub type Hull = Vec<[f64; 3]>;

/// `_z_slice_cluster`: rasterise the points (at least ten) and label the
/// dilated occupancy.  Returns the slab and the mask of the largest
/// cluster, the lowest label among equals.
fn rasterise(pts: &[[f64; 2]], grid_res: f64) -> Option<(Slab, Vec<bool>)> {
    let Grid {
        nx,
        ny,
        xy_min,
        cell,
    } = grid(pts, grid_res)?;
    let mut occ = vec![false; nx * ny];
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
            off: [0, 0],
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
            s.xy_min[0] + (i + s.off[0]) as f64 * grid_res + grid_res / 2.0,
            s.xy_min[1] + (j + s.off[1]) as f64 * grid_res + grid_res / 2.0,
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
            pk,
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
        pk,
    }
}

/// Distance from cell `pk` to the nearest cell outside `filled`.
fn section_radius(filled: &[bool], nx: usize, ny: usize, pk: usize, grid_res: f64) -> f64 {
    let outside: Vec<bool> = filled.iter().map(|&f| !f).collect();
    (sq_distance_to(&outside, nx, ny)[pk] as f64).sqrt() * grid_res
}

/// `_z_slice_void`: the void enclosed by the soma cluster `region`, and the
/// section's radius at it.
fn slice_void(s: &Slab, region: &[bool], grid_res: f64) -> Option<(Void, f64)> {
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
    let v = locate_void(s, &occ_s, &void, grid_res);
    let sec = section_radius(&filled, nx, ny, v.pk, grid_res);
    Some((v, sec))
}

/// A level's largest cluster: which of its points fall in it, and its void
/// with the section's radius at it.
type Cluster = (Vec<bool>, Option<(Void, f64)>);

/// [`rasterise`]'s largest cluster and [`slice_void`]'s void in it, without
/// rasterising the slab's box, which on a cell a millimetre wide is mostly
/// empty: the occupied cells are labelled as [`dilate3`] and the
/// 4-connected [`label`] label them — two join when their 3x3 blocks overlap
/// or share a side, within three cells along one axis and two along the
/// other — and the largest (by dilated cells; among equals the first
/// `label` meets, the one whose first cell comes first in row-major order)
/// is rasterised alone, in a window one cell past its dilated cells.  The
/// window holds every cell the void, the filled section and both distance
/// transforms read, so they come out as on the whole slab.  Returns which
/// points fall in the cluster, and its void with the section's radius.
fn largest_section(pts: &[[f64; 2]], grid_res: f64) -> Option<Cluster> {
    let g = grid(pts, grid_res)?;
    let (nx, ny) = (g.nx as i64, g.ny as i64);
    let mut occ: Vec<usize> = g.cell.clone();
    occ.sort_unstable();
    occ.dedup();
    // union-find keeping each component's first cell, row-major, its root
    let mut parent: Vec<usize> = (0..occ.len()).collect();
    fn root(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for (k, &c) in occ.iter().enumerate() {
        let (i, j) = (c as i64 / ny, c as i64 % ny);
        // the cells joined to it that come after it in row-major order
        for di in 0..=3i64 {
            let span = if di == 3 { 2 } else { 3 };
            for dj in -span..=span {
                if (di == 0 && dj <= 0) || !(0..ny).contains(&(j + dj)) {
                    continue;
                }
                if let Ok(m) = occ.binary_search(&(((i + di) * ny + j + dj) as usize)) {
                    let (a, b) = (root(&mut parent, k), root(&mut parent, m));
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let roots: Vec<usize> = (0..occ.len()).map(|k| root(&mut parent, k)).collect();
    // each component's dilated cells, counted once
    let mut dilated: Vec<(usize, usize)> = Vec::with_capacity(9 * occ.len());
    for (k, &c) in occ.iter().enumerate() {
        let (i, j) = (c as i64 / ny, c as i64 % ny);
        for a in (i - 1).max(0)..=(i + 1).min(nx - 1) {
            for b in (j - 1).max(0)..=(j + 1).min(ny - 1) {
                dilated.push((roots[k], (a * ny + b) as usize));
            }
        }
    }
    dilated.sort_unstable();
    dilated.dedup();
    let mut best = (0usize, usize::MAX);
    let mut run = 0usize;
    for (n, &(r, _)) in dilated.iter().enumerate() {
        run += 1;
        if dilated.get(n + 1).is_none_or(|&(next, _)| next != r) {
            if run > best.0 || (run == best.0 && r < best.1) {
                best = (run, r);
            }
            run = 0;
        }
    }
    let lid = best.1;

    // the window: the cluster's dilated cells and one more
    let members: Vec<(i64, i64)> = (0..occ.len())
        .filter(|&k| roots[k] == lid)
        .map(|k| (occ[k] as i64 / ny, occ[k] as i64 % ny))
        .collect();
    let (mut lo, mut hi) = ([i64::MAX; 2], [i64::MIN; 2]);
    for &(i, j) in &members {
        lo = [lo[0].min(i), lo[1].min(j)];
        hi = [hi[0].max(i), hi[1].max(j)];
    }
    let (i0, j0) = ((lo[0] - 2).max(0), (lo[1] - 2).max(0));
    let (i1, j1) = ((hi[0] + 2).min(nx - 1), (hi[1] + 2).min(ny - 1));
    let (wnx, wny) = ((i1 - i0 + 1) as usize, (j1 - j0 + 1) as usize);
    let at = |i: i64, j: i64| (i - i0) as usize * wny + (j - j0) as usize;
    let mut w_occ = vec![false; wnx * wny];
    let mut region = vec![false; wnx * wny];
    for &(i, j) in &members {
        w_occ[at(i, j)] = true;
        for a in (i - 1).max(0)..=(i + 1).min(nx - 1) {
            for b in (j - 1).max(0)..=(j + 1).min(ny - 1) {
                region[at(a, b)] = true;
            }
        }
    }
    let window = Slab {
        nx: wnx,
        ny: wny,
        xy_min: g.xy_min,
        off: [i0 as usize, j0 as usize],
        occ: w_occ,
        cell: Vec::new(),
        labeled: Vec::new(),
        n_labels: 1,
    };
    let mask = g
        .cell
        .iter()
        .map(|c| roots[occ.binary_search(c).expect("an occupied cell")] == lid)
        .collect();
    Some((mask, slice_void(&window, &region, grid_res)))
}

/// The vertices within `z_tol` of `z`, ascending: a run of `by_z` (the
/// vertices sorted by Z, their Zs `z_of`), since `|z' - z|` shrinks as `z'`
/// nears `z` from either side, rounded as it is.
fn slab_vertices(by_z: &[u32], z_of: &[f64], z: f64, z_tol: f64) -> Vec<u32> {
    let near = |v: f64| (v - z).abs() < z_tol;
    let lo = z_of.partition_point(|&v| v < z && !near(v));
    let hi = lo + z_of[lo..].partition_point(|&v| v <= z || near(v));
    let mut vi = by_z[lo..hi].to_vec();
    vi.sort_unstable();
    vi
}

/// Chains of two or more consecutive levels with a void of at least
/// `min_void_r`, each centred less than `max_shift` from the one before.
fn chains(levels: &[Level], min_void_r: f64, max_shift: f64) -> Vec<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for (i, l) in levels.iter().enumerate() {
        if l.void_r < min_void_r || l.cx.is_nan() {
            if !cur.is_empty() {
                runs.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if let Some(&last) = cur.last() {
            let (dx, dy) = (l.cx - levels[last].cx, l.cy - levels[last].cy);
            if (dx * dx + dy * dy).sqrt() >= max_shift {
                runs.push(std::mem::take(&mut cur));
            }
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs.retain(|r| r.len() >= 2);
    runs
}

/// The nucleus chain past `min_void_r`: the longest, the first among
/// equals — or, given a point on or in the soma, the chain with a level's
/// void centre nearest the point, the first among equals.  The point picks
/// among the chains the void minimum lets through; it lets none through
/// itself.
fn nucleus_chain(
    levels: &[Level],
    min_void_r: f64,
    max_shift: f64,
    near: Option<[f64; 3]>,
) -> Option<Vec<usize>> {
    let reach = |c: &[usize], p: [f64; 3]| {
        c.iter()
            .map(|&i| {
                let l = &levels[i];
                let (dx, dy, dz) = (l.cx - p[0], l.cy - p[1], l.z - p[2]);
                dx * dx + dy * dy + dz * dz
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut best: Option<(f64, Vec<usize>)> = None;
    for r in chains(levels, min_void_r, max_shift) {
        let better = match (near, &best) {
            (_, None) => true,
            (None, Some(b)) => r.len() > b.1.len(),
            (Some(p), Some(b)) => reach(&r, p) < b.0,
        };
        if better {
            best = Some((near.map_or(0.0, |p| reach(&r, p)), r));
        }
    }
    best.map(|b| b.1)
}

/// The biggest void anywhere else in the cell than its body, when the
/// body's void beats it.  The candidate is the chain of voided levels
/// holding the cell's biggest void, its peak the first level with that
/// void; its reach is its widest section's radius past the farthest of its
/// centres from the peak's, in XY.  The body is every level within that
/// reach of the peak's void centre, and above or below it, within its XY
/// reach, every level whose section is still at least half the widest — a
/// ring doubled from its exit's is the body, as the soma's stub erosion has
/// it — so a tall soma stays whole and a neurite leaving it straight up or
/// down counts as elsewhere.  A nucleus is the cell's one void a neurite
/// cannot hold: the peak must beat the biggest void elsewhere by more than
/// the grid resolves (a cell: each radius is uncertain by half of one);
/// `None` when it does not — no nucleus.
fn void_floor(levels: &[Level], grid_res: f64, max_shift: f64) -> Option<f64> {
    let peak = |c: &[usize]| c.iter().map(|&i| levels[i].void_r).fold(0.0, f64::max);
    let mut cand: Option<Vec<usize>> = None;
    for c in chains(levels, f64::MIN_POSITIVE, max_shift) {
        if cand.as_ref().is_none_or(|b| peak(&c) > peak(b)) {
            cand = Some(c);
        }
    }
    let cand = cand?;
    let top = peak(&cand);
    let pk = &levels[*cand.iter().find(|&&i| levels[i].void_r == top)?];
    let (px, py, pz) = (pk.cx, pk.cy, pk.z);
    let spread = cand
        .iter()
        .map(|&i| {
            let l = &levels[i];
            ((l.cx - px) * (l.cx - px) + (l.cy - py) * (l.cy - py)).sqrt()
        })
        .fold(0.0, f64::max);
    let widest = cand.iter().map(|&i| levels[i].sec_r).fold(0.0, f64::max);
    let foot = widest + spread;
    let other = levels
        .iter()
        .filter(|l| {
            let (dx, dy, dz) = (l.cx - px, l.cy - py, l.z - pz);
            let flat = (dx * dx + dy * dy).sqrt();
            let body = (dx * dx + dy * dy + dz * dz).sqrt() <= foot
                || (flat <= foot && 2.0 * l.sec_r >= widest);
            l.void_r > 0.0 && !l.cx.is_nan() && !body
        })
        .map(|l| l.void_r)
        .fold(0.0, f64::max);
    (top - other > grid_res).then_some(other)
}

/// The scan of one mesh.
pub struct ZScan {
    pub levels: Vec<Level>,
    /// Whether each level had enough vertices to rasterise.
    has_slab: Vec<bool>,
    /// Indices into `levels` of the nucleus chain.
    pub best: Option<Vec<usize>>,
    /// The void a level needs to join a chain — given, or a grid cell past
    /// the biggest void elsewhere in the cell ([`void_floor`]) — or `None`
    /// when the mesh has no nucleus.
    pub min_void_r: Option<f64>,
    xy: Vec<[f64; 2]>,
    p: ZParams,
}

impl ZScan {
    /// Pass 1: every level on its own, then the void minimum and the
    /// longest chain past it.
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
        // the vertices by Z, each level's slab a run of them
        let mut by_z: Vec<u32> = (0..verts.len() as u32)
            .filter(|&v| !verts[v as usize][2].is_nan())
            .collect();
        let z_order = |a: &u32, b: &u32| verts[*a as usize][2].total_cmp(&verts[*b as usize][2]);
        if by_z.len() < threads::MIN_PARALLEL {
            by_z.sort_unstable_by(z_order);
        } else {
            threads::install(|| by_z.par_sort_unstable_by(z_order));
        }
        let z_of: Vec<f64> = by_z.iter().map(|&v| verts[v as usize][2]).collect();
        let (levels, has_slab): (Vec<Level>, Vec<bool>) = threads::install(|| {
            zs.par_iter()
                .map(|&z| {
                    let vi = slab_vertices(&by_z, &z_of, z, p.z_tol);
                    let pts: Vec<[f64; 2]> = vi.iter().map(|&v| xy[v as usize]).collect();
                    let mut level = Level {
                        z,
                        cx: f64::NAN,
                        cy: f64::NAN,
                        void_r: 0.0,
                        sec_r: 0.0,
                        void_xy: None,
                        vi,
                        soma_mask: None,
                    };
                    let Some((mask, void)) = largest_section(&pts, p.grid_res) else {
                        return (level, false);
                    };
                    level.soma_mask = Some(mask);
                    if let Some((v, sec)) = void {
                        level.cx = v.cx;
                        level.cy = v.cy;
                        level.void_r = v.void_r;
                        level.sec_r = sec;
                        level.void_xy = v.void_xy;
                    }
                    (level, true)
                })
                .unzip()
        });
        // the chain's levels beat the void elsewhere by a cell, too
        let min_void_r = match p.min_void_r {
            Some(r) => Some(r),
            None => void_floor(&levels, p.grid_res, p.max_shift).map(|o| o + p.grid_res),
        };
        let best = min_void_r.and_then(|r| nucleus_chain(&levels, r, p.max_shift, p.near));
        ZScan {
            levels,
            has_slab,
            best,
            min_void_r,
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
        let (Some((lo, _)), Some(min_void_r)) = (self.chain_ends(), self.min_void_r) else {
            return;
        };
        let p = self.p;
        // a level whose largest cluster lies outside the chain end's hull
        // (its centroid, within a cell, as the second look takes clusters)
        // saw another structure, not a broken soma ring: it is not looked
        // at again
        let todo: Vec<usize> = self
            .retry_levels()
            .into_iter()
            .filter(|&i| {
                let l = &self.levels[i];
                let hull = if i < lo { hull_lo } else { hull_hi };
                let ours = match (hull, self.soma_vertices(i)) {
                    (Some(h), Some(sv)) if !sv.is_empty() => {
                        let n = sv.len() as f64;
                        let col = |k: usize| {
                            numpy_sum(
                                &sv.iter()
                                    .map(|&v| self.xy[v as usize][k])
                                    .collect::<Vec<_>>(),
                            ) / n
                        };
                        let (x, y) = (col(0), col(1));
                        h.iter().all(|e| x * e[0] + y * e[1] + e[2] <= p.grid_res)
                    }
                    _ => true,
                };
                !(l.void_r >= min_void_r && !l.cx.is_nan()) && ours
            })
            .collect();
        let (levels, xy) = (&self.levels, &self.xy);
        let found: Vec<(usize, (Void, f64), Vec<bool>)> = threads::install(|| {
            todo.par_iter()
                .filter_map(|&i| {
                    let hull = if i < lo { hull_lo } else { hull_hi }?;
                    let pts: Vec<[f64; 2]> = levels[i].vi.iter().map(|&v| xy[v as usize]).collect();
                    let (slab, _) = rasterise(&pts, p.grid_res)?;
                    hull_guided_void(&slab, hull, p.grid_res).map(|(v, m)| (i, v, m))
                })
                .collect()
        });
        for (i, (v, sec), mask) in found {
            let l = &mut self.levels[i];
            l.cx = v.cx;
            l.cy = v.cy;
            l.void_r = v.void_r;
            l.sec_r = sec;
            l.void_xy = v.void_xy;
            l.soma_mask = Some(mask);
        }
        if let Some(b) = nucleus_chain(&self.levels, min_void_r, p.max_shift, p.near) {
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
/// Returns the void with the section's radius at it (the hull's), and the
/// new soma mask over the level's vertices.
fn hull_guided_void(s: &Slab, hull: &Hull, grid_res: f64) -> Option<((Void, f64), Vec<bool>)> {
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
    let v = locate_void(s, &occ_h, &void, grid_res);
    let sec = section_radius(&hull_mask, nx, ny, v.pk, grid_res);
    Some(((v, sec), mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*, in [0, 1).
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// What the whole slab gives for the largest cluster: which points fall
    /// in it, and its void with the section's radius.
    type Section = (
        Vec<bool>,
        Option<(f64, f64, f64, Option<Vec<[f64; 2]>>, f64)>,
    );

    fn on_the_whole_slab(pts: &[[f64; 2]], grid_res: f64) -> Option<Section> {
        let (slab, region) = rasterise(pts, grid_res)?;
        let mask = slab.cell.iter().map(|&c| region[c]).collect();
        let void = slice_void(&slab, &region, grid_res);
        Some((
            mask,
            void.map(|(v, s)| (v.cx, v.cy, v.void_r, v.void_xy, s)),
        ))
    }

    fn in_its_window(pts: &[[f64; 2]], grid_res: f64) -> Option<Section> {
        let (mask, void) = largest_section(pts, grid_res)?;
        Some((
            mask,
            void.map(|(v, s)| (v.cx, v.cy, v.void_r, v.void_xy, s)),
        ))
    }

    /// The largest cluster found from its occupied cells and rasterised in
    /// a window is the whole slab's, to the bit: hollow rings among specks
    /// spread far, specks on a lattice of 2–4 cells (blocks that touch, by
    /// a side or a corner, and that do not), rings of one size (a tie), a
    /// slab of a few points.
    #[test]
    fn the_largest_section_is_the_whole_slabs() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let grid_res = 200.0;
        let (mut voids, mut mixed) = (0, 0);
        for case in 0..300 {
            let mut pts: Vec<[f64; 2]> = Vec::new();
            let rings = 1 + case % 3;
            let r = 400.0 + 3600.0 * uniform(&mut s);
            for k in 0..rings {
                let (cx, cy) = if case % 7 == 0 {
                    // the same ring twice: a tie
                    (k as f64 * 20_000.0, 0.0)
                } else {
                    (40_000.0 * uniform(&mut s), 40_000.0 * uniform(&mut s))
                };
                let rr = if case % 7 == 0 {
                    r
                } else {
                    400.0 + 3600.0 * uniform(&mut s)
                };
                let n = (rr / 20.0) as usize;
                for q in 0..n {
                    let a = q as f64 * std::f64::consts::TAU / n as f64;
                    pts.push([cx + rr * a.cos(), cy + rr * a.sin()]);
                }
            }
            let spread = [1_000.0, 30_000.0, 200_000.0][case % 3];
            for _ in 0..(case * 7) % 300 {
                pts.push([spread * uniform(&mut s), spread * uniform(&mut s)]);
            }
            let step = grid_res * (2.0 + (case % 3) as f64);
            for a in 0..(case % 5) {
                for b in 0..(case % 4) {
                    pts.push([-20_000.0 + a as f64 * step, b as f64 * step]);
                }
            }
            if case % 11 == 0 {
                pts.truncate(10 + case % 5);
            }
            let got = in_its_window(&pts, grid_res);
            assert_eq!(got, on_the_whole_slab(&pts, grid_res), "case {case}");
            let (mask, void) = got.expect("ten points or more");
            voids += usize::from(void.is_some());
            mixed += usize::from(mask.iter().any(|&m| m) && mask.iter().any(|&m| !m));
        }
        // the fixtures: most slabs hold a void and more than one cluster
        assert!(voids > 200 && mixed > 200, "{voids} voids, {mixed} mixed");
        let few = [[0.0, 0.0]; 9];
        assert!(in_its_window(&few, grid_res).is_none());
    }

    /// A level's vertices taken from the Z-sorted run are the ones the plain
    /// filter keeps, in their order.
    #[test]
    fn a_slab_is_a_run_of_the_vertices_by_z() {
        let mut s = 7u64;
        let zs: Vec<f64> = (0..5000)
            .map(|k| {
                if k % 9 == 0 {
                    // coarse values: ties, and points on the slab's edges
                    (uniform(&mut s) * 40.0).floor() * 150.0
                } else {
                    uniform(&mut s) * 6000.0
                }
            })
            .collect();
        let mut by_z: Vec<u32> = (0..zs.len() as u32).collect();
        by_z.sort_unstable_by(|&a, &b| zs[a as usize].total_cmp(&zs[b as usize]));
        let z_of: Vec<f64> = by_z.iter().map(|&v| zs[v as usize]).collect();
        for k in 0..400 {
            let z = k as f64 * 15.0;
            let want: Vec<u32> = (0..zs.len() as u32)
                .filter(|&v| (zs[v as usize] - z).abs() < 150.0)
                .collect();
            assert_eq!(slab_vertices(&by_z, &z_of, z, 150.0), want, "z {z}");
        }
    }

    /// A soma's chain broken in two by levels with no ring: the scan takes
    /// the longer run; a point on the soma takes the run nearest it, and a
    /// point lets through no chain the void minimum stops.
    #[test]
    fn a_point_picks_the_chain_nearest_it() {
        let mut verts = Vec::new();
        rings(&mut verts, [0.0, 0.0, 0.0], 3000.0, 30);
        rings(&mut verts, [0.0, 0.0, 3800.0], 3000.0, 20);
        let at = |near: Option<[f64; 3]>| {
            let p = ZParams {
                grid_res: 200.0,
                z_tol: 150.0,
                min_void_r: None,
                max_shift: 3000.0,
                near,
            };
            let scan = ZScan::new(&verts, p);
            let best = scan.best.clone().expect("a chain");
            (scan.levels[best[0]].z, scan.levels[best[best.len() - 1]].z)
        };
        let (lo, hi) = at(None);
        assert!(hi < 3000.0, "the longer run, below the gap: {lo}–{hi}");
        let (lo, _) = at(Some([0.0, 3000.0, 4600.0]));
        assert!(lo > 3600.0, "the run the point is on, above the gap: {lo}");
        assert_eq!(at(Some([0.0, 3000.0, 1500.0])), at(None));

        // two rings alike: no nucleus, wherever the point is
        let mut twins = Vec::new();
        rings(&mut twins, [0.0, 0.0, 0.0], 600.0, 20);
        rings(&mut twins, [20000.0, 0.0, 3000.0], 600.0, 20);
        let p = ZParams {
            grid_res: 200.0,
            z_tol: 150.0,
            min_void_r: None,
            max_shift: 3000.0,
            near: Some([0.0, 600.0, 1000.0]),
        };
        assert!(ZScan::new(&twins, p).best.is_none());
    }

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
            min_void_r: Some(500.0),
            max_shift: 3000.0,
            near: None,
        };
        let scan = ZScan::new(&verts, p);
        let best = scan.best.clone().expect("a chain");
        assert_eq!(best.len(), scan.levels.len());
        let c = scan.center().unwrap();
        assert!(c[0].abs() < 300.0 && c[1].abs() < 300.0, "{c:?}");
        assert!(scan.levels[best[0]].void_r > 2000.0);
    }

    /// Hollow rings of radius `r` round `(x, y)`, stacked 100 apart over
    /// `n` levels from `z`.
    fn rings(verts: &mut Vec<[f64; 3]>, [x, y, z]: [f64; 3], r: f64, n: usize) {
        for zi in 0..n {
            for k in 0..400 {
                let a = k as f64 * std::f64::consts::TAU / 400.0;
                for rr in [r, r + 100.0] {
                    verts.push([x + rr * a.cos(), y + rr * a.sin(), z + zi as f64 * 100.0]);
                }
            }
        }
    }

    /// The mesh sets the void minimum: a wide hollow standing out of the
    /// cell's other voids is the nucleus, its chain a grid cell past the
    /// biggest of them; two alike stand out of nothing.
    #[test]
    fn the_void_minimum_is_the_biggest_void_elsewhere() {
        let p = ZParams {
            grid_res: 200.0,
            z_tol: 150.0,
            min_void_r: None,
            max_shift: 3000.0,
            near: None,
        };
        let (mut body, mut twins) = (Vec::new(), Vec::new());
        rings(&mut body, [0.0, 0.0, 0.0], 3000.0, 20);
        rings(&mut body, [20000.0, 0.0, 3000.0], 600.0, 20);
        let scan = ZScan::new(&body, p);
        let floor = scan.min_void_r.expect("a nucleus");
        // a grid cell past the thin hollow's void
        assert!((700.0..900.0).contains(&floor), "{floor}");
        let c = scan.center().expect("a chain");
        assert!(
            c[0].abs() < 300.0 && c[1].abs() < 300.0 && c[2] < 2000.0,
            "{c:?}"
        );

        rings(&mut twins, [0.0, 0.0, 0.0], 600.0, 20);
        rings(&mut twins, [20000.0, 0.0, 3000.0], 600.0, 20);
        let scan = ZScan::new(&twins, p);
        assert!(scan.min_void_r.is_none() && scan.best.is_none());
    }
}
