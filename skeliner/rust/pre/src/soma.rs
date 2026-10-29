//! `skeliner.pre.find_soma` (the ring cutoff) and `_assign_soma_verts`.
//!
//! Rings of vertices grow outward from the nucleus seeds over the soma's
//! component; the soma ends where the largest connected piece of a ring
//! stops growing.  An ellipsoid fitted to the rings then claims vertices,
//! absorbs pockets, sheds neurite stubs, and is refitted as its vertex set
//! changes.
//!
//! The fits themselves are the caller's (`Soma.fit`, numpy's covariance and
//! eigen-solver), so the ellipsoids are bit-identical to the reference's.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};

use mesh::components::LowestRootSets;
use mesh::geometry::{numpy_argmin, numpy_order, numpy_sum, row_times_matrix};
use rayon::prelude::*;

/// An ellipsoid as `skeliner.dataclass.Soma` tests it: a point is inside
/// when `|(x - center) @ w|^2 <= 1`, where `w = R / axes`.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsoid {
    pub center: [f64; 3],
    pub w: [[f64; 3]; 3],
}

impl Ellipsoid {
    /// `Soma._body_coords`: `(x - center) @ w`, rounded as numpy's matmul.
    pub fn body_coords(&self, x: &[f64; 3]) -> [f64; 3] {
        let d = [
            x[0] - self.center[0],
            x[1] - self.center[1],
            x[2] - self.center[2],
        ];
        row_times_matrix(&d, &self.w)
    }

    /// `|body_coords(x)|^2`, summed in numpy's order.
    pub fn rho2(&self, x: &[f64; 3]) -> f64 {
        let xi = self.body_coords(x);
        xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
    }

    pub fn contains(&self, x: &[f64; 3]) -> bool {
        self.rho2(x) <= 1.0
    }
}

/// A fitted soma: its ellipsoid and smallest semi-axis.
#[derive(Debug, Clone, Copy)]
pub struct Fit {
    pub ellipsoid: Ellipsoid,
    pub axes_min: f64,
}

/// Which fit the caller is asked for.  The first one, to the BFS rings,
/// must succeed; a refit may fail (`Soma.fit`'s `ValueError`), which keeps
/// the previous ellipsoid with the new vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitKind {
    Initial,
    Refit,
}

/// The inputs [`ring_cutoff`] reads about the mesh.
pub struct SomaMesh<'a> {
    pub faces: &'a [[u32; 3]],
    pub verts: &'a [[f64; 3]],
    /// Face components; degenerate faces are -2.
    pub labels: &'a [i64],
    pub main: i64,
    /// Organelle faces.
    pub organelle: &'a [bool],
    /// Mean unique-edge length.
    pub avg_edge: f64,
}

/// What `find_nucleus_center` found.
pub struct Nucleus<'a> {
    pub center: [f64; 3],
    /// `z_hi - z_lo` of the nucleus chain.
    pub z_span: f64,
    /// Soma-cluster vertices at the chain's middle level.
    pub seed_vi: &'a [u32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SomaFound {
    /// Index, among the fits that succeeded, of the one that sets the
    /// final ellipsoid.
    pub fit: usize,
    /// The soma's vertices, ascending.
    pub verts: Vec<u32>,
    pub cutoff: usize,
    pub max_ring: usize,
}

#[derive(Debug)]
pub enum SomaError<E> {
    /// The caller's fit failed.
    Fit(E),
    /// No ring past the seeds: numpy's argmax of an empty slice.
    NoRings,
}

/// Vertex adjacency of the soma's component, as the reference's
/// `defaultdict(list)` holds it: neighbour lists in insertion order, keys
/// in first-seen order.
struct Graph {
    start: Vec<usize>,
    nbr: Vec<u32>,
    keys: Vec<u32>,
    is_key: Vec<bool>,
}

impl Graph {
    fn new(faces: &[[u32; 3]], soma_fi: &[u32], nv: usize) -> Graph {
        let mut deg = vec![0usize; nv];
        let mut is_key = vec![false; nv];
        let mut keys = Vec::new();
        for &f in soma_fi {
            let v = faces[f as usize];
            for i in 0..3 {
                let (a, b) = (v[i], v[(i + 1) % 3]);
                for x in [a, b] {
                    if !is_key[x as usize] {
                        is_key[x as usize] = true;
                        keys.push(x);
                    }
                }
                deg[a as usize] += 1;
                deg[b as usize] += 1;
            }
        }
        let mut start = vec![0usize; nv + 1];
        for v in 0..nv {
            start[v + 1] = start[v] + deg[v];
        }
        let mut fill = start.clone();
        let mut nbr = vec![0u32; start[nv]];
        for &f in soma_fi {
            let v = faces[f as usize];
            for i in 0..3 {
                let (a, b) = (v[i], v[(i + 1) % 3]);
                nbr[fill[a as usize]] = b;
                fill[a as usize] += 1;
                nbr[fill[b as usize]] = a;
                fill[b as usize] += 1;
            }
        }
        Graph {
            start,
            nbr,
            keys,
            is_key,
        }
    }

    fn nbrs(&self, v: u32) -> &[u32] {
        &self.nbr[self.start[v as usize]..self.start[v as usize + 1]]
    }

    /// Connected pieces of the vertices `inside` marks, visited from each
    /// of `starts` in turn.
    fn pieces(&self, starts: &[u32], inside: &[bool], seen: &mut [bool]) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut queue = VecDeque::new();
        for &s in starts {
            if seen[s as usize] {
                continue;
            }
            let mut comp = Vec::new();
            queue.push_back(s);
            while let Some(v) = queue.pop_front() {
                if seen[v as usize] {
                    continue;
                }
                seen[v as usize] = true;
                comp.push(v);
                for &nv in self.nbrs(v) {
                    if inside[nv as usize] && !seen[nv as usize] {
                        queue.push_back(nv);
                    }
                }
            }
            out.push(comp);
        }
        out
    }

    /// The connected pieces of the vertices `inside` marks, neighbours in
    /// one piece where `joins(v, u)`; found in parallel, the same pieces
    /// whatever the order.
    fn pieces_of(&self, inside: &[bool], joins: impl Fn(u32, u32) -> bool + Sync) -> Pieces {
        let roots: Vec<u32> = threads::install(|| {
            let sets = LowestRootSets::new(inside.len());
            (0..inside.len() as u32).into_par_iter().for_each(|v| {
                if inside[v as usize] {
                    for &u in self.nbrs(v) {
                        if u > v && inside[u as usize] && joins(v, u) {
                            sets.join(v, u);
                        }
                    }
                }
            });
            (0..inside.len() as u32)
                .into_par_iter()
                .map(|v| {
                    if inside[v as usize] {
                        sets.root(v)
                    } else {
                        NONE
                    }
                })
                .collect()
        });
        let mut size = vec![0usize; roots.len()];
        for &r in &roots {
            if r != NONE {
                size[r as usize] += 1;
            }
        }
        let list = (0..roots.len() as u32)
            .filter(|&v| roots[v as usize] == v)
            .collect();
        Pieces { roots, size, list }
    }
}

/// Outside every piece.
const NONE: u32 = u32::MAX;

/// Connected pieces, each named by its lowest vertex.
struct Pieces {
    /// Each vertex's piece, [`NONE`] outside them.
    roots: Vec<u32>,
    /// Each piece's size, at its name.
    size: Vec<usize>,
    /// The pieces, ascending: in the order a search from the lowest
    /// vertex up meets them.
    list: Vec<u32>,
}

impl Pieces {
    fn size(&self, piece: u32) -> usize {
        self.size[piece as usize]
    }

    /// The vertices of the pieces `chosen` marks (by name), ascending.
    fn members<'a>(&'a self, chosen: &'a [bool]) -> impl Iterator<Item = u32> + 'a {
        (0..self.roots.len() as u32).filter(move |&v| {
            let r = self.roots[v as usize];
            r != NONE && chosen[r as usize]
        })
    }
}

/// A vertex set kept as a mask plus its size.
struct Set {
    mask: Vec<bool>,
    len: usize,
}

impl Set {
    fn empty(n: usize) -> Set {
        Set {
            mask: vec![false; n],
            len: 0,
        }
    }

    fn from(n: usize, items: impl IntoIterator<Item = u32>) -> Set {
        let mut s = Set::empty(n);
        for v in items {
            s.insert(v);
        }
        s
    }

    fn has(&self, v: u32) -> bool {
        self.mask[v as usize]
    }

    fn insert(&mut self, v: u32) {
        if !self.mask[v as usize] {
            self.mask[v as usize] = true;
            self.len += 1;
        }
    }

    fn remove(&mut self, v: u32) {
        if self.mask[v as usize] {
            self.mask[v as usize] = false;
            self.len -= 1;
        }
    }

    fn sorted(&self) -> Vec<u32> {
        (0..self.mask.len() as u32)
            .filter(|&v| self.mask[v as usize])
            .collect()
    }
}

/// `_otsu_threshold`: the midpoint between consecutive distinct sorted
/// values that maximises the between-class variance, with numpy's sums.
pub fn otsu_threshold(values: &[f64]) -> (f64, f64) {
    let mut x = values.to_vec();
    x.sort_by(numpy_order);
    let n = x.len();
    if n < 2 {
        return (x[0], 0.0);
    }
    let mean = numpy_sum(&x) / n as f64;
    let sq: Vec<f64> = x.iter().map(|&v| (v - mean) * (v - mean)).collect();
    let total_var = numpy_sum(&sq) / n as f64;
    if total_var < 1e-12 {
        return (x[0], 0.0);
    }
    let mut best_thresh = x[0];
    let mut best_between = 0.0;
    for i in 1..n {
        if x[i] == x[i - 1] {
            continue;
        }
        let (left, right) = (&x[..i], &x[i..]);
        let w0 = i as f64 / n as f64;
        let w1 = (n - i) as f64 / n as f64;
        let d = numpy_sum(left) / i as f64 - numpy_sum(right) / (n - i) as f64;
        let between = w0 * w1 * (d * d);
        if between > best_between {
            best_between = between;
            best_thresh = (x[i - 1] + x[i]) / 2.0;
        }
    }
    (best_thresh, best_between / total_var)
}

/// `!(a <= b)`: true for NaN, as the reference's `if d <= b ... else`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn not_within(a: f64, b: f64) -> bool {
    !(a <= b)
}

/// numpy's `max`: NaN if any value is NaN.
fn nan_max(x: impl Iterator<Item = f64>) -> f64 {
    x.fold(f64::NEG_INFINITY, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    })
}

/// `numpy.percentile(x, 25)` (linear interpolation, NaN if any value is
/// NaN); `None` when empty.
fn percentile25(x: &mut [f64]) -> Option<f64> {
    let n = x.len();
    if n == 0 {
        return None;
    }
    if x.iter().any(|v| v.is_nan()) {
        return Some(f64::NAN);
    }
    x.sort_by(numpy_order);
    let v = (n - 1) as f64 * 0.25;
    if v >= (n - 1) as f64 {
        return Some(x[n - 1]);
    }
    let prev = v.floor();
    let (a, b) = (x[prev as usize], x[prev as usize + 1]);
    let t = v - prev;
    let d = b - a;
    Some(if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    })
}

/// The soma's evolving state: the vertex set and which fit stands.
struct Current {
    fit: Fit,
    index: usize,
    n_fits: usize,
    verts: Vec<u32>,
    /// A neurite small enough to pass for a pocket by its size was kept out
    /// of the soma: the exits are eroded as after a pruned tube.
    held_out: bool,
    /// The vertices of the pieces outside the soma reaching out past three
    /// times their junction's radius: tubes, their bases in it stubs.
    tube: Vec<bool>,
}

impl Current {
    /// The reference's refit: four or more vertices are fitted (keeping the
    /// old ellipsoid if the fit fails); the soma takes the vertices either way.
    fn refit<E>(
        &mut self,
        sv: Vec<u32>,
        fit: &mut impl FnMut(&[u32], FitKind) -> Result<Option<Fit>, E>,
    ) -> Result<(), SomaError<E>> {
        if sv.len() >= 4
            && let Some(f) = fit(&sv, FitKind::Refit).map_err(SomaError::Fit)?
        {
            self.fit = f;
            self.index = self.n_fits;
            self.n_fits += 1;
        }
        self.verts = sv;
        Ok(())
    }
}

/// `_assign_soma_verts` with the soma's component given: containment,
/// dilation along edges, pocket absorption, vertices of other components
/// inside the ellipsoid, then a refit.
fn assign_soma_verts<E>(
    m: &SomaMesh<'_>,
    g: &Graph,
    cur: &mut Current,
    fit: &mut impl FnMut(&[u32], FitKind) -> Result<Option<Fit>, E>,
) -> Result<(), SomaError<E>> {
    let nv = m.verts.len();
    let soma = cur.fit;
    let e = &soma.ellipsoid;
    let inside = |on_component: bool| -> Vec<u32> {
        threads::install(|| {
            (0..nv as u32)
                .into_par_iter()
                .filter(|&v| {
                    g.is_key[v as usize] == on_component && e.contains(&m.verts[v as usize])
                })
                .collect()
        })
    };
    let contained = inside(true);
    let mut soma_set = Set::from(nv, contained.iter().copied());

    // dilation along edges, a ring a round until a round adds nothing:
    // everything reached from the contained vertices through vertices
    // under the limit
    let edge_in_body = m.avg_edge / soma.axes_min;
    let dilation_limit = 1.0 + 3.0 * edge_in_body;
    let mut stack = contained;
    while let Some(v) = stack.pop() {
        for &u in g.nbrs(v) {
            if !soma_set.has(u) && e.rho2(&m.verts[u as usize]).sqrt() < dilation_limit {
                soma_set.insert(u);
                stack.push(u);
            }
        }
    }

    // pockets: the pieces of the component outside the soma that do not
    // stand out of it like a neurite, absorbed while smaller than the soma
    // (a round at a time, as it grows).  Every piece is bounded by the soma
    // alone (a vertex's neighbours are on the component: in the soma or in
    // its piece), and absorbing pieces leaves the others as they were, so
    // the pieces are found once for every round.
    let outside: Vec<bool> = threads::install(|| {
        (0..nv as u32)
            .into_par_iter()
            .map(|v| g.is_key[v as usize] && !soma_set.has(v))
            .collect()
    });
    let pieces = g.pieces_of(&outside, |_, _| true);
    let (neurite, tube) = stands_out(m, g, e, &soma_set, &pieces);
    // a neurite under Otsu's split of the sizes of the pieces smaller than
    // the soma passes for a pocket by its size: its base inside the
    // dilation is a stub
    let first: Vec<u32> = pieces
        .list
        .iter()
        .copied()
        .filter(|&p| pieces.size(p) < soma_set.len)
        .collect();
    let first_thresh = if first.len() >= 2 {
        let sizes: Vec<f64> = first.iter().map(|&p| pieces.size(p) as f64).collect();
        otsu_threshold(&sizes).0
    } else {
        0.0
    };
    cur.held_out = first
        .iter()
        .any(|&p| neurite[p as usize] && pieces.size(p) as f64 <= first_thresh);
    cur.tube = (0..nv)
        .map(|v| pieces.roots[v] != NONE && tube[pieces.roots[v] as usize])
        .collect();
    let mut left: Vec<u32> = pieces
        .list
        .iter()
        .copied()
        .filter(|&p| !neurite[p as usize])
        .collect();
    let mut absorb = vec![false; nv];
    let mut soma_len = soma_set.len;
    for _ in 0..10 {
        let mut absorbed = 0;
        for &p in &left {
            if pieces.size(p) < soma_len {
                absorbed += pieces.size(p);
                absorb[p as usize] = true;
            }
        }
        if absorbed == 0 {
            break;
        }
        soma_len += absorbed;
        left.retain(|&p| !absorb[p as usize]);
    }
    for v in pieces.members(&absorb) {
        soma_set.insert(v);
    }

    // vertices of other components inside the ellipsoid
    for v in inside(false) {
        soma_set.insert(v);
    }
    cur.refit(soma_set.sorted(), fit)
}

/// Which of the `pieces` outside the soma stand out of it like a neurite,
/// by name: never a pocket of its surface, as `break_up_mesh` never absorbs
/// one.  Its farthest vertex beyond the ellipsoid `e` — radially, from the
/// centre — lies past half the ellipsoid's equivalent radius and three
/// times its junction's radius, the mean distance from their centroid of
/// its vertices next to the soma.  With them, which are tubes: past three
/// times the junction's radius, however short of half the radius.
fn stands_out(
    m: &SomaMesh<'_>,
    g: &Graph,
    e: &Ellipsoid,
    soma: &Set,
    pieces: &Pieces,
) -> (Vec<bool>, Vec<bool>) {
    let nv = m.verts.len();
    let axes = [0, 1, 2].map(|k| {
        1.0 / (e.w[0][k] * e.w[0][k] + e.w[1][k] * e.w[1][k] + e.w[2][k] * e.w[2][k]).sqrt()
    });
    let equiv = (axes[0] * axes[1] * axes[2]).powf(1.0 / 3.0);
    let mut reach = vec![f64::NEG_INFINITY; nv];
    let mut rim: Vec<(u32, u32)> = Vec::new();
    for v in 0..nv as u32 {
        let r = pieces.roots[v as usize];
        if r == NONE {
            continue;
        }
        let x = &m.verts[v as usize];
        let d = [x[0] - e.center[0], x[1] - e.center[1], x[2] - e.center[2]];
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let rho = e.rho2(x).sqrt();
        let beyond = if rho > 1.0 {
            dist * (1.0 - 1.0 / rho)
        } else {
            0.0
        };
        if beyond > reach[r as usize] {
            reach[r as usize] = beyond;
        }
        if g.nbrs(v).iter().any(|&u| soma.has(u)) {
            rim.push((r, v));
        }
    }
    rim.sort_unstable();
    let mut out = vec![false; nv];
    let mut tube = vec![false; nv];
    let mut has_rim = vec![false; nv];
    for run in rim.chunk_by(|a, b| a.0 == b.0) {
        let r = run[0].0 as usize;
        has_rim[r] = true;
        let pts: Vec<[f64; 3]> = run.iter().map(|&(_, v)| m.verts[v as usize]).collect();
        let n = pts.len() as f64;
        let centre =
            [0, 1, 2].map(|k| numpy_sum(&pts.iter().map(|p| p[k]).collect::<Vec<_>>()) / n);
        let dists: Vec<f64> = pts
            .iter()
            .map(|p| {
                let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect();
        let junction = numpy_sum(&dists) / n;
        tube[r] = reach[r] > 3.0 * junction;
        out[r] = reach[r] > 0.5 * equiv && tube[r];
    }
    for &p in &pieces.list {
        if !has_rim[p as usize] {
            out[p as usize] = reach[p as usize] > 0.5 * equiv;
        }
    }
    (out, tube)
}

/// `find_soma` from the nucleus on: everything but the
/// nucleus scan, the ellipsoid fits (`fit`) and the verbose report.
pub fn ring_cutoff<E>(
    m: &SomaMesh<'_>,
    nuc: &Nucleus<'_>,
    mut fit: impl FnMut(&[u32], FitKind) -> Result<Option<Fit>, E>,
) -> Result<SomaFound, SomaError<E>> {
    let nv = m.verts.len();
    let nf = m.faces.len();

    // the component the seeds lie on: the most common among their
    // vertices' components (each vertex takes its last face's label)
    let mut soma_ci = m.main;
    if !nuc.seed_vi.is_empty() {
        let mut vert_comp = vec![-1i64; nv];
        for (f, face) in m.faces.iter().enumerate() {
            for &v in face {
                vert_comp[v as usize] = m.labels[f];
            }
        }
        let mut counts: Vec<(i64, usize)> = Vec::new();
        for &v in nuc.seed_vi {
            let c = vert_comp[v as usize];
            if c < 0 {
                continue;
            }
            match counts.iter_mut().find(|(k, _)| *k == c) {
                Some((_, n)) => *n += 1,
                None => counts.push((c, 1)),
            }
        }
        if let Some(&(c, _)) = counts
            .iter()
            .reduce(|best, x| if x.1 > best.1 { x } else { best })
        {
            soma_ci = c;
        }
    }
    let soma_fi: Vec<u32> = threads::install(|| {
        (0..nf as u32)
            .into_par_iter()
            .filter(|&f| m.labels[f as usize] == soma_ci)
            .collect()
    });
    let g = Graph::new(m.faces, &soma_fi, nv);

    // BFS rings from the seeds on the component, or from the component's
    // vertex nearest the nucleus centre
    let mut seeds: Vec<u32> = nuc
        .seed_vi
        .iter()
        .copied()
        .filter(|&v| g.is_key[v as usize])
        .collect();
    if seeds.is_empty() {
        let dist = |v: u32| {
            let p = m.verts[v as usize];
            let d = [
                p[0] - nuc.center[0],
                p[1] - nuc.center[1],
                p[2] - nuc.center[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        if !g.keys.is_empty() {
            let d: Vec<f64> = g.keys.iter().map(|&v| dist(v)).collect();
            seeds.push(g.keys[numpy_argmin(&d)]);
        }
    }
    // the search's order is ring after ring, ring 0 the seeds as listed
    let mut level = vec![u32::MAX; nv];
    let mut order: Vec<u32> = Vec::new();
    for &s in &seeds {
        level[s as usize] = 0;
        order.push(s);
    }
    let mut next = 0;
    while next < order.len() {
        let v = order[next];
        next += 1;
        let lv = level[v as usize];
        for &u in g.nbrs(v) {
            if level[u as usize] == u32::MAX {
                level[u as usize] = lv + 1;
                order.push(u);
            }
        }
    }
    let max_ring = order.last().map_or(0, |&v| level[v as usize] as usize);

    // largest connected piece of each ring, all rings at once
    let reached: Vec<bool> = level.iter().map(|&l| l != u32::MAX).collect();
    let ring_pieces = g.pieces_of(&reached, |v, u| level[v as usize] == level[u as usize]);
    let mut largest = vec![0.0f64; max_ring + 1];
    for &p in &ring_pieces.list {
        let l = &mut largest[level[p as usize] as usize];
        *l = l.max(ring_pieces.size(p) as f64);
    }

    // the cutoff: past the peak (searched within three nucleus heights),
    // the first ring whose largest piece falls under Otsu's threshold
    let max_soma_rings = ((nuc.z_span * 3.0 / m.avg_edge) as i64).max(10);
    let search_end = ((max_soma_rings + 1) as usize).min(max_ring + 1);
    if search_end <= 1 {
        return Err(SomaError::NoRings);
    }
    let mut peak_ring = 1;
    for lv in 2..search_end {
        if largest[lv] > largest[peak_ring] {
            peak_ring = lv;
        }
    }
    let post_peak = &largest[peak_ring..];
    let count_thresh = if post_peak.len() > 1 {
        otsu_threshold(post_peak).0
    } else {
        0.0
    };
    let mut cutoff = (peak_ring..=max_ring)
        .find(|&lv| largest[lv] < count_thresh)
        .unwrap_or(max_ring);
    let min_cutoff = ((nuc.z_span / m.avg_edge) as i64).max(1) as usize;
    cutoff = cutoff.max(min_cutoff);

    let last = cutoff.min(max_ring);
    let mut bfs: Vec<u32> =
        order[..order.partition_point(|&v| level[v as usize] as usize <= last)].to_vec();
    bfs.sort_unstable();
    let initial = fit(&bfs, FitKind::Initial)
        .map_err(SomaError::Fit)?
        .expect("the first fit raises rather than fails");
    let mut cur = Current {
        fit: initial,
        index: 0,
        n_fits: 1,
        verts: Vec::new(),
        held_out: false,
        tube: Vec::new(),
    };
    assign_soma_verts(m, &g, &mut cur, &mut fit)?;
    prune(m, &g, &initial.ellipsoid, &mut cur, &mut fit)?;

    // organelle-only vertices leave the soma
    if m.organelle.iter().any(|&o| o) {
        // which faces each vertex is on: bit 1 organelle, bit 2 other
        let (org, other) = (1u8, 2u8);
        let sides: Vec<AtomicU8> = (0..nv).map(|_| AtomicU8::new(0)).collect();
        threads::install(|| {
            m.faces
                .par_iter()
                .zip(m.organelle.par_iter())
                .for_each(|(face, &o)| {
                    for &v in face {
                        sides[v as usize].fetch_or(if o { org } else { other }, Ordering::Relaxed);
                    }
                });
        });
        let kept: Vec<u32> = cur
            .verts
            .iter()
            .copied()
            .filter(|&v| sides[v as usize].load(Ordering::Relaxed) != org)
            .collect();
        if kept.len() < cur.verts.len() {
            cur.refit(kept, &mut fit)?;
        }
    }

    Ok(SomaFound {
        fit: cur.index,
        verts: cur.verts,
        cutoff,
        max_ring,
    })
}

/// Step 6 of `find_soma`: drop neurite tubes reaching far
/// out of the initial ellipsoid, erode the stubs at each neurite exit, keep
/// the largest connected piece, and refit if the vertex count changed.
fn prune<E>(
    m: &SomaMesh<'_>,
    g: &Graph,
    initial: &Ellipsoid,
    cur: &mut Current,
    fit: &mut impl FnMut(&[u32], FitKind) -> Result<Option<Fit>, E>,
) -> Result<(), SomaError<E>> {
    let nv = m.verts.len();
    if cur.verts.len() < 4 {
        return Ok(());
    }
    let mut soma_set = Set::from(nv, cur.verts.iter().copied());
    let outside_soma: Vec<u32> = cur
        .verts
        .iter()
        .copied()
        .filter(|&v| not_within(initial.rho2(&m.verts[v as usize]).sqrt(), 1.0))
        .collect();

    // 6a: tubes reaching far past the ellipsoid, on the side of a neurite
    let mut pruned = false;
    let mut cut = vec![false; nv];
    if !outside_soma.is_empty() {
        let non_soma: Vec<bool> = threads::install(|| {
            (0..nv as u32)
                .into_par_iter()
                .map(|v| g.is_key[v as usize] && !soma_set.has(v))
                .collect()
        });
        let ns = g.pieces_of(&non_soma, |_, _| true);
        // a neurite tree: the pieces Otsu calls large, or the only piece
        let neurite_tree = match ns.list.len() {
            0 => false,
            1 => true,
            _ => {
                let sizes: Vec<f64> = ns.list.iter().map(|&p| ns.size(p) as f64).collect();
                let thresh = otsu_threshold(&sizes).0;
                sizes.iter().any(|&s| s > thresh)
            }
        };
        if neurite_tree {
            let mut in_out = vec![false; nv];
            for &v in &outside_soma {
                in_out[v as usize] = true;
            }
            let out = g.pieces_of(&in_out, |_, _| true);
            // each piece's farthest reach past the ellipsoid
            let mut reach = vec![f64::NEG_INFINITY; nv];
            for &v in &outside_soma {
                let p = out.roots[v as usize] as usize;
                let x = initial.rho2(&m.verts[v as usize]).sqrt();
                reach[p] = nan_max([reach[p], x].into_iter());
            }
            let comps: Vec<u32> = out
                .list
                .iter()
                .copied()
                .filter(|&p| out.size(p) >= 20)
                .collect();
            let mut ext: Vec<f64> = comps.iter().map(|&p| reach[p as usize]).collect();
            // no outside piece of 20 vertices or more: nothing to prune
            if let Some(q1) = percentile25(&mut ext) {
                let ext_thresh = 2.0 * q1;
                let mut far = vec![false; nv];
                for &p in &comps {
                    if reach[p as usize] > ext_thresh {
                        pruned = true;
                        far[p as usize] = true;
                    }
                }
                if pruned {
                    for &v in &outside_soma {
                        if far[out.roots[v as usize] as usize] {
                            soma_set.remove(v);
                            cut[v as usize] = true;
                        }
                    }
                }
            }
        }
    }

    // 6b: erode each neurite exit inward until its ring widens.  An exit
    // is the soma's edge onto a tube left outside it or cut off by 6a:
    // walking in from the rim of a bulge or an organelle left outside
    // would cut a ring of the soma round it.
    let non_comp_soma: Vec<u32> = soma_set
        .sorted()
        .into_iter()
        .filter(|&v| !g.is_key[v as usize])
        .collect();
    let mut exit = Set::empty(nv);
    if pruned || cur.held_out {
        for v in soma_set.sorted() {
            if g.is_key[v as usize]
                && g.nbrs(v)
                    .iter()
                    .any(|&u| !soma_set.has(u) && (cur.tube[u as usize] || cut[u as usize]))
            {
                exit.insert(v);
            }
        }
    }
    if exit.len > 0 {
        let soma_main: Vec<bool> = (0..nv as u32)
            .map(|v| soma_set.has(v) && g.is_key[v as usize])
            .collect();
        let n_main = soma_main.iter().filter(|&&b| b).count();
        let mut seen = vec![false; nv];
        let clusters = g.pieces(&exit.sorted(), &exit.mask, &mut seen);
        let mut all_stub = Set::empty(nv);
        let mut global_visited = vec![false; nv];
        let mut visited = vec![false; nv];
        for cluster in clusters {
            let mut touched: Vec<u32> = cluster.clone();
            for &v in &cluster {
                visited[v as usize] = true;
            }
            let mut stub: Vec<u32> = cluster.clone();
            let mut ring = cluster.clone();
            let first = cluster.len();
            let mut next: Vec<u32> = Vec::new();
            for _ in 0..200 {
                next.clear();
                for &v in &ring {
                    for &nv in g.nbrs(v) {
                        let u = nv as usize;
                        if soma_main[u] && !visited[u] && !global_visited[u] {
                            visited[u] = true;
                            touched.push(nv);
                            next.push(nv);
                        }
                    }
                }
                // the ring doubling from the exit is the soma's body; not a
                // branch: dendrites leaving close together share a trunk,
                // which is not soma either
                if next.is_empty() || next.len() > 2 * first {
                    break;
                }
                stub.extend_from_slice(&next);
                std::mem::swap(&mut ring, &mut next);
            }
            if !next.is_empty() {
                for &v in &stub {
                    all_stub.insert(v);
                    global_visited[v as usize] = true;
                }
            }
            for v in touched {
                visited[v as usize] = false;
            }
        }
        if all_stub.len > 0 && (all_stub.len as f64) < n_main as f64 * 0.5 {
            for v in all_stub.sorted() {
                soma_set.remove(v);
            }
            for &v in &non_comp_soma {
                soma_set.insert(v);
            }
        }
    }

    // 6c: only the largest connected piece on the component stays
    let main_final: Vec<bool> = (0..nv as u32)
        .map(|v| soma_set.has(v) && g.is_key[v as usize])
        .collect();
    let fin = g.pieces_of(&main_final, |_, _| true);
    // the first largest, as a search from the lowest vertex meets them
    let mut largest: Option<u32> = None;
    for &p in &fin.list {
        if largest.is_none_or(|b| fin.size(p) > fin.size(b)) {
            largest = Some(p);
        }
    }
    let n_main = main_final.iter().filter(|&&b| b).count();
    if let Some(p) = largest
        && n_main > fin.size(p)
    {
        let mut chosen = vec![false; nv];
        chosen[p as usize] = true;
        soma_set = Set::from(
            nv,
            fin.members(&chosen).chain(non_comp_soma.iter().copied()),
        );
    }

    if soma_set.len != cur.verts.len() {
        cur.refit(soma_set.sorted(), fit)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_found_at_once_are_the_pieces_a_search_finds() {
        // random faces in clusters of 30 vertices, half the vertices inside
        let mut state = 0xbb67_ae85_84ca_a73bu64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % n) as u32
        };
        let nv = 30_000;
        let faces: Vec<[u32; 3]> = (0..40_000)
            .map(|_| {
                let base = 30 * next(1000);
                [base + next(30), base + next(30), base + next(30)]
            })
            .collect();
        let soma_fi: Vec<u32> = (0..faces.len() as u32).filter(|_| next(4) > 0).collect();
        let g = Graph::new(&faces, &soma_fi, nv);
        let inside: Vec<bool> = (0..nv).map(|v| g.is_key[v] && next(2) == 0).collect();
        let starts: Vec<u32> = (0..nv as u32).filter(|&v| inside[v as usize]).collect();
        let searched: Vec<Vec<u32>> = g
            .pieces(&starts, &inside, &mut vec![false; nv])
            .into_iter()
            .map(|mut c| {
                c.sort_unstable();
                c
            })
            .collect();
        let found = g.pieces_of(&inside, |_, _| true);
        let listed: Vec<Vec<u32>> = found
            .list
            .iter()
            .map(|&p| {
                (0..nv as u32)
                    .filter(|&v| found.roots[v as usize] == p)
                    .collect()
            })
            .collect();
        assert_eq!(listed, searched);
        assert!(searched.len() > 1000 && searched.iter().any(|c| c.len() > 5));
        for (&p, c) in found.list.iter().zip(&listed) {
            assert_eq!(found.size(p), c.len());
        }
    }

    /// A grid whose soma is a disc with a hole in it, a bulge of the surface
    /// left outside, and a strip of tube leaving the disc; with a neurite
    /// held out, the erosion walks in from the strip, never from the hole's
    /// rim nor the disc's edge.
    #[test]
    fn the_stub_erosion_starts_at_tubes_alone() {
        let n = 41u32;
        let mut verts = Vec::new();
        for i in 0..n {
            for j in 0..n {
                verts.push([i as f64 * 100.0 - 2000.0, j as f64 * 100.0 - 2000.0, 0.0]);
            }
        }
        let mut faces = Vec::new();
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                let (a, b) = (i * n + j, (i + 1) * n + j);
                faces.push([a, b, b + 1]);
                faces.push([a, b + 1, a + 1]);
            }
        }
        let nv = verts.len();
        let from = |v: usize, c: [f64; 2]| (verts[v][0] - c[0]).hypot(verts[v][1] - c[1]);
        let hole = |v: usize| from(v, [800.0, 0.0]) <= 300.0;
        let soma: Vec<u32> = (0..nv as u32)
            .filter(|&v| from(v as usize, [0.0, 0.0]) <= 1500.0 && !hole(v as usize))
            .collect();
        let labels = vec![0i64; faces.len()];
        let organelle = vec![false; faces.len()];
        let m = SomaMesh {
            faces: &faces,
            verts: &verts,
            labels: &labels,
            main: 0,
            organelle: &organelle,
            avg_edge: 100.0,
        };
        let all: Vec<u32> = (0..faces.len() as u32).collect();
        let g = Graph::new(&faces, &all, nv);
        // wide enough to hold the grid: nothing past it to prune
        let s = 1.0 / 5000.0;
        let wide = Ellipsoid {
            center: [0.0; 3],
            w: [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]],
        };
        let mut cur = Current {
            fit: Fit {
                ellipsoid: wide,
                axes_min: 5000.0,
            },
            index: 0,
            n_fits: 1,
            verts: soma.clone(),
            held_out: true,
            tube: (0..nv)
                .map(|v| verts[v][1] == 0.0 && verts[v][0] < -1500.0)
                .collect(),
        };
        prune(&m, &g, &wide, &mut cur, &mut |_, _| Ok::<_, ()>(None)).unwrap();
        let kept = Set::from(nv, cur.verts.iter().copied());
        let rim: Vec<u32> = soma
            .iter()
            .copied()
            .filter(|&v| g.nbrs(v).iter().any(|&u| hole(u as usize)))
            .collect();
        assert!(rim.len() > 10 && rim.iter().all(|&v| kept.has(v)));
        // the strip's stub went, and nothing else
        let lost: Vec<u32> = soma.iter().copied().filter(|&v| !kept.has(v)).collect();
        assert!(!lost.is_empty() && lost.iter().all(|&v| verts[v as usize][0] < -1000.0));
    }

    #[test]
    fn otsu_splits_two_clusters() {
        let (t, sep) = otsu_threshold(&[1.0, 2.0, 1.5, 10.0, 11.0, 12.0]);
        assert_eq!(t, (2.0 + 10.0) / 2.0);
        assert!(sep > 0.9);
        assert_eq!(otsu_threshold(&[3.0]), (3.0, 0.0));
        assert_eq!(otsu_threshold(&[3.0, 3.0]), (3.0, 0.0));
    }

    #[test]
    fn percentile_interpolates_like_numpy() {
        assert_eq!(percentile25(&mut [4.0, 1.0, 3.0, 2.0]), Some(1.75));
        assert_eq!(percentile25(&mut [5.0]), Some(5.0));
        assert_eq!(percentile25(&mut []), None);
        assert_eq!(percentile25(&mut [1.0, 2.0, 3.0, 4.0, 5.0]), Some(2.0));
    }
}
