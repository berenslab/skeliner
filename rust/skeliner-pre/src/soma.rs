//! `skeliner.pre.find_soma_via_ring_cutoff` and `_assign_soma_verts`.
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

use meshcore::geometry::{numpy_order, numpy_sum, row_times_matrix};

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
    /// No outside component to take the percentile of: numpy's
    /// percentile of an empty array.
    NoExtents,
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
    let mut main_verts = g.keys.clone();
    main_verts.sort_unstable();
    let mut soma_set = Set::from(
        nv,
        main_verts
            .iter()
            .copied()
            .filter(|&v| e.contains(&m.verts[v as usize])),
    );

    let edge_in_body = m.avg_edge / soma.axes_min;
    let dilation_limit = 1.0 + 3.0 * edge_in_body;
    if soma_set.len > 0 {
        let mut in_frontier = vec![false; nv];
        loop {
            let mut frontier = Vec::new();
            for v in soma_set.sorted() {
                for &nv in g.nbrs(v) {
                    if !soma_set.has(nv) && !in_frontier[nv as usize] {
                        in_frontier[nv as usize] = true;
                        frontier.push(nv);
                    }
                }
            }
            for &u in &frontier {
                in_frontier[u as usize] = false;
            }
            if frontier.is_empty() {
                break;
            }
            let accept: Vec<u32> = frontier
                .into_iter()
                .filter(|&u| e.rho2(&m.verts[u as usize]).sqrt() < dilation_limit)
                .collect();
            if accept.is_empty() {
                break;
            }
            for u in accept {
                soma_set.insert(u);
            }
        }
    }

    // pockets: pieces of the component outside the soma, absorbed unless
    // Otsu on their sizes calls them neurites
    for _ in 0..10 {
        let outside: Vec<bool> = (0..nv as u32)
            .map(|v| g.is_key[v as usize] && !soma_set.has(v))
            .collect();
        let starts: Vec<u32> = main_verts
            .iter()
            .copied()
            .filter(|&v| outside[v as usize])
            .collect();
        let mut seen = vec![false; nv];
        let mut trapped: Vec<Vec<u32>> = Vec::new();
        for comp in g.pieces(&starts, &outside, &mut seen) {
            let in_comp = |v: u32| outside[v as usize];
            let closed = comp
                .iter()
                .all(|&v| g.nbrs(v).iter().all(|&nv| soma_set.has(nv) || in_comp(nv)));
            if closed && comp.len() < soma_set.len {
                trapped.push(comp);
            }
        }
        if trapped.is_empty() {
            break;
        }
        let size_thresh = if trapped.len() >= 2 {
            let sizes: Vec<f64> = trapped.iter().map(|c| c.len() as f64).collect();
            otsu_threshold(&sizes).0
        } else {
            0.0
        };
        let mut absorbed = 0;
        for comp in trapped {
            if comp.len() as f64 <= size_thresh {
                absorbed += comp.len();
                for v in comp {
                    soma_set.insert(v);
                }
            }
        }
        if absorbed == 0 {
            break;
        }
    }

    // vertices of other components inside the ellipsoid
    for v in 0..nv as u32 {
        if !g.is_key[v as usize] && e.contains(&m.verts[v as usize]) {
            soma_set.insert(v);
        }
    }
    cur.refit(soma_set.sorted(), fit)
}

/// `find_soma_via_ring_cutoff` from the nucleus on: everything but the
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
    let soma_fi: Vec<u32> = (0..nf as u32)
        .filter(|&f| m.labels[f as usize] == soma_ci)
        .collect();
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
        let mut best: Option<(u32, f64)> = None;
        for &v in &g.keys {
            let d = dist(v);
            if best.is_none_or(|(_, b)| d < b) {
                best = Some((v, d));
            }
        }
        seeds.extend(best.map(|(v, _)| v));
    }
    let mut level = vec![usize::MAX; nv];
    let mut rings: Vec<Vec<u32>> = vec![Vec::new()];
    let mut queue = VecDeque::new();
    for &s in &seeds {
        level[s as usize] = 0;
        queue.push_back(s);
        rings[0].push(s);
    }
    while let Some(v) = queue.pop_front() {
        let lv = level[v as usize];
        for &nv in g.nbrs(v) {
            if level[nv as usize] == usize::MAX {
                level[nv as usize] = lv + 1;
                queue.push_back(nv);
                if rings.len() <= lv + 1 {
                    rings.push(Vec::new());
                }
                rings[lv + 1].push(nv);
            }
        }
    }
    let max_ring = rings.len() - 1;

    // largest connected piece of each ring
    let mut in_ring = vec![false; nv];
    let mut seen = vec![false; nv];
    let largest: Vec<f64> = rings
        .iter()
        .map(|ring| {
            for &v in ring {
                in_ring[v as usize] = true;
            }
            let size = g
                .pieces(ring, &in_ring, &mut seen)
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0);
            for &v in ring {
                in_ring[v as usize] = false;
                seen[v as usize] = false;
            }
            size as f64
        })
        .collect();

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

    let mut bfs: Vec<u32> = rings[..=cutoff.min(max_ring)].concat();
    bfs.sort_unstable();
    let initial = fit(&bfs, FitKind::Initial)
        .map_err(SomaError::Fit)?
        .expect("the first fit raises rather than fails");
    let mut cur = Current {
        fit: initial,
        index: 0,
        n_fits: 1,
        verts: Vec::new(),
    };
    assign_soma_verts(m, &g, &mut cur, &mut fit)?;
    prune(m, &g, &initial.ellipsoid, &mut cur, &mut fit)?;

    // organelle-only vertices leave the soma
    if m.organelle.iter().any(|&o| o) {
        let mut on_org = vec![false; nv];
        let mut on_other = vec![false; nv];
        for (f, face) in m.faces.iter().enumerate() {
            let side = if m.organelle[f] {
                &mut on_org
            } else {
                &mut on_other
            };
            for &v in face {
                side[v as usize] = true;
            }
        }
        let kept: Vec<u32> = cur
            .verts
            .iter()
            .copied()
            .filter(|&v| !(on_org[v as usize] && !on_other[v as usize]))
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

/// Step 6 of `find_soma_via_ring_cutoff`: drop neurite tubes reaching far
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
    if !outside_soma.is_empty() {
        let non_soma: Vec<bool> = (0..nv as u32)
            .map(|v| g.is_key[v as usize] && !soma_set.has(v))
            .collect();
        let starts: Vec<u32> = (0..nv as u32).filter(|&v| non_soma[v as usize]).collect();
        let mut seen = vec![false; nv];
        let ns = g.pieces(&starts, &non_soma, &mut seen);
        let mut neurite_tree = Set::empty(nv);
        if ns.len() >= 2 {
            let sizes: Vec<f64> = ns.iter().map(|c| c.len() as f64).collect();
            let thresh = otsu_threshold(&sizes).0;
            for c in ns.iter().filter(|c| c.len() as f64 > thresh) {
                for &v in c {
                    neurite_tree.insert(v);
                }
            }
        } else if let Some(c) = ns.first() {
            for &v in c {
                neurite_tree.insert(v);
            }
        }
        if neurite_tree.len > 0 {
            let mut in_out = vec![false; nv];
            for &v in &outside_soma {
                in_out[v as usize] = true;
            }
            let mut seen = vec![false; nv];
            let comps: Vec<(Vec<u32>, f64)> = g
                .pieces(&outside_soma, &in_out, &mut seen)
                .into_iter()
                .filter(|c| c.len() >= 20)
                .map(|c| {
                    let ext = nan_max(c.iter().map(|&v| initial.rho2(&m.verts[v as usize]).sqrt()));
                    (c, ext)
                })
                .collect();
            let mut ext: Vec<f64> = comps.iter().map(|(_, e)| *e).collect();
            let ext_thresh = 2.0 * percentile25(&mut ext).ok_or(SomaError::NoExtents)?;
            for (c, e) in &comps {
                if *e > ext_thresh {
                    pruned = true;
                    for &v in c {
                        soma_set.remove(v);
                    }
                }
            }
        }
    }

    // 6b: erode each neurite exit inward until its ring widens or branches
    let non_comp_soma: Vec<u32> = soma_set
        .sorted()
        .into_iter()
        .filter(|&v| !g.is_key[v as usize])
        .collect();
    let non_soma: Vec<bool> = (0..nv as u32)
        .map(|v| g.is_key[v as usize] && !soma_set.has(v))
        .collect();
    let mut exit = Set::empty(nv);
    if pruned && non_soma.iter().any(|&b| b) {
        for v in soma_set.sorted() {
            if g.is_key[v as usize] && g.nbrs(v).iter().any(|&nv| non_soma[nv as usize]) {
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
        let mut in_ring = vec![false; nv];
        let mut seen_ring = vec![false; nv];
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
                if next.is_empty() || next.len() > 2 * first {
                    break;
                }
                // stop at a real branch: two or more pieces each a quarter
                for &v in &next {
                    in_ring[v as usize] = true;
                }
                let quarter = next.len() as f64 * 0.25;
                let big = g
                    .pieces(&next, &in_ring, &mut seen_ring)
                    .iter()
                    .filter(|c| c.len() as f64 >= quarter)
                    .count();
                for &v in &next {
                    in_ring[v as usize] = false;
                    seen_ring[v as usize] = false;
                }
                if big >= 2 {
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
    let starts: Vec<u32> = (0..nv as u32).filter(|&v| main_final[v as usize]).collect();
    let mut seen = vec![false; nv];
    let mut largest: Vec<u32> = Vec::new();
    for c in g.pieces(&starts, &main_final, &mut seen) {
        if c.len() > largest.len() {
            largest = c;
        }
    }
    if starts.len() > largest.len() {
        soma_set = Set::from(nv, largest.into_iter().chain(non_comp_soma.iter().copied()));
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
