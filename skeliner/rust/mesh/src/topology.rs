//! Edge and face incidence, with no manifoldness assumed.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

/// Marks a face slot with no edge: the face is degenerate.
pub const NO_EDGE: u32 = u32::MAX;

/// The three vertex-column pairs that make a face's edges, in the order
/// skeliner's Python edge maps visit them.
pub const EDGE_COLS: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];

/// Which faces share which edges.
///
/// A face with a repeated vertex is *degenerate*: it keeps its index but has
/// no edges, so it is adjacent to nothing.  Every other face has three
/// edges, and an edge may belong to any number of faces — one on a
/// boundary, two on a manifold sheet, three or more where sheets are fused.
///
/// The faces of an edge are listed by the vertex-column pair the edge
/// occupies in them, then by face index — the order in which a pass over
/// column pairs, each over all faces, first meets them.
#[derive(Debug, Clone)]
pub struct Topology {
    n_vertices: usize,
    degenerate: Vec<bool>,
    edges: Vec<[u32; 2]>,
    edge_ptr: Vec<usize>,
    edge_faces: Vec<u32>,
    edge_cols: Vec<u8>,
    face_edges: Vec<[u32; 3]>,
    /// Vertex → faces as CSR arrays, built when first asked for.
    vertex_index: OnceLock<(Vec<usize>, Vec<u32>)>,
}

impl Topology {
    /// Build the incidence of `faces` over `n_vertices` vertices, in
    /// parallel on [`threads`]' pool.
    ///
    /// Every index in `faces` must be below `n_vertices`; [`crate::Mesh`]
    /// guarantees it.
    pub fn new(faces: &[[u32; 3]], n_vertices: usize) -> Self {
        threads::install(|| Self::build(faces, n_vertices))
    }

    fn build(faces: &[[u32; 3]], n_vertices: usize) -> Self {
        let degenerate: Vec<bool> = faces
            .par_iter()
            .map(|f| f[0] == f[1] || f[1] == f[2] || f[0] == f[2])
            .collect();

        // One entry per (edge, column pair, face); sorting groups each edge
        // and puts its faces in column-pair-then-face order.  No two
        // entries are equal, so the order is the same however it is sorted.
        let mut entries: Vec<(u64, u8, u32)> = faces
            .par_iter()
            .enumerate()
            .filter(|&(fi, _)| !degenerate[fi])
            .flat_map_iter(|(fi, f)| {
                EDGE_COLS
                    .iter()
                    .enumerate()
                    .map(move |(c, &(a, b))| (edge_key(f[a], f[b]), c as u8, fi as u32))
            })
            .collect();
        entries.par_sort_unstable();

        // An edge's entries start where the key changes.
        let mut edge_ptr: Vec<usize> = (0..entries.len())
            .into_par_iter()
            .filter(|&i| i == 0 || entries[i - 1].0 != entries[i].0)
            .collect();
        let edges: Vec<[u32; 2]> = edge_ptr
            .par_iter()
            .map(|&i| {
                let key = entries[i].0;
                [(key >> 32) as u32, key as u32]
            })
            .collect();
        edge_ptr.push(entries.len());
        let edge_faces: Vec<u32> = entries.par_iter().map(|e| e.2).collect();
        let edge_cols: Vec<u8> = entries.par_iter().map(|e| e.1).collect();
        // each (face, column pair) slot is written by its one entry
        let slots: Vec<AtomicU32> = (0..3 * faces.len())
            .into_par_iter()
            .map(|_| AtomicU32::new(NO_EDGE))
            .collect();
        edge_ptr.par_windows(2).enumerate().for_each(|(e, w)| {
            for &(_, c, fi) in &entries[w[0]..w[1]] {
                slots[3 * fi as usize + c as usize].store(e as u32, Ordering::Relaxed);
            }
        });
        let face_edges: Vec<[u32; 3]> = slots
            .par_chunks(3)
            .map(|s| [0, 1, 2].map(|c| s[c].load(Ordering::Relaxed)))
            .collect();

        Self {
            n_vertices,
            degenerate,
            edges,
            edge_ptr,
            edge_faces,
            edge_cols,
            face_edges,
            vertex_index: OnceLock::new(),
        }
    }

    /// Vertex → faces over the faces that have edges, in face order.  A
    /// face's vertices are the ends of its first two edges, three different
    /// ones.  Built by one thread: the first call may come from inside a
    /// parallel loop, and a pool thread waiting in a parallel build could be
    /// handed that loop's next call, which would wait on the build itself.
    fn vertex_index(&self) -> &(Vec<usize>, Vec<u32>) {
        self.vertex_index.get_or_init(|| {
            let n = self.n_vertices;
            let vertices_of = |fi: usize| {
                let [a, b, _] = self.face_edges[fi];
                let ([v0, v1], [v2, v3]) = (self.edges[a as usize], self.edges[b as usize]);
                [v0, v1, if v2 == v0 || v2 == v1 { v3 } else { v2 }]
            };
            let mut ptr = vec![0usize; n + 1];
            for fi in (0..self.n_faces()).filter(|&fi| !self.degenerate[fi]) {
                for v in vertices_of(fi) {
                    ptr[v as usize + 1] += 1;
                }
            }
            for v in 0..n {
                ptr[v + 1] += ptr[v];
            }
            let mut fill = ptr.clone();
            let mut faces = vec![0u32; ptr[n]];
            for fi in (0..self.n_faces()).filter(|&fi| !self.degenerate[fi]) {
                for v in vertices_of(fi) {
                    faces[fill[v as usize]] = fi as u32;
                    fill[v as usize] += 1;
                }
            }
            (ptr, faces)
        })
    }

    /// Number of faces, degenerate ones included.
    pub fn n_faces(&self) -> usize {
        self.degenerate.len()
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// Number of distinct edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Whether face `fi` has a repeated vertex.
    pub fn is_degenerate(&self, fi: usize) -> bool {
        self.degenerate[fi]
    }

    /// Distinct edges as `[low, high]` vertex pairs, in ascending order.
    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    /// Id of the edge between vertices `a` and `b`, if any face has it.
    pub fn edge_id(&self, a: u32, b: u32) -> Option<usize> {
        let key = [a.min(b), a.max(b)];
        self.edges.binary_search(&key).ok()
    }

    /// Faces sharing edge `e`, in column-pair-then-face order.
    pub fn edge_faces(&self, e: usize) -> &[u32] {
        &self.edge_faces[self.edge_ptr[e]..self.edge_ptr[e + 1]]
    }

    /// The vertex-column pair edge `e` occupies in each of its faces.
    pub fn edge_cols(&self, e: usize) -> &[u8] {
        &self.edge_cols[self.edge_ptr[e]..self.edge_ptr[e + 1]]
    }

    /// Edge ids of face `fi` per vertex-column pair, or `None` if degenerate.
    pub fn face_edges(&self, fi: usize) -> Option<[u32; 3]> {
        (!self.degenerate[fi]).then(|| self.face_edges[fi])
    }

    /// Faces sharing an edge with face `fi`, ascending, `fi` itself excluded.
    pub fn face_neighbors(&self, fi: usize) -> Neighbors {
        let mut out = Neighbors::default();
        let Some(edges) = self.face_edges(fi) else {
            return out;
        };
        for e in edges {
            for &g in self.edge_faces(e as usize) {
                if g as usize != fi {
                    out.push(g);
                }
            }
        }
        out.sort_dedup();
        out
    }

    /// Non-degenerate faces using vertex `v`, in face order.  The first call
    /// builds the lists for every vertex.
    pub fn vertex_faces(&self, v: usize) -> &[u32] {
        let (ptr, faces) = self.vertex_index();
        &faces[ptr[v]..ptr[v + 1]]
    }

    /// Edge ids in the order a pass over column pairs, each over all faces,
    /// first meets them.
    pub fn first_seen_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.n_edges()).collect();
        order.sort_unstable_by_key(|&e| {
            let lo = self.edge_ptr[e];
            (self.edge_cols[lo], self.edge_faces[lo])
        });
        order
    }

    /// CSR arrays of edge → faces: `(offsets, faces)`.
    pub fn edge_faces_csr(&self) -> (&[usize], &[u32]) {
        (&self.edge_ptr, &self.edge_faces)
    }
}

/// A face's neighbours ([`Topology::face_neighbors`]): held inline, with no
/// allocation, unless fused edges bring more than fit.
#[derive(Debug, Clone, Default)]
pub struct Neighbors {
    inline: [u32; INLINE],
    len: usize,
    spilled: Vec<u32>,
}

const INLINE: usize = 9;

impl Neighbors {
    fn push(&mut self, g: u32) {
        if self.spilled.is_empty() && self.len < INLINE {
            self.inline[self.len] = g;
            self.len += 1;
        } else {
            if self.spilled.is_empty() {
                self.spilled.extend_from_slice(&self.inline[..self.len]);
            }
            self.spilled.push(g);
        }
    }

    fn sort_dedup(&mut self) {
        if self.spilled.is_empty() {
            let held = &mut self.inline[..self.len];
            held.sort_unstable();
            let mut kept = 0;
            for i in 0..held.len() {
                if i == 0 || held[i] != held[kept - 1] {
                    held[kept] = held[i];
                    kept += 1;
                }
            }
            self.len = kept;
        } else {
            self.spilled.sort_unstable();
            self.spilled.dedup();
        }
    }
}

impl std::ops::Deref for Neighbors {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        if self.spilled.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spilled
        }
    }
}

impl IntoIterator for Neighbors {
    type Item = u32;
    type IntoIter = NeighborsIter;

    fn into_iter(self) -> NeighborsIter {
        NeighborsIter { of: self, at: 0 }
    }
}

/// The faces of a [`Neighbors`], in order.
#[derive(Debug, Clone)]
pub struct NeighborsIter {
    of: Neighbors,
    at: usize,
}

impl Iterator for NeighborsIter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let g = self.of.get(self.at).copied();
        self.at += 1;
        g
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.of.len().saturating_sub(self.at);
        (left, Some(left))
    }
}

impl ExactSizeIterator for NeighborsIter {}

fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    (u64::from(lo) << 32) | u64::from(hi)
}

/// Every distinct edge of `faces` as `[low, high]`, ordered by `high` and
/// then `low`: trimesh's `edges_unique`, which keeps a degenerate face's
/// edges too — `[v, v]` for its repeated vertex.
///
/// Every index in `faces` must be below `n_vertices`.
pub fn unique_edges(faces: &[[u32; 3]], n_vertices: usize) -> Vec<[u32; 2]> {
    // Bucket each face edge's low end by its high end, then sort and
    // deduplicate each bucket: a few entries each, so no global sort.
    let ends = |f: &[u32; 3]| [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])];
    let mut start = vec![0usize; n_vertices + 1];
    for f in faces {
        for (a, b) in ends(f) {
            start[a.max(b) as usize + 1] += 1;
        }
    }
    for v in 0..n_vertices {
        start[v + 1] += start[v];
    }
    let mut fill = start.clone();
    let mut low = vec![0u32; start[n_vertices]];
    for f in faces {
        for (a, b) in ends(f) {
            let hi = a.max(b) as usize;
            low[fill[hi]] = a.min(b);
            fill[hi] += 1;
        }
    }
    let mut edges = Vec::with_capacity(low.len() / 2);
    for hi in 0..n_vertices {
        let bucket = &mut low[start[hi]..start[hi + 1]];
        bucket.sort_unstable();
        for (i, &lo) in bucket.iter().enumerate() {
            if i == 0 || bucket[i - 1] != lo {
                edges.push([lo, hi as u32]);
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three triangles fanned around edge (0, 1), a fourth hanging off
    /// (1, 2), and a degenerate face.
    fn fan() -> (Vec<[u32; 3]>, usize) {
        let faces = vec![
            [0, 1, 2], // 0
            [1, 0, 3], // 1
            [0, 1, 4], // 2: third face on (0, 1)
            [2, 1, 5], // 3
            [5, 5, 2], // 4: degenerate
        ];
        (faces, 6)
    }

    #[test]
    fn unique_edges_come_by_high_end_then_low_with_degenerate_ones_kept() {
        let (faces, n) = fan();
        assert_eq!(
            unique_edges(&faces, n),
            [
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 3],
                [1, 3],
                [0, 4],
                [1, 4],
                [1, 5],
                [2, 5],
                [5, 5],
            ]
        );
        assert!(unique_edges(&[], 3).is_empty());
    }

    #[test]
    fn edges_are_distinct_and_sorted() {
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        let e = t.edges();
        assert!(e.windows(2).all(|w| w[0] < w[1]));
        // 0-1 0-2 0-3 0-4 1-2 1-3 1-4 1-5 2-5
        assert_eq!(t.n_edges(), 9);
    }

    #[test]
    fn a_fused_edge_lists_all_its_faces() {
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        let e = t.edge_id(1, 0).unwrap();
        assert_eq!(t.edge_faces(e), &[0, 1, 2]);
        assert_eq!(t.edge_faces(t.edge_id(0, 2).unwrap()), &[0]);
    }

    #[test]
    fn faces_on_an_edge_come_in_column_pair_order() {
        // Edge (1, 2) is column pair (1, 2) in face 0 and (0, 1) in face 3,
        // so face 3 comes first.
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        let e = t.edge_id(1, 2).unwrap();
        assert_eq!(t.edge_faces(e), &[3, 0]);
        assert_eq!(t.edge_cols(e), &[0, 1]);
    }

    #[test]
    fn a_degenerate_face_has_no_edges() {
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        assert!(t.is_degenerate(4));
        assert_eq!(t.face_edges(4), None);
        assert_eq!(t.edge_id(5, 5), None);
        // its (2, 5) edge exists only through face 3
        assert_eq!(t.edge_faces(t.edge_id(2, 5).unwrap()), &[3]);
    }

    #[test]
    fn face_edges_point_back_at_the_face() {
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        for (fi, face) in faces.iter().enumerate() {
            let Some(es) = t.face_edges(fi) else { continue };
            for (c, &e) in es.iter().enumerate() {
                let (a, b) = EDGE_COLS[c];
                assert_eq!(t.edge_id(face[a], face[b]), Some(e as usize));
                assert!(t.edge_faces(e as usize).contains(&(fi as u32)));
            }
        }
    }

    #[test]
    fn vertex_faces_skip_degenerate_faces() {
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        assert_eq!(t.vertex_faces(5), &[3]);
        assert_eq!(t.vertex_faces(1), &[0, 1, 2, 3]);
    }

    #[test]
    fn first_seen_order_follows_column_pairs_then_faces() {
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        let order: Vec<[u32; 2]> = t
            .first_seen_order()
            .into_iter()
            .map(|e| t.edges()[e])
            .collect();
        // Columns (0, 1) over faces 0-3 meet 0-1 then 1-2 (face 3); columns
        // (1, 2) meet 0-3, 1-4, 1-5; columns (0, 2) meet 0-2, 1-3, 0-4, 2-5.
        let expected = vec![
            [0, 1],
            [1, 2],
            [0, 3],
            [1, 4],
            [1, 5],
            [0, 2],
            [1, 3],
            [0, 4],
            [2, 5],
        ];
        assert_eq!(order, expected);
    }

    #[test]
    fn a_large_mesh_is_built_as_one_face_at_a_time_builds_it() {
        // random faces over few vertices: repeated vertices, fused edges
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % n) as u32
        };
        let n = 3000;
        let faces: Vec<[u32; 3]> = (0..60_000).map(|_| [next(n), next(n), next(n)]).collect();
        let t = Topology::new(&faces, n as usize);

        let mut by_edge: std::collections::BTreeMap<[u32; 2], Vec<(u8, u32)>> =
            std::collections::BTreeMap::new();
        let mut by_vertex = vec![Vec::new(); n as usize];
        for (fi, f) in faces.iter().enumerate() {
            if f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
                assert!(t.is_degenerate(fi) && t.face_edges(fi).is_none());
                continue;
            }
            for (c, &(a, b)) in EDGE_COLS.iter().enumerate() {
                let key = [f[a].min(f[b]), f[a].max(f[b])];
                by_edge.entry(key).or_default().push((c as u8, fi as u32));
            }
            for &v in f {
                by_vertex[v as usize].push(fi as u32);
            }
        }
        assert!((0..faces.len()).any(|f| t.is_degenerate(f)));
        assert!(by_edge.values().any(|l| l.len() >= 3), "no fused edge");
        let keys: Vec<[u32; 2]> = by_edge.keys().copied().collect();
        assert_eq!(t.edges(), keys.as_slice());
        for (e, list) in by_edge.values_mut().enumerate() {
            list.sort_unstable();
            let cols: Vec<u8> = list.iter().map(|x| x.0).collect();
            let fs: Vec<u32> = list.iter().map(|x| x.1).collect();
            assert_eq!((t.edge_cols(e), t.edge_faces(e)), (&cols[..], &fs[..]));
            for (&c, &f) in cols.iter().zip(&fs) {
                assert_eq!(t.face_edges(f as usize).unwrap()[c as usize], e as u32);
            }
        }
        for (v, fs) in by_vertex.iter().enumerate() {
            assert_eq!(t.vertex_faces(v), fs.as_slice());
        }
    }

    #[test]
    fn neighbours_come_ascending_once_each_however_many_there_are() {
        let (faces, n) = fan();
        let t = Topology::new(&faces, n);
        assert_eq!(&*t.face_neighbors(0), &[1, 2, 3]);
        assert_eq!(t.face_neighbors(4).len(), 0);
        // twelve faces fused on one edge: more than fit inline
        let mut faces = vec![[0, 1, 2]];
        faces.extend((3..14).map(|v| [1, 0, v]));
        faces.push([2, 1, 14]);
        let t = Topology::new(&faces, 15);
        let all: Vec<u32> = (1..13).collect();
        assert_eq!(t.face_neighbors(0).into_iter().collect::<Vec<_>>(), all);
        assert_eq!(&*t.face_neighbors(12), &[0]);
        assert_eq!(t.face_neighbors(3).len(), 11);
    }

    #[test]
    fn the_vertex_lists_may_first_be_asked_for_from_a_parallel_loop() {
        // a parallel loop where one item in a thousand wants a fresh
        // topology's lists: threads done with their items take parts of the
        // lists' build, and the building thread takes up the loop's items
        // meanwhile, some of which want the lists too
        let faces: Vec<[u32; 3]> = (0..400_000u32).map(|i| [i, i + 1, i + 2]).collect();
        for _ in 0..10 {
            let t = Topology::new(&faces, 400_002);
            let total: u64 = threads::install(|| {
                (0..200_000u64)
                    .into_par_iter()
                    .map(|v| {
                        if v % 1000 == 7 {
                            t.vertex_faces(v as usize).len() as u64
                        } else {
                            (0..v % 64).fold(v, |a, k| a.rotate_left(5) ^ k) & 1
                        }
                    })
                    .sum()
            });
            assert!(total > 0);
        }
    }

    #[test]
    fn an_empty_mesh_has_no_edges() {
        let t = Topology::new(&[], 0);
        assert_eq!(t.n_edges(), 0);
        assert_eq!(t.first_seen_order(), Vec::<usize>::new());
    }
}
