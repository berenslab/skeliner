//! Edge and face incidence, with no manifoldness assumed.

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    n_vertices: usize,
    degenerate: Vec<bool>,
    edges: Vec<[u32; 2]>,
    edge_ptr: Vec<usize>,
    edge_faces: Vec<u32>,
    edge_cols: Vec<u8>,
    face_edges: Vec<[u32; 3]>,
    vertex_ptr: Vec<usize>,
    vertex_faces: Vec<u32>,
}

impl Topology {
    /// Build the incidence of `faces` over `n_vertices` vertices.
    ///
    /// Every index in `faces` must be below `n_vertices`; [`crate::Mesh`]
    /// guarantees it.
    pub fn new(faces: &[[u32; 3]], n_vertices: usize) -> Self {
        let degenerate: Vec<bool> = faces
            .iter()
            .map(|f| f[0] == f[1] || f[1] == f[2] || f[0] == f[2])
            .collect();

        // One entry per (edge, column pair, face); sorting groups each edge
        // and puts its faces in column-pair-then-face order.
        let mut entries: Vec<(u64, u8, u32)> = Vec::with_capacity(3 * faces.len());
        for (fi, f) in faces.iter().enumerate() {
            if degenerate[fi] {
                continue;
            }
            for (c, &(a, b)) in EDGE_COLS.iter().enumerate() {
                entries.push((edge_key(f[a], f[b]), c as u8, fi as u32));
            }
        }
        entries.sort_unstable();

        let mut edges = Vec::new();
        let mut edge_ptr = vec![0];
        let mut edge_faces = Vec::with_capacity(entries.len());
        let mut edge_cols = Vec::with_capacity(entries.len());
        let mut face_edges = vec![[NO_EDGE; 3]; faces.len()];
        for (i, &(key, c, fi)) in entries.iter().enumerate() {
            if i == 0 || entries[i - 1].0 != key {
                if i > 0 {
                    edge_ptr.push(edge_faces.len());
                }
                edges.push([(key >> 32) as u32, key as u32]);
            }
            face_edges[fi as usize][c as usize] = (edges.len() - 1) as u32;
            edge_faces.push(fi);
            edge_cols.push(c);
        }
        if !entries.is_empty() {
            edge_ptr.push(edge_faces.len());
        }

        // Vertex → faces over the faces that have edges, in face order.
        let mut vertex_ptr = vec![0usize; n_vertices + 1];
        for (fi, f) in faces.iter().enumerate() {
            if !degenerate[fi] {
                for &v in f {
                    vertex_ptr[v as usize + 1] += 1;
                }
            }
        }
        for v in 0..n_vertices {
            vertex_ptr[v + 1] += vertex_ptr[v];
        }
        let mut fill = vertex_ptr.clone();
        let mut vertex_faces = vec![0u32; vertex_ptr[n_vertices]];
        for (fi, f) in faces.iter().enumerate() {
            if !degenerate[fi] {
                for &v in f {
                    vertex_faces[fill[v as usize]] = fi as u32;
                    fill[v as usize] += 1;
                }
            }
        }

        Self {
            n_vertices,
            degenerate,
            edges,
            edge_ptr,
            edge_faces,
            edge_cols,
            face_edges,
            vertex_ptr,
            vertex_faces,
        }
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
    pub fn face_neighbors(&self, fi: usize) -> Vec<u32> {
        let Some(edges) = self.face_edges(fi) else {
            return Vec::new();
        };
        let mut out: Vec<u32> = edges
            .iter()
            .flat_map(|&e| self.edge_faces(e as usize).iter().copied())
            .filter(|&g| g as usize != fi)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Non-degenerate faces using vertex `v`, in face order.
    pub fn vertex_faces(&self, v: usize) -> &[u32] {
        &self.vertex_faces[self.vertex_ptr[v]..self.vertex_ptr[v + 1]]
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

fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    (u64::from(lo) << 32) | u64::from(hi)
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
    fn an_empty_mesh_has_no_edges() {
        let t = Topology::new(&[], 0);
        assert_eq!(t.n_edges(), 0);
        assert_eq!(t.first_seen_order(), Vec::<usize>::new());
    }
}
