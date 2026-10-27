//! Coincident vertices: vertices at the same position, which a mesh stored
//! in chunks keeps once per chunk along every seam.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use rayon::prelude::*;

/// A multiply-rotate hash: one pass over a million vertices in milliseconds,
/// where std's SipHash takes several times longer.
#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    // MurmurHash3's finaliser: coordinates on a voxel grid end in zero bits,
    // and without it so would every hash, which the table indexes by
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

fn fast_map<K, V>(capacity: usize) -> FastMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, Default::default())
}

/// A position's three coordinates as bits, −0.0 read as 0.0.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Position([u64; 3]);

impl Hash for Position {
    fn hash<H: Hasher>(&self, h: &mut H) {
        for &x in &self.0 {
            h.write_u64(x);
        }
    }
}

/// Two or three vertex ids in two words.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Ids(u64, u64);

impl Ids {
    fn pair(a: u32, b: u32) -> Self {
        Ids(u64::from(a) << 32 | u64::from(b), 0)
    }

    /// A face's vertices in any order.
    fn face(mut f: [u32; 3]) -> Self {
        f.sort_unstable();
        Ids(u64::from(f[0]) << 32 | u64::from(f[1]), u64::from(f[2]))
    }
}

impl Hash for Ids {
    fn hash<H: Hasher>(&self, h: &mut H) {
        h.write_u64(self.0);
        h.write_u64(self.1);
    }
}

/// `v` as a [`Position`]; None for one with a NaN, which equals nothing.
fn position(v: &[f64; 3]) -> Option<Position> {
    // adding 0.0 turns −0.0 into 0.0 and leaves every other value
    (!v.iter().any(|x| x.is_nan())).then(|| Position(v.map(|x| (x + 0.0).to_bits())))
}

/// For `ids`, ascending: each at the position of one before it, with the
/// first there, `[copy, first]`.
fn copies_among(vertices: &[[f64; 3]], ids: &[u32]) -> Vec<[u32; 2]> {
    let mut first: FastMap<Position, u32> = fast_map(ids.len());
    ids.iter()
        .filter_map(|&i| {
            let f = *first.entry(position(&vertices[i as usize])?).or_insert(i);
            (f != i).then_some([i, f])
        })
        .collect()
}

/// Each vertex at the position of one before it, with the first vertex
/// there: `[copy, first]`. Positions are equal as floats are: −0.0 is 0.0,
/// and a NaN equals nothing. From [`threads::MIN_PARALLEL`] vertices on,
/// they are dealt into shards by their positions' hashes, each shard's
/// table built on a thread of its own.
fn copies(vertices: &[[f64; 3]]) -> Vec<[u32; 2]> {
    const SHARDS: usize = 256;
    if vertices.len() < threads::MIN_PARALLEL {
        let all: Vec<u32> = (0..vertices.len() as u32).collect();
        return copies_among(vertices, &all);
    }
    let shard_of = threads::map(vertices, |v| {
        position(v).map_or(u16::MAX, |p| {
            let mut h = FastHasher::default();
            p.hash(&mut h);
            (h.finish() >> 56) as u16
        })
    });
    // each shard's vertices, ascending
    let mut start = vec![0; SHARDS + 1];
    for &s in &shard_of {
        if s != u16::MAX {
            start[s as usize + 1] += 1;
        }
    }
    for s in 0..SHARDS {
        start[s + 1] += start[s];
    }
    let mut order = vec![0; start[SHARDS]];
    let mut next = start.clone();
    for (i, &s) in shard_of.iter().enumerate() {
        if s != u16::MAX {
            order[next[s as usize]] = i as u32;
            next[s as usize] += 1;
        }
    }
    threads::install(|| {
        (0..SHARDS)
            .into_par_iter()
            .flat_map_iter(|s| copies_among(vertices, &order[start[s]..start[s + 1]]))
            .collect()
    })
}

/// The smallest vertex of `v`'s set.
fn find(low: &mut [u32], v: u32) -> u32 {
    let mut root = v;
    while low[root as usize] != root {
        root = low[root as usize];
    }
    let mut x = v;
    while low[x as usize] != root {
        x = std::mem::replace(&mut low[x as usize], root);
    }
    root
}

/// Join the sets of `a` and `b`: whether they were two.
fn join(low: &mut [u32], a: u32, b: u32) -> bool {
    let (a, b) = (find(low, a), find(low, b));
    low[a.max(b) as usize] = a.min(b);
    a != b
}

/// Each vertex's first vertex at its position that a seam joins it to,
/// itself when none comes before it; None when no vertex is joined to
/// another. Two vertices at one position are joined when an edge of each
/// ends at one position too, at one vertex or at two more joined so: a seam
/// is a chain of edges both sides of it repeat. Vertices that only touch
/// stay apart — `remove_fusions` cuts a vertex shared by branches that
/// share no edge into such copies, and joining them would fuse the branches
/// again. Edges are those of faces of three vertices; a face naming one
/// vertex twice joins nothing.
fn seam_joined(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Option<Vec<u32>> {
    let copies = copies(vertices);
    if copies.is_empty() {
        return None;
    }
    let mut first: Vec<u32> = (0..vertices.len() as u32).collect();
    let mut shared = vec![false; vertices.len()];
    for &[v, f] in &copies {
        first[v as usize] = f;
        shared[v as usize] = true;
        shared[f as usize] = true;
    }
    // a face's edges with an end at a shared position, keyed by the two
    // positions, from the lower to the higher
    let seam_edges = |f: &[u32; 3]| {
        let whole = f[0] != f[1] && f[1] != f[2] && f[2] != f[0];
        [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])].map(|(a, b)| {
            let (pa, pb) = (first[a as usize], first[b as usize]);
            if !whole || pa == pb || !(shared[a as usize] || shared[b as usize]) {
                None
            } else if pa < pb {
                Some((Ids::pair(pa, pb), [a, b]))
            } else {
                Some((Ids::pair(pb, pa), [b, a]))
            }
        })
    };
    let edges: Vec<(Ids, [u32; 2])> = if faces.len() < threads::MIN_PARALLEL {
        faces.iter().flat_map(seam_edges).flatten().collect()
    } else {
        threads::install(|| {
            faces
                .par_iter()
                .flat_map_iter(|f| seam_edges(f).into_iter().flatten())
                .collect()
        })
    };
    // by the two positions: the first edge between them
    let mut between: FastMap<Ids, [u32; 2]> = fast_map(edges.len());
    let mut low: Vec<u32> = (0..vertices.len() as u32).collect();
    let mut any = false;
    for (key, ends) in edges {
        let there = *between.entry(key).or_insert(ends);
        any |= join(&mut low, ends[0], there[0]);
        any |= join(&mut low, ends[1], there[1]);
    }
    any.then(|| (0..low.len() as u32).map(|v| find(&mut low, v)).collect())
}

/// What [`merge_coincident`] made of a mesh's faces.
#[derive(Debug, PartialEq, Eq)]
pub struct Merged {
    /// The faces, each vertex replaced by the first a seam joins it to, and
    /// each face the merge made a copy of another collapsed to `[0, 0, 0]`.
    pub faces: Vec<[u32; 3]>,
    /// The vertices merged into an earlier one.
    pub copies: usize,
    /// The faces collapsed.
    pub collapsed: usize,
}

/// *faces* with every vertex replaced by the first vertex at its position
/// that a seam joins it to ([`seam_joined`]). Chunks overlap along a seam,
/// so merging their copies makes copies of the seam's faces too: of the
/// faces with the same three vertices, those the merge left alone all stay
/// (a duplicate the mesh had already is the fusions' evidence), and when it
/// changed every one of them the first stays; the rest collapse to
/// `[0, 0, 0]`. The vertices and the faces keep their places. None when no
/// vertex is merged. Every index in `faces` must name one of `vertices`.
pub fn merge_coincident(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Option<Merged> {
    let joined = seam_joined(vertices, faces)?;
    let mut kept = vec![false; joined.len()];
    let mut copies = 0;
    for (v, &j) in joined.iter().enumerate() {
        if j as usize != v {
            kept[j as usize] = true;
            copies += 1;
        }
    }
    let mut merged = threads::map(faces, |f| f.map(|v| joined[v as usize]));
    // by the vertices of a face the merge changed: the first such face, and
    // whether a face it left alone has them too (one naming a vertex that
    // others merged into)
    let mut groups: FastMap<Ids, (u32, bool)> = fast_map(0);
    for (i, (f, m)) in faces.iter().zip(&merged).enumerate() {
        if f != m {
            groups.entry(Ids::face(*m)).or_insert((i as u32, false));
        }
    }
    for (f, m) in faces.iter().zip(&merged) {
        if f == m
            && f.iter().any(|&v| kept[v as usize])
            && let Some(group) = groups.get_mut(&Ids::face(*f))
        {
            group.1 = true;
        }
    }
    let mut collapsed = 0;
    for (i, (f, m)) in faces.iter().zip(merged.iter_mut()).enumerate() {
        if f == m {
            continue;
        }
        let (first_face, any_unchanged) = groups[&Ids::face(*m)];
        if any_unchanged || first_face as usize != i {
            *m = [0, 0, 0];
            collapsed += 1;
        }
    }
    Some(Merged {
        faces: merged,
        copies,
        collapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_name_the_first_at_their_position() {
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -0.0, 0.0],
            [1.0, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
        ];
        assert_eq!(copies(&v), vec![[2, 0], [3, 1]]);
    }

    #[test]
    fn the_shards_find_what_one_table_finds() {
        // enough vertices for the shards: a grid of 7,000 points visited
        // three times over, the last point of the second visit at −0.0 and
        // one of the first visit's at a NaN
        let n = 3 * 7_000;
        assert!(n >= threads::MIN_PARALLEL);
        let mut v: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let k = (i * 7_919) % 7_000;
                [
                    (k % 20) as f64 * 100.0,
                    (k / 20 % 25) as f64 * 100.0,
                    (k / 500) as f64,
                ]
            })
            .collect();
        let signed = (7_000..14_000).rev().find(|&i| v[i][0] == 0.0).unwrap();
        v[signed][0] = -0.0;
        v[9][1] = f64::NAN;
        let mut sharded = copies(&v);
        sharded.sort_unstable();
        let all: Vec<u32> = (0..n as u32).collect();
        assert_eq!(sharded, copies_among(&v, &all));
        assert!(!sharded.iter().flatten().any(|&i| i == 9));
        let found = sharded.iter().find(|c| c[0] as usize == signed).unwrap();
        assert!((found[1] as usize) < 7_000);
    }

    #[test]
    fn a_seam_is_merged_and_its_copied_faces_collapse() {
        // two chunks share the edge 1-2; the second copies it as 4-5 and
        // also copies the first chunk's face 0-1-2 as 3-4-5
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let m = merge_coincident(&v, &[[0, 1, 2], [3, 4, 5], [4, 6, 5]]).unwrap();
        assert_eq!(m.faces, vec![[0, 1, 2], [0, 0, 0], [1, 6, 2]]);
        assert_eq!((m.copies, m.collapsed), (3, 1));
    }

    #[test]
    fn duplicates_the_mesh_had_stay() {
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let m = merge_coincident(&v, &[[0, 1, 2], [0, 2, 1], [0, 3, 2]]).unwrap();
        // the copy made by the merge goes; the two the mesh had stay
        assert_eq!(m.faces, vec![[0, 1, 2], [0, 2, 1], [0, 0, 0]]);
        assert_eq!((m.copies, m.collapsed), (1, 1));
    }

    #[test]
    fn vertices_that_only_touch_stay_apart() {
        // two triangles meeting at (0, 0, 0), each with its own vertex there,
        // as `remove_fusions` cuts a bow-tie
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        assert_eq!(merge_coincident(&v, &[[0, 1, 2], [3, 4, 5]]), None);
    }

    #[test]
    fn a_face_naming_a_vertex_twice_joins_nothing() {
        // only the needle 0-0-1 has the edge 0-1 that 3-1 repeats: no seam
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let faces = [[0, 0, 1], [0, 2, 5], [3, 4, 1]];
        assert_eq!(merge_coincident(&v, &faces), None);
    }

    #[test]
    fn touching_copies_stay_beside_a_seam() {
        // 0 and 3 are joined by the seam 0-1 / 3-4 (1 and 4 coincide too);
        // 6, at their position as well, only touches them
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
        ];
        let m = merge_coincident(&v, &[[0, 1, 2], [3, 5, 4], [6, 7, 8]]).unwrap();
        assert_eq!(m.faces, vec![[0, 1, 2], [0, 5, 1], [6, 7, 8]]);
        assert_eq!((m.copies, m.collapsed), (2, 0));
        let joined = seam_joined(&v, &[[0, 1, 2], [3, 5, 4], [6, 7, 8]]).unwrap();
        assert_eq!(joined[6], 6);
    }

    #[test]
    fn nothing_coincides() {
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(merge_coincident(&v, &[[0, 1, 2]]), None);
    }
}
