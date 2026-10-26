//! The bins — the mesh vertices each node owns — checked against
//! `vert2node` and the mesh: `sk.dx.check_bins` and `face_owner`.

use rayon::prelude::*;

/// What `check_bins` reports of the bins (the first is the soma's `verts`,
/// not a bin) against `vert2node`.
#[derive(Debug, PartialEq)]
pub struct BinCheck {
    /// Vertices some bin past the first owns.
    pub n_owned: usize,
    /// Vertices listed more than once in the bins past the first, ascending.
    pub duplicated: Vec<usize>,
    /// Vertices whose `vert2node` entry is not the last bin listing them
    /// (−1 for none on either side), ascending.
    pub mismatched: Vec<usize>,
    /// Nodes with an empty bin.
    pub empty_bins: Vec<usize>,
    /// Distinct vertices of the first bin that another bin owns too.
    pub soma_overlap: usize,
}

/// `check_bins` on the bins and `vert2node` as `keys` → `vals`.
pub fn check_bins(bins: &[&[usize]], keys: &[usize], vals: &[i64]) -> BinCheck {
    let size = bins
        .iter()
        .flat_map(|b| b.iter())
        .chain(keys)
        .max()
        .map_or(0, |&v| v + 1);
    let mut count = vec![0u32; size];
    let mut expect = vec![-1i64; size];
    for (nid, bin) in bins.iter().enumerate() {
        for &v in bin.iter() {
            if nid > 0 {
                count[v] += 1;
            }
            expect[v] = nid as i64;
        }
    }
    let mut actual = vec![-1i64; size];
    for (&k, &v) in keys.iter().zip(vals) {
        actual[k] = v;
    }
    let mut in_soma = vec![false; size];
    if let Some(soma) = bins.first() {
        for &v in soma.iter() {
            in_soma[v] = true;
        }
    }
    BinCheck {
        n_owned: count.iter().filter(|&&c| c > 0).count(),
        duplicated: (0..size).filter(|&v| count[v] > 1).collect(),
        mismatched: (0..size).filter(|&v| expect[v] != actual[v]).collect(),
        empty_bins: (0..bins.len()).filter(|&i| bins[i].is_empty()).collect(),
        soma_overlap: (0..size).filter(|&v| in_soma[v] && count[v] > 0).count(),
    }
}

/// The bins past the first that lie in more than one patch of surface: the
/// mesh `edges` whose two ends one bin owns (the last bin listing a vertex
/// owns it), joined; for each bin of more than one vertex, how many patches
/// its vertices fall in, when more than one. Ascending by node.
pub fn fragmented_bins(
    bins: &[&[usize]],
    edges: &[[usize; 2]],
    n_vertices: usize,
) -> Vec<(usize, usize)> {
    let mut owner = vec![0usize; n_vertices];
    for (nid, bin) in bins.iter().enumerate().skip(1) {
        for &v in bin.iter() {
            owner[v] = nid;
        }
    }
    let mut root: Vec<usize> = (0..n_vertices).collect();
    fn find(root: &mut [usize], mut v: usize) -> usize {
        while root[v] != v {
            root[v] = root[root[v]];
            v = root[v];
        }
        v
    }
    for &[a, b] in edges {
        if owner[a] != 0 && owner[a] == owner[b] {
            let (ra, rb) = (find(&mut root, a), find(&mut root, b));
            if ra != rb {
                root[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    let mut stamp = vec![usize::MAX; n_vertices];
    let mut out = Vec::new();
    for (nid, bin) in bins.iter().enumerate().skip(1) {
        if bin.len() < 2 {
            continue;
        }
        let mut pieces = 0;
        for &v in bin.iter() {
            let r = find(&mut root, v);
            if stamp[r] != nid {
                stamp[r] = nid;
                pieces += 1;
            }
        }
        if pieces > 1 {
            out.push((nid, pieces));
        }
    }
    out
}

/// `face_owner`: per face, the owner (by `owner`, −1 for none) of at least
/// two of its three vertices, or −1. Faces in parallel.
pub fn face_owner(faces: &[[usize; 3]], owner: &[i64]) -> Vec<i64> {
    let one = |f: &[usize; 3]| -> i64 {
        let mut o = f.map(|v| owner[v]);
        o.sort_unstable();
        if o[0] == o[1] {
            o[0]
        } else if o[1] == o[2] {
            o[1]
        } else {
            -1
        }
    };
    threads::install(|| faces.par_iter().map(one).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vertex_in_two_bins() {
        // node 1 and node 2 both list vertex 3; node 0 (the soma) lists 1
        let bins: [&[usize]; 4] = [&[0, 1], &[1, 2, 3], &[3, 4], &[]];
        let check = check_bins(&bins, &[0, 1, 2, 3, 4], &[0, 1, 1, 2, 2]);
        assert_eq!(check.n_owned, 4);
        assert_eq!(check.duplicated, [3]);
        assert_eq!(check.mismatched, Vec::<usize>::new());
        assert_eq!(check.empty_bins, [3]);
        assert_eq!(check.soma_overlap, 1);
        // vertex 4 left out of vert2node, vertex 5 named by it alone
        let check = check_bins(&bins, &[0, 1, 2, 3, 5], &[0, 1, 1, 2, 2]);
        assert_eq!(check.mismatched, [4, 5]);
    }

    #[test]
    fn a_bin_in_two_patches() {
        // a path 0-1-2-3-4: bin 1 holds 0, 1 and 3 (3 cut off by 2)
        let bins: [&[usize]; 3] = [&[], &[0, 1, 3], &[2, 4]];
        let edges = [[0, 1], [1, 2], [2, 3], [3, 4]];
        assert_eq!(fragmented_bins(&bins, &edges, 5), [(1, 2), (2, 2)]);
        let owner = [1, 1, 2, -1, 2];
        assert_eq!(
            face_owner(&[[0, 1, 2], [2, 3, 4], [0, 2, 3]], &owner),
            [1, 2, -1]
        );
    }
}
