//! `skeliner._state`: the arrays a skeleton keeps in step.

/// `vert2node` from the bins (each node's mesh vertices, nodes in order):
/// its keys — each vertex the first time a bin lists it — and, by vertex,
/// the node of the last bin listing it (−1 for none), `n_vertices` long.
/// Every vertex id must be below `n_vertices`.
pub fn vertex_map(bins: &[&[usize]], n_vertices: usize) -> (Vec<usize>, Vec<i64>) {
    let mut node = vec![-1i64; n_vertices];
    let mut keys = Vec::new();
    for (nid, bin) in bins.iter().enumerate() {
        for &v in bin.iter() {
            if node[v] < 0 {
                keys.push(v);
            }
            node[v] = nid as i64;
        }
    }
    (keys, node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_come_first_seen_and_the_last_bin_wins() {
        // vertex 3 in nodes 0 and 2, vertex 1 twice in node 1
        let bins: [&[usize]; 3] = [&[3, 0], &[1, 1, 4], &[3]];
        let (keys, node) = vertex_map(&bins, 6);
        assert_eq!(keys, [3, 0, 1, 4]);
        assert_eq!(node, [0, 1, -1, 2, 1, -1]);
    }
}
