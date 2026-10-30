//! `skeliner.dataclass.Skeleton`'s own arrays.

/// Each node's neighbours as a CSR `(ptr, idx)` over `n` nodes: in the
/// order the edges name them, an edge's first end before its second, so a
/// self-loop lists its node twice — numpy's stable argsort of the edges'
/// ends.  Every node id must be below `n`.
pub fn node_neighbors(edges: &[[usize; 2]], n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut ptr = vec![0usize; n + 1];
    for &[a, b] in edges {
        ptr[a + 1] += 1;
        ptr[b + 1] += 1;
    }
    for i in 0..n {
        ptr[i + 1] += ptr[i];
    }
    let mut next = ptr[..n].to_vec();
    let mut idx = vec![0usize; 2 * edges.len()];
    for &[a, b] in edges {
        idx[next[a]] = b;
        next[a] += 1;
        idx[next[b]] = a;
        next[b] += 1;
    }
    (ptr, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_come_in_edge_order_and_a_loop_twice() {
        let (ptr, idx) = node_neighbors(&[[2, 0], [1, 1], [0, 1], [2, 2]], 4);
        assert_eq!(ptr, [0, 2, 5, 8, 8]);
        // node 0: 2 (edge 0), 1 (edge 2); node 1: itself twice, then 0
        assert_eq!(idx, [2, 1, 1, 1, 0, 0, 2, 2]);
        assert_eq!(node_neighbors(&[], 2), (vec![0, 0, 0], vec![]));
    }
}
