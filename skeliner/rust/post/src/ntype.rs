//! Node types (`ntype`, SWC's) carried through edits that merge, drop or
//! renumber nodes, and zero labels filled from their neighbours.

/// The order a tie between labels goes: the soma, the root's other mark,
/// the neurite types, unlabelled.
const PRIORITY: [i64; 8] = [1, -1, 2, 3, 4, 5, 6, 0];

/// The label of a node made of nodes labelled `labels`: their most common
/// non-zero label (the most common of all when every one is zero), a tie
/// going by `PRIORITY`, then to the smallest; 0 for no labels.
fn choose_label(labels: &[i64]) -> i64 {
    if let [only] = labels {
        return *only;
    }
    let mut vals: Vec<i64> = labels.iter().copied().filter(|&l| l != 0).collect();
    if vals.is_empty() {
        vals = labels.to_vec();
    }
    if vals.is_empty() {
        return 0;
    }
    vals.sort_unstable();
    let mut counts: Vec<(i64, usize)> = Vec::new();
    for v in vals {
        match counts.last_mut() {
            Some((u, c)) if *u == v => *c += 1,
            _ => counts.push((v, 1)),
        }
    }
    let most = counts.iter().map(|&(_, c)| c).max().unwrap_or(0);
    let tied: Vec<i64> = counts
        .iter()
        .filter(|&&(_, c)| c == most)
        .map(|&(u, _)| u)
        .collect();
    PRIORITY
        .iter()
        .copied()
        .find(|p| tied.contains(p))
        .unwrap_or(tied[0])
}

/// The root (node 0) marked `1` when `root_label` is 1 and `-1` otherwise;
/// every other node marked `1` or `-1` unlabelled.
pub fn ensure_root_label(labels: &mut [i64], root_label: i64) {
    let Some((first, rest)) = labels.split_first_mut() else {
        return;
    };
    *first = if root_label == 1 { 1 } else { -1 };
    for l in rest {
        if *l == 1 || *l == -1 {
            *l = 0;
        }
    }
}

/// The most common of `vals` as numpy's `bincount(clip(vals, 0, max))`
/// counts them, `max` taken with 0 as its floor: negative values count as
/// 0, and a tie goes to the smallest value.
fn mode_int(vals: &[i64]) -> i64 {
    let top = vals.iter().copied().fold(0, i64::max);
    let mut counts = vec![0usize; top as usize + 1];
    for &v in vals {
        counts[v.clamp(0, top) as usize] += 1;
    }
    let mut best = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c > counts[best] {
            best = i;
        }
    }
    best as i64
}

/// Zero labels filled from their neighbours over the undirected `edges`:
/// sweeps over the nodes labelled 0 at the sweep's start, in order, each
/// taking the most common neighbouring label other than 0, 1 and -1
/// (`mode_int`; a neighbour counts once per edge, and a fill earlier in
/// the sweep is seen), until a sweep fills nothing, at most one sweep per
/// node. `edges` must name nodes of `labels`.
pub fn fill_ntype_gaps(labels: &[i64], edges: &[[usize; 2]]) -> Vec<i64> {
    let n = labels.len();
    let mut filled = labels.to_vec();
    if n == 0 || edges.is_empty() {
        return filled;
    }
    let mut adj = vec![Vec::new(); n];
    for &[a, b] in edges {
        if a != b {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    for _ in 0..n {
        let zeros: Vec<usize> = (0..n).filter(|&i| filled[i] == 0).collect();
        if zeros.is_empty() {
            break;
        }
        let mut changed = false;
        for z in zeros {
            let around: Vec<i64> = adj[z]
                .iter()
                .map(|&j| filled[j])
                .filter(|&l| l != 0 && l != 1 && l != -1)
                .collect();
            if around.is_empty() {
                continue;
            }
            let label = mode_int(&around);
            if label != 0 {
                filled[z] = label;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    filled
}

/// Why node types could not be remapped.
#[derive(Debug, PartialEq)]
pub enum RemapError {
    /// `old2new` and the labels differ in length.
    Length,
    /// An entry of `old2new` past the new node count.
    Target(i64),
}

impl std::fmt::Display for RemapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemapError::Length => write!(f, "old2new must have one entry per node"),
            RemapError::Target(t) => write!(f, "old2new names node {t}, past the new nodes"),
        }
    }
}

/// Node types after the nodes are renumbered by `old2new` (−1 for a node
/// dropped) into `new_len` nodes: each new node takes `choose_label` of
/// the old nodes mapped to it, the root is marked by `root_label`
/// (`ensure_root_label`), and with `fill_gaps` the zero labels are filled
/// over `edges` (between new nodes) and the root marked again.
pub fn remap_ntype(
    labels: &[i64],
    old2new: &[i64],
    new_len: usize,
    root_label: i64,
    edges: &[[usize; 2]],
    fill_gaps: bool,
) -> Result<Vec<i64>, RemapError> {
    if old2new.len() != labels.len() {
        return Err(RemapError::Length);
    }
    let mut buckets: Vec<Vec<i64>> = vec![Vec::new(); new_len];
    for (&label, &new) in labels.iter().zip(old2new) {
        if new < 0 {
            continue;
        }
        let bucket = usize::try_from(new)
            .ok()
            .and_then(|i| buckets.get_mut(i))
            .ok_or(RemapError::Target(new))?;
        bucket.push(label);
    }
    let mut mapped: Vec<i64> = buckets.iter().map(|b| choose_label(b)).collect();
    ensure_root_label(&mut mapped, root_label);
    if fill_gaps {
        mapped = fill_ntype_gaps(&mapped, edges);
        ensure_root_label(&mut mapped, root_label);
    }
    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_merged_node_takes_the_most_common_label_ties_by_priority() {
        assert_eq!(choose_label(&[]), 0);
        assert_eq!(choose_label(&[0, 0]), 0);
        assert_eq!(choose_label(&[0, 3, 3, 2]), 3);
        assert_eq!(choose_label(&[3, 2, 0, 0, 0]), 2);
        assert_eq!(choose_label(&[1, 4]), 1);
        assert_eq!(choose_label(&[9, 7]), 7);
    }

    #[test]
    fn gaps_fill_from_neighbours_in_one_sweep_or_several() {
        // 3 - 0 - 0 - 4 on a path: the first zero fills from 3, and the
        // second sees that fill (3 and 4 tie, the smaller wins)
        let filled = fill_ntype_gaps(&[3, 0, 0, 4], &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(filled, vec![3, 3, 3, 4]);
        // negative labels count as 0, which wins a tie
        assert_eq!(
            fill_ntype_gaps(&[0, -2, 5], &[[0, 1], [0, 2]]),
            vec![0, -2, 5]
        );
        assert_eq!(fill_ntype_gaps(&[0, 2], &[]), vec![0, 2]);
    }

    #[test]
    fn remapping_merges_marks_the_root_and_fills() {
        // nodes 1 and 2 merge into new node 1; node 3 is dropped
        let got = remap_ntype(
            &[1, 3, 0, 2, 0],
            &[0, 1, 1, -1, 2],
            3,
            1,
            &[[0, 1], [1, 2]],
            true,
        );
        assert_eq!(got, Ok(vec![1, 3, 3]));
        assert_eq!(
            remap_ntype(&[1, 3], &[0], 1, 1, &[], true),
            Err(RemapError::Length)
        );
        assert_eq!(
            remap_ntype(&[1, 3], &[0, 5], 1, 1, &[], true),
            Err(RemapError::Target(5))
        );
        // a second root mark is cleared
        assert_eq!(
            remap_ntype(&[-1, 1, 2], &[0, 1, 2], 3, -1, &[], false),
            Ok(vec![-1, 0, 2])
        );
    }
}
