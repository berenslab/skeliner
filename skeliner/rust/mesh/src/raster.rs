//! Boolean 2D grids and the `scipy.ndimage` operations run on them.
//!
//! A grid is `nx` rows of `ny` cells, row-major, as numpy lays out an
//! `(nx, ny)` array; "connected" means sharing a side, scipy's default
//! structure in 2D.

use std::collections::VecDeque;

/// Neighbours of cell `c` sharing a side, in the grid `nx` x `ny`.
fn sides(c: usize, nx: usize, ny: usize) -> impl Iterator<Item = usize> {
    let (i, j) = (c / ny, c % ny);
    let up = (i > 0).then(|| c - ny);
    let down = (i + 1 < nx).then(|| c + ny);
    let left = (j > 0).then(|| c - 1);
    let right = (j + 1 < ny).then(|| c + 1);
    [up, down, left, right].into_iter().flatten()
}

/// `binary_dilation` with a 3x3 block: a cell is set when any cell of its
/// 3x3 neighbourhood is; cells off the grid count as unset.
pub fn dilate3(m: &[bool], nx: usize, ny: usize) -> Vec<bool> {
    let mut rows = vec![false; nx * ny];
    for (row, out) in m.chunks_exact(ny).zip(rows.chunks_exact_mut(ny)) {
        for (j, o) in out.iter_mut().enumerate() {
            let lo = j.saturating_sub(1);
            let hi = (j + 1).min(ny - 1);
            *o = row[lo..=hi].iter().any(|&b| b);
        }
    }
    let mut out = vec![false; nx * ny];
    for i in 0..nx {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(nx - 1);
        for j in 0..ny {
            out[i * ny + j] = (lo..=hi).any(|r| rows[r * ny + j]);
        }
    }
    out
}

/// `scipy.ndimage.label`: components of set cells, numbered from 1 in the
/// order a row-major scan first meets them; unset cells are 0.  Returns the
/// labels and the number of components.
pub fn label(m: &[bool], nx: usize, ny: usize) -> (Vec<u32>, u32) {
    let mut lab = vec![0u32; nx * ny];
    let mut n = 0u32;
    let mut queue = VecDeque::new();
    for start in 0..nx * ny {
        if !m[start] || lab[start] != 0 {
            continue;
        }
        n += 1;
        lab[start] = n;
        queue.push_back(start);
        while let Some(c) = queue.pop_front() {
            for nb in sides(c, nx, ny) {
                if m[nb] && lab[nb] == 0 {
                    lab[nb] = n;
                    queue.push_back(nb);
                }
            }
        }
    }
    (lab, n)
}

/// `binary_fill_holes`: set cells plus every unset cell not connected to
/// the grid's edge through unset cells.
pub fn fill_holes(m: &[bool], nx: usize, ny: usize) -> Vec<bool> {
    let mut outside = vec![false; nx * ny];
    let mut queue = VecDeque::new();
    for c in 0..nx * ny {
        let (i, j) = (c / ny, c % ny);
        let edge = i == 0 || j == 0 || i + 1 == nx || j + 1 == ny;
        if edge && !m[c] {
            outside[c] = true;
            queue.push_back(c);
        }
    }
    while let Some(c) = queue.pop_front() {
        for nb in sides(c, nx, ny) {
            if !m[nb] && !outside[nb] {
                outside[nb] = true;
                queue.push_back(nb);
            }
        }
    }
    outside.iter().map(|&o| !o).collect()
}

/// Cells with a set cell at or before them and at or after them along both
/// axes — the 4-ray enclosure `maximum.accumulate` computes forwards and
/// backwards per axis.
pub fn enclosed(m: &[bool], nx: usize, ny: usize) -> Vec<bool> {
    let mut out = vec![true; nx * ny];
    let mut seen = vec![false; ny];
    for i in 0..nx {
        for j in 0..ny {
            seen[j] |= m[i * ny + j];
            out[i * ny + j] &= seen[j];
        }
    }
    seen.fill(false);
    for i in (0..nx).rev() {
        for j in 0..ny {
            seen[j] |= m[i * ny + j];
            out[i * ny + j] &= seen[j];
        }
    }
    for i in 0..nx {
        let mut fwd = false;
        for j in 0..ny {
            fwd |= m[i * ny + j];
            out[i * ny + j] &= fwd;
        }
        let mut bwd = false;
        for j in (0..ny).rev() {
            bwd |= m[i * ny + j];
            out[i * ny + j] &= bwd;
        }
    }
    out
}

/// Squared distance from every cell to the nearest `feature` cell, exactly
/// (`i64::MAX` when there is none): a column pass, then the lower envelope
/// of parabolas along each row with intersections compared as fractions.
pub fn sq_distance_to(feature: &[bool], nx: usize, ny: usize) -> Vec<i64> {
    const INF: i64 = i64::MAX;
    // distance along the column to the nearest feature
    let mut col = vec![INF; nx * ny];
    for j in 0..ny {
        let mut last: Option<usize> = None;
        for i in 0..nx {
            if feature[i * ny + j] {
                last = Some(i);
            }
            if let Some(l) = last {
                col[i * ny + j] = (i - l) as i64;
            }
        }
        last = None;
        for i in (0..nx).rev() {
            if feature[i * ny + j] {
                last = Some(i);
            }
            if let Some(l) = last {
                let d = (l - i) as i64;
                if d < col[i * ny + j] {
                    col[i * ny + j] = d;
                }
            }
        }
    }
    let mut out = vec![INF; nx * ny];
    // per row: parabolas centred at q with height g(q)^2
    let mut v: Vec<i64> = Vec::with_capacity(ny);
    // boundaries between successive parabolas, as numerator / denominator
    let mut z: Vec<(i64, i64)> = Vec::with_capacity(ny + 1);
    for i in 0..nx {
        let g = &col[i * ny..(i + 1) * ny];
        let f = |q: i64| {
            let d = g[q as usize];
            d * d + q * q
        };
        v.clear();
        z.clear();
        for q in 0..ny as i64 {
            if g[q as usize] == INF {
                continue;
            }
            loop {
                let Some(&p) = v.last() else {
                    v.push(q);
                    break;
                };
                // where parabola q overtakes parabola p
                let s = (f(q) - f(p), 2 * (q - p));
                let behind = z
                    .last()
                    .is_some_and(|&(zn, zd)| s.0 as i128 * zd as i128 <= zn as i128 * s.1 as i128);
                if behind {
                    v.pop();
                    z.pop();
                } else {
                    z.push(s);
                    v.push(q);
                    break;
                }
            }
        }
        if v.is_empty() {
            continue;
        }
        let mut k = 0;
        for j in 0..ny as i64 {
            // advance while the next boundary lies before j
            while k < z.len() && (z[k].0 as i128) < j as i128 * z[k].1 as i128 {
                k += 1;
            }
            let q = v[k];
            let d = g[q as usize];
            out[i * ny + j as usize] = d * d + (j - q) * (j - q);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> (Vec<bool>, usize, usize) {
        let nx = rows.len();
        let ny = rows[0].len();
        let m = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| b == b'#'))
            .collect();
        (m, nx, ny)
    }

    #[test]
    fn labels_follow_the_row_major_scan_and_ignore_diagonals() {
        let (m, nx, ny) = grid(&["#..#", ".#.#", "...."]);
        let (lab, n) = label(&m, nx, ny);
        assert_eq!(n, 3);
        assert_eq!(lab, vec![1, 0, 0, 2, 0, 3, 0, 2, 0, 0, 0, 0]);
    }

    #[test]
    fn holes_touching_the_edge_stay_open() {
        let (m, nx, ny) = grid(&["#####", "#...#", "#####", "#..##"]);
        let filled = fill_holes(&m, nx, ny);
        assert!(filled[6] && filled[7] && filled[8]);
        assert!(!filled[16] && !filled[17]);
    }

    #[test]
    fn dilation_and_enclosure() {
        let (m, nx, ny) = grid(&["....", ".#..", "....", "...."]);
        let d = dilate3(&m, nx, ny);
        let set: Vec<usize> = (0..16).filter(|&c| d[c]).collect();
        assert_eq!(set, vec![0, 1, 2, 4, 5, 6, 8, 9, 10]);
        let (m, nx, ny) = grid(&[".#.", "#.#", ".#."]);
        let e = enclosed(&m, nx, ny);
        assert!(e[4] && !e[0]);
    }

    #[test]
    fn squared_distances_match_brute_force() {
        let mut state = 12345u64;
        for _ in 0..200 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let nx = 1 + (state >> 59) as usize;
            let ny = 1 + ((state >> 53) & 31) as usize;
            let feat: Vec<bool> = (0..nx * ny)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 60) < 3
                })
                .collect();
            let got = sq_distance_to(&feat, nx, ny);
            for (c, &d) in got.iter().enumerate() {
                let (i, j) = ((c / ny) as i64, (c % ny) as i64);
                let want = (0..nx * ny)
                    .filter(|&f| feat[f])
                    .map(|f| {
                        let (a, b) = ((f / ny) as i64, (f % ny) as i64);
                        (a - i) * (a - i) + (b - j) * (b - j)
                    })
                    .min()
                    .unwrap_or(i64::MAX);
                assert_eq!(d, want, "{nx}x{ny} cell {c}");
            }
        }
    }
}
