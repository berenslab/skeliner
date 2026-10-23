//! `skeliner.dx`'s kernels: distances from points to a skeleton.

#![forbid(unsafe_code)]

/// `skeliner.dx.distance` in centreline mode against a whitelist: per point,
/// the distance to the nearest of the `centres` (node ids) or of the
/// `segments` (node-id pairs), each computed as `_point_segment_distance`
/// computes it; infinity when there are neither. Every id indexes `nodes`.
pub fn segment_distances(
    points: &[[f64; 3]],
    nodes: &[[f64; 3]],
    centres: &[usize],
    segments: &[(usize, usize)],
) -> Vec<f64> {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    points
        .iter()
        .map(|&p| {
            let mut best = f64::INFINITY;
            for &c in centres {
                let d = norm(sub(nodes[c], p));
                if d < best {
                    best = d;
                }
            }
            for &(a, b) in segments {
                let (s, e) = (nodes[a], nodes[b]);
                let v = sub(e, s);
                let len2 = dot(v, v);
                let d = if len2 <= 0.0 {
                    norm(sub(p, s))
                } else {
                    // Python's min(1.0, max(0.0, t)), which sends NaN to 0
                    let t = dot(sub(p, s), v) / len2;
                    let t = if t > 0.0 { t } else { 0.0 };
                    let t = if t < 1.0 { t } else { 1.0 };
                    norm(sub(p, [s[0] + t * v[0], s[1] + t * v[1], s[2] + t * v[2]]))
                };
                if d < best {
                    best = d;
                }
            }
            best
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_of_centres_and_segments() {
        let nodes = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 5.0, 0.0]];
        let pts = [[1.0, 1.0, 0.0], [3.0, 0.0, 0.0], [5.0, 5.0, 1.0]];
        let d = segment_distances(&pts, &nodes, &[2], &[(0, 1)]);
        assert_eq!(d, vec![1.0, 1.0, 1.0]);
        assert_eq!(
            segment_distances(&pts, &nodes, &[], &[]),
            vec![f64::INFINITY; 3]
        );
    }
}
