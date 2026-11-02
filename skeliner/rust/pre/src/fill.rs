//! Closing the holes in a cell's surface — the organelles' mouths and the
//! mesh's own openings — each with a patch of triangles about as long as the
//! median edge, shaped like the tube it sits in.
//!
//! The surface is the usable faces that are no organelle. A hole is a loop
//! of its border that lies on its mesh component's body (its largest piece,
//! the one `break_up_mesh` measures the others against), that no soma vertex
//! touches (the soma is cut out next, along its junctions), and that opens
//! the side of the surface rather than ending it: the faces along it face
//! the way the loop spans more than a direction at random would (a mean
//! `|cos|` over 1/2), where round a tube's cut end they face across it.
//!
//! Each hole is triangulated from its loop alone with the least folding
//! (Liepa 2003, after Barequet & Sharir 1995: the largest dihedral angle,
//! then the area), refined until no triangle is long against the median
//! edge (Liepa's centroid splits and Delaunay swaps), and faired: the patch
//! and the hole's lip take the shape of least thin-plate energy — the
//! squared umbrella Laplacian — with the surface beyond held. Over its
//! tangent plane a quadric's height (a tube's side, a sphere's cap, a
//! saddle) is biharmonic, so the patch continues the tube, curved as the
//! tube is, rather than spanning it flat or sagging into the mouth.
//!
//! The lip is the loop itself — where the organelle happened to end, not
//! where the tube does — and the rings of surface round it that dip into
//! the mouth: a band of rings round the hole is faired with the patch first,
//! to the smooth surface the tube would have there, and the rings from the
//! loop out that lie below it on average are the lip; the band widens (2,
//! 4, 8 .. rings, up to the hole's radius) until the lip ends inside it.

use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, SQRT_2};
use std::ops::Range;

use mesh::Topology;
use mesh::components::{LowestRootSets, face_edge_components_without};
use rayon::prelude::*;

use crate::breakup::usable_faces;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// `a` scaled to length 1, or zero.
fn unit(a: V3) -> V3 {
    let l = norm(a);
    if l > 0.0 { scale(a, 1.0 / l) } else { [0.0; 3] }
}

/// The angle between `a` and `b`.
fn angle(a: V3, b: V3) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

/// The vertex of `f` that is neither `a` nor `b`.
fn third(f: [u32; 3], a: u32, b: u32) -> u32 {
    f.into_iter().find(|&v| v != a && v != b).unwrap_or(f[0])
}

/// Whether face `f` runs from `a` to `b`.
fn runs(f: [u32; 3], a: u32, b: u32) -> bool {
    (f[0] == a && f[1] == b) || (f[1] == a && f[2] == b) || (f[2] == a && f[0] == b)
}

/// A loop of a face set's border: edge `i` runs from `verts[i]` to
/// `verts[i + 1]` (the last back to the first) along `faces[i]`, the member
/// face on it, and runs as that face does for most of the loop's edges.
#[derive(Clone, Debug)]
pub struct Loop {
    pub verts: Vec<u32>,
    pub faces: Vec<u32>,
}

/// The border of the faces `member` marks, as loops passing no vertex
/// twice. A border edge is an edge of a member face with one member face on
/// it, or three or more; each is followed to the next round its end vertex,
/// through the member faces there, so the faces meeting at a pinch vertex
/// part their loops instead of joining them. A walk that does not close is
/// dropped.
pub fn border_loops(topo: &Topology, faces: &[[u32; 3]], member: &[bool]) -> Vec<Loop> {
    let members_on = |e: usize| {
        topo.edge_faces(e)
            .iter()
            .copied()
            .filter(|&f| member[f as usize])
    };
    // the border's edges, once per member face on them: (low, high, face)
    let half: Vec<(u32, u32, u32)> = threads::install(|| {
        let mut half: Vec<(u32, u32, u32)> = (0..faces.len())
            .into_par_iter()
            .filter(|&f| member[f])
            .flat_map_iter(|f| {
                topo.face_edges(f)
                    .into_iter()
                    .flatten()
                    .filter(|&e| members_on(e as usize).count() != 2)
                    .map(move |e| {
                        let [a, b] = topo.edges()[e as usize];
                        (a.min(b), a.max(b), f as u32)
                    })
            })
            .collect();
        half.par_sort_unstable();
        half
    });
    let find = |a: u32, b: u32, f: u32| half.binary_search(&(a.min(b), a.max(b), f)).ok();
    let mut used = vec![false; half.len()];
    let mut loops = Vec::new();
    for start in 0..half.len() {
        if used[start] {
            continue;
        }
        let (lo, hi, f0) = half[start];
        let (mut cur, mut f, mut from, mut at) = (start, f0, lo, hi);
        let (mut verts, mut on) = (Vec::new(), Vec::new());
        let closed = loop {
            used[cur] = true;
            verts.push(from);
            on.push(f);
            // round `at`, face by face, to its next border edge
            let (mut g, mut x) = (f, third(faces[f as usize], from, at));
            let mut next = None;
            for _ in 0..=topo.vertex_faces(at as usize).len() {
                if let Some(i) = find(at, x, g) {
                    next = Some((i, g, x));
                    break;
                }
                let Some(e) = topo.edge_id(at, x) else { break };
                let Some(h) = members_on(e).find(|&h| h != g) else {
                    break;
                };
                x = third(faces[h as usize], at, x);
                g = h;
            }
            match next {
                Some((i, _, _)) if i == start => break true,
                Some((i, g, x)) if !used[i] => (cur, f, from, at) = (i, g, at, x),
                _ => break false,
            }
        };
        if closed {
            loops.extend(simple(orient(faces, Loop { verts, faces: on })));
        }
    }
    loops
}

/// `lp` turned, if most of its faces run against it.
fn orient(faces: &[[u32; 3]], lp: Loop) -> Loop {
    let n = lp.verts.len();
    let along = (0..n)
        .filter(|&i| {
            runs(
                faces[lp.faces[i] as usize],
                lp.verts[i],
                lp.verts[(i + 1) % n],
            )
        })
        .count();
    if 2 * along >= n {
        return lp;
    }
    Loop {
        verts: (0..n).map(|j| lp.verts[(n - j) % n]).collect(),
        faces: (0..n).map(|j| lp.faces[n - 1 - j]).collect(),
    }
}

/// `lp` cut at each vertex it passes twice, into loops that pass none
/// twice; those of fewer than three vertices are dropped.
fn simple(lp: Loop) -> Vec<Loop> {
    let mut out = Vec::new();
    let mut todo = vec![lp];
    while let Some(lp) = todo.pop() {
        let mut first: HashMap<u32, usize> = HashMap::new();
        let mut again = None;
        for (j, &v) in lp.verts.iter().enumerate() {
            if let Some(&i) = first.get(&v) {
                again = Some((i, j));
                break;
            }
            first.insert(v, j);
        }
        match again {
            None if lp.verts.len() >= 3 => out.push(lp),
            None => {}
            Some((i, j)) => {
                todo.push(Loop {
                    verts: lp.verts[i..j].to_vec(),
                    faces: lp.faces[i..j].to_vec(),
                });
                todo.push(Loop {
                    verts: lp.verts[j..]
                        .iter()
                        .chain(&lp.verts[..i])
                        .copied()
                        .collect(),
                    faces: lp.faces[j..]
                        .iter()
                        .chain(&lp.faces[..i])
                        .copied()
                        .collect(),
                });
            }
        }
    }
    out
}

/// Per face: whether it lies on its mesh component's body, the piece of
/// `surface` with the most faces there (the lowest-numbered among equals).
fn on_body(topo: &Topology, usable: &[bool], surface: &[bool]) -> Vec<bool> {
    let off: Vec<bool> = surface.iter().map(|&s| !s).collect();
    let (piece, _) = face_edge_components_without(topo, &off);
    let off: Vec<bool> = usable.iter().map(|&u| !u).collect();
    let (comp, _) = face_edge_components_without(topo, &off);
    let n_pieces = piece.iter().map(|&p| p + 1).max().unwrap_or(0).max(0) as usize;
    let mut size = vec![0usize; n_pieces];
    for &p in piece.iter().filter(|&&p| p >= 0) {
        size[p as usize] += 1;
    }
    let mut best: HashMap<i64, i64> = HashMap::new();
    for (&p, &c) in piece.iter().zip(&comp).filter(|(p, _)| **p >= 0) {
        let b = best.entry(c).or_insert(p);
        let (sp, sb) = (size[p as usize], size[*b as usize]);
        if sp > sb || (sp == sb && p < *b) {
            *b = p;
        }
    }
    let mut body = vec![false; n_pieces];
    for &p in best.values() {
        body[p as usize] = true;
    }
    piece.iter().map(|&p| p >= 0 && body[p as usize]).collect()
}

/// What became of a loop of the surface's border.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Patched.
    Filled,
    /// On a piece other than its mesh component's body.
    OffBody,
    /// A soma vertex on it.
    Soma,
    /// An edge of it with three or more surface faces.
    NonManifold,
    /// The faces along it face across the way it spans: a tube's end.
    TubeEnd,
    /// No span, or no triangulation without an edge the surface has.
    Untriangulable,
}

/// A loop of the surface's border and what became of it.
#[derive(Clone, Debug)]
pub struct Hole {
    pub verts: Vec<u32>,
    pub decision: Decision,
    /// Its patch's faces, in [`Filled::faces`].
    pub faces: Range<usize>,
    /// The vertices its patch added.
    pub n_new: usize,
    /// The vertices of its lip, freed to move with the patch.
    pub n_lip: usize,
}

/// What [`fill_holes`] did.
pub struct Filled {
    /// The vertices: the input's, those of the lips moved, then the
    /// patches' new ones.
    pub verts: Vec<V3>,
    /// The patches' faces, to append to the input's.
    pub faces: Vec<[u32; 3]>,
    /// Every loop of the surface's border, in the order traced.
    pub holes: Vec<Hole>,
    /// The input's vertices that moved.
    pub n_moved: usize,
    /// Conjugate-gradient iterations, summed over the fairing's systems.
    pub iterations: usize,
    /// When each stage ended, in seconds from the start.
    pub times: Vec<(&'static str, f64)>,
}

/// Patch the holes in the surface of `faces` — the usable faces that are
/// not `organelle` (shorter than `faces`: the rest are none) — with
/// triangles about `med` long, and fair each patch and its lip with the
/// surface around; `soma` marks the soma's vertices (shorter: the rest are
/// not).
pub fn fill_holes(
    verts: &[V3],
    faces: &[[u32; 3]],
    organelle: &[bool],
    soma: &[bool],
    med: f64,
) -> Filled {
    threads::install(|| fill(verts, faces, organelle, soma, med))
}

/// A hole's patch: its new vertices, and triangles over the loop's
/// positions (below the loop's length) and the new vertices after them.
struct Patch {
    new: Vec<V3>,
    tris: Vec<[u32; 3]>,
}

fn fill(verts: &[V3], faces: &[[u32; 3]], organelle: &[bool], soma: &[bool], med: f64) -> Filled {
    let (nv, nf) = (verts.len(), faces.len());
    let clock = std::time::Instant::now();
    let mut times = Vec::new();
    let lap = |name: &'static str, times: &mut Vec<(&'static str, f64)>| {
        times.push((name, clock.elapsed().as_secs_f64()))
    };
    let topo = Topology::new(faces, nv);
    if nv > 0 {
        // built here, once, rather than by the first of many workers
        let _ = topo.vertex_faces(0);
    }
    let usable = usable_faces(verts, faces);
    let surface: Vec<bool> = (0..nf)
        .map(|f| usable[f] && !organelle.get(f).copied().unwrap_or(false))
        .collect();
    let is_soma = |v: u32| soma.get(v as usize).copied().unwrap_or(false);
    let body = on_body(&topo, &usable, &surface);
    lap("topology", &mut times);
    let loops = border_loops(&topo, faces, &surface);
    lap("loops", &mut times);
    let surface_on = |a: u32, b: u32| {
        topo.edge_id(a, b).map_or(0, |e| {
            topo.edge_faces(e)
                .iter()
                .filter(|&&f| surface[f as usize])
                .count()
        })
    };

    // which loops are holes
    let judged: Vec<Decision> = loops
        .par_iter()
        .map(|lp| {
            let n = lp.verts.len();
            if !body[lp.faces[0] as usize] {
                return Decision::OffBody;
            }
            if lp.verts.iter().any(|&v| is_soma(v)) {
                return Decision::Soma;
            }
            if (0..n).any(|i| surface_on(lp.verts[i], lp.verts[(i + 1) % n]) >= 3) {
                return Decision::NonManifold;
            }
            let p: Vec<V3> = lp.verts.iter().map(|&v| verts[v as usize]).collect();
            let c = scale(p.iter().fold([0.0; 3], |s, &q| add(s, q)), 1.0 / n as f64);
            let span = (0..n).fold([0.0; 3], |s, i| {
                add(s, cross(sub(p[i], c), sub(p[(i + 1) % n], c)))
            });
            let way = unit(span);
            if way == [0.0; 3] {
                return Decision::Untriangulable;
            }
            let mut along: Vec<u32> = lp.faces.clone();
            along.sort_unstable();
            along.dedup();
            let (mut facing, mut total) = (0.0, 0.0);
            for &f in &along {
                let [a, b, d] = faces[f as usize].map(|v| verts[v as usize]);
                let x = cross(sub(b, a), sub(d, a));
                facing += dot(x, way).abs();
                total += norm(x);
            }
            if 2.0 * facing <= total {
                return Decision::TubeEnd;
            }
            Decision::Filled
        })
        .collect();

    lap("judged", &mut times);
    let patches: Vec<Option<Patch>> = loops
        .par_iter()
        .zip(&judged)
        .map(|(lp, &d)| {
            if d != Decision::Filled {
                return None;
            }
            patch(lp, verts, faces, &topo, &surface, med)
        })
        .collect();
    let decisions: Vec<Decision> = judged
        .iter()
        .zip(&patches)
        .map(|(&d, p)| {
            if d == Decision::Filled && p.is_none() {
                Decision::Untriangulable
            } else {
                d
            }
        })
        .collect();

    // vertices that stay: the soma's, and those of loops left open
    let mut pinned: Vec<bool> = (0..nv as u32).map(is_soma).collect();
    for (lp, &d) in loops.iter().zip(&decisions) {
        if d != Decision::Filled {
            for &v in &lp.verts {
                pinned[v as usize] = true;
            }
        }
    }

    // the patches' vertices and faces, numbered after the input's
    let mut out_verts = verts.to_vec();
    let mut out_faces: Vec<[u32; 3]> = Vec::new();
    let mut spans = Vec::with_capacity(loops.len());
    let mut vspans = Vec::with_capacity(loops.len());
    for (lp, p) in loops.iter().zip(&patches) {
        let start = out_faces.len();
        let vstart = out_verts.len() as u32;
        if let Some(p) = p {
            let n = lp.verts.len() as u32;
            out_verts.extend_from_slice(&p.new);
            out_faces.extend(p.tris.iter().map(|t| {
                t.map(|i| {
                    if i < n {
                        lp.verts[i as usize]
                    } else {
                        vstart + i - n
                    }
                })
            }));
        }
        spans.push(start..out_faces.len());
        vspans.push(vstart..out_verts.len() as u32);
    }
    let new: Vec<u32> = (nv as u32..out_verts.len() as u32).collect();

    // first each patch with a band round it, to the smooth surface the tube
    // would have there, the band widened (2, 4, 8 .. rings, up to the
    // hole's radius) until the rings below that surface — the lip — end
    // inside it; then the patches with only their lips, the rest of the
    // bands back where they were
    lap("patches", &mut times);
    let mut lips: Vec<Option<Vec<u32>>> = decisions
        .iter()
        .map(|&d| (d != Decision::Filled).then(Vec::new))
        .collect();
    let mut iterations = 0;
    let mut widest = 2;
    loop {
        let open: Vec<usize> = (0..loops.len()).filter(|&i| lips[i].is_none()).collect();
        if open.is_empty() {
            break;
        }
        let bands: Vec<(Vec<Vec<u32>>, bool)> = open
            .par_iter()
            .map(|&i| {
                band(
                    &loops[i], verts, faces, &topo, &surface, &body, &pinned, med, widest,
                )
            })
            .collect();
        let mut free: Vec<u32> = bands
            .iter()
            .flat_map(|(b, _)| b.iter().flatten())
            .copied()
            .collect();
        for &i in &open {
            free.extend(vspans[i].clone());
        }
        free.sort_unstable();
        free.dedup();
        iterations += fair(
            &mut out_verts,
            &free,
            &out_faces,
            faces,
            &topo,
            &surface,
            &pinned,
        );
        let normal = vertex_normals(&out_verts, &free, nv, &out_faces, faces, &topo, &surface);
        for (&i, (b, last)) in open.iter().zip(&bands) {
            let (lip, rings) = lip(b, verts, &out_verts, &normal);
            if rings < b.len() || *last {
                lips[i] = Some(lip);
            }
        }
        for &v in free.iter().filter(|&&v| (v as usize) < nv) {
            out_verts[v as usize] = verts[v as usize];
        }
        widest *= 2;
    }
    let lips: Vec<Vec<u32>> = lips.into_iter().map(Option::unwrap_or_default).collect();
    lap("bands", &mut times);
    let mut free: Vec<u32> = lips.iter().flatten().copied().collect();
    free.extend_from_slice(&new);
    free.sort_unstable();
    free.dedup();
    iterations += fair(
        &mut out_verts,
        &free,
        &out_faces,
        faces,
        &topo,
        &surface,
        &pinned,
    );
    lap("lips", &mut times);
    let n_moved = free
        .iter()
        .filter(|&&v| (v as usize) < nv && out_verts[v as usize] != verts[v as usize])
        .count();

    let holes = loops
        .iter()
        .zip(&decisions)
        .zip(spans.into_iter().zip(&patches).zip(&lips))
        .map(|((lp, &d), ((faces, p), lip))| Hole {
            verts: lp.verts.clone(),
            decision: d,
            faces,
            n_new: p.as_ref().map_or(0, |p| p.new.len()),
            n_lip: lip.len(),
        })
        .collect();
    Filled {
        verts: out_verts,
        faces: out_faces,
        holes,
        n_moved,
        iterations,
        times,
    }
}

/// Unit normals at the vertices `at`, from the surface's faces and the
/// patches' round each (areas weighting), at positions `pos`.
fn vertex_normals(
    pos: &[V3],
    at: &[u32],
    nv: usize,
    patch: &[[u32; 3]],
    faces: &[[u32; 3]],
    topo: &Topology,
    surface: &[bool],
) -> HashMap<u32, V3> {
    let mut sum: HashMap<u32, V3> = at.iter().map(|&v| (v, [0.0; 3])).collect();
    let mut add_face = |f: [u32; 3], v: u32| {
        let [a, b, c] = f.map(|w| pos[w as usize]);
        if let Some(s) = sum.get_mut(&v) {
            *s = add(*s, cross(sub(b, a), sub(c, a)));
        }
    };
    for &f in patch {
        for v in f {
            add_face(f, v);
        }
    }
    for &v in at.iter().filter(|&&v| (v as usize) < nv) {
        for &f in topo.vertex_faces(v as usize) {
            if surface[f as usize] {
                add_face(faces[f as usize], v);
            }
        }
    }
    sum.into_iter().map(|(v, s)| (v, unit(s))).collect()
}

/// A hole's patch: Liepa's triangulation of its loop, refined.
fn patch(
    lp: &Loop,
    verts: &[V3],
    faces: &[[u32; 3]],
    topo: &Topology,
    surface: &[bool],
    med: f64,
) -> Option<Patch> {
    let n = lp.verts.len();
    let p: Vec<V3> = lp.verts.iter().map(|&v| verts[v as usize]).collect();
    // the surface's normal along each edge, as the loop runs
    let outer: Vec<V3> = (0..n)
        .map(|i| {
            let c = third(
                faces[lp.faces[i] as usize],
                lp.verts[i],
                lp.verts[(i + 1) % n],
            );
            unit(cross(
                sub(p[(i + 1) % n], p[i]),
                sub(verts[c as usize], p[i]),
            ))
        })
        .collect();
    // the surface's own edges between vertices of the loop not next on it
    let at: HashMap<u32, usize> = lp.verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut banned: HashSet<(usize, usize)> = HashSet::new();
    for (i, &v) in lp.verts.iter().enumerate() {
        for &f in topo.vertex_faces(v as usize) {
            if !surface[f as usize] {
                continue;
            }
            for w in faces[f as usize] {
                if let Some(&j) = at.get(&w)
                    && j != i
                    && (j + 1) % n != i
                    && (i + 1) % n != j
                {
                    banned.insert((i.min(j), i.max(j)));
                }
            }
        }
    }
    let tris = triangulate(&p, &outer, &banned)?;
    Some(refine(p, tris, &banned, med))
}

/// Liepa's triangulation of the loop through `p`, `outer[i]` the unit
/// normal of the surface face along edge `(i, i + 1)` as the loop runs:
/// triangles `[k, m, i]` (`i < m < k`, wound against the loop, as a face
/// across it from the surface is) whose largest dihedral angle — with each
/// other and with the surface across the loop — is least, then whose area
/// is; `None` when every way uses a `banned` edge or spans no area.
fn triangulate(p: &[V3], outer: &[V3], banned: &HashSet<(usize, usize)>) -> Option<Vec<[u32; 3]>> {
    const NONE: u32 = u32::MAX;
    let n = p.len();
    let face = |i: usize, m: usize, k: usize| -> Option<(V3, f64)> {
        let c = cross(sub(p[m], p[k]), sub(p[i], p[k]));
        let l = norm(c);
        (l > 0.0).then(|| (scale(c, 1.0 / l), 0.5 * l))
    };
    let at = |i: usize, k: usize| i * n + k;
    // per span (i, k): the largest fold (1 - cos) and the area of the best
    // triangulation of the loop from i to k, its apex over (i, k), and the
    // normal of the triangle there
    #[derive(Clone, Copy)]
    struct Best {
        worst: f64,
        area: f64,
        apex: u32,
        normal: V3,
    }
    const UNSET: Best = Best {
        worst: f64::INFINITY,
        area: f64::INFINITY,
        apex: NONE,
        normal: [0.0; 3],
    };
    let mut table = vec![UNSET; n * n];
    for i in 0..n - 1 {
        table[at(i, i + 1)] = Best {
            worst: 0.0,
            area: 0.0,
            ..UNSET
        };
    }
    // each span from the shorter ones; a diagonal's spans are apart, so a
    // long one is shared out over the threads
    for len in 2..n {
        let span = |i: usize| -> Best {
            let k = i + len;
            let closing = i == 0 && k == n - 1;
            let mut best = UNSET;
            if !closing && banned.contains(&(i, k)) {
                return best;
            }
            for m in i + 1..k {
                let (a, b) = (&table[at(i, m)], &table[at(m, k)]);
                if !a.worst.is_finite() || !b.worst.is_finite() {
                    continue;
                }
                let Some((nt, area)) = face(i, m, k) else {
                    continue;
                };
                let left = if m == i + 1 { outer[i] } else { a.normal };
                let right = if k == m + 1 { outer[m] } else { b.normal };
                let mut worst = a
                    .worst
                    .max(b.worst)
                    .max(1.0 - dot(nt, left))
                    .max(1.0 - dot(nt, right));
                if closing {
                    worst = worst.max(1.0 - dot(nt, outer[n - 1]));
                }
                let total = a.area + b.area + area;
                if worst < best.worst || (worst == best.worst && total < best.area) {
                    best = Best {
                        worst,
                        area: total,
                        apex: m as u32,
                        normal: nt,
                    };
                }
            }
            best
        };
        let row: Vec<Best> = if (n - len) * len >= threads::MIN_PARALLEL {
            (0..n - len).into_par_iter().map(span).collect()
        } else {
            (0..n - len).map(span).collect()
        };
        for (i, b) in row.into_iter().enumerate() {
            table[at(i, i + len)] = b;
        }
    }
    if table[at(0, n - 1)].apex == NONE {
        return None;
    }
    let mut tris = Vec::with_capacity(n - 2);
    let mut todo = vec![(0usize, n - 1)];
    while let Some((i, k)) = todo.pop() {
        if k < i + 2 {
            continue;
        }
        let m = table[at(i, k)].apex as usize;
        tris.push([k as u32, m as u32, i as u32]);
        todo.push((i, m));
        todo.push((m, k));
    }
    Some(tris)
}

/// A patch's triangles, wound, and per directed edge the triangle on it.
struct Tris {
    tris: Vec<[u32; 3]>,
    on: HashMap<(u32, u32), usize>,
}

impl Tris {
    fn new(tris: Vec<[u32; 3]>) -> Self {
        let mut on = HashMap::new();
        for (t, &[a, b, c]) in tris.iter().enumerate() {
            on.insert((a, b), t);
            on.insert((b, c), t);
            on.insert((c, a), t);
        }
        Self { tris, on }
    }

    /// Triangle `t` made `tri` (`t` past the end: appended).
    fn put(&mut self, mut t: usize, tri: [u32; 3]) {
        if let Some(&[a, b, c]) = self.tris.get(t) {
            for e in [(a, b), (b, c), (c, a)] {
                if self.on.get(&e) == Some(&t) {
                    self.on.remove(&e);
                }
            }
            self.tris[t] = tri;
        } else {
            t = self.tris.len();
            self.tris.push(tri);
        }
        let [a, b, c] = tri;
        self.on.insert((a, b), t);
        self.on.insert((b, c), t);
        self.on.insert((c, a), t);
    }

    /// Swap edge `u`–`v` for the other diagonal of its two triangles when
    /// it fails the Delaunay criterion (its two opposite angles sum past
    /// π), unless it is on the patch's border, or the diagonal is an edge
    /// already or a `banned` one between loop vertices (below `n_loop`).
    fn swap(
        &mut self,
        u: u32,
        v: u32,
        pos: &[V3],
        n_loop: u32,
        banned: &HashSet<(usize, usize)>,
    ) -> bool {
        let (Some(&t1), Some(&t2)) = (self.on.get(&(u, v)), self.on.get(&(v, u))) else {
            return false;
        };
        let c = third(self.tris[t1], u, v);
        let d = third(self.tris[t2], v, u);
        if c == d || self.on.contains_key(&(c, d)) || self.on.contains_key(&(d, c)) {
            return false;
        }
        if c < n_loop && d < n_loop && banned.contains(&((c.min(d)) as usize, (c.max(d)) as usize))
        {
            return false;
        }
        let (pu, pv) = (pos[u as usize], pos[v as usize]);
        let (pc, pd) = (pos[c as usize], pos[d as usize]);
        if angle(sub(pu, pc), sub(pv, pc)) + angle(sub(pu, pd), sub(pv, pd)) <= PI + 1e-12 {
            return false;
        }
        self.put(t1, [u, d, c]);
        self.put(t2, [v, c, d]);
        true
    }
}

/// Liepa's refinement of a patch over the loop through `loop_pos`: each
/// triangle whose centroid lies farther than `med / √2` from all three of
/// its corners is split there, its edges swapped to the Delaunay criterion,
/// and once a pass splits nothing more, every edge is; until none splits.
fn refine(
    loop_pos: Vec<V3>,
    tris: Vec<[u32; 3]>,
    banned: &HashSet<(usize, usize)>,
    med: f64,
) -> Patch {
    let n_loop = loop_pos.len() as u32;
    let mut pos = loop_pos;
    let mut t = Tris::new(tris);
    let reach = med / SQRT_2;
    loop {
        let mut split = false;
        for i in 0..t.tris.len() {
            let [a, b, c] = t.tris[i];
            let (pa, pb, pc) = (pos[a as usize], pos[b as usize], pos[c as usize]);
            let mid = scale(add(add(pa, pb), pc), 1.0 / 3.0);
            if [pa, pb, pc].iter().all(|&q| norm(sub(mid, q)) > reach) {
                let x = pos.len() as u32;
                pos.push(mid);
                t.put(i, [x, b, c]);
                t.put(usize::MAX, [a, x, c]);
                t.put(usize::MAX, [a, b, x]);
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    t.swap(u, v, &pos, n_loop, banned);
                }
                split = true;
            }
        }
        if !split {
            break;
        }
        // relax every edge until none swaps; a flip cycle on a curved
        // patch is cut off after as many passes as there are edges
        let mut passes = t.on.len();
        loop {
            let mut edges: Vec<(u32, u32)> = t.on.keys().copied().filter(|&(u, v)| u < v).collect();
            edges.sort_unstable();
            let mut swapped = false;
            for (u, v) in edges {
                swapped |= t.swap(u, v, &pos, n_loop, banned);
            }
            passes -= 1;
            if !swapped || passes == 0 {
                break;
            }
        }
    }
    Patch {
        new: pos.split_off(n_loop as usize),
        tris: t.tris,
    }
}

/// The band round a hole, as rings of vertices that may move, at most
/// `widest` of them: ring 0 its loop, each next one the vertices of the
/// body's surface faces at the last ring not reached before, while the ring
/// lies within the hole's radius (its perimeter over 2π) of the loop,
/// `med` a ring; pinned vertices left out. And whether that is as wide as
/// it gets.
#[allow(clippy::too_many_arguments)]
fn band(
    lp: &Loop,
    verts: &[V3],
    faces: &[[u32; 3]],
    topo: &Topology,
    surface: &[bool],
    body: &[bool],
    pinned: &[bool],
    med: f64,
    widest: usize,
) -> (Vec<Vec<u32>>, bool) {
    let n = lp.verts.len();
    let radius = (0..n)
        .map(|i| {
            norm(sub(
                verts[lp.verts[(i + 1) % n] as usize],
                verts[lp.verts[i] as usize],
            ))
        })
        .sum::<f64>()
        / (2.0 * PI);
    let mut seen: HashSet<u32> = lp.verts.iter().copied().collect();
    let mut ring: Vec<u32> = lp.verts.clone();
    let mut rings = Vec::new();
    for k in 0.. {
        if ring.is_empty() || k as f64 * med >= radius {
            return (rings, true);
        }
        if k == widest {
            return (rings, false);
        }
        let next: Vec<u32> = ring
            .iter()
            .flat_map(|&v| topo.vertex_faces(v as usize).iter().copied())
            .filter(|&f| surface[f as usize] && body[f as usize])
            .flat_map(|f| faces[f as usize])
            .filter(|&v| seen.insert(v))
            .collect();
        rings.push(ring.into_iter().filter(|&v| !pinned[v as usize]).collect());
        ring = next;
    }
    unreachable!()
}

/// A hole's lip, out of its `band`: the loop itself — where the organelle
/// happened to end, not where the tube does — and the rings out from it
/// whose vertices lie, on average, below the smooth surface the band was
/// faired to (`smooth`, `normal` there), where the surface dips into the
/// mouth rather than going on as the tube. Its vertices, and its rings.
fn lip(
    band: &[Vec<u32>],
    verts: &[V3],
    smooth: &[V3],
    normal: &HashMap<u32, V3>,
) -> (Vec<u32>, usize) {
    let mut out = Vec::new();
    for (k, ring) in band.iter().enumerate() {
        let depth: f64 = ring
            .iter()
            .map(|&v| {
                let n = normal.get(&v).copied().unwrap_or([0.0; 3]);
                dot(sub(verts[v as usize], smooth[v as usize]), n)
            })
            .sum();
        if ring.is_empty() || (k > 0 && depth >= 0.0) {
            return (out, k);
        }
        out.extend_from_slice(ring);
    }
    (out, band.len())
}

/// One equation of the fairing: the umbrella Laplacian at a vertex — the
/// mean of its neighbours less itself — over the free vertices (`cols`,
/// with coefficients), plus what the held ones add (`known`).
struct Row {
    cols: Vec<(usize, f64)>,
    known: V3,
    /// a held vertex is in it
    held: bool,
}

/// Move the `free` vertices (ascending) of `pos` to the least sum of
/// squared umbrella Laplacians — at each free vertex, and at each held one
/// beside them whose ring is whole (not on a loop left open) — over the
/// surface's faces and the `patch` faces; each set of free vertices that
/// share an equation solved apart, by conjugate gradients on the normal
/// equations (CGLS). A set with no held vertex beside it stays. Returns
/// the iterations.
fn fair(
    pos: &mut [V3],
    free: &[u32],
    patch: &[[u32; 3]],
    faces: &[[u32; 3]],
    topo: &Topology,
    surface: &[bool],
    pinned: &[bool],
) -> usize {
    if free.is_empty() {
        return 0;
    }
    let nv = topo.n_vertices();
    let mut patch_nbrs: HashMap<u32, Vec<u32>> = HashMap::new();
    for &[a, b, c] in patch {
        for (u, v, w) in [(a, b, c), (b, c, a), (c, a, b)] {
            let e = patch_nbrs.entry(u).or_default();
            e.push(v);
            e.push(w);
        }
    }
    let nbrs = |v: u32| -> Vec<u32> {
        let mut out: Vec<u32> = patch_nbrs.get(&v).cloned().unwrap_or_default();
        if (v as usize) < nv {
            for &f in topo.vertex_faces(v as usize) {
                if surface[f as usize] {
                    out.extend(faces[f as usize].into_iter().filter(|&w| w != v));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    };
    let index: HashMap<u32, usize> = free.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    // the equations: at each free vertex, and at each held one beside them
    let mut centres: Vec<u32> = free.to_vec();
    let mut beside: Vec<u32> = Vec::new();
    for &v in free {
        for w in nbrs(v) {
            if !index.contains_key(&w) && !((w as usize) < nv && pinned[w as usize]) {
                beside.push(w);
            }
        }
    }
    beside.sort_unstable();
    beside.dedup();
    centres.extend(beside);
    let rows: Vec<Row> = centres
        .par_iter()
        .map(|&r| {
            let around = nbrs(r);
            let share = 1.0 / around.len().max(1) as f64;
            let mut cols = Vec::with_capacity(around.len() + 1);
            let mut known = [0.0; 3];
            let mut held = false;
            for w in around {
                match index.get(&w) {
                    Some(&j) => cols.push((j, share)),
                    None => {
                        known = add(known, scale(pos[w as usize], share));
                        held = true;
                    }
                }
            }
            match index.get(&r) {
                Some(&j) => cols.push((j, -1.0)),
                None => {
                    known = sub(known, pos[r as usize]);
                    held = true;
                }
            }
            Row { cols, known, held }
        })
        .collect();
    // the free vertices sharing equations, and each set's equations
    let sets = LowestRootSets::new(free.len());
    for row in &rows {
        if let Some(&(first, _)) = row.cols.first() {
            for &(j, _) in &row.cols[1..] {
                sets.join(first as u32, j as u32);
            }
        }
    }
    let root = sets.roots();
    let mut held = vec![false; free.len()];
    let mut eqs: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        let Some(&(first, _)) = row.cols.first() else {
            continue;
        };
        let r = root[first];
        eqs.entry(r).or_default().push(i);
        held[r as usize] |= row.held;
    }
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for (j, &r) in root.iter().enumerate() {
        members.entry(r).or_default().push(j);
    }
    let mut systems: Vec<(Vec<usize>, Vec<usize>)> = members
        .into_iter()
        .filter(|(r, _)| held[*r as usize])
        .map(|(r, m)| (m, eqs.remove(&r).unwrap_or_default()))
        .collect();
    systems.sort_unstable_by_key(|(m, _)| m[0]);
    let start: Vec<V3> = free.iter().map(|&v| pos[v as usize]).collect();
    let solved: Vec<(Vec<usize>, Vec<V3>, usize)> = systems
        .par_iter()
        .map(|(m, e)| {
            let (x, it) = cgls(m, e, &rows, &start);
            (m.clone(), x, it)
        })
        .collect();
    let mut iterations = 0;
    for (m, x, it) in solved {
        iterations += it;
        for (&j, xj) in m.iter().zip(x) {
            pos[free[j] as usize] = xj;
        }
    }
    iterations
}

/// Least squares over the free vertices `members` of the equations `eqs`
/// (rows of `rows`), from `start`, by CGLS — each coordinate on its own —
/// until the normal equations' residual falls below 1e-10 of where it
/// began, or after as many iterations as unknowns, ten times over.
fn cgls(members: &[usize], eqs: &[usize], rows: &[Row], start: &[V3]) -> (Vec<V3>, usize) {
    let local: HashMap<usize, usize> = members.iter().enumerate().map(|(i, &j)| (j, i)).collect();
    let a: Vec<Vec<(usize, f64)>> = eqs
        .iter()
        .map(|&r| rows[r].cols.iter().map(|&(j, w)| (local[&j], w)).collect())
        .collect();
    let b: Vec<V3> = eqs.iter().map(|&r| scale(rows[r].known, -1.0)).collect();
    let n = members.len();
    let times = |x: &[V3]| -> Vec<V3> {
        a.iter()
            .map(|row| {
                row.iter()
                    .fold([0.0; 3], |s, &(j, w)| add(s, scale(x[j], w)))
            })
            .collect()
    };
    let times_t = |y: &[V3]| -> Vec<V3> {
        let mut out = vec![[0.0; 3]; n];
        for (row, &yr) in a.iter().zip(y) {
            for &(j, w) in row {
                out[j] = add(out[j], scale(yr, w));
            }
        }
        out
    };
    let sq = |v: &[V3]| -> V3 {
        v.iter().fold([0.0; 3], |s, q| {
            add(s, [q[0] * q[0], q[1] * q[1], q[2] * q[2]])
        })
    };
    let mut x: Vec<V3> = members.iter().map(|&j| start[j]).collect();
    let ax = times(&x);
    let mut r: Vec<V3> = b.iter().zip(&ax).map(|(&bi, &ai)| sub(bi, ai)).collect();
    let mut s = times_t(&r);
    let mut p = s.clone();
    let mut gamma = sq(&s);
    let gamma0 = gamma;
    let mut it = 0;
    while it < 10 * n {
        if (0..3).all(|c| gamma[c] <= 1e-20 * gamma0[c]) {
            break;
        }
        it += 1;
        let q = times(&p);
        let delta = sq(&q);
        let alpha: V3 = std::array::from_fn(|c| {
            if delta[c] > 0.0 {
                gamma[c] / delta[c]
            } else {
                0.0
            }
        });
        for (xi, pi) in x.iter_mut().zip(&p) {
            *xi = add(*xi, [alpha[0] * pi[0], alpha[1] * pi[1], alpha[2] * pi[2]]);
        }
        for (ri, qi) in r.iter_mut().zip(&q) {
            *ri = sub(*ri, [alpha[0] * qi[0], alpha[1] * qi[1], alpha[2] * qi[2]]);
        }
        s = times_t(&r);
        let next = sq(&s);
        let beta: V3 = std::array::from_fn(|c| {
            if gamma[c] > 0.0 {
                next[c] / gamma[c]
            } else {
                0.0
            }
        });
        for (pi, si) in p.iter_mut().zip(&s) {
            *pi = [
                si[0] + beta[0] * pi[0],
                si[1] + beta[1] * pi[1],
                si[2] + beta[2] * pi[2],
            ];
        }
        gamma = next;
    }
    (x, it)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tube round the z axis, `rings` rings of `around` vertices `step`
    /// apart, radius `r`, its faces wound outwards; open at both ends.
    fn tube(around: usize, rings: usize, r: f64, step: f64) -> (Vec<V3>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        for j in 0..rings {
            for i in 0..around {
                let t = 2.0 * PI * (i as f64 + 0.5 * (j % 2) as f64) / around as f64;
                verts.push([r * t.cos(), r * t.sin(), j as f64 * step]);
            }
        }
        let mut faces = Vec::new();
        for j in 0..rings - 1 {
            for i in 0..around {
                let a = (j * around + i) as u32;
                let b = (j * around + (i + 1) % around) as u32;
                let c = ((j + 1) * around + i) as u32;
                let d = ((j + 1) * around + (i + 1) % around) as u32;
                if j % 2 == 0 {
                    faces.push([a, b, c]);
                    faces.push([b, d, c]);
                } else {
                    faces.push([a, b, d]);
                    faces.push([a, d, c]);
                }
            }
        }
        (verts, faces)
    }

    /// The faces whose centres lie within `half` radians and `z` of
    /// `(0, z0)` on the tube.
    fn window(verts: &[V3], faces: &[[u32; 3]], half: f64, z0: f64, z: f64) -> Vec<bool> {
        faces
            .iter()
            .map(|f| {
                let c = scale(
                    f.iter().fold([0.0; 3], |s, &v| add(s, verts[v as usize])),
                    1.0 / 3.0,
                );
                c[1].atan2(c[0]).abs() < half && (c[2] - z0).abs() < z
            })
            .collect()
    }

    fn rho(p: V3) -> f64 {
        p[0].hypot(p[1])
    }

    /// The vertices only organelle faces use, pushed `depth` towards the
    /// axis: the pocket behind the mouth.
    fn deepen(verts: &mut [V3], faces: &[[u32; 3]], org: &[bool], depth: f64) {
        let mut other = vec![false; verts.len()];
        for (f, &o) in faces.iter().zip(org) {
            if !o {
                for &v in f {
                    other[v as usize] = true;
                }
            }
        }
        for (v, p) in verts.iter_mut().enumerate() {
            if !other[v]
                && faces
                    .iter()
                    .zip(org)
                    .any(|(f, &o)| o && f.contains(&(v as u32)))
            {
                let k = 1.0 - depth / rho(*p);
                *p = [p[0] * k, p[1] * k, p[2]];
            }
        }
    }

    /// The surface and the patches together: their border's loops.
    fn open_after(faces: &[[u32; 3]], org: &[bool], done: &Filled) -> usize {
        let all: Vec<[u32; 3]> = faces.iter().chain(&done.faces).copied().collect();
        let topo = Topology::new(&all, done.verts.len());
        let member: Vec<bool> = (0..all.len())
            .map(|f| f >= faces.len() || !org[f])
            .collect();
        border_loops(&topo, &all, &member).len()
    }

    #[test]
    fn a_side_hole_is_patched_on_the_tube_and_its_ends_left_open() {
        let r = 10.0;
        let (mut verts, faces) = tube(63, 30, r, 1.0);
        let org = window(&verts, &faces, 0.35, 15.0, 3.0);
        deepen(&mut verts, &faces, &org, 3.0);
        let done = fill_holes(&verts, &faces, &org, &[], 1.0);
        let kinds: Vec<Decision> = done.holes.iter().map(|h| h.decision).collect();
        assert_eq!(
            kinds.iter().filter(|&&d| d == Decision::Filled).count(),
            1,
            "{kinds:?}"
        );
        assert_eq!(
            kinds.iter().filter(|&&d| d == Decision::TubeEnd).count(),
            2,
            "{kinds:?}"
        );
        let hole = done
            .holes
            .iter()
            .find(|h| h.decision == Decision::Filled)
            .unwrap();
        assert!(
            hole.n_new > 10,
            "the patch has inner vertices: {}",
            hole.n_new
        );
        // the two ends are all that is open
        assert_eq!(open_after(&faces, &org, &done), 2);
        // on the tube, not across it: to within a tenth of an edge
        for p in &done.verts[verts.len()..] {
            assert!((rho(*p) - r).abs() < 0.1, "off the tube: {}", rho(*p));
        }
        // edges about the median's length
        for f in &done.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                let l = norm(sub(done.verts[a as usize], done.verts[b as usize]));
                assert!(l > 0.3 && l < 2.0, "edge {l}");
            }
        }
    }

    #[test]
    fn a_rim_dipping_into_its_mouth_comes_back_out_to_the_tube() {
        let r = 10.0;
        let (mut verts, faces) = tube(63, 30, r, 1.0);
        let org = window(&verts, &faces, 0.35, 15.0, 3.0);
        deepen(&mut verts, &faces, &org, 3.0);
        // the rim, and the ring out from it, pulled in: a funnel
        let mut on_org = vec![false; verts.len()];
        let mut on_surf = vec![false; verts.len()];
        for (f, &o) in faces.iter().zip(&org) {
            for &v in f {
                if o {
                    on_org[v as usize] = true;
                } else {
                    on_surf[v as usize] = true;
                }
            }
        }
        let rim: Vec<usize> = (0..verts.len())
            .filter(|&v| on_org[v] && on_surf[v])
            .collect();
        let mut next = vec![false; verts.len()];
        for f in faces.iter().zip(&org).filter(|(_, o)| !**o).map(|(f, _)| f) {
            if f.iter().any(|&v| rim.contains(&(v as usize))) {
                for &v in f {
                    next[v as usize] = !rim.contains(&(v as usize));
                }
            }
        }
        for v in 0..verts.len() {
            let pull = if rim.contains(&v) {
                0.6
            } else if next[v] {
                0.3
            } else {
                continue;
            };
            let p = verts[v];
            let k = 1.0 - pull / rho(p);
            verts[v] = [p[0] * k, p[1] * k, p[2]];
        }
        let done = fill_holes(&verts, &faces, &org, &[], 1.0);
        let hole = done
            .holes
            .iter()
            .find(|h| h.decision == Decision::Filled)
            .unwrap();
        let pulled = rim.len() + next.iter().filter(|&&x| x).count();
        assert!(
            hole.n_lip >= pulled,
            "the lip moves: {} of {pulled}",
            hole.n_lip
        );
        // back out to the tube, to within a tenth of an edge
        for v in (0..verts.len()).filter(|&v| rim.contains(&v) || next[v]) {
            let off = (r - rho(done.verts[v])).abs();
            assert!(
                off < 0.1,
                "lip vertex {v}: {} in after",
                r - rho(done.verts[v])
            );
        }
        for p in &done.verts[verts.len()..] {
            assert!((rho(*p) - r).abs() < 0.1, "off the tube: {}", rho(*p));
        }
    }

    #[test]
    fn a_hole_the_soma_touches_stays_open() {
        let (mut verts, faces) = tube(63, 30, 10.0, 1.0);
        let org = window(&verts, &faces, 0.35, 15.0, 3.0);
        deepen(&mut verts, &faces, &org, 3.0);
        let mut soma = vec![false; verts.len()];
        let edge = faces
            .iter()
            .zip(&org)
            .filter(|(_, o)| !**o)
            .flat_map(|(f, _)| f.iter().copied())
            .find(|&v| faces.iter().zip(&org).any(|(f, &o)| o && f.contains(&v)))
            .unwrap();
        soma[edge as usize] = true;
        let done = fill_holes(&verts, &faces, &org, &soma, 1.0);
        assert!(done.faces.is_empty());
        assert!(done.holes.iter().any(|h| h.decision == Decision::Soma));
        assert_eq!(done.verts, verts);
    }

    #[test]
    fn an_island_in_an_organelle_is_not_patched_but_the_hole_round_it_is() {
        let (mut verts, faces) = tube(63, 30, 10.0, 1.0);
        let wide = window(&verts, &faces, 0.5, 15.0, 5.0);
        let inner = window(&verts, &faces, 0.12, 15.0, 1.2);
        let org: Vec<bool> = wide.iter().zip(&inner).map(|(&w, &i)| w && !i).collect();
        deepen(&mut verts, &faces, &org, 1.0);
        let done = fill_holes(&verts, &faces, &org, &[], 1.0);
        let kinds: Vec<Decision> = done.holes.iter().map(|h| h.decision).collect();
        assert!(kinds.contains(&Decision::OffBody), "{kinds:?}");
        assert_eq!(
            kinds.iter().filter(|&&d| d == Decision::Filled).count(),
            1,
            "{kinds:?}"
        );
    }

    #[test]
    fn loops_pass_no_vertex_twice_and_run_as_their_faces() {
        let (verts, faces) = tube(63, 30, 10.0, 1.0);
        let org = window(&verts, &faces, 0.35, 15.0, 3.0);
        let topo = Topology::new(&faces, verts.len());
        let member: Vec<bool> = org.iter().map(|&o| !o).collect();
        for lp in border_loops(&topo, &faces, &member) {
            let mut seen = lp.verts.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), lp.verts.len());
            let n = lp.verts.len();
            let along = (0..n)
                .filter(|&i| {
                    runs(
                        faces[lp.faces[i] as usize],
                        lp.verts[i],
                        lp.verts[(i + 1) % n],
                    )
                })
                .count();
            assert_eq!(along, n);
        }
    }
}
