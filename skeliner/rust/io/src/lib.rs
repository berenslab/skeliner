//! `skeliner.io`'s mesh readers: OBJ as trimesh reads it
//! (`trimesh.load_mesh(path, process=False)`), for the files of plain
//! vertices and triangles skeliner's pipeline writes and reads; and
//! OpenCTM, only for reading EyeWire 1's meshes
//! (<https://drive.google.com/file/d/1-qmE1sEL3AIGpNfi0w50MEOgm2WIF4_1/view>).
//!
//! Each reader returns `None` for anything it does not read, so the caller
//! hands that file to trimesh instead.

#![forbid(unsafe_code)]

/// A triangle mesh: its vertices, and faces naming them.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// A decimal number as trimesh's parser reads it, or `None` for anything
/// else (`inf`, `nan`, hex, a stray character), which trimesh may read
/// differently.
fn decimal(token: &str) -> Option<f64> {
    let ok = !token.is_empty()
        && token
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.' | b'e' | b'E'))
        && token.bytes().any(|b| b.is_ascii_digit());
    if ok { token.parse().ok() } else { None }
}

/// A 1-based OBJ index as a 0-based one; `None` for zero, a relative
/// (negative) index or anything not a plain number.
fn obj_index(token: &str) -> Option<u32> {
    if token.is_empty() || !token.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    token.parse::<u32>().ok()?.checked_sub(1)
}

/// Only the vertices some face names, in their order, the faces renumbered:
/// what trimesh's OBJ loader keeps.
fn referenced_only(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> Mesh {
    let mut used = vec![false; vertices.len()];
    for &v in faces.iter().flatten() {
        used[v as usize] = true;
    }
    if used.iter().all(|&u| u) {
        return Mesh { vertices, faces };
    }
    let mut new_id = vec![0u32; vertices.len()];
    let mut kept = Vec::new();
    for (i, v) in vertices.into_iter().enumerate() {
        if used[i] {
            new_id[i] = kept.len() as u32;
            kept.push(v);
        }
    }
    let faces = faces
        .into_iter()
        .map(|f| f.map(|v| new_id[v as usize]))
        .collect();
    Mesh {
        vertices: kept,
        faces,
    }
}

/// An OBJ of `v x y z` and triangle `f a b c` lines (1-based indices), with
/// comments and blank lines between and at most one `o` line before the
/// first face (trimesh's own export names its object): vertices no face
/// names are dropped, as trimesh does. `None` for anything more — normals,
/// texture coordinates, more objects, groups, materials, polygons, slashes,
/// relative or out-of-range indices, extra vertex values, no faces at all.
pub fn read_obj(text: &[u8]) -> Option<Mesh> {
    if !text.is_ascii() || text.contains(&b'\\') {
        return None;
    }
    let text = std::str::from_utf8(text).ok()?;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut named = false;
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.split_ascii_whitespace().next().is_none() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("o ") {
            // one object for every face, or trimesh splits the mesh
            if named || !faces.is_empty() {
                return None;
            }
            named = true;
        } else if let Some(rest) = line.strip_prefix("v ") {
            let mut it = rest.split_ascii_whitespace();
            let v = [
                decimal(it.next()?)?,
                decimal(it.next()?)?,
                decimal(it.next()?)?,
            ];
            if it.next().is_some() {
                return None;
            }
            vertices.push(v);
        } else {
            // a triangle, or nothing this reader reads
            let rest = line.strip_prefix("f ")?;
            let mut it = rest.split_ascii_whitespace();
            let f = [
                obj_index(it.next()?)?,
                obj_index(it.next()?)?,
                obj_index(it.next()?)?,
            ];
            if it.next().is_some() {
                return None;
            }
            faces.push(f);
        }
    }
    let n = vertices.len();
    if faces.is_empty() || faces.iter().flatten().any(|&v| v as usize >= n) {
        return None;
    }
    Some(referenced_only(vertices, faces))
}

/// Why an OpenCTM file could not be read: it is malformed, or a packed
/// block would not unpack.
#[derive(Debug, PartialEq)]
pub enum CtmError<E> {
    Malformed(&'static str),
    Unpack(E),
}

/// An OpenCTM file, read front to back.
struct Ctm<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Ctm<'a> {
    fn take<E>(&mut self, n: usize) -> Result<&'a [u8], CtmError<E>> {
        let out = self
            .data
            .get(self.pos..self.pos + n)
            .ok_or(CtmError::Malformed("the CTM file ends early"))?;
        self.pos += n;
        Ok(out)
    }

    fn u32<E>(&mut self) -> Result<u32, CtmError<E>> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().expect("four bytes")))
    }

    fn f32<E>(&mut self) -> Result<f32, CtmError<E>> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn count<E>(&mut self) -> Result<usize, CtmError<E>> {
        let n = self.u32()?;
        i32::try_from(n)
            .map(|n| n as usize)
            .map_err(|_| CtmError::Malformed("a CTM count is negative"))
    }

    fn section<E>(&mut self, tag: &[u8; 4]) -> Result<(), CtmError<E>> {
        if self.take(4)? != tag {
            return Err(CtmError::Malformed(
                "a CTM section is missing or out of place",
            ));
        }
        Ok(())
    }

    /// A packed block of `count` × `size` 32-bit words: its packed length,
    /// five LZMA property bytes and the LZMA1 data, unpacked by `unlzma`
    /// into four byte planes, most significant first, each plane component
    /// by component.
    fn packed<E>(
        &mut self,
        count: usize,
        size: usize,
        unlzma: &mut impl FnMut(&[u8], &[u8], usize) -> Result<Vec<u8>, E>,
    ) -> Result<Vec<u32>, CtmError<E>> {
        let n = self.u32()? as usize;
        let props = self.take(5)?;
        let packed = self.take(n)?;
        let total = count * size;
        let raw = unlzma(props, packed, 4 * total).map_err(CtmError::Unpack)?;
        if raw.len() != 4 * total {
            return Err(CtmError::Malformed("a CTM block unpacks to the wrong size"));
        }
        let mut words = vec![0u32; total];
        for i in 0..count {
            for k in 0..size {
                let at = i + k * count;
                words[i * size + k] = (u32::from(raw[at]) << 24)
                    | (u32::from(raw[at + total]) << 16)
                    | (u32::from(raw[at + 2 * total]) << 8)
                    | u32::from(raw[at + 3 * total]);
            }
        }
        Ok(words)
    }
}

/// MG1's and MG2's triangles from their deltas: the first index from the
/// previous triangle's first, the third from the first, the second from the
/// previous second where the first repeats and from the first otherwise —
/// in the reference library's `u32` arithmetic.
fn restore_indices(words: &[u32]) -> Vec<[u32; 3]> {
    let mut out: Vec<[u32; 3]> = Vec::with_capacity(words.len() / 3);
    for w in words.as_chunks::<3>().0 {
        let mut t = *w;
        let prev = out.last().copied();
        if let Some(p) = prev {
            t[0] = t[0].wrapping_add(p[0]);
        }
        t[2] = t[2].wrapping_add(t[0]);
        t[1] = match prev {
            Some(p) if t[0] == p[0] => t[1].wrapping_add(p[1]),
            _ => t[1].wrapping_add(t[0]),
        };
        out.push(t);
    }
    out
}

/// An OpenCTM v5 file's vertices and triangles, as the reference library
/// decodes them (vertices in single precision, as stored, then widened),
/// only for reading EyeWire 1's meshes
/// (<https://drive.google.com/file/d/1-qmE1sEL3AIGpNfi0w50MEOgm2WIF4_1/view>):
/// RAW, MG1 or MG2 compression, every vertex kept; normals, UV maps and
/// attribute maps, which those files do not have, are not read. `unlzma(props,
/// packed, size)` unpacks an LZMA1 block of `size` bytes given its five
/// property bytes. `None` for a file that is not OpenCTM v5 or uses another
/// compression.
pub fn read_ctm<E>(
    data: &[u8],
    unlzma: &mut impl FnMut(&[u8], &[u8], usize) -> Result<Vec<u8>, E>,
) -> Result<Option<Mesh>, CtmError<E>> {
    let mut f = Ctm { data, pos: 0 };
    if f.take::<E>(4).ok() != Some(&b"OCTM"[..]) || f.u32::<E>().ok() != Some(5) {
        return Ok(None);
    }
    let method: [u8; 4] = f.take(4)?.try_into().expect("four bytes");
    let n_vertices = f.count()?;
    let n_triangles = f.count()?;
    let _uv_maps = f.u32()?;
    let _attribute_maps = f.u32()?;
    let _flags = f.u32()?;
    let comment = f.count()?;
    f.take(comment)?;
    let (vertices, triangles): (Vec<[f32; 3]>, Vec<[u32; 3]>) = match &method {
        b"RAW\0" => {
            f.section(b"INDX")?;
            let mut triangles = Vec::with_capacity(n_triangles);
            for _ in 0..n_triangles {
                triangles.push([f.u32()?, f.u32()?, f.u32()?]);
            }
            f.section(b"VERT")?;
            let mut vertices = Vec::with_capacity(n_vertices);
            for _ in 0..n_vertices {
                vertices.push([f.f32()?, f.f32()?, f.f32()?]);
            }
            (vertices, triangles)
        }
        b"MG1\0" => {
            f.section(b"INDX")?;
            let triangles = restore_indices(&f.packed(n_triangles, 3, unlzma)?);
            f.section(b"VERT")?;
            let words = f.packed(n_vertices, 3, unlzma)?;
            let vertices = words
                .as_chunks::<3>()
                .0
                .iter()
                .map(|w| w.map(f32::from_bits))
                .collect();
            (vertices, triangles)
        }
        b"MG2\0" => {
            f.section(b"MG2H")?;
            let precision = f.f32()?;
            let _normal_precision = f.f32()?;
            let lo = [f.f32()?, f.f32()?, f.f32()?];
            let hi = [f.f32()?, f.f32()?, f.f32()?];
            let div = [f.u32()?, f.u32()?, f.u32()?];
            if div.contains(&0) {
                return Err(CtmError::Malformed("the CTM grid has a zero division"));
            }
            f.section(b"VERT")?;
            let ints = f.packed(n_vertices, 3, unlzma)?;
            f.section(b"GIDX")?;
            let deltas = f.packed(n_vertices, 1, unlzma)?;
            f.section(b"INDX")?;
            let triangles = restore_indices(&f.packed(n_triangles, 3, unlzma)?);
            // the grid: cells of (hi - lo) / div, numbered x fastest
            let size = [0, 1, 2].map(|k| (hi[k] - lo[k]) / div[k] as f32);
            let cells = u64::from(div[0]) * u64::from(div[1]) * u64::from(div[2]);
            let mut vertices = Vec::with_capacity(n_vertices);
            let (mut cell, mut prev_dx) = (0u64, 0i64);
            for i in 0..n_vertices {
                let prev_cell = cell;
                cell += u64::from(deltas[i]);
                if cell >= cells {
                    return Err(CtmError::Malformed("a CTM vertex lies outside its grid"));
                }
                let (dx, dy) = (u64::from(div[0]), u64::from(div[1]));
                let g = [cell % dx, (cell / dx) % dy, cell / (dx * dy)];
                let origin = [0, 1, 2].map(|k| g[k] as f32 * size[k] + lo[k]);
                // x is a delta from the previous vertex in the same cell
                let mut x = i64::from(ints[3 * i] as i32);
                if i > 0 && cell == prev_cell {
                    x += prev_dx;
                }
                prev_dx = x;
                let y = i64::from(ints[3 * i + 1] as i32);
                let z = i64::from(ints[3 * i + 2] as i32);
                vertices.push([
                    precision * x as f32 + origin[0],
                    precision * y as f32 + origin[1],
                    precision * z as f32 + origin[2],
                ]);
            }
            (vertices, triangles)
        }
        _ => return Ok(None),
    };
    if triangles
        .iter()
        .flatten()
        .any(|&v| v as usize >= n_vertices)
    {
        return Err(CtmError::Malformed(
            "a CTM triangle names a vertex past the last",
        ));
    }
    Ok(Some(Mesh {
        vertices: vertices.iter().map(|v| v.map(f64::from)).collect(),
        faces: triangles,
    }))
}

/// An OBJ of `vertices` and `faces` as trimesh's `export_obj` writes a
/// plain mesh (no normals, colours, texture or name): its header comment,
/// a `v` line per vertex with eight decimals (Python's `{:.8f}`, which
/// rounds half to even as Rust's does; NaN as `nan`), an `f` line per face
/// numbered from one, and a blank line to end.
pub fn write_obj(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<u8> {
    use std::fmt::Write;

    let mut out = String::with_capacity(36 + vertices.len() * 52 + faces.len() * 24);
    out.push_str("# https://github.com/mikedh/trimesh\nv ");
    let coord = |out: &mut String, x: f64| {
        if x.is_nan() {
            out.push_str("nan");
        } else {
            write!(out, "{x:.8}").expect("writing to a string");
        }
    };
    for (i, v) in vertices.iter().enumerate() {
        if i > 0 {
            out.push_str("\nv ");
        }
        coord(&mut out, v[0]);
        out.push(' ');
        coord(&mut out, v[1]);
        out.push(' ');
        coord(&mut out, v[2]);
    }
    out.push_str("\nf ");
    for (i, f) in faces.iter().enumerate() {
        if i > 0 {
            out.push_str("\nf ");
        }
        write!(out, "{} {} {}", f[0] + 1, f[1] + 1, f[2] + 1).expect("writing to a string");
    }
    out.push_str("\n\n");
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_obj_is_written_as_trimesh_writes_it() {
        let v = [[331737.0, -0.0, 1.0 / 512.0], [f64::NAN, 2.5, 1e20]];
        let text = String::from_utf8(write_obj(&v, &[[0, 1, 1]])).expect("utf-8");
        assert_eq!(
            text,
            "# https://github.com/mikedh/trimesh\n\
             v 331737.00000000 -0.00000000 0.00195312\n\
             v nan 2.50000000 100000000000000000000.00000000\n\
             f 1 2 2\n\n"
        );
        let empty = String::from_utf8(write_obj(&[], &[])).expect("utf-8");
        assert_eq!(empty, "# https://github.com/mikedh/trimesh\nv \nf \n\n");
        // read back as written; the empty one has a bare `f` line, which the
        // reader leaves to trimesh
        let tri = [[331737.0, 0.5, 2.25], [1.0, 2.0, 3.0], [-4.0, 0.125, 9.0]];
        let m = read_obj(&write_obj(&tri, &[[0, 1, 2]])).expect("plain OBJ");
        assert_eq!((m.vertices, m.faces), (tri.to_vec(), vec![[0, 1, 2]]));
        assert!(read_obj(&write_obj(&[], &[])).is_none());
    }

    use super::*;

    /// Words into the byte planes a packed CTM block unpacks to.
    fn planes(words: &[u32], count: usize, size: usize) -> Vec<u8> {
        let total = count * size;
        let mut raw = vec![0u8; 4 * total];
        for i in 0..count {
            for k in 0..size {
                let (w, at) = (words[i * size + k], i + k * count);
                raw[at] = (w >> 24) as u8;
                raw[at + total] = (w >> 16) as u8;
                raw[at + 2 * total] = (w >> 8) as u8;
                raw[at + 3 * total] = w as u8;
            }
        }
        raw
    }

    /// A packed block holding `raw` as it is: the tests unpack it unchanged.
    fn block(raw: &[u8]) -> Vec<u8> {
        let mut b = (raw.len() as u32).to_le_bytes().to_vec();
        b.extend_from_slice(&[0u8; 5]);
        b.extend_from_slice(raw);
        b
    }

    /// The MG1 / MG2 index deltas, as the reference encoder makes them.
    fn index_deltas(tris: &[[u32; 3]]) -> Vec<u32> {
        let mut w: Vec<u32> = tris.iter().flatten().copied().collect();
        for i in (0..tris.len()).rev() {
            if i >= 1 && w[3 * i] == w[3 * (i - 1)] {
                w[3 * i + 1] = w[3 * i + 1].wrapping_sub(w[3 * (i - 1) + 1]);
            } else {
                w[3 * i + 1] = w[3 * i + 1].wrapping_sub(w[3 * i]);
            }
            w[3 * i + 2] = w[3 * i + 2].wrapping_sub(w[3 * i]);
            if i >= 1 {
                w[3 * i] = w[3 * i].wrapping_sub(w[3 * (i - 1)]);
            }
        }
        w
    }

    fn ctm_header(method: &[u8; 4], nv: u32, nt: u32) -> Vec<u8> {
        let mut out = b"OCTM".to_vec();
        for word in [5u32, u32::from_le_bytes(*method), nv, nt, 0, 0, 0] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(b"note");
        out
    }

    fn unchanged(_props: &[u8], packed: &[u8], _size: usize) -> Result<Vec<u8>, ()> {
        Ok(packed.to_vec())
    }

    const TRIS: [[u32; 3]; 4] = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];

    #[test]
    fn a_ctm_reads_raw_and_mg1() {
        let verts = [
            [0.0f32, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, -3.25],
        ];
        let want = Mesh {
            vertices: verts.iter().map(|v| v.map(f64::from)).collect(),
            faces: TRIS.to_vec(),
        };
        let mut raw = ctm_header(b"RAW\0", 4, 4);
        raw.extend_from_slice(b"INDX");
        for v in TRIS.iter().flatten() {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        raw.extend_from_slice(b"VERT");
        for c in verts.iter().flatten() {
            raw.extend_from_slice(&c.to_le_bytes());
        }
        assert_eq!(read_ctm(&raw, &mut unchanged), Ok(Some(want.clone())));

        let mut mg1 = ctm_header(b"MG1\0", 4, 4);
        mg1.extend_from_slice(b"INDX");
        mg1.extend(block(&planes(&index_deltas(&TRIS), 4, 3)));
        mg1.extend_from_slice(b"VERT");
        let bits: Vec<u32> = verts.iter().flatten().map(|c| c.to_bits()).collect();
        mg1.extend(block(&planes(&bits, 4, 3)));
        assert_eq!(read_ctm(&mg1, &mut unchanged), Ok(Some(want)));
        // the block's size is checked, and an unpacking error comes back
        let mut wrong = |_: &[u8], p: &[u8], _: usize| Ok::<_, ()>(p[1..].to_vec());
        assert_eq!(
            read_ctm(&mg1, &mut wrong),
            Err(CtmError::Malformed("a CTM block unpacks to the wrong size"))
        );
        let mut failing = |_: &[u8], _: &[u8], _: usize| Err::<Vec<u8>, _>("lzma");
        assert_eq!(read_ctm(&mg1, &mut failing), Err(CtmError::Unpack("lzma")));
    }

    #[test]
    fn a_ctm_reads_mg2_vertices_off_its_grid() {
        // a 2 x 2 x 2 grid of 2-unit cells from the origin, precision 0.5:
        // two vertices in cell 0 (x deltas 1 then 2), one in cell 1, one in 7
        let ints = [[1u32, 1, 1], [2, 0, 2], [0, 2, 1], [1, 1, 1]];
        let cell_deltas = [0u32, 0, 1, 6];
        let mut f = ctm_header(b"MG2\0", 4, 4);
        f.extend_from_slice(b"MG2H");
        for x in [0.5f32, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0] {
            f.extend_from_slice(&x.to_le_bytes());
        }
        for d in [2u32, 2, 2] {
            f.extend_from_slice(&d.to_le_bytes());
        }
        f.extend_from_slice(b"VERT");
        let words: Vec<u32> = ints.iter().flatten().copied().collect();
        f.extend(block(&planes(&words, 4, 3)));
        f.extend_from_slice(b"GIDX");
        f.extend(block(&planes(&cell_deltas, 4, 1)));
        f.extend_from_slice(b"INDX");
        f.extend(block(&planes(&index_deltas(&TRIS), 4, 3)));
        let m = read_ctm(&f, &mut unchanged)
            .expect("well formed")
            .expect("MG2");
        assert_eq!(
            m.vertices,
            vec![
                [0.5, 0.5, 0.5],
                [1.5, 0.0, 1.0],
                [2.0, 1.0, 0.5],
                [2.5, 2.5, 2.5]
            ]
        );
        assert_eq!(m.faces, TRIS.to_vec());
    }

    #[test]
    fn a_ctm_that_is_not_one_or_breaks_off() {
        assert_eq!(read_ctm(b"# an OBJ\n", &mut unchanged), Ok(None));
        let mut other = ctm_header(b"MG3\0", 0, 0);
        assert_eq!(read_ctm(&other, &mut unchanged), Ok(None));
        other = ctm_header(b"RAW\0", 4, 4);
        other.extend_from_slice(b"VERT");
        assert_eq!(
            read_ctm(&other, &mut unchanged),
            Err(CtmError::Malformed(
                "a CTM section is missing or out of place"
            ))
        );
        let mut raw = ctm_header(b"RAW\0", 3, 1);
        raw.extend_from_slice(b"INDX");
        for v in [0u32, 1, 3] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        raw.extend_from_slice(b"VERT");
        assert_eq!(
            read_ctm(&raw, &mut unchanged),
            Err(CtmError::Malformed("the CTM file ends early"))
        );
        raw.extend_from_slice(&[0u8; 36]);
        assert_eq!(
            read_ctm(&raw, &mut unchanged),
            Err(CtmError::Malformed(
                "a CTM triangle names a vertex past the last"
            ))
        );
    }

    #[test]
    fn one_object_name_before_the_faces_is_read() {
        let m = read_obj(b"# trimesh\n\no cell.obj\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        assert_eq!(m.map(|m| m.faces), Some(vec![[0, 1, 2]]));
    }

    #[test]
    fn an_obj_keeps_the_referenced_vertices_in_order() {
        let text = b"# a comment\nv 0 0 0\nv 1 0 0\nv 9 9 9\n\nv 0 1 0\nf 1 2 4\r\n";
        let m = read_obj(text).expect("plain");
        assert_eq!(m.vertices, vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(m.faces, vec![[0, 1, 2]]);
    }

    #[test]
    fn an_obj_with_more_than_vertices_and_triangles_is_left_to_trimesh() {
        for text in [
            &b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1 2 3\n"[..],
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1 2/2 3/3\n",
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 4 3\n",
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n",
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n",
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\no thing\nf 1 3 2\n",
            b"o a\no b\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            b"v 0 0 0 1 1 1\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            b"v inf 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            b"v 0 0 0\nv 1 0 0\n",
        ] {
            assert_eq!(read_obj(text), None, "{}", String::from_utf8_lossy(text));
        }
    }
}
