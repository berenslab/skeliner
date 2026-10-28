"""``skeliner.io`` on the Python path: OBJ read as trimesh reads it
(``trimesh.load_mesh(path, process=False)``), for the files of plain vertices
and triangles skeliner's pipeline writes and reads; and OpenCTM, only for
reading EyeWire 1's meshes
(https://drive.google.com/file/d/1-qmE1sEL3AIGpNfi0w50MEOgm2WIF4_1/view).

Each reader returns None for anything it does not read, so the caller hands
that file to trimesh instead.
"""

from __future__ import annotations

import lzma
import re
import struct
from pathlib import Path

import numpy as np

# the whitespace Rust's split_ascii_whitespace splits on
_SPACE = re.compile(rb"[ \t\n\x0c\r]+")
_DECIMAL_CHARS = re.compile(rb"[0-9+\-.eE]+")


def _split(line: bytes) -> list[bytes]:
    return [t for t in _SPACE.split(line) if t]


def _decimal(token: bytes) -> float | None:
    """A decimal number as trimesh's parser reads it, or None for anything
    else (``inf``, ``nan``, hex, a stray character), which trimesh may read
    differently."""
    if not _DECIMAL_CHARS.fullmatch(token) or not any(
        c in b"0123456789" for c in token
    ):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _obj_index(token: bytes) -> int | None:
    """A 1-based OBJ index as a 0-based one; None for zero, a relative
    (negative) index or anything not a plain number."""
    if not token.isdigit():
        return None
    value = int(token)
    return value - 1 if 1 <= value < 2**32 else None


def _referenced_only(vertices: np.ndarray, faces: np.ndarray):
    """Only the vertices some face names, in their order, the faces
    renumbered: what trimesh's OBJ loader keeps."""
    used = np.zeros(len(vertices), dtype=bool)
    used[faces] = True
    new_id = np.cumsum(used) - 1
    return vertices[used], new_id[faces]


def read_obj(path) -> tuple[np.ndarray, np.ndarray] | None:
    """The OBJ at *path* — ``v x y z`` and triangle ``f a b c`` lines
    (1-based indices), with comments and blank lines between and at most one
    ``o`` line before the first face (trimesh's own export names its object)
    — as ``(vertices, faces)``: vertices no face names are dropped, as
    trimesh does. None for anything more — normals, texture coordinates,
    more objects, groups, materials, polygons, slashes, relative or
    out-of-range indices, extra vertex values, no faces at all."""
    data = Path(path).read_bytes()
    if not data.isascii() or b"\\" in data:
        return None
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    named = False
    for line in data.split(b"\n"):
        if line.endswith(b"\r"):
            line = line[:-1]
        if not _split(line) or line.startswith(b"#"):
            continue
        if line.startswith(b"o "):
            # one object for every face, or trimesh splits the mesh
            if named or faces:
                return None
            named = True
        elif line.startswith(b"v "):
            tokens = _split(line[2:])
            v = [_decimal(t) for t in tokens]
            if len(v) != 3 or None in v:
                return None
            vertices.append(v)
        elif line.startswith(b"f "):
            tokens = _split(line[2:])
            f = [_obj_index(t) for t in tokens]
            if len(f) != 3 or None in f:
                return None
            faces.append(f)
        else:
            return None
    if not faces or max(max(f) for f in faces) >= len(vertices):
        return None
    v, f = _referenced_only(
        np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        np.asarray(faces, dtype=np.int64),
    )
    return v, f


def _unlzma(props: bytes, packed: bytes, size: int) -> bytes:
    """An LZMA1 block of *size* bytes from its five property bytes and data,
    read as the reference library reads it: up to *size* bytes, whatever
    follows. As an LZMA "alone" stream (the property bytes, the unpacked
    size, the data) with the size known; where the data ends in an end
    marker, which liblzma before 5.4 refuses with the size known, again with
    the size unknown."""
    try:
        return lzma.decompress(
            props + struct.pack("<Q", size) + packed, format=lzma.FORMAT_ALONE
        )
    except lzma.LZMAError:
        return lzma.decompress(props + b"\xff" * 8 + packed, format=lzma.FORMAT_ALONE)


class _Ctm:
    """An OpenCTM file, read front to back."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        if len(out) != n:
            raise ValueError("the CTM file ends early")
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def f32(self) -> np.float32:
        return np.frombuffer(self.take(4), dtype="<f4")[0]

    def count(self) -> int:
        n = self.u32()
        if n >= 2**31:
            raise ValueError("a CTM count is negative")
        return n

    def section(self, tag: bytes) -> None:
        if self.take(4) != tag:
            raise ValueError("a CTM section is missing or out of place")

    def packed(self, count: int, size: int) -> np.ndarray:
        """A packed block of *count* × *size* 32-bit words, ``(count,
        size)``: its packed length, five LZMA property bytes and the LZMA1
        data, which unpack to four byte planes, most significant first, each
        plane component by component."""
        n = self.u32()
        props = self.take(5)
        packed = self.take(n)
        total = count * size
        raw = _unlzma(props, packed, 4 * total)
        if len(raw) != 4 * total:
            raise ValueError("a CTM block unpacks to the wrong size")
        planes = np.frombuffer(raw, dtype=np.uint8).reshape(4, size, count)
        words = (
            (planes[0].astype(np.uint32) << 24)
            | (planes[1].astype(np.uint32) << 16)
            | (planes[2].astype(np.uint32) << 8)
            | planes[3].astype(np.uint32)
        )
        return words.T


def _restore_indices(words: np.ndarray) -> np.ndarray:
    """MG1's and MG2's triangles from their deltas: the first index from the
    previous triangle's first, the third from the first, the second from
    the previous second where the first repeats and from the first
    otherwise — in the reference library's 32-bit unsigned arithmetic."""
    tris = [[int(a), int(b), int(c)] for a, b, c in words]
    wrap = 0xFFFFFFFF
    for i, t in enumerate(tris):
        if i >= 1:
            t[0] = (t[0] + tris[i - 1][0]) & wrap
        t[2] = (t[2] + t[0]) & wrap
        if i >= 1 and t[0] == tris[i - 1][0]:
            t[1] = (t[1] + tris[i - 1][1]) & wrap
        else:
            t[1] = (t[1] + t[0]) & wrap
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def read_ctm(path) -> tuple[np.ndarray, np.ndarray] | None:
    """The vertices and triangles of the OpenCTM v5 file at *path*, as the
    reference library decodes them (vertices in single precision, as stored,
    then widened): ``(vertices, faces)``.

    Only for reading EyeWire 1's meshes
    (https://drive.google.com/file/d/1-qmE1sEL3AIGpNfi0w50MEOgm2WIF4_1/view):
    RAW, MG1 or MG2 compression, every vertex kept; normals, UV maps and
    attribute maps, which those files do not have, are not read. None for a
    file that is not OpenCTM v5 or uses another compression.
    """
    f = _Ctm(Path(path).read_bytes())
    if f.data[:4] != b"OCTM" or f.data[4:8] != struct.pack("<I", 5):
        return None
    f.pos = 8
    method = f.take(4)
    n_vertices = f.count()
    n_triangles = f.count()
    f.u32()  # UV maps
    f.u32()  # attribute maps
    f.u32()  # flags
    f.take(f.count())  # comment
    if method == b"RAW\x00":
        f.section(b"INDX")
        tris = np.frombuffer(f.take(12 * n_triangles), dtype="<u4").reshape(-1, 3)
        f.section(b"VERT")
        verts = np.frombuffer(f.take(12 * n_vertices), dtype="<f4").reshape(-1, 3)
    elif method == b"MG1\x00":
        f.section(b"INDX")
        tris = _restore_indices(f.packed(n_triangles, 3))
        f.section(b"VERT")
        # one run of 3 × n floats, each vertex's x, y and z in turn
        words = np.ascontiguousarray(f.packed(3 * n_vertices, 1))
        verts = words.view(np.float32).reshape(-1, 3)
    elif method == b"MG2\x00":
        f.section(b"MG2H")
        precision = f.f32()
        f.f32()  # normal precision
        lo = np.array([f.f32() for _ in range(3)], dtype=np.float32)
        hi = np.array([f.f32() for _ in range(3)], dtype=np.float32)
        div = [f.u32() for _ in range(3)]
        if 0 in div:
            raise ValueError("the CTM grid has a zero division")
        f.section(b"VERT")
        ints = f.packed(n_vertices, 3).astype(np.int32).astype(np.int64)
        f.section(b"GIDX")
        cells = np.cumsum(f.packed(n_vertices, 1)[:, 0].astype(np.int64))
        f.section(b"INDX")
        tris = _restore_indices(f.packed(n_triangles, 3))
        if len(cells) and cells[-1] >= div[0] * div[1] * div[2]:
            raise ValueError("a CTM vertex lies outside its grid")
        # the grid: cells of (hi - lo) / div, numbered x fastest
        size = (hi - lo) / np.asarray(div, dtype=np.float32)
        g = np.stack(
            [cells % div[0], (cells // div[0]) % div[1], cells // (div[0] * div[1])],
            axis=1,
        )
        origin = g.astype(np.float32) * size + lo
        # x is a delta from the previous vertex in the same cell
        x = ints[:, 0].copy()
        for i in range(1, n_vertices):
            if cells[i] == cells[i - 1]:
                x[i] += x[i - 1]
        q = np.stack([x, ints[:, 1], ints[:, 2]], axis=1).astype(np.float32)
        verts = precision * q + origin
    else:
        return None
    if len(tris) and tris.max() >= n_vertices:
        raise ValueError("a CTM triangle names a vertex past the last")
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def write_obj(path, vertices, faces) -> None:
    """Write *vertices* ``(N, 3)`` and *faces* ``(F, 3)`` to *path* as
    trimesh's ``export_obj`` writes a plain mesh (no normals, colours,
    texture or name): its header comment, a ``v`` line per vertex with eight
    decimals, an ``f`` line per face numbered from one, and a blank line to
    end."""
    from .mesh import checked_faces, rows3

    vertices = rows3("V", vertices)
    faces, _ = checked_faces(faces, len(vertices))
    v = "\nv ".join(
        " ".join(format(x, ".8f") for x in row) for row in vertices.tolist()
    )
    f = "\nf ".join(f"{a} {b} {c}" for a, b, c in (faces + 1).tolist())
    text = "\n".join(["# https://github.com/mikedh/trimesh", f"v {v}\nf {f}", "\n"])
    Path(path).write_bytes(text.encode())
