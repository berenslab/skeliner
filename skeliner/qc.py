"""skeliner.qc – what may be wrong with each of many cells, and which to look at.

Run over thousands of cells, the automatic pipeline (:func:`~skeliner.pre.
preprocess`, then :func:`~skeliner.skeletonize` with its components) leaves
most of them right and some with a defect worth fixing by hand, and nobody
opens thousands of cells to find which.  This module writes down, cell by
cell, what the run decided and what its outputs measure (:func:`report`,
:func:`save_report`), collects the reports of many cells into one table
(:func:`collect`, :func:`save_table`) and lists the cells to look at, each
with its reasons (:func:`triage`).

A report holds facts and measurements, never a verdict: what is unusual
depends on the cells a cell is compared with, so :func:`triage` decides —
from the facts, and from each cell's rank among the cells collected.  How
many to look at is the caller's to say.

    import skeliner as sk

    for path in mesh_paths:
        mesh, comps = sk.pre.preprocess(sk.io.load_mesh(path))
        skel = sk.skeletonize(mesh, components=comps)
        rep = sk.qc.report(mesh, comps, skel, meta={"cell": path.parent.name})
        sk.qc.save_report(rep, path.parent / "qc.json")

    rows = sk.qc.collect(sorted(root.glob("*/qc.json")))
    sk.qc.save_table(rows, root / "qc.csv")
    for cell in sk.qc.triage(rows, top=50):
        print(cell["cell"], [r["reason"] for r in cell["reasons"]])
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .dataclass import mesh_args

__all__ = [
    "MEASURES",
    "collect",
    "load_report",
    "report",
    "save_report",
    "save_table",
    "triage",
]

#: The version of a report's layout; :func:`load_report` reads this one.
SCHEMA = 1

#: What :func:`triage` ranks the cells by, larger meaning more unusual:
#: the summary's column, and what it measures.  Each is free of the cell's
#: size — a share of its faces, a length against the mesh's own scale — so
#: the biggest cells do not head every list.
MEASURES: dict[str, str] = {
    "bridge_longest_edges": "the longest bridge, in median edges",
    "bridge_most_relative": "a bridge long against the piece it joins",
    "fusion_share": "share of the faces cut out as fusions",
    "parallel_patch_share": "share of the faces cut out as parallel patches",
    "discarded_share": "share of the faces discarded",
    "discarded_soma_share": "the largest discarded piece at the soma "
    "(share of the faces)",
    "discarded_organelle_share": "the largest discarded piece cut off by "
    "organelles (share of the faces)",
    "discarded_reach_share": "the largest loose discarded piece within "
    "bridging reach of the arbor (share of the faces)",
    "stem_longest": "the longest jump from the soma to a neurite",
}

#: The facts, as :func:`triage` words them.
FLAGS: dict[str, str] = {
    "no_soma": "no soma found",
    "no_soma_at_point": "no soma at the point given",
    "skeleton_forest": "the skeleton is more than one tree",
    "skeleton_loops": "the skeleton has loops",
}


# ── one cell ──────────────────────────────────────────────────────────


@mesh_args()
def report(
    mesh,
    components,
    skeleton=None,
    *,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    meta: dict | None = None,
) -> dict:
    """What may be wrong with one cell, as plain JSON-ready data.

    From the record :func:`~skeliner.pre.preprocess` keeps on its components
    (what it decided: where it cut parallel patches and fusions, bridged
    gaps, how it settled the soma) and from the outputs themselves: the
    soma, the organelles' share of the surface, the discarded pieces in
    three kinds — cut off at the soma (touching it), cut off by organelles
    (touching one, not the soma), and loose (a piece of the mesh no bridge
    joined), the last, given the skeleton, within the reach
    :func:`~skeliner.pre.find_gaps` would bridge a piece of its size or not
    — and the skeleton: its trees, its loops and the soma's stems (how far
    each neurite's first node lies off the soma's surface).  Counts of
    faces are also given as shares of the cell's, so cells of any size
    compare.

    Nothing is judged here: a measurement is unusual only against other
    cells', which is :func:`triage`'s to say.  The facts are flagged.

    Parameters
    ----------
    mesh : Mesh
        The mesh :func:`~skeliner.pre.preprocess` returned, compacted or not.
    components : MeshComponents
        Its components.  Made by anything but ``preprocess`` they carry no
        record, and the report says only what the outputs show.
    skeleton : Skeleton, optional
        :func:`~skeliner.skeletonize`'s, from these components.
    bounds : (lo, hi), optional
        The box of the volume the cell was segmented in, in the mesh's
        units.  A cell meeting its edge was cut there — a soma missing for
        that reason is no miss — so the report says whether it does, and
        where (within a median edge of a face of the box).
    meta : dict, optional
        What names the cell, ``{"cell": ..., "type": ...}``: carried into
        the table, and ``by=`` of :func:`triage` can group by it.

    Returns
    -------
    dict
        ``qc`` (the layout's version), ``skeliner``, ``meta``; ``summary``,
        one flat row of numbers (None where one does not apply — no bridge,
        no skeleton); ``flags``, the facts (``no_soma``,
        ``no_soma_at_point``, ``skeleton_forest``, ``skeleton_loops``); and
        ``sites``, each place worth a look: its ``kind``, ``position`` and
        ``extent`` (the radius of what it covers), in the mesh's units, and
        what was measured there.  Dropped into the viewer with the cell's
        mesh, a report's sites are listed as markers.
    """
    from . import __version__
    from .pre import _median_edge

    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    me = _median_edge(mesh)
    summary: dict[str, Any] = {
        "faces": len(faces),
        "vertices": len(verts),
        "median_edge": me,
    }
    flags: list[str] = []
    sites: list[dict] = []

    rec = components.record
    summary["record"] = rec is not None
    if rec is not None:
        _from_record(rec, summary, sites)
    _soma(components, rec, summary, flags, sites)
    _surface(faces, verts, me, components, skeleton, summary, sites)
    if skeleton is not None:
        _skeleton(skeleton, components.soma is not None, summary, flags, sites)
    if bounds is not None:
        _bounds(verts, me, bounds, summary, sites)

    return _plain(
        {
            "qc": SCHEMA,
            "skeliner": __version__,
            "meta": dict(meta or {}),
            "summary": summary,
            "flags": flags,
            "sites": sites,
        }
    )


def _from_record(rec: dict, summary: dict, sites: list) -> None:
    """The run's own decisions: what it cut, bridged and filled."""
    summary["input_faces"] = rec["input"]["faces"]
    summary["soma_asked"] = rec["soma"]["asked"]
    n_in = rec["input"]["faces"]
    patches = rec.get("parallel_patches", [])
    summary["parallel_patches"] = len(patches)
    summary["parallel_patch_faces"] = sum(p["faces"] for p in patches)
    summary["parallel_patch_share"] = (
        summary["parallel_patch_faces"] / n_in if n_in else None
    )
    for p in patches:
        sites.append(
            {
                "kind": "parallel_patch",
                "position": p["center"],
                "extent": p["diag"] / 2,
                "faces": p["faces"],
                "axis": p["axis"],
            }
        )
    fusions = rec.get("fusions", [])
    summary["fusions"] = len(fusions)
    summary["fusion_faces"] = sum(f["faces"] for f in fusions)
    summary["fusion_share"] = summary["fusion_faces"] / n_in if n_in else None
    for f in fusions:
        sites.append(
            {
                "kind": "fusion",
                "position": f["center"],
                "extent": f["diag"] / 2,
                "faces": f["faces"],
            }
        )
    summary["disconnected"] = rec["disconnected"]["pieces"]

    bridges = rec.get("bridges", [])
    scale = rec.get("median_edge")
    summary["bridges"] = len(bridges)
    edges, relative = [], []
    for b in bridges:
        a, c = np.asarray(b["a"]), np.asarray(b["b"])
        e = b["length"] / scale if scale else None
        r = b["length"] / b["diag"] if b["diag"] else None
        edges.append(e)
        relative.append(r)
        sites.append(
            {
                "kind": "bridge",
                "position": (a + c) / 2,
                "extent": b["length"] / 2,
                "a": b["a"],
                "b": b["b"],
                "length": b["length"],
                "edges": e,
                "relative": r,
                "pieces": b["pieces"],
            }
        )
    summary["bridge_longest_edges"] = _max(edges)
    summary["bridge_most_relative"] = _max(relative)
    summary["fill_faces"] = rec["fill"]["faces"]


def _soma(components, rec, summary: dict, flags: list, sites: list) -> None:
    soma = components.soma
    summary["soma"] = soma is not None
    summary["soma_length"] = None
    if soma is not None:
        length = 2.0 * float(np.max(soma.axes))
        summary["soma_length"] = length
        sites.append(
            {
                "kind": "soma",
                "position": soma.center,
                "extent": length / 2,
                "length": length,
            }
        )
        return
    asked = rec["soma"] if rec is not None else None
    if asked is not None and asked["asked"] == "point":
        flags.append("no_soma_at_point")
        sites.append({"kind": "soma_point", "position": asked["point"], "extent": 0.0})
    elif asked is None or asked["asked"] != "off":
        # told it has none (soma=False), the cell is not missing one
        flags.append("no_soma")


def _surface(faces, verts, me, components, skeleton, summary, sites) -> None:
    """The organelles' share, and the discarded pieces in three kinds: cut
    off at the soma (touching it), cut off by organelles (touching one, not
    the soma), and loose (touching neither: a piece of the mesh left
    unbridged), the last with how far from the arbor against the reach it
    would be bridged across."""
    from .pre import _bridge_reach

    n_f, n_v = len(faces), len(verts)
    org = np.zeros(n_f, dtype=bool)
    mask = np.asarray(components.organelles.mask, dtype=bool)[:n_f]
    org[: len(mask)] = mask
    summary["organelle_share"] = float(org.sum()) / n_f if n_f else None

    neurites = [np.asarray(p, dtype=np.int64) for p in components.neurites]
    summary["neurites"] = len(neurites)
    pieces = [np.asarray(p, dtype=np.int64) for p in components.discarded]
    sizes = np.array([len(p) for p in pieces], dtype=np.int64)
    summary["discarded"] = len(pieces)
    summary["discarded_share"] = float(sizes.sum()) / n_f if n_f else None
    summary["discarded_largest"] = int(sizes.max()) if len(sizes) else None
    for kind in ("soma", "organelle", "reach"):
        summary[f"discarded_{kind}_largest"] = None
        summary[f"discarded_{kind}_share"] = None
    summary["discarded_at_soma"] = 0
    summary["discarded_at_organelle"] = 0
    summary["discarded_within_reach"] = None
    if not pieces:
        return

    def largest(kind: str, which: np.ndarray) -> None:
        top = _max(sizes[which].tolist())
        summary[f"discarded_{kind}_largest"] = top
        summary[f"discarded_{kind}_share"] = top / n_f if top is not None else None

    piece = np.repeat(np.arange(len(pieces)), sizes)
    pf = faces[np.concatenate(pieces)]

    def touching(vmask: np.ndarray) -> np.ndarray:
        hit = vmask[pf].any(axis=1)
        return np.bincount(piece, weights=hit, minlength=len(pieces)) > 0

    soma = components.soma
    soma_v = np.zeros(n_v, dtype=bool)
    if soma is not None and soma.verts is not None:
        soma_v[np.asarray(soma.verts, dtype=np.int64)] = True
    org_v = np.zeros(n_v, dtype=bool)
    org_v[faces[org].ravel()] = True
    at_soma = touching(soma_v)
    at_org = touching(org_v) & ~at_soma
    loose = ~at_soma & ~at_org
    summary["discarded_at_soma"] = int(at_soma.sum())
    summary["discarded_at_organelle"] = int(at_org.sum())
    largest("soma", at_soma)
    largest("organelle", at_org)

    # each piece's box, the size find_gaps caps a bridge by
    p3 = verts[pf].reshape(-1, 3)
    pv = np.repeat(piece, 3)
    lo = np.full((len(pieces), 3), np.inf)
    hi = np.full((len(pieces), 3), -np.inf)
    np.minimum.at(lo, pv, p3)
    np.maximum.at(hi, pv, p3)
    centre = np.zeros((len(pieces), 3))
    np.add.at(centre, pv, p3)
    centre /= 3.0 * sizes[:, None]
    diag = np.linalg.norm(hi - lo, axis=1)

    far = reach = None
    near = np.zeros(len(pieces), dtype=bool)
    if skeleton is not None:
        from . import dx

        used = np.unique(pf)
        d_v = np.full(n_v, np.inf)
        d_v[used] = dx.distance(skeleton, verts[used])
        far = np.full(len(pieces), np.inf)
        np.minimum.at(far, piece, d_v[pf].min(axis=1))
        reach = np.array([_bridge_reach(d, me) for d in diag])
        near = loose & (far <= reach)
        summary["discarded_within_reach"] = int(near.sum())
        largest("reach", near)

    for i in range(len(pieces)):
        if not (at_soma[i] or at_org[i] or near[i]):
            continue
        sites.append(
            {
                "kind": "discarded",
                "position": centre[i],
                "extent": diag[i] / 2,
                "index": i,
                "faces": int(sizes[i]),
                "at_soma": bool(at_soma[i]),
                "at_organelle": bool(at_org[i]),
                "arbor_distance": None if far is None else float(far[i]),
                "reach": None if reach is None else float(reach[i]),
            }
        )


def _skeleton(skel, has_soma: bool, summary, flags, sites) -> None:
    """Trees, loops and the soma's stems."""
    from ._backend import kernels
    from .dx import _edges

    edges = _edges(skel)
    n = len(skel.nodes)
    nodes = np.asarray(skel.nodes, dtype=np.float64)
    radius = np.asarray(skel.r, dtype=np.float64)
    label = np.asarray(kernels.dx.components(edges, n)) if n else np.zeros(0)
    trees = int(label.max()) + 1 if n else 0
    loops = len(edges) - n + trees
    summary["skel_nodes"] = n
    summary["skel_trees"] = trees
    summary["skel_loops"] = loops
    if trees > 1:
        flags.append("skeleton_forest")
        for t in range(trees):
            if n and t != label[0]:
                members = np.flatnonzero(label == t)
                box = nodes[members]
                sites.append(
                    {
                        "kind": "tree",
                        "position": nodes[members[0]],
                        "extent": float(np.linalg.norm(np.ptp(box, axis=0))) / 2,
                        "node": int(members[0]),
                        "nodes": len(members),
                    }
                )
    if loops > 0:
        flags.append("skeleton_loops")
        for cycle in kernels.dx.cycle_basis(edges, n):
            u = int(edges[int(cycle[0])][0])
            sites.append(
                {
                    "kind": "loop",
                    "position": nodes[u],
                    "extent": float(radius[u]),
                    "node": u,
                }
            )

    summary["stem_longest"] = None
    if not has_soma or not n:
        return
    stems = np.unique(edges[(edges == 0).any(axis=1)].ravel())
    stems = stems[stems != 0]
    if not len(stems):
        return
    gap = np.maximum(0.0, np.atleast_1d(skel.soma.distance_to_surface(nodes[stems])))
    summary["stem_longest"] = float(gap.max())
    for v, g in zip(stems.tolist(), gap.tolist()):
        sites.append(
            {
                "kind": "stem",
                "position": nodes[v],
                "extent": float(radius[v]),
                "node": v,
                "gap": g,
            }
        )


def _bounds(verts, me, bounds, summary, sites) -> None:
    """Where the cell meets the volume's box: within a median edge of one
    of its faces."""
    lo = np.asarray(bounds[0], dtype=np.float64).reshape(3)
    hi = np.asarray(bounds[1], dtype=np.float64).reshape(3)
    at = np.zeros(len(verts), dtype=bool)
    for axis in range(3):
        for side, hit in (
            ("lo", verts[:, axis] - lo[axis] <= me),
            ("hi", hi[axis] - verts[:, axis] <= me),
        ):
            at |= hit
            if hit.any():
                v = verts[hit]
                sites.append(
                    {
                        "kind": "bounds",
                        "position": v.mean(axis=0),
                        "extent": float(np.linalg.norm(np.ptp(v, axis=0))) / 2,
                        "axis": axis,
                        "side": side,
                        "vertices": int(hit.sum()),
                    }
                )
    summary["bounds_vertices"] = int(at.sum())
    summary["touches_bounds"] = bool(at.any())


def _max(values: Iterable) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _plain(x):
    """JSON-ready: numpy scalars and arrays as Python numbers and lists, a
    number that is not finite as None."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def save_report(rep: dict, path: str | Path) -> None:
    """Write a :func:`report` as JSON."""
    Path(path).write_text(json.dumps(_plain(rep), indent=1), encoding="utf-8")


def load_report(path: str | Path) -> dict:
    """Read a report :func:`save_report` wrote."""
    rep = json.loads(Path(path).read_text(encoding="utf-8"))
    if rep.get("qc") != SCHEMA:
        raise ValueError(
            f"{path}: a qc report of layout {rep.get('qc')!r}, not {SCHEMA}"
        )
    return rep


# ── many cells ────────────────────────────────────────────────────────


def collect(reports: Iterable[dict | str | Path]) -> list[dict]:
    """One row per cell: its ``meta``, its ``summary`` and its ``flags``.

    *reports* are reports or paths of saved ones.  A report whose meta
    names no ``cell`` is named after its file's folder (``<cell>/qc.json``)
    or, given in memory, its place in the list.
    """
    rows = []
    for i, item in enumerate(reports):
        if isinstance(item, dict):
            rep, name = item, str(i)
        else:
            rep, name = load_report(item), Path(item).parent.name
        meta = dict(rep.get("meta") or {})
        row = {"cell": str(meta.pop("cell", name)), **meta}
        row.update(rep["summary"])
        row["flags"] = list(rep.get("flags", []))
        rows.append(row)
    return rows


def save_table(rows: Sequence[dict], path: str | Path) -> None:
    """Write rows — :func:`collect`'s or :func:`triage`'s — as CSV: every
    column any row has, in the order first met; a list as its items joined
    by ``"; "``, a reason as ``"what (value, rank/of)"``."""
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=list(columns))
        out.writeheader()
        for row in rows:
            out.writerow({k: _cell(row.get(k)) for k in columns})


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "; ".join(_cell(x) for x in v)
    if isinstance(v, dict) and "reason" in v:
        if v.get("rank") is None:
            return str(v["reason"])
        return f"{v['reason']} ({_short(v['value'])}, {v['rank']}/{v['of']})"
    return str(v)


def _short(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def triage(
    rows: Sequence[dict],
    *,
    top: int | None = None,
    by: str | None = None,
    measures: Iterable[str] | None = None,
) -> list[dict]:
    """The cells to look at, each with its reasons.

    A reason is a fact (a flag of its report: no soma found, none at the
    point given, a skeleton of more than one tree or with loops), or, with
    *top*, a measure in which the cell is among the *top* most extreme of
    the cells collected (of its group, with *by*).  No threshold decides:
    what is extreme is set by the other cells, and how many to look at by
    *top* — the time there is to look.

    Parameters
    ----------
    rows : list of dict
        :func:`collect`'s.
    top : int, optional
        How many cells to take per measure; None takes the facts alone.
    by : str, optional
        A column to rank within — a ``meta`` key such as ``"type"`` — so a
        cell is compared with its own kind.
    measures : iterable of str, optional
        The columns to rank by, larger meaning more unusual; by default
        :data:`MEASURES`.

    Returns
    -------
    list of dict
        One per cell with a reason: ``cell``, ``by``'s value when given, and
        ``reasons`` — each ``{"reason", "value", "rank", "of"}``, ``rank``
        None for a fact.  Cells with facts first (most first), then by
        their best rank.
    """
    found: dict[str, dict] = {}

    def add(row, reason, value=None, rank=None, of=None):
        entry = found.setdefault(
            row["cell"],
            {"cell": row["cell"], **({by: row.get(by)} if by else {}), "reasons": []},
        )
        entry["reasons"].append(
            {"reason": reason, "value": value, "rank": rank, "of": of}
        )

    for row in rows:
        for flag in row.get("flags") or ():
            what = FLAGS.get(flag, flag)
            if flag == "no_soma" and row.get("touches_bounds"):
                what += " (the cell meets the volume's edge)"
            add(row, what)

    if top:
        groups: dict[Any, list[dict]] = {}
        for row in rows:
            groups.setdefault(row.get(by) if by else None, []).append(row)
        for name in measures or MEASURES:
            label = MEASURES.get(name, name)
            for members in groups.values():
                have = [r for r in members if _number(r.get(name)) is not None]
                have.sort(key=lambda r: -_number(r[name]))
                for rank, row in enumerate(have[:top], start=1):
                    add(row, label, _number(row[name]), rank, len(have))

    def order(entry):
        ranks = [r["rank"] for r in entry["reasons"] if r["rank"] is not None]
        facts = sum(r["rank"] is None for r in entry["reasons"])
        return (-facts, min(ranks, default=math.inf), -len(ranks))

    return sorted(found.values(), key=order)


def _number(v) -> float | None:
    """A table's value as a number: None for an empty one (a CSV's "")."""
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
