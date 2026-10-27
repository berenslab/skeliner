"""CLI entrypoint for skeliner.

Usage:
    skeliner view [path/to/mesh.obj]
    skeliner view-contacts A.obj B.obj [sites.npz]
    python -m skeliner view path/to/mesh.obj
"""

import argparse
import sys
from pathlib import Path


def _add_server_args(parser):
    parser.add_argument("--port", type=int, default=8777, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )


def _mesh_name(path: str) -> str:
    """A mesh's name in the contact panel: the file with its folder, which
    names the cell when the file is ``<cell id>/mesh.obj``."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


def main():
    parser = argparse.ArgumentParser(prog="skeliner", description="Skeliner CLI")
    sub = parser.add_subparsers(dest="command")

    # ── view ──────────────────────────────────────────────────────────
    view_parser = sub.add_parser("view", help="Launch interactive mesh viewer")
    view_parser.add_argument(
        "mesh", nargs="?", default=None, help="Path to mesh file (.obj, .ply, etc.)"
    )
    _add_server_args(view_parser)

    # ── view-contacts ─────────────────────────────────────────────────
    contacts_parser = sub.add_parser(
        "view-contacts",
        help="Launch the viewer on two meshes and the contact sites between them",
        description=(
            "Show two meshes and the sites where they touch. Without a sites"
            " file (one saved with sk.io.save_contact_sites, found on these"
            " two meshes) the sites are found with sk.pair.scan_contacts."
        ),
    )
    contacts_parser.add_argument("mesh_a", help="Mesh A (.obj, .ply, etc.)")
    contacts_parser.add_argument("mesh_b", help="Mesh B")
    contacts_parser.add_argument(
        "sites", nargs="?", default=None, help="Contact sites (.npz) of A and B"
    )
    contacts_parser.add_argument(
        "--sides",
        default="A",
        choices=["A", "B", "both"],
        help="Whose side of the sites to show first (the panel switches it)",
    )
    contacts_parser.add_argument(
        "--color-by",
        default="kind",
        choices=["kind", "site"],
        help="Colour the sites by their kind or each in turn",
    )
    contacts_parser.add_argument(
        "--boxes",
        default="union",
        choices=["union", "split", "none"],
        help="A box around both sides of a site, one per side, or none at first",
    )
    _add_server_args(contacts_parser)

    args = parser.parse_args()

    if args.command == "view":
        from skeliner.viewer.viewer import start

        start(args.mesh, host=args.host, port=args.port, no_browser=args.no_browser)
    elif args.command == "view-contacts":
        import skeliner as sk
        from skeliner.viewer.viewer import view_contacts

        meshes = [sk.io.load_mesh(p) for p in (args.mesh_a, args.mesh_b)]
        sites = sk.io.load_contact_sites(args.sites) if args.sites else None
        view_contacts(
            *meshes,
            sites,
            names=(_mesh_name(args.mesh_a), _mesh_name(args.mesh_b)),
            sides=args.sides,
            show_aabb=args.boxes != "none",
            aabb_mode="split" if args.boxes == "split" else "union",
            color_by=args.color_by,
            host=args.host,
            port=args.port,
            no_browser=args.no_browser,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
