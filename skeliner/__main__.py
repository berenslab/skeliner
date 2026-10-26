"""CLI entrypoint for skeliner.

Usage:
    skeliner view path/to/mesh.obj
    python -m skeliner view path/to/mesh.obj
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="skeliner", description="Skeliner CLI")
    sub = parser.add_subparsers(dest="command")

    # ── view ──────────────────────────────────────────────────────────
    view_parser = sub.add_parser("view", help="Launch interactive mesh viewer")
    view_parser.add_argument(
        "mesh", nargs="?", default=None, help="Path to mesh file (.obj, .ply, etc.)"
    )
    view_parser.add_argument("--port", type=int, default=8777, help="Server port")
    view_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    view_parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )

    args = parser.parse_args()

    if args.command == "view":
        from skeliner.viewer.viewer import view

        view(args.mesh, host=args.host, port=args.port, no_browser=args.no_browser)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
