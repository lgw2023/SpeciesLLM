from __future__ import annotations

import argparse
from pathlib import Path

from training_timeline.api import create_app
from training_timeline.indexer import index_source_roots


def main() -> None:
    parser = argparse.ArgumentParser(description="SpeciesLLM training timeline dashboard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rebuild = subparsers.add_parser("rebuild", help="Rebuild the local SQLite index")
    rebuild.add_argument("--db", type=Path, required=True)
    rebuild.add_argument("--source", type=Path, action="append", required=True)

    serve = subparsers.add_parser("serve", help="Serve the local FastAPI backend")
    serve.add_argument("--db", type=Path, required=True)
    serve.add_argument("--source", type=Path, action="append", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    sources = [source.resolve() for source in args.source]
    if args.command == "rebuild":
        result = index_source_roots(args.db, sources)
        print(result)
        return
    if args.command == "serve":
        import uvicorn

        app = create_app(args.db, sources)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
