"""Command-line interface for managing the Poisson stage cache and hashing files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cache import Cache
from .hashing import blake3_file, blake3_version


def _format_bytes(num: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{num} B"
        value /= 1024.0
    return f"{num} B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m poisson_runner",
        description="Manage the BLAKE3 content-addressed stage cache for Poisson.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cache_p = sub.add_parser("cache", help="Inspect or clean the stage cache")
    cache_p.add_argument("--cache-dir", default=None, help="Cache directory (default: POISSON_CACHE_DIR or .poisson_cache)")
    cache_p.add_argument(
        "--lock-mode",
        default=None,
        choices=["flock", "lockfile", "none"],
        help="Override cache lock mode (default: cache .lock_mode marker, POISSON_CACHE_LOCK_MODE, or flock)",
    )
    cache_sub = cache_p.add_subparsers(dest="cache_cmd", required=True)

    cache_sub.add_parser("status", help="Show cache disk usage and entry counts")

    clean_p = cache_sub.add_parser("clean", help="Remove temporary files and/or evict cached objects")
    group = clean_p.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Remove all cached objects and request metadata")
    group.add_argument("--max-gb", type=float, default=None, help="Trim cache objects via LRU down to this size (GiB)")

    inspect_p = cache_sub.add_parser("inspect", help="Show provenance metadata for cached entries")
    inspect_p.add_argument("prefix", nargs="?", default="", help="Optional request_key or content_id hex prefix")

    hash_p = sub.add_parser("hash", help="Compute BLAKE3 hex digest of one or more files")
    hash_p.add_argument("files", nargs="+", type=Path, help="Files to hash")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hash":
        for path in args.files:
            print(f"{blake3_file(path)}  {path}")
        return 0

    if args.command == "cache":
        cache = Cache(root=args.cache_dir, lock_mode=args.lock_mode)
        if args.cache_cmd == "status":
            st = cache.stats()
            print(f"Cache root:    {st.root}")
            print(f"Lock mode:     {cache.lock_mode.value}")
            print(f"BLAKE3:        v{blake3_version()}")
            print(f"Objects:       {st.object_count} ({_format_bytes(st.object_bytes)})")
            print(f"Requests:      {st.request_count} ({_format_bytes(st.metadata_bytes)})")
            print(f"Ephemeral:     {st.ephemeral_count} ({_format_bytes(st.ephemeral_bytes)})")
            print(f"Temp files:    {st.temp_count}")
            limit = _format_bytes(st.max_bytes) if st.max_bytes is not None else "unlimited"
            print(f"Max size cap:  {limit}")
            return 0

        if args.cache_cmd == "clean":
            if args.max_gb is not None and args.max_gb < 0:
                print(f"error: --max-gb must be non-negative, got {args.max_gb}", file=sys.stderr)
                return 2
            max_bytes = int(args.max_gb * (1024**3)) if args.max_gb is not None else None
            res = cache.clean(max_bytes=max_bytes, all_entries=args.all)
            print(
                f"Removed {res['removed_objects']} objects, "
                f"{res['removed_requests']} requests, {res['removed_tmp']} temp files, "
                f"{res['removed_ephemeral']} ephemeral files."
            )
            return 0

        if args.cache_cmd == "inspect":
            records = cache.inspect(args.prefix)
            print(json.dumps(records, indent=2))
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
