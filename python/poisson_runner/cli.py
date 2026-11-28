"""Command-line interface for managing the Poisson stage cache and hashing files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle import bundle_saved_run, format_bytes, saved_run_status
from .cache import Cache
from .hashing import blake3_file, blake3_version


def _format_bytes(num: int) -> str:
    return format_bytes(num)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m poisson_runner",
        description="Manage the BLAKE3 content-addressed stage cache, saved runs, and file hashing.",
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

    run_p = sub.add_parser("run", help="Inspect or bundle artifacts of a saved experiment run")
    run_sub = run_p.add_subparsers(dest="run_cmd", required=True)

    bundle_p = run_sub.add_parser(
        "bundle",
        help="Hard-link or copy a saved run's artifacts into <dir>/objects/ so it survives cache clean or /scratch purge",
    )
    bundle_p.add_argument("directory", type=Path, help="Saved run directory (or path to report.json)")
    bundle_p.add_argument("--link-only", action="store_true", help="Only hard-link; do not copy across filesystems")
    bundle_p.add_argument("--stages", action="store_true", help="Also bundle intermediate stage outputs (not just final outputs)")
    bundle_p.add_argument("--cache-dir", default=None, help="Override cache directory for finding unbundled objects")

    status_p = run_sub.add_parser("status", help="Report whether a saved run's artifacts are bundled, in cache only, or missing")
    status_p.add_argument("directory", type=Path, help="Saved run directory (or path to report.json)")
    status_p.add_argument("--verify", action="store_true", help="Verify BLAKE3 digests of all present artifacts")
    status_p.add_argument("--stages", action="store_true", help="Also check intermediate stage outputs")
    status_p.add_argument("--cache-dir", default=None, help="Override cache directory for finding unbundled objects")

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
            own_count = st.object_count - st.shared_count
            own_bytes = st.object_bytes - st.shared_bytes
            print(f"Cache root:    {st.root}")
            print(f"Lock mode:     {cache.lock_mode.value}")
            print(f"BLAKE3:        v{blake3_version()}")
            print(f"Objects:       {st.object_count} ({_format_bytes(st.object_bytes)})")
            if st.shared_count:
                print(f"  Cache-only:  {own_count} ({_format_bytes(own_bytes)})")
                print(f"  Shared runs: {st.shared_count} ({_format_bytes(st.shared_bytes)})")
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

    if args.command == "run":
        include = "stages" if args.stages else "outputs"
        if args.run_cmd == "bundle":
            mode = "link" if args.link_only else "link_or_copy"
            summary = bundle_saved_run(args.directory, mode=mode, include=include, cache_dir=args.cache_dir)
            parts = [
                f"{summary.linked} linked ({_format_bytes(summary.linked_bytes)})",
                f"{summary.copied} copied ({_format_bytes(summary.copied_bytes)})",
                f"{summary.already_present} already bundled",
            ]
            if summary.left_in_cache:
                parts.append(f"{summary.left_in_cache} left in cache ({_format_bytes(summary.left_in_cache_bytes)})")
            if summary.missing:
                parts.append(f"{summary.missing} missing")
            print(f"Bundled {args.directory}: {', '.join(parts)}.")
            if summary.missing:
                shown = ", ".join(summary.missing_objects[:10])
                more = f" (and {summary.missing - 10} more)" if summary.missing > 10 else ""
                print(f"error: missing objects: {shown}{more}", file=sys.stderr)
                return 1
            return 0

        if args.run_cmd == "status":
            st = saved_run_status(args.directory, include=include, cache_dir=args.cache_dir, verify=args.verify)
            print(f"Saved run:     {args.directory}")
            print(f"Scope:         {include}")
            print(f"Bundled:       {st.bundled}/{st.total} ({_format_bytes(st.bundled_bytes)})")
            print(f"Cache-only:    {st.cache_only}/{st.total} ({_format_bytes(st.cache_only_bytes)})")
            print(f"Missing:       {st.missing}/{st.total}")
            if st.external_missing:
                print(f"Missing inputs: {len(st.external_missing)} ({', '.join(st.external_missing[:5])})")
            if args.verify:
                print(f"Corrupted:     {len(st.corrupted)}")
            if st.missing:
                print(f"Missing objects: {', '.join(st.missing_objects[:10])}", file=sys.stderr)
            if st.corrupted:
                print(f"Corrupted objects: {', '.join(st.corrupted[:10])}", file=sys.stderr)
            return 1 if (st.missing or st.corrupted) else 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
