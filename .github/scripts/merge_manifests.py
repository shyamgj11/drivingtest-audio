#!/usr/bin/env python3
"""Merge multiple per-shard manifest.json files into a single combined manifest.

Each shard's generate_audio.py run produces its own manifest.json with the
entries for that shard's slice. Run this from the directory containing all
shard outputs (e.g. after `actions/download-artifact` flattens them).

Usage:
  python scripts/merge_manifests.py --inputs shard-0/manifest.json shard-1/manifest.json ... \\
                                    --output manifest.json

Exits non-zero on any conflict between shards (same qid in two manifests with
different hashes — should not happen but worth catching).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Paths to per-shard manifest.json files.")
    ap.add_argument("--output", required=True,
                    help="Destination merged manifest path.")
    args = ap.parse_args()

    merged = {"version": 1, "voice": None, "files": {}}
    conflicts: list[str] = []

    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f"[merge] WARN: missing {p}, skipping", file=sys.stderr)
            continue
        try:
            data = json.loads(p.read_text("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[merge] ERROR: {p} is not valid JSON: {e}", file=sys.stderr)
            return 2

        if merged["voice"] is None:
            merged["voice"] = data.get("voice")
        elif data.get("voice") and data["voice"] != merged["voice"]:
            print(f"[merge] ERROR: voice mismatch — {merged['voice']} vs {data['voice']} in {p}",
                  file=sys.stderr)
            return 2

        for qid, entry in data.get("files", {}).items():
            if qid in merged["files"]:
                existing = merged["files"][qid]
                if existing.get("q_hash") != entry.get("q_hash"):
                    conflicts.append(f"{qid} (in {p})")
                    continue
            merged["files"][qid] = entry

    if conflicts:
        print(f"[merge] ERROR: {len(conflicts)} duplicate qids with mismatching hashes:",
              file=sys.stderr)
        for c in conflicts[:20]:
            print(f"  - {c}", file=sys.stderr)
        return 3

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), "utf-8")

    print(f"[merge] OK — {len(merged['files'])} entries from {len(args.inputs)} "
          f"shards → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
