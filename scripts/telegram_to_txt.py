#!/usr/bin/env python3
"""Convert Telegram JSON exports to plain text files for RAG ingestion.

Extracts message text only (no usernames, dates, or service messages).
Filters out very short messages (< MIN_CHARS chars after extraction).
Links are stripped; surrounding plain text is kept.

Usage:
    python scripts/telegram_to_txt.py
    python scripts/telegram_to_txt.py --min-chars 30
    python scripts/telegram_to_txt.py --export-dir datasets/telegram_export/japan-justice_2026-03
"""

import argparse
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
EXPORT_DIR = BASE_DIR / "datasets" / "telegram_export"
OUTPUT_DIR = BASE_DIR / "data" / "documents"

# Map export folder prefix → output filename and RAG document set name
GROUPS = {
    "japan-justice": "japan-justice",
    "japan-move-school-visa": "japan-move-school-visa",
    "vmeste-japan": "vmeste-japan",
}


def extract_text(text_field) -> str:
    """Extract plain text from a message's text field.

    The field can be:
    - A plain string
    - A list of strings and/or dicts like {"type": "link", "text": "https://..."}

    Links (URLs) are dropped; all other text parts are joined.
    """
    if isinstance(text_field, str):
        return text_field.strip()
    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") != "link":
                parts.append(item.get("text", ""))
        return "".join(parts).strip()
    return ""


def convert(json_path: Path, out_path: Path, min_chars: int) -> tuple[int, int]:
    """Convert a single result.json to a plain text file.

    Returns (total_messages, kept_messages).
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    lines = []
    kept = 0

    for msg in messages:
        if msg.get("type") != "message":
            continue
        text = extract_text(msg.get("text", ""))
        if len(text) < min_chars:
            continue
        lines.append(text)
        kept += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
        f.write("\n")

    return len(messages), kept


def find_export_dir(prefix: str) -> Path | None:
    """Find the export directory matching a group prefix."""
    matches = sorted(EXPORT_DIR.glob(f"{prefix}_*"))
    return matches[-1] if matches else None


def resolve_export_dir(path_arg: str) -> tuple[Path, str]:
    """Resolve --export-dir argument to (directory, output_name).

    Accepts an absolute path or a path relative to the repo root.
    Output name is derived by matching the folder name against GROUPS prefixes;
    falls back to the folder name itself.
    """
    p = Path(path_arg)
    if not p.is_absolute():
        p = BASE_DIR / p
    folder_name = p.name
    name = folder_name  # default: use folder name as-is
    for prefix, group_name in GROUPS.items():
        if folder_name.startswith(prefix):
            name = group_name
            break
    return p, name


def main():
    parser = argparse.ArgumentParser(description="Convert Telegram exports to plain text")
    parser.add_argument(
        "--min-chars", type=int, default=20,
        help="Minimum character count to keep a message (default: 20)"
    )
    parser.add_argument(
        "--export-dir", metavar="DIR",
        help="Process a single export folder (relative to repo root or absolute). "
             "When omitted, all groups in GROUPS are processed."
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.export_dir:
        export_dir, name = resolve_export_dir(args.export_dir)
        jobs = [(export_dir, name)]
    else:
        jobs = []
        for prefix, name in GROUPS.items():
            export_dir = find_export_dir(prefix)
            if not export_dir:
                print(f"[SKIP] No export dir found for '{prefix}'")
                continue
            jobs.append((export_dir, name))

    for export_dir, name in jobs:
        json_path = export_dir / "result.json"
        if not json_path.exists():
            print(f"[SKIP] result.json not found in {export_dir}")
            continue

        out_path = OUTPUT_DIR / f"{name}.txt"
        total, kept = convert(json_path, out_path, args.min_chars)
        skipped = total - kept
        size_kb = out_path.stat().st_size // 1024
        print(
            f"[OK] {name}: {total} messages → {kept} kept, {skipped} skipped "
            f"(min_chars={args.min_chars}) → {out_path.name} ({size_kb} KB)"
        )


if __name__ == "__main__":
    main()
