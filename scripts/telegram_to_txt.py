#!/usr/bin/env python3
"""Convert Telegram JSON exports to plain text files for RAG ingestion.

Extracts message text only (no usernames, dates, or service messages).
Filters out very short messages (< MIN_CHARS chars after extraction).
Links are stripped; surrounding plain text is kept.

Usage:
    python scripts/telegram_to_txt.py
    python scripts/telegram_to_txt.py --min-chars 30
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


def main():
    parser = argparse.ArgumentParser(description="Convert Telegram exports to plain text")
    parser.add_argument(
        "--min-chars", type=int, default=20,
        help="Minimum character count to keep a message (default: 20)"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for prefix, name in GROUPS.items():
        export_dir = find_export_dir(prefix)
        if not export_dir:
            print(f"[SKIP] No export dir found for '{prefix}'")
            continue

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
