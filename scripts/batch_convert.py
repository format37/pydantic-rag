#!/usr/bin/env python3
"""Batch-convert all PDFs in a directory to Markdown.

Same flags as `convert_pdf.py`, but loads Marker's models once and reuses
them for every PDF — avoids the ~30s startup cost per file. Failures on
individual PDFs are logged and the loop continues.

Usage:
    python scripts/batch_convert.py datasets/RL --output-dir data/documents/RL
    python scripts/batch_convert.py datasets/RL --output-dir data/documents/RL --use-llm --cleanup
    python scripts/batch_convert.py datasets/RL --output-dir data/documents/RL --skip-existing
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# Make the project's `scripts` package importable so Marker's
# `--llm_service scripts.marker_sdk_service.…` resolves via importlib.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.convert_pdf import build_converter, convert_one  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("input_dir", type=Path, help="Directory containing PDFs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write <stem>.md and the <stem>/ image folder for each PDF",
    )
    parser.add_argument("--use-llm", action="store_true", help="Marker LLM cleanup pass")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Pass-2 SDK reconciliation against the source PDF (per file).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs whose <stem>.md already exists in --output-dir.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for files inside input_dir (default: *.pdf).",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Not a directory: {args.input_dir}")

    pdfs = sorted(args.input_dir.glob(args.pattern))
    if not pdfs:
        sys.exit(f"No PDFs matching {args.pattern} in {args.input_dir}")

    print(f"Loading Marker models (one-time)...")
    converter = build_converter(use_llm=args.use_llm)

    cleanup_run = None
    if args.cleanup:
        import anyio  # noqa: F401  (used below)

        from scripts.cleanup_md_with_pdf import run as _cleanup_run
        cleanup_run = _cleanup_run

    failures: list[tuple[str, str]] = []
    skipped: list[str] = []
    total = len(pdfs)

    for i, pdf in enumerate(pdfs, 1):
        stem = pdf.stem
        md_path = args.output_dir / f"{stem}.md"

        if args.skip_existing and md_path.exists():
            print(f"[{i}/{total}] SKIP (exists): {pdf.name}")
            skipped.append(pdf.name)
            continue

        t0 = time.monotonic()
        print(f"[{i}/{total}] Converting {pdf.name}...")
        try:
            md_path = convert_one(converter, pdf, args.output_dir)
            n_images = len(list((args.output_dir / stem).iterdir())) if (args.output_dir / stem).exists() else 0
            print(
                f"     pass 1 done in {time.monotonic()-t0:.1f}s  "
                f"→ {md_path.name} ({md_path.stat().st_size:,} bytes), {n_images} image(s)"
            )

            if cleanup_run is not None:
                import anyio
                t1 = time.monotonic()
                print(f"     pass 2: SDK cleanup against {pdf.name}...")
                anyio.run(cleanup_run, pdf, md_path)
                print(f"     pass 2 done in {time.monotonic()-t1:.1f}s")
        except Exception as exc:
            traceback.print_exc()
            failures.append((pdf.name, f"{type(exc).__name__}: {exc}"))

    converted = total - len(failures) - len(skipped)
    print(f"\n=== {converted}/{total} converted, {len(skipped)} skipped, {len(failures)} failed ===")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
