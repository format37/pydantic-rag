#!/usr/bin/env python3
"""Convert a scientific PDF to Markdown using Marker (VikParuchuri/marker).

Output layout:
    <output-dir>/<pdf-stem>.md
    <output-dir>/<pdf-stem>/<image>.{png,jpg}

Image references inside the .md are rewritten to `<pdf-stem>/<image>.{ext}`
so the .md sits as a sibling of its image folder and can be moved as a pair.

Usage:
    python scripts/convert_pdf.py datasets/RL/foo.pdf --output-dir data/documents/RL
"""

import argparse
import re
import sys
from pathlib import Path

# Make the project's `scripts` package importable so Marker's
# `--llm_service scripts.marker_sdk_service.…` can resolve via importlib.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def build_converter(use_llm: bool = False):
    """Construct a PdfConverter (loads Marker's models — slow, ~30s).

    Reuse the returned object across many PDFs to avoid reloading models.
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config: dict = {}
    llm_service: str | None = None
    if use_llm:
        config["use_llm"] = True
        llm_service = "scripts.marker_sdk_service.ClaudeAgentSdkService"

    return PdfConverter(
        artifact_dict=create_model_dict(),
        config=config or None,
        llm_service=llm_service,
    )


def convert_one(converter, pdf_path: Path, output_dir: Path) -> Path:
    """Convert a single PDF using a pre-built converter. Returns the .md path."""
    from marker.output import text_from_rendered

    rendered = converter(str(pdf_path))
    text, _ext, images = text_from_rendered(rendered)

    stem = pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{stem}.md"
    images_dir = output_dir / stem

    if images:
        images_dir.mkdir(exist_ok=True)
        for name, img in images.items():
            target = images_dir / name
            fmt = "JPEG" if target.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
            img.save(target, format=fmt)
            text = text.replace(f"]({name})", f"]({stem}/{name})")

    md_path.write_text(text, encoding="utf-8")
    return md_path


def convert(pdf_path: Path, output_dir: Path, use_llm: bool = False) -> Path:
    """Build a converter and process one PDF. Convenience wrapper for single-shot use."""
    return convert_one(build_converter(use_llm), pdf_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown via Marker.")
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/documents"),
        help="Output directory for the .md and image folder (default: data/documents)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Run Marker's LLM cleanup pass via the Claude Agent SDK (uses Max subscription).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="After Marker, run a second SDK pass that reconciles the markdown "
        "against the source PDF and fixes residual conversion errors "
        "(HTML <sup>/<sub>, detached minus signs, broken pseudocode, etc.).",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    md_path = convert(args.pdf, args.output_dir, use_llm=args.use_llm)
    images_dir = args.output_dir / args.pdf.stem
    n_images = len(list(images_dir.iterdir())) if images_dir.exists() else 0
    n_chars = md_path.stat().st_size
    print(f"Wrote {md_path} ({n_chars:,} bytes) and {n_images} image(s) in {images_dir}/")

    if args.cleanup:
        import anyio

        from scripts.cleanup_md_with_pdf import run as cleanup_run

        print(f"\n--- Pass 2: SDK cleanup against {args.pdf} ---")
        anyio.run(cleanup_run, args.pdf, md_path)


if __name__ == "__main__":
    main()
