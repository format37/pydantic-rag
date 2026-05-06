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


def convert(pdf_path: Path, output_dir: Path) -> Path:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
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


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown via Marker.")
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/documents"),
        help="Output directory for the .md and image folder (default: data/documents)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    md_path = convert(args.pdf, args.output_dir)
    images_dir = args.output_dir / args.pdf.stem
    n_images = len(list(images_dir.iterdir())) if images_dir.exists() else 0
    n_chars = md_path.stat().st_size
    print(f"Wrote {md_path} ({n_chars:,} bytes) and {n_images} image(s) in {images_dir}/")


if __name__ == "__main__":
    main()
