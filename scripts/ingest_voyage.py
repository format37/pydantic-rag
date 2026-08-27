#!/usr/bin/env python3
"""Ingest documents (.md/.txt/code, plus .pdf via pypdf) into Weaviate using
voyage-context-3 embeddings.

The Voyage `contextualized_embed` API takes a *list of chunk lists per document*
and embeds each chunk conditioned on the full document — better than independent
chunk embeddings on long-form scientific text. We therefore:

1. Read each Markdown file in --documents-dir
2. Chunk it (reuses chunking logic from scripts.ingest)
3. Call `vo.contextualized_embed([chunks])` once per document
4. Insert each chunk into Weaviate with the matching explicit vector

Weaviate is configured with `vectorizer=none` for this collection, so query-time
search must also supply explicit vectors (computed via voyage-context-3 with
`input_type="query"`). See README for the search-side wiring.

Env (loaded from .env):
    VOYAGEAI_API_KEY  required

Usage:
    python scripts/ingest_voyage.py --reset
    python scripts/ingest_voyage.py --name "RL" --documents-dir data/documents/RL
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

# Load .env BEFORE anything that needs the key
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Reuse chunking + ChunkInfo from the existing ingest module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger("scripts.ingest_voyage")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.ingest import (  # noqa: E402
    ChunkInfo,
    chunk_text,
    estimate_tokens,
    get_page_for_position,
    read_pdf_file,
    read_text_file,
)

import voyageai
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter


COLLECTION_NAME = "Document"
VOYAGE_MODEL = "voyage-context-3"
VOYAGE_DIM = 1024  # voyage-context-3 default; also supports 256 / 512 / 2048
VOYAGE_DOC_TOKEN_LIMIT = 30000  # margin under voyage-context-3's 32K per-doc cap
IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def create_collection(client: weaviate.WeaviateClient) -> None:
    """Create Document collection with vectorizer=none (we supply vectors)."""
    if client.collections.exists(COLLECTION_NAME):
        logger.warning("Collection '%s' exists; deleting.", COLLECTION_NAME)
        client.collections.delete(COLLECTION_NAME)

    logger.info("Creating collection '%s' (vectorizer=none, dim=%d).", COLLECTION_NAME, VOYAGE_DIM)
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="folder", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="name", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="file_type", data_type=DataType.TEXT),
            Property(name="start_char", data_type=DataType.INT),
            Property(name="end_char", data_type=DataType.INT),
            Property(name="start_line", data_type=DataType.INT),
            Property(name="end_line", data_type=DataType.INT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="image_paths", data_type=DataType.TEXT_ARRAY),
        ],
    )


def extract_image_paths(content: str) -> list[str]:
    """Pull markdown image references out of a chunk for downstream agents."""
    return IMAGE_REF_RE.findall(content)


def split_for_token_limit(
    vo: voyageai.Client, chunks: list[str], limit: int = VOYAGE_DOC_TOKEN_LIMIT
) -> list[list[str]]:
    """Greedy-pack chunks into sub-batches each ≤ `limit` tokens.

    voyage-context-3 caps a single contextualized_embed input at 32K tokens.
    Long documents must be split; chunks within a sub-batch share document
    context, chunks across sub-batches do not.
    """
    counts = [vo.count_tokens([c], model=VOYAGE_MODEL) for c in chunks]
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for chunk, n in zip(chunks, counts):
        if n > limit:
            logger.warning("  Single chunk exceeds %d tokens (%d); voyage will reject.", limit, n)
        if cur and cur_tokens + n > limit:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(chunk)
        cur_tokens += n
    if cur:
        batches.append(cur)
    return batches


def embed_document(vo: voyageai.Client, chunks: list[str]) -> list[list[float]]:
    """Voyage embedding for a whole document, split as needed for the 32K cap.

    Returns one vector per input chunk (in input order).
    """
    sub_batches = split_for_token_limit(vo, chunks)
    if len(sub_batches) > 1:
        logger.info("  Splitting into %d sub-batches for voyage's 32K context cap.", len(sub_batches))
    vectors: list[list[float]] = []
    for sub in sub_batches:
        result = vo.contextualized_embed(
            inputs=[sub],
            model=VOYAGE_MODEL,
            input_type="document",
            output_dimension=VOYAGE_DIM,
        )
        vectors.extend(result.results[0].embeddings)
    return vectors


def ingest_file(
    client: weaviate.WeaviateClient,
    vo: voyageai.Client,
    filepath: Path,
    documents_dir: Path,
    name: str,
    replace_existing: bool = False,
) -> int:
    rel_path = filepath.relative_to(documents_dir)
    folder = str(rel_path.parent) if rel_path.parent != Path(".") else ""

    page_boundaries = None
    if filepath.suffix.lower() == ".pdf":
        pdf_content = read_pdf_file(filepath)
        text = pdf_content.text
        page_boundaries = pdf_content.page_boundaries
    else:
        text = read_text_file(filepath)
    chunks: list[ChunkInfo] = list(chunk_text(text))
    if not chunks:
        logger.warning("  No chunks produced for %s", rel_path)
        return 0
    if page_boundaries:
        for c in chunks:
            c.page_number = get_page_for_position(page_boundaries, c.start_char)

    logger.info("  %d chunks (%d est. tokens)", len(chunks), estimate_tokens(text))

    if replace_existing:
        collection = client.collections.get(COLLECTION_NAME)
        delete_result = collection.data.delete_many(
            where=Filter.by_property("source").equal(str(rel_path))
        )
        cleared = getattr(delete_result, "successful", 0)
        if cleared:
            logger.info("  Cleared %d existing chunks (source=%s).", cleared, rel_path)

    t0 = time.monotonic()
    vectors = embed_document(vo, [c.content for c in chunks])
    logger.info("  Voyage embedding: %.1fs", time.monotonic() - t0)

    if len(vectors) != len(chunks):
        logger.error(
            "  Voyage returned %d vectors for %d chunks; aborting file.",
            len(vectors), len(chunks),
        )
        return 0

    collection = client.collections.get(COLLECTION_NAME)
    inserted = 0
    with collection.batch.dynamic() as batch:
        for chunk, vec in zip(chunks, vectors):
            batch.add_object(
                properties={
                    "content": chunk.content,
                    "filename": filepath.name,
                    "folder": folder,
                    "source": str(rel_path),
                    "name": name,
                    "chunk_index": chunk.chunk_index,
                    "file_type": "pdf" if page_boundaries else "text",
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "page_number": chunk.page_number,
                    "image_paths": extract_image_paths(chunk.content),
                },
                vector=vec,
            )
            inserted += 1

    failed = collection.batch.failed_objects
    if failed:
        logger.error("  %d objects failed: %s", len(failed), failed[0].message[:200])
    return inserted - len(failed)


def main():
    parser = argparse.ArgumentParser(description="Ingest markdown via voyage-context-3.")
    parser.add_argument(
        "--weaviate-url",
        default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"),
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=Path("data/documents"),
        help="Folder of documents to ingest (recursive), or a single file.",
    )
    parser.add_argument("--name", help="Document-set label stored on each chunk.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Document collection before ingesting.",
    )
    parser.add_argument(
        "--extensions",
        default=".md",
        help="Comma-separated extensions to ingest (default: .md). "
        "PDFs are extracted with pypdf, e.g. --extensions .md,.txt,.pdf",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Before inserting each file's chunks, delete any existing chunks "
        "whose `source` field matches its relative path. Makes re-runs idempotent.",
    )
    args = parser.parse_args()

    if not os.environ.get("VOYAGEAI_API_KEY"):
        sys.exit("VOYAGEAI_API_KEY not set (check .env)")

    extensions = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    extensions = {e if e.startswith(".") else f".{e}" for e in extensions}

    from urllib.parse import urlparse
    parsed = urlparse(args.weaviate_url)
    client = weaviate.connect_to_local(
        host=parsed.hostname or "localhost",
        port=parsed.port or 8080,
    )

    try:
        if not client.is_ready():
            sys.exit("Weaviate not ready")

        if args.reset or not client.collections.exists(COLLECTION_NAME):
            create_collection(client)
            if args.reset and not args.name:
                logger.info("Reset complete; pass --name and --documents-dir to ingest.")
                return

        if not args.name:
            sys.exit("--name is required when ingesting")

        if args.documents_dir.is_file():
            # Single file passed directly: ingest just it, skip the extensions filter.
            documents_dir = args.documents_dir.parent
            files = [args.documents_dir]
        elif args.documents_dir.is_dir():
            documents_dir = args.documents_dir
            files = sorted(
                f for f in documents_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in extensions
            )
            if not files:
                sys.exit(f"No files matching {extensions} in {documents_dir}")
        else:
            sys.exit(f"--documents-dir not found: {args.documents_dir}")

        vo = voyageai.Client(api_key=os.environ["VOYAGEAI_API_KEY"])

        logger.info("Ingesting %d file(s) into '%s' as set '%s'.", len(files), COLLECTION_NAME, args.name)
        total = 0
        for i, f in enumerate(files, 1):
            logger.info("[%d/%d] %s", i, len(files), f.relative_to(documents_dir))
            try:
                total += ingest_file(
                    client, vo, f, documents_dir, args.name,
                    replace_existing=args.replace_existing,
                )
            except Exception as exc:
                logger.error("  Failed: %s", exc)
        logger.info("Ingestion complete: %d chunks stored.", total)
    finally:
        client.close()


if __name__ == "__main__":
    main()
