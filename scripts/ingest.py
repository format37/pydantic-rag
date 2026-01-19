#!/usr/bin/env python3
"""
Document ingestion script for Weaviate vector database.

Reads documents from data/documents/ (including subfolders), chunks them,
and stores in Weaviate with position metadata for source linking.
Weaviate's text2vec-ollama module handles embedding automatically.

Usage:
    python scripts/ingest.py --name "my project"                      # Ingest with name/label (required)
    python scripts/ingest.py --name "my project" --reset              # Delete collection and re-ingest
    python scripts/ingest.py --reset                                  # Reset collection only (no ingestion)
    python scripts/ingest.py --name "docs" --documents-dir ./my-docs  # Custom source folder
    python scripts/ingest.py --name "code" --extensions .py,.md       # Only specific file types
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Setup logging from app module
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger("scripts.ingest")

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType

# Supported file extensions
TEXT_EXTENSIONS = {".txt", ".md"}
CODE_EXTENSIONS = {".py", ".c", ".h", ".cu", ".cpp", ".js", ".ts"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | CODE_EXTENSIONS | PDF_EXTENSIONS
MULTIMODAL_EXTENSIONS = SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS

# Chunking parameters
CHUNK_SIZE = 800  # tokens (approximate as chars / 4)
CHUNK_OVERLAP = 200  # tokens


@dataclass
class ChunkInfo:
    """Metadata for a text chunk including position information."""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    page_number: int | None = None  # Only for PDFs


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 chars for English)."""
    return len(text) // 4


def count_lines_before(text: str, position: int) -> int:
    """Count number of newlines before a position in text."""
    return text[:position].count('\n') + 1


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[ChunkInfo]:
    """
    Split text into overlapping chunks with position tracking.

    Uses character-based splitting with token estimation.
    Tries to split on paragraph boundaries when possible.

    Yields:
        ChunkInfo objects with content and position metadata.
    """
    # Convert token counts to character counts
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4

    if len(text) <= char_chunk_size:
        yield ChunkInfo(
            content=text,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            start_line=1,
            end_line=count_lines_before(text, len(text)),
        )
        return

    chunk_index = 0
    start = 0
    while start < len(text):
        end = start + char_chunk_size

        # If not at the end, try to find a good break point
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + char_chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                for sep in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + char_chunk_size // 2:
                        end = sent_break + len(sep)
                        break

        chunk = text[start:end].strip()
        if chunk:
            # Find actual positions of stripped content
            strip_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            strip_end = end - (len(text[start:end]) - len(text[start:end].rstrip()))

            yield ChunkInfo(
                content=chunk,
                chunk_index=chunk_index,
                start_char=strip_start,
                end_char=strip_end,
                start_line=count_lines_before(text, strip_start),
                end_line=count_lines_before(text, strip_end),
            )
            chunk_index += 1

        # Move start, accounting for overlap
        start = end - char_overlap
        if start >= len(text):
            break


def read_text_file(filepath: Path) -> str:
    """Read a plain text or markdown file."""
    return filepath.read_text(encoding="utf-8", errors="ignore")


@dataclass
class PDFContent:
    """PDF content with page boundary information."""
    text: str
    page_boundaries: list[tuple[int, int, int]]  # (page_num, start_char, end_char)


def read_pdf_file(filepath: Path) -> PDFContent:
    """Read a PDF file using pypdf, tracking page boundaries."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    reader = PdfReader(filepath)
    text_parts = []
    page_boundaries = []
    current_pos = 0

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text:
            start_pos = current_pos
            text_parts.append(text)
            current_pos += len(text) + 2  # +2 for "\n\n" separator
            page_boundaries.append((page_num, start_pos, current_pos - 2))

    return PDFContent(
        text="\n\n".join(text_parts),
        page_boundaries=page_boundaries,
    )


def get_page_for_position(page_boundaries: list[tuple[int, int, int]], char_pos: int) -> int | None:
    """Find which page a character position belongs to."""
    for page_num, start, end in page_boundaries:
        if start <= char_pos <= end:
            return page_num
    return None


def read_code_file(filepath: Path) -> str:
    """Read a code file with language annotation."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    ext = filepath.suffix.lower()

    # Add language hint for better context
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".c": "c",
        ".h": "c header",
        ".cpp": "c++",
        ".cu": "cuda",
    }
    lang = lang_map.get(ext, "code")

    return f"[{lang} code]\n{content}"


def read_document(filepath: Path) -> str:
    """Read a document based on its extension.

    Unknown extensions are treated as plain text files.
    """
    ext = filepath.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return read_text_file(filepath)
    elif ext in CODE_EXTENSIONS:
        return read_code_file(filepath)
    elif ext in PDF_EXTENSIONS:
        return read_pdf_file(filepath)
    else:
        # Treat unknown extensions as plain text
        return read_text_file(filepath)


def get_file_type(filepath: Path) -> str:
    """Get file type category from extension.

    Unknown extensions are categorized as 'text'.
    """
    ext = filepath.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    elif ext in CODE_EXTENSIONS:
        return "code"
    elif ext in PDF_EXTENSIONS:
        return "pdf"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    return "text"  # Treat unknown as text


def create_collection(client: weaviate.WeaviateClient, ollama_url: str) -> None:
    """Create the Document collection with text2vec-ollama vectorizer."""
    collection_name = "Document"

    # Delete existing collection if it exists
    if client.collections.exists(collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Deleting...")
        client.collections.delete(collection_name)

    logger.info(f"Creating collection '{collection_name}'...")
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(
            api_endpoint=ollama_url,
            model="nomic-embed-text"
        ),
        generative_config=Configure.Generative.ollama(
            api_endpoint=ollama_url,
            model="llama3.2"
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="folder", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),  # Full relative path
            Property(name="name", data_type=DataType.TEXT),  # Document set name/label
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="file_type", data_type=DataType.TEXT),
            # Position metadata for source linking
            Property(name="start_char", data_type=DataType.INT),
            Property(name="end_char", data_type=DataType.INT),
            Property(name="start_line", data_type=DataType.INT),
            Property(name="end_line", data_type=DataType.INT),
            Property(name="page_number", data_type=DataType.INT),  # For PDFs, nullable
        ]
    )
    logger.info(f"Collection '{collection_name}' created successfully.")


def create_multimodal_collection(client: weaviate.WeaviateClient) -> None:
    """Create MultimodalDocument collection with multi2vec-clip vectorizer."""
    collection_name = "MultimodalDocument"

    # Delete existing collection if it exists
    if client.collections.exists(collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Deleting...")
        client.collections.delete(collection_name)

    logger.info(f"Creating collection '{collection_name}'...")
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.multi2vec_clip(
            image_fields=["image"],
            text_fields=["content", "caption"],
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT),       # Text content or empty for images
            Property(name="caption", data_type=DataType.TEXT),       # AI-generated for images
            Property(name="image", data_type=DataType.BLOB),         # Base64 image data
            Property(name="content_type", data_type=DataType.TEXT),  # "text" | "image"
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="folder", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),        # Full relative path
            Property(name="name", data_type=DataType.TEXT),          # Document set name/label
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="file_type", data_type=DataType.TEXT),
            # Position metadata (for text chunks)
            Property(name="start_char", data_type=DataType.INT),
            Property(name="end_char", data_type=DataType.INT),
            Property(name="start_line", data_type=DataType.INT),
            Property(name="end_line", data_type=DataType.INT),
            Property(name="page_number", data_type=DataType.INT),
        ]
    )
    logger.info(f"Collection '{collection_name}' created successfully.")


def ingest_documents(
    client: weaviate.WeaviateClient,
    documents_dir: Path,
    name: str,
    batch_size: int = 10,
    extensions: set[str] | None = None,
    filenames: set[str] | None = None
) -> int:
    """Ingest documents from directory (recursively) into Weaviate.

    Args:
        client: Weaviate client connection
        documents_dir: Directory to scan for documents
        name: Name/label for the ingested documents (e.g., 'pydantic rag project')
        batch_size: Number of chunks to insert per batch
        extensions: Set of file extensions to include (e.g., {'.py', '.md'}).
                   If None and filenames is None, uses all SUPPORTED_EXTENSIONS.
        filenames: Set of exact filenames to include (e.g., {'Dockerfile', 'Makefile'}).
    """
    collection = client.collections.get("Document")

    total_chunks = 0

    # Determine filtering mode
    use_custom_filter = extensions or filenames

    def file_matches(f: Path) -> bool:
        """Check if file matches the filter criteria."""
        if not use_custom_filter:
            # Default: use SUPPORTED_EXTENSIONS
            return f.suffix.lower() in SUPPORTED_EXTENSIONS
        # Custom filter: match extension OR exact filename
        if extensions and f.suffix.lower() in extensions:
            return True
        if filenames and f.name in filenames:
            return True
        return False

    # Find all matching files recursively
    files = [f for f in documents_dir.rglob("*") if f.is_file() and file_matches(f)]

    if not files:
        logger.warning(f"No documents found in {documents_dir}")
        if use_custom_filter:
            parts = []
            if extensions:
                parts.append(f"extensions: {', '.join(sorted(extensions))}")
            if filenames:
                parts.append(f"filenames: {', '.join(sorted(filenames))}")
            logger.info(f"Looking for {'; '.join(parts)}")
        else:
            logger.info(f"Looking for extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return 0

    logger.info(f"Found {len(files)} document(s) to process")

    for filepath in files:
        # Calculate relative path from documents_dir
        rel_path = filepath.relative_to(documents_dir)
        folder = str(rel_path.parent) if rel_path.parent != Path(".") else ""
        filename = filepath.name
        source = str(rel_path)  # Full relative path

        logger.info(f"Processing: {source}")

        try:
            file_type = get_file_type(filepath)
            page_boundaries = None

            # Read document (handle PDFs specially for page tracking)
            if file_type == "pdf":
                pdf_content = read_pdf_file(filepath)
                text = pdf_content.text
                page_boundaries = pdf_content.page_boundaries
            else:
                text = read_document(filepath)

            # Chunk document with position tracking
            chunks = list(chunk_text(text))
            logger.info(f"  {len(chunks)} chunks ({estimate_tokens(text)} estimated tokens)")

            # Insert chunks in small batches to avoid Ollama embedding timeouts
            inserted = 0
            failed = 0
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                objects = []

                for chunk_info in batch_chunks:
                    # Determine page number for PDFs
                    page_num = None
                    if page_boundaries:
                        page_num = get_page_for_position(page_boundaries, chunk_info.start_char)

                    objects.append({
                        "content": chunk_info.content,
                        "filename": filename,
                        "folder": folder,
                        "source": source,
                        "name": name,
                        "chunk_index": chunk_info.chunk_index,
                        "file_type": file_type,
                        "start_char": chunk_info.start_char,
                        "end_char": chunk_info.end_char,
                        "start_line": chunk_info.start_line,
                        "end_line": chunk_info.end_line,
                        "page_number": page_num,
                    })

                try:
                    result = collection.data.insert_many(objects)
                    if result.has_errors:
                        for err in result.errors.values():
                            logger.error(f"  Batch error: {err.message}")
                        failed += len([e for e in result.errors.values()])
                        inserted += len(batch_chunks) - len(result.errors)
                    else:
                        inserted += len(batch_chunks)

                    # Progress indicator
                    logger.debug(f"  Progress: {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"  Batch {i//batch_size + 1} failed: {e}")
                    failed += len(batch_chunks)

            total_chunks += inserted
            logger.info(f"  Inserted {inserted} chunks" + (f" ({failed} failed)" if failed else ""))

        except Exception as e:
            logger.error(f"  Error processing {source}: {e}")

    return total_chunks


def ingest_multimodal_documents(
    client: weaviate.WeaviateClient,
    documents_dir: Path,
    name: str,
    batch_size: int = 10,
    extensions: set[str] | None = None,
    filenames: set[str] | None = None
) -> int:
    """Ingest documents and images into Weaviate MultimodalDocument collection.

    Images are stored as raw blobs without caption generation. The VLM (mistral-small3.1)
    will analyze images at query time, providing more accurate and context-aware responses.

    Args:
        client: Weaviate client connection
        documents_dir: Directory to scan for documents and images
        name: Name/label for the ingested documents
        batch_size: Number of chunks to insert per batch
        extensions: Set of file extensions to include.
                   If None and filenames is None, uses MULTIMODAL_EXTENSIONS.
        filenames: Set of exact filenames to include.
    """
    import base64

    collection = client.collections.get("MultimodalDocument")

    total_items = 0

    # Determine filtering mode
    use_custom_filter = extensions or filenames

    def file_matches(f: Path) -> bool:
        """Check if file matches the filter criteria."""
        if not use_custom_filter:
            # Default: use MULTIMODAL_EXTENSIONS (includes images)
            return f.suffix.lower() in MULTIMODAL_EXTENSIONS
        # Custom filter: match extension OR exact filename
        if extensions and f.suffix.lower() in extensions:
            return True
        if filenames and f.name in filenames:
            return True
        return False

    # Find all matching files recursively
    files = [f for f in documents_dir.rglob("*") if f.is_file() and file_matches(f)]

    if not files:
        logger.warning(f"No documents found in {documents_dir}")
        if use_custom_filter:
            parts = []
            if extensions:
                parts.append(f"extensions: {', '.join(sorted(extensions))}")
            if filenames:
                parts.append(f"filenames: {', '.join(sorted(filenames))}")
            logger.info(f"Looking for {'; '.join(parts)}")
        else:
            logger.info(f"Looking for extensions: {', '.join(sorted(MULTIMODAL_EXTENSIONS))}")
        return 0

    logger.info(f"Found {len(files)} file(s) to process")

    for filepath in files:
        # Calculate relative path from documents_dir
        rel_path = filepath.relative_to(documents_dir)
        folder = str(rel_path.parent) if rel_path.parent != Path(".") else ""
        filename = filepath.name
        source = str(rel_path)  # Full relative path

        logger.info(f"Processing: {source}")

        try:
            file_type = get_file_type(filepath)

            # Handle images differently from text
            if file_type == "image":
                logger.info("  Storing image (no caption generation - VLM will analyze at query time)...")

                # Read image as base64
                with open(filepath, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode()

                obj = {
                    "content": "",  # No text content for images
                    "caption": "",  # Empty - VLM will analyze at query time
                    "image": image_b64,
                    "content_type": "image",
                    "filename": filename,
                    "folder": folder,
                    "source": source,
                    "name": name,
                    "chunk_index": 0,
                    "file_type": "image",
                    "start_char": None,
                    "end_char": None,
                    "start_line": None,
                    "end_line": None,
                    "page_number": None,
                }

                try:
                    collection.data.insert(obj)
                    total_items += 1
                    logger.info(f"  Inserted image ({len(image_b64)} bytes base64)")
                except Exception as e:
                    logger.error(f"  Error inserting image: {e}")

            else:
                # Handle text documents (same logic as ingest_documents)
                page_boundaries = None

                if file_type == "pdf":
                    pdf_content = read_pdf_file(filepath)
                    text = pdf_content.text
                    page_boundaries = pdf_content.page_boundaries
                else:
                    text = read_document(filepath)

                # Chunk document with position tracking
                chunks = list(chunk_text(text))
                logger.info(f"  {len(chunks)} chunks ({estimate_tokens(text)} estimated tokens)")

                # Insert chunks in small batches
                inserted = 0
                failed = 0
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    objects = []

                    for chunk_info in batch_chunks:
                        # Determine page number for PDFs
                        page_num = None
                        if page_boundaries:
                            page_num = get_page_for_position(page_boundaries, chunk_info.start_char)

                        objects.append({
                            "content": chunk_info.content,
                            "caption": "",  # No caption for text
                            "image": None,  # No image data for text
                            "content_type": "text",
                            "filename": filename,
                            "folder": folder,
                            "source": source,
                            "name": name,
                            "chunk_index": chunk_info.chunk_index,
                            "file_type": file_type,
                            "start_char": chunk_info.start_char,
                            "end_char": chunk_info.end_char,
                            "start_line": chunk_info.start_line,
                            "end_line": chunk_info.end_line,
                            "page_number": page_num,
                        })

                    try:
                        result = collection.data.insert_many(objects)
                        if result.has_errors:
                            for err in result.errors.values():
                                logger.error(f"  Batch error: {err.message}")
                            failed += len([e for e in result.errors.values()])
                            inserted += len(batch_chunks) - len(result.errors)
                        else:
                            inserted += len(batch_chunks)

                        # Progress indicator
                        logger.debug(f"  Progress: {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
                    except Exception as e:
                        logger.error(f"  Batch {i//batch_size + 1} failed: {e}")
                        failed += len(batch_chunks)

                total_items += inserted
                logger.info(f"  Inserted {inserted} chunks" + (f" ({failed} failed)" if failed else ""))

        except Exception as e:
            logger.error(f"  Error processing {source}: {e}")

    return total_items


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Weaviate")
    parser.add_argument(
        "--weaviate-url",
        default=os.environ.get("WEAVIATE_URL", "http://localhost:8080"),
        help="Weaviate URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_DOCKER_URL", "http://ollama:11434"),
        help="Ollama URL for Weaviate to use (Docker network, default: http://ollama:11434)"
    )
    parser.add_argument(
        "--documents-dir",
        default="data/documents",
        help="Directory containing documents to ingest (default: data/documents)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset (delete and recreate) the collection before ingesting"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated extensions or filenames (e.g., '.py,.yml,Dockerfile'). Default: all supported"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name/label for the ingested documents (e.g., 'pydantic rag project'). Required when ingesting."
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Use multimodal mode with CLIP embeddings (creates MultimodalDocument collection)"
    )
    args = parser.parse_args()

    # Resolve documents directory relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    documents_dir = project_root / args.documents_dir

    if not documents_dir.exists():
        logger.error(f"Documents directory not found: {documents_dir}")
        logger.error("Please create it and add documents to ingest:")
        logger.error(f"  mkdir -p {documents_dir}")
        sys.exit(1)

    logger.info(f"Connecting to Weaviate at {args.weaviate_url}...")

    # Parse URL to get host and port
    from urllib.parse import urlparse
    parsed = urlparse(args.weaviate_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080

    try:
        client = weaviate.connect_to_local(host=host, port=port)

        # Check connection
        if not client.is_ready():
            logger.error("Weaviate is not ready")
            sys.exit(1)

        logger.info("Connected to Weaviate")

        # Determine which collection to use based on mode
        if args.multimodal:
            collection_name = "MultimodalDocument"
            logger.info("Mode: Multimodal (CLIP + LLaVA)")
        else:
            collection_name = "Document"
            logger.info("Mode: Text-only (nomic-embed-text + llama3.2)")

        # Create or reset collection
        if args.reset or not client.collections.exists(collection_name):
            if args.multimodal:
                create_multimodal_collection(client)
            else:
                create_collection(client, args.ollama_url)
            # If reset-only (no --name), exit after resetting
            if args.reset and not args.name:
                logger.info(f"Collection '{collection_name}' reset complete. Use --name to ingest documents.")
                return
        else:
            logger.info(f"Using existing '{collection_name}' collection (use --reset to recreate)")

        # Require --name for ingestion
        if not args.name:
            logger.error("--name is required when ingesting documents")
            sys.exit(1)

        # Parse extensions/filenames if provided
        extensions = None
        filenames = None
        if args.extensions:
            raw_filters = {f.strip() for f in args.extensions.split(",") if f.strip()}

            if not raw_filters:
                logger.error("No extensions specified")
                sys.exit(1)

            # Separate extensions (start with .) from exact filenames
            extensions = set()
            filenames = set()
            for f in raw_filters:
                if f.startswith("."):
                    extensions.add(f.lower())
                elif "." in f:
                    # Has extension like "file.txt" - treat as extension
                    extensions.add(f".{f.rsplit('.', 1)[1].lower()}")
                else:
                    # No dot - exact filename like "Dockerfile", "Makefile"
                    filenames.add(f)

            # Report what we're filtering for
            filter_parts = []
            if extensions:
                filter_parts.append(f"extensions: {', '.join(sorted(extensions))}")
            if filenames:
                filter_parts.append(f"filenames: {', '.join(sorted(filenames))}")
            logger.info(f"Filtering for {'; '.join(filter_parts)}")

        # Ingest documents
        logger.info(f"Ingesting with name: '{args.name}'")
        if args.multimodal:
            total = ingest_multimodal_documents(
                client, documents_dir,
                name=args.name,
                extensions=extensions,
                filenames=filenames
            )
            logger.info(f"Ingestion complete: {total} items stored in Weaviate (MultimodalDocument)")
        else:
            total = ingest_documents(client, documents_dir, name=args.name, extensions=extensions, filenames=filenames)
            logger.info(f"Ingestion complete: {total} chunks stored in Weaviate (Document)")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
