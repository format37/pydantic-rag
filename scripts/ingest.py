#!/usr/bin/env python3
"""
Document ingestion script for Weaviate vector database.

Reads documents from data/documents/, chunks them, and stores in Weaviate.
Weaviate's text2vec-ollama module handles embedding automatically.

Usage:
    python scripts/ingest.py [--weaviate-url URL] [--ollama-url URL]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType

# Supported file extensions
TEXT_EXTENSIONS = {".txt", ".md"}
CODE_EXTENSIONS = {".py", ".c", ".h", ".cu", ".cpp", ".js", ".ts"}
PDF_EXTENSIONS = {".pdf"}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | CODE_EXTENSIONS | PDF_EXTENSIONS

# Chunking parameters
CHUNK_SIZE = 800  # tokens (approximate as chars / 4)
CHUNK_OVERLAP = 200  # tokens


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 chars for English)."""
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[str]:
    """
    Split text into overlapping chunks.

    Uses character-based splitting with token estimation.
    Tries to split on paragraph boundaries when possible.
    """
    # Convert token counts to character counts
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4

    if len(text) <= char_chunk_size:
        yield text
        return

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
            yield chunk

        # Move start, accounting for overlap
        start = end - char_overlap
        if start >= len(text):
            break


def read_text_file(filepath: Path) -> str:
    """Read a plain text or markdown file."""
    return filepath.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(filepath: Path) -> str:
    """Read a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Error: pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    reader = PdfReader(filepath)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)


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
    """Read a document based on its extension."""
    ext = filepath.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return read_text_file(filepath)
    elif ext in CODE_EXTENSIONS:
        return read_code_file(filepath)
    elif ext in PDF_EXTENSIONS:
        return read_pdf_file(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_file_type(filepath: Path) -> str:
    """Get file type category from extension."""
    ext = filepath.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    elif ext in CODE_EXTENSIONS:
        return "code"
    elif ext in PDF_EXTENSIONS:
        return "pdf"
    return "unknown"


def create_collection(client: weaviate.WeaviateClient, ollama_url: str) -> None:
    """Create the Document collection with text2vec-ollama vectorizer."""
    collection_name = "Document"

    # Delete existing collection if it exists
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting...")
        client.collections.delete(collection_name)

    print(f"Creating collection '{collection_name}'...")
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
            Property(name="source", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="file_type", data_type=DataType.TEXT),
        ]
    )
    print(f"Collection '{collection_name}' created successfully.")


def ingest_documents(
    client: weaviate.WeaviateClient,
    documents_dir: Path,
    batch_size: int = 10
) -> int:
    """Ingest documents from directory into Weaviate."""
    collection = client.collections.get("Document")

    total_chunks = 0

    # Find all supported files
    files = [f for f in documents_dir.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print(f"No supported documents found in {documents_dir}")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return 0

    print(f"Found {len(files)} document(s) to process")

    for filepath in files:
        print(f"\nProcessing: {filepath.name}")

        try:
            # Read document
            text = read_document(filepath)
            file_type = get_file_type(filepath)

            # Chunk document
            chunks = list(chunk_text(text))
            print(f"  - {len(chunks)} chunks ({estimate_tokens(text)} estimated tokens)")

            # Insert chunks in small batches to avoid Ollama embedding timeouts
            inserted = 0
            failed = 0
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                objects = [
                    {
                        "content": chunk,
                        "source": filepath.name,
                        "chunk_index": i + idx,
                        "file_type": file_type,
                    }
                    for idx, chunk in enumerate(batch_chunks)
                ]

                try:
                    result = collection.data.insert_many(objects)
                    if result.has_errors:
                        for err in result.errors.values():
                            print(f"    Batch error: {err.message}")
                        failed += len([e for e in result.errors.values()])
                        inserted += len(batch_chunks) - len(result.errors)
                    else:
                        inserted += len(batch_chunks)

                    # Progress indicator
                    print(f"  - Progress: {min(i + batch_size, len(chunks))}/{len(chunks)} chunks", end="\r")
                except Exception as e:
                    print(f"    Batch {i//batch_size + 1} failed: {e}")
                    failed += len(batch_chunks)

            total_chunks += inserted
            print(f"  - Inserted {inserted} chunks" + (f" ({failed} failed)" if failed else ""))

        except Exception as e:
            print(f"  - Error processing {filepath.name}: {e}")

    return total_chunks


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
    args = parser.parse_args()

    # Resolve documents directory relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    documents_dir = project_root / args.documents_dir

    if not documents_dir.exists():
        print(f"Error: Documents directory not found: {documents_dir}")
        print("Please create it and add documents to ingest:")
        print(f"  mkdir -p {documents_dir}")
        sys.exit(1)

    print(f"Connecting to Weaviate at {args.weaviate_url}...")

    # Parse URL to get host and port
    from urllib.parse import urlparse
    parsed = urlparse(args.weaviate_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080

    try:
        client = weaviate.connect_to_local(host=host, port=port)

        # Check connection
        if not client.is_ready():
            print("Error: Weaviate is not ready")
            sys.exit(1)

        print("Connected to Weaviate")

        # Create or reset collection
        if args.reset or not client.collections.exists("Document"):
            create_collection(client, args.ollama_url)
        else:
            print("Using existing 'Document' collection (use --reset to recreate)")

        # Ingest documents
        total = ingest_documents(client, documents_dir)

        print(f"\nIngestion complete: {total} chunks stored in Weaviate")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
