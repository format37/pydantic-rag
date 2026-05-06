#!/usr/bin/env python3
"""MCP server exposing Weaviate hybrid search as Claude Code tools.

Run as a subprocess via Claude Code's MCP stdio transport.
Connects to Weaviate on WEAVIATE_URL (default: http://localhost:8080).

Usage:
    pip install mcp weaviate-client
    python mcp_server.py  # started automatically by Claude Code
"""

import os
from pathlib import Path

# Load .env from the project root so VOYAGEAI_API_KEY is available regardless
# of the cwd from which Claude Code spawned this subprocess.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import weaviate
from mcp.server.fastmcp import FastMCP
from weaviate.classes.query import Filter

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
VOYAGE_API_KEY = os.getenv("VOYAGEAI_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-context-3")
VOYAGE_DIM = int(os.getenv("VOYAGE_DIM", "1024"))

_voyage_client = None  # lazy

mcp = FastMCP("pydantic-rag")


def _voyage():
    """Return a cached voyage client if VOYAGEAI_API_KEY is set, else None."""
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client
    if not VOYAGE_API_KEY:
        return None
    try:
        import voyageai
        _voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        return _voyage_client
    except ImportError:
        return None


def _embed_query(query: str) -> list[float] | None:
    """Embed a search query via voyage-context-3 (input_type='query').

    Returns None if voyage isn't configured/available — caller should fall back
    to BM25-only search.
    """
    vo = _voyage()
    if vo is None:
        return None
    result = vo.contextualized_embed(
        inputs=[[query]],
        model=VOYAGE_MODEL,
        input_type="query",
        output_dimension=VOYAGE_DIM,
    )
    return result.results[0].embeddings[0]


def _connect() -> weaviate.WeaviateClient:
    host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
    port = int(WEAVIATE_URL.split(":")[-1])
    secure = WEAVIATE_URL.startswith("https")
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=secure,
        grpc_host=host,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
    )


@mcp.tool()
def search_documents(
    query: str,
    n_chunks: int = 5,
    collection: str = "Document",
    name_filter: list[str] | None = None,
    chunk_content_size: int = 1000,
    alpha: float | None = None,
) -> str:
    """Search the RAG knowledge base using hybrid BM25 + vector search.

    If VOYAGEAI_API_KEY is configured, the query is embedded via
    voyage-context-3 (input_type='query') and a hybrid search is performed
    against Weaviate (default alpha=0.5). Otherwise falls back to pure BM25.

    Args:
        query: The search query.
        n_chunks: Number of chunks to retrieve (1–20). Default 5.
        collection: Weaviate collection — typically "Document". Default "Document".
        name_filter: Optional list of document-set names to restrict search to.
        chunk_content_size: Max characters per chunk shown (100–4000). Default 1000.
        alpha: Hybrid weight (0=BM25 only, 1=vector only). Default 0.5 when
               voyage is available; 0 (BM25-only) otherwise.

    Returns:
        Formatted search results with source paths, image refs, and excerpts.
    """
    client = _connect()
    try:
        coll = client.collections.get(collection)

        where_filter = None
        if name_filter:
            where_filter = Filter.by_property("name").contains_any(name_filter)

        query_vector = _embed_query(query)
        if alpha is None:
            alpha = 0.5 if query_vector is not None else 0.0

        hybrid_kwargs = dict(
            query=query,
            alpha=alpha,
            limit=max(1, min(20, n_chunks)),
            return_metadata=["score"],
            filters=where_filter,
        )
        if query_vector is not None:
            hybrid_kwargs["vector"] = query_vector

        response = coll.query.hybrid(**hybrid_kwargs)

        if not response.objects:
            return "No relevant documents found for this query."

        max_size = max(100, min(4000, chunk_content_size))
        results = []
        for i, obj in enumerate(response.objects, 1):
            props = obj.properties
            content = props.get("content", "")
            source = props.get("source", "Unknown source")
            chunk_index = props.get("chunk_index", "?")
            start_line = props.get("start_line")
            end_line = props.get("end_line")
            page_number = props.get("page_number")
            image_paths = props.get("image_paths") or []

            location_parts = []
            if page_number:
                location_parts.append(f"page {page_number}")
            if start_line and end_line:
                location_parts.append(
                    f"line {start_line}" if start_line == end_line
                    else f"lines {start_line}-{end_line}"
                )
            location = ", ".join(location_parts) if location_parts else f"chunk {chunk_index}"

            if len(content) > max_size:
                content = content[:max_size] + "..."

            block = f"[Result {i}] Source: {source} ({location})\n"
            if image_paths:
                block += f"Images: {', '.join(image_paths)}\n"
            block += f"Content: {content}\n"
            results.append(block)

        return "\n---\n".join(results)

    except Exception as e:
        return f"Error searching documents: {e}"
    finally:
        client.close()


@mcp.tool()
def list_document_sets(collection: str = "Document") -> str:
    """List available document set names in the RAG knowledge base.

    Args:
        collection: Weaviate collection to inspect. Default "Document".

    Returns:
        Newline-separated list of document set names, or a message if empty.
    """
    client = _connect()
    try:
        coll = client.collections.get(collection)
        result = coll.aggregate.over_all(group_by="name")
        names = sorted(g.grouped_by.value for g in result.groups if g.grouped_by.value)
        if not names:
            return "No document sets found in collection."
        return "Available document sets:\n" + "\n".join(f"  - {n}" for n in names)
    except Exception as e:
        return f"Error listing document sets: {e}"
    finally:
        client.close()


if __name__ == "__main__":
    mcp.run()
