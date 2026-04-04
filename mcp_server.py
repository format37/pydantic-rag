#!/usr/bin/env python3
"""MCP server exposing Weaviate hybrid search as Claude Code tools.

Run as a subprocess via Claude Code's MCP stdio transport.
Connects to Weaviate on WEAVIATE_URL (default: http://localhost:8080).

Usage:
    pip install mcp weaviate-client
    python mcp_server.py  # started automatically by Claude Code
"""

import os

import weaviate
from mcp.server.fastmcp import FastMCP
from weaviate.classes.query import Filter

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

mcp = FastMCP("pydantic-rag")


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
) -> str:
    """Search the RAG knowledge base using hybrid BM25 + vector search.

    Args:
        query: The search query.
        n_chunks: Number of chunks to retrieve (1–20). Default 5.
        collection: Weaviate collection — "Document" (text-only) or
                    "MultimodalDocument" (multimodal). Default "Document".
        name_filter: Optional list of document-set names to restrict search to.
        chunk_content_size: Max characters per chunk shown (100–4000). Default 1000.

    Returns:
        Formatted search results with source paths and content excerpts.
    """
    client = _connect()
    try:
        coll = client.collections.get(collection)

        where_filter = None
        if name_filter:
            where_filter = Filter.by_property("name").contains_any(name_filter)

        response = coll.query.hybrid(
            query=query,
            alpha=0,  # BM25-only; no Ollama/embeddings required
            limit=max(1, min(20, n_chunks)),
            return_metadata=["score"],
            filters=where_filter,
        )

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

            results.append(
                f"[Result {i}] Source: {source} ({location})\n"
                f"Content: {content}\n"
            )

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
