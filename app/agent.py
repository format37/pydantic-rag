"""RAG Agent module with Pydantic AI and Weaviate hybrid search."""

import os
from dataclasses import dataclass

import weaviate
from pydantic_ai import Agent, RunContext
from weaviate.classes.query import Filter
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

# System prompts for different RAG modes
SYSTEM_PROMPT_AUTO = """You are a helpful assistant with access to a document search tool.

Use the search_documents tool when:
- The user asks about specific information that might be in documents
- You need to look up facts, regulations, or detailed information
- The question relates to document content

Do NOT use search_documents when:
- The user asks about the conversation itself (e.g., "what was my first message?")
- The user asks general knowledge questions you can answer confidently
- The user wants a simple greeting or casual chat

Always cite sources when using retrieved information. Be concise and helpful."""

SYSTEM_PROMPT_FORCE = """You are a RAG (Retrieval-Augmented Generation) assistant.

IMPORTANT INSTRUCTIONS:
1. ALWAYS use the search_documents tool before answering any question.
2. Base your answers ONLY on the retrieved context from the search results.
3. If the search returns no relevant results, clearly state that you couldn't find information about the topic.
4. Always cite the source documents when providing information (mention the source filename/title).
5. If asked about something not in the documents, say you don't have that information in the available documents.

Be concise and accurate in your responses."""

SYSTEM_PROMPT_DISABLED = """You are a helpful general-purpose assistant.
Answer questions using your general knowledge. Be helpful, accurate, and concise.
You do not have access to any document search capabilities in this mode."""


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""

    weaviate_client: weaviate.WeaviateClient
    collection_name: str = "Document"
    # Configurable retrieval parameters
    num_chunks: int = 5  # Number of chunks to retrieve
    chunk_content_size: int = 500  # Max chars to show per chunk
    name_filter: list[str] | None = None  # Filter by document set names


def create_weaviate_client() -> weaviate.WeaviateClient:
    """Create and return a Weaviate client."""
    return weaviate.connect_to_custom(
        http_host=WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0],
        http_port=int(WEAVIATE_URL.split(":")[-1]),
        http_secure=WEAVIATE_URL.startswith("https"),
        grpc_host=WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0],
        grpc_port=50051,
        grpc_secure=False,
    )


def get_available_names(client: weaviate.WeaviateClient) -> list[str]:
    """Fetch distinct document set names from Weaviate."""
    collection = client.collections.get("Document")
    result = collection.aggregate.over_all(group_by="name")
    return sorted([group.grouped_by.value for group in result.groups if group.grouped_by.value])


def create_agent(rag_mode: str = "auto") -> Agent[RAGDeps, str]:
    """Create and configure the RAG agent with the specified mode.

    Args:
        rag_mode: One of "auto", "force", or "disabled".
            - auto: Agent decides when to use search tool
            - force: Always use search tool before answering
            - disabled: No search tool available, plain chat
    """
    # Select system prompt based on mode
    system_prompts = {
        "auto": SYSTEM_PROMPT_AUTO,
        "force": SYSTEM_PROMPT_FORCE,
        "disabled": SYSTEM_PROMPT_DISABLED,
    }
    system_prompt = system_prompts.get(rag_mode.lower(), SYSTEM_PROMPT_AUTO)

    model = OpenAIChatModel(
        model_name=CHAT_MODEL,
        provider=OllamaProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
    )

    agent = Agent(
        model,
        deps_type=RAGDeps,
        system_prompt=system_prompt,
    )

    # Only register search tool if not in disabled mode
    if rag_mode.lower() != "disabled":

        @agent.tool
        async def search_documents(ctx: RunContext[RAGDeps], query: str) -> str:
            """Search documents using hybrid search (combining keyword and semantic search).

            Args:
                ctx: The run context containing dependencies.
                query: The search query to find relevant documents.

            Returns:
                Formatted search results with content, source citations, and position info.
            """
            try:
                collection = ctx.deps.weaviate_client.collections.get(ctx.deps.collection_name)

                # Build filter if document set names are specified
                where_filter = None
                if ctx.deps.name_filter:
                    where_filter = Filter.by_property("name").contains_any(ctx.deps.name_filter)

                response = collection.query.hybrid(
                    query=query,
                    alpha=0.5,  # Balance between keyword (0) and vector (1) search
                    limit=ctx.deps.num_chunks,
                    return_metadata=["score"],
                    filters=where_filter,
                )

                if not response.objects:
                    return "No relevant documents found for this query."

                results = []
                for i, obj in enumerate(response.objects, 1):
                    props = obj.properties
                    content = props.get("content", "No content available")
                    source = props.get("source", "Unknown source")
                    chunk_index = props.get("chunk_index", "?")

                    # Position metadata
                    start_line = props.get("start_line")
                    end_line = props.get("end_line")
                    page_number = props.get("page_number")

                    # Build location string
                    location_parts = []
                    if page_number:
                        location_parts.append(f"page {page_number}")
                    if start_line and end_line:
                        if start_line == end_line:
                            location_parts.append(f"line {start_line}")
                        else:
                            location_parts.append(f"lines {start_line}-{end_line}")
                    location_str = ", ".join(location_parts) if location_parts else f"chunk {chunk_index}"

                    # Truncate content based on configurable size
                    max_size = ctx.deps.chunk_content_size
                    if len(content) > max_size:
                        content = content[:max_size] + "..."

                    results.append(
                        f"[Result {i}] Source: {source} ({location_str})\n"
                        f"Content: {content}\n"
                    )

                return "\n---\n".join(results)

            except Exception as e:
                return f"Error searching documents: {e}"

    return agent


# Cache agents by mode to avoid recreating them unnecessarily
_agents: dict[str, Agent[RAGDeps, str]] = {}


def get_agent(rag_mode: str = "auto") -> Agent[RAGDeps, str]:
    """Get or create an agent instance for the specified mode.

    Args:
        rag_mode: One of "auto", "force", or "disabled".
    """
    mode = rag_mode.lower()
    if mode not in _agents:
        _agents[mode] = create_agent(mode)
    return _agents[mode]
