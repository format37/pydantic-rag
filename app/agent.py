"""RAG Agent module with Pydantic AI and Weaviate hybrid search."""

import os
from dataclasses import dataclass

import weaviate
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a helpful RAG (Retrieval-Augmented Generation) assistant.

IMPORTANT INSTRUCTIONS:
1. ALWAYS use the search_documents tool before answering any question about documents or information.
2. Base your answers ONLY on the retrieved context from the search results.
3. If the search returns no relevant results, clearly state that you couldn't find information about the topic.
4. Always cite the source documents when providing information (mention the source filename/title).
5. If asked about something not in the documents, say you don't have that information in the available documents.

Be concise and accurate in your responses."""


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""

    weaviate_client: weaviate.WeaviateClient
    collection_name: str = "Document"


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


def create_agent() -> Agent[RAGDeps, str]:
    """Create and configure the RAG agent."""
    model = OpenAIChatModel(
        model_name=CHAT_MODEL,
        provider=OllamaProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
    )

    agent = Agent(
        model,
        deps_type=RAGDeps,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    async def search_documents(ctx: RunContext[RAGDeps], query: str) -> str:
        """Search documents using hybrid search (combining keyword and semantic search).

        Args:
            ctx: The run context containing dependencies.
            query: The search query to find relevant documents.

        Returns:
            Formatted search results with content and source citations.
        """
        try:
            collection = ctx.deps.weaviate_client.collections.get(ctx.deps.collection_name)

            response = collection.query.hybrid(
                query=query,
                alpha=0.5,  # Balance between keyword (0) and vector (1) search
                limit=5,
                return_metadata=["score"],
            )

            if not response.objects:
                return "No relevant documents found for this query."

            results = []
            for i, obj in enumerate(response.objects, 1):
                props = obj.properties
                content = props.get("content", "No content available")
                source = props.get("source", "Unknown source")
                chunk_index = props.get("chunk_index", "?")

                # Truncate content if too long
                if len(content) > 500:
                    content = content[:500] + "..."

                results.append(
                    f"[Result {i}] Source: {source} (chunk {chunk_index})\n"
                    f"Content: {content}\n"
                )

            return "\n---\n".join(results)

        except Exception as e:
            return f"Error searching documents: {e}"

    return agent


# Singleton agent instance
_agent: Agent[RAGDeps, str] | None = None


def get_agent() -> Agent[RAGDeps, str]:
    """Get or create the singleton agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent
