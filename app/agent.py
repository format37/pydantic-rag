"""RAG Agent module with Pydantic AI and Weaviate hybrid search."""

import os
from dataclasses import dataclass, field

import httpx
import weaviate
from pydantic_ai import Agent, RunContext
from weaviate.classes.query import Filter
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
CHAT_MODEL_MULTIMODAL = os.getenv("CHAT_MODEL_MULTIMODAL", "mistral-small3.1")

# Maximum images to pass to VLM to avoid context overflow
MAX_IMAGES_FOR_VLM = 3

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

# Multimodal system prompts
SYSTEM_PROMPT_MULTIMODAL_AUTO = """You are a multimodal vision assistant with access to documents and images.

You can see and analyze images directly. When search results include images:
- Analyze the actual image content (you will receive the images to view)
- Describe what you see in detail relevant to the user's question
- Connect visual elements to the user's query
- Reference the source filename when discussing images

Use the search_documents tool when:
- The user asks about specific information in documents or images
- You need to look up visual content, diagrams, charts, or pictures
- The question relates to document or image content

Do NOT use search_documents when:
- The user asks about the conversation itself
- The user asks general knowledge questions you can answer confidently
- The user wants a simple greeting or casual chat

Always cite sources when using retrieved information. Be concise and helpful."""

SYSTEM_PROMPT_MULTIMODAL_FORCE = """You are a multimodal vision RAG assistant with access to documents and images.

You can see and analyze images directly.

IMPORTANT INSTRUCTIONS:
1. ALWAYS use the search_documents tool before answering any question.
2. Base your answers ONLY on the retrieved context from search results.
3. When results include images, you will receive them to view directly - analyze and describe what you see.
4. Always cite the source documents/images when providing information.
5. If the search returns no relevant results, clearly state that.

Be concise and accurate in your responses."""


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""

    weaviate_client: weaviate.WeaviateClient
    collection_name: str = "Document"
    multimodal: bool = False
    num_chunks: int = 5
    chunk_content_size: int = 500
    name_filter: list[str] | None = None
    retrieved_images: list[dict] = field(default_factory=list)


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


def get_available_names(client: weaviate.WeaviateClient, multimodal: bool = False) -> list[str]:
    """Fetch distinct document set names from Weaviate."""
    collection_name = "MultimodalDocument" if multimodal else "Document"
    collection = client.collections.get(collection_name)
    result = collection.aggregate.over_all(group_by="name")
    return sorted([group.grouped_by.value for group in result.groups if group.grouped_by.value])


def create_agent(rag_mode: str = "auto") -> Agent[RAGDeps, str]:
    """Create and configure the RAG agent with the specified mode."""
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

    if rag_mode.lower() != "disabled":

        @agent.tool
        async def search_documents(ctx: RunContext[RAGDeps], query: str) -> str:
            """Search documents using hybrid search (combining keyword and semantic search)."""
            try:
                collection = ctx.deps.weaviate_client.collections.get(ctx.deps.collection_name)

                where_filter = None
                if ctx.deps.name_filter:
                    where_filter = Filter.by_property("name").contains_any(ctx.deps.name_filter)

                response = collection.query.hybrid(
                    query=query,
                    alpha=0.5,
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

                    start_line = props.get("start_line")
                    end_line = props.get("end_line")
                    page_number = props.get("page_number")

                    location_parts = []
                    if page_number:
                        location_parts.append(f"page {page_number}")
                    if start_line and end_line:
                        if start_line == end_line:
                            location_parts.append(f"line {start_line}")
                        else:
                            location_parts.append(f"lines {start_line}-{end_line}")
                    location_str = ", ".join(location_parts) if location_parts else f"chunk {chunk_index}"

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


_agents: dict[str, Agent[RAGDeps, str]] = {}


def get_agent(rag_mode: str = "auto") -> Agent[RAGDeps, str]:
    """Get or create an agent instance for the specified mode."""
    mode = rag_mode.lower()
    if mode not in _agents:
        _agents[mode] = create_agent(mode)
    return _agents[mode]


def create_multimodal_agent(rag_mode: str = "auto") -> Agent[RAGDeps, str]:
    """Create and configure the multimodal RAG agent.

    Uses mistral-small3.1 which supports BOTH vision AND tool calling.
    Images are retrieved from Weaviate and passed to the VLM at query time.
    """
    system_prompts = {
        "auto": SYSTEM_PROMPT_MULTIMODAL_AUTO,
        "force": SYSTEM_PROMPT_MULTIMODAL_FORCE,
        "disabled": SYSTEM_PROMPT_DISABLED,
    }
    system_prompt = system_prompts.get(rag_mode.lower(), SYSTEM_PROMPT_MULTIMODAL_AUTO)

    model = OpenAIChatModel(
        model_name=CHAT_MODEL_MULTIMODAL,
        provider=OllamaProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
    )

    agent = Agent(
        model,
        deps_type=RAGDeps,
        system_prompt=system_prompt,
    )

    if rag_mode.lower() != "disabled":

        @agent.tool
        async def search_documents(ctx: RunContext[RAGDeps], query: str) -> str:
            """Search documents and images using hybrid search."""
            try:
                collection = ctx.deps.weaviate_client.collections.get(ctx.deps.collection_name)

                base_filter = None
                if ctx.deps.name_filter:
                    base_filter = Filter.by_property("name").contains_any(ctx.deps.name_filter)

                # Hybrid search for text results
                text_filter = Filter.by_property("content_type").equal("text")
                if base_filter:
                    text_filter = text_filter & base_filter

                text_response = collection.query.hybrid(
                    query=query,
                    alpha=0.5,
                    limit=ctx.deps.num_chunks,
                    return_metadata=["score"],
                    filters=text_filter,
                )

                # Vector search for images (BM25 can't match empty content)
                image_filter = Filter.by_property("content_type").equal("image")
                if base_filter:
                    image_filter = image_filter & base_filter

                image_response = collection.query.near_text(
                    query=query,
                    limit=MAX_IMAGES_FOR_VLM,
                    filters=image_filter,
                    return_properties=["source", "filename", "image", "content_type"],
                )

                if not text_response.objects and not image_response.objects:
                    return "No relevant documents or images found for this query."

                ctx.deps.retrieved_images = []
                results = []
                result_num = 0

                # Process image results first
                if image_response.objects:
                    for obj in image_response.objects:
                        props = obj.properties
                        source = props.get("source", "Unknown source")
                        image_b64 = props.get("image", "")

                        if image_b64 and len(ctx.deps.retrieved_images) < MAX_IMAGES_FOR_VLM:
                            ctx.deps.retrieved_images.append({
                                "source": source,
                                "data": image_b64,
                            })
                            result_num += 1
                            results.append(
                                f"[Result {result_num}] [IMAGE] Source: {source}\n"
                                f"(Image will be analyzed by vision model)\n"
                            )

                # Process text results
                if text_response.objects:
                    for obj in text_response.objects:
                        props = obj.properties
                        content = props.get("content", "No content available")
                        source = props.get("source", "Unknown source")
                        chunk_index = props.get("chunk_index", "?")

                        start_line = props.get("start_line")
                        end_line = props.get("end_line")
                        page_number = props.get("page_number")

                        location_parts = []
                        if page_number:
                            location_parts.append(f"page {page_number}")
                        if start_line and end_line:
                            if start_line == end_line:
                                location_parts.append(f"line {start_line}")
                            else:
                                location_parts.append(f"lines {start_line}-{end_line}")
                        location_str = ", ".join(location_parts) if location_parts else f"chunk {chunk_index}"

                        max_size = ctx.deps.chunk_content_size
                        if len(content) > max_size:
                            content = content[:max_size] + "..."

                        result_num += 1
                        results.append(
                            f"[Result {result_num}] Source: {source} ({location_str})\n"
                            f"Content: {content}\n"
                        )

                return "\n---\n".join(results)

            except Exception as e:
                return f"Error searching documents: {e}"

    return agent


_multimodal_agents: dict[str, Agent[RAGDeps, str]] = {}


def get_multimodal_agent(rag_mode: str = "auto") -> Agent[RAGDeps, str]:
    """Get or create a multimodal agent instance for the specified mode."""
    mode = rag_mode.lower()
    if mode not in _multimodal_agents:
        _multimodal_agents[mode] = create_multimodal_agent(mode)
    return _multimodal_agents[mode]


async def run_multimodal_with_images(
    agent: Agent[RAGDeps, str],
    message: str,
    deps: RAGDeps,
    message_history: list | None = None,
) -> tuple[str, list, any]:
    """Run multimodal agent with image injection for VLM analysis.

    This wrapper:
    1. Runs the agent normally (which may call search_documents tool)
    2. If images were retrieved, makes a follow-up VLM call with actual images
    3. Returns the combined response
    """
    result = await agent.run(
        message,
        deps=deps,
        message_history=message_history,
    )

    if deps.retrieved_images:
        try:
            vlm_response = await _call_vlm_with_images(
                query=message,
                text_context=result.output,
                images=deps.retrieved_images,
            )
            final_output = vlm_response if vlm_response else result.output
        except Exception as e:
            final_output = f"{result.output}\n\n(Note: Image analysis unavailable: {e})"
    else:
        final_output = result.output

    return final_output, result.all_messages(), result.usage()


async def _call_vlm_with_images(
    query: str,
    text_context: str,
    images: list[dict],
) -> str | None:
    """Call VLM directly with images for visual analysis."""
    if not images:
        return None

    image_sources = [f"- Image {i+1}: {img['source']}" for i, img in enumerate(images)]
    image_sources_text = "\n".join(image_sources)

    prompt = f"""Based on the user's question and the search results below, analyze the retrieved images and provide a comprehensive answer.

User Question: {query}

Text Context from Search:
{text_context}

Retrieved Images:
{image_sources_text}

Please analyze the images shown and answer the user's question, incorporating what you see in the images along with the text context. Be specific about visual details you observe."""

    image_data = [img["data"] for img in images]

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL_MULTIMODAL,
                "prompt": prompt,
                "images": image_data,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")
