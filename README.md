# Pydantic RAG

Agentic RAG chatbot built with Pydantic AI, Weaviate vector database, and local Ollama inference. Features hybrid search, conversation memory, and multiple RAG modes.

## Features

- **Hybrid Search** - Combines BM25 keyword search with semantic vector search (configurable alpha)
- **3 RAG Modes** - Auto (agent decides), Force (always search), Disabled (plain chat)
- **Multimodal Support** - Image + text RAG with CLIP embeddings and VLM analysis at query time
- **Conversation Memory** - Multi-turn conversations with full context via `message_history`
- **Token Tracking** - Real-time token usage display with context limit warnings
- **Local Inference** - GPU-accelerated LLM and embedding generation via Ollama
- **MCP Server** - Expose the knowledge base as a Claude Code tool via Model Context Protocol

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | [Pydantic AI](https://ai.pydantic.dev/) |
| Vector Database | [Weaviate](https://weaviate.io/) (with text2vec-ollama) |
| LLM Inference | [Ollama](https://ollama.ai/) (llama3.2) |
| Embeddings | nomic-embed-text (768 dimensions) |
| Web UI | [Gradio](https://gradio.app/) |
| Orchestration | Docker Compose |

## Prerequisites

- Docker with Compose v2
- NVIDIA GPU with CUDA drivers (for GPU inference)
- NVIDIA Container Toolkit (`nvidia-docker`)

## Quick Start

1. **Clone and start services**:
   ```bash
   git clone <repo-url>
   cd pydantic-rag
   docker compose up -d
   ```

2. **Wait for model downloads** (first run only):
   ```bash
   # Watch Ollama logs until models are ready
   docker logs -f ollama
   ```

3. **Access the UI** at http://localhost:7860

## Document Ingestion

**Important**: place each document set in its own subdirectory under `data/documents/`. The `--name` label applies to all files found in the target directory, so mixing sources in one directory mislabels chunks.

```
data/documents/
├── my-project/          ← ingest with --name "my-project"
│   └── report.pdf
└── another-source/      ← ingest with --name "another-source"
    └── notes.txt
```

```bash
# Install dependencies
pip install weaviate-client pypdf

# Reset collection and ingest each group separately
python scripts/ingest.py --reset
python scripts/ingest.py --name "my-project"   --documents-dir data/documents/my-project
python scripts/ingest.py --name "other-source" --documents-dir data/documents/other-source
```

**Ingestion options**:
```bash
python scripts/ingest.py --help
python scripts/ingest.py --reset                                        # Reset collection only (no ingestion)
python scripts/ingest.py --name "my project" --reset                    # Delete and recreate collection, then ingest
python scripts/ingest.py --name "eu ai regulations" --extensions .pdf   # Only PDFs with label
python scripts/ingest.py --name "docs" --documents-dir ./my-docs        # Custom source folder
python scripts/ingest.py --name "code" --extensions .py,.md             # Only Python and Markdown files
python scripts/ingest.py --name "config" --extensions .yml,Dockerfile   # Extensions and exact filenames
python scripts/ingest.py --name "images" --multimodal                   # Multimodal mode with CLIP (includes images)
python scripts/ingest.py --name "mm-docs" --multimodal --reset          # Reset and reingest in multimodal mode
```

> **Note**: The `--name` option is required when ingesting documents. Use `--reset` alone to recreate the schema without ingesting.

Documents are chunked (800 tokens, 200 overlap) and embedded automatically by Weaviate's text2vec-ollama module (`bge-m3` by default — supports Russian and 100+ languages).

### Telegram Export Ingestion

To ingest Telegram chat exports (from Telegram Desktop → Export chat history → JSON format):

1. Place export directories under `datasets/telegram_export/` with a `result.json` inside each
2. Convert to plain text (strips usernames, dates, service messages, and URLs):
   ```bash
   # Process all groups defined in GROUPS dict
   python scripts/telegram_to_txt.py

   # Process a single export folder (useful when appending a new export)
   python scripts/telegram_to_txt.py --export-dir datasets/telegram_export/japan-justice_2026-03

   # Optional: adjust minimum message length (default 20 chars)
   python scripts/telegram_to_txt.py --min-chars 30
   ```
   `--export-dir` accepts a path relative to the repo root or an absolute path.
   The output name is derived by matching the folder name against `GROUPS` prefixes;
   unrecognised folders use the folder name as-is.

   This writes one `.txt` per group into `data/documents/<group>/`.

3. Reset and re-ingest:
   ```bash
   python scripts/ingest.py --reset
   for group in japan-justice japan-move-school-visa vmeste-japan; do
       python scripts/ingest.py --name "$group" --documents-dir "data/documents/$group" --extensions .txt
   done
   ```

The conversion script is configured in `scripts/telegram_to_txt.py` → `GROUPS` dict. Add new groups there to extend.

**Current knowledge base** (as of 2026-04-01):

| Document set | Source | Chunks |
|---|---|---|
| `japan-justice` | Telegram: Япония: Юридический Чат | 2,590 |
| `japan-move-school-visa` | Telegram: Japan move/school/visa group | 5,350 |
| `vmeste-japan` | Telegram: Вместе в Японию | 497 |

## Usage

### RAG Modes

| Mode | Behavior |
|------|----------|
| **Auto** | Agent decides when to search documents based on the question |
| **Force** | Always searches documents before answering |
| **Disabled** | Plain chat without document retrieval |

### Chat Interface

1. Select a RAG mode
2. Type your question and press Enter
3. View token usage in the top-right display
4. Click "Reset Chat" to clear conversation history

### Status Checks

Use the "Check Ollama" and "Check Weaviate" buttons to verify connections and see available models/collections.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Gradio UI     │────▶│   Pydantic AI   │────▶│     Ollama      │
│   (port 7860)   │     │      Agent      │     │   (port 11434)  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │    Weaviate     │
                        │   (port 8080)   │
                        └─────────────────┘
```

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `WEAVIATE_URL` | `http://weaviate:8080` | Weaviate endpoint |
| `CHAT_MODEL` | `llama3.2` | LLM for chat |
| `EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |
| `CHAT_MODEL_MULTIMODAL` | `mistral-small3.1` | Vision-language model for multimodal |
| `MULTIMODAL_MODE` | `false` | Enable multimodal mode |

## Configuration Modes

The system supports two operating modes:

### Text-Only Mode (Default)

Uses text embeddings for document retrieval.

| Component | Value |
|-----------|-------|
| Embedding | `nomic-embed-text` via text2vec-ollama |
| Chat Model | `llama3.2` |
| Collection | `Document` |
| Best for | Pure text documents |

### Multimodal Mode

Uses CLIP embeddings for cross-modal (text + image) retrieval with VLM analysis at query time.

| Component | Value |
|-----------|-------|
| Embedding | `CLIP ViT-B-32` via multi2vec-clip |
| Chat Model | `mistral-small3.1` (vision + tool calling) |
| Collection | `MultimodalDocument` |
| Best for | Mixed text and images |

**Architecture:**
```
Ingestion:
  Image → CLIP embedding → Store embedding + raw blob in Weaviate
  Text  → CLIP embedding → Store embedding + content in Weaviate

Query time:
  Query → CLIP embedding → Retrieve top-K results
                              ↓
              For image results: extract raw blobs
                              ↓
              Pass query + images to mistral-small3.1
                              ↓
              VLM reasons over actual images (not pre-generated captions)
```

**To enable multimodal mode:**

1. Ensure `multi2vec-clip` container is running (included in docker-compose.yml)
2. Set `MULTIMODAL_MODE=true` in docker-compose.yml for the app service
3. Ingest documents with the `--multimodal` flag:
   ```bash
   python scripts/ingest.py --name "my docs" --multimodal
   ```
4. Restart the app:
   ```bash
   docker compose up -d --build app
   ```

**Notes:**
- `mistral-small3.1` is ~13GB quantized, requires ~24GB VRAM for good performance
- Images are stored as raw blobs during ingestion (no caption generation - faster ingestion)
- At query time, retrieved images are passed directly to the VLM for analysis
- Hybrid search: text uses BM25 + vector, images use vector-only search (CLIP)
- Up to 3 images are passed to the VLM per query to avoid context overflow

## MCP Server (Claude Code Integration)

The knowledge base can be exposed as a native tool for [Claude Code](https://claude.ai/code) via the Model Context Protocol (MCP). This lets Claude Code search your documents directly without any manual prompting.

### Setup

1. **Install dependencies on the host** (use the same Python where `weaviate-client` is installed):
   ```bash
   pip install mcp weaviate-client
   ```

2. **Add the server to `~/.config/Claude/claude_desktop_config.json`**:
   ```json
   {
     "mcpServers": {
       "pydantic-rag": {
         "command": "/path/to/your/python3",
         "args": ["/path/to/pydantic-rag/mcp_server.py"],
         "env": {
           "WEAVIATE_URL": "http://localhost:8080",
           "WEAVIATE_GRPC_PORT": "50051"
         }
       }
     }
   }
   ```
   > Replace `/path/to/your/python3` with the Python that has the packages installed (e.g. `/home/alex/anaconda3/bin/python3`). Use `which python3` to find it.

3. **Make sure Weaviate is running**:
   ```bash
   docker compose up -d weaviate
   ```

4. **Restart Claude Desktop** to pick up the new server.

### Available Tools

| Tool | Description |
|------|-------------|
| `search_documents` | Hybrid BM25 + vector search returning ranked excerpts with source paths |
| `list_document_sets` | List available document set names in the collection |

### Tool Parameters

**`search_documents`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | — | Search query (required) |
| `n_chunks` | `5` | Number of chunks to retrieve (1–20) |
| `collection` | `"Document"` | `"Document"` or `"MultimodalDocument"` |
| `name_filter` | `null` | List of document set names to restrict search to |
| `chunk_content_size` | `1000` | Max characters per chunk in output (100–4000) |

**`list_document_sets`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `collection` | `"Document"` | Weaviate collection to inspect |

### Notes

- The MCP server connects directly to Weaviate on the host (port 8080 must be accessible)
- Uses stdio transport — Claude Code spawns it as a subprocess, no network port needed
- For text-only mode use collection `"Document"`; for multimodal use `"MultimodalDocument"`

## Troubleshooting

**Models not loading**: Check Ollama logs with `docker logs ollama`. First startup downloads ~2GB of models (more for multimodal mode with mistral-small3.1).

**Weaviate connection errors**: Ensure Weaviate is healthy with `docker compose ps`. The app will show connection status.

**Out of GPU memory**: llama3.2 (3B) requires ~4GB VRAM. For multimodal mode, mistral-small3.1 requires ~24GB VRAM.

**Images not being analyzed**: Ensure you're using "Force" RAG mode, as "Auto" mode may not always trigger search. Images require explicit BLOB property retrieval from Weaviate.

## Benchmarking

The system has been evaluated on [MRAG-Bench](https://huggingface.co/datasets/uclanlp/MRAG-Bench), a multimodal RAG benchmark testing vision-language models on multi-image reasoning tasks.

| Metric | Value |
|--------|-------|
| Overall Accuracy | 33.7% |
| Best Scenario | Scope (48.0%) |
| Best Aspect | Perspective (44.3%) |
| Questions | 1,353 |

The system performs best on perspective-based reasoning where CLIP retrieval finds relevant reference images. See [benchmark/REPORT.md](benchmark/REPORT.md) for detailed analysis, scenario breakdowns, and baseline comparisons.

For benchmark setup and running your own evaluation, see [benchmark/README.md](benchmark/README.md).
