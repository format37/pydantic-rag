# Pydantic RAG

RAG knowledge base with MCP interface for Claude Code and a Gradio chat UI. Powered by Weaviate (vector DB) and Ollama (local LLM + embeddings). Supports Russian and 100+ languages via `bge-m3` embeddings.

---

## MCP Quick Start

Expose your documents as a native tool in Claude Code — no GPU required for search.

### 1. Start Weaviate

```bash
git clone https://github.com/format37/pydantic-rag
cd pydantic-rag
docker compose up -d        # starts Weaviate only (no GPU needed)
```

### 2. Ingest documents

Place documents under `data/documents/<set-name>/`, then:

```bash
pip install weaviate-client pypdf

# Start Ollama for embedding (needed only during ingestion)
docker compose --profile app up -d ollama

# Ingest
python scripts/ingest.py --name "my-docs" --documents-dir data/documents/my-docs

# Stop Ollama when done (optional — not needed for MCP search)
docker compose --profile app stop ollama
```

> Ollama can run on CPU (no GPU required); ingestion will be slower but works fine.

### 3. Add MCP server to Claude Code

```bash
pip install mcp weaviate-client

claude mcp add-json pydantic-rag \
  '{"type":"stdio","command":"/path/to/python3","args":["/path/to/pydantic-rag/mcp_server.py"],"env":{"WEAVIATE_URL":"http://localhost:8080","WEAVIATE_GRPC_PORT":"50051"}}' \
  --scope user
```

Replace `/path/to/python3` with the Python that has the packages installed (use `which python3`).

Verify:
```bash
claude mcp list   # should show pydantic-rag as connected
```

**Claude Desktop** — add to `~/.config/Claude/claude_desktop_config.json` instead:
```json
{
  "mcpServers": {
    "pydantic-rag": {
      "command": "/path/to/python3",
      "args": ["/path/to/pydantic-rag/mcp_server.py"],
      "env": {
        "WEAVIATE_URL": "http://localhost:8080",
        "WEAVIATE_GRPC_PORT": "50051"
      }
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `search_documents` | BM25 keyword search returning ranked excerpts with source paths |
| `list_document_sets` | List available document set names in the collection |

**`search_documents` parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | — | Search query (required) |
| `n_chunks` | `5` | Number of chunks to retrieve (1–20) |
| `collection` | `"Document"` | `"Document"` or `"MultimodalDocument"` |
| `name_filter` | `null` | List of document set names to restrict search to |
| `chunk_content_size` | `1000` | Max characters per chunk in output (100–4000) |

> The MCP server uses BM25-only search (`alpha=0`) — Ollama is **not** needed while searching.

---

## Full App Quick Start (Gradio UI)

Requires Ollama for LLM inference. GPU is recommended but not required (CPU works, slower).

```bash
docker compose --profile app up -d
```

First run downloads models (~2GB for text mode). Watch progress:
```bash
docker logs -f ollama
```

Open **http://localhost:7860**

---

## Document Ingestion

Place each document set in its own subdirectory:

```
data/documents/
├── my-project/          ← ingest with --name "my-project"
│   └── report.pdf
└── another-source/      ← ingest with --name "another-source"
    └── notes.txt
```

```bash
# Reset collection and ingest each group separately
python scripts/ingest.py --reset
python scripts/ingest.py --name "my-project"   --documents-dir data/documents/my-project
python scripts/ingest.py --name "other-source" --documents-dir data/documents/other-source
```

**All options:**
```bash
python scripts/ingest.py --help
python scripts/ingest.py --reset                                        # Reset collection only
python scripts/ingest.py --name "my project" --reset                    # Reset then ingest
python scripts/ingest.py --name "eu ai regulations" --extensions .pdf   # Only PDFs
python scripts/ingest.py --name "docs" --documents-dir ./my-docs        # Custom folder
python scripts/ingest.py --name "code" --extensions .py,.md             # Python + Markdown
python scripts/ingest.py --name "config" --extensions .yml,Dockerfile
python scripts/ingest.py --name "images" --multimodal                   # CLIP mode (images + text)
```

Documents are chunked (800 tokens, 200 overlap) and embedded via Weaviate's `text2vec-ollama` module (`bge-m3` by default).

### Telegram Export Ingestion

To ingest Telegram chat exports (Telegram Desktop → Export chat history → JSON format):

1. Place export directories under `datasets/telegram_export/` with a `result.json` inside each
2. Convert to plain text:
   ```bash
   python scripts/telegram_to_txt.py
   # or a single export:
   python scripts/telegram_to_txt.py --export-dir datasets/telegram_export/japan-justice_2026-03
   ```
3. Reset and re-ingest:
   ```bash
   python scripts/ingest.py --reset
   for group in japan-justice japan-move-school-visa vmeste-japan; do
       python scripts/ingest.py --name "$group" --documents-dir "data/documents/$group" --extensions .txt
   done
   ```

The conversion script is configured in `scripts/telegram_to_txt.py` → `GROUPS` dict.

**Current knowledge base** (as of 2026-04-01):

| Document set | Source | Chunks |
|---|---|---|
| `japan-justice` | Telegram: Япония: Юридический Чат | 2,590 |
| `japan-move-school-visa` | Telegram: Japan move/school/visa group | 5,350 |
| `vmeste-japan` | Telegram: Вместе в Японию | 497 |

---

## Features

- **MCP Server** — expose the knowledge base as a Claude Code tool via Model Context Protocol
- **Hybrid Search** — BM25 keyword search; optionally combined with vector search (configurable alpha)
- **3 RAG Modes** — Auto (agent decides), Force (always search), Disabled (plain chat)
- **Multilingual** — `bge-m3` embeddings support Russian and 100+ languages
- **Multimodal Support** — Image + text RAG with CLIP embeddings and VLM analysis at query time
- **Conversation Memory** — multi-turn conversations with full context via `message_history`
- **Token Tracking** — real-time token usage display with context limit warnings

---

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

Claude Code ──MCP──▶ mcp_server.py ──▶ Weaviate (BM25, no Ollama needed)
```

**Services** (GPU-dependent services are in the `app` profile):
- **weaviate** — always starts with `docker compose up -d`, no GPU needed
- **ollama** — `--profile app`, GPU recommended (CPU works)
- **multi2vec-clip** — `--profile app`, for multimodal mode only
- **app** — `--profile app`, Gradio web UI

---

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API endpoint |
| `WEAVIATE_URL` | `http://weaviate:8080` | Weaviate endpoint |
| `CHAT_MODEL` | `llama3.2` | LLM for chat |
| `EMBED_MODEL` | `bge-m3` | Embedding model (100+ languages) |
| `CHAT_MODEL_MULTIMODAL` | `mistral-small3.1` | Vision-language model for multimodal |
| `MULTIMODAL_MODE` | `false` | Enable multimodal mode |

### Multimodal Mode

Uses CLIP embeddings for cross-modal (text + image) retrieval with VLM analysis at query time.

**To enable:**
1. Set `MULTIMODAL_MODE=true` in `docker-compose.yml`
2. Add `multi2vec-clip` and `CLIP_INFERENCE_API` to `ENABLE_MODULES` in the weaviate service env
3. Ingest with `--multimodal` flag
4. Restart: `docker compose --profile app up -d --build app`

**Notes:**
- `mistral-small3.1` (~13GB) requires ~24GB VRAM for good performance
- Images are stored as raw blobs; VLM analyzes them at query time (no pre-generated captions)
- Up to 3 images passed to VLM per query

---

## Troubleshooting

**Models not loading**: `docker logs -f ollama` — first startup downloads ~2GB of models.

**Weaviate connection errors**: `docker compose ps` — verify Weaviate is healthy.

**Slow inference without GPU**: llama3.2 (3B) runs on CPU but takes longer. GPU (~4GB VRAM) gives best performance.

**Images not being analyzed**: Use "Force" RAG mode — "Auto" may skip search. Multimodal requires `MULTIMODAL_MODE=true` and CLIP ingestion.

---

## Benchmarking

Evaluated on [MRAG-Bench](https://huggingface.co/datasets/uclanlp/MRAG-Bench):

| Metric | Value |
|--------|-------|
| Overall Accuracy | 33.7% |
| Best Scenario | Scope (48.0%) |
| Best Aspect | Perspective (44.3%) |
| Questions | 1,353 |

See [benchmark/REPORT.md](benchmark/REPORT.md) for detailed analysis and [benchmark/README.md](benchmark/README.md) for evaluation setup.
