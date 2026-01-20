# MRAG-Bench Evaluation

This directory contains scripts for evaluating the multimodal RAG system using the [MRAG-Bench](https://huggingface.co/datasets/uclanlp/MRAG-Bench) benchmark.

## Overview

MRAG-Bench is a benchmark for evaluating multimodal retrieval-augmented generation systems. It contains 1,353 multiple-choice questions across 9 scenarios, where answering correctly requires retrieving and understanding relevant images from a corpus of ~16K ground-truth images.

## Quick Start

```bash
# 1. Install dependencies
pip install -r benchmark/requirements.txt

# 2. Create output directory (if data/ is root-owned from Docker)
sudo mkdir -p data/mrag_bench && sudo chown -R $USER:$USER data/mrag_bench

# 3. Download the dataset
python -m benchmark.mrag_bench.download --output data/mrag_bench

# 4. Prepare corpus for ingestion
python -m benchmark.mrag_bench.prepare_corpus

# 5. Ingest into Weaviate (multimodal mode)
python scripts/ingest.py --name mrag_bench --multimodal

# 6. Start services
docker compose up -d

# 7. Run evaluation (subset for testing)
python -m benchmark.mrag_bench.evaluate --limit 10 --output benchmark/results/test_run.json

# 8. Run full evaluation
python -m benchmark.mrag_bench.evaluate --output benchmark/results/full_run.json
```

## Dataset Structure

After downloading, the dataset is organized as:

```
data/mrag_bench/
├── metadata.json          # Questions, choices, answers, image mappings
├── images/                # Query images (1,353)
│   ├── question_0.png
│   ├── question_1.png
│   └── ...
└── corpus/                # Ground-truth images (~16K)
    ├── scenario_name/
    │   ├── image_0.png
    │   └── ...
    └── ...
```

## Scripts

| Script | Description |
|--------|-------------|
| `download.py` | Download MRAG-Bench from HuggingFace |
| `prepare_corpus.py` | Symlink corpus for ingestion pipeline |
| `ingest.py` | Wrapper for document ingestion |
| `evaluate.py` | Main evaluation client (Gradio API) |
| `metrics.py` | Accuracy calculation and reporting |
| `answer_extractor.py` | Extract A/B/C/D from VLM responses |
| `baselines.py` | Published baseline results |

## Evaluation Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Download   │────▶│   Prepare   │────▶│   Ingest    │
│  Dataset    │     │   Corpus    │     │  to Weaviate│
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Report    │◀────│   Compute   │◀────│  Evaluate   │
│  Results    │     │   Metrics   │     │  via Gradio │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Evaluation Details

The evaluation script:
1. Connects to the Gradio app via `gradio_client`
2. For each question:
   - Formats a multiple-choice prompt with options A/B/C/D
   - Uploads the query image and sends the question
   - Extracts the answer (A/B/C/D) from the VLM response
   - Compares with ground truth
3. Saves results with checkpointing for resumption
4. Computes accuracy metrics

## Metrics

Results include:
- **Overall accuracy**: Percentage of correct answers
- **Per-scenario accuracy**: Breakdown by 9 MRAG-Bench scenarios
- **Per-aspect accuracy**: Perspective, Transformative, Other
- **Extraction failures**: Questions where answer couldn't be parsed

## Published Baselines

| Model | Accuracy |
|-------|----------|
| GPT-4o | 74.5% |
| Human | 71.6% |
| GPT-4V | 65.3% |
| Gemini 1.5 Pro | 62.4% |
| Claude 3 Opus | 59.8% |
| LLaVA-1.5 | 41.2% |

## Command Reference

```bash
# Download with custom output
python -m benchmark.mrag_bench.download --output /path/to/output

# Evaluate with limit and specific output
python -m benchmark.mrag_bench.evaluate \
    --limit 100 \
    --output benchmark/results/run_001.json \
    --gradio-url http://localhost:7860

# Resume from checkpoint
python -m benchmark.mrag_bench.evaluate \
    --resume benchmark/results/run_001.json \
    --output benchmark/results/run_001.json

# View metrics from saved results
python -m benchmark.mrag_bench.metrics benchmark/results/run_001.json
```

## Scenarios

MRAG-Bench includes 9 scenarios:

1. **Artwork** - Art identification and analysis
2. **Document** - Document understanding
3. **Illustration** - Illustrated content
4. **Infographic** - Information graphics
5. **Map** - Geographic and spatial content
6. **Medical** - Medical imaging
7. **Photo** - Photographic content
8. **Poster** - Poster and advertisement content
9. **Table** - Tabular data

## Aspects

Questions are categorized by aspect:
- **Perspective** - Different viewpoints of the same subject
- **Transformative** - Transformed versions (crops, edits, etc.)
- **Other** - Other retrieval challenges
