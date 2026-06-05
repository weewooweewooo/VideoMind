# VideoMind

> Fully local multimodal video RAG for lecture videos: archive.org discovery, in-memory ingestion, CLIP retrieval, Redis Stack vector search, and Ollama answer generation with conversation memory.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-orange)
![Redis Stack](https://img.shields.io/badge/Vector%20Store-Redis%20Stack-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## Overview

VideoMind is a local research system for asking natural-language questions about lecture videos. It discovers lecture content, processes video streams without storing the source videos on disk, aligns visual frames with transcript context, stores CLIP embeddings in Redis Stack, and answers with timestamp citations through a local Ollama model.

The project is built for lecture-domain retrieval. Vanilla CLIP performs poorly on slides, whiteboards, derivations, and code walkthroughs because those visuals are underrepresented in generic image-text training data. VideoMind addresses that gap by building a lecture-specific dataset and fine-tuning CLIP ViT-B/32 with an InfoNCE contrastive loss. Fine-tuning is implemented and in progress; reported retrieval numbers below are the pre-fine-tuning baseline.

VideoMind is fully local for AI inference: no OpenAI, Claude, or hosted LLM APIs. Discovery uses public archive.org metadata, while transcription, embedding, vector search, RAG, and answer generation run on local infrastructure.

## Current Results

Baseline retrieval quality before fine-tuning:

| Metric | Vanilla CLIP | Fine-tuned CLIP |
|---|---:|---:|
| Mean cosine similarity | 0.86+ | After adding transcript text embeddings |
| Recall@1 | 0% | Pending training |

Fine-tuned CLIP results will be added after the current training run and evaluation pass complete.

## What Works Today

VideoMind currently supports intelligent content discovery via LLaMA sector categorization, archive.org video search, in-memory frame extraction with decord, in-memory transcription with faster-whisper, parallel frame/transcript processing, Redis Stack HNSW vector search, FastAPI ingestion/query endpoints, and per-session conversation memory.

The ingestion path avoids source video storage. Frames are represented as PIL images in memory, transcripts are held as dictionaries in memory, embeddings are generated directly from those in-memory objects, and Redis stores vector records with video name, transcript context, and start/end timestamps.

## Architecture

```text
User discovers a topic
  -> LLaMA categorizes archive.org results into sectors
  -> User selects sector and videos
  -> Stream video
  -> decord extracts scene-change frames in memory
  -> content-aware filtering removes blank and presenter-only frames
  -> slide regions cropped from mixed frames
  -> transcript segments chunked and embedded as text for high-quality retrieval
  -> faster-whisper transcribes audio in memory
  -> frame extraction and transcription run in parallel
  -> CLIP embeds frames with open-clip-torch ViT-B/32
  -> frames align to transcript segments by timestamp
  -> Redis Stack stores HNSW vectors and metadata
  -> User asks a question
  -> CLIP embeds the query
  -> Redis vector search retrieves relevant moments
  -> Ollama LLaMA 3.2 3B generates an answer with timestamps
  -> conversation memory is maintained per session
```

## Tech Stack

| Layer | Current implementation |
|---|---|
| Content discovery | archive.org search plus LLaMA sector categorization |
| Frame extraction | decord, in-memory, no frame files written |
| Speech-to-text | faster-whisper medium (int8), local model path preferred |
| Embeddings | open-clip-torch ViT-B/32 |
| Fine-tuning | PyTorch with symmetric InfoNCE contrastive loss |
| Vector store | Redis Stack HNSW vector search via RedisVL |
| Answer generation | Ollama LLaMA 3.1 8B via langchain-ollama |
| API | FastAPI |
| Session memory | Per-session VideoMindPipeline instances with 1-hour expiry |
| NPU optimization | OpenVINO planned for benchmarking/export |

## Repository Layout

```text
videomind/
  src/
    ingestion/
      downloader.py        CLI entry point
      archive_search.py    archive.org search and metadata filtering
      archive_utils.py     shared archive.org metadata/direct URL helpers
      sector_analyzer.py   LLaMA-based sector categorization
      stream_processor.py  discovery and streaming ingestion orchestration
      extractor.py         decord in-memory frame extraction
      transcriber.py       faster-whisper and archive.org transcript loading
    dataset/
      builder.py           transcript chunk dataset builder
      loader.py            OpenCLIP-ready PyTorch dataset
    training/
      train.py             CLIP fine-tuning loop
      loss.py              InfoNCE contrastive loss
      evaluate.py          Recall@K evaluation
    retrieval/
      embedder.py          OpenCLIP embedding wrapper
      store.py             Redis Stack vector search wrapper
      pipeline.py          RAG pipeline with conversation memory
    utils/
      cleanup.py           pair/Redis cleanup helpers
    main.py                FastAPI app
  scripts/
    cleanup.py             cleanup CLI
  docker-compose.yml       Redis Stack, Ollama, API stack
  Dockerfile               FastAPI API image
  requirements.txt         CPU dependencies
  requirements-gpu.txt     GPU dependencies
```

## Setup

Prerequisites:

- Docker and Docker Compose
- Ollama
- Python 3.11
- Anaconda or Miniconda

Clone the repository:

```bash
git clone https://github.com/yourusername/videomind.git
cd videomind
```

Create the local Python environment:

```bash
conda create -n videomind python=3.11 xz -y
conda activate videomind
```

Install dependencies. Use the CPU file for the work laptop and Docker-compatible environments:

```bash
pip install -r requirements.txt
```

Use the GPU file on the RTX 3060 machine:

```bash
pip install -r requirements-gpu.txt
```

Start local infrastructure:

```bash
docker-compose up -d
```

This starts Redis Stack, Redis Insight, Ollama, and the API container. If you want to run the API directly from the conda environment during development, start only the local services:

```bash
docker-compose up -d redis ollama
```

Pull the local LLM:

```bash
ollama pull llama3.1:8b
```

Download or copy the faster-whisper base model into:

```text
models/faster-whisper-base/
```

The local app expects:

```env
REDIS_URL=redis://localhost:6379
OLLAMA_HOST=http://localhost:11434
WHISPER_MODEL=medium
KMP_DUPLICATE_LIB_OK=TRUE
DEVICE=cpu
```

## Usage

Discover and ingest lecture videos from archive.org:

```bash
python -m src.ingestion.downloader --discover "machine learning"
```

Start the FastAPI app locally:

```bash
uvicorn src.main:app --reload
```

Query indexed videos:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is backpropagation?"}'
```

The response includes a `session_id`. Pass it back for follow-up questions so VideoMind can use conversation history:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you explain that more simply?", "session_id": "returned-session-id"}'
```

Session helpers:

```bash
GET    /sessions/{session_id}/history
DELETE /sessions/{session_id}
```

Cleanup local pairs and Redis vectors:

```bash
python scripts/cleanup.py --all
python scripts/cleanup.py --all --targets pairs redis
```

## Training

The fine-tuning path is implemented for OpenCLIP ViT-B/32. The dataset builder creates train/validation/test splits from transcript chunks, and training optimizes image/text alignment with a symmetric InfoNCE loss.

Run training on the GPU laptop:

```bash
python -m src.training.train \
  --dataset data/pairs \
  --epochs 20 \
  --batch-size 32 \
  --device cuda \
  --output checkpoints/clip-lecture
```

Evaluate retrieval after training:

```bash
python -m src.training.evaluate \
  --model checkpoints/clip-lecture \
  --test-split data/pairs/test
```

## Hardware

| Machine | Specs | Role |
|---|---|---|
| Work laptop | Intel Core Ultra 7 165U, Intel AI Boost NPU | Development, ingestion, API testing, future NPU benchmarking |
| Personal laptop | NVIDIA RTX 3060 4GB | CLIP fine-tuning and GPU inference experiments |

## Roadmap

Completed:

- [x] Environment setup
- [x] Intelligent content discovery with LLaMA
- [x] Streaming in-memory pipeline with decord and faster-whisper
- [x] Scene-change frame extraction
- [x] Parallel frame and transcript processing
- [x] Dataset builder with train/validation/test split
- [x] CLIP fine-tuning pipeline with InfoNCE loss
- [x] Redis Stack vector store
- [x] Conversation memory and session management
- [x] FastAPI backend
- [x] Docker Compose stack

Pending:

- [ ] CLIP fine-tuning results
- [ ] OpenVINO NPU benchmarking
- [ ] Confidence thresholding
- [ ] arXiv paper

## Author

Sean, BSc Computer Science (Artificial Intelligence), Multimedia University
