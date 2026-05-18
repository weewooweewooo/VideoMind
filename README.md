# VideoMind

> A fully local Multimodal Video RAG system for lecture and educational videos — fine-tuned CLIP meets local LLM, zero external APIs.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## What is VideoMind?

VideoMind lets you upload any lecture or educational video and ask natural language questions about it. Instead of scrubbing through hours of content, you get precise, timestamped answers citing the exact moment the concept was explained.

**Fully local. No OpenAI. No Claude API. No cost per query.**

---

## Motivation

Vanilla CLIP was trained on general image-text pairs scraped from the internet — photographs, memes, product images. It has never seen a lecture slide, a whiteboard derivation, or a code walkthrough. When you ask it to retrieve the frame where a professor explains backpropagation, it struggles because the visual domain is completely foreign to it.

VideoMind addresses this by fine-tuning CLIP on a custom dataset of lecture frame-transcript pairs, teaching the model to align visual slide content with spoken explanations. The result is a retrieval model that actually understands educational video.

---

## ML Contributions

This is not a wrapper project. The core ML work includes:

- **Custom dataset pipeline** — scraped lecture videos via `yt-dlp`, extracted frames with `FFmpeg`, transcribed audio with `Whisper medium`, and aligned frames with transcript chunks by timestamp to produce `(frame, text)` contrastive pairs
- **CLIP fine-tuning** — fine-tuned `CLIP ViT-B/32` using InfoNCE contrastive loss on the lecture domain dataset, trained on a local RTX 3060
- **Retrieval evaluation** — measured Recall@1, Recall@5, Recall@10 before and after fine-tuning on a held-out lecture test set
- **NPU inference benchmarking** — optimised the fine-tuned CLIP model for Intel NPU (AI Boost) via OpenVINO INT8 quantisation and benchmarked latency against CPU baseline

---

## Results

| Metric | Vanilla CLIP | Fine-tuned CLIP | Improvement |
|---|---|---|---|
| Recall@1 | — | — | — |
| Recall@5 | — | — | — |
| Recall@10 | — | — | — |

| Backend | Avg inference latency | Speedup |
|---|---|---|
| CPU (Intel Core Ultra 7 165U) | — | baseline |
| NPU (Intel AI Boost, INT8) | — | —x |

> Results will be updated upon training completion.

---

## System Architecture

```
Input Video
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  Ingestion Pipeline                  │
│                                                     │
│  yt-dlp ──► FFmpeg ──► Frames    ──► CLIP (FT) ──► │
│                    └──► Whisper ──► Transcript ──►  │
│                                  Timestamp align    │
└─────────────────────────┬───────────────────────────┘
                          │ (frame, text, timestamp) pairs
                          ▼
               ┌──────────────────┐
               │    ChromaDB      │
               │  Vector Store    │
               └────────┬─────────┘
                        │
              Query ───►│
                        ▼
               ┌──────────────────┐
               │  LangChain RAG   │
               │  Retrieval Layer │
               └────────┬─────────┘
                        │ top-k chunks + timestamps
                        ▼
               ┌──────────────────┐
               │ Ollama LLaMA 3.2 │
               │  (fully local)   │
               └────────┬─────────┘
                        │
                        ▼
          Answer + Timestamp Citations
```

---

## Tech Stack

| Component | Tool | Runs |
|---|---|---|
| Speech-to-text | Whisper medium | Local |
| Frame-text embeddings | CLIP ViT-B/32 (fine-tuned) | Local |
| Vector store | ChromaDB | Local |
| Retrieval pipeline | LangChain | Local |
| LLM answer generation | Ollama + LLaMA 3.2 3B | Local |
| Backend API | FastAPI | Local |
| NPU optimisation | OpenVINO | Local |
| Training framework | PyTorch | Local |
| Data collection | yt-dlp + FFmpeg | Local |

---

## Project Structure

```
videomind/
├── data/
│   ├── videos/          # raw downloaded lecture videos
│   ├── frames/          # extracted frames per video
│   ├── transcripts/     # Whisper output JSON
│   ├── pairs/           # (frame, text) contrastive dataset
│   └── chroma/          # ChromaDB persistent store
├── src/
│   ├── ingestion/
│   │   ├── downloader.py      # yt-dlp wrapper
│   │   ├── extractor.py       # FFmpeg frame extraction
│   │   └── transcriber.py     # Whisper transcription
│   ├── dataset/
│   │   ├── builder.py         # timestamp alignment, pair creation
│   │   └── loader.py          # PyTorch Dataset class
│   ├── training/
│   │   ├── train.py           # CLIP fine-tuning loop
│   │   ├── loss.py            # InfoNCE contrastive loss
│   │   └── evaluate.py        # Recall@K evaluation
│   ├── retrieval/
│   │   ├── embedder.py        # CLIP inference wrapper
│   │   ├── store.py           # ChromaDB operations
│   │   └── pipeline.py        # LangChain RAG chain
│   ├── inference/
│   │   └── openvino_export.py # OpenVINO INT8 export
│   └── main.py                # FastAPI app
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_retrieval_eval.ipynb
├── tests/
├── .env
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Anaconda or Miniconda
- FFmpeg
- Ollama

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/videomind.git
cd videomind
```

### 2. Create conda environment

```bash
conda create -n videomind python=3.11 -y
conda activate videomind
conda install -c conda-forge ffmpeg -y
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull LLaMA 3.2

```bash
# Download Ollama from https://ollama.com/download
ollama pull llama3.2:3b
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env with your paths
```

---

## Usage

### Ingest a video

```bash
python -m src.ingestion.downloader --url "https://youtube.com/watch?v=..." --output data/videos
python -m src.ingestion.extractor --video data/videos/lecture.mp4
python -m src.ingestion.transcriber --video data/videos/lecture.mp4
```

### Build dataset and index

```bash
python -m src.dataset.builder --videos-dir data/videos
python -m src.retrieval.store --index
```

### Query a video

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "When does the professor explain backpropagation?"}'
```

### Fine-tune CLIP

```bash
# Run on GPU machine
python -m src.training.train \
  --dataset data/pairs \
  --epochs 20 \
  --batch-size 32 \
  --output checkpoints/clip-lecture
```

### Evaluate retrieval

```bash
python -m src.training.evaluate \
  --model checkpoints/clip-lecture \
  --test-split data/pairs/test
```

---

## Roadmap

- [x] Project structure and environment setup
- [ ] Data ingestion pipeline (yt-dlp + FFmpeg + Whisper)
- [ ] Contrastive dataset builder
- [ ] CLIP fine-tuning training loop
- [ ] Recall@K evaluation harness
- [ ] ChromaDB vector store integration
- [ ] LangChain RAG pipeline
- [ ] Ollama LLaMA 3.2 integration
- [ ] FastAPI backend
- [ ] OpenVINO NPU export and benchmarking
- [ ] Docker deployment
- [ ] arXiv paper

---

## Hardware

Developed and tested on:

- **Development + NPU benchmarking** — Intel Core Ultra 7 165U, Intel AI Boost NPU, 16GB RAM
- **Model training** — NVIDIA RTX 3060 4GB VRAM

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Sean Lee** — Full-Stack AI Developer  
BSc Computer Science (Artificial Intelligence), Multimedia University