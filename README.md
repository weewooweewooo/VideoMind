# VideoMind

VideoMind is a small CPU-oriented project for searching the spoken content of
local videos. It transcribes a local video with Faster-Whisper, groups the
transcript into timestamped chunks, and ranks those chunks with local TF-IDF
retrieval.

## Architecture

```text
local video
  -> Faster-Whisper transcription
  -> timestamped transcript chunks
  -> local TF-IDF retrieval
  -> ranked chunks with timestamps
```

The transcription command produces JSON. The retrieval command reads that JSON
directly. Retrieval is lexical: it ranks shared query and transcript terms, not
semantic similarity.

## Environment setup

Python 3.11 or newer is recommended.

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`faster-whisper` is the only direct runtime dependency. Its package installation
provides the required decoding and inference dependencies. TF-IDF retrieval uses
only the Python standard library.

The defaults are CPU-friendly:

```text
WHISPER_MODEL=base
WHISPER_COMPUTE_TYPE=int8
WHISPER_BEAM_SIZE=5
TRANSCRIPT_CHUNK_WORDS=70
DEVICE=cpu
```

The first use of a named Faster-Whisper model may download model files. Use a
local model path with `--model` when an offline run is required.

## Transcription

Transcribe one local video and write timestamped segments and chunks:

```powershell
python -m src.ingestion.transcriber data\video.mp4 --output transcript.json
```

Useful options include `--model`, `--compute-type`, `--beam-size`, `--language`,
and `--chunk-words`. Run the following for the complete command reference:

```powershell
python -m src.ingestion.transcriber --help
```

## Local retrieval

Search a transcript:

```powershell
python -m src.retrieval.local_retriever transcript.json "machine learning"
```

The command returns ranked chunks containing their original text, start and end
timestamps, chunk IDs, and cosine scores. Use `--top-k` and `--min-score` to
control result selection.

```powershell
python -m src.retrieval.local_retriever --help
```

## Current limitations

- Real Faster-Whisper inference depends on local hardware, media compatibility,
  and model availability.
- TF-IDF retrieval is lexical rather than semantic.
- The tokenizer is intentionally ASCII-focused.
- Each CLI invocation searches one transcript and builds an in-memory index.
- Transcription and retrieval are currently separate commands.

## Next planned milestone

Add a small `src/videomind.py` orchestration CLI that runs transcription and
retrieval as one local workflow.
