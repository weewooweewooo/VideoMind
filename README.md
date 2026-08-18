# VideoMind

VideoMind transcribes one local media file and retrieves the strongest
transcript chunk as focused evidence for a question. It uses deterministic
lexical retrieval and returns transcript evidence rather than generating an
answer.

## Architecture

```text
local media file
-> Faster-Whisper transcription
-> stable transcript cache
-> timestamped transcript chunks
-> BM25S tokenization with English stopwords and stemming
-> BM25S chunk retrieval
-> exact transcript chunk evidence
```

The core flow is divided across these modules:

- `ingestion.py`: validates a local path, transcribes the media lazily, reads
  and writes the transcript cache, validates timestamps, and builds chunks.
- `retrieval.py`: tokenizes text with BM25S, builds the in-memory chunk index,
  and deterministically ranks transcript chunks.
- `videomind.py`: orchestrates the CLI, reuses one prepared session for
  interactive questions, and prints plain-text or JSON output.
- `config.py`: contains transcription, chunking, and retrieval defaults.

## Install

Python 3.11 or newer is recommended.

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` provides Faster-Whisper transcription and BM25S retrieval.
The first use of the configured Faster-Whisper `small` model may download model
files. Faster-Whisper reads supported media directly through PyAV; VideoMind
does not create an intermediate WAV file.

## Command-line usage

Pass a local media path followed by a question. Plain text is the default
output:

```powershell
py -3.13 -m src.videomind .\data\test1.mp4 "What did HCA automate?"
```

When no positive-scoring transcript evidence is found, plain-text mode prints:

```text
No relevant evidence found in the video.
```

Use `--json` for a single-line JSON object:

```powershell
py -3.13 -m src.videomind .\data\test1.mp4 "What did HCA automate?" --json
```

```json
{"query": "What did HCA automate?", "focused_evidence": "Exact transcript evidence."}
```

Use `--pretty` with `--json` for indented JSON:

```powershell
py -3.13 -m src.videomind .\data\test1.mp4 "What did HCA automate?" --json --pretty
```

`--pretty` affects JSON formatting only. JSON mode preserves the query and uses
`null` when no evidence is found.

## Interactive use

```powershell
py -3.13 -m src.videomind .\data\test1.mp4 --interactive
```

The media file is ingested once, and the prepared transcript, BM25S tokens,
and chunk index are reused for each independent question. Use `:help` for
interactive commands and `:quit` or `:exit` to finish. Interactive mode does
not add conversation memory.

## How retrieval works

Chunks and questions use BM25S tokenization with built-in English stopwords and
the supported English stemmer. The retriever discards non-positive scores,
then orders chunks by score and chunk ID for deterministic results. The
highest-ranked chunk text is returned exactly as it appears in the transcript.

Full chunk text, IDs, timestamps, and raw BM25 scores remain available through
the internal `search()` API but are not included in the focused CLI response.

## Transcript cache

VideoMind uses one inspectable JSON cache file per normalized, resolved local
media path. A cache entry is reused only when the file size, nanosecond
modification time, and fixed Faster-Whisper profile still match. Changing or
moving the media file therefore causes a fresh transcription. Cache entries
are written atomically.

On Windows, the cache is stored under
`%LOCALAPPDATA%\VideoMind\cache`. On other platforms, it is stored under
`~/.cache/videomind`.

## Source structure

```text
src/
|-- config.py
|-- ingestion.py
|-- retrieval.py
`-- videomind.py
```

Faster-Whisper is imported only when a cache miss requires transcription, so
importing VideoMind or using a warm cache does not load the model.

## Current capabilities

- Local media input.
- Direct Faster-Whisper transcription.
- Persistent transcript caching.
- Timestamped transcript chunking.
- BM25S tokenization with English stopwords and stemming.
- Deterministic BM25S chunk retrieval.
- Plain-text, JSON, and interactive CLI modes.

## Current limitations

VideoMind does not provide semantic retrieval, answer generation, speaker
diarization, main-topic analysis, visual understanding, multi-video indexing,
or a persistent retrieval index.

BM25 remains keyword-based, so it does not understand synonyms or semantic
relationships. Strongly paraphrased or unrelated questions may return weak or
coincidental lexical evidence. BM25 scores are ranking values, not confidence
values. Only one transcript chunk is returned, so useful surrounding context
may be omitted. Transcription also depends on local hardware, media
compatibility, and model availability.
