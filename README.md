# VideoMind

VideoMind transcribes one local video and retrieves the strongest exact
transcript sentence as focused evidence for a question. It uses deterministic
lexical retrieval and intentionally returns transcript evidence rather than
generating an answer.

## Architecture

```text
Video
→ Faster-Whisper transcription
→ stable transcript cache
→ timestamped transcript chunks
→ custom lexical normalization
→ BM25 chunk retrieval
→ BM25 sentence scoring
→ exact transcript evidence
```

The current implementation is divided into five modules:

- `ingestion.py`: validates the video, transcribes it with Faster-Whisper,
  reads and writes the persistent transcript cache, validates transcript
  segments, and builds timestamped transcript chunks.
- `normalization.py`: tokenizes text, filters stopwords, applies conservative
  light stemming, discovers and applies corpus-aware compound splits, and
  splits transcript text into sentences.
- `retrieval.py`: builds the BM25 chunk index, ranks chunks, scores eligible
  sentences with BM25, and deterministically selects focused evidence.
- `videomind.py`: orchestrates the CLI, reuses a prepared session for
  interactive questions, and prints plain-text or JSON output.
- `config.py`: contains transcription settings, chunking settings, lexical
  constants, and default retrieval settings.

## Install

Python 3.11 or newer is recommended.

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` provides Faster-Whisper transcription and BM25 retrieval.
The first use of the configured Faster-Whisper `base` model may download model
files.

## Command-line usage

The CLI accepts the video path followed by the question. Plain text is the
default output:

```powershell
py -3.13 -m src.videomind data\WjNlodSXlmI.mp4 "What did HCA automate?"
```

```text
Using MEDALAM, we have automated things like documentation, summarizing insights from medical records.
```

When no relevant transcript evidence is found, plain-text mode prints:

```text
No relevant evidence found in the video.
```

Use `--json` for a single-line JSON object:

```powershell
py -3.13 -m src.videomind data\WjNlodSXlmI.mp4 "What did HCA automate?" --json
```

```json
{"query": "What did HCA automate?", "focused_evidence": "Using MEDALAM, we have automated things like documentation, summarizing insights from medical records."}
```

Use `--pretty` with `--json` for indented JSON:

```powershell
py -3.13 -m src.videomind data\WjNlodSXlmI.mp4 "What did HCA automate?" --json --pretty
```

```json
{
  "query": "What did HCA automate?",
  "focused_evidence": "Using MEDALAM, we have automated things like documentation, summarizing insights from medical records."
}
```

`--pretty` affects JSON formatting only. In JSON mode, an unsupported question
keeps the original query and returns `null` for `focused_evidence`.

## Interactive use

```powershell
py -3.13 -m src.videomind data\WjNlodSXlmI.mp4 --interactive
```

The video is ingested once, and the prepared transcript, compound mappings,
BM25 chunk index, and sentence candidates are reused for each independent
question. Use `:help` for interactive commands and `:quit` or `:exit` to
finish. Interactive mode does not add conversation memory.

## How retrieval works

Chunks and questions pass through the same lexical normalization. The
retriever first checks for meaningful vocabulary overlap, ranks positive-score
chunks with BM25, then scores sentences from the retrieved chunks with BM25.
Focused-evidence selection is deterministic: score, parent chunk rank,
sentence position, and chunk ID provide stable ordering. The returned text is
copied exactly from the transcript.

Full chunk text, IDs, timestamps, and raw BM25 scores remain available through
the internal `search()` API but are not included in the focused CLI response.

## Transcript cache

VideoMind uses one stable, inspectable JSON cache file per resolved video path.
Each record contains `source`, `profile`, `language`, `duration`, and
`segments`. Source size and modification time and the fixed transcription
profile determine whether the cached transcript remains valid. If either
changes, VideoMind retranscribes the video and atomically replaces the cache
entry.

On Windows, the cache is stored under
`%LOCALAPPDATA%\VideoMind\cache`. On other platforms, it is stored under
`~/.cache/videomind`.

## Source structure

```text
src/
├── config.py
├── ingestion.py
├── normalization.py
├── retrieval.py
└── videomind.py
```

Transcription is loaded lazily, so importing VideoMind does not import
Faster-Whisper.

## Current capabilities

- Local video transcription.
- Persistent transcript caching.
- Timestamped transcript chunking.
- Custom lexical normalization.
- Conservative morphological matching.
- Corpus-aware compound matching.
- BM25 chunk retrieval.
- Sentence-level focused evidence.
- Deterministic repeated results.
- Plain-text and JSON CLI modes.
- Safe `null` JSON output or a plain-text fallback when no evidence is found.

## Current limitations

VideoMind currently does not provide:

- Semantic embedding retrieval.
- CrossEncoder reranking.
- Extractive QA.
- Answer generation or abstractive summarization.
- External knowledge.
- Speaker diarization.
- Main-topic analysis.
- Persistent vocabulary learning; compound mappings are corpus-aware and
  process-local.
- Multi-video indexing.

BM25 remains keyword-based, so it does not understand synonyms and strongly
paraphrased questions may return weak or no evidence. BM25 scores are ranking
values, not confidence values. Only one transcript sentence is returned, so
useful surrounding context may be omitted. Real transcription also depends on
local hardware, media compatibility, and model availability.
