# VideoMind

VideoMind transcribes one video, builds a lightweight BM25 lexical index, and
retrieves transcript passages matching the user's question. It returns
timestamped evidence rather than a generated answer.

```text
one video
  -> transcript or transcript cache
  -> transcript chunks
  -> reusable BM25 index
  -> matching transcript evidence
```

## Install

Python 3.11 or newer is recommended.

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`requirements.txt` provides Faster-Whisper transcription and BM25 retrieval.
The first use of a named Faster-Whisper model may download model files; use a
local model path when an offline run is needed.

## Ask one question

```powershell
python -m src.videomind `
  .\data\WjNlodSXlmI.mp4 `
  "What has HCA Health Care automated?"
```

VideoMind transcribes the video when necessary, caches the normalized
transcript, builds one in-memory BM25 index, and prints matching transcript
evidence as JSON.

## Ask multiple questions

```powershell
python -m src.videomind `
  .\data\WjNlodSXlmI.mp4 `
  --interactive
```

The video is ingested once and the same BM25 index is reused for every
independent question. Use `:help` for interactive commands and `:quit` or
`:exit` to finish.

## How retrieval works

```text
validated transcript chunks
  -> tokenize and remove generic stopwords
  -> BM25 index
  -> meaningful-vocabulary-overlap check
  -> BM25 scores
  -> deterministic ranked evidence
```

BM25 is a keyword-based information-retrieval technique, not an embedding model
or LLM. It generally ranks lexical matches better than basic term-frequency
weighting, but it does not understand synonyms and strong paraphrases may fail.
Unrelated queries with no meaningful vocabulary overlap return no evidence.
Higher scores mean stronger lexical matches; they are raw BM25 ranking values,
not confidence, probability, or percentages, and are not directly comparable
with the former cosine scores.

## Transcript cache

The transcript cache is keyed by the video contents and transcription
configuration. It supports cache hits, misses, refreshes, disabled operation,
configuration changes, and corrupt-entry recovery.

```powershell
# Use a custom cache directory.
python -m src.videomind .\data\WjNlodSXlmI.mp4 "What is discussed?" `
  --cache-dir "D:\VideoMindCache"

# Retranscribe and replace the matching entry.
python -m src.videomind .\data\WjNlodSXlmI.mp4 "What is discussed?" `
  --refresh-cache

# Disable cache reads and writes.
python -m src.videomind .\data\WjNlodSXlmI.mp4 "What is discussed?" `
  --no-cache
```

`--cache-dir` takes precedence over `VIDEOMIND_CACHE_DIR`. By default, Windows
uses `%LOCALAPPDATA%\VideoMind\cache`; other platforms use
`~/.cache/videomind`.

## Diagnostic transcript paths

`--transcript-input` loads a saved transcript without running transcription:

```powershell
python -m src.videomind `
  --transcript-input .\transcript.json `
  "What is discussed?" `
  --pretty
```

`--save-transcript` saves the prepared transcript without changing the normal
single-video flow:

```powershell
python -m src.videomind `
  .\data\WjNlodSXlmI.mp4 `
  "What is discussed?" `
  --save-transcript .\transcript.json
```

## Source structure

```text
src/
|-- ingestion.py
|-- retrieval.py
`-- videomind.py
```

The application call graph is:

```text
main()
  -> ingest_video()
  -> build_retriever()
  -> search one question or enter the interactive loop
  -> print matching evidence
```

Transcription remains lazy: importing VideoMind does not import
Faster-Whisper.

## Current limitations

- BM25 does not understand synonyms.
- Strongly paraphrased questions may return weak or no evidence.
- BM25 scores are ranking values, not confidence.
- Natural-language answer generation is not implemented.
- Questions are independent; interactive mode has no conversation memory.
- Transcript and retrieval indexes are process-local. Only normalized
  transcript segments are cached between processes.
- Real transcription depends on local hardware, media compatibility, and model
  availability.
