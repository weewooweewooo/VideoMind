# VideoMind

VideoMind is a small CPU-oriented project for searching the spoken content of
local videos. It transcribes a local video with Faster-Whisper, groups the
transcript into timestamped chunks, and ranks those chunks with a selected local
retrieval backend.

## Architecture

```text
local video
  -> transcript cache lookup or Faster-Whisper transcription
  -> timestamped transcript chunks
  -> selected local retrieval backend
     |-> TF-IDF (default)
     |-> semantic embeddings (optional)
     `-> hybrid TF-IDF + semantic RRF (optional)
  -> ranked chunks with timestamps
```

The transcription command produces JSON. The retrieval command reads that JSON
directly. TF-IDF remains the default dependency-free backend. Optional semantic
retrieval ranks conceptual similarity using a small local embedding model.

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
VIDEOMIND_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

The first use of a named Faster-Whisper model may download model files. Use a
local model path with `--model` when an offline run is required.

## Unified video query

Transcribe and search one local video without writing an intermediate
transcript:

```powershell
py -3.13 -m src.videomind `
  data\example.mp4 `
  "your question"
```

To search an existing transcript without invoking Faster-Whisper:

```powershell
py -3.13 -m src.videomind `
  --transcript-input path\to\transcript.json `
  "your question" `
  --pretty
```

Use `--save-transcript path\to\transcript.json` in video mode when the complete
transcription should be retained. Existing files are not overwritten.

The transcription and retrieval commands below remain available as lower-level
tools.

## Persistent transcript cache

Unified video mode caches timestamped transcript segments by default, so a new
process can avoid retranscribing an unchanged video with the same transcription
configuration:

```powershell
python -m src.videomind `
  data\video.mp4 `
  "care transformation"
```

Use a custom cache directory when required:

```powershell
python -m src.videomind `
  data\video.mp4 `
  "care transformation" `
  --cache-dir "D:\VideoMindCache"
```

Disable both cache reads and writes:

```powershell
python -m src.videomind `
  data\video.mp4 `
  "care transformation" `
  --no-cache
```

When combined with `--no-cache`, `--cache-dir` is accepted but ignored.

Ignore and atomically replace an existing matching entry:

```powershell
python -m src.videomind `
  data\video.mp4 `
  "care transformation" `
  --refresh-cache
```

`--cache-dir` takes precedence over `VIDEOMIND_CACHE_DIR`. When neither is set,
Windows uses `%LOCALAPPDATA%\VideoMind\cache`; other platforms use
`~/.cache/videomind`.

Cache identity includes the video content hash, cache schema version, Whisper
model identifier or resolved local path, automatic or explicit language, beam
size, device, and compute type. Chunk size, chunk overlap, retrieval backend,
embedding model, score thresholds, and questions are not part of that identity,
so those settings can change without retranscription.

Cache files contain transcript text, timestamps, duration, language, diagnostic
source path, and transcription configuration. They do not contain the video,
semantic embeddings, TF-IDF vectors, retrieval results, questions, or
interactive history. Transcript-input mode does not use the automatic cache.
Removing the cache directory is safe because VideoMind can regenerate it.
Future cache schema versions are not guaranteed to remain compatible.

## Retrieval backends

Default TF-IDF retrieval needs no optional package:

```powershell
python -m src.videomind `
  --transcript-input transcript.json `
  "care transformation"
```

Install the single optional semantic dependency after the base requirements:

```powershell
python -m pip install -r requirements-semantic.txt
```

Run semantic retrieval explicitly:

```powershell
python -m src.videomind `
  --transcript-input transcript.json `
  "Who is improving healthcare delivery?" `
  --retriever semantic `
  --pretty
```

Hybrid retrieval uses both indexes and reciprocal-rank fusion:

```powershell
python -m src.videomind `
  --transcript-input transcript.json `
  "How is generated medical content protected for organizations?" `
  --retriever hybrid `
  --semantic-min-score 0.55 `
  --pretty
```

| Backend | Best use | Limitation |
| --- | --- | --- |
| TF-IDF | Exact phrases and known terminology | Weak on paraphrases |
| Semantic | Paraphrases and conceptual questions | Model-dependent similarity can admit unrelated content |
| Hybrid | Mixed natural-language search with lexical fallback | Extra model cost and lexical terms can alter semantic ordering |

Semantic retrieval uses `BAAI/bge-small-en-v1.5` on CPU by default. Override it
with a supported FastEmbed model name through `--embedding-model` or
`VIDEOMIND_EMBEDDING_MODEL`. An existing local directory is treated as a local
copy of the default BGE-small model. Semantic scores are cosine similarities
specific to the selected model; they are not calibrated against TF-IDF scores
and do not imply factual understanding. Semantic mode defaults to
`--min-score 0.4` to suppress weak matches; tune that threshold for the selected
model and content. `--semantic-min-score` is the clearer threshold option for
semantic and hybrid modes; the existing `--min-score` remains supported when the
semantic-specific option is omitted.

Hybrid mode uses reciprocal-rank fusion with `k=60`; it does not compare raw
TF-IDF and semantic score scales. A chunk is admitted when TF-IDF finds a
positive lexical match or its semantic score reaches the configured threshold.
In the current five-minute evaluation, a `0.55` threshold rejected three
unrelated queries. For the enterprise-security paraphrase above, hybrid at 100
words ranked the expected section first while semantic retrieval ranked it
third. This is one-video evidence, not a claim that hybrid or `0.55` is
universally optimal. TF-IDF, 70-word chunks, zero overlap, and semantic threshold
`0.4` remain the defaults.

To rebuild chunks from an existing transcript without retranscribing:

```powershell
python -m src.videomind `
  --transcript-input transcript.json `
  "your question" `
  --chunk-words 100 `
  --chunk-overlap-words 20 `
  --retriever hybrid `
  --semantic-min-score 0.55
```

Positive overlap reuses complete trailing transcript segments. The actual word
overlap can therefore exceed the requested approximation; it never cuts a
segment or invents partial timestamps.

## Interactive query sessions

Interactive mode loads one transcript, optionally rebuilds its chunks, and
constructs the selected in-memory index once. Each question then reuses that
same index.

Start a dependency-free TF-IDF session:

```powershell
python -m src.videomind `
  --transcript-input transcript.json `
  --retriever tfidf `
  --interactive
```

Start a hybrid session with 100-word chunks:

```powershell
python -m src.videomind `
  --transcript-input transcript.json `
  --retriever hybrid `
  --semantic-min-score 0.55 `
  --chunk-words 100 `
  --interactive
```

Use `:help` to list the interactive commands, and use `:quit` or `:exit` to
finish successfully. An initial positional question may be supplied before
`--interactive`; VideoMind processes it before reading more questions.

Questions are independent. Interactive mode does not retain conversation
memory, is not a chatbot, and returns timestamped transcript evidence rather
than generated answers. The transcript, embedding model, chunk embeddings, and
retrieval index remain process-local and are discarded when the session exits.
One session searches one transcript.

## Small local video libraries

Search the supported top-level media files in one directory:

```powershell
python -m src.videomind `
  --library .\videos `
  "patient safety" `
  --retriever tfidf
```

Video libraries discover `.mp4`, `.m4a`, `.mov`, `.mkv`, and `.webm` files
non-recursively and in deterministic filename order. Each distinct video uses
the persistent transcript cache. Duplicate video contents are hashed, reported
on stderr, and indexed once under the first deterministic filename.

Search a directory containing transcript JSON files without Faster-Whisper:

```powershell
python -m src.videomind `
  --transcript-library .\transcripts `
  "healthcare delivery" `
  --retriever hybrid `
  --semantic-min-score 0.55
```

Every top-level `.json` file is treated as a transcript and validated. Duplicate
normalized transcript content is indexed once. Library initialization is
fail-fast: an invalid or unreadable selected file aborts the operation and its
path is reported rather than silently omitting it.

Start an interactive video library:

```powershell
python -m src.videomind `
  --library .\videos `
  --retriever hybrid `
  --semantic-min-score 0.55 `
  --chunk-words 100 `
  --interactive
```

VideoMind validates and chunks each transcript separately, assigns globally
unique library chunk IDs, and builds one combined TF-IDF, semantic, or hybrid
index. IDF, semantic embeddings, and hybrid candidate ranking therefore span
the complete library. Results retain the source video, original per-video chunk
ID, text, and timestamps. Interactive questions reuse that one process-local
index and remain independent.

Library mode is intended for small local collections. It needs no database,
vector service, cloud API, or persistent embedding index.

## Transcription

Transcribe one local video and write timestamped segments and chunks:

```powershell
python -m src.ingestion.transcriber data\video.mp4 --output transcript.json
```

Useful options include `--model`, `--compute-type`, `--beam-size`, `--language`,
`--chunk-words`, and `--chunk-overlap-words`. Run the following for the complete
command reference:

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
- Semantic retrieval is similarity search rather than reasoning.
- Hybrid retrieval is rank fusion rather than reasoning or reranking.
- The optional embedding model requires a local download, memory, and CPU time.
- The tokenizer is intentionally ASCII-focused.
- Each CLI invocation searches one transcript; interactive mode reuses its
  in-memory index across independent questions.
- Transcript chunks and embeddings are process-local and are not persisted;
  only timestamped transcription segments can be cached between processes.
- A combined library index is intended only for a small local collection and
  is rebuilt in each process.
- VideoMind returns ranked transcript evidence, not a generated answer.
