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
```

`requirements.txt` provides Faster-Whisper transcription and BM25 retrieval.
The first use of the fixed Faster-Whisper `base` model may download model files.

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
  -> shared lowercase tokenization
  -> conservative English morphological normalization
  -> corpus-aware compound splitting
  -> remove generic stopwords
  -> BM25 index
  -> meaningful-vocabulary-overlap check
  -> BM25 scores
  -> deterministic ranked evidence
```

Chunks and questions use the same normalization pipeline. BM25 remains a
keyword-based information-retrieval technique, not an embedding model or LLM.
It does not provide semantic understanding: speaker intent, main-topic
inference, and general paraphrases may still fail. Unrelated queries with no
meaningful vocabulary overlap return no evidence. Higher scores mean stronger
lexical matches; they are raw BM25 ranking values, not confidence, probability,
or percentages, and are not directly comparable with the former cosine scores.

## Transcript cache

The inspectable JSON transcript cache uses one stable file per resolved video
path. The file stores exactly five top-level fields: `source`, `profile`,
`language`, `duration`, and `segments`. The source field contains the video size
and modification timestamp, while the profile records the fixed transcription
configuration. When the source or transcription profile changes, VideoMind
retranscribes and atomically replaces the same cache file. The source fingerprint
is intended for local cache invalidation, not cryptographic content-integrity
verification. Windows uses `%LOCALAPPDATA%\VideoMind\cache`; other platforms use
`~/.cache/videomind`.

## Source structure

```text
src/
|-- config.py
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
