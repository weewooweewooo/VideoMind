# VideoMind

VideoMind transcribes one video, retrieves relevant transcript chunks with
BM25, and returns the strongest exact transcript sentence as focused evidence.

```text
video
  -> cached Faster-Whisper transcript
  -> normalized BM25 chunk retrieval
  -> deterministic sentence-level reranking
  -> exact focused transcript evidence
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
transcript, builds one in-memory BM25 chunk index, and prints focused evidence:

```json
{
  "query": "What has HCA Health Care automated?",
  "focused_evidence": "Using MEDALAM, we have automated things like documentation, summarizing insights from medical records."
}
```

The public output contains only `query` and `focused_evidence`. Unsupported
questions return `null` focused evidence.

## Ask multiple questions

```powershell
python -m src.videomind `
  .\data\WjNlodSXlmI.mp4 `
  --interactive
```

The video is ingested once and the same prepared transcript, BM25 chunk index,
compound mappings, and sentence candidates are reused for every independent
question. Use `:help` for interactive commands and `:quit` or `:exit` to
finish.

## How retrieval works

```text
validated transcript chunks
  -> shared lowercase tokenization
  -> conservative English morphological normalization
  -> corpus-aware compound splitting
  -> remove generic stopwords
  -> BM25 index
  -> meaningful-vocabulary-overlap check
  -> deterministic BM25 chunk ranking
  -> split sentences in the retrieved chunks
  -> sentence-level BM25 reranking with the same normalization
  -> strongest exact transcript sentence
```

Chunks and questions use the same normalization pipeline. BM25 remains a
keyword-based information-retrieval technique, not an embedding model or LLM.
The focused evidence is extractive: it is copied from the transcript without
rewriting, generation, or abstractive summarization. No generative model,
embedding model, or model training is used. Full chunk text, IDs, timestamps,
and raw BM25 scores remain available through the internal `search()` API but
are not exposed in the normal focused response.

VideoMind does not provide semantic understanding. Unrelated queries with no
meaningful vocabulary overlap return no evidence, and speaker identification
and main-topic extraction remain unsupported.

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
  -> retrieve full BM25 chunks
  -> rerank their exact transcript sentences
  -> print focused evidence
```

Transcription remains lazy: importing VideoMind does not import
Faster-Whisper.

## Current limitations

- BM25 does not understand synonyms.
- Strongly paraphrased questions may return weak or no evidence.
- BM25 scores are ranking values, not confidence.
- Only one exact transcript sentence is returned, so useful surrounding context
  may be omitted.
- Speaker identification and main-topic extraction are not implemented.
- Natural-language answer generation and abstractive summarization are not
  implemented.
- Questions are independent; interactive mode has no conversation memory.
- Transcript and retrieval indexes are process-local. Only normalized
  transcript segments are cached between processes.
- Real transcription depends on local hardware, media compatibility, and model
  availability.
