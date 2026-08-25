# VideoMind

VideoMind transcribes one local media file into validated timestamped segments.
It stops at the clean Faster-Whisper segment output and does not build a
downstream text representation.

## Architecture

```text
local media file
-> Faster-Whisper transcription
-> validated and normalized timestamped segments
```

The transcript cache avoids repeating expensive transcription. Cached entries
contain the source identity, fixed Faster-Whisper profile, language, duration,
and normalized segments. They do not contain downstream derived objects.

Sentence reconstruction is intentionally the next development stage and is not
implemented yet.

## Install

Python 3.11 or newer is recommended.

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first use of the configured Faster-Whisper `small` model may download model
files. Faster-Whisper reads supported media directly through PyAV; VideoMind
does not create an intermediate WAV file.

## Command-line usage

Pass one local media path:

```powershell
py -3.13 -m src.videomind .\data\test1.mp4
```

The command prints a JSON array of clean timestamped segments:

```json
[
  {
    "start": 0.0,
    "end": 4.2,
    "text": "Normalized transcript text."
  }
]
```

Each segment has finite, non-negative, forward-moving timestamps and non-empty
whitespace-normalized text. Segments remain in deterministic time order.

## Transcript cache

VideoMind uses one inspectable JSON cache file per normalized, resolved local
media path. A cache entry is reused only when the file size, nanosecond
modification time, and fixed Faster-Whisper profile still match. Changing or
moving the media file therefore causes a fresh transcription. Cache entries are
written atomically.

On Windows, the cache is stored under
`%LOCALAPPDATA%\VideoMind\cache`. On other platforms, it is stored under
`~/.cache/videomind`.

Faster-Whisper is imported only when a cache miss requires transcription, so
importing VideoMind or using a warm cache does not load the model.

## Source structure

```text
src/
|-- config.py
|-- ingestion.py
`-- videomind.py
```

- `config.py` contains the fixed Faster-Whisper profile.
- `ingestion.py` validates the local path, handles transcription and caching,
  and returns normalized segments.
- `videomind.py` is the single command-line entry point.

The frozen `data/test2.small.transcript.json` fixture is retained for the next
separately authorized sentence-reconstruction stage without rerunning
Faster-Whisper. VideoMind currently stops before that stage.
