# src/ingestion/transcriber.py
import whisper
import json
import os
from pathlib import Path
from tqdm import tqdm


def transcribe_video(
    video_path: str,
    output_dir: str = "data/transcripts",
    model_size: str = "base"
) -> dict:
    """
    Transcribe a video using Whisper.
    Returns transcript with word-level timestamps.
    Saves output as JSON.
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{video_name}.json"

    # Load Whisper model
    print(f"Loading Whisper {model_size}...")
    model = whisper.load_model(model_size, device="cpu")

    # Transcribe
    print(f"Transcribing: {video_path.name}")
    print("This may take a few minutes on CPU...")
    result = model.transcribe(
        str(video_path),
        verbose=False,
        language="en",
        task="transcribe",
        word_timestamps=True
    )

    # Structure output
    transcript = {
        "video": video_path.name,
        "duration": result.get("duration", 0),
        "language": result.get("language", "en"),
        "segments": []
    }

    for seg in tqdm(result["segments"], desc="Processing segments"):
        transcript["segments"].append({
            "id": seg["id"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            "words": [
                {
                    "word": w["word"].strip(),
                    "start": round(w["start"], 2),
                    "end": round(w["end"], 2)
                }
                for w in seg.get("words", [])
            ]
        })

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript saved → {output_file}")
    print(f"Total segments: {len(transcript['segments'])}")
    print(f"\nSample segment:")
    print(f"  [{transcript['segments'][0]['start']}s → {transcript['segments'][0]['end']}s]")
    print(f"  {transcript['segments'][0]['text']}")

    return transcript


if __name__ == "__main__":
    video_file = "data/videos/MIT6_034F10_lec01_300k.mp4"
    transcript = transcribe_video(video_file, model_size="base")
    print(f"\nDone. {len(transcript['segments'])} segments transcribed.")