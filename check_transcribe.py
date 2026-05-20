import time
import os
from src.ingestion.transcriber import transcribe_to_memory

os.environ["WHISPER_MODEL_PATH"] = "D:/models/faster-whisper-base"

url = "https://archive.org/download/CS607Lecture07/CS607_Lecture07.mp4"
video_name = "test_lecture"

print("Starting transcription...")
start = time.time()
result = transcribe_to_memory(url, video_name)
elapsed = time.time() - start

print(f"Time: {elapsed:.1f}s")
print(f"Segments: {len(result['segments'])}")
if result["segments"]:
    print(f"Sample: {result['segments'][0]['text']}")
else:
    print("NO SEGMENTS")