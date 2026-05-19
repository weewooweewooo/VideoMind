import yt_dlp
import os
from pathlib import Path


def download_lecture(url: str, output_dir: str = "data/videos") -> str:
    """
    Download a lecture video from YouTube.
    Returns the path to the downloaded file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "best[height<=480][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Ensure .mp4 extension
        if not filename.endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4"
        print(f"\nDownloaded: {filename}")
        return filename


if __name__ == "__main__":
    # Test with Karpathy's Let's build GPT — great lecture content
    url = "https://www.youtube.com/watch?v=kCc8FmEb1nY"
    path = download_lecture(url)
    print(f"Saved to: {path}")
