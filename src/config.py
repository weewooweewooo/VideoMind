"""Fixed configuration for the current VideoMind application."""

import re


YOUTUBE_CANONICAL_SCHEME = "https"
YOUTUBE_SUPPORTED_SCHEMES = frozenset({"http", YOUTUBE_CANONICAL_SCHEME})
YOUTUBE_ROOT_HOST = "youtube.com"
YOUTUBE_CANONICAL_HOST = f"www.{YOUTUBE_ROOT_HOST}"
YOUTUBE_SHORT_HOST = "youtu.be"
YOUTUBE_WATCH_HOSTS = frozenset(
    {
        YOUTUBE_ROOT_HOST,
        YOUTUBE_CANONICAL_HOST,
        f"m.{YOUTUBE_ROOT_HOST}",
        f"music.{YOUTUBE_ROOT_HOST}",
    }
)
YOUTUBE_WATCH_PATH = "/watch"
YOUTUBE_VIDEO_ID_QUERY_KEY = "v"
YOUTUBE_VIDEO_ID_LENGTH = 11
YOUTUBE_VIDEO_ID_PATTERN = re.compile(
    rf"^[A-Za-z0-9_-]{{{YOUTUBE_VIDEO_ID_LENGTH}}}$"
)

STOPWORDS = frozenset(
    (
        "a an and are as at be been being but by can could did do does doing "
        "for from had has have having he her hers him his how i if in into is "
        "it its itself may me might my of on or our ours she should so that "
        "the their theirs them themselves then there these they this those "
        "through to was we were what when where which while who why will with "
        "work would you your yours"
    ).split()
)
TITLE_ABBREVIATIONS = frozenset(
    ("dr.", "jr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "st.")
)
COMPOUND_MIN_PART_LENGTH = 3
COMPOUND_MIN_DOCUMENT_FREQUENCY = 2
DEFAULT_TOP_K = 5

WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BEAM_SIZE = 5

TRANSCRIPT_CHUNK_WORDS = 70
