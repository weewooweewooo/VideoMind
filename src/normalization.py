"""Shared lexical normalization for transcript retrieval."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from src.config import (
    COMPOUND_MIN_DOCUMENT_FREQUENCY,
    COMPOUND_MIN_PART_LENGTH,
    STOPWORDS,
    TITLE_ABBREVIATIONS,
)


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_SENTENCE_END_PATTERN = re.compile(r"[.!?]+(?=\s|$)")


def _light_stem(token: str) -> str:
    """Normalize a small set of safe English suffix patterns."""
    if (
        len(token) <= 3
        or token.isdigit()
        or token in STOPWORDS
        or "'" in token
    ):
        return token

    replacements = (
        ("izations", "ize"),
        ("ization", "ize"),
        ("izing", "ize"),
        ("ized", "ize"),
        ("ating", "ate"),
        ("ated", "ate"),
        ("izes", "ize"),
        ("ates", "ate"),
    )
    for suffix, replacement in replacements:
        if token.endswith(suffix):
            stem = token[: -len(suffix)] + replacement
            return stem if len(stem) >= 4 else token

    for suffix in ("ations", "ation"):
        if token.endswith(suffix):
            root = token[: -len(suffix)]
            if len(root) < 4:
                return token
            if root.endswith(("form", "ment")):
                return root
            if root.endswith("vers"):
                return root + "e"
            return root + "ate"

    # Restrict -ies -> -y to common long consonant-y plural families.
    if len(token) >= 8 and token.endswith(
        ("anies", "aries", "icies", "ities", "ogies")
    ):
        return token[:-3] + "y"

    if token.endswith("s") and not token.endswith(
        ("es", "is", "ss", "us", "ys")
    ):
        stem = token[:-1]
        return stem if len(stem) >= 4 else token
    return token


def _base_tokens(text: str) -> list[str]:
    return [
        _light_stem(match.group(0).lower())
        for match in _TOKEN_PATTERN.finditer(text)
    ]


def _discover_compound_splits(
    tokenized_chunks: list[list[str]],
) -> dict[str, tuple[str, str]]:
    document_frequencies = Counter(
        token
        for tokens in tokenized_chunks
        for token in set(tokens)
        if token not in STOPWORDS
    )
    splits: dict[str, tuple[str, str]] = {}
    for token in sorted(document_frequencies):
        if len(token) < COMPOUND_MIN_PART_LENGTH * 2 or token.isdigit():
            continue
        candidates: list[tuple[int, int, int, str, str]] = []
        for position in range(
            COMPOUND_MIN_PART_LENGTH,
            len(token) - COMPOUND_MIN_PART_LENGTH + 1,
        ):
            left, right = token[:position], token[position:]
            left_frequency = document_frequencies.get(left, 0)
            right_frequency = document_frequencies.get(right, 0)
            if (
                left_frequency < COMPOUND_MIN_DOCUMENT_FREQUENCY
                or right_frequency < COMPOUND_MIN_DOCUMENT_FREQUENCY
            ):
                continue
            candidates.append(
                (
                    left_frequency + right_frequency,
                    -abs(len(left) - len(right)),
                    -position,
                    left,
                    right,
                )
            )
        if candidates:
            _, _, _, left, right = max(candidates)
            splits[token] = (left, right)
    return splits


def _apply_compound_splits(
    base_tokens: list[str],
    compound_splits: Mapping[str, tuple[str, str]],
) -> list[str]:
    tokens: list[str] = []
    for token in base_tokens:
        tokens.extend(compound_splits.get(token, (token,)))
    return [token for token in tokens if token not in STOPWORDS]


def _tokenize(
    text: str,
    compound_splits: Mapping[str, tuple[str, str]],
) -> list[str]:
    return _apply_compound_splits(_base_tokens(text), compound_splits)


def _split_sentences(text: str) -> list[str]:
    """Split transcript text on its punctuation while preserving display text."""
    fragments: list[str] = []
    start = 0
    for match in _SENTENCE_END_PATTERN.finditer(text):
        fragment = text[start : match.end()].strip()
        if (
            match.group(0) == "."
            and fragment
            and fragment.lower().split()[-1] in TITLE_ABBREVIATIONS
        ):
            continue
        if fragment and _TOKEN_PATTERN.search(fragment):
            fragments.append(fragment)
        start = match.end()

    final_fragment = text[start:].strip()
    if final_fragment and _TOKEN_PATTERN.search(final_fragment):
        fragments.append(final_fragment)

    merged: list[str] = []
    for fragment in fragments:
        if merged and len(_TOKEN_PATTERN.findall(fragment)) < 3:
            merged[-1] = f"{merged[-1]} {fragment}"
        else:
            merged.append(fragment)
    if len(merged) > 1 and len(_TOKEN_PATTERN.findall(merged[0])) < 3:
        merged[1] = f"{merged[0]} {merged[1]}"
        del merged[0]
    return merged
