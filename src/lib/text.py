"""Text cleaning for review bodies.

Sentiment scoring and bag-of-words modelling want different inputs, so this
module exposes two cleaners rather than one.

VADER derives much of its intensity signal from punctuation runs, ALL-CAPS
emphasis and emoji -- ``"This is AMAZING!!!"`` is a stronger reading than
``"this is amazing"``. Anything scored with VADER goes through
:func:`for_vader`, which preserves all three.

TF-IDF, LDA and K-Means operate on lowercased token counts and cannot use
punctuation or casing at all. Those consumers get :func:`for_bagofwords`.
"""

from __future__ import annotations

import html
import re

__all__ = ["for_vader", "for_bagofwords", "word_count"]

# Expansion is anchored to word boundaries: an unanchored replace would rewrite
# "Lolita" and "lollipop" via the "lol" entry, which matters on a clothing
# corpus where those terms are frequent.
_ABBREVIATIONS = {
    "brb": "be right back",
    "lol": "laughing out loud",
    "idk": "i do not know",
    "imo": "in my opinion",
    "tbh": "to be honest",
}

_ABBREV_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, _ABBREVIATIONS)) + r")\b",
    flags=re.IGNORECASE,
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]")


def _base(text: object) -> str:
    """Unescape entities, drop markup, normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _expand(text: str) -> str:
    return _ABBREV_RE.sub(lambda m: _ABBREVIATIONS[m.group(0).lower()], text)


def for_vader(text: object) -> str:
    """Clean a review for lexicon sentiment scoring.

    Markup and stray whitespace are removed; punctuation, casing and emoji are
    preserved, because VADER reads them as intensity.
    """
    return _expand(_base(text))


def for_bagofwords(text: object) -> str:
    """Clean a review for TF-IDF, LDA or K-Means.

    Lowercased and stripped of punctuation. Abbreviations are expanded first,
    while the word boundaries that expansion depends on still exist.
    """
    text = _expand(_base(text))
    text = _NON_WORD_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def word_count(text: object) -> int:
    """Whitespace token count of a review body."""
    return len(_base(text).split())
