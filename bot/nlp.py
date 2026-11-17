"""Text preprocessing and entity extraction (NLTK based).

Only NLTK components that ship with the library are used (regex tokenizer,
Porter stemmer, n-gram helpers), so the bot works without downloading any
corpora at runtime.
"""
from __future__ import annotations

import re

from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

_stemmer = PorterStemmer()

# Question words (how/what/when/...) and negations are deliberately kept: they
# carry meaning for follow-up questions such as "how long will it take?".
STOPWORDS = frozenset(
    """a an the is are was were be been am i me my we our you your it its to of in
    on at for and or so do does did can could would will shall please just
    there this that these those with about what which""".split()
)

_CONTRACTIONS = [
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'m\b", " am"),
    (r"'ll\b", " will"),
    (r"'ve\b", " have"),
    (r"'d\b", " would"),
    (r"'s\b", ""),
]

ORDER_ID_RE = re.compile(r"(?<![\w.])#?(\d{5,8})(?![\w.])")
NAME_RE = re.compile(
    r"\b(?:my name is|call me|you can call me)\s+([A-Za-z][A-Za-z\-']{1,24})",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_NOT_NAMES = {"a", "an", "the", "not", "just", "very", "so", "to", "in", "on", "me", "my"}


def normalize(text: str) -> str:
    text = (text or "").lower().replace("’", "'")
    for pattern, repl in _CONTRACTIONS:
        text = re.sub(pattern, repl, text)
    text = EMAIL_RE.sub(" emailaddr ", text)
    text = ORDER_ID_RE.sub(" ordernum ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens with contractions expanded."""
    return [t for t in wordpunct_tokenize(normalize(text)) if t.isalnum()]


def stem(token: str) -> str:
    return _stemmer.stem(token)


def preprocess(text: str) -> list[str]:
    """Normalise -> tokenise -> drop stopwords -> stem."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def extract_entities(text: str) -> dict[str, str]:
    """Pull simple slots (order id, name, email) out of a user message."""
    entities: dict[str, str] = {}
    m = ORDER_ID_RE.search(text or "")
    if m:
        entities["order_id"] = m.group(1)
    m = NAME_RE.search(text or "")
    if m and m.group(1).lower() not in _NOT_NAMES:
        entities["name"] = m.group(1).capitalize()
    m = EMAIL_RE.search(text or "")
    if m:
        entities["email"] = m.group(0)
    return entities
