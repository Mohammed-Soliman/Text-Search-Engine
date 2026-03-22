"""
preprocessing.py
----------------
Phase 1 · Preprocessing Pipeline

Steps (all toggleable):
  1. Lowercasing
  2. Tokenization   (regex-based, no external tokenizer needed)
  3. Stopword removal
  4. Stemming       (Porter Stemmer)
  5. Lemmatization  (WordNetLemmatizer) — [BONUS: spaCy NER filtering]

Returns clean token lists ready for indexing.
"""

import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# ─────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────

STEMMER    = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
TOKENIZER = nltk.tokenize
STOP_WORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    tokens = TOKENIZER.word_tokenize(text)
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def remove_punctuation_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if not all(c in string.punctuation for c in t)]


def stem(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(t) for t in tokens]


@lru_cache(maxsize=50_000)
def _lemmatize_word(word: str) -> str:
    return LEMMATIZER.lemmatize(word)


def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatize_word(t) for t in tokens]


def remove_short(tokens: list[str], min_len: int = 2) -> list[str]:
    return [t for t in tokens if len(t) >= min_len]


# ─────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────

def preprocess(
    text: str,
    do_stem: bool = False,       # Porter stemming
    do_lemma: bool = True,       # WordNet lemmatization (default)
    do_stopwords: bool = True,
    min_token_len: int = 2
) -> list[str]:
    """
    Full preprocessing pipeline.

    Args:
        text:           raw input string
        do_stem:        apply Porter stemming (mutually exclusive with lemma)
        do_lemma:       apply WordNet lemmatization
        do_stopwords:   remove stopwords
        min_token_len:  drop tokens shorter than this

    Returns:
        list of clean, normalized tokens
    """
    text   = text.lower()
    tokens = tokenize(text)
    tokens = remove_punctuation_tokens(tokens)
    tokens = remove_short(tokens, min_token_len)

    if do_stopwords:
        tokens = remove_stopwords(tokens)

    if do_stem:
        tokens = stem(tokens)
    elif do_lemma:
        tokens = lemmatize(tokens)

    return tokens


def preprocess_query(query: str, **kwargs) -> list[str]:
    """Thin wrapper — same pipeline, used for queries."""
    return preprocess(query, **kwargs)


# ─────────────────────────────────────────────────────────────
# [BONUS] spaCy NER — filter out named entities before indexing
# ─────────────────────────────────────────────────────────────

def preprocess_with_ner(text: str, remove_entities: bool = True) -> list[str]:
    """
    [BONUS] Use spaCy to:
      - Lemmatize tokens
      - Optionally remove named entities (PERSON, ORG, GPE …)
    """
    import spacy

    _nlp = spacy.load("en_core_web_sm", disable=["parser"])
    doc = _nlp(text[:100_000])   # spaCy max length guard

    entity_spans = set()
    if remove_entities:
        for ent in doc.ents:
            for tok in ent:
                entity_spans.add(tok.i)

    tokens = []
    for i, token in enumerate(doc):
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if remove_entities and i in entity_spans:
            continue
        lemma = token.lemma_.lower()
        if len(lemma) >= 2:
            tokens.append(lemma)

    return tokens


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Mohammed says Information Retrieval systems are designed to retrieve relevant documents!"
    print("Raw:       ", sample)
    print("Standard:  ", preprocess(sample))
    print("Stemmed:   ", preprocess(sample, do_stem=True, do_lemma=False))
    print("No Named Entites:   ", preprocess_with_ner(sample))
