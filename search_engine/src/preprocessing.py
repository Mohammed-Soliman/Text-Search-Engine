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

import spacy

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

def remove_ner(tokens: list[str], ner_tags: set[str] = {"PERSON", "ORG", "GPE"}) -> list[str]:
    return [t for t in tokens if t.ent_type_ not in ner_tags]

def remove_short(tokens: list[str], min_len: int = 2) -> list[str]:
    return [t for t in tokens if len(t) >= min_len]

def remove_long(tokens: list[str], max_len: int = 45) -> list[str]:
    return [t for t in tokens if len(t) <= max_len]

# ─────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────

def preprocess(
    text: str,
    do_stem: bool = False,       # Porter stemming
    do_lemma: bool = True,       # WordNet lemmatization (default)
    do_stopwords: bool = True,
    do_ner: bool = False,       # [BONUS] spaCy NER filtering
    min_token_len: int = 2,
    longest_token_len: int = 45     # Longest English word is 45 chars
) -> list[str]:
    """
    Full preprocessing pipeline.

    Args:
        text:           raw input string
        do_stem:        apply Porter stemming (mutually exclusive with lemma)
        do_lemma:       apply WordNet lemmatization
        do_stopwords:   remove stopwords
        do_ner:          apply spaCy NER filtering
        min_token_len:  drop tokens shorter than this
        longest_token_len: drop tokens longer than this
    Returns:
        list of clean, normalized tokens
    """
    text   = text.lower()
    tokens = tokenize(text)
    tokens = remove_punctuation_tokens(tokens)
    tokens = remove_short(tokens, min_token_len)
    tokens = remove_long(tokens, longest_token_len)

    if do_stopwords:
        tokens = remove_stopwords(tokens)

    if do_stem:
        tokens = stem(tokens)
    elif do_lemma:
        tokens = lemmatize(tokens)

    if do_ner:
        tokens = remove_ner(tokens)

    return tokens



# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Mohammed says Information Retrieval systems are designed to retrieve relevant documents!"
    print("Raw:       ", sample)
    print("Standard:  ", preprocess(sample))
    print("Stemmed:   ", preprocess(sample, do_stem=True, do_lemma=False))
    print("No Named Entites:   ", preprocess(sample, do_ner=True))
