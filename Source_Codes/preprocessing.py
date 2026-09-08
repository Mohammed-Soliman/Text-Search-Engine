import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def _ensure_nltk_data():
    """Auto-download required NLTK corpora on first run.

    This matters for deployment: a fresh host (Render, Railway, a Docker
    container, etc.) won't have these packages pre-installed the way a dev
    machine might. Without this, the very first request would crash with a
    LookupError instead of the app just quietly fetching what it needs.
    """
    required = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for find_path, package in required:
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                print(f"[nltk] Could not download '{package}': {e}")


_ensure_nltk_data()

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def tokenize(text):
    return nltk.word_tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]


def clean_text(text):
    text = re.sub(r'\\+', ' ', text)
    text = re.sub(r'#\d+;', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def stem(tokens):
    return [stemmer.stem(t) for t in tokens]


@lru_cache(maxsize=50_000)
def _lemmatize_word(word):
    return lemmatizer.lemmatize(word)


def lemmatize(tokens):
    return [_lemmatize_word(t) for t in tokens]


def remove_short(tokens, min_len=2):
    return [t for t in tokens if len(t) >= min_len]


def remove_long(tokens, max_len=45):
    return [t for t in tokens if len(t) <= max_len]


def preprocess(text, do_stem=False, do_lemma=True, do_stopwords=True, min_token_len=2, longest_token_len=45):

    text = clean_text(text.lower())
    tokens = tokenize(text)
    tokens = remove_short(tokens, min_token_len)
    tokens = remove_long(tokens, longest_token_len)

    if do_stopwords:
        tokens = remove_stopwords(tokens)

    if do_stem:
        tokens = stem(tokens)
    elif do_lemma:
        tokens = lemmatize(tokens)

    return tokens