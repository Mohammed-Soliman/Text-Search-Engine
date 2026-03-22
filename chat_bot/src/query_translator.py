"""
query_translator.py
-------------------
The ONLY file that touches the Anthropic API.
Job: take a natural-language sentence → return a short keyword query.

That's it. No answer generation, no summarization.

Two modes:
  - "api"   : one tiny Anthropic call, returns 3-7 keywords
  - "nltk"  : pure NLTK fallback (no API key needed)

The NLTK fallback works surprisingly well because your preprocessing
pipeline already does stopword removal + lemmatization.
"""