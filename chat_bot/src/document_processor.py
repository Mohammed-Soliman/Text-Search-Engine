"""
document_processor.py
---------------------
Upload handler.

What it stores per document:
  - Chunks in the InvertedIndex (for BM25/TF-IDF scoring)
  - Sentences with their positions (for the "found here" result display)

Sentence store structure (in-memory dict):
  {
    "doc_id": {
      "filename":  "paper.pdf",
      "sentences": [
        {
          "idx":      0,           ← sentence number in file
          "page":     1,           ← estimated page (250 words/page)
          "text":     "Machine learning models are ...",
          "tokens":   {"machine", "learning", "model"}  ← preprocessed
        },
        ...
      ]
    }
  }

This lets the search layer look up exactly which sentences contain
the query terms without touching the LLM at all.
"""