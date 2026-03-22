"""
indexer.py
----------
Phase 1 · Inverted Index

Builds and persists an inverted index from a document corpus.

Index structure (stored as JSON):
  {
    "term": {
      "df":      document frequency (int),
      "postings": {
        "doc_id": tf  (int)
      }
    }
  }

Also stores:
  - doc_lengths: { doc_id: number_of_tokens }
  - doc_store:   { doc_id: { title, text } }   ← for snippet display
  - N:           total number of documents
"""

import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from preprocessing import preprocess


# ─────────────────────────────────────────────────────────────
# InvertedIndex class
# ─────────────────────────────────────────────────────────────

class InvertedIndex:

    def __init__(self):
        # { term: {"df": int, "postings": {doc_id: tf}} }
        self.index: dict = defaultdict(lambda: {"df": 0, "postings": {}})

        self.doc_lengths: dict[str, int] = {}   # { doc_id: token_count }
        self.doc_store:   dict[str, dict] = {}   # { doc_id: {title, text} }
        self.N: int = 0                           # total documents

    # ── build ────────────────────────────────────────────────

    def build(self, docs: list[dict], **preprocess_kwargs) -> None:
        """
        Build the index from a list of document dicts.

        Each doc must have: id, body
        Optional fields:    title, category
        """
        print(f"[indexer] Building index for {len(docs)} documents ...")
        self.N = len(docs)

        for doc in tqdm(docs, desc="Indexing"):
            doc_id = doc["id"]
            text   = doc.get("title", "") + " " + doc.get("body", "")
            tokens = preprocess(text, **preprocess_kwargs)

            # doc store
            self.doc_store[doc_id] = {
                "title":    doc.get("title", ""),
                "body":  doc.get("body", "")[:300],
                "category": doc.get("category", "")
            }

            # term frequencies for this doc
            tf_map: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1

            self.doc_lengths[doc_id] = len(tokens)

            # update index
            for term, tf in tf_map.items():
                self.index[term]["postings"][doc_id] = tf
                self.index[term]["df"] += 1

        self.index = dict(self.index)
        print(f"[indexer] Done. Vocabulary size: {len(self.index):,}")

    # ── query helpers ────────────────────────────────────────

    def get_postings(self, term: str) -> dict[str, int]:
        """Return { doc_id: tf } for a term (empty dict if absent)."""
        entry = self.index.get(term, {})
        return entry.get("postings", {})

    def get_df(self, term: str) -> int:
        return self.index.get(term, {}).get("df", 0)

    def idf(self, term: str) -> float:
        df = self.get_df(term)
        if df == 0:
            return 0.0
        return math.log((self.N + 1) / (df + 1)) + 1

    @property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def vocab(self) -> list[str]:
        return list(self.index.keys())

    # ── persist ──────────────────────────────────────────────

    def save(self, path: str = "data/processed/index.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[indexer] Index saved -> {path}")

    def load(self, path: str = "data/processed/index.pkl") -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        print(f"[indexer] Index loaded <- {path}  "f"(N={self.N}, vocab={len(self.index):,})")

    # ── stats ────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_docs":       self.N,
            "vocab_size":       len(self.index),
            "avg_doc_length":   round(self.avg_doc_length, 2),
            "total_postings":   sum(len(v["postings"]) for v in self.index.values())
        }


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Quick test
    docs = [
        {"id": "doc1", "title": "Hello World", "body": "This is a test."},
        {"id": "doc2", "title": "Another Test", "body": "Testing the indexer."},
        {"id": "doc3", "title": "Hello Again", "body": "Hello world again!"},
    ]

    index = InvertedIndex()
    index.build(docs, do_stem=True)
    print(index.stats())
    print("Hello Docs: ", index.get_postings("hello"))