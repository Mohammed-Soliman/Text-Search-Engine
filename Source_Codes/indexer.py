import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from preprocessing import preprocess


class InvertedIndex:

    def __init__(self):
        self.index = defaultdict( lambda: {"df": 0, "postings": {}})
        self.doc_lengths = {}
        self.doc_collection = {}
        self.N = 0

    def build(self, docs, **argments):
        print(f"building index for {len(docs)} documents ...")
        self.N = len(docs)

        for doc in tqdm(docs, desc="Indexing"):
            doc_id = doc["index"]
            text = doc.get("text", "")
            tokens = preprocess(text, **argments)

            self.doc_collection[doc_id] = {
                "category": doc.get("category", ""),
                "title": doc.get("title", ""),
                "body": doc.get("text", "")
            }

            tf_map = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1

            self.doc_lengths[doc_id] = len(tokens)

            for term, tf in tf_map.items():
                self.index[term]["postings"][doc_id] = tf
                self.index[term]["df"] += 1

        self.index = dict(self.index)
        print(f"done! vocab size: {len(self.index):,}")

    def get_postings(self, term):
        entry = self.index.get(term, {})
        return entry.get("postings", {})

    def get_df(self, term):
        return self.index.get(term, {}).get("df", 0)

    def idf(self, term):
        df = self.get_df(term)
        if df == 0:
            return 0.0
        return math.log((self.N + 1) / (df + 1)) + 1

    def avg_doc_length(self):
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def vocab(self):
        return list(self.index.keys())

    def save(self, path="data/processed/index.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"index saved to {path}")

    def load(self, path="data/processed/index.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        print(f"index loaded from {path} (N={self.N}, vocab={len(self.index):,})")

    def stats(self):
        return {
            "total_docs": self.N,
            "vocab_size": len(self.index),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "total_postings": sum(len(v["postings"]) for v in self.index.values())
        }

    def return_inverted_index_dict(self):
        return self.index