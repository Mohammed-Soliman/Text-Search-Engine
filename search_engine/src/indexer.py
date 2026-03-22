"""
indexer.py
----------
Phase 1 · Inverted Index

TODO: Implement InvertedIndex class to build and persist inverted index

InvertedIndex Class:
  - __init__(): Initialize empty structures:
    self.index = defaultdict with {term: {df, postings}},
    self.doc_lengths, self.doc_store, self.N
  
  - build(docs, **preprocess_kwargs): For each doc, preprocess text, calculate TFs,
    store metadata in doc_store, update index with postings and DF.
  
  - get_postings(term): Return {doc_id: tf} for term (empty dict if absent).
  
  - get_df(term): Return document frequency (0 if absent).
  
  - idf(term): Return log((N+1)/(df+1)) + 1 (0.0 if df=0).
  
  - avg_doc_length: Property that returns average doc length or 0.0.
  
  - vocab(): Return list of all terms.
  
  - save(path): Pickle __dict__ to file (create dirs if needed).
  
  - load(path): Unpickle file and update __dict__.
  
  - stats(): Return dict with total_docs, vocab_size, avg_doc_length, total_postings.
"""

import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from preprocessing import preprocess
