"""
retrieval.py
------------
Phase 2 · Query Processing & Ranking

TODO: Implement TFIDFRanker, BM25Ranker, and SearchEngine classes

TFIDFRanker Class (Vector Space Model with cosine similarity):
  - __init__(index): Store index reference.
  
  - score(query_terms, top_k): Score documents using TF-IDF.
    For each unique term: accumulate dot products (IDF * doc_weight * query_weight).
    Normalize by document norms (cosine similarity). Return top_k as (doc_id, score).
  
  - _doc_norms(query_terms): Compute L2 norm for each doc containing query terms.
    Weight = (1 + log(tf)) * idf. Return dict: doc_id → sqrt(sum of weights²).

BM25Ranker Class (Okapi BM25):
  - __init__(index, k1=1.5, b=0.75): Store parameters.
  
  - score(query_terms, top_k): Score using BM25 formula.
    For each term: score += IDF * [tf * (k1+1)] / [tf + k1*(1-b+b*|d|/avgdl)].
    Return top_k as (doc_id, score).
  
  - _idf(term): Return log((N - df + 0.5) / (df + 0.5) + 1).

SearchEngine Class (unified interface):
  - __init__(index): Create TFIDFRanker and BM25Ranker instances.
  
  - search(query, model, top_k, **kwargs): Preprocess query, select ranker,
    get scores, enrich with metadata from doc_store.
    Return list of dicts: {rank, doc_id, score, title, snippet, category}.
  
  - search_boolean(query): AND of all query terms.
    Start with first term's postings, intersect with others.
    Return list of matching doc_ids.
"""

import math
from collections import defaultdict

from preprocessing import preprocess_query
