"""
query_expansion.py
------------------
Phase 2 · Query Expansion

TODO: Implement three query expansion strategies

wordnet_expand(query, max_synonyms_per_term, pos_filter):
  Add synonyms from WordNet for each query term. Return expanded token list.

BERTExpander Class:
  - __init__(index, model_name): Load sentence-transformer model.
    Pre-compute embeddings for top-5000 vocab terms (by DF).
  
  - expand(query, top_k): Encode query, find top_k similar vocab terms by cosine sim.
    Return original_terms + expansion terms.

RocchioExpander Class (Pseudo-Relevance Feedback):
  - __init__(index, alpha, beta, gamma): Store Rocchio coefficients.
  
  - _doc_vector(doc_id, vocab): Build TF-IDF vector for doc.
    Return dict: term → (1 + log(tf)) * idf.
  
  - expand(query_terms, relevant_docs, nonrelevant_docs, top_expansion):
    Apply Rocchio: q_new = α·q + β·rel_centroid - γ·nonrel_centroid.
    Return original_terms + top_expansion new terms.

expand_query(query, method, index, bert_expander, rocchio_expander, top_docs):
  Dispatch to wordnet_expand, BERTExpander.expand, or RocchioExpander.expand.
  Return expanded token list.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from preprocessing import preprocess_query

if TYPE_CHECKING:
    from indexer import InvertedIndex
