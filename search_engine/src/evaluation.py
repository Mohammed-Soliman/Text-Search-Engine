"""
evaluation.py
-------------
Phase 2 · Evaluation  (1 point + bonus visualizations)

TODO: Implement all evaluation metrics and the Evaluator class

Core Functions to Implement:
  - precision(retrieved, relevant): Calculate precision (hits / len(retrieved))
  - recall(retrieved, relevant): Calculate recall (hits / len(relevant))
  - f1(p, r): Calculate F1 score
  - fbeta(p, r, beta): Calculate F-beta score
  - precision_at_k(retrieved, relevant, k): Precision at top k
  - average_precision(retrieved, relevant): AP for MAP calculation
  - dcg(retrieved, relevant): Discounted Cumulative Gain
  - ndcg(retrieved, relevant): Normalized DCG (between 0-1)
  - interpolated_pr_curve(retrieved, relevant, recall_levels): 11-point P-R curve
  
Evaluator Class:
  - __init__(engine, qrels, model, top_k): Store parameters
  - run(queries): For each query, retrieve results and calculate all metrics.
    Return DataFrame with per-query metrics + MAP score.

Helper Functions:
  - load_qrels(path): Parse TREC qrels file (query_id, 0, doc_id, relevance)
    Return dict: query_id → set of relevant doc_ids (where relevance >= 1)
  
  - plot_pr_curve(retrieved, relevant, query_label): Create Plotly figure
    showing precision-recall curve using interpolated_pr_curve()
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
