"""
clustering.py
-------------
[BONUS] Document Clustering using K-Means on TF-IDF vectors

TODO: Implement DocumentClusterer class

DocumentClusterer Class:
  - __init__(index, n_clusters, use_svd, n_components): Store parameters
  
  - _build_tfidf_matrix(): Build doc × term TF-IDF matrix from index.
    For each term-doc pair: weight = (1 + log(tf)) * idf(term).
    L2 normalize each row. Return (matrix, doc_ids, vocab).
  
  - fit(): Build TF-IDF matrix, optionally apply SVD for dimensionality reduction,
    then fit MiniBatchKMeans. Store labels in _labels.
  
  - predict(doc_id): Return cluster label for a document.
  
  - cluster_docs(cluster_id): Return all doc_ids in a cluster.
  
  - cluster_keywords(cluster_id, top_n): Return top N keywords for cluster
    by averaging TF-IDF vectors of docs in cluster.
  
  - summary(): Return list of dicts: {cluster, size, keywords} for each cluster.
  
  - save(path): Pickle the clusterer to file.
  
  - load(path): Unpickle the clusterer from file.
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from indexer import InvertedIndex
