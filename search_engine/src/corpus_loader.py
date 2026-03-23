"""
corpus_loader.py
----------------
Phase 1 · Data Collection

TODO: Implement data loading functions

Functions to Implement:
  - load_20newsgroups(categories, max_docs, subset): Fetch from sklearn dataset.
    Return list of dicts with: id, title, text, category.
  
  - load_local_files(raw_dir): Load all .txt and .json files from directory.
    For each file create dict with: id, title, text.
    Return list of dicts.
  
  - scrape_urls(urls): [BONUS] Scrape text from URLs using requests + BeautifulSoup.
    Extract title and <p> text. Return list of dicts with: id, title, text, url.
  
  - save_corpus(docs, path): Write docs list to JSON file (create dirs if needed).
    Print confirmation.
  
  - load_corpus(path): Read docs list from JSON file. Print confirmation.
    Return list of dicts.
"""

import json
import os
from pathlib import Path
from typing import Optional

from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
##done 

