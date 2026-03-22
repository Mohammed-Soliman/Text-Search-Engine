"""
search_handler.py
-----------------
Pure IR search — no answer generation.

Pipeline:
  1. query_translator.py  →  keywords
  2. SpellCorrector        →  fix typos in those keywords
  3. query_expansion.py   →  optional expansion
  4. SearchEngine (BM25/TF-IDF)  →  ranked doc chunks
  5. DocumentProcessor.find_sentences()  →  actual sentences per doc
  6. Format result:  "Found in doc1, doc2 …\ndoc1: '…' (page X)"

Nothing here touches the LLM except the tiny keyword translation call.
"""
