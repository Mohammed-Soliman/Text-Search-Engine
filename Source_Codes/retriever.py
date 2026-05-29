import math
from collections import defaultdict

from indexer import InvertedIndex
from preprocessing import preprocess


class Retriever:
    def __init__(self, index):
        self.index = index

    def boolean_search(self, query):
        tokens = preprocess(query)
        if not tokens:
            return []

        result = set(self.index.get_postings(tokens[0]).keys())

        for token in tokens[1:]:
            postings = set(self.index.get_postings(token).keys())
            result = result & postings
            if not result:
                break

        return list(result)

    def search_tfidf(self, query, top_k=10):
        tokens = preprocess(query)
        if not tokens:
            return []

        scores = defaultdict(float)

        for term in tokens:
            postings = self.index.get_postings(term)
            idf = self.index.idf(term)

            for doc_id, tf in postings.items():
                tf_weight = 1 + math.log(tf) if tf > 0 else 0
                scores[doc_id] += tf_weight * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_bm25(self, query, top_k=10, k1=1.5, b=0.75):
        tokens = preprocess(query)
        if not tokens:
            return []

        avgdl = self.index.avg_doc_length()
        scores = defaultdict(float)

        for term in tokens:
            postings = self.index.get_postings(term)
            idf = self.index.idf(term)

            for doc_id, tf in postings.items():
                dl = self.index.doc_lengths.get(doc_id, 1)

                numerator   = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (dl / avgdl))
                bm25_tf = numerator / denominator

                scores[doc_id] += idf * bm25_tf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _get_doc_tfidf_vector(self, doc_id):
        vector = {}
        for term, entry in self.index.index.items():
            if doc_id in entry["postings"]:
                tf = entry["postings"][doc_id]
                tf_weight = 1 + math.log(tf) if tf > 0 else 0
                vector[term] = tf_weight * self.index.idf(term)
        return vector

    def _cosine_similarity(self, vec_a, vec_b):
        dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_b)

        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def search_vsm(self, query, top_k=10):
        tokens = preprocess(query)
        if not tokens:
            return []

        query_vec = defaultdict(float)
        for token in tokens:
            query_vec[token] += 1

        for token in query_vec:
            query_vec[token] *= self.index.idf(token)

        candidate_docs = set()
        for token in tokens:
            candidate_docs.update(self.index.get_postings(token).keys())

        scores = {}
        for doc_id in candidate_docs:
            doc_vec = self._get_doc_tfidf_vector(doc_id)
            scores[doc_id] = self._cosine_similarity(query_vec, doc_vec)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search(self, query, method="bm25", top_k=10):
        if method == "tfidf":
            return self.search_tfidf(query, top_k)
        elif method == "bm25":
            return self.search_bm25(query, top_k)
        elif method == "vsm":
            return self.search_vsm(query, top_k)
        elif method == "boolean":
            doc_ids = self.boolean_search(query)
            return [(doc_id, 1.0) for doc_id in doc_ids[:top_k]]
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: bm25, tfidf, vsm, boolean")