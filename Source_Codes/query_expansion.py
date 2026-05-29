import math
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import wordnet

from preprocessing import preprocess


class QueryExpander:

    def __init__(self, index, retriever):
        self.index     = index
        self.retriever = retriever

    def _doc_vector(self, doc_id):
        vector = {}
        for term, entry in self.index.index.items():
            if doc_id in entry["postings"]:
                tf = entry["postings"][doc_id]
                tf_weight = 1 + math.log(tf) if tf > 0 else 0
                vector[term] = tf_weight * self.index.idf(term)
        return vector

    def _centroid(self, doc_ids):
        if not doc_ids:
            return {}

        centroid = defaultdict(float)
        for doc_id in doc_ids:
            for term, score in self._doc_vector(doc_id).items():
                centroid[term] += score

        for term in centroid:
            centroid[term] /= len(doc_ids)

        return dict(centroid)

    def _cosine_similarity_matrix(self, query_vec, matrix):
        dot_products = matrix @ query_vec.T

        query_norm  = np.linalg.norm(query_vec)
        matrix_norm = np.linalg.norm(matrix, axis=1)

        denom = query_norm * matrix_norm
        if denom == 0:
            return 0

        return dot_products / denom

    def rocchio(self, query_tokens, relevant_ids, non_relevant_ids=None,alpha=1.0, beta=0.75, gamma=0.15, top_terms=20):

        new_query = defaultdict(float)
        for token in query_tokens:
            new_query[token] += alpha

        if relevant_ids:
            rel_centroid = self._centroid(relevant_ids)
            for term, score in rel_centroid.items():
                new_query[term] += beta * score

        if non_relevant_ids:
            non_rel_centroid = self._centroid(non_relevant_ids)
            for term, score in non_rel_centroid.items():
                new_query[term] -= gamma * score

        new_query = {t: w for t, w in new_query.items() if w > 0}
        ranked_terms = sorted(new_query, key=lambda t: new_query[t], reverse=True)

        return ranked_terms[:top_terms]

    def pseudo_relevance_feedback(self, query, method="bm25",top_k=5, top_terms=15):
        print(f"[PRF] Running initial search for: '{query}'")
        initial_results = self.retriever.search(query, method=method, top_k=top_k)

        if not initial_results:
            print("[PRF] No initial results. Returning original query.")
            return query, []

        relevant_ids  = [doc_id for doc_id, _ in initial_results]
        query_tokens  = preprocess(query)

        expanded_tokens = self.rocchio(query_tokens, relevant_ids, top_terms=top_terms)
        expanded_query  = " ".join(expanded_tokens)

        print(f"[PRF] Expanded query: '{expanded_query}'")
        new_results = self.retriever.search(expanded_query, method=method, top_k=10)
        return expanded_query, new_results

    def expand_with_wordnet(self, query, max_synonyms_per_word=2):
        tokens   = preprocess(query)
        vocab    = set(self.index.vocab())
        expanded = list(tokens)

        for token in tokens:
            count = 0
            for synset in wordnet.synsets(token):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace("_", " ").lower()
                    if (synonym != token and synonym not in expanded and " " not in synonym and synonym in vocab):
                        expanded.append(synonym)
                        count += 1
                if count >= max_synonyms_per_word:
                    break

        expanded_query = " ".join(expanded)
        print(f"[WordNet] Original : {' '.join(tokens)}")
        print(f"[WordNet] Expanded : {expanded_query}")
        return expanded_query

    def expand_with_sbert(self, query, sbert_model, top_n=3, min_similarity=0.45):
        tokens = preprocess(query)
        vocab_sample  = list(self.index.vocab())

        query_str  = " ".join(tokens)
        all_texts  = [query_str] + vocab_sample
        embeddings = sbert_model.encode(all_texts)

        query_emb = embeddings[0].reshape(1, -1)
        vocab_embs = embeddings[1:]

        sims = self._cosine_similarity_matrix(query_emb, vocab_embs)

        ranked_indices = np.argsort(sims)[::-1]

        expanded = list(tokens)
        count = 0
        for idx in ranked_indices:
            word = vocab_sample[idx]
            if (word not in expanded and sims[idx] >= min_similarity and " " not in word):
                expanded.append(word)
                count += 1
            if count >= top_n:
                break

        expanded_query = " ".join(expanded)
        print(f"[SBERT]    Original : {query_str}")
        print(f"[SBERT]    Expanded : {expanded_query}")
        return expanded_query

    def full_expansion(self, query, method="bm25", top_k_prf=5):
        wordnet_query = self.expand_with_wordnet(query)
        expanded_query, results = self.pseudo_relevance_feedback(wordnet_query, method=method, top_k=top_k_prf)
        return expanded_query, results

    def embedding_expansion(self, query, w2v_model, method="bm25", top_k_prf=5):
        w2v_query = self.expand_with_word2vec(query, w2v_model)
        expanded_query, results = self.pseudo_relevance_feedback( w2v_query, method=method, top_k=top_k_prf)
        return expanded_query, results