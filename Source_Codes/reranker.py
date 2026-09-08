from sentence_transformers import SentenceTransformer, util
import torch

class BERTReranker:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading BERT model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("BERT model ready")

        self._doc_embeddings = {}
        self._cached_index_id = None

    def _doc_text(self, doc_id, index):
        doc_info = index.doc_collection.get(doc_id, {})
        title = doc_info.get("title", "")
        body = doc_info.get("body", "")
        return (title + " " + body[:300]).strip()

    def clear_cache(self):
        self._doc_embeddings = {}
        self._cached_index_id = None

    def _get_doc_embeddings(self, doc_ids, index):
        if self._cached_index_id != id(index):
            # a different corpus is loaded now - old embeddings don't apply
            self._doc_embeddings = {}
            self._cached_index_id = id(index)

        missing = [d for d in doc_ids if d not in self._doc_embeddings]
        if missing:
            texts = [self._doc_text(d, index) for d in missing]
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            for doc_id, emb in zip(missing, embeddings):
                self._doc_embeddings[doc_id] = emb

        return torch.stack([self._doc_embeddings[d] for d in doc_ids])

    def precompute(self, index, batch_size=64):
        """Optionally warm the cache for the whole corpus up front (e.g. right
        after building the index), so the very first SBERT search is fast
        instead of paying the embedding cost on that first request."""
        doc_ids = list(index.doc_collection.keys())
        for i in range(0, len(doc_ids), batch_size):
            self._get_doc_embeddings(doc_ids[i:i + batch_size], index)

    def rerank(self, query, initial_results, index, top_k=10):
        if not initial_results:
            return []

        doc_ids = [doc_id for doc_id, _ in initial_results]
        doc_embeddings = self._get_doc_embeddings(doc_ids, index)

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarity_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

        results = [
            (doc_ids[i], float(similarity_scores[i]))
            for i in range(len(doc_ids))
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]
    
    def rerank_and_show(self, query, initial_results, index, top_k=10):
        reranked = self.rerank(query, initial_results, index, top_k)
        
        original_rank = {}
        for i, (doc_id, _) in enumerate(initial_results):
            original_rank[doc_id] = i + 1
        
        print(f"\nBERT Re-ranked Results for: '{query}'")
        print("-" * 65)
        
        for new_rank, (doc_id, bert_score) in enumerate(reranked, start=1):
            old_rank = original_rank.get(doc_id, "")
            movement = ""
            if isinstance(old_rank, int):
                if new_rank < old_rank:
                    movement = f"  *down* was #{old_rank}"
                elif new_rank > old_rank:
                    movement = f"  *up* was #{old_rank}"
            
            print(f"[{new_rank}] Doc {doc_id} | Score: {bert_score:.4f} {movement}")
            print()
        
        return reranked
    
    def search_and_rerank(self, query, retriever, index, bm25_top_k=50, final_top_k=10):
        print(f"Step 1: Getting {bm25_top_k} results from BM25 ...")
        bm25_results = retriever.search(query, method="bm25", top_k=bm25_top_k)
        
        print(f"Step 2: Re-ranking with BERT to get top {final_top_k} ...")
        final_results = self.rerank_and_show(query, bm25_results, index, top_k=final_top_k)
        
        return final_results
