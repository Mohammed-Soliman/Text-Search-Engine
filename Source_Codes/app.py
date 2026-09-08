import json
import os
import time
from collections import defaultdict

from flask import Flask, jsonify, request
from corpus_loader import CorpusLoader
from evaluator import Evaluator
from feedback_store import FeedbackStore
from indexer import InvertedIndex
from preprocessing import preprocess
from query_expansion import QueryExpander
from retriever import Retriever
from reranker import BERTReranker
from spell_correction import SpellCorrector

app = Flask(__name__)

index = InvertedIndex()
retriever = None
expander = None
bert_reranker = None
spell_corrector = None
loader = CorpusLoader()
feedback_store = FeedbackStore(path=os.path.join(os.path.dirname(__file__), "data", "feedback.json"))

demo_docs = loader.load_demo()
index.build(demo_docs)
retriever = Retriever(index)
expander = QueryExpander(index, retriever)
spell_corrector = SpellCorrector(index)

try:
    bert_reranker = BERTReranker(model_name="all-MiniLM-L6-v2")
    print("BERT reranker ready")
except:
    print("BERT not available. Install: pip install sentence-transformers torch")

print("Demo corpus loaded. Server running!")


def make_result(doc_id, score, rank):
    info = index.doc_collection.get(doc_id, {})
    body = info.get("body", "")
    snippet = body[:200].replace("\n", " ")
    if len(body) > 200:
        snippet = snippet + "..."
    
    return {
        "rank": rank,
        "doc_id": doc_id,
        "title": info.get("title", f"Document {doc_id}"),
        "category": info.get("category", ""),
        "snippet": snippet,
        "score": round(float(score), 4)
    }


@app.route("/")
def home():
    html_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/api/status")
def get_status():
    return jsonify({
        "corpus_loaded": index.N > 0,
        "doc_count": index.N,
        "vocab_size": len(index.index),
        "avg_doc_length": round(index.avg_doc_length(), 1) if index.N > 0 else 0,
        "bert_available": bert_reranker is not None
    })


@app.route("/api/upload", methods=["POST"])
def upload_corpus():
    global index, retriever, expander, spell_corrector
    
    if request.form.get("use_demo"):
        docs = loader.load_demo()
        name = "Demo corpus"
    
    elif "file" in request.files:
        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        save_folder = os.path.join(os.path.dirname(__file__), "uploads_temp")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, uploaded.filename)
        uploaded.save(save_path)
        
        try:
            docs = loader.load(save_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        
        name = uploaded.filename
    else:
        return jsonify({"error": "No file or demo flag provided"}), 400
    
    index = InvertedIndex()
    index.build(docs)
    retriever = Retriever(index)
    expander = QueryExpander(index, retriever)
    spell_corrector.rebuild(index)
    
    return jsonify({
        "message": f"Loaded '{name}' successfully",
        "doc_count": index.N,
        "vocab_size": len(index.index),
        "avg_doc_length": round(index.avg_doc_length(), 1)
    })


@app.route("/api/autocomplete")
def autocomplete():
    query = request.args.get("q", "").strip().lower()
    
    if len(query) < 2 or index.N == 0:
        return jsonify([])
    
    words = query.split()
    last_word = words[-1]
    vocab = index.index
    
    matches = []
    for word in vocab:
        if word.startswith(last_word) and word != last_word:
            matches.append((word, vocab[word]["df"]))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    suggestions = [w for w, _ in matches[:8]]
    
    if len(words) > 1:
        complete_words = words[:-1]
        matching_docs = None
        
        for w in complete_words:
            docs_with_word = set(index.get_postings(w).keys())
            if docs_with_word:
                if matching_docs is None:
                    matching_docs = docs_with_word
                else:
                    matching_docs = matching_docs & docs_with_word
        
        if matching_docs:
            word_counts = defaultdict(int)
            for term, info in vocab.items():
                for doc_id in matching_docs:
                    if doc_id in info["postings"]:
                        word_counts[term] += 1
            
            extra_suggestions = [
                w for w, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                if w not in suggestions and w not in complete_words and not w.startswith(last_word)
            ]
            suggestions = suggestions[:4] + extra_suggestions[:4]
    
    return jsonify(suggestions[:8])


@app.route("/api/search", methods=["POST"])
def search():
    if index.N == 0:
        return jsonify({"error": "No corpus loaded"}), 400
    
    data = request.get_json() or {}
    user_query = data.get("query", "").strip()
    method = data.get("method", "bm25")
    expansion = data.get("expansion", "none")
    top_k = int(data.get("top_k", 10))
    
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    valid_methods = ["bm25", "tfidf", "vsm", "boolean", "sbert"]
    if method not in valid_methods:
        return jsonify({"error": f"Invalid method. Use: {', '.join(valid_methods)}"}), 400
    
    if method == "sbert" and bert_reranker is None:
        return jsonify({"error": "SBERT not available. Install: pip install sentence-transformers torch"}), 400
    
    start = time.time()
    
    # ?? Spelling / "did you mean" correction ?????????????????????????????????
    # Corrections are proposed against the *index vocabulary* (not a generic
    # dictionary) so a suggestion is guaranteed to actually be searchable.
    corrected_query, spelling_corrections = spell_corrector.correct_query(user_query)
    
    use_corrected_flag = data.get("use_corrected", None)
    auto_corrected = False
    base_query = user_query
    
    if use_corrected_flag is True:
        # user explicitly clicked "Search instead for <corrected>"
        base_query = corrected_query
        auto_corrected = True
    elif use_corrected_flag is False:
        # user explicitly clicked "No, search for what I typed"
        base_query = user_query
        auto_corrected = False
    elif spelling_corrections:
        # no explicit choice: auto-apply the correction only when the query,
        # as typed, wouldn't match anything in the index at all
        original_tokens = set(preprocess(user_query))
        has_known_token = any(t in index.index for t in original_tokens)
        if not has_known_token:
            base_query = corrected_query
            auto_corrected = True
    
    expanded = base_query
    expansion_words = []
    
    # ?? Query Expansion ????????????????????????????????????????????????????
    if expansion == "wordnet":
        try:
            expanded = expander.expand_with_wordnet(base_query, max_synonyms_per_word=2)
        except Exception as e:
            print(f"[WordNet] Error: {e}")
    
    elif expansion == "rocchio":
        try:
            expanded, _ = expander.pseudo_relevance_feedback(
                base_query, method=method if method != "sbert" else "bm25", top_k=5, top_terms=10
            )
        except Exception as e:
            print(f"[Rocchio] Error: {e}")
    
    elif expansion == "sbert":
        try:
            if bert_reranker is None:
                return jsonify({"error": "SBERT expansion not available"}), 400
            expanded = expander.expand_with_sbert(
                base_query, 
                bert_reranker.model, 
                top_n=3, 
                min_similarity=0.45
            )
            print(f"[SBERT Expansion] Original: '{base_query}' ? Expanded: '{expanded}'")
        except Exception as e:
            print(f"[SBERT Expansion] Error: {e}")
            expanded = base_query
    
    # ?? Calculate expansion words ??????????????????????????????????????????
    original_set = set(preprocess(base_query))
    expanded_set = set(preprocess(expanded))
    expansion_words = list(expanded_set - original_set)[:8]
    
    # ?? Retrieval ??????????????????????????????????????????????????????????
    if method == "sbert":
        # Use BM25 candidates, then rerank with SBERT
        candidate_k = max(50, top_k * 5)
        bm25_results = retriever.search(expanded, method="bm25", top_k=candidate_k)
        results = bert_reranker.rerank(base_query, bm25_results, index, top_k=top_k)
    else:
        # Standard retrieval with the selected method
        results = retriever.search(expanded, method=method, top_k=top_k)
    
    elapsed = (time.time() - start) * 1000
    
    formatted = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        formatted.append(make_result(doc_id, score, rank))
    
    # surface any past feedback recorded for this exact query, so the UI can
    # optionally show "you previously marked N docs as relevant"
    past_relevant, past_non_relevant = feedback_store.aggregate_for_query(user_query, method=method)
    
    return jsonify({
        "query": user_query,
        "expanded_query": expanded,
        "method": method,
        "expansion": expansion,
        "results": formatted,
        "total": len(formatted),
        "time_ms": round(elapsed, 1),
        "expansion_terms": expansion_words,
        "corrected_query": corrected_query if spelling_corrections else None,
        "spelling_corrections": spelling_corrections,
        "auto_corrected": auto_corrected,
        "past_feedback": {
            "relevant": past_relevant,
            "non_relevant": past_non_relevant
        }
    })


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Record explicit relevance feedback and return a re-ranked result set.

    The user marks individual results as relevant / not relevant in the UI.
    We log that judgement (for future queries and analysis) and immediately
    fold it into the current query via Rocchio expansion: relevant docs pull
    the query vector towards their terms, non-relevant docs push it away.
    """
    if index.N == 0:
        return jsonify({"error": "No corpus loaded"}), 400

    data = request.get_json() or {}
    query = data.get("query", "").strip()
    method = data.get("method", "bm25")
    relevant_ids = data.get("relevant", []) or []
    non_relevant_ids = data.get("non_relevant", []) or []
    top_k = int(data.get("top_k", 10))

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if not relevant_ids and not non_relevant_ids:
        return jsonify({"error": "Mark at least one result as relevant or not relevant first"}), 400

    valid_methods = ["bm25", "tfidf", "vsm", "boolean", "sbert"]
    if method not in valid_methods:
        return jsonify({"error": f"Invalid method. Use: {', '.join(valid_methods)}"}), 400

    def _normalize_ids(ids):
        normalized = []
        for d in ids:
            try:
                normalized.append(int(d))
            except (TypeError, ValueError):
                normalized.append(d)
        return normalized

    relevant_ids = _normalize_ids(relevant_ids)
    non_relevant_ids = _normalize_ids(non_relevant_ids)

    # persist the judgement so it's available for this query in the future
    feedback_store.record(query, method, relevant_ids, non_relevant_ids)

    start = time.time()

    query_tokens = preprocess(query)
    expanded_terms = expander.rocchio(
        query_tokens, relevant_ids, non_relevant_ids,
        alpha=1.0, beta=0.75, gamma=0.15, top_terms=15
    )
    expanded_query = " ".join(expanded_terms) if expanded_terms else query

    retrieval_method = "bm25" if method == "sbert" else method

    if method == "sbert":
        if bert_reranker is None:
            return jsonify({"error": "SBERT not available. Install: pip install sentence-transformers torch"}), 400
        candidate_k = max(50, top_k * 5)
        bm25_results = retriever.search(expanded_query, method="bm25", top_k=candidate_k)
        results = bert_reranker.rerank(query, bm25_results, index, top_k=top_k + len(non_relevant_ids))
    else:
        results = retriever.search(expanded_query, method=retrieval_method, top_k=top_k + len(non_relevant_ids))

    # honor the user's explicit "not relevant" judgement: never show those
    # docs back to them, even if Rocchio still ranks them highly
    non_rel_set = set(non_relevant_ids)
    results = [(doc_id, score) for doc_id, score in results if doc_id not in non_rel_set][:top_k]

    elapsed = (time.time() - start) * 1000

    formatted = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        formatted.append(make_result(doc_id, score, rank))

    original_set = set(query_tokens)
    expanded_set = set(preprocess(expanded_query))
    expansion_words = list(expanded_set - original_set)[:8]

    return jsonify({
        "query": query,
        "expanded_query": expanded_query,
        "method": method,
        "results": formatted,
        "total": len(formatted),
        "time_ms": round(elapsed, 1),
        "expansion_terms": expansion_words,
        "feedback_applied": {
            "relevant": len(relevant_ids),
            "non_relevant": len(non_relevant_ids)
        }
    })


@app.route("/api/feedback/stats")
def feedback_stats():
    return jsonify(feedback_store.stats())


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    if index.N == 0:
        return jsonify({"error": "No corpus loaded"}), 400
    
    data = request.get_json() or {}
    test_queries = data.get("test_queries", [])
    k = int(data.get("k", 10))
    
    if not test_queries:
        return jsonify({"error": "No test queries"}), 400
    
    evaluator = Evaluator()
    methods = ["bm25", "tfidf", "vsm"]
    if bert_reranker is not None:
        methods.append("sbert")
    
    scores = {}
    
    for method in methods:
        map_scores = []
        ndcg_scores = []
        prec_scores = []
        
        for item in test_queries:
            q = item.get("query", "")
            relevant = item.get("relevant", [])
            
            if not q or not relevant:
                continue
            
            if method == "sbert":
                bm25_raw = retriever.search(q, method="bm25", top_k=k*5)
                raw = bert_reranker.rerank(q, bm25_raw, index, top_k=k)
            else:
                raw = retriever.search(q, method=method, top_k=k)
            
            doc_ids = [doc_id for doc_id, _ in raw]
            
            map_scores.append(evaluator.average_precision(doc_ids, relevant))
            ndcg_scores.append(evaluator.ndcg_at_k(doc_ids, relevant, k))
            prec_scores.append(evaluator.precision_at_k(doc_ids, relevant, k))
        
        n = len(map_scores) or 1
        scores[method] = {
            "MAP": round(sum(map_scores) / n, 4),
            f"NDCG@{k}": round(sum(ndcg_scores) / n, 4),
            f"P@{k}": round(sum(prec_scores) / n, 4)
        }
    
    return jsonify({
        "methods": scores,
        "k": k,
        "queries": len(test_queries)
    })


print("Search Engine Ready")
print("http://localhost:5000")

if __name__ == "__main__":
    # PORT is read from the environment so this also works unchanged on
    # hosts like Render/Railway/Fly.io, which inject their own port.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)