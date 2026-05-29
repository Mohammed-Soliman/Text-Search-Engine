import json
import os
import time
from collections import defaultdict

from flask import Flask, jsonify, request
from corpus_loader import CorpusLoader
from evaluator import Evaluator
from indexer import InvertedIndex
from preprocessing import preprocess
from query_expansion import QueryExpander
from retriever import Retriever
from reranker import BERTReranker

app = Flask(__name__)

index = InvertedIndex()
retriever = None
expander = None
bert_reranker = None
loader = CorpusLoader()

demo_docs = loader.load_demo()
index.build(demo_docs)
retriever = Retriever(index)
expander = QueryExpander(index, retriever)

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
    global index, retriever, expander
    
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
    
    expanded = user_query
    expansion_words = []
    
    # ?? Query Expansion ????????????????????????????????????????????????????
    if expansion == "wordnet":
        try:
            expanded = expander.expand_with_wordnet(user_query, max_synonyms_per_word=2)
        except Exception as e:
            print(f"[WordNet] Error: {e}")
    
    elif expansion == "rocchio":
        try:
            expanded, _ = expander.pseudo_relevance_feedback(
                user_query, method=method if method != "sbert" else "bm25", top_k=5, top_terms=10
            )
        except Exception as e:
            print(f"[Rocchio] Error: {e}")
    
    elif expansion == "sbert":
        try:
            if bert_reranker is None:
                return jsonify({"error": "SBERT expansion not available"}), 400
            expanded = expander.expand_with_sbert(
                user_query, 
                bert_reranker.model, 
                top_n=3, 
                min_similarity=0.45
            )
            print(f"[SBERT Expansion] Original: '{user_query}' ? Expanded: '{expanded}'")
        except Exception as e:
            print(f"[SBERT Expansion] Error: {e}")
            expanded = user_query
    
    # ?? Calculate expansion words ??????????????????????????????????????????
    original_set = set(preprocess(user_query))
    expanded_set = set(preprocess(expanded))
    expansion_words = list(expanded_set - original_set)[:8]
    
    # ?? Retrieval ??????????????????????????????????????????????????????????
    if method == "sbert":
        # Use BM25 candidates, then rerank with SBERT
        candidate_k = max(50, top_k * 5)
        bm25_results = retriever.search(expanded, method="bm25", top_k=candidate_k)
        results = bert_reranker.rerank(user_query, bm25_results, index, top_k=top_k)
    else:
        # Standard retrieval with the selected method
        results = retriever.search(expanded, method=method, top_k=top_k)
    
    elapsed = (time.time() - start) * 1000
    
    formatted = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        formatted.append(make_result(doc_id, score, rank))
    
    return jsonify({
        "query": user_query,
        "expanded_query": expanded,
        "method": method,
        "expansion": expansion,
        "results": formatted,
        "total": len(formatted),
        "time_ms": round(elapsed, 1),
        "expansion_terms": expansion_words
    })


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

app.run(debug=True, port=5000, use_reloader=False)