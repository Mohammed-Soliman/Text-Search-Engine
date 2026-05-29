import math

class Evaluator:

    def precision_at_k(self, retrieved, relevant, k):
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = 0
        for doc_id in top_k:
            if doc_id in relevant_set:
                hits = hits + 1
        return hits / k if k > 0 else 0.0

    def recall_at_k(self, retrieved, relevant, k):
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = 0
        for doc_id in top_k:
            if doc_id in relevant_set:
                hits = hits + 1
        return hits / len(relevant_set) if relevant_set else 0.0

    def f1_at_k(self, retrieved, relevant, k):
        p = self.precision_at_k(retrieved, relevant, k)
        r = self.recall_at_k(retrieved, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def average_precision(self, retrieved, relevant):
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        
        hits = 0
        sum_precision = 0.0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                hits = hits + 1
                sum_precision = sum_precision + (hits / (i + 1))
        
        return sum_precision / len(relevant_set)

    def mean_average_precision(self, queries_results):
        if not queries_results:
            return 0.0
        ap_scores = []
        for retrieved, relevant in queries_results:
            ap_scores.append(self.average_precision(retrieved, relevant))
        return sum(ap_scores) / len(ap_scores)

    def ndcg_at_k(self, retrieved, relevant, k):
        relevant_set = set(relevant)
        dcg = 0.0
        idcg = 0.0
        
        for i in range(k):
            if i < len(retrieved) and retrieved[i] in relevant_set:
                dcg = dcg + (1 / math.log2(i + 2))
            
            if i < len(relevant_set):
                idcg = idcg + (1 / math.log2(i + 2))
        
        return dcg / idcg if idcg > 0 else 0.0