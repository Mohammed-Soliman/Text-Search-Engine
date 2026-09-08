import json
import os
import time


class FeedbackStore:

    def __init__(self, path="data/feedback.json"):
        self.path = path
        self.events = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.events = json.load(f)
            except Exception as e:
                print(f"[FeedbackStore] Could not load {self.path}: {e}")
                self.events = []

    def _save(self):
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                # cap the log so it doesn't grow unbounded on a long-running demo
                json.dump(self.events[-5000:], f, indent=2)
        except Exception as e:
            print(f"[FeedbackStore] Could not save {self.path}: {e}")

    def record(self, query, method, relevant_ids, non_relevant_ids):
        event = {
            "query": query,
            "normalized_query": query.strip().lower(),
            "method": method,
            "relevant": list(relevant_ids),
            "non_relevant": list(non_relevant_ids),
            "timestamp": time.time(),
        }
        self.events.append(event)
        self._save()
        return event

    def history_for_query(self, query, method=None, limit=50):
        normalized = query.strip().lower()
        matches = [
            e for e in self.events
            if e["normalized_query"] == normalized and (method is None or e["method"] == method)
        ]
        return matches[-limit:]

    def aggregate_for_query(self, query, method=None):
        matches = self.history_for_query(query, method=method)
        relevant = set()
        non_relevant = set()
        for e in matches:
            relevant.update(e["relevant"])
            non_relevant.update(e["non_relevant"])
        non_relevant -= relevant
        return list(relevant), list(non_relevant)

    def stats(self):
        return {
            "total_events": len(self.events),
            "unique_queries": len({e["normalized_query"] for e in self.events}),
        }
