"""
feedback_store.py

Persists explicit user relevance feedback ("this doc is relevant" / "this doc
is not relevant") so that:

1. It can immediately be used to re-rank the *current* search via Rocchio
   query expansion (see QueryExpander.rocchio in query_expansion.py), and
2. It accumulates over time as a lightweight feedback log that could later
   be used for evaluation, analytics, or to auto-suggest previously-marked
   documents the next time the same query is issued.

Feedback is stored as a flat, append-only JSON list on disk so it survives
server restarts. This is intentionally simple (no database) to match the
rest of the project's "basic" scope.
"""

import json
import os
import time


class FeedbackStore:

    def __init__(self, path="data/feedback.json"):
        self.path = path
        self.events = []
        self._load()

    # -- persistence ----------------------------------------------------------

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

    # -- writing ----------------------------------------------------------------

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

    # -- reading ------------------------------------------------------------------

    def history_for_query(self, query, method=None, limit=50):
        normalized = query.strip().lower()
        matches = [
            e for e in self.events
            if e["normalized_query"] == normalized and (method is None or e["method"] == method)
        ]
        return matches[-limit:]

    def aggregate_for_query(self, query, method=None):
        """Union of all relevant/non-relevant doc ids ever marked for this query.

        Useful for pre-filling feedback the next time someone runs a query
        that's already been judged before. Relevant wins over non-relevant
        if a doc was marked both ways at different times.
        """
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
