import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z']+")
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class SpellCorrector:

    def __init__(self, index=None):
        self.vocab_freq = Counter()
        self.total = 1
        if index is not None:
            self.rebuild(index)

    def rebuild(self, index):
        self.vocab_freq = Counter()
        for term, entry in index.index.items():
            self.vocab_freq[term] = entry.get("df", 1)
        self.total = sum(self.vocab_freq.values()) or 1


    def _edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in _ALPHABET]
        inserts = [L + c + R for L, R in splits for c in _ALPHABET]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}

    def _known(self, words):
        return {w for w in words if w in self.vocab_freq}

    def _candidates(self, word):
        return (
            self._known([word])
            or self._known(self._edits1(word))
            or self._known(self._edits2(word))
            or {word}
        )

    def correct_word(self, word):
        if not word:
            return word

        lower = word.lower()
        if lower in self.vocab_freq or len(lower) < 3:
            return word

        candidates = self._candidates(lower)
        if candidates == {lower}:
            return word  # nothing better found

        best = max(candidates, key=lambda w: self.vocab_freq.get(w, 0))
        return best

    def correct_query(self, query):
        tokens = _WORD_RE.findall(query)
        if not tokens:
            return query, []

        corrected_tokens = []
        corrections = []

        for token in tokens:
            fixed = self.correct_word(token)
            corrected_tokens.append(fixed)
            if fixed.lower() != token.lower():
                corrections.append({"original": token, "corrected": fixed})

        corrected_query = " ".join(corrected_tokens)
        return corrected_query, corrections

    def has_unknown_words(self, query):
        tokens = _WORD_RE.findall(query.lower())
        return any(t not in self.vocab_freq for t in tokens if len(t) >= 3)
