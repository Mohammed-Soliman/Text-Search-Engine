"""
spell_correction.py

Lightweight "did you mean?" query correction, in the spirit of Peter Norvig's
classic spelling corrector (https://norvig.com/spell-correct.html), adapted to
correct search queries against the search engine's own vocabulary rather than
a generic English dictionary.

Why correct against the index vocabulary instead of a dictionary?
- A word can be spelled "correctly" in English but still be a typo for this
  corpus (e.g. "footbal" vs "football").
- Correcting to a real dictionary word that isn't in the corpus wouldn't help
  the user - it would still return zero results.
- Correcting to a term that *is* in the index guarantees the suggestion is
  actually searchable.

The corrector only ever proposes replacements for words that are Out-Of-
Vocabulary (OOV). Words already present in the index vocabulary are left
untouched, even if a "more frequent" neighbour exists (we don't want to
"fix" a query that is already valid).
"""

import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z']+")
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class SpellCorrector:
    """Suggests corrections for query words that don't appear in the index."""

    def __init__(self, index=None):
        self.vocab_freq = Counter()
        self.total = 1
        if index is not None:
            self.rebuild(index)

    def rebuild(self, index):
        """(Re)build the frequency table from an InvertedIndex.

        Called once at startup and again whenever a new corpus is uploaded,
        so corrections always reflect the currently loaded index.
        """
        self.vocab_freq = Counter()
        for term, entry in index.index.items():
            # document frequency is a reasonable proxy for "how common/likely"
            # a term is within this specific corpus
            self.vocab_freq[term] = entry.get("df", 1)
        self.total = sum(self.vocab_freq.values()) or 1

    # -- Norvig-style edit-distance candidate generation --------------------

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
        """Best-effort candidate set: exact match > 1 edit > 2 edits > give up."""
        return (
            self._known([word])
            or self._known(self._edits1(word))
            or self._known(self._edits2(word))
            or {word}
        )

    # -- Public API -----------------------------------------------------------

    def correct_word(self, word):
        """Return the best in-vocabulary correction for a single word.

        If the word is already known (in the vocabulary) or no correction
        candidate can be found, the original word is returned unchanged.
        """
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
        """Correct every OOV word in a query string.

        Returns:
            (corrected_query, corrections) where `corrections` is a list of
            {"original": ..., "corrected": ...} dicts, one per word that was
            actually changed.
        """
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
        """True if the query contains at least one word absent from the vocab."""
        tokens = _WORD_RE.findall(query.lower())
        return any(t not in self.vocab_freq for t in tokens if len(t) >= 3)
