"""
spell_corrector.py
------------------
[BONUS] Query Spell Correction

TODO: Implement SpellCorrector class using pyspellchecker

SpellCorrector Class:
  - __init__(vocab): Initialize with optional corpus vocabulary list.
    Try to import pyspellchecker.SpellChecker. If successful, teach it the vocab.
    Set _available flag based on import success.
  
  - correct(query): Correct misspelled words in query string.
    For each word: if in vocab keep it, else get correction from spellchecker.
    Return (corrected_query, list of (original, corrected) changes).
  
  - suggest(word): Return list of alternative spellings for a word.
"""

from __future__ import annotations
