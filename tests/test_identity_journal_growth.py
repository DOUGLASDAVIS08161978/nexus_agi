#!/usr/bin/env python3
"""
tests/test_identity_journal_growth.py -- pins the honest growth metric
in IdentityJournal.analyze_growth / analyze_sentiment (round 4).

The old analyze_sentiment trained a LogisticRegression to predict
(compound > 0) from the compound score itself -- the target is a
deterministic function of the feature, so the reported "accuracy" was
meaningless (and sklearn rejects the 1-D feature list). The new version
reports the trend directly with no model. This test pins that: no
"accuracy" key, trend values are the compound scores themselves, and
no sklearn import is attempted.

The module imports groq and experience_logger at the top and nltk
lazily inside analyze_growth; none of those are installed in this
sandbox, so they are stubbed in sys.modules before import. Stdlib only.
"""

import os
import sys
import types
import unittest

# --- stub external deps before importing the module under test ---
_groq = types.ModuleType("groq")


class _Groq:
    def __init__(self, *a, **k):
        pass


_groq.Groq = _Groq
sys.modules.setdefault("groq", _groq)

_exp = types.ModuleType("experience_logger")


class _ExperienceLogger:
    def __init__(self, *a, **k):
        pass


_exp.ExperienceLogger = _ExperienceLogger
sys.modules.setdefault("experience_logger", _exp)

_nltk = types.ModuleType("nltk")
_sentiment = types.ModuleType("nltk.sentiment")


class _SIA:
    def polarity_scores(self, text):
        # Deterministic toy sentiment: +0.5 if "good" in the text.
        return {"compound": 0.5 if "good" in text else -0.5}


_sentiment.SentimentIntensityAnalyzer = _SIA
_nltk.sentiment = _sentiment
sys.modules.setdefault("nltk", _nltk)
sys.modules.setdefault("nltk.sentiment", _sentiment)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "emergence"))

from identity_journal import IdentityJournal  # noqa: E402


def _journal_with(entries):
    # Bypass __init__ (it would create a Groq client and a memory dir).
    journal = object.__new__(IdentityJournal)
    journal.get_all_entries = lambda: entries
    return journal


class GrowthMetricTests(unittest.TestCase):
    def test_no_accuracy_claim_and_trend_is_the_scores(self):
        journal = _journal_with([
            {"id": 1, "entry": "a good day"},
            {"id": 2, "entry": "a bad day"},
            {"id": 3, "entry": "a good day indeed"},
            {"id": 4, "entry": "a good day again"},
        ])
        result = journal.analyze_growth()
        self.assertNotIn("accuracy", result)
        self.assertEqual(result["sentiment_trend"], [0.5, -0.5, 0.5, 0.5])
        self.assertAlmostEqual(result["mean_compound"], 0.25)
        self.assertAlmostEqual(result["fraction_positive"], 0.75)
        self.assertAlmostEqual(result["trend_slope"], 0.1)
        self.assertAlmostEqual(result["first_half_mean"], 0.0)
        self.assertAlmostEqual(result["second_half_mean"], 0.5)
        # No sklearn import was attempted anywhere in the path.
        self.assertNotIn("sklearn", sys.modules)

    def test_analyze_sentiment_empty_input(self):
        journal = object.__new__(IdentityJournal)
        result = journal.analyze_sentiment([])
        self.assertEqual(result["sentiment_trend"], [])
        self.assertEqual(result["mean_compound"], 0.0)
        self.assertEqual(result["trend_slope"], 0.0)

    def test_analyze_sentiment_known_slope(self):
        journal = object.__new__(IdentityJournal)
        scores = [{"compound": c} for c in (0.0, 1.0, 2.0, 3.0)]
        result = journal.analyze_sentiment(scores)
        self.assertEqual(result["sentiment_trend"], [0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["mean_compound"], 1.5)
        self.assertAlmostEqual(result["fraction_positive"], 0.75)
        self.assertAlmostEqual(result["trend_slope"], 1.0)
        self.assertAlmostEqual(result["first_half_mean"], 0.5)
        self.assertAlmostEqual(result["second_half_mean"], 2.5)

    def test_result_is_json_safe(self):
        import json
        journal = _journal_with([{"id": 1, "entry": "a good day"}])
        json.dumps(journal.analyze_growth())  # must not raise


if __name__ == "__main__":
    unittest.main()
