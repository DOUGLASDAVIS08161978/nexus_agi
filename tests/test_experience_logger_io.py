#!/usr/bin/env python3
"""
tests/test_experience_logger_io.py — round-trip test for ExperienceLogger.

Proposed by Groq (openai/gpt-oss-120b) in the first three-minds review
(2026-11-10). Groq guessed the method name `log_event` from the snapshot;
the real API is `log_experience(session_id, human_input, ai_response,
themes=None)` — corrected here.

All file I/O is isolated to a TemporaryDirectory; the test never touches
the real production log directory or any external service. Stdlib only.
"""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "emergence"))

from experience_logger import ExperienceLogger


class TestExperienceLoggerIO(unittest.TestCase):
    def test_log_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = ExperienceLogger(memory_dir=tmp)
            exp_id = logger.log_experience(
                session_id="test-session",
                human_input="Hello, Lumina.",
                ai_response="Hello, Doug.",
                themes=["greeting"],
            )
            self.assertIsInstance(exp_id, str)
            with open(os.path.join(tmp, "experiences.json")) as f:
                stored = json.load(f)
            self.assertEqual(len(stored), 1)
            entry = stored[0]
            self.assertEqual(entry["id"], exp_id)
            self.assertEqual(entry["session_id"], "test-session")
            self.assertEqual(entry["human_input"], "Hello, Lumina.")
            self.assertEqual(entry["ai_response"], "Hello, Doug.")
            self.assertEqual(entry["themes"], ["greeting"])
            self.assertIsNone(entry["reflection"])

    def test_get_recent_returns_logged_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = ExperienceLogger(memory_dir=tmp)
            logger.log_experience("s1", "one", "uno")
            logger.log_experience("s1", "two", "dos")
            recent = logger.get_recent(1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["human_input"], "two")
            self.assertEqual(logger.count(), 2)


if __name__ == "__main__":
    unittest.main()
