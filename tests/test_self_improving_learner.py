#!/usr/bin/env python3
"""
tests/test_self_improving_learner.py -- fast checks for
emergence/self_improving_learner.py.

Stdlib only, no API calls, no files written (all memory is in-memory).
Small generations, well under a minute (the module's own selftest takes
~0.3s). Added in round 4 (2026-11-10) with the adapted module.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "emergence"))

from self_improving_learner import (  # noqa: E402
    AgentMemory,
    _sealed_eval,
    evolve,
    make_worlds,
    run_external_critique,
    selftest,
)


class SelfImprovingLearnerTests(unittest.TestCase):
    def test_module_selftest_passes(self):
        # The module's own offline selftest (15 checks, no API calls).
        self.assertTrue(selftest(out=lambda *a: None))

    def test_evolve_never_regresses_and_returns_valid_genome(self):
        mem = AgentMemory(path=None)  # in-memory: no files written
        best, rep = evolve(seed=7, generations=2, population=4,
                           mem=mem, verbose=False)
        self.assertIn(best["learner"], ("memorizer", "nearest", "rule"))
        self.assertEqual(len(rep["history"]), 2)
        hist = rep["history"]
        for i in range(len(hist) - 1):
            self.assertGreaterEqual(hist[i + 1]["fitness"],
                                    hist[i]["fitness"] - 1e-9)
        self.assertEqual(mem.stats()["records"], 4)  # 2 gen + 2 reflection

    def test_sealed_eval_scores_all_four_worlds(self):
        # A different seed than the reserved sealed seed, so this never
        # pollutes the module's sealed-seed guard.
        worlds = make_worlds(7)
        best, _rep = evolve(seed=7, generations=2, population=4,
                            mem=AgentMemory(path=None), verbose=False)
        sealed = _sealed_eval(best, worlds, 999979)
        self.assertEqual(set(sealed["per_world"]),
                         {"curve", "parity", "composition", "sequence"})
        for v in sealed["per_world"].values():
            self.assertGreaterEqual(v["gen"], 0.0)
            self.assertLessEqual(v["gen"], 1.0)

    def test_external_critique_is_a_noop_without_a_hook(self):
        mem = AgentMemory(path=None)
        self.assertIsNone(run_external_critique(mem))
        self.assertEqual(mem.stats()["records"], 0)

    def test_external_critique_uses_a_supplied_hook(self):
        mem = AgentMemory(path=None)
        best, _rep = evolve(seed=7, generations=1, population=3,
                            mem=mem, verbose=False)
        seen = []
        reply = run_external_critique(
            mem, genome=best,
            scores={"fitness": 0.5},
            critique_hook=lambda snap: (seen.append(snap), "looks fine")[1])
        self.assertEqual(reply, "looks fine")
        self.assertIn("Best genome", seen[0])
        self.assertEqual(mem.stats()["kinds"].get("critique"), 1)


if __name__ == "__main__":
    unittest.main()
