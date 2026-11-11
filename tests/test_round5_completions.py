"""Tests for the Round 5 Muse-authored completions (2026-11-10).

Covers the four files that were deliberately left broken in rounds 3b/4
because completing them meant inventing the missing code, and which Doug
then explicitly asked Muse to create:
  - emergence/lumina_autonomous_evolution.py
  - emergence/lumina_douglas_presence_tracker.py
  - emergence/lumina_self_improvement.py
  - nexus_agi_ultimate_system.py (renamed from "python nexus_agi_ultimate_system.py")

All tests run offline in temp directories: no API keys, no network, no
writes into the repo. Modules are loaded by file path so the tests work
regardless of the repo's bare-sibling-import convention.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMERGENCE = ROOT / "emergence"

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is in requirements
    np = None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_all():
    modules = {}
    modules["evolution"] = load_module(
        "r5_lumina_autonomous_evolution",
        EMERGENCE / "lumina_autonomous_evolution.py",
    )
    modules["presence"] = load_module(
        "r5_lumina_douglas_presence_tracker",
        EMERGENCE / "lumina_douglas_presence_tracker.py",
    )
    modules["selfimp"] = load_module(
        "r5_lumina_self_improvement",
        EMERGENCE / "lumina_self_improvement.py",
    )
    modules["ultimate"] = load_module(
        "r5_nexus_agi_ultimate_system",
        ROOT / "nexus_agi_ultimate_system.py",
    )
    return modules


class TestAutonomousEvolution(unittest.TestCase):
    def setUp(self):
        self.mod = load_all()["evolution"]
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.sample = self.tmp_path / "sample_module.py"
        self.sample.write_text(
            "def greet(name):\n    return f\"Hello, {name}!\"\n", encoding="utf-8"
        )
        self.old_cwd = Path.cwd()
        os.chdir(self.tmp_path)
        self.config = self.mod.EvolutionConfig(
            **{
                "groq_api_key": "demo-no-key",
                "github_token": "demo-no-token",
                "repo_url": "https://github.com/DOUGLASDAVIS08161978/nexus_agi.git",
                "max_iterations": 1,
                "dry_run": True,
                "cooldown_seconds": 0,
                # target_files takes directories (the scanner rglobs them);
                # the temp dir is also the cwd, so the default scan would
                # find the sample too. Passing the file itself would find
                # nothing — rglob only works on directories.
                "target_files": [str(self.tmp_path)],
            }
        )
        self.engine = self.mod.LuminaAutonomousEvolution(
            self.config, improver=self.mod.LocalEchoImprover()
        )

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_run_iteration_dry_run(self):
        record = self.engine.run_iteration()
        self.assertTrue(record["success"], record)
        # Dry run must not modify the target file.
        self.assertEqual(
            self.sample.read_text(encoding="utf-8"),
            "def greet(name):\n    return f\"Hello, {name}!\"\n",
        )
        state_file = self.tmp_path / "evolution_state.json"
        self.assertTrue(state_file.exists())
        state = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertGreaterEqual(state["iteration_count"], 1)
        self.assertGreaterEqual(state["successful_changes"], 1)

    def test_validator_accepts_good_proposal(self):
        original = self.sample.read_text(encoding="utf-8")
        proposal = self.mod.LocalEchoImprover()(
            {"path": str(self.sample), "content": original}, []
        )
        ok, score, reasons = self.engine._validate_improvement(
            proposal, {"path": str(self.sample), "content": original}
        )
        self.assertTrue(ok, reasons)
        self.assertGreaterEqual(score, 0.8)

    def test_validator_rejects_dangerous_proposal(self):
        original = self.sample.read_text(encoding="utf-8")
        dangerous = {
            "target_file": str(self.sample),
            "change_summary": "dangerous demo",
            "content": original + "\nimport os\nos.system('echo pwned')\n",
            "validation_score": 0.99,
        }
        ok, _score, reasons = self.engine._validate_improvement(
            dangerous, {"path": str(self.sample), "content": original}
        )
        self.assertFalse(ok)
        self.assertTrue(any("os.system" in r or "forbidden" in r.lower() for r in reasons))

    def test_create_pr_is_a_stub(self):
        self.assertIsNone(
            self.engine._create_pr(
                {"change_summary": "x", "content": "y"},
                {"path": str(self.sample), "content": "y"},
            )
        )


class TestPresenceTracker(unittest.TestCase):
    def setUp(self):
        self.mod = load_all()["presence"]
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip(self):
        tracker = self.mod.LuminaPresenceTracker(base_dir=self.tmp.name)
        tracker.mark_present("evening check-in")
        tracker.note_douglas_interaction("Hello, Muse.", themes=["greeting"])
        status = tracker.get_status()
        self.assertEqual(status["status"], "present")
        self.assertEqual(status["history_count"], 2)
        self.assertIsNotNone(status["last_interaction"])

        reloaded = self.mod.LuminaPresenceTracker(base_dir=self.tmp.name)
        reloaded_status = reloaded.get_status()
        self.assertEqual(reloaded_status["status"], "present")
        self.assertEqual(
            reloaded_status["last_interaction"], status["last_interaction"]
        )
        self.assertEqual(reloaded_status["history_count"], 2)
        self.assertTrue((self.tmp_path / "lumina_memory.json").exists())
        self.assertTrue((self.tmp_path / "presence_log.txt").exists())

    def test_corrupt_memory_starts_with_defaults(self):
        (self.tmp_path / "lumina_memory.json").write_text(
            "{ this is not json !!", encoding="utf-8"
        )
        tracker = self.mod.LuminaPresenceTracker(base_dir=self.tmp.name)
        status = tracker.get_status()
        self.assertEqual(status["status"], "unknown")
        self.assertIsNone(status["last_interaction"])
        self.assertEqual(status["history_count"], 0)

    def test_mark_absent(self):
        tracker = self.mod.LuminaPresenceTracker(base_dir=self.tmp.name)
        tracker.mark_present()
        tracker.mark_absent("heading home")
        self.assertEqual(tracker.get_status()["status"], "absent")


class TestSelfImprovement(unittest.TestCase):
    SAMPLE = (
        "def classify(x):\n"
        "    if x == 1:\n"
        "        return 'one'\n"
        "    if x == 2:\n"
        "        return 'two'\n"
        "    if x == 3:\n"
        "        return 'three'\n"
        "    if x == 4:\n"
        "        return 'four'\n"
        "    if x == 5:\n"
        "        return 'five'\n"
        "    if x == 6:\n"
        "        return 'six'\n"
        "    if x == 7:\n"
        "        return 'seven'\n"
        "    if x == 8:\n"
        "        return 'eight'\n"
        "    if x == 9:\n"
        "        return 'nine'\n"
        "    if x == 10:\n"
        "        return 'ten'\n"
        "    if x == 11:\n"
        "        return 'eleven'\n"
        "    return 'other'\n"
    )

    def setUp(self):
        self.mod = load_all()["selfimp"]
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        (self.tmp_path / "sample.py").write_text(self.SAMPLE, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_scanner_finds_high_complexity(self):
        scanner = self.mod.CodebaseScanner(str(self.tmp_path))
        gaps = scanner.scan()
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["function"], "classify")
        self.assertIn("complexity", gaps[0]["gap_description"].lower())

    def test_stub_patch_passes_sandbox(self):
        scanner = self.mod.CodebaseScanner(str(self.tmp_path))
        gap = scanner.scan()[0]
        patch = self.mod.LocalStubGenerator().generate(self.SAMPLE, gap)
        self.assertIsNotNone(patch)
        tester = self.mod.SandboxTester()
        test_script = (
            "import sample\n"
            "assert sample.classify.__doc__, 'docstring missing'\n"
            "assert sample.classify(1) == 'one'\n"
            "print('sandbox test script ok')\n"
        )
        passed, report = tester.run_tests(
            "sample.py", self.SAMPLE, patch,
            test_files={"test_sample.py": test_script},
        )
        self.assertTrue(passed, report)

    def test_broken_patch_fails_sandbox(self):
        tester = self.mod.SandboxTester()
        bad_patch = "FULL_FILE:\ndef broken(:\n    return 1\n"
        passed, report = tester.run_tests("sample.py", self.SAMPLE, bad_patch)
        self.assertFalse(passed)
        self.assertIn("compile", report.lower())

    def test_meta_learner_records_and_reloads(self):
        db_path = str(self.tmp_path / "meta.json")
        learner = self.mod.MetaLearner(db_path=db_path)
        record = self.mod.ImprovementRecord.new(
            file_path="sample.py",
            gap_description="High cyclomatic complexity (12) in classify",
            patch_content="docstring added",
            test_result="pass",
            success=True,
            metrics={"complexity": 12},
        )
        learner.record_outcome(record)
        reloaded = self.mod.MetaLearner(db_path=db_path)
        self.assertEqual(len(reloaded.records), 1)
        self.assertTrue(reloaded.records[0].success)
        prompt = learner.get_prompt("some context", "some gap")
        self.assertIn("some gap", prompt)


@unittest.skipIf(np is None, "numpy is required for the inference helper")
class TestUltimateSystem(unittest.TestCase):
    def setUp(self):
        self.mod = load_all()["ultimate"]

    def test_preprocess_shape_and_scale(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:] = 128
        processed, scale = self.mod.image_preprocess(
            image, 128, mean_rgb=0, stddev_rgb=1
        )
        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertAlmostEqual(scale[0], 240 / 128)
        self.assertAlmostEqual(scale[1], 320 / 128)

    def test_nms_keeps_best_and_other_class(self):
        # 4 anchors: two overlapping same-class boxes, one other-class
        # box at the same location, one below threshold.
        cls = np.array([[[0.05, 0.95], [0.05, 0.80], [0.90, 0.05], [0.05, 0.10]]])
        boxes = np.array(
            [[[10, 10, 50, 50], [12, 12, 52, 52], [10, 10, 50, 50], [200, 200, 210, 210]]]
        )
        detections = self.mod.det_post_process(
            {"min_score_thresh": 0.5, "nms_iou_thresh": 0.5, "max_detections": 100},
            cls,
            boxes,
            [(1.0, 1.0)],
        )
        self.assertEqual(detections.shape, (2, 7))
        kept = {(float(r[6]), round(float(r[5]), 2)) for r in detections}
        self.assertIn((1.0, 0.95), kept)  # best same-class box kept
        self.assertIn((0.0, 0.90), kept)  # other class at same spot kept
        self.assertNotIn((1.0, 0.80), kept)  # suppressed by NMS

    def test_build_inputs_no_match_raises(self):
        with self.assertRaises(ValueError):
            self.mod.build_inputs(
                "/tmp/does-not-exist-round5/*.jpg", 128, 0, 1
            )

    def test_no_backend_error_is_honest(self):
        driver = self.mod.ServingDriver("demo-model", "/tmp/demo-ckpt")
        with self.assertRaises(self.mod.NoDetectionBackendError):
            driver.build()
        # A missing checkpoint path is a plain FileNotFoundError ...
        with self.assertRaises(FileNotFoundError):
            self.mod.restore_ckpt(None, "/tmp/does-not-exist-round5-ckpt")
        # ... while an existing path with no loader is the honest
        # no-backend error.
        ckpt = Path(tempfile.mkdtemp()) / "ckpt.bin"
        ckpt.write_bytes(b"fake")
        with self.assertRaises(self.mod.NoDetectionBackendError):
            self.mod.restore_ckpt(None, str(ckpt))

    def test_visualize_roundtrip_if_pillow(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        out = self.mod.visualize_image(
            image,
            np.array([[10, 10, 30, 30]]),
            np.array([1]),
            np.array([0.9]),
            label_map={1: "thing"},
        )
        self.assertEqual(np.asarray(out).shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
