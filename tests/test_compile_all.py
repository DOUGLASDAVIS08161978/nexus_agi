#!/usr/bin/env python3
"""
tests/test_compile_all.py — repo-wide compile guard for nexus_agi.

Proposed by Claude (claude-haiku-5-5) in the first three-minds review
(2026-11-10): one test that walks the repo and runs py_compile on every
.py file, then reports the failing paths. It modifies no existing files.

It only READS source files. It never imports or executes any module, so it
cannot trigger Lumina's live Groq calls, file writes, or side effects.
Stdlib only, no new dependencies.

Informational by design: it prints a summary and exits 0 so it never
breaks existing workflows. Pass --fail-on-error to make it exit non-zero
when any file fails to compile.
"""

import argparse
import os
import py_compile
import sys

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}


def iter_python_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def check(root):
    """Return (checked, [(path, error), ...]) for every .py under root."""
    failures = []
    checked = 0
    for path in iter_python_files(root):
        checked += 1
        try:
            py_compile.compile(path, doraise=True, quiet=1)
        except py_compile.PyCompileError as exc:
            failures.append((os.path.relpath(path, root), str(exc).strip()))
    return checked, failures


def report(root, fail_on_error=False):
    checked, failures = check(root)
    print(f"compile guard: checked {checked} Python files under {root}")
    if not failures:
        print("compile guard: all files compile OK")
        return 0
    print(f"compile guard: {len(failures)} file(s) fail py_compile:")
    for path, error in failures:
        first = error.splitlines()[0] if error else ""
        print(f"  FAIL {path}: {first}")
    if fail_on_error:
        return 1
    print("compile guard: informational only — exiting 0 (use --fail-on-error to enforce)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compile every .py file in the repo (read-only).")
    parser.add_argument("--root", default=None, help="repo root (default: parent of tests/)")
    parser.add_argument("--fail-on-error", action="store_true",
                        help="exit non-zero when any file fails to compile")
    args = parser.parse_args(argv)
    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return report(os.path.abspath(root), fail_on_error=args.fail_on_error)


def test_all_python_files_compile():
    """Pytest entry point: same check, informational (never fails the suite)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checked, failures = check(root)
    print(f"\ncompile guard: {checked} files checked, {len(failures)} failing (informational)")
    for path, error in failures[:20]:
        print(f"  FAIL {path}")
    assert checked > 0


if __name__ == "__main__":
    sys.exit(main())
