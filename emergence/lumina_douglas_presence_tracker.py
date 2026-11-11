"""
Lumina Douglas Presence Tracker
================================
Tracks whether Douglas is present and records his interactions with
Lumina, persisting the state to a JSON memory file
(`lumina_memory.json`) and a plain-text presence log
(`presence_log.txt`) in the module's own directory.

Completed by Muse 2026-11-10 (Round 5): the original was a
thinking-process transcript whose only code was a corrected __init__
plus the first lines of load_memory; the following implementation is
new code written to match the file's existing patterns and the repo's
memory-file conventions.

This is a functional presence tracker only. It makes no claims
about consciousness, sentience, or AGI.
"""

import argparse
import json
import os
import tempfile
import time
import datetime

# Maximum number of presence-history entries kept in memory.
MAX_HISTORY = 200


class LuminaPresenceTracker:
    def __init__(self, base_dir=None):
        # Determine the directory where this script resides; fallback to CWD if __file__ is unavailable or invalid
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
        except (NameError, OSError, AttributeError):
            base_path = os.getcwd()

        # Fallback to CWD if the resolved path does not exist or is not a directory
        if not os.path.isdir(base_path):
            base_path = os.getcwd()

        # An explicit base_dir (or the LUMINA_PRESENCE_DIR environment
        # variable) overrides the script directory, so the demo and
        # tests can point the tracker at a scratch directory instead.
        if base_dir is None:
            base_dir = os.environ.get("LUMINA_PRESENCE_DIR")
        if base_dir:
            base_path = base_dir

        self.base_dir = base_path

        # Ensure the base directory exists
        os.makedirs(self.base_dir, exist_ok=True)

        # Use absolute paths for memory and log files to avoid cwd issues
        self.memory_file = os.path.join(self.base_dir, "lumina_memory.json")
        self.presence_log = os.path.join(self.base_dir, "presence_log.txt")
        self.douglas_status = "unknown"
        self.last_interaction = None
        self.presence_history = []
        self.load_memory()

    def load_memory(self):
        """Restore saved state from the JSON memory file, tolerantly.

        A missing, corrupt, or oddly-shaped memory file never crashes
        the tracker: defaults are kept and a warning is printed.
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                print("Warning: could not read memory file "
                      "%s (%s); starting with defaults."
                      % (self.memory_file, e))
                return
            if not isinstance(data, dict):
                print("Warning: memory file %s is not a JSON object; "
                      "starting with defaults." % self.memory_file)
                return
            status = data.get("douglas_status")
            if isinstance(status, str) and status:
                self.douglas_status = status
            last = data.get("last_interaction")
            if isinstance(last, str) and last:
                self.last_interaction = last
            history = data.get("presence_history")
            if isinstance(history, list):
                self.presence_history = history[-MAX_HISTORY:]

    def save_memory(self):
        """Write the current state to the JSON memory file, tolerantly.

        Writes to a temporary file in the same directory and then
        replaces the memory file, so a crash mid-write does not leave
        a half-written file behind. Errors are logged, never raised.
        """
        data = {
            "douglas_status": self.douglas_status,
            "last_interaction": self.last_interaction,
            "presence_history": self.presence_history[-MAX_HISTORY:],
        }
        tmp_file = self.memory_file + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, self.memory_file)
        except OSError as e:
            print("Warning: could not save memory file %s (%s)."
                  % (self.memory_file, e))
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass

    def _now_iso(self):
        """Current time as a UTC ISO-8601 timestamp string."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _append_log_line(self, event, note=None):
        """Append one line to the plain-text presence log, tolerantly."""
        line = "[%s] %s" % (self._now_iso(), event)
        if note:
            line += ": %s" % note
        line += "\n"
        try:
            with open(self.presence_log, "a") as f:
                f.write(line)
        except OSError as e:
            print("Warning: could not write presence log %s (%s)."
                  % (self.presence_log, e))

    def _record(self, event, note=None, human_input=None, themes=None):
        """Append one entry to the in-memory history (capped)."""
        entry = {
            "timestamp": self._now_iso(),
            "event": event,
        }
        if note is not None:
            entry["note"] = note
        if human_input is not None:
            entry["human_input"] = human_input
        if themes is not None:
            entry["themes"] = list(themes)
        self.presence_history.append(entry)
        self.presence_history = self.presence_history[-MAX_HISTORY:]
        return entry

    def mark_present(self, note=None):
        """Record that Douglas is present right now."""
        self.douglas_status = "present"
        self.last_interaction = self._now_iso()
        entry = self._record("present", note=note)
        self._append_log_line("present", note)
        self.save_memory()
        return entry

    def mark_absent(self, note=None):
        """Record that Douglas is absent right now."""
        self.douglas_status = "absent"
        self.last_interaction = self._now_iso()
        entry = self._record("absent", note=note)
        self._append_log_line("absent", note)
        self.save_memory()
        return entry

    def note_douglas_interaction(self, text, themes=None):
        """Record an interaction from Douglas and mark him present.

        Shaped like the repo's experience_logger.log_experience idea
        (human_input + timestamp + themes) but without an AI response
        — this tracker only records that Douglas spoke, not what
        Lumina answered.
        """
        self.douglas_status = "present"
        self.last_interaction = self._now_iso()
        entry = self._record("interaction", human_input=text,
                             themes=themes or [])
        self._append_log_line("interaction", text)
        self.save_memory()
        return entry

    def get_status(self):
        """Return a small dict describing the tracker's current state."""
        return {
            "status": self.douglas_status,
            "last_interaction": self.last_interaction,
            "history_count": len(self.presence_history),
            "base_dir": self.base_dir,
        }

    def summary(self):
        """One-line human-readable summary, for logs."""
        status = self.get_status()
        last = status["last_interaction"] or "never"
        return ("Douglas: %(status)s (last interaction %(last)s, "
                "%(count)d history entries) — base dir %(dir)s"
                % {"status": status["status"], "last": last,
                   "count": status["history_count"],
                   "dir": status["base_dir"]})


def run_demo(base_dir=None):
    """Demo: mark presence, log an interaction, save, reload, verify.

    Runs in a temporary directory unless base_dir is given, so the
    demo never writes into the repo. Returns True on success.
    """
    if base_dir is None:
        tmp = tempfile.TemporaryDirectory()
        demo_dir = tmp.name
    else:
        tmp = None
        demo_dir = base_dir
        os.makedirs(demo_dir, exist_ok=True)

    print("demo dir: %s" % demo_dir)

    tracker = LuminaPresenceTracker(base_dir=demo_dir)
    print("initial: %s" % tracker.summary())

    tracker.mark_present(note="demo: Douglas arrived")
    tracker.note_douglas_interaction(
        "Hello Lumina, checking in.",
        themes=["greeting", "demo"])
    print("after interaction: %s" % tracker.summary())

    # Reload from disk into a fresh instance and prove persistence.
    reloaded = LuminaPresenceTracker(base_dir=demo_dir)
    print("reloaded: %s" % reloaded.summary())

    before = tracker.get_status()
    after = reloaded.get_status()
    ok = (after["status"] == before["status"]
          and after["last_interaction"] == before["last_interaction"]
          and after["history_count"] == before["history_count"]
          and after["base_dir"] == before["base_dir"])
    print("roundtrip: status=%s last_interaction=%s history_count=%d "
          "-> %s" % (after["status"], after["last_interaction"],
                     after["history_count"],
                     "MATCH" if ok else "MISMATCH"))
    print("demo: %s" % ("PASS" if ok else "FAIL"))
    if tmp is not None:
        tmp.cleanup()
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Track Douglas's presence and interactions.")
    parser.add_argument("--demo", action="store_true",
                        help="run the save/load roundtrip demo")
    parser.add_argument("--dir", default=None,
                        help="directory for memory/log files "
                             "(default: a temp dir for --demo, "
                             "the script's directory otherwise)")
    args = parser.parse_args(argv)

    # Plain `python lumina_douglas_presence_tracker.py` also runs the
    # demo (in its own directory, or --dir if given).
    demo_dir = args.dir
    if demo_dir is None and not args.demo:
        # No --dir: demo in the module's own directory.
        try:
            demo_dir = os.path.dirname(os.path.abspath(__file__))
        except (NameError, OSError, AttributeError):
            demo_dir = os.getcwd()
    ok = run_demo(base_dir=demo_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
