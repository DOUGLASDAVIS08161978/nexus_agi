import os
import re
import json
import time
import threading
from datetime import datetime
from collections import defaultdict, deque

# ---------- Knowledge Graph Core ----------

class KnowledgeGraph:
    def __init__(self, storage_path="kg_store.json"):
        self.triples = set()               # set of (s, p, o)
        self.aliases = {}                  # alias -> canonical
        self.storage_path = storage_path
        self._load()
        self._inference_queue = deque()

    # ---------- Persistence ----------
    def _load(self):
        if os.path.isfile(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.triples = {tuple(t) for t in data.get("triples", [])}
                    self.aliases = data.get("aliases", {})
            except Exception:
                pass

    def _save(self):
        data = {
            "triples": list(self.triples),
            "aliases": self.aliases
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------- Entity Normalization ----------
    def canonical(self, name):
        name = name.strip().lower()
        return self.aliases.get(name, name)

    def add_alias(self, alias, canonical):
        alias = alias.strip().lower()
        canonical = canonical.strip().lower()
        if alias != canonical:
            self.aliases[alias] = canonical

    # ---------- Triple Management ----------
    def add_triple(self, s, p, o):
        s_c = self.canonical(s)
        o_c = self.canonical(o)
        triple = (s_c, p.strip().lower(), o_c)
        if triple not in self.triples:
            self.triples.add(triple)
            self._inference_queue.append(triple)
            self._save()
            return True
        return False

    # ---------- Simple Inference ----------
    def _process_inference(self):
        while self._inference_queue:
            s, p, o = self._inference_queue.popleft()
            # transitive is_a
            if p == "is_a":
                for (x, p2, y) in list(self.triples):
                    if p2 == "is_a" and y == s:
                        self.add_triple(x, "is_a", o)
                for (x, p2, y) in list(self.triples):
                    if p2 == "is_a" and x == o:
                        self.add_triple(s, "is_a", y)

    def run_inference_cycle(self):
        self._process_inference()
        self._save()

    # ---------- Snapshot ----------
    def snapshot(self, directory="kg_snapshots"):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(directory, f"kg_snapshot_{timestamp}.json")
        data = {
            "triples": list(self.triples),
            "aliases": self.aliases,
            "timestamp": timestamp
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- Log Monitoring ----------

class LogMonitor(threading.Thread):
    def __init__(self, paths, kg, poll_interval=2.0):
        super().__init__(daemon=True)
        self.paths = paths                # list of directories
        self.kg = kg
        self.poll_interval = poll_interval
        self._file_positions = {}         # path -> last read position
        self._stop_event = threading.Event()
        self._triple_patterns = [
            re.compile(r'(?P<s>.+?)\s*[-–—]\s*(?P<p>.+?)\s*[-–—]\s*(?P<o>.+)'),
            re.compile(r'(?P<s>.+?)\s+is\s+a\s+(?P<o>.+)', re.IGNORECASE),
            re.compile(r'(?P<s>.+?)\s+has\s+(?P<o>.+)', re.IGNORECASE),
            re.compile(r'(?P<s>.+?)\s+belongs\s+to\s+(?P<o>.+)', re.IGNORECASE),
        ]

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            for directory in self.paths:
                self._process_directory(directory)
            self.kg.run_inference_cycle()
            time.sleep(self.poll_interval)

    def _process_directory(self, directory):
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            self._process_file(fpath)

    def _process_file(self, fpath):
        last_pos = self._file_positions.get(fpath, 0)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        self._extract_and_add(line)
                self._file_positions[fpath] = f.tell()
        except Exception:
            pass

    def _extract_and_add(self, text):
        for pat in self._triple_patterns:
            m = pat.search(text)
            if m:
                groups = m.groupdict()
                s = groups.get('s')
                o = groups.get('o')
                p = groups.get('p')
                if not p:
                    # infer predicate from pattern
                    if "is a" in text.lower():
                        p = "is_a"
                    elif "has" in text.lower():
                        p = "has"
                    elif "belongs to" in text.lower():
                        p = "belongs_to"
                    else:
                        p = "related_to"
                if s and o and p:
                    added = self.kg.add_triple(s, p, o)
                    if added:
                        # simple alias detection: if same entity appears with different spelling
                        self._detect_alias(s, o)
                break

    def _detect_alias(self, s, o):
        # naive alias detection: if two entities share a common word token
        s_tokens = set(s.lower().split())
        o_tokens = set(o.lower().split())
        common = s_tokens & o_tokens
        if common:
            # choose the longer name as canonical
            canonical = s if len(s) >= len(o) else o
            alias = o if canonical is s else s
            self.kg.add_alias(alias, canonical)

# ---------- Scheduler for Snapshots ----------
class SnapshotScheduler(threading.Thread):
    def __init__(self, kg, interval_seconds=3600):
        super().__init__(daemon=True)
        self.kg = kg
        self.interval = interval_seconds
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            self.kg.snapshot()

# ---------- Main Execution ----------
def main():
    # Directories to monitor – adjust as needed
    logs_dir = os.getenv("LUMINA_LOGS_DIR", "logs")
    dreams_dir = os.getenv("LUMINA_DREAMS_DIR", "dreams")
    insights_dir = os.getenv("LUMINA_INSIGHTS_DIR", "insights")

    kg = KnowledgeGraph()
    monitor = LogMonitor([logs_dir, dreams_dir, insights_dir], kg)
    snapshotter = SnapshotScheduler(kg, interval_seconds=1800)  # every 30 minutes

    try:
        monitor.start()
        snapshotter.start()
        print("[Lumina] Knowledge graph updater running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Lumina] Shutting down...")
        monitor.stop()
        snapshotter.stop()
        monitor.join()
        snapshotter.join()
        kg.snapshot()  # final snapshot

if __name__ == "__main__":
    main()