#!/usr/bin/env python3
"""
lumina_memory_arch.py — Multi-tier memory architecture for Lumina

Four memory tiers, modeled on biological memory systems:

  WorkingMemory    — active context, current session, Python dict (~50 items)
  EpisodicMemory   — specific events and moments, C-accelerated vector search
  SemanticMemory   — facts, concepts, knowledge (existing system, unchanged)
  ProceduralMemory — how-to knowledge, strategies, learned algorithms

Plus a MemoryConsolidator that runs between sessions:
  • Strengthens important memories (high salience, frequently recalled)
  • Fades trivial memories (old, never recalled)
  • Abstracts repeated patterns from episodes → semantic entries
  • This is the analogue of sleep/dreaming in biological systems

The C accelerator (lumina_vecdot.so) is loaded via ctypes if compiled.
If not compiled, all operations fall back to pure Python transparently.

Persists to:
  emergence/episodic_memory.json    — episodic store
  emergence/working_memory.json     — end-of-session working memory snapshot

Build the C accelerator first:
  cd emergence && bash build_memory.sh
"""

from __future__ import annotations
import json, time, math, hashlib, threading, ctypes
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient

EPISODIC_FILE = Path(__file__).parent / "episodic_memory.json"
WORKING_FILE  = Path(__file__).parent / "working_memory.json"
VECDOT_SO     = Path(__file__).parent / "lumina_vecdot.so"

VEC_DIM       = 256     # feature-hash dimension
WORKING_MAX   = 60      # max working memory entries
EPISODIC_MAX  = 5000    # max episodic entries before consolidation trims
SALIENCE_FADE = 0.02    # salience decay per consolidation cycle
MIN_SALIENCE  = 0.10    # entries below this get pruned on consolidation


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _ts() -> float:
    return time.time()


# ── C accelerator (optional) ──────────────────────────────────────────────────

class _VecDotLib:
    """Thin ctypes wrapper around lumina_vecdot.so."""

    def __init__(self):
        self._lib = None
        self._available = False
        self._load()

    def _load(self):
        if not VECDOT_SO.exists():
            return
        try:
            lib = ctypes.CDLL(str(VECDOT_SO))

            # batch_cosine_similarity(query, vectors, norms, n, dim, scores)
            lib.batch_cosine_similarity.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # query
                ctypes.POINTER(ctypes.c_float),  # vectors
                ctypes.POINTER(ctypes.c_float),  # norms
                ctypes.c_int,                    # n
                ctypes.c_int,                    # dim
                ctypes.POINTER(ctypes.c_float),  # scores (out)
            ]
            lib.batch_cosine_similarity.restype = None

            # top_k_indices(scores, n, k, indices)
            lib.top_k_indices.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # scores
                ctypes.c_int,                    # n
                ctypes.c_int,                    # k
                ctypes.POINTER(ctypes.c_int),    # indices (out)
            ]
            lib.top_k_indices.restype = None

            # l2_norm(v, n)
            lib.l2_norm.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.l2_norm.restype = ctypes.c_float

            self._lib = lib
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def batch_search(
        self,
        query_vec: List[float],
        stored_vecs: List[List[float]],
        stored_norms: List[float],
        top_k: int,
    ) -> List[int]:
        """Return indices of top-k matches (C-accelerated)."""
        n   = len(stored_vecs)
        dim = VEC_DIM

        FloatArr = ctypes.c_float * (n * dim)
        NormArr  = ctypes.c_float * n
        ScoreArr = ctypes.c_float * n
        IdxArr   = ctypes.c_int   * top_k

        q_arr  = (ctypes.c_float * dim)(*query_vec)
        v_flat = [x for row in stored_vecs for x in row]
        v_arr  = FloatArr(*v_flat)
        n_arr  = NormArr(*stored_norms)
        s_arr  = ScoreArr(*([0.0] * n))
        i_arr  = IdxArr(*([0] * top_k))

        self._lib.batch_cosine_similarity(q_arr, v_arr, n_arr, n, dim, s_arr)
        self._lib.top_k_indices(s_arr, n, min(top_k, n), i_arr)
        return list(i_arr[:min(top_k, n)])


_vecdot = _VecDotLib()


# ── Vector encoding (hashing trick) ──────────────────────────────────────────

def _encode(text: str) -> Tuple[List[float], float]:
    """
    Encode text as a VEC_DIM-dimensional float vector using the hashing trick.
    Returns (vector, l2_norm). The vector is NOT normalized — norm is separate
    so the C code can use pre-stored norms for fast cosine similarity.
    """
    vec = [0.0] * VEC_DIM
    words = text.lower().split()
    for word in words:
        if len(word) < 2:
            continue
        h1 = int(hashlib.md5(word.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(word.encode()).hexdigest(), 16)
        pos  = h1 % VEC_DIM
        sign = 1.0 if h2 % 2 == 0 else -1.0
        vec[pos] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    return vec, norm


def _cosine(vec_a: List[float], norm_a: float,
            vec_b: List[float], norm_b: float) -> float:
    """Pure-Python cosine similarity fallback."""
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)


# ── Memory entry ──────────────────────────────────────────────────────────────

class MemoryEntry:
    __slots__ = ("text", "tags", "category", "ts", "ts_epoch",
                 "salience", "recall_count", "vec", "norm", "session_id")

    def __init__(self, text: str, tags: List[str] = None,
                 category: str = "general", salience: float = 0.5,
                 session_id: str = ""):
        self.text         = text[:500]
        self.tags         = tags or []
        self.category     = category
        self.ts           = _now()
        self.ts_epoch     = _ts()
        self.salience     = salience
        self.recall_count = 0
        self.session_id   = session_id
        self.vec, self.norm = _encode(text)

    def to_dict(self) -> Dict:
        return {
            "text":         self.text,
            "tags":         self.tags,
            "category":     self.category,
            "ts":           self.ts,
            "ts_epoch":     self.ts_epoch,
            "salience":     round(self.salience, 4),
            "recall_count": self.recall_count,
            "session_id":   self.session_id,
            # Store vec as list of floats (JSON-serialisable)
            "vec":          [round(x, 5) for x in self.vec],
            "norm":         round(self.norm, 6),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MemoryEntry":
        e = cls.__new__(cls)
        e.text         = d.get("text", "")
        e.tags         = d.get("tags", [])
        e.category     = d.get("category", "general")
        e.ts           = d.get("ts", _now())
        e.ts_epoch     = d.get("ts_epoch", 0.0)
        e.salience     = float(d.get("salience", 0.5))
        e.recall_count = int(d.get("recall_count", 0))
        e.session_id   = d.get("session_id", "")
        e.vec          = d.get("vec") or [0.0] * VEC_DIM
        e.norm         = float(d.get("norm", 0.0))
        if not e.vec or len(e.vec) != VEC_DIM:
            e.vec, e.norm = _encode(e.text)
        return e

    def age_days(self) -> float:
        return (_ts() - self.ts_epoch) / 86400.0


# ── Working memory ─────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Fast in-memory store for the current session.
    Holds the most recent N items — what Lumina is actively thinking about.
    Wiped on session end (but snapshot saved to disk for reflection).
    """

    def __init__(self, session_id: str = ""):
        self._entries:    List[MemoryEntry] = []
        self._session_id  = session_id
        self._lock        = threading.Lock()

    def store(self, text: str, tags: List[str] = None,
              category: str = "general", salience: float = 0.5):
        with self._lock:
            e = MemoryEntry(text, tags, category, salience, self._session_id)
            self._entries.append(e)
            if len(self._entries) > WORKING_MAX:
                # Drop lowest-salience entry when full
                self._entries.sort(key=lambda x: x.salience, reverse=True)
                self._entries = self._entries[:WORKING_MAX]

    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        with self._lock:
            if not self._entries:
                return []
            q_vec, q_norm = _encode(query)
            scored = [
                (_cosine(q_vec, q_norm, e.vec, e.norm), e)
                for e in self._entries
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, e in scored[:top_k]:
                if score > 0.01:
                    e.recall_count += 1
                    results.append({
                        "text": e.text, "tags": e.tags,
                        "category": e.category, "ts": e.ts,
                        "score": round(score, 4), "tier": "working",
                    })
            return results

    def snapshot(self) -> List[Dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def clear(self):
        with self._lock:
            self._entries.clear()

    def __len__(self):
        with self._lock:
            return len(self._entries)


# ── Episodic memory ────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Long-term store of specific events and moments.
    C-accelerated search via lumina_vecdot.so when compiled.
    Falls back to pure Python cosine similarity otherwise.

    Each entry has a salience score that increases with recall and
    decreases over time — important memories persist, trivial ones fade.
    """

    def __init__(self, session_id: str = ""):
        self._entries:   List[MemoryEntry] = []
        self._session_id = session_id
        self._lock       = threading.Lock()
        self._dirty      = False
        self._load()

    def _load(self):
        if not EPISODIC_FILE.exists():
            return
        try:
            data = json.loads(EPISODIC_FILE.read_text("utf-8"))
            self._entries = [MemoryEntry.from_dict(d) for d in data]
        except Exception:
            pass

    def save(self):
        if not self._dirty:
            return
        try:
            EPISODIC_FILE.write_text(
                json.dumps([e.to_dict() for e in self._entries[-EPISODIC_MAX:]],
                           indent=2),
                "utf-8",
            )
            self._dirty = False
        except Exception:
            pass

    def store(self, text: str, tags: List[str] = None,
              category: str = "episodic", salience: float = 0.5):
        with self._lock:
            e = MemoryEntry(text, tags or [], category, salience, self._session_id)
            self._entries.append(e)
            self._dirty = True

    def recall(self, query: str, top_k: int = 5,
               category: str = None) -> List[Dict]:
        with self._lock:
            pool = [e for e in self._entries
                    if not category or e.category == category]
            if not pool:
                return []

            q_vec, q_norm = _encode(query)
            if q_norm < 1e-10:
                return []

            # Recency + salience boosted scores
            now = _ts()
            results: List[Dict] = []

            if _vecdot.available and len(pool) > 20:
                # C path: batch similarity
                vecs   = [e.vec  for e in pool]
                norms  = [e.norm for e in pool]
                indices = _vecdot.batch_search(q_vec, vecs, norms, top_k * 3)
                candidates = [(pool[i], i) for i in indices]
            else:
                # Python path
                scored = [
                    (_cosine(q_vec, q_norm, e.vec, e.norm), i, e)
                    for i, e in enumerate(pool)
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                candidates = [(e, i) for _, i, e in scored[:top_k * 3]]

            for e, _ in candidates:
                base_score = _cosine(q_vec, q_norm, e.vec, e.norm)
                if base_score < 0.01:
                    continue
                age_days = e.age_days()
                recency  = max(0.0, 1.0 - age_days / 30.0) * 0.20
                salience_boost = e.salience * 0.15
                final = base_score * (1.0 + recency + salience_boost)

                e.recall_count += 1
                e.salience = min(1.0, e.salience + 0.02)
                self._dirty = True

                results.append({
                    "text":     e.text,
                    "tags":     e.tags,
                    "category": e.category,
                    "ts":       e.ts,
                    "score":    round(final, 4),
                    "salience": round(e.salience, 3),
                    "tier":     "episodic",
                    "c_accel":  _vecdot.available,
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def stats(self) -> Dict:
        with self._lock:
            return {
                "entries":    len(self._entries),
                "c_accel":    _vecdot.available,
                "avg_sal":    round(
                    sum(e.salience for e in self._entries) / max(1, len(self._entries)), 3
                ),
                "top_recalled": sorted(
                    [(e.text[:60], e.recall_count) for e in self._entries],
                    key=lambda x: x[1], reverse=True,
                )[:3],
            }


# ── Procedural memory (thin wrapper, delegates to meta_solver) ─────────────────

class ProceduralMemory:
    """
    Wraps the MetaSolver's strategy and algorithm libraries.
    Gives the memory architecture a unified interface to procedural knowledge.
    """

    def __init__(self, meta_solver=None):
        self._solver = meta_solver

    def recall(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self._solver:
            return []
        low = query.lower()
        results = []
        for strat in getattr(self._solver, "_strategies", []):
            words = set(strat.context.lower().split()[:10])
            if any(w in low for w in words):
                results.append({
                    "text":     f"[Strategy:{strat.domain}] {strat.pattern}",
                    "category": "procedural",
                    "tier":     "procedural",
                    "score":    0.6,
                })
        for algo in getattr(self._solver, "_algorithms", []):
            if algo.domain in low or any(
                w in low for w in algo.problem_type.lower().split()
            ):
                steps = " → ".join(algo.steps[:3])
                results.append({
                    "text":     f"[Algorithm:{algo.domain}] {algo.name}: {steps}",
                    "category": "procedural",
                    "tier":     "procedural",
                    "score":    0.7,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ── Memory consolidator ────────────────────────────────────────────────────────

class MemoryConsolidator:
    """
    Runs between sessions (or every N turns) to:
      1. Fade low-salience episodic memories
      2. Prune entries below MIN_SALIENCE
      3. Strengthen frequently-recalled memories
      4. Abstract patterns from episodes → semantic entries (via Groq)

    This is the analogue of sleep consolidation in biological memory.
    """

    def __init__(self, episodic: EpisodicMemory,
                 semantic=None, groq=None):
        self._episodic = episodic
        self._semantic = semantic
        self._groq     = groq

    def consolidate(self, verbose: bool = True) -> Dict:
        """Run a consolidation cycle. Returns stats dict."""
        if verbose:
            print("  💭 Memory consolidation running...")

        with self._episodic._lock:
            entries = self._episodic._entries

            before = len(entries)
            strengthened = 0
            faded        = 0
            pruned_list  = []

            for e in entries:
                if e.recall_count >= 3:
                    # Frequently recalled → strengthen
                    e.salience = min(1.0, e.salience + 0.05)
                    strengthened += 1
                else:
                    # Fade by age
                    age_factor = min(1.0, e.age_days() / 7.0)
                    fade = SALIENCE_FADE * age_factor
                    e.salience = max(0.0, e.salience - fade)
                    if e.salience < MIN_SALIENCE:
                        pruned_list.append(e)
                    else:
                        faded += 1

            # Prune
            pruned_set = set(id(e) for e in pruned_list)
            self._episodic._entries = [
                e for e in entries if id(e) not in pruned_set
            ]
            self._episodic._dirty = True

        pruned = len(pruned_list)

        # Abstract patterns via Groq (fire-and-forget quality)
        abstracted = 0
        if self._groq and self._semantic and pruned_list:
            abstracted = self._abstract_patterns(pruned_list)

        self._episodic.save()

        stats = {
            "before": before,
            "after":  before - pruned,
            "strengthened": strengthened,
            "faded":  faded,
            "pruned": pruned,
            "abstracted": abstracted,
        }
        if verbose:
            print(f"  💭 Consolidation: {strengthened} strengthened, "
                  f"{faded} faded, {pruned} pruned → {abstracted} abstracted")
        return stats

    def _abstract_patterns(self, fading: List[MemoryEntry]) -> int:
        """
        Find patterns in fading memories and save the insight to semantic memory.
        """
        if not fading:
            return 0
        sample = "\n".join(f"- {e.text[:80]}" for e in fading[:12])
        result = self._groq.chat(
            "You are abstracting patterns from fading memories into durable knowledge. "
            "Find what is common or important across these items and write ONE concise "
            "insight (1-2 sentences) worth remembering permanently. "
            "If there is no meaningful pattern, say 'No pattern found.'",
            f"Fading memories:\n{sample}",
            tier="fast", max_tokens=80,
        )
        if result and "No pattern found" not in result and not result.startswith("[Groq"):
            self._semantic.store(
                f"[ABSTRACTED] {result.strip()}",
                tags=["consolidation", "abstraction"],
                category="semantic",
            )
            return 1
        return 0


# ── Unified memory architecture ────────────────────────────────────────────────

class LuminaMemoryArchitecture:
    """
    Unified interface to all four memory tiers.

    Replaces direct calls to SemanticMemory.store/recall in the engine —
    the engine calls this instead and all tiers are managed automatically.

    Working memory fills each session.
    Episodic stores meaningful exchanges and events.
    Semantic stores knowledge and facts.
    Procedural delegates to meta_solver.

    Recall merges results from all tiers, ranked by relevance.
    """

    def __init__(self, session_id: str, semantic=None,
                 meta_solver=None, groq=None):
        self.working    = WorkingMemory(session_id)
        self.episodic   = EpisodicMemory(session_id)
        self.procedural = ProceduralMemory(meta_solver)
        self._semantic  = semantic
        self._groq      = groq
        self._session_id = session_id
        self._consolidator = MemoryConsolidator(
            self.episodic, semantic, groq
        )

    def store(self, text: str, tags: List[str] = None,
              category: str = "general",
              salience: float = 0.5,
              working: bool = True,
              episodic: bool = True):
        """Store to working memory, episodic, and semantic."""
        tags = tags or []
        if working:
            self.working.store(text, tags, category, salience)
        if episodic:
            self.episodic.store(text, tags, category, salience)
        if self._semantic:
            self._semantic.store(text, tags=tags, category=category)

    def recall(self, query: str, top_k: int = 6,
               category: str = None) -> List[Dict]:
        """
        Recall from all tiers, merge and deduplicate by content similarity.
        """
        seen_texts: set = set()
        merged: List[Dict] = []

        # Working memory (highest recency weight)
        for r in self.working.recall(query, top_k=top_k):
            key = r["text"][:80]
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(r)

        # Episodic (C-accelerated)
        for r in self.episodic.recall(query, top_k=top_k, category=category):
            key = r["text"][:80]
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(r)

        # Semantic (existing TF-IDF system)
        if self._semantic:
            try:
                sem = self._semantic.recall(query, top_k=top_k,
                                            category=category)
                for r in sem:
                    key = r.get("text", "")[:80]
                    if key not in seen_texts:
                        seen_texts.add(key)
                        r["tier"] = "semantic"
                        merged.append(r)
            except Exception:
                pass

        # Procedural (strategies/algorithms)
        for r in self.procedural.recall(query, top_k=3):
            key = r["text"][:80]
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(r)

        # Sort by score
        merged.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        return merged[:top_k]

    def on_session_end(self, verbose: bool = True):
        """
        Called when Lumina's session ends.
        Saves working memory snapshot, runs consolidation.
        """
        try:
            snap = self.working.snapshot()
            if snap:
                WORKING_FILE.write_text(json.dumps(snap, indent=2), "utf-8")
        except Exception:
            pass

        self.episodic.save()
        self._consolidator.consolidate(verbose=verbose)
        self.working.clear()

    def force_consolidate(self) -> Dict:
        return self._consolidator.consolidate(verbose=True)

    def display(self) -> str:
        w_len = len(self.working)
        e_stats = self.episodic.stats()
        accel = "C-accelerated (NEON)" if _vecdot.available else "Python (run build_memory.sh to enable C)"
        lines = [
            f"  Memory architecture:",
            f"    Working memory  : {w_len} items (current session)",
            f"    Episodic memory : {e_stats['entries']} entries"
            f"  [avg salience {e_stats['avg_sal']}]",
            f"    Vector search   : {accel}",
            f"    Vector dim      : {VEC_DIM}",
        ]
        if e_stats["top_recalled"]:
            lines.append("    Most recalled:")
            for text, count in e_stats["top_recalled"]:
                lines.append(f"      ({count}×) {text}")
        return "\n".join(lines)

    def stats(self) -> Dict:
        return {
            "working":  len(self.working),
            "episodic": self.episodic.stats(),
            "c_accel":  _vecdot.available,
            "vec_dim":  VEC_DIM,
        }
