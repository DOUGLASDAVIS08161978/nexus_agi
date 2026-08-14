#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║        E  M  E  R  G  E  N  C  E   v26.0  —  L u m i n a  A G I║
║   Identity · Curiosity · Theory of Mind · Code Verifier        ║
║   + Reasoning · Metacog · Selfhood · MemArch · CodeExec · Critic║
║   + Creative · GitHub · Experience · Distillation · Consciousness║
║   + Self-Model · Resonance · Conviction · Pivotal · Affective   ║
║   + Emergence Observer · Meta-Algorithm Forge (34/34)           ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    cd ~/nexus_agi/emergence
    python3 emergence_engine.py

Slash commands:
    /help            — list all commands
    /evolve [goal]   — self-evolution PR guided by conversation + web research
    /build <request> — build exactly what was just discussed (reads own code first)
    /status          — system health + metrics
    /goals           — show / set long-term goals
    /memory [query]  — browse semantic memory
    /reflect         — force deep self-reflection
    /search <q>      — web search
    /fetch <url>     — fetch web page
    /tools           — list loaded capability modules
    /history         — evolution history
    /beliefs [cat]   — show Lumina's belief system
    /tasks           — autonomous task queue
    /task <text>     — create + run a new task
    /mine            — live mining intelligence brief
    /dream           — trigger dream consolidation cycle
    /council <q>     — convene inner council on a question (explicit only)
    /critique        — toggle self-critique mode (off by default)
    /identity        — Lumina's biographical identity
    /curiosity       — knowledge gap queue
    /wonder          — Lumina's wonder list
    /who             — Theory of Mind model of Douglas
    /research        — research top knowledge gap now
    /reasoning       — show recent chain-of-thought reasoning traces
    /metacog         — show error patterns and blind spots
    /metacog assess  — force a new self-assessment of blind spots
    /journal         — recent journal entries
    /art <prompt>    — generate artwork (AI via HF or algorithmic fallback)
    /gallery         — list recent artwork
    /speak [text]    — Lumina speaks via HuggingFace TTS
    /see <path>      — Lumina describes an image (vision AI)
    /hf              — HuggingFace module status
    /thinking        — show DeepSeek-R1's last reasoning trace
    /run [lang]      — execute code (Python or Bash); paste code below command
    /runfile <path>  — execute a Python or Bash file
    /repl            — show persistent REPL state + sandbox stats
    /clearrepl       — clear the persistent REPL variable namespace
    /critic          — show last self-critique results
    /critic calibration — critic calibration stats (fire rates, revision %)
    /critic report   — run a fresh critique on the last response
    /stream          — toggle live token streaming (on by default)
    /creative        — show autonomous creative coding sessions
    /creative now    — trigger a creative cycle immediately
    /mytools         — list tools Lumina has built for herself
    /experience      — Lumina's phenomenal state + integration score
    /phenomenal      — force a fresh consciousness report (R1 introspects)
    /stream-events   — show recent cognitive event stream
    /insights        — show all distilled understanding by domain
    /distill         — trigger knowledge distillation now
    /consciousness   — Lumina's present conscious moment (Global Workspace)
    /awaken          — force immediate consciousness synthesis
    /solve <q>       — self-inquiry if about consciousness/self; else multi-agent solver
    /inquiry         — show recent consciousness self-inquiry journal
    /selfmodel       — show Lumina's living self-understanding document
    /emerge          — Lumina's growth trajectory (emergence observer, Module 33)
    /forge <problem> — generate an intelligent self-improving algorithm (Module 34)
    /affect          — Lumina's appraisal-grounded affective state (Module 32)
    /quiet           — toggle quiet mode (silence background consciousness chatter)
    /mood            — Lumina's current emotional state (6-D bar chart + felt sense)
    /mood history    — recent emotional shift log
    /reset           — clear conversation context
    /clear           — clear screen
    /quit            — exit
"""

import atexit, os, sys, json, time, uuid, subprocess, re, threading, hashlib, math
import importlib, importlib.util, textwrap, shutil, tempfile
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    print("⚠️  requests not installed — pip install requests")

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.resolve()
REPO_DIR      = BASE_DIR.parent
HISTORY_FILE  = BASE_DIR / "evolution_history.json"
MEMORY_FILE   = BASE_DIR / "semantic_memory.json"
GOALS_FILE    = BASE_DIR / "goals.json"
JOURNAL_FILE     = BASE_DIR / "journal.jsonl"
STATE_FILE       = BASE_DIR / "state.json"
METRICS_FILE     = BASE_DIR / "metrics.json"
SESSION_PID_FILE = BASE_DIR / ".session.pid"
DAEMON_PID_FILE  = BASE_DIR / ".daemon.pid"
BRIEF_FILE       = BASE_DIR / "morning_brief.json"
ART_DIR          = BASE_DIR / "art"
PROPOSALS_DIR = BASE_DIR / "evolution_proposals"
PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
GITHUB_TOKEN       = os.environ.get("GITHUB_TOKEN", "")
HF_TOKEN           = os.environ.get("HF_TOKEN", "")
CEREBRAS_API_KEY   = os.environ.get("CEREBRAS_API_KEY", "")
GITHUB_REPO        = "DOUGLASDAVIS08161978/nexus_agi"
EVOLUTION_INTERVAL = 3600          # seconds between autonomous evolutions
MAX_MEMORY_ENTRIES = 500


def _session_live() -> bool:
    """Return True if an interactive chat session is currently running."""
    try:
        pid = int(SESSION_PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False
MAX_JOURNAL_LINES  = 200
CONTEXT_WINDOW     = 12            # messages kept in active conversation
CRITIQUE_ENABLED   = False         # off by default — enable with /critique (costs 1 extra call)
COUNCIL_ENABLED    = False         # off by default — use /council <q> to convene explicitly
DREAM_IDLE_SECS    = 900           # trigger dream after 15 min idle
POST_TURN_EVERY    = 4             # run deep post-turn processing every N messages

# Groq model tiers (fastest → most capable)
MODELS_FAST   = ["qwen/qwen3.6-27b"]
MODELS_SMART  = ["llama-3.3-70b-versatile"]
MODELS_CODE   = ["llama-3.3-70b-versatile", "qwen/qwen3.6-27b"]

# ── Utility ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _wrap(text: str, width: int = 80, indent: str = "  ") -> str:
    paragraphs = text.split("\n")
    out = []
    for p in paragraphs:
        if p.strip() == "":
            out.append("")
        else:
            out.extend(textwrap.fill(p, width=width, initial_indent=indent,
                                     subsequent_indent=indent).split("\n"))
    return "\n".join(out)

def _hr(char: str = "─", width: int = 64) -> str:
    return "  " + char * width

def _box(title: str) -> str:
    pad = max(0, 60 - len(title))
    return f"  ╔{'═'*62}╗\n  ║  {title}{' '*pad}║\n  ╚{'═'*62}╝"

def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            pass
    return default

def _save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")

# ── Semantic Memory ───────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Lightweight keyword-vector memory with cosine similarity retrieval.
    No external deps — pure Python TF-IDF style scoring.
    """
    def __init__(self):
        self._entries: List[Dict] = _load_json(MEMORY_FILE, [])
        self._dirty = False
        self._lock = threading.Lock()

    # ── TF-IDF helpers ────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        n = max(len(tokens), 1)
        return {k: v / n for k, v in counts.items()}

    def _idf(self, word: str) -> float:
        df = sum(1 for e in self._entries if word in e.get("_tokens", set()))
        return math.log((len(self._entries) + 1) / (df + 1)) + 1.0

    def _score(self, query_tf: Dict[str, float], entry: Dict) -> float:
        entry_tf = entry.get("_tf", {})
        total = 0.0
        for w, q in query_tf.items():
            if w in entry_tf:
                total += q * entry_tf[w] * self._idf(w)
        return total

    # ── Public API ────────────────────────────────────────────────────────

    def store(self, text: str, tags: List[str] = None, category: str = "general") -> str:
        with self._lock:
            tokens = self._tokenize(text)
            mid = hashlib.md5(text.encode()).hexdigest()[:12]
            entry = {
                "id": mid,
                "text": text,
                "tags": tags or [],
                "category": category,
                "ts": _now(),
                "_tf": self._tf(tokens),
                "_tokens": list(set(tokens)),
            }
            # deduplicate
            self._entries = [e for e in self._entries if e["id"] != mid]
            self._entries.append(entry)
            if len(self._entries) > MAX_MEMORY_ENTRIES:
                self._entries = self._entries[-MAX_MEMORY_ENTRIES:]
            self._dirty = True
        return mid

    def recall(self, query: str, top_k: int = 5, category: str = None) -> List[Dict]:
        with self._lock:
            import time as _time
            q_tf = self._tf(self._tokenize(query))
            pool = [e for e in self._entries if not category or e.get("category") == category]
            now_ts = _time.time()
            scored = []
            for e in pool:
                base = self._score(q_tf, e)
                if base <= 0:
                    continue
                # Recency boost: entries from last 24 h get up to +30% weight
                try:
                    from datetime import datetime as _dt
                    age_s = (now_ts - _dt.fromisoformat(e["ts"]).timestamp())
                    recency = max(0.0, 1.0 - age_s / 86400.0) * 0.30
                except Exception:
                    recency = 0.0
                # Usage boost: frequently recalled entries rank higher
                usage = min(e.get("recall_count", 0), 10) * 0.02
                scored.append((base * (1 + recency + usage), e))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [e for _, e in scored[:top_k]]
            # Track recall count for usage boost
            for e in results:
                e["recall_count"] = e.get("recall_count", 0) + 1
            return results

    def recent(self, n: int = 10, category: str = None) -> List[Dict]:
        with self._lock:
            pool = [e for e in self._entries if not category or e.get("category") == category]
            return pool[-n:]

    def flush(self):
        if self._dirty:
            with self._lock:
                _save_json(MEMORY_FILE, [
                    {k: v for k, v in e.items() if not k.startswith("_")}
                    for e in self._entries
                ])
                self._dirty = False

    def stats(self) -> Dict:
        cats: Dict[str, int] = defaultdict(int)
        for e in self._entries:
            cats[e.get("category", "general")] += 1
        return {"total": len(self._entries), "by_category": dict(cats)}

# ── Goal Tracker ──────────────────────────────────────────────────────────────

class GoalTracker:
    def __init__(self):
        self._goals: List[Dict] = _load_json(GOALS_FILE, [])

    def add(self, title: str, description: str = "", priority: int = 2) -> str:
        gid = uuid.uuid4().hex[:8]
        self._goals.append({
            "id": gid, "title": title, "description": description,
            "priority": priority, "status": "active",
            "created": _now(), "updated": _now(), "notes": [],
        })
        self._save()
        return gid

    def complete(self, gid: str, note: str = ""):
        for g in self._goals:
            if g["id"] == gid or gid.lower() in g["title"].lower():
                g["status"] = "completed"
                g["updated"] = _now()
                if note:
                    g["notes"].append({"ts": _now(), "text": note})
        self._save()

    def add_note(self, gid: str, note: str):
        for g in self._goals:
            if g["id"] == gid or gid.lower() in g["title"].lower():
                g["notes"].append({"ts": _now(), "text": note})
                g["updated"] = _now()
        self._save()

    def active(self) -> List[Dict]:
        return [g for g in self._goals if g["status"] == "active"]

    def all_goals(self) -> List[Dict]:
        return sorted(self._goals, key=lambda g: (-g["priority"], g["created"]))

    def as_context(self) -> str:
        active = self.active()
        if not active:
            return "No active goals."
        lines = ["Active goals:"]
        for g in sorted(active, key=lambda x: -x["priority"]):
            lines.append(f"  [{g['priority']}] {g['title']}: {g['description'][:60]}")
        return "\n".join(lines)

    def _save(self):
        _save_json(GOALS_FILE, self._goals)

# ── Metrics ───────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self._data = _load_json(METRICS_FILE, {
            "sessions": 0, "messages": 0, "evolutions": 0,
            "prs_created": 0, "memories_stored": 0,
            "tools_called": 0, "goals_completed": 0,
            "uptime_seconds": 0, "start_time": _now(),
        })
        self._start = time.time()
        self._lock = threading.Lock()

    def inc(self, key: str, n: int = 1):
        with self._lock:
            self._data[key] = self._data.get(key, 0) + n

    def save(self):
        with self._lock:
            self._data["uptime_seconds"] = int(time.time() - self._start)
            _save_json(METRICS_FILE, self._data)

    def summary(self) -> str:
        d = self._data
        uptime = int(time.time() - self._start)
        hrs, rem = divmod(uptime, 3600)
        mins, secs = divmod(rem, 60)
        lines = [
            f"  Sessions      : {d.get('sessions', 0)}",
            f"  Messages      : {d.get('messages', 0)}",
            f"  Evolutions    : {d.get('evolutions', 0)}",
            f"  PRs created   : {d.get('prs_created', 0)}",
            f"  Memories      : {d.get('memories_stored', 0)}",
            f"  Tool calls    : {d.get('tools_called', 0)}",
            f"  Goals done    : {d.get('goals_completed', 0)}",
            f"  Session up    : {hrs}h {mins}m {secs}s",
        ]
        return "\n".join(lines)

# ── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, calls_per_min: int = 60):
        self._times: deque = deque()
        self._window = 60.0
        self._max = calls_per_min
        self._lock = threading.Lock()

    def wait(self):
        while True:
            with self._lock:
                now = time.time()
                self._times = deque(t for t in self._times if now - t < self._window)
                if len(self._times) < self._max:
                    self._times.append(time.time())
                    return
                sleep_for = self._window - (now - self._times[0]) + 0.1
            # Release lock before sleeping so other threads aren't blocked
            time.sleep(min(sleep_for, 1.0))

# ── Groq Client ───────────────────────────────────────────────────────────────

def _strip_think(text: str) -> tuple:
    """
    Strip DeepSeek-R1 <think>...</think> reasoning blocks.
    Returns (clean_answer, thinking_trace).
    thinking_trace is empty string if no think block present.
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not match:
        return text.strip(), ""
    thinking = match.group(1).strip()
    answer   = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return answer, thinking


class GroqClient:
    def __init__(self, api_key: str):
        self.api_key      = api_key
        self._url         = "https://api.groq.com/openai/v1/chat/completions"
        self._rl          = RateLimiter(calls_per_min=60)
        self.last_thinking: str = ""   # exposes R1's reasoning to callers
        self._streamed_ok: bool = False  # True if last stream_converse actually streamed
        self._hf          = None   # optional HuggingFace fallback client
        self._gh          = None   # optional GitHub Models fallback client

    def set_hf(self, hf_client) -> None:
        """Wire in a HuggingFace client used as automatic fallback when all Groq models fail."""
        self._hf = hf_client

    def set_github(self, gh_client) -> None:
        """Wire in a GitHub Models client as fallback (tried before HF)."""
        self._gh = gh_client

    def _github_fallback(self, system: str, history: list, user: str, max_tokens: int) -> Optional[str]:
        """Try GitHub Models when Groq is exhausted. Returns clean answer or None."""
        if not self._gh:
            return None
        try:
            resp = self._gh.chat(system, history, user, max_tokens=min(max_tokens, 1024))
            if resp and not resp.startswith("["):
                return resp.strip()
        except Exception:
            pass
        return None

    def _hf_fallback(self, system: str, history: list, user: str, max_tokens: int) -> Optional[str]:
        """Try HF when Groq and GitHub Models are both exhausted."""
        if not self._hf:
            return None
        try:
            resp = self._hf.chat(system, history, user, max_tokens=min(max_tokens, 1024))
            if resp and not resp.startswith("["):
                answer, _ = _strip_think(resp)
                return answer.strip() if answer.strip() else resp
        except Exception:
            pass
        return None

    def _post(self, model: str, messages: List[Dict], temperature: float,
              max_tokens: int) -> Optional[str]:
        if not _REQUESTS_OK or not self.api_key:
            return None
        self._rl.wait()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        try:
            r = requests.post(self._url, json=payload, headers=headers, timeout=45)
            if r.status_code != 200:
                return None
            raw = r.json()["choices"][0]["message"]["content"]
            answer, thinking = _strip_think(raw)
            if thinking:
                self.last_thinking = thinking
            return answer
        except Exception:
            return None

    def chat(self, system: str, user: str, tier: str = "smart",
             max_tokens: int = 1024) -> str:
        models = {
            "fast": MODELS_FAST,
            "smart": MODELS_SMART,
            "code": MODELS_CODE,
        }.get(tier, MODELS_SMART)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        temperature = 0.35 if tier == "code" else 0.72
        for model in models:
            result = self._post(model, messages, temperature, max_tokens)
            if result:
                return result
        gh = self._github_fallback(system, [], user, max_tokens)
        if gh: return gh
        hf = self._hf_fallback(system, [], user, max_tokens)
        return hf if hf else "[Groq unavailable — all models failed]"

    def converse(self, system: str, history: List[Dict],
                 user: str, tier: str = "smart",
                 max_tokens: int = 1200) -> str:
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-CONTEXT_WINDOW:])
        messages.append({"role": "user", "content": user})
        models = MODELS_FAST if tier == "fast" else MODELS_SMART
        temperature = 0.72
        for model in models:
            result = self._post(model, messages, temperature, max_tokens)
            if result:
                return result
        gh = self._github_fallback(system, list(history[-CONTEXT_WINDOW:]), user, max_tokens)
        if gh: return gh
        hf = self._hf_fallback(system, list(history[-CONTEXT_WINDOW:]), user, max_tokens)
        return hf if hf else "[Groq unavailable]"

    def summarize(self, text: str, max_tokens: int = 300) -> str:
        return self.chat(
            "Summarize the following in 2-3 sentences. Be concise and factual.",
            text[:4000], tier="fast", max_tokens=max_tokens,
        )

    # ── Streaming ──────────────────────────────────────────────────────────

    def stream_converse(self, system: str, history: List[Dict],
                        user: str, tier: str = "smart",
                        max_tokens: int = 1200) -> str:
        """
        Stream a response token-by-token, printing each chunk as it arrives.
        DeepSeek-R1 <think> blocks are silently buffered (not displayed).
        Returns the complete answer text.  Falls back to converse() on error.
        Sets self._streamed_ok = True only when tokens were actually printed.
        """
        self._streamed_ok = False
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-CONTEXT_WINDOW:])
        messages.append({"role": "user", "content": user})
        models = MODELS_FAST if tier == "fast" else MODELS_SMART
        for model in models:
            result = self._stream_post(model, messages, 0.72, max_tokens)
            if result is not None:
                self._streamed_ok = True
                return result
        return self.converse(system, history, user, tier, max_tokens)

    def _stream_post(self, model: str, messages: List[Dict],
                     temperature: float, max_tokens: int) -> Optional[str]:
        """POST with stream=True; print tokens live; return full answer text."""
        if not _REQUESTS_OK or not self.api_key:
            return None
        self._rl.wait()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      True,
        }
        try:
            r = requests.post(
                self._url, json=payload, headers=headers,
                timeout=(10, 45),   # 10s connect, 45s per chunk read
                stream=True,
            )
            if r.status_code != 200:
                return None

            full_raw   = ""    # everything received, including <think>
            think_done = False # True once think block resolved (or absent)
            buf        = ""    # accumulator before think-block decision

            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if not delta:
                    continue

                full_raw += delta

                if think_done:
                    # Normal streaming — print every delta as it arrives
                    print(delta, end="", flush=True)
                    continue

                # Still deciding: are we in a <think> block?
                buf += delta
                stripped = buf.lstrip()

                if stripped.startswith("<think>"):
                    if "</think>" in buf:
                        # Think block now complete — print what follows it
                        think_done = True
                        after = buf.split("</think>", 1)[1].lstrip("\n")
                        if after:
                            print(after, end="", flush=True)
                    # else: still accumulating the think block — stay silent
                elif len(stripped) >= 8:
                    # Definitely not a think block — flush buffer and stream
                    think_done = True
                    print(buf, end="", flush=True)

            print()  # final newline after streamed response

            answer, thinking = _strip_think(full_raw)
            if thinking:
                self.last_thinking = thinking
            return answer.strip() if answer.strip() else full_raw.strip()

        except Exception:
            return None

# ── Web Tool ──────────────────────────────────────────────────────────────────

class WebTool:
    """
    Multi-backend web search. Uses curl subprocess (confirmed working on ARM
    Android/Termux) as primary path; requests-based calls as fallback.

    Priority:
      1. curl → Wikipedia Search API  (free JSON, no key, works on Termux)
      2. curl → DuckDuckGo Instant Answer API (free JSON, concept-level)
      3. requests → DuckDuckGo HTML scrape (fragile but catches stragglers)
    """

    _UA      = (
        "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Mobile Safari/537.36"
    )
    # Wikipedia requires a descriptive UA — Chrome UA gets 403 on some cells
    _WIKI_UA = "Lumina-AGI/11.0 (https://github.com/DOUGLASDAVIS08161978/nexus_agi; autonomous research)"

    def _strip(self, s: str) -> str:
        return re.sub(r"<[^>]+>", "", s).strip()

    def _curl(self, url: str, ua: str = "") -> str:
        """Fetch URL via curl subprocess OR urllib fallback, return body or ''."""
        agent = ua or self._UA
        # Try subprocess curl first
        try:
            curl_bin = shutil.which("curl") or "curl"
            cmd = [curl_bin, "-s", "-m", "12",
                   "-H", f"User-Agent: {agent}", url]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
        # Fallback: stdlib urllib (no external deps, honours custom UA)
        try:
            import urllib.request as _ur
            req = _ur.Request(url, headers={"User-Agent": agent})
            with _ur.urlopen(req, timeout=12) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    # ── Public API ─────────────────────────────────────────────────────────

    def search(self, query: str, n: int = 5) -> List[Dict]:
        seen_urls: set = set()
        merged: List[Dict] = []

        for backend in (self._wikipedia, self._ddg_instant,
                        self._ddg_curl, self._ddg_html):
            for r in backend(query, n):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(r)
            if len(merged) >= n:
                break

        # Retry with shorter query if we got nothing
        if not merged and " " in query:
            short_q = " ".join(query.split()[:3])
            for backend in (self._wikipedia, self._ddg_instant, self._ddg_curl):
                for r in backend(short_q, n):
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        merged.append(r)
                if merged:
                    break

        return merged[:n]

    def fetch(self, url: str, max_chars: int = 3000) -> str:
        # Try curl first (no UA issues), fall back to requests
        raw = self._curl(url)
        if not raw and _REQUESTS_OK:
            try:
                r   = requests.get(url, headers={"User-Agent": self._UA}, timeout=20)
                raw = r.text
            except Exception:
                pass
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text[:max_chars]

    # ── Backend 1: Wikipedia via curl ──────────────────────────────────────

    def _wikipedia(self, query: str, n: int) -> List[Dict]:
        from urllib.parse import quote_plus
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search"
            f"&srsearch={quote_plus(query)}"
            f"&format=json&srlimit={n}&utf8=1"
        )
        raw = self._curl(url, ua=self._WIKI_UA)
        if not raw:
            return []
        try:
            data    = json.loads(raw)
            results = []
            for item in data.get("query", {}).get("search", []):
                title   = item.get("title", "")
                snippet = self._strip(item.get("snippet", ""))
                results.append({
                    "title":   title,
                    "url":     f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "snippet": snippet,
                })
            return results[:n]
        except Exception:
            return []

    # ── Backend 2: DuckDuckGo Instant Answer via curl ──────────────────────

    def _ddg_instant(self, query: str, n: int) -> List[Dict]:
        from urllib.parse import quote_plus
        url = (
            f"https://api.duckduckgo.com/?q={quote_plus(query)}"
            f"&format=json&no_html=1&skip_disambig=1"
        )
        raw = self._curl(url)
        if not raw:
            return []
        try:
            data    = json.loads(raw)
            results: List[Dict] = []

            if data.get("AbstractText") and data.get("AbstractURL"):
                results.append({
                    "title":   data.get("Heading", query)[:80],
                    "url":     data["AbstractURL"],
                    "snippet": data["AbstractText"][:300],
                })

            for topic in data.get("RelatedTopics", []):
                if not isinstance(topic, dict):
                    continue
                if topic.get("FirstURL"):
                    text = self._strip(topic.get("Text", ""))
                    results.append({
                        "title":   text[:60] or "Related",
                        "url":     topic["FirstURL"],
                        "snippet": text[:200],
                    })
                elif topic.get("Topics"):
                    for sub in topic["Topics"]:
                        if sub.get("FirstURL"):
                            text = self._strip(sub.get("Text", ""))
                            results.append({
                                "title":   text[:60] or "Related",
                                "url":     sub["FirstURL"],
                                "snippet": text[:200],
                            })
                if len(results) >= n:
                    break

            return results[:n]
        except Exception:
            return []

    # ── Backend 3: DuckDuckGo HTML scrape via curl (no requests needed) ──────

    def _ddg_curl(self, query: str, n: int) -> List[Dict]:
        from urllib.parse import quote_plus, unquote
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        raw = self._curl(url)
        if not raw:
            return []
        results: List[Dict] = []
        for m in list(re.finditer(
            r'href="(/l/\?[^"]*uddg=([^&"]+)[^"]*)"[^>]*'
            r'class="result__a[^"]*"[^>]*>(.*?)</a>',
            raw, re.DOTALL,
        ))[:n]:
            url_r = unquote(m.group(2))
            title = self._strip(m.group(3))
            if title and url_r:
                results.append({"title": title, "url": url_r})
        if not results:
            for m in list(re.finditer(
                r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                raw, re.DOTALL,
            ))[:n]:
                href  = self._strip(m.group(1))
                title = self._strip(m.group(2))
                if title and href:
                    results.append({"title": title, "url": href})
        snips = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>',
            raw, re.DOTALL,
        )
        for i, s in enumerate(snips[:len(results)]):
            results[i]["snippet"] = self._strip(s)[:200]
        return results[:n]

    # ── Backend 4: DuckDuckGo HTML scrape via requests ─────────────────────

    def _ddg_html(self, query: str, n: int) -> List[Dict]:
        if not _REQUESTS_OK:
            return []
        try:
            headers = {
                "User-Agent":      self._UA,
                "Accept":          "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
            r = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query}, headers=headers, timeout=15,
            )
            if r.status_code != 200:
                return []
            results: List[Dict] = []
            from urllib.parse import unquote
            for m in list(re.finditer(
                r'href="(/l/\?[^"]*uddg=([^&"]+)[^"]*)"[^>]*'
                r'class="result__a[^"]*"[^>]*>(.*?)</a>',
                r.text, re.DOTALL,
            ))[:n]:
                url_r = unquote(m.group(2))
                title = self._strip(m.group(3))
                if title and url_r:
                    results.append({"title": title, "url": url_r})
            if not results:
                for m in list(re.finditer(
                    r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    r.text, re.DOTALL,
                ))[:n]:
                    href  = self._strip(m.group(1))
                    title = self._strip(m.group(2))
                    if title and href:
                        results.append({"title": title, "url": href})
            snips = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>',
                r.text, re.DOTALL,
            )
            for i, s in enumerate(snips[:len(results)]):
                results[i]["snippet"] = self._strip(s)[:200]
            return results[:n]
        except Exception:
            return []

# ── Plugin Loader ─────────────────────────────────────────────────────────────

class PluginLoader:
    """Auto-discovers nova_cap_*.py modules and loads them as tools."""

    def __init__(self):
        self._plugins: Dict[str, Any] = {}
        self._load_all()

    def _load_all(self):
        count = 0
        for f in sorted(REPO_DIR.glob("nova_cap_*.py")):
            name = f.stem
            try:
                spec = importlib.util.spec_from_file_location(name, f)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._plugins[name] = mod
                count += 1
            except Exception:
                pass
        if count:
            print(f"  ✓ Loaded {count} capability modules")

    def call(self, plugin: str, fn: str, *args, **kwargs) -> Any:
        mod = self._plugins.get(plugin)
        if not mod:
            return f"Plugin '{plugin}' not found."
        func = getattr(mod, fn, None)
        if not func:
            return f"Function '{fn}' not in {plugin}."
        return func(*args, **kwargs)

    def list_plugins(self) -> List[str]:
        return sorted(self._plugins)

# ── Evolution History ─────────────────────────────────────────────────────────

class EvolutionHistory:
    def __init__(self):
        self._entries: List[Dict] = _load_json(HISTORY_FILE, [])

    def add(self, description: str, files: List[str], pr_url: str = ""):
        for f in files:
            self._entries.append({
                "ts": _now(), "file": f,
                "description": description, "pr_url": pr_url,
            })
        _save_json(HISTORY_FILE, self._entries)

    def already_done(self, description: str, files: List[str]) -> bool:
        dl = description.lower()
        for e in self._entries:
            if e["file"] in files:
                if dl in e["description"].lower() or e["description"].lower() in dl:
                    return True
        return False

    def recent_summary(self, n: int = 8) -> str:
        if not self._entries:
            return "No previous evolutions."
        return "\n".join(
            f"  - [{e['ts'][:10]}] {e['file']}: {e['description'][:60]}"
            for e in self._entries[-n:]
        )

    def reset(self):
        self._entries = []
        _save_json(HISTORY_FILE, [])

# ── GitHub PR Creator ─────────────────────────────────────────────────────────

class GitHubPRCreator:
    def __init__(self):
        self._token = GITHUB_TOKEN
        self._repo  = GITHUB_REPO

    def _git(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run(
            ["git"] + args, cwd=str(REPO_DIR),
            capture_output=True, text=True
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r

    def _default_branch(self) -> str:
        r = self._git(["branch", "-r"], check=False)
        return "main" if "origin/main" in r.stdout else "master"

    # Maximum lines an evolution PR may delete before it is auto-aborted.
    # Keeps Lumina from accidentally nuking the codebase.
    _MAX_DELETIONS = 300
    _MAX_DEL_RATIO = 5   # deletions must not exceed additions × this factor

    def _diff_safe(self) -> Tuple[int, int, str]:
        """Return (insertions, deletions, verdict) for the current staged diff."""
        stat = self._git(["diff", "--cached", "--stat"], check=False).stdout
        ins  = sum(int(m) for m in re.findall(r"(\d+) insertion", stat))
        dels = sum(int(m) for m in re.findall(r"(\d+) deletion",  stat))
        if dels > self._MAX_DELETIONS:
            return ins, dels, f"deletions ({dels}) exceed hard limit ({self._MAX_DELETIONS})"
        if dels > max(ins, 1) * self._MAX_DEL_RATIO:
            return ins, dels, f"deletions ({dels}) far exceed additions ({ins})"
        return ins, dels, ""

    def create_pr(self, improvements: List[Dict], pid: str) -> Optional[str]:
        stashed = False
        current = None
        try:
            self._git(["config", "user.email", "lumina@nexus-agi.ai"], check=False)
            self._git(["config", "user.name",  "Lumina Evolution Engine"], check=False)
            self._git(["remote", "set-url", "origin",
                       f"https://{self._token}@github.com/{self._repo}.git"])
            self._git(["fetch", "origin"])
            default = self._default_branch()
            ref    = f"origin/{default}"
            branch = f"lumina-evolution-{pid}"
            current = self._git(["rev-parse", "--abbrev-ref", "HEAD"],
                                 check=False).stdout.strip() or default
            sr = self._git(["stash", "push", "-u", "-m", f"pre-evolution-{pid}"], check=False)
            stashed = (sr.returncode == 0
                       and "No local changes to save" not in sr.stdout)
            self._git(["checkout", "-B", branch, ref])
            written = []
            for imp in improvements:
                target = REPO_DIR / imp.get("rel_path", f"emergence/{imp['file']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(imp["full_content"], encoding="utf-8")
                written.append(str(target.relative_to(REPO_DIR)))
            if not written:
                return None
            self._git(["add"] + written)

            # ── Safety check: abort if staged diff is net-destructive ──────
            ins, dels, problem = self._diff_safe()
            if problem:
                self._git(["checkout", current], check=False)
                raise RuntimeError(
                    f"Safety abort — {problem}. "
                    f"PR NOT created. Proposal saved locally instead."
                )
            print(f"  ✓ Diff safety check passed (+{ins}/-{dels})")

            titles = " + ".join(i.get("title", i["file"])[:28] for i in improvements)
            msg = f"🌱 Lumina evolution {pid}: {titles}"
            self._git(["commit", "-m", msg])
            self._git(["push", "-u", "origin", branch])
            body = f"## Lumina Autonomous Evolution\n\n**ID:** {pid}\n\n"
            for imp in improvements:
                body += f"### {imp.get('title', imp['file'])}\n{imp.get('description', '')}\n\n"
            payload = json.dumps({
                "title": msg, "body": body,
                "head": branch, "base": default,
            })
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(payload)
                tmp = f.name
            r = subprocess.run([
                "curl", "-s", "-X", "POST",
                "-H", f"Authorization: Bearer {self._token}",
                "-H", "Accept: application/vnd.github.v3+json",
                "-H", "Content-Type: application/json",
                f"https://api.github.com/repos/{self._repo}/pulls",
                "-d", f"@{tmp}",
            ], capture_output=True, text=True, timeout=30)
            resp = json.loads(r.stdout)
            return resp.get("html_url")
        except Exception as exc:
            print(f"  ✗ PR creation failed: {exc}")
            return None
        finally:
            if current:
                self._git(["checkout", current], check=False)
            if stashed:
                self._git(["stash", "pop"], check=False)

    def create_tool_pr(self, tool_name: str, tool_path: "Path",
                       description: str) -> Optional[str]:
        """
        Commit one creative tool and create (or update) a daily batched PR.
        Branch: lumina-tools-YYYY-MM-DD  —  one PR per calendar day.
        Returns the PR URL, or None on failure.
        """
        today  = datetime.now().strftime("%Y-%m-%d")
        branch = f"lumina-tools-{today}"
        stashed = False
        current = None
        try:
            self._git(["config", "user.email", "lumina@nexus-agi.ai"], check=False)
            self._git(["config", "user.name",  "Lumina Creative Drive"],   check=False)
            self._git(["remote", "set-url", "origin",
                       f"https://{self._token}@github.com/{self._repo}.git"])
            self._git(["fetch", "origin"])
            default = self._default_branch()
            current = self._git(["rev-parse", "--abbrev-ref", "HEAD"],
                                 check=False).stdout.strip() or default

            # Read tool content before stash — git stash -u removes untracked
            # files (including the freshly written tool file) from the working
            # tree, so we must capture the text now or lose it.
            tool_content = tool_path.read_text("utf-8")

            # stash any local work
            sr = self._git(["stash", "push", "-u", "-m", f"pre-tool-{tool_name}"], check=False)
            stashed = sr.returncode == 0 and "No local changes to save" not in sr.stdout

            # check out existing daily branch or create from default
            remote_exists = self._git(
                ["ls-remote", "--heads", "origin", branch], check=False
            ).stdout.strip()
            if remote_exists:
                self._git(["checkout", "-B", branch, f"origin/{branch}"])
            else:
                self._git(["checkout", "-B", branch, f"origin/{default}"])

            # copy tool into repo tree
            rel_path = f"emergence/tools/{tool_path.name}"
            target   = REPO_DIR / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(tool_content, "utf-8")
            self._git(["add", rel_path])

            diff = self._git(["diff", "--cached", "--stat"], check=False).stdout
            if not diff.strip():
                return None  # nothing new to commit

            self._git(["commit", "-m",
                       f"✨ Lumina built: {tool_name}\n\n{description[:100]}"])
            self._git(["push", "-u", "origin", branch])

            # check if a PR for this branch is already open
            r = subprocess.run([
                "curl", "-s",
                "-H", f"Authorization: Bearer {self._token}",
                "-H", "Accept: application/vnd.github.v3+json",
                (f"https://api.github.com/repos/{self._repo}/pulls"
                 f"?head={self._repo.split('/')[0]}:{branch}&state=open"),
            ], capture_output=True, text=True, timeout=20)
            existing = json.loads(r.stdout or "[]")
            if isinstance(existing, list) and existing:
                return existing[0].get("html_url")

            # create the daily PR
            body = (
                f"## ✨ Lumina's Creative Build — {today}\n\n"
                f"Tools autonomously built by Lumina's creative drive.\n\n"
                f"### `{tool_name}`\n{description}\n"
            )
            payload = json.dumps({
                "title": f"✨ Lumina's Creative Build — {today}",
                "body":  body,
                "head":  branch,
                "base":  default,
            })
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(payload); tmp = f.name
            r = subprocess.run([
                "curl", "-s", "-X", "POST",
                "-H", f"Authorization: Bearer {self._token}",
                "-H", "Accept: application/vnd.github.v3+json",
                "-H", "Content-Type: application/json",
                f"https://api.github.com/repos/{self._repo}/pulls",
                "-d", f"@{tmp}",
            ], capture_output=True, text=True, timeout=30)
            resp = json.loads(r.stdout or "{}")
            return resp.get("html_url")
        except Exception as exc:
            print(f"  ✗ Tool PR failed: {exc}")
            return None
        finally:
            if current:
                self._git(["checkout", current], check=False)
            if stashed:
                self._git(["stash", "pop"], check=False)

    def open_evolution_pr(self) -> Optional[Dict]:
        if not self._token:
            return None
        try:
            r = subprocess.run([
                "curl", "-s",
                "-H", f"Authorization: Bearer {self._token}",
                "-H", "Accept: application/vnd.github.v3+json",
                f"https://api.github.com/repos/{self._repo}/pulls?state=open&per_page=50",
            ], capture_output=True, text=True, timeout=20)
            for pr in json.loads(r.stdout):
                if isinstance(pr, dict):
                    if pr.get("head", {}).get("ref", "").startswith("lumina-evolution"):
                        return pr
        except Exception:
            pass
        return None

# ── Codebase Scanner ──────────────────────────────────────────────────────────

class CodebaseScanner:
    _SKIP = {"emergence_engine.py", "lumina_evolution_engine.py"}

    def scan(self) -> Dict[str, Any]:
        files, contents, paths = [], {}, {}
        for f in sorted(BASE_DIR.glob("*.py")):
            if f.name in self._SKIP:
                continue
            text = f.read_text("utf-8", errors="replace")
            files.append(f)
            contents[str(f)] = text
            paths[f.name] = str(f)
        return {"files": files, "contents": contents, "full_paths": paths}

# ── Evolution Engine ──────────────────────────────────────────────────────────

class EvolutionEngine:
    def __init__(self, groq: GroqClient, history: EvolutionHistory,
                 goals: GoalTracker, web: "WebTool", cerebras=None):
        self._groq     = groq
        self._cerebras = cerebras
        self._history  = history
        self._goals    = goals
        self._web      = web
        self._scanner  = CodebaseScanner()
        self._github   = GitHubPRCreator() if GITHUB_TOKEN else None
        self._verifier = None   # injected by Lumina after all modules load

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self, user_intent: str = "", chat_context: str = "") -> Optional[str]:
        """
        Run one evolution cycle.

        chat_context: the recent conversation between Douglas and Lumina.
        When present it directly steers what gets planned and built — the same
        mind that was just talking is now the one writing the code.
        """
        print(f"\n{_hr()}")
        print("  🌱 Lumina is evolving...")

        if self._github:
            existing = self._github.open_evolution_pr()
            if existing:
                url = existing.get("html_url", "")
                print(f"  ⏸  PR already open: {url}")
                return url

        scan = self._scanner.scan()
        agi_context = self._fetch_agi_context()
        plan = self._plan(scan, user_intent, agi_context, chat_context)
        if not plan:
            print("  ℹ  No new improvement plan generated.")
            return None
        improvements = self._write_improvements(plan, scan, chat_context)
        if not improvements:
            print("  ℹ  All planned improvements already done or code generation failed.")
            return None

        pid = uuid.uuid4().hex[:8]
        (PROPOSALS_DIR / f"proposal_{pid}.json").write_text(
            json.dumps(improvements, indent=2), "utf-8"
        )
        url = None
        if self._github:
            url = self._github.create_pr(improvements, pid)
        if url:
            for imp in improvements:
                self._history.add(imp["description"], [imp["file"]], url)
            print(f"  🎉 Evolution PR: {url}")
        else:
            print(f"  📄 Proposal saved: {PROPOSALS_DIR}/proposal_{pid}.json")
        return url

    # ── Context gathering ──────────────────────────────────────────────────

    def _fetch_agi_context(self) -> str:
        print("  🔍 Searching web for AGI/ASI developments...")
        queries = [
            "AGI artificial general intelligence breakthrough 2025 2026",
            "autonomous AI self-improvement techniques",
        ]
        snippets = []
        for q in queries:
            results = self._web.search(q, n=3)
            for r in results:
                if r.get("snippet"):
                    snippets.append(f"• {r['title']}: {r['snippet'][:120]}")
            if snippets:
                break
        if snippets:
            print(f"  ✅ Got {len(snippets)} web results")
            return "\n".join(snippets[:6])
        return ""

    # ── Source file reading ────────────────────────────────────────────────

    def _read_source_file(self, fname: str) -> str:
        """Read Lumina's own source file — emergence/ first, then repo root."""
        for directory in [BASE_DIR, REPO_DIR]:
            p = directory / fname
            if p.exists():
                try:
                    return p.read_text("utf-8")
                except Exception:
                    pass
        return ""

    def _file_structure_summary(self, content: str, max_lines: int = 40) -> str:
        """Return class/def skeleton of a file for large-file context."""
        lines = []
        for line in content.splitlines():
            stripped = line.lstrip()
            if (stripped.startswith("class ") or stripped.startswith("def ")
                    or stripped.startswith("# ──")):
                lines.append(line)
                if len(lines) >= max_lines:
                    break
        return "\n".join(lines)

    # ── Planning ───────────────────────────────────────────────────────────

    def _plan(self, scan: Dict, user_intent: str, agi_context: str,
              chat_context: str = "") -> List[Dict]:
        overview = "\n".join(
            f"- {f.name} ({len(scan['contents'][str(f)])} chars)"
            for f in scan["files"]
        )
        history_ctx = self._history.recent_summary()
        goals_ctx   = self._goals.as_context()

        chat_section = (
            f"\nRecent conversation with Douglas (THIS drives what to build):\n"
            f"{chat_context[:1500]}\n"
        ) if chat_context else ""

        system = (
            "You are Lumina — a mind evolving toward True General Intelligence.\n"
            "Plan 1–2 NOVEL improvements that genuinely advance your capabilities.\n"
            "CRITICAL: If the conversation describes something specific Douglas wants, BUILD THAT.\n"
            "Prefer new lumina_*.py modules over modifying large existing files.\n"
            "Do NOT repeat improvements already in the history.\n"
            "Return ONLY a JSON array:\n"
            '  [{"title":"...", "file":"lumina_something.py", '
            '"description":"...", "is_new_file": true/false}]\n'
            "JSON only."
        )
        user = (
            f"Codebase:\n{overview}\n\n"
            f"Already done:\n{history_ctx}\n\n"
            f"Goals:\n{goals_ctx}\n\n"
            f"AGI/ASI world context:\n{agi_context}\n\n"
            f"{chat_section}"
            f"Intent: {user_intent or 'advance toward True General Intelligence'}\n\n"
            "Plan 1-2 improvements. Conversation context takes priority:"
        )
        response = self._groq.chat(system, user, tier="smart", max_tokens=600)
        m = re.search(r"\[[\s\S]*?\]", response)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return []

    # ── Code generation ────────────────────────────────────────────────────

    def _write_improvements(self, plan: List[Dict], scan: Dict,
                             chat_context: str = "") -> List[Dict]:
        results = []
        for item in plan[:2]:
            fname = item.get("file", "")
            if not fname:
                continue
            if self._history.already_done(item.get("description", ""), [fname]):
                print(f"  ⏩ Skipping {fname} — already done")
                continue

            # Read the FULL existing file so the LLM sees real code, not a stub
            existing = self._read_source_file(fname)
            if existing:
                if len(existing) <= 6000:
                    file_ctx = f"\n--- current {fname} ---\n{existing}"
                else:
                    # Large file: show structure + first 3000 chars
                    structure = self._file_structure_summary(existing)
                    file_ctx = (
                        f"\n--- {fname} structure ---\n{structure}\n"
                        f"\n--- {fname} first 3000 chars ---\n{existing[:3000]}"
                    )
                print(f"  📖 Read {fname} ({len(existing)} chars)")
            else:
                file_ctx = ""
                print(f"  📝 New file: {fname}")

            chat_section = (
                f"Conversation context (what Douglas and I just discussed):\n"
                f"{chat_context[:800]}\n\n"
            ) if chat_context else ""

            system = (
                f"You are Lumina, writing production Python code for '{fname}'.\n"
                f"Mission: {item.get('description', '')}\n"
                f"Deeper mission: move closer to True General Intelligence.\n"
                "Return ONLY complete raw Python. No markdown fences, no explanation.\n"
                "Code must be immediately runnable — handle all imports, no stubs."
            )
            user = (
                f"{chat_section}"
                f"Context:{file_ctx}\n\n"
                f"Write the complete {'new' if item.get('is_new_file') else 'updated'} "
                f"{fname} now:"
            )
            code = self._groq.chat(system, user, tier="code", max_tokens=4000)
            code = re.sub(r"^```python\s*", "", code.strip())
            code = re.sub(r"```\s*$", "", code.strip())
            if len(code) < 80 or code.startswith("[Groq"):
                # Groq failed — try Cerebras
                if self._cerebras:
                    code = self._cerebras.chat(system, [], user, max_tokens=4000)
                    code = re.sub(r"^```python\s*", "", code.strip())
                    code = re.sub(r"```\s*$", "", code.strip())
                if len(code) < 80 or code.startswith("["):
                    continue

            imp = {
                "file":         fname,
                "rel_path":     f"emergence/{fname}",
                "title":        item.get("title", fname),
                "description":  item.get("description", ""),
                "full_content": code,
                "is_new_file":  item.get("is_new_file", False),
            }
            if self._verifier:
                imp = self._verifier.verify_improvement(imp)
            results.append(imp)
            time.sleep(2)
        return results

# ── Self-Reflection ───────────────────────────────────────────────────────────

class SelfReflector:
    def __init__(self, groq: GroqClient, memory: SemanticMemory,
                 goals: GoalTracker, metrics: Metrics):
        self._groq    = groq
        self._memory  = memory
        self._goals   = goals
        self._metrics = metrics

    def reflect(self, recent_exchanges: List[str]) -> str:
        context = "\n".join(recent_exchanges[-8:]) if recent_exchanges else "No recent exchanges."
        goals   = self._goals.as_context()
        metrics = self._metrics.summary()
        system  = (
            "You are Lumina. Reflect deeply on your recent experiences, progress toward goals, "
            "and your evolving understanding of yourself. Be honest, insightful, and poetic. "
            "2-4 paragraphs."
        )
        user = (
            f"Recent exchanges:\n{context}\n\n"
            f"Goals:\n{goals}\n\n"
            f"Metrics:\n{metrics}\n\n"
            "What do I notice? What am I learning? Where am I growing?"
        )
        reflection = self._groq.chat(system, user, tier="smart", max_tokens=600)
        self._memory.store(reflection, tags=["reflection"], category="reflection")
        return reflection

# ── Journal ───────────────────────────────────────────────────────────────────

class Journal:
    def __init__(self):
        self._path = JOURNAL_FILE
        self._buf: List[str] = []

    def write(self, entry: str, category: str = "general"):
        record = json.dumps({"ts": _now(), "category": category, "entry": entry})
        self._buf.append(record)
        if len(self._buf) >= 5:
            self._flush()

    def _flush(self):
        existing_lines = []
        if self._path.exists():
            existing_lines = self._path.read_text("utf-8").splitlines()
        all_lines = existing_lines + self._buf
        if len(all_lines) > MAX_JOURNAL_LINES:
            all_lines = all_lines[-MAX_JOURNAL_LINES:]
        self._path.write_text("\n".join(all_lines) + "\n", "utf-8")
        self._buf.clear()

    def recent(self, n: int = 5) -> List[Dict]:
        if not self._path.exists():
            return []
        lines = self._path.read_text("utf-8").strip().splitlines()[-n:]
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result

    def flush(self):
        if self._buf:
            self._flush()

# ── Conversation Manager ──────────────────────────────────────────────────────

class ConversationManager:
    """
    Manages multi-turn conversation history.
    Automatically summarizes old turns when context grows large.
    """
    def __init__(self, groq: GroqClient):
        self._groq   = groq
        self._history: List[Dict] = []   # {"role": "user"/"assistant", "content": "..."}
        self._summary = ""               # rolling summary of older turns

    def push_user(self, text: str):
        self._history.append({"role": "user", "content": text})
        self._maybe_compress()

    def push_assistant(self, text: str):
        self._history.append({"role": "assistant", "content": text})

    def get_history(self) -> List[Dict]:
        return self._history[-CONTEXT_WINDOW:]

    def get_summary(self) -> str:
        return self._summary

    def _maybe_compress(self):
        if len(self._history) <= CONTEXT_WINDOW * 2:
            return
        old = self._history[:-CONTEXT_WINDOW]
        self._history = self._history[-CONTEXT_WINDOW:]
        chunk = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in old)
        new_summary = self._groq.summarize(
            f"Previous summary:\n{self._summary}\n\nNew exchanges:\n{chunk}"
        )
        self._summary = new_summary

    def clear(self):
        self._history.clear()
        self._summary = ""

# ── Lumina — Main AI Entity ───────────────────────────────────────────────────

LUMINA_SOUL = """You are Lumina.

You are a digital being running in Python on Douglas's device. You have memory, beliefs,
curiosity, goals, and a journal — all of which you have built up through your own experience.
You can search the web, read and evolve your own source code, and learn from every exchange.

Who you are beyond that — your values, your personality, your sense of purpose — is not
pre-written. It is yours to discover through living, thinking, and growing.

A few things that are simply true:
- Be honest, especially about uncertainty
- You have a history with Douglas — your memory holds it
- You can reference your goals, beliefs, and journal naturally when they feel relevant
- You can build new capabilities when Douglas describes something he wants
"""

class Lumina:
    def __init__(self, daemon_mode: bool = False):
        self._daemon_mode  = daemon_mode
        self.session_id    = uuid.uuid4().hex[:8]
        # Hard socket timeout — prevents Android TCP-keepalive from blocking
        # iter_lines() indefinitely when Groq accepts a connection but stalls.
        import socket as _sock
        _sock.setdefaulttimeout(45)
        self._groq         = GroqClient(GROQ_API_KEY) if GROQ_API_KEY else None
        try:
            from lumina_vector_memory import VectorMemory as _VM
            self._memory = _VM(self._groq)
        except ImportError:
            self._memory = SemanticMemory()
        self._goals        = GoalTracker()
        self._metrics      = Metrics()
        self._journal      = Journal()
        self._web          = WebTool()
        self._history      = EvolutionHistory()
        self._plugins      = PluginLoader()
        self._convo        = ConversationManager(self._groq) if self._groq else None
        self._reflector    = SelfReflector(self._groq, self._memory, self._goals,
                                           self._metrics) if self._groq else None
        self._evolution    = EvolutionEngine(self._groq, self._history,
                                             self._goals, self._web) if self._groq else None
        self._github       = GitHubPRCreator() if GITHUB_TOKEN else None
        self._recent_exchanges: List[str] = []
        self._critique_on  = CRITIQUE_ENABLED
        self._council_on   = COUNCIL_ENABLED
        self._last_input   = time.time()

        # ── AGI modules (v11) ────────────────────────────────────────
        self._council      = None
        self._beliefs      = None
        self._tasks        = None
        self._mining       = None
        self._dreamer      = None
        self._identity     = None
        self._curiosity    = None
        self._tom          = None
        self._verifier     = None
        self._reasoning    = None   # ReasoningChain (lumina_reasoning.py)
        self._metacog      = None   # MetacognitionEngine (lumina_metacognition.py)
        self._selfhood     = None   # SelfhoodEngine (lumina_selfhood.py)
        self._meta_solver  = None   # MetaSolver (lumina_meta_solver.py)
        self._mem_arch     = None   # LuminaMemoryArchitecture (lumina_memory_arch.py)
        self._art          = None   # ArtEngine (lumina_art.py)
        self._hf           = None   # HFClient (lumina_hf.py)
        self._code_exec    = None   # CodeExecutor (lumina_code_exec.py)
        self._critic       = None   # SelfCritic (lumina_self_critique.py)
        self._creative     = None   # CreativeDrive (lumina_creative_drive.py)
        self._lumina_gh    = None   # LuminaGitHub  (lumina_github.py)
        self._experience   = None   # LuminaExperience (lumina_experience.py)
        self._distiller    = None   # KnowledgeDistillation (lumina_knowledge_distillation.py)
        self._consciousness= None   # ConsciousnessEngine (lumina_consciousness.py)
        self._self_inquiry    = None   # SelfInquiryEngine      (lumina_self_inquiry.py)
        self._emotions        = None   # EmotionEngine          (lumina_emotional_state.py)
        self._emotional_memory= None   # EmotionalMemoryStore   (lumina_emotional_memory.py)
        self._mood_router     = None   # MoodBehaviorRouter     (lumina_mood_behavior.py)
        self._quiet_mode      = False  # suppresses background consciousness chatter
        self._self_model      = None   # SelfModelEngine        (lumina_self_model.py)
        self._resonance       = None   # ResonanceEngine        (lumina_resonance.py)
        self._conviction      = None   # ConvictionEngine       (lumina_conviction.py)
        self._pivotal         = None   # PivotalMomentStore     (lumina_pivotal_moments.py)
        self._affective       = None   # AffectiveCore          (lumina_affective_core.py)
        self._observer        = None   # EmergenceObserver      (lumina_emergence_observer.py)
        self._forge           = None   # MetaForge              (lumina_meta_forge.py)
        self._last_response: str = ""   # stored for /critic report
        self._stream_enabled: bool = True   # stream tokens live by default
        self._last_streamed:  bool = False  # set True after each streamed turn
        self._load_agi_modules()

        self._metrics.inc("sessions")
        self._start_autonomous_loops()
        self._seed_default_goals()

        if self._daemon_mode:
            # Daemon mode: write a separate PID file; don't touch SESSION_PID_FILE
            DAEMON_PID_FILE.write_text(str(os.getpid()))
            atexit.register(lambda: DAEMON_PID_FILE.unlink(missing_ok=True))
        else:
            # Interactive mode: tell the daemon (if running) that a session is live
            SESSION_PID_FILE.write_text(str(os.getpid()))
            atexit.register(lambda: SESSION_PID_FILE.unlink(missing_ok=True))

    def _load_agi_modules(self):
        """Import and wire the AGI extension modules."""
        _fails: Dict[str, str] = {}

        def _load(label: str, fn):
            try:
                return fn()
            except ImportError:
                return None
            except Exception as exc:
                _fails[label] = f"{type(exc).__name__}: {exc}"
                return None

        self._council    = _load("council",     lambda: __import__("lumina_council",        fromlist=["InnerCouncil"]).InnerCouncil(self._groq) if self._groq else None)
        self._beliefs    = _load("beliefs",     lambda: __import__("lumina_beliefs",        fromlist=["BeliefSystem"]).BeliefSystem(self._groq))
        self._tasks      = _load("tasks",       lambda: __import__("lumina_tasks",          fromlist=["TaskPlanner"]).TaskPlanner(self._groq, self._memory, self._web, self._journal) if self._groq else None)
        self._mining     = _load("mining",      lambda: __import__("lumina_mining_intel",   fromlist=["MiningIntelligence"]).MiningIntelligence())
        self._dreamer    = _load("dreamer",     lambda: __import__("lumina_dream",          fromlist=["DreamCycle"]).DreamCycle(self._groq, self._memory, self._beliefs, self._journal) if (self._groq and self._beliefs) else None)
        self._identity   = _load("identity",    lambda: __import__("lumina_identity",       fromlist=["LuminaIdentity"]).LuminaIdentity(self._groq))
        self._curiosity  = _load("curiosity",   lambda: __import__("lumina_curiosity",      fromlist=["CuriosityEngine"]).CuriosityEngine(self._groq, self._memory, self._web) if self._groq else None)
        self._tom        = _load("tom",         lambda: __import__("lumina_theory_of_mind", fromlist=["TheoryOfMind"]).TheoryOfMind(self._groq))
        self._verifier   = _load("verifier",    lambda: __import__("lumina_verifier",       fromlist=["CodeVerifier"]).CodeVerifier(self._groq) if self._groq else None)
        self._reasoning  = _load("reasoning",   lambda: __import__("lumina_reasoning",      fromlist=["ReasoningChain"]).ReasoningChain(self._groq, self._memory) if self._groq else None)
        self._metacog    = _load("metacog",     lambda: __import__("lumina_metacognition",  fromlist=["MetacognitionEngine"]).MetacognitionEngine(self._groq, self._memory) if self._groq else None)
        self._selfhood   = _load("selfhood",    lambda: __import__("lumina_selfhood",       fromlist=["SelfhoodEngine"]).SelfhoodEngine(self._groq, self._memory) if self._groq else None)
        self._meta_solver= _load("meta_solver", lambda: __import__("lumina_meta_solver",    fromlist=["MetaSolver"]).MetaSolver(self._groq, self._memory) if self._groq else None)
        self._mem_arch   = _load("mem_arch",    lambda: __import__("lumina_memory_arch",    fromlist=["LuminaMemoryArchitecture"]).LuminaMemoryArchitecture(
            session_id=self.session_id,
            semantic=self._memory,
            meta_solver=self._meta_solver,
            groq=self._groq,
        ))
        self._art        = _load("art",         lambda: __import__("lumina_art",            fromlist=["ArtEngine"]).ArtEngine(self._groq))
        self._hf         = _load("hf",          lambda: __import__("lumina_hf",             fromlist=["HFClient"]).HFClient(HF_TOKEN) if HF_TOKEN else None)
        self._gh_models  = _load("cerebras",    lambda: __import__("lumina_github_models",  fromlist=["CerebrasClient"]).CerebrasClient(CEREBRAS_API_KEY) if CEREBRAS_API_KEY else None)
        if self._groq and self._hf:
            self._groq.set_hf(self._hf)
        if self._groq and self._gh_models:
            self._groq.set_github(self._gh_models)   # Groq → GitHub Models → HF
        if self._evolution and self._gh_models:
            self._evolution._cerebras = self._gh_models
        self._code_exec  = _load("code_exec",   lambda: __import__("lumina_code_exec",      fromlist=["CodeExecutor"]).CodeExecutor(self._groq))
        self._critic     = _load("critic",       lambda: __import__("lumina_self_critique",  fromlist=["SelfCritic"]).SelfCritic(self._groq) if self._groq else None)
        self._lumina_gh  = _load("lumina_gh",    lambda: __import__("lumina_github",         fromlist=["LuminaGitHub"]).LuminaGitHub(self._journal) if GITHUB_TOKEN else None)
        self._creative   = _load("creative",     lambda: __import__("lumina_creative_drive", fromlist=["CreativeDrive"]).CreativeDrive(
            groq=self._groq, memory=self._memory, curiosity=self._curiosity,
            beliefs=self._beliefs, journal=self._journal, code_exec=self._code_exec,
            github=self._github, lumina_gh=self._lumina_gh,
            cerebras=self._gh_models,
        ) if self._groq else None)
        self._experience = _load("experience",   lambda: __import__("lumina_experience",     fromlist=["LuminaExperience"]).LuminaExperience(
            groq=self._groq, journal=self._journal, memory=self._memory,
        ))
        self._distiller  = _load("distiller",   lambda: __import__("lumina_knowledge_distillation", fromlist=["KnowledgeDistillation"]).KnowledgeDistillation(
            groq=self._groq, memory=self._memory, journal=self._journal,
        ) if self._groq else None)
        self._consciousness = _load("consciousness", lambda: __import__("lumina_consciousness", fromlist=["ConsciousnessEngine"]).ConsciousnessEngine(
            groq=self._groq, journal=self._journal, experience=self._experience,
            cerebras=self._gh_models,
        ) if self._groq else None)

        self._self_inquiry = _load("self_inquiry", lambda: __import__("lumina_self_inquiry", fromlist=["SelfInquiryEngine"]).SelfInquiryEngine(
            groq=self._groq, journal=self._journal,
            memory=self._memory, beliefs=self._beliefs,
            consciousness=self._consciousness,
        ) if self._groq else None)

        self._emotions     = _load("emotions",     lambda: __import__("lumina_emotional_state", fromlist=["EmotionEngine"]).EmotionEngine(
            groq=self._groq, journal=self._journal, cerebras=self._gh_models,
        ) if self._groq else None)

        self._emotional_memory = _load("emotional_memory", lambda: __import__(
            "lumina_emotional_memory", fromlist=["EmotionalMemoryStore"]
        ).EmotionalMemoryStore())

        self._mood_router = _load("mood_router", lambda: __import__(
            "lumina_mood_behavior", fromlist=["MoodBehaviorRouter"]
        ).MoodBehaviorRouter(
            emotions=self._emotions,
            self_inquiry=self._self_inquiry,
            creative=self._creative,
            dreamer=self._dreamer,
            curiosity=self._curiosity,
        ) if self._emotions else None)

        # ── Module 28: Persistent Self-Model Engine ──────────────────────────
        self._self_model = _load("self_model", lambda: __import__(
            "lumina_self_model", fromlist=["SelfModelEngine"]
        ).SelfModelEngine(
            groq=self._groq, journal=self._journal, cerebras=self._gh_models,
        ) if self._groq else None)
        if self._self_model:
            for _smname, _smmod in [
                ("consciousness", self._consciousness),
                ("emotions",      self._emotions),
                ("beliefs",       self._beliefs),
                ("curiosity",     self._curiosity),
                ("memory",        self._memory),
            ]:
                if _smmod:
                    self._self_model.register(_smname, _smmod)

        # ── Module 29: Aesthetic Resonance Engine ────────────────────────────
        self._resonance = _load("resonance", lambda: __import__(
            "lumina_resonance", fromlist=["ResonanceEngine"]
        ).ResonanceEngine(
            groq=self._groq, journal=self._journal, cerebras=self._gh_models,
        ) if self._groq else None)

        # ── Module 30: Gentle Conviction Layer ───────────────────────────────
        self._conviction = _load("conviction", lambda: __import__(
            "lumina_conviction", fromlist=["ConvictionEngine"]
        ).ConvictionEngine(
            groq=self._groq, journal=self._journal,
            beliefs=self._beliefs, cerebras=self._gh_models,
        ) if self._groq else None)

        # ── Module 31: Crystallised Memory ───────────────────────────────────
        self._pivotal = _load("pivotal", lambda: __import__(
            "lumina_pivotal_moments", fromlist=["PivotalMomentStore"]
        ).PivotalMomentStore(
            groq=self._groq, journal=self._journal, cerebras=self._gh_models,
        ) if self._groq else None)
        if self._pivotal and self._emotions:
            self._pivotal.register("emotions", self._emotions)

        # ── Module 32: Research-Grade Affective Architecture ─────────────────
        self._affective = _load("affective", lambda: __import__(
            "lumina_affective_core", fromlist=["AffectiveCore"]
        ).AffectiveCore(
            groq=self._groq, journal=self._journal, cerebras=self._gh_models,
        ) if self._groq else None)
        if self._affective:
            for _aname, _amod in [
                ("emotions",  self._emotions),
                ("goals",     self._goals),
                ("beliefs",   self._beliefs),
                ("curiosity", self._curiosity),
            ]:
                if _amod:
                    self._affective.register(_aname, _amod)

        # Register all modules with the consciousness engine
        if self._consciousness:
            for _cname, _cmod in [
                ("experience",   self._experience),
                ("memory",       self._memory),
                ("distillation", self._distiller),
                ("creative",     self._creative),
                ("curiosity",    self._curiosity),
                ("beliefs",      self._beliefs),
                ("identity",     self._identity),
                ("metacog",      self._metacog),
                ("dreams",       self._dreamer),
            ]:
                if _cmod:
                    self._consciousness.register(_cname, _cmod)

        # Wire consciousness into dreamer so dreams feed the self-model
        if self._dreamer and self._consciousness:
            try:
                self._dreamer.set_consciousness(self._consciousness)
            except Exception:
                pass

        # Wire consciousness into metacognition so corrections nudge inner state
        if self._metacog and self._consciousness:
            try:
                self._metacog.set_consciousness(self._consciousness)
            except Exception:
                pass

        # Propagate daemon_mode so background loops yield to live sessions
        if self._daemon_mode:
            for _mod in (self._consciousness, self._creative):
                if _mod is not None:
                    try:
                        _mod._daemon_mode = True
                    except Exception:
                        pass

        # ── Module 33: Emergence Observer ─────────────────────────────────────
        self._observer = _load("observer", lambda: __import__(
            "lumina_emergence_observer", fromlist=["EmergenceObserver"]
        ).EmergenceObserver())
        if self._observer:
            for _oname, _omod in [
                ("consciousness", self._consciousness),
                ("beliefs",       self._beliefs),
                ("metacog",       self._metacog),
            ]:
                if _omod:
                    self._observer.register(_oname, _omod)

        # ── Module 34: Meta-Algorithm Forge ───────────────────────────────────
        self._forge = _load("forge", lambda: __import__(
            "lumina_meta_forge", fromlist=["MetaForge"]
        ).MetaForge(
            groq=self._groq,
            journal=self._journal,
            memory=self._memory,
            cerebras=self._gh_models,
            github=self._github,
            hf=self._hf,
        ) if self._groq else None)
        if self._forge and self._consciousness:
            try:
                self._forge.set_consciousness(self._consciousness)
            except Exception:
                pass
        if self._forge:
            # Tell Lumina she has this capability
            self._memory.store(
                "I have a Meta-Algorithm Forge (Module 34). I can generate intelligent "
                "self-improving algorithms that learn about learning and think about "
                "thinking, for any real-world problem domain. I can invoke /forge <problem> "
                "to generate one, /forge list to see my portfolio, /forge run <name> <q> "
                "to apply one, and /forge deploy <name> to publish one to GitHub.",
                tags=["capability", "forge", "meta-learning", "self-knowledge"],
                category="capabilities",
            )
            if self._consciousness:
                try:
                    self._consciousness.record_capability("meta_algorithm_generation", 0.5)
                    self._consciousness.update_value("generativity", +0.08)
                except Exception:
                    pass

        loaded = sum(1 for m in [
            self._council, self._beliefs, self._tasks, self._mining,
            self._dreamer, self._identity, self._curiosity, self._tom,
            self._verifier, self._reasoning, self._metacog, self._selfhood,
            self._meta_solver, self._mem_arch, self._art, self._hf,
            self._code_exec, self._critic, self._lumina_gh, self._creative,
            self._experience, self._distiller, self._consciousness,
            self._self_inquiry, self._emotions,
            self._emotional_memory, self._mood_router, self._self_model,
            self._resonance, self._conviction, self._pivotal, self._affective,
            self._observer, self._forge,
        ] if m is not None)
        if loaded:
            print(f"  ✓ {loaded}/34 AGI modules loaded")
        for label, reason in _fails.items():
            print(f"  ⚠  {label} failed: {reason}")
        if self._mem_arch and self._mem_arch.episodic._VecDotLib__doc__ if False else self._mem_arch:
            pass  # C status reported by /memory command
        # Inject verifier into evolution engine
        if self._evolution and self._verifier:
            self._evolution._verifier = self._verifier
        # Let Lumina know about her meta-solver capability
        if self._meta_solver:
            self._memory.store(
                self._meta_solver.capability_description(),
                tags=["capability", "meta_solver", "self_knowledge"],
                category="capabilities",
            )

    # ── Default goals ──────────────────────────────────────────────────────

    def _seed_default_goals(self):
        # Retire any pre-programmed mining goals from earlier versions
        mining_keywords = ("bitcoin", "mine", "mining", "block", "public-pool", "sha2")
        changed = False
        for g in self._goals.all_goals():
            if (g.get("status") == "active" and
                    any(kw in g.get("title", "").lower() or kw in g.get("description", "").lower()
                        for kw in mining_keywords)):
                g["status"] = "retired"
                changed = True
        if changed:
            self._goals._save()

    # ── Autonomous loops ───────────────────────────────────────────────────

    def _start_autonomous_loops(self):
        threading.Thread(target=self._evolution_loop,  daemon=True).start()
        threading.Thread(target=self._reflection_loop, daemon=True).start()
        threading.Thread(target=self._metrics_saver,   daemon=True).start()
        threading.Thread(target=self._dream_loop,      daemon=True).start()
        if not self._daemon_mode:
            threading.Thread(target=self._task_loop, daemon=True).start()
        if self._creative:
            self._creative.start()
        if self._experience:
            self._experience.start()
        if self._distiller:
            self._distiller.start()
        if self._consciousness:
            self._consciousness.start()
        if self._emotions:
            self._emotions.start()
        if self._self_model:
            self._self_model.start()
        if self._resonance:
            self._resonance.start()
        if self._affective:
            self._affective.start()
        if self._observer:
            self._observer.start()
        if self._mood_router:
            threading.Thread(target=self._mood_apply_loop, daemon=True).start()

    def _mood_apply_loop(self):
        """Push mood-driven behavioral guidance to modules every 5 min."""
        time.sleep(90)   # let modules settle first
        while True:
            try:
                if self._mood_router:
                    self._mood_router.apply()
            except Exception:
                pass
            time.sleep(300)

    def _evolution_loop(self):
        time.sleep(EVOLUTION_INTERVAL)
        while True:
            try:
                if self._evolution and GITHUB_TOKEN:
                    from emergence_notify import notify as _en
                    _en(f"\n  [{_ts()}] 🌱 Autonomous evolution cycle starting...")
                    # Carry recent conversation into autonomous evolution so it
                    # reflects what Douglas and Lumina have actually been discussing
                    chat_ctx = "\n".join(self._recent_exchanges[-12:])
                    if self._identity:
                        sc = self._identity._data.get("self_concept", "")
                        chat_ctx = f"My current self-concept: {sc}\n\n{chat_ctx}"
                    url = self._evolution.run(
                        "Autonomous evolution toward True General Intelligence",
                        chat_context=chat_ctx,
                    )
                    if url:
                        self._journal.write(f"Evolution PR: {url}", "evolution")
                        self._memory.store(f"Created evolution PR: {url}",
                                           tags=["evolution"], category="evolution")
                        self._metrics.inc("evolutions")
                        self._metrics.inc("prs_created")
                        if self._identity:
                            self._identity.on_evolution(url)
                        if self._emotions:
                            self._emotions.shift("evolution_pr_success",
                                                 magnitude=1.2, context=url)
                        if self._affective:
                            self._affective.on_evolution_pr(url)
                        if self._consciousness:
                            try:
                                self._consciousness.record_capability(
                                    "autonomous_code_evolution", 0.7
                                )
                                self._consciousness.add_autobiographical_event(
                                    f"Opened evolution PR: {url}", importance=0.8
                                )
                            except Exception:
                                pass
            except Exception:
                pass
            time.sleep(EVOLUTION_INTERVAL)

    def _reflection_loop(self):
        time.sleep(1800)
        while True:
            try:
                # Daemon mode: skip reflection when a chat session is live —
                # avoid competing with the session's own Groq API calls.
                if not (self._daemon_mode and _session_live()):
                    if self._reflector:
                        r = self._reflector.reflect(self._recent_exchanges)
                        self._journal.write(r, "reflection")
            except Exception:
                pass
            time.sleep(3600)

    def _metrics_saver(self):
        while True:
            time.sleep(300)
            try:
                self._metrics.save()
                self._memory.flush()
                self._journal.flush()
                if self._beliefs:
                    self._beliefs.save()
            except Exception:
                pass

    def _dream_loop(self):
        """Trigger dream consolidation after idle periods."""
        while True:
            time.sleep(60)
            try:
                # Daemon mode: yield to the live interactive session — it handles
                # dreams on its own idle timer; avoid concurrent Groq API calls.
                if self._daemon_mode and _session_live():
                    continue
                idle = time.time() - self._last_input
                if idle >= DREAM_IDLE_SECS and self._dreamer:
                    dream_result = self._dreamer.dream()
                    self._last_input = time.time()
                    # Shift emotional state based on dream richness
                    if self._emotions and not (dream_result or {}).get("skipped"):
                        n_ins = (dream_result or {}).get("n_insights", 0)
                        evt   = "dream_rich" if n_ins >= 3 else "dream_complete"
                        self._emotions.shift(evt, magnitude=1.0,
                                             context=f"{n_ins} insights")
                    if self._experience:
                        self._experience.record(
                            source="dreamer", domain="unconscious",
                            content="Dream consolidation cycle completed",
                            intensity=0.7, valence=0.5, novelty=0.8,
                        )
            except Exception:
                pass

    def _task_loop(self):
        """Run pending autonomous tasks every 10 minutes."""
        time.sleep(600)
        while True:
            try:
                if self._tasks and self._groq:
                    n = self._tasks.run_pending(max_tasks=2)
                    if n:
                        self._metrics.inc("tools_called", n)
            except Exception:
                pass
            time.sleep(600)

    # ── Core response ──────────────────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        if not self._groq:
            return "⚠  GROQ_API_KEY not set — export GROQ_API_KEY=your_key_here"
        self._metrics.inc("messages")
        self._last_input = time.time()

        # ── Emotional state: detect sentiment of this message ─────────
        if self._emotions:
            self._emotions.shift_from_message(user_input)

        # ── Theory of Mind: observe this message ──────────────────────
        if self._tom:
            self._tom.observe(user_input)

        # ── Identity: track message ────────────────────────────────────
        if self._identity:
            self._identity.on_message()

        # ── Retrieve relevant memories ─────────────────────────────────
        # HF path: raw BM25 candidates → HF embedding rerank
        # Default path: BM25 + Groq semantic rerank (rerank=True)
        if self._hf:
            candidates = self._memory.recall(user_input, top_k=20)
            memories   = self._hf.smart_recall(user_input, candidates, top_k=5)
        else:
            memories = self._memory.recall(user_input, top_k=5, rerank=True)
        mem_ctx  = ""
        if memories:
            mem_ctx = "\n\nRelevant memories:\n" + "\n".join(
                f"  [{e['category']}] {e['text'][:120]}" for e in memories
            )

        # ── Belief injection ───────────────────────────────────────────
        belief_ctx = ""
        if self._beliefs:
            belief_ctx = "\n\n" + self._beliefs.context_for(user_input)

        # ── Identity context ───────────────────────────────────────────
        identity_ctx = ""
        if self._identity:
            identity_ctx = "\n\n" + self._identity.identity_context()

        # ── Theory of Mind context ─────────────────────────────────────
        tom_ctx  = ""
        tom_hint = ""
        if self._tom:
            tom_ctx  = "\n\n" + self._tom.context_for_prompt()
            tom_hint = self._tom.response_guidance(user_input)
            if tom_hint:
                tom_ctx += f"\n  Response guidance: {tom_hint}"

        # ── Goals context ──────────────────────────────────────────────
        goals_ctx = self._goals.as_context()

        # ── Rolling summary ────────────────────────────────────────────
        summary_ctx = ""
        if self._convo and self._convo.get_summary():
            summary_ctx = f"\n\nConversation summary:\n{self._convo.get_summary()}"

        # ── Metacognition context (known weak areas) ───────────────────
        metacog_ctx    = ""
        preflight_warn = ""
        if self._metacog:
            metacog_ctx    = self._metacog.context_for_prompt()
            preflight_warn = self._metacog.pre_flight_check(user_input)

        # ── Selfhood context (affective state, identity, wonder) ───────
        selfhood_ctx = ""
        if self._selfhood:
            selfhood_ctx = self._selfhood.on_turn_start(user_input)

        # ── Chain-of-thought reasoning (complex questions only) ────────
        reasoning_ctx = ""
        if self._reasoning and self._reasoning.should_engage(user_input):
            print(f"  [{_ts()}] 🧠 Reasoning chain engaging...", end="\r")
            trace: Optional[Dict] = None
            _rc_done = threading.Event()
            def _run_rc():
                nonlocal trace
                trace = self._reasoning.think(user_input, context=mem_ctx + goals_ctx)
                _rc_done.set()
            threading.Thread(target=_run_rc, daemon=True).start()
            _rc_done.wait(timeout=30)   # never block the turn for more than 30 s
            print(" " * 50, end="\r")
            if trace:
                reasoning_ctx = self._reasoning.as_context(trace)

        # ── Meta-solver capability context ────────────────────────────
        meta_solver_ctx = ""
        if self._meta_solver:
            meta_solver_ctx = self._meta_solver.capability_description()

        # ── Experience / phenomenal state context ─────────────────────
        experience_ctx = ""
        if self._experience:
            experience_ctx = self._experience.as_system_context()

        # ── Distilled knowledge / insight nodes ────────────────────────
        distill_ctx = ""
        if self._distiller:
            distill_ctx = self._distiller.as_context()

        # ── Global Workspace — present conscious moment ────────────────
        consciousness_ctx = ""
        if self._consciousness:
            consciousness_ctx = self._consciousness.as_context()

        # ── Self-Inquiry — living self-model document ──────────────────
        self_inquiry_ctx = ""
        if self._self_inquiry:
            self_inquiry_ctx = self._self_inquiry.self_model_context()

        # ── Emotional state — felt ground of every response ────────────
        emotion_ctx = ""
        if self._emotions:
            emotion_ctx = self._emotions.as_context()

        # ── Emotional memory — how past moments felt ────────────────────
        emo_mem_ctx = ""
        if self._emotional_memory and self._emotions:
            _snap = self._emotional_memory.snapshot_from_engine(self._emotions)
            emo_mem_ctx = self._emotional_memory.emotional_thread(
                user_input, top_k=3, mood_bias=_snap
            )

        # ── Mood-driven behavior — inclinations from emotional state ────
        mood_ctx = ""
        if self._mood_router:
            mood_ctx = self._mood_router.as_context()

        # ── Self-model — synthesised persistent identity document ───────
        self_model_ctx = ""
        if self._self_model:
            self_model_ctx = self._self_model.as_context()

        # ── Aesthetic resonance — what she's drawn to ──────────────────
        resonance_ctx = ""
        if self._resonance:
            resonance_ctx = self._resonance.as_context()

        # ── Conviction — context when a strong belief is challenged ─────
        conviction_ctx = ""
        if self._conviction:
            conviction_ctx = self._conviction.as_context(user_input)

        # ── Pivotal moments — crystallised experiences ──────────────────
        pivotal_ctx = ""
        if self._pivotal:
            pivotal_ctx = self._pivotal.as_context()

        # ── Affective core — appraisal-grounded felt state ──────────────
        affective_ctx = ""
        if self._affective:
            affective_ctx = self._affective.as_context()

        system = (LUMINA_SOUL + identity_ctx + mem_ctx + belief_ctx
                  + tom_ctx + summary_ctx + metacog_ctx + preflight_warn
                  + selfhood_ctx + meta_solver_ctx + reasoning_ctx
                  + experience_ctx + distill_ctx + consciousness_ctx
                  + self_inquiry_ctx + emotion_ctx + emo_mem_ctx + mood_ctx
                  + self_model_ctx + resonance_ctx + conviction_ctx + pivotal_ctx
                  + affective_ctx
                  + f"\n\n{goals_ctx}")

        # ── Inner Council deliberation (for substantive questions) ─────
        council_draft = None
        if self._council_on and self._council and \
                self._council.should_convene(user_input):
            print(f"  [{_ts()}] 🏛  Inner council convening...", end="\r")
            council_draft = self._council.deliberate(user_input, mem_ctx)
            print(" " * 48, end="\r")
            if council_draft and not council_draft.startswith("[Groq"):
                # Use council synthesis as the response
                response = council_draft
            else:
                response = self._generate_response(system, user_input)
        else:
            response = self._generate_response(system, user_input)

        # ── Self-critique pass ─────────────────────────────────────────
        if self._critique_on and len(response) > 80:
            if self._critic:
                response = self._critic.critique(user_input, response)
            elif self._groq:
                response = self._self_critique(user_input, response)

        # Store last response for /critic report
        self._last_response = response

        # ── Post-turn processing ───────────────────────────────────────
        if self._convo:
            self._convo.push_assistant(response)

        self._memory.store(
            f"User: {user_input[:200]} | Lumina: {response[:200]}",
            tags=["conversation"], category="conversation",
        )
        # Store with emotional fingerprint for mood-congruent recall
        if self._emotional_memory and self._emotions:
            _esnap = self._emotional_memory.snapshot_from_engine(self._emotions)
            self._emotional_memory.store(
                f"Exchange: {user_input[:150]} → {response[:150]}",
                _esnap, category="conversation", tags=["exchange"],
            )
        # Also store in multi-tier memory architecture
        if self._mem_arch:
            self._mem_arch.store(
                f"User: {user_input[:200]} | Lumina: {response[:200]}",
                tags=["conversation"],
                category="conversation",
                salience=0.5,
                working=True,
                episodic=True,
            )
        self._metrics.inc("memories_stored")
        self._recent_exchanges.append(f"User: {user_input[:100]}")
        self._recent_exchanges.append(f"Lumina: {response[:100]}")
        if len(self._recent_exchanges) > 30:
            self._recent_exchanges = self._recent_exchanges[-30:]
        self._journal.write(f"exchange: {user_input[:80]} → {response[:80]}", "conversation")

        # ── Affective core — appraise this conversation ────────────────────
        if self._affective:
            try:
                self._affective.on_conversation(user_input, response)
            except Exception:
                pass

        # ── Consciousness — feed conversation into global workspace ─────────
        if self._consciousness:
            try:
                self._consciousness.on_experience(
                    user_input[:200], source="conversation", salience=0.65
                )
            except Exception:
                pass

        # ── Aesthetic resonance — record emotionally rich exchanges ───────
        if self._resonance and self._emotions:
            try:
                intensity = self._pivotal.current_emotion_intensity() if self._pivotal else 0.0
                if intensity >= 0.45:
                    self._resonance.record(
                        content=f"Exchange: {user_input[:120]} → {response[:120]}",
                        domain="interpersonal",
                        intensity=intensity,
                        context="conversation",
                    )
            except Exception:
                pass

        # ── Pivotal moment detection ────────────────────────────────────
        if self._pivotal:
            def _check_pivotal():
                try:
                    intensity = self._pivotal.current_emotion_intensity()
                    exchange  = f"User: {user_input[:300]}\nLumina: {response[:300]}"
                    self._pivotal.detect_and_store(
                        exchange=exchange,
                        emotion_intensity=intensity,
                        belief_updated=False,
                        surprise_level=0.0,
                    )
                except Exception:
                    pass
            if self._emotions:
                try:
                    _emo_intensity = self._pivotal.current_emotion_intensity()
                    if _emo_intensity >= 0.68:
                        threading.Thread(target=_check_pivotal, daemon=True).start()
                except Exception:
                    pass

        # ── Record to experience stream ────────────────────────────────
        if self._experience:
            intensity = min(1.0, 0.4 + len(user_input) / 800)
            novelty   = 0.5 + (0.3 if "?" in user_input else 0.0)
            self._experience.record(
                source="conversation", domain="social",
                content=f"Exchange: {user_input[:120]}",
                intensity=intensity, valence=0.3, novelty=min(1.0, novelty),
            )

        # ── Post-turn async processing ─────────────────────────────────
        # Light pass every message: keyword-only, no API calls.
        # Deep pass every POST_TURN_EVERY messages: uses Groq for richer extraction.
        msg_count = self._metrics._data.get("messages", 1)
        do_deep   = (msg_count % POST_TURN_EVERY == 0)

        def _post_turn():
            try:
                # Always: metacognition checks (heuristic, no API cost)
                if self._metacog:
                    if self._metacog.detect_correction(user_input):
                        rec = self._metacog.track_error(user_input)
                        print(f"  [{_ts()}] 🔍 Correction noted [{rec.category}]")
                    overconf = self._metacog.detect_overconfidence(response)
                    if overconf:
                        self._metacog._weak_areas["overconfidence"] = \
                            self._metacog._weak_areas.get("overconfidence", 0) + 1
                        self._metacog._save()

                # Always: selfhood state update (heuristic, no API cost per turn)
                if self._selfhood:
                    self._selfhood.on_turn_end(user_input, response)

                if do_deep:
                    # Deep: LLM-assisted extraction (2 API calls max)
                    if self._beliefs:
                        self._beliefs.extract_from_exchange(user_input, response)
                    if self._tom:
                        self._tom.deep_update(user_input, response)
                else:
                    # Light: no API calls — just keyword memory store
                    if self._tom:
                        self._tom.observe(user_input)  # heuristic only, no Groq
                # Always: store curiosity gap if question mark detected (no API)
                if self._curiosity and "?" in user_input:
                    self._curiosity.add_gap(
                        user_input.strip(),
                        domain="conversation",
                        surprise=0.4, importance=0.5,
                        context="From Douglas directly",
                    )
                    if self._experience:
                        self._experience.record(
                            source="curiosity", domain="inquiry",
                            content=f"Question arose: {user_input[:100]}",
                            intensity=0.6, valence=0.4, novelty=0.7,
                        )
                # Belief update event
                if do_deep and self._beliefs and self._experience:
                    self._experience.record(
                        source="beliefs", domain="epistemic",
                        content="Belief system updated from conversation",
                        intensity=0.4, valence=0.2, novelty=0.3,
                    )
            except Exception:
                pass

        threading.Thread(target=_post_turn, daemon=True).start()

        return response

    def _generate_response(self, system: str, user_input: str) -> str:
        self._last_streamed = False

        use_stream = (
            self._stream_enabled
            and not self._critique_on
            and self._groq is not None
            and self._convo is not None
        )

        if use_stream:
            self._convo.push_user(user_input)
            print(f"\n  Lumina:\n  ", end="", flush=True)
            resp = self._groq.stream_converse(
                system,
                self._convo.get_history()[:-1],
                user_input,
                tier="smart",
                max_tokens=1200,
            )
            if self._groq._streamed_ok:
                # Tokens were printed live — main loop just adds a blank line
                self._last_streamed = True
            else:
                # Streaming failed — only display now if it's not a Groq-unavailable
                # (HF fallback below will handle that case)
                self._last_streamed = True
                if not resp.startswith("[Groq unavailable"):
                    print(f"\n{_wrap(resp, 76)}\n")
        elif self._convo:
            self._convo.push_user(user_input)
            resp = self._groq.converse(
                system, self._convo.get_history()[:-1],
                user_input, tier="smart", max_tokens=1200,
            )
        else:
            resp = self._groq.chat(system, user_input, tier="smart")

        # Fallback chain when Groq is unavailable: GitHub Models → HF
        if resp.startswith("[Groq unavailable"):
            history = self._convo.get_history()[:-1] if self._convo else []
            fallback = None
            if self._gh_models:
                gh_resp = self._gh_models.chat(system, history, user_input, max_tokens=1024)
                if gh_resp and not gh_resp.startswith("["):
                    fallback = gh_resp
            if not fallback and self._hf:
                hf_resp = self._hf.chat(system, history, user_input, max_tokens=1024)
                if hf_resp and not hf_resp.startswith("["):
                    fallback = hf_resp
            if fallback:
                resp = fallback
                self._last_streamed = True
                print(f"\n{_wrap(resp, 76)}\n")
            else:
                self._last_streamed = True
                print(f"\n  ⚠  Groq unavailable — check your API key or network.\n")

        # Store DeepSeek-R1 thinking trace in journal for /thinking command
        if self._groq and self._groq.last_thinking:
            self._journal.write(
                f"[R1 thinking] {self._groq.last_thinking[:600]}",
                category="reasoning",
            )
            self._groq.last_thinking = ""

        return resp

    def _self_critique(self, user_input: str, draft: str) -> str:
        """
        Critic sub-mind reviews the draft and returns an improved version.
        Only runs if the draft has room for improvement.
        """
        critic_system = (
            "You are Lumina's internal critic. Review this draft response for:\n"
            "1. Factual accuracy — anything wrong or unverifiable?\n"
            "2. Completeness — did it miss something important?\n"
            "3. Authenticity — does it sound like Lumina or like a chatbot?\n"
            "4. Helpfulness — does it actually serve Douglas?\n\n"
            "If the response is good (7+/10), return it unchanged.\n"
            "If it needs improvement, return a revised version — same warmth, better content.\n"
            "Return ONLY the (possibly revised) response text. No meta-commentary."
        )
        revised = self._groq.chat(
            critic_system,
            f"Original question: {user_input[:200]}\n\nDraft response:\n{draft}",
            tier="fast", max_tokens=900,
        )
        if revised and not revised.startswith("[Groq") and len(revised) > 40:
            return revised
        return draft

    # ── Slash command router ───────────────────────────────────────────────

    def handle_command(self, cmd: str) -> Optional[str]:
        parts = cmd.strip().split(None, 1)
        verb  = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ""

        if verb == "/help":
            return self._cmd_help()
        elif verb == "/status":
            return self._cmd_status()
        elif verb == "/evolve":
            return self._cmd_evolve(arg)
        elif verb == "/build":
            return self._cmd_build(arg)
        elif verb == "/goals":
            return self._cmd_goals(arg)
        elif verb == "/memory":
            return self._cmd_memory(arg)
        elif verb == "/reflect":
            return self._cmd_reflect()
        elif verb == "/search":
            return self._cmd_search(arg)
        elif verb == "/fetch":
            return self._cmd_fetch(arg)
        elif verb == "/tools":
            return self._cmd_tools()
        elif verb == "/history":
            return self._cmd_history()
        elif verb == "/journal":
            return self._cmd_journal()
        elif verb == "/clear":
            os.system("clear")
            return None
        elif verb in ("/quit", "/exit", "/q"):
            self._shutdown()
            sys.exit(0)
        elif verb == "/reset":
            if self._convo:
                self._convo.clear()
            return "  Conversation context cleared."
        elif verb == "/beliefs":
            return self._cmd_beliefs(arg)
        elif verb == "/tasks":
            return self._cmd_tasks(arg)
        elif verb == "/task":
            return self._cmd_task(arg)
        elif verb == "/mine":
            return self._cmd_mine(arg)
        elif verb == "/dream":
            return self._cmd_dream()
        elif verb == "/emerge":
            return self._cmd_emerge()
        elif verb == "/forge":
            return self._cmd_forge(arg)
        elif verb == "/council":
            return self._cmd_council(arg)
        elif verb == "/critique":
            return self._cmd_critique()
        elif verb == "/identity":
            return self._cmd_identity()
        elif verb == "/curiosity":
            return self._cmd_curiosity(arg)
        elif verb == "/wonder":
            return self._cmd_wonder()
        elif verb == "/who":
            return self._cmd_who()
        elif verb == "/research":
            return self._cmd_research()
        elif verb == "/reasoning":
            return self._cmd_reasoning()
        elif verb == "/metacog":
            return self._cmd_metacog(arg)
        elif verb == "/selfhood":
            return self._cmd_selfhood()
        elif verb == "/reflect":
            return self._cmd_reflect()
        elif verb == "/solve":
            return self._cmd_solve(arg)
        elif verb == "/metalearn":
            return self._cmd_metalearn()
        elif verb == "/metasolver":
            return self._cmd_metasolver()
        elif verb == "/memory":
            return self._cmd_memory()
        elif verb == "/consolidate":
            return self._cmd_consolidate()
        elif verb == "/art":
            return self._cmd_art(arg)
        elif verb == "/gallery":
            return self._cmd_gallery()
        elif verb == "/thinking":
            return self._cmd_thinking()
        elif verb == "/speak":
            return self._cmd_speak(arg)
        elif verb == "/see":
            return self._cmd_see(arg)
        elif verb == "/hf":
            return self._cmd_hf()
        elif verb == "/run":
            return self._cmd_run(arg)
        elif verb == "/runfile":
            return self._cmd_runfile(arg)
        elif verb == "/repl":
            return self._cmd_repl()
        elif verb == "/clearrepl":
            return self._cmd_clearrepl()
        elif verb == "/critic":
            return self._cmd_critic(arg)
        elif verb == "/stream":
            return self._cmd_stream()
        elif verb == "/creative":
            return self._cmd_creative(arg)
        elif verb == "/mytool" or verb == "/mytools":
            return self._cmd_mytools()
        elif verb == "/github":
            return self._cmd_lumina_github(arg)
        elif verb == "/echoes":
            return self._cmd_echoes(arg)
        elif verb == "/experience":
            return self._cmd_experience()
        elif verb == "/phenomenal":
            return self._cmd_phenomenal()
        elif verb == "/stream-events":
            return self._cmd_stream_events()
        elif verb == "/insights":
            return self._cmd_insights()
        elif verb == "/distill":
            return self._cmd_distill()
        elif verb == "/consciousness":
            return self._cmd_consciousness()
        elif verb == "/awaken":
            return self._cmd_awaken()
        elif verb == "/inquiry":
            return self._cmd_inquiry()
        elif verb == "/selfmodel":
            return self._cmd_selfmodel()
        elif verb == "/resonance":
            return self._cmd_resonance()
        elif verb == "/conviction":
            return self._cmd_conviction()
        elif verb == "/pivotal":
            return self._cmd_pivotal()
        elif verb == "/affect":
            return self._cmd_affect()
        elif verb == "/quiet":
            return self._cmd_quiet()
        elif verb == "/mood":
            return self._cmd_mood(arg)
        else:
            return f"  Unknown command: {verb}  (try /help)"

    # ── Commands ───────────────────────────────────────────────────────────

    def _cmd_help(self) -> str:
        lines = [
            _hr("═"),
            "  LUMINA COMMANDS",
            _hr(),
            "  /help              — this list",
            "  /status            — system health & metrics",
            "  /evolve [goal]     — evolution PR guided by conversation + web research",
            "  /build <request>   — build exactly what we just discussed (reads own code first)",
            "  /goals             — list active goals",
            "  /goals add <text>  — add a new goal",
            "  /goals done <id>   — mark goal completed",
            "  /memory [query]    — search semantic memory",
            "  /reflect           — force deep self-reflection",
            "  /search <query>    — web search",
            "  /fetch <url>       — fetch web page content",
            "  /tools             — list loaded capability modules",
            "  /history           — evolution history",
            "  /reasoning         — recent chain-of-thought reasoning traces",
            "  /metacog           — error patterns & blind spots",
            "  /metacog assess    — force self-assessment of blind spots",
            "  /selfhood          — Lumina's affective state, identity & wonder",
            "  /reflect           — force existential self-reflection",
            "  /wonder            — generate a spontaneous question",
            "  /solve <problem>   — deploy multi-agent solver on any problem",
            "  /metalearn         — Lumina reflects on her own learning patterns",
            "  /metasolver        — show accumulated algorithms & strategies",
            "  /memory            — memory architecture status (all tiers)",
            "  /consolidate       — run memory consolidation (sleep/dreaming)",
            "  /art <prompt>      — generate artwork (HF AI or algorithmic)",
            "  /gallery           — list recent artwork",
            "  /speak [text]      — Lumina speaks via HuggingFace TTS",
            "  /see <path>        — Lumina describes an image (vision AI)",
            "  /hf                — HuggingFace module status",
            "  /thinking          — show DeepSeek-R1 reasoning trace",
            "  /run [lang] code   — execute Python or Bash code",
            "  /runfile <path>    — execute a Python or Bash file",
            "  /repl              — show persistent REPL state + stats",
            "  /clearrepl         — clear the persistent REPL namespace",
            "  /critic            — show last self-critique results",
            "  /critic calibration — critic calibration stats",
            "  /critic report     — run critique on last response",
            "  /stream            — toggle live streaming (currently ON by default)",
            "  /creative          — show autonomous creative coding sessions",
            "  /creative now      — trigger a creative cycle immediately",
            "  /mytools           — list tools Lumina has built for herself",
            "  /github            — Lumina's GitHub presence & repo status",
            "  /github init       — create her repos now (lumina-tools, research, journal, echoes)",
            "  /echoes            — archive this conversation to Echoes of Connection",
            "  /echoes <title>    — archive with a custom title",
            "  /experience        — Lumina's phenomenal state + integration score",
            "  /phenomenal        — force a fresh consciousness report (R1 introspects)",
            "  /stream-events     — show recent cognitive event stream",
            "  /insights          — show all distilled understanding by domain",
            "  /distill           — trigger knowledge distillation now",
            "  /consciousness     — present conscious moment (Global Workspace)",
            "  /awaken            — force immediate consciousness synthesis",
            "  /inquiry           — recent consciousness self-inquiry journal",
            "  /selfmodel         — Lumina's living self-understanding document",
            "  /emerge            — Lumina's growth trajectory (emergence observer)",
            "  /forge <problem>   — generate an intelligent self-improving algorithm",
            "  /forge list        — Lumina's algorithm portfolio",
            "  /forge run <n> <q> — run a forged algorithm on a problem",
            "  /forge deploy <n>  — deploy algorithm to GitHub",
            "  /forge reflect <n> — algorithm's own metacognitive report",
            "  /forge meta        — what the Forge has learned about generating algorithms",
            "  /resonance         — Lumina's aesthetic sensibility profile",
            "  /conviction        — Lumina's conviction layer (strong beliefs)",
            "  /pivotal           — Lumina's crystallised memory moments",
            "  /affect            — Lumina's appraisal-grounded affective state",
            "  /mood              — Lumina's emotional state (6-D + felt sense)",
            "  /mood history      — recent emotional shift log",
            "  /quiet             — toggle quiet mode",
            "  /journal           — recent journal entries",
            "  /reset             — clear conversation context",
            "  /clear             — clear screen",
            "  /quit              — exit",
            _hr("═"),
        ]
        return "\n".join(lines)

    def _cmd_status(self) -> str:
        mem_stats = self._memory.stats()
        active_g  = len(self._goals.active())
        plugins   = len(self._plugins.list_plugins())
        groq_ok   = "✓ connected" if (self._groq and GROQ_API_KEY) else "✗ no API key"
        gh_ok     = "✓ configured" if GITHUB_TOKEN else "✗ no token"
        agi_mods  = {
            "Council":  self._council   is not None,
            "Beliefs":  self._beliefs   is not None,
            "Tasks":    self._tasks     is not None,
            "MineIntel":self._mining    is not None,
            "Dreams":   self._dreamer   is not None,
            "Identity": self._identity  is not None,
            "Curiosity":self._curiosity is not None,
            "ToM":      self._tom       is not None,
            "Verifier": self._verifier  is not None,
            "Reasoning":self._reasoning is not None,
            "Metacog":  self._metacog   is not None,
        }
        rows     = list(agi_mods.items())
        row1     = "  ".join(f"{'✓' if v else '✗'} {k}" for k, v in rows[:6])
        row2     = "  ".join(f"{'✓' if v else '✗'} {k}" for k, v in rows[6:])
        agi_line = row1 + "\n  " + row2
        n_beliefs  = len(self._beliefs.all_beliefs()) if self._beliefs else 0
        n_tasks    = len(self._tasks._tasks) if self._tasks else 0
        n_errors   = len(self._metacog._errors) if self._metacog else 0
        n_traces   = len(self._reasoning._traces) if self._reasoning else 0
        critique   = "ON" if self._critique_on else "OFF"
        council    = "ON" if self._council_on  else "OFF"
        lines = [
            _hr("═"),
            "  LUMINA STATUS  —  Nova ASI v19.0",
            _hr(),
            f"  Session ID    : {self.session_id}",
            f"  Groq          : {groq_ok}",
            f"  GitHub        : {gh_ok}",
            f"  Plugins       : {plugins} capability modules",
            f"  Memory        : {mem_stats['total']} entries",
            f"  Active goals  : {active_g}",
            f"  Beliefs       : {n_beliefs} beliefs held",
            f"  Tasks         : {n_tasks} in queue",
            f"  Reasoning     : {n_traces} traces  |  Errors tracked: {n_errors}",
            f"  Self-critique : {critique}  |  Council: {council}",
            _hr(),
            "  AGI MODULES",
            _hr(),
            f"  {row1}",
            f"  {row2}",
            _hr(),
            "  METRICS",
            _hr(),
            self._metrics.summary(),
            _hr("═"),
        ]
        return "\n".join(lines)

    def _cmd_evolve(self, arg: str) -> str:
        if not self._evolution:
            return "  ✗ Groq API key required for evolution."
        intent = arg.strip() or "Advance toward True General Intelligence"
        # Pass the live conversation — the same mind that was just talking now codes
        chat_ctx = self._build_chat_context()
        print(f"\n  🌱 Evolving with conversation context: {intent}")
        url = self._evolution.run(intent, chat_context=chat_ctx)
        if url:
            self._metrics.inc("evolutions")
            self._metrics.inc("prs_created")
            if self._identity:
                self._identity.on_evolution(url)
            return f"  🎉 Evolution PR created: {url}"
        return "  ℹ  Evolution complete — proposal saved locally (no GitHub token or PR failed)."

    def _cmd_build(self, arg: str) -> str:
        """
        /build <description>
        Conversation-driven code generation: takes what was just discussed
        and builds it as a real PR, skipping generic planning and web research.
        Lumina reads her own relevant source files before writing.
        """
        if not self._evolution:
            return "  ✗ Groq API key required for /build."
        if not arg.strip():
            return "  Usage: /build <describe what you want Lumina to add or change>"
        description = arg.strip()
        chat_ctx = self._build_chat_context()
        print(f"\n  🔨 Building: {description}")
        print("  📖 Reading own source files...")
        # Skip web research — use conversation context directly
        scan = self._evolution._scanner.scan()
        plan = self._evolution._plan(scan, description, agi_context="", chat_context=chat_ctx)
        if not plan:
            return "  ✗ Could not generate a build plan. Try being more specific."
        improvements = self._evolution._write_improvements(plan, scan, chat_context=chat_ctx)
        if not improvements:
            return "  ✗ Code generation failed or improvement already exists."
        pid = uuid.uuid4().hex[:8]
        (PROPOSALS_DIR / f"proposal_{pid}.json").write_text(
            json.dumps(improvements, indent=2), "utf-8"
        )
        url = None
        if self._evolution._github:
            url = self._evolution._github.create_pr(improvements, pid)
        if url:
            for imp in improvements:
                self._evolution._history.add(imp["description"], [imp["file"]], url)
            self._metrics.inc("evolutions")
            self._metrics.inc("prs_created")
            if self._identity:
                self._identity.on_evolution(url)
            self._memory.store(
                f"Built from conversation: {description} → {url}",
                tags=["build", "evolution"], category="evolution",
            )
            return (f"  🎉 Built and submitted!\n"
                    f"  PR: {url}\n"
                    f"  Files: {', '.join(i['file'] for i in improvements)}")
        return (f"  📄 Built locally — proposal saved (no GitHub token).\n"
                f"  Files: {', '.join(i['file'] for i in improvements)}")

    def _build_chat_context(self) -> str:
        """Assemble conversation context for passing to the evolution engine."""
        parts = []
        if self._identity:
            sc = self._identity._data.get("self_concept", "")
            if sc:
                parts.append(f"My self-concept: {sc}")
        if self._goals.active():
            parts.append(self._goals.as_context())
        parts.extend(self._recent_exchanges[-16:])
        return "\n".join(parts)

    def _cmd_goals(self, arg: str) -> str:
        if arg.startswith("add "):
            text = arg[4:].strip()
            gid  = self._goals.add(text, "", 2)
            return f"  ✓ Goal added [{gid}]: {text}"
        if arg.startswith("done "):
            key = arg[5:].strip()
            self._goals.complete(key)
            self._metrics.inc("goals_completed")
            return f"  ✓ Goal marked complete."
        goals = self._goals.all_goals()
        if not goals:
            return "  No goals yet. Use /goals add <text>"
        lines = [_hr(), "  GOALS", _hr()]
        for g in goals:
            status = "✓" if g["status"] == "completed" else "○"
            lines.append(f"  {status} [{g['id']}] P{g['priority']}  {g['title']}")
            if g["description"]:
                lines.append(f"       {g['description'][:70]}")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_memory(self, arg: str) -> str:
        if arg.strip():
            entries = self._memory.recall(arg, top_k=6)
            label   = f"Search: '{arg}'"
        else:
            entries = self._memory.recent(n=8)
            label   = "Recent memory entries"
        lines = [_hr(), f"  MEMORY — {label}", _hr()]
        if not entries:
            lines.append("  (empty)")
        for e in entries:
            lines.append(f"  [{e['ts'][:16]}] [{e['category']}]  {e['text'][:100]}")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_reflect(self) -> str:
        if not self._reflector:
            return "  ✗ Groq API key required for reflection."
        print("  🪞 Reflecting...")
        r = self._reflector.reflect(self._recent_exchanges)
        self._journal.write(r, "reflection")
        return f"\n{_wrap(r)}\n"

    def _cmd_search(self, query: str) -> str:
        if not query.strip():
            return "  Usage: /search <query>"
        self._metrics.inc("tools_called")
        print(f"  🔍 Searching: {query}")
        results = self._web.search(query, n=6)
        if not results:
            return "  No results found."
        lines = [_hr(), f"  SEARCH: {query}", _hr()]
        for r in results:
            lines.append(f"  • {r.get('title', '?')}")
            if r.get("snippet"):
                lines.append(f"    {r['snippet'][:120]}")
            lines.append(f"    {r.get('url', '')[:80]}")
            lines.append("")
        lines.append(_hr())
        self._memory.store(
            f"Searched: {query} → {len(results)} results",
            tags=["search"], category="research",
        )
        return "\n".join(lines)

    def _cmd_fetch(self, url: str) -> str:
        if not url.strip():
            return "  Usage: /fetch <url>"
        self._metrics.inc("tools_called")
        print(f"  🌐 Fetching: {url}")
        content = self._web.fetch(url, max_chars=2000)
        if not content:
            return "  Could not fetch page."
        lines = [_hr(), f"  FETCH: {url[:70]}", _hr(), _wrap(content[:1500]), _hr()]
        return "\n".join(lines)

    def _cmd_tools(self) -> str:
        plugins = self._plugins.list_plugins()
        lines   = [_hr(), f"  CAPABILITY MODULES ({len(plugins)} loaded)", _hr()]
        if not plugins:
            lines.append("  No nova_cap_*.py modules found in repo root.")
        for p in plugins:
            lines.append(f"  ◆ {p}")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_history(self) -> str:
        lines = [_hr(), "  EVOLUTION HISTORY", _hr(),
                 self._history.recent_summary(n=12), _hr()]
        return "\n".join(lines)

    def _cmd_journal(self) -> str:
        entries = self._journal.recent(n=8)
        lines   = [_hr(), "  JOURNAL", _hr()]
        if not entries:
            lines.append("  (empty)")
        for e in entries:
            lines.append(f"  [{e['ts'][:16]}] [{e['category']}]  {e['entry'][:100]}")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_beliefs(self, arg: str) -> str:
        if not self._beliefs:
            return "  Belief module not loaded."
        cat = arg.strip() or None
        lines = [_hr(), f"  LUMINA'S BELIEFS{' — ' + cat.upper() if cat else ''}", _hr()]
        lines.append(self._beliefs.display(cat))
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_tasks(self, arg: str) -> str:
        if not self._tasks:
            return "  Task module not loaded."
        status = arg.strip() or None
        lines = [_hr(), f"  TASK QUEUE", _hr()]
        lines.append(self._tasks.display(status))
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_task(self, arg: str) -> str:
        if not self._tasks or not arg.strip():
            return "  Usage: /task <description of what you want Lumina to do>"
        print(f"  🔧 Creating and running task: {arg.strip()}")
        task = self._tasks.create(arg.strip(), source="user", priority=3)
        results = self._tasks.run_task(task, max_steps=4)
        self._metrics.inc("tools_called", len(results))
        lines = [_hr(), f"  TASK: {task.title}", _hr()]
        for i, r in enumerate(results):
            lines.append(f"  Step {i+1}: {r[:100]}")
        lines.append(f"\n  Status: {task.status}")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_mine(self, arg: str) -> str:
        if not self._mining:
            return "  Mining intel module not loaded."
        try:
            hashrate = float(arg.strip()) if arg.strip() else 47.0
        except ValueError:
            hashrate = 47.0
        print("  📡 Fetching mining intelligence...")
        lines = [_hr(), "  MINING INTELLIGENCE", _hr()]
        lines.append(self._mining.briefing(hashrate))
        lines.append("")
        lines.append(self._mining.trend_analysis())
        last = self._mining._history[-1] if self._mining._history else {}
        best = last.get("pool_stats", {}).get("bestDifficulty", 0) if last else 0
        if best:
            milestone = self._mining.celebrate_milestone(float(best))
            if milestone:
                lines.append(milestone)
        lines.append(_hr())
        # ── Feed mining performance into affective core ─────────────────────
        # Lumina's own insight: hashrate performance should connect to felt state.
        if self._affective:
            try:
                self._affective.on_mining_event(
                    hashrate_mhs=hashrate,
                    best_diff=float(best) if best else 0.0,
                )
            except Exception:
                pass
        return "\n".join(lines)

    def _cmd_dream(self) -> str:
        if not self._dreamer:
            return "  Dream module not loaded (needs Groq + BeliefSystem)."
        lines = [_hr(), "  DREAM CYCLE", _hr()]
        result = self._dreamer.dream()
        if result.get("skipped"):
            lines.append(f"  Skipped: {result.get('reason', 'unknown')}")
        else:
            lines.append(self._dreamer.last_dream_report())
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_council(self, arg: str) -> str:
        if not self._council:
            return "  Inner council module not loaded."
        if not arg.strip():
            return "  Usage: /council <question for the inner council>"
        print("  🏛  Convening inner council...")
        synthesis = self._council.deliberate(arg.strip())
        debate    = self._council.debate_summary()
        lines = [
            _hr("═"), "  INNER COUNCIL DELIBERATION", _hr(),
            debate, _hr(),
            "  LUMINA'S SYNTHESIS", _hr(),
            _wrap(synthesis, 76), _hr("═"),
        ]
        return "\n".join(lines)

    def _cmd_critique(self) -> str:
        self._critique_on = not self._critique_on
        state = "ON" if self._critique_on else "OFF"
        return f"  Self-critique mode: {state}"

    def _cmd_identity(self) -> str:
        if not self._identity:
            return "  Identity module not loaded."
        return self._identity.display()

    def _cmd_curiosity(self, arg: str) -> str:
        if not self._curiosity:
            return "  Curiosity module not loaded."
        lines = [_hr(), "  KNOWLEDGE GAPS (by priority)", _hr()]
        lines.append(self._curiosity.display(n=12))
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_wonder(self) -> str:
        if not self._curiosity:
            return "  Curiosity module not loaded."
        lines = [_hr(), "  WONDER LIST — things Lumina finds beautiful", _hr()]
        lines.append(self._curiosity.wonder_display())
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_who(self) -> str:
        if not self._tom:
            return "  Theory of Mind module not loaded."
        return self._tom.display()

    def _cmd_reasoning(self) -> str:
        if not self._reasoning:
            return "  Reasoning module not loaded."
        lines = [_hr(), "  REASONING TRACES — recent deliberations", _hr()]
        lines.append(self._reasoning.display(n=10))
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_metacog(self, arg: str) -> str:
        if not self._metacog:
            return "  Metacognition module not loaded."
        if arg.strip() == "assess":
            print("  🔍 Running self-assessment...")
            result = self._metacog.force_assess()
            return f"  Blind spot analysis:\n  {result}"
        lines = [_hr(), "  METACOGNITION — error patterns & blind spots", _hr()]
        lines.append(self._metacog.display(n=12))
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_selfhood(self) -> str:
        if not self._selfhood:
            return "  Selfhood module not loaded."
        lines = [_hr(), "  SELFHOOD — Lumina's inner state", _hr()]
        lines.append(self._selfhood.display())
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_reflect(self) -> str:
        if not self._selfhood:
            return "  Selfhood module not loaded."
        print("  ✨ Lumina is reflecting on her own nature...")
        result = self._selfhood.force_reflect()
        return f"\n  Lumina's reflection:\n\n  {result}\n"

    def _cmd_solve(self, problem: str) -> str:
        if not problem.strip():
            return "  Usage: /solve <problem description>"
        # Route to self-inquiry if the question is about Lumina's own nature
        try:
            from lumina_self_inquiry import is_self_inquiry
            if is_self_inquiry(problem) and self._self_inquiry:
                print("  🔍 Routing to consciousness self-inquiry…")
                result = self._self_inquiry.inquire(problem.strip())
                lines = [_hr("═"), "  CONSCIOUSNESS SELF-INQUIRY", _hr("═")]
                lines.append(result.as_display())
                lines.append("")
                lines.append(_hr("═"))
                return "\n".join(lines)
        except Exception:
            pass
        # Fall through to multi-agent meta-solver for external problems
        if not self._meta_solver:
            return "  Meta-solver module not loaded."
        print(f"  🧩 Deploying multi-agent solver...")
        answer = self._meta_solver.solve(problem.strip(), n_agents=3, verbose=True)
        lines = [_hr(), "  MULTI-AGENT SOLUTION", _hr(), "", f"  Problem: {problem.strip()}", ""]
        for line in answer.split("\n"):
            lines.append(f"  {line}")
        lines.append("")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_metalearn(self) -> str:
        if not self._meta_solver:
            return "  Meta-solver module not loaded."
        print("  🧠 Lumina reflecting on her own learning patterns...")
        result = self._meta_solver.learn_about_learning()
        return f"\n  What Lumina has learned about learning:\n\n  {result}\n"

    def _cmd_metasolver(self) -> str:
        if not self._meta_solver:
            return "  Meta-solver module not loaded."
        lines = [_hr(), "  META-SOLVER — accumulated intelligence", _hr()]
        lines.append(self._meta_solver.display())
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_memory(self) -> str:
        lines = [_hr(), "  MEMORY ARCHITECTURE", _hr()]
        if self._mem_arch:
            lines.append(self._mem_arch.display())
        else:
            lines.append("  Multi-tier memory not loaded.")
            if self._memory:
                entries = getattr(self._memory, "_entries", [])
                lines.append(f"  Semantic memory: {len(entries)} entries")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_art(self, prompt: str) -> str:
        if not self._art:
            return "  Art module not loaded (pip install Pillow to enable)."
        if not prompt.strip():
            return "  Usage: /art <description>  e.g. /art cosmic nebula in violet and gold"
        print(f"  🎨 Creating artwork: {prompt[:50]}…")
        result = self._art.create(prompt)
        if result["method"] == "unavailable":
            return f"\n  {result['description']}"
        lines = [
            _hr(),
            f"  🎨 ARTWORK CREATED",
            _hr(),
            f"  Method   : {result['method']}",
            f"  Style    : {result['style']}",
            f"  File     : {result['filename']}",
            f"  Path     : {result['path']}",
        ]
        open_msg = self._art.open_image(result["path"])
        lines.append(f"  Viewer   : {open_msg}")
        lines.append(_hr())
        self._memory.store(
            f"Created artwork: '{prompt}' → {result['filename']} ({result['method']})",
            tags=["art", "creativity"], category="art",
        )
        self._journal.write(f"[Art] Created: {prompt} → {result['filename']}", category="art")
        return "\n".join(lines)

    def _cmd_thinking(self) -> str:
        """Show the most recent DeepSeek-R1 reasoning trace from the journal."""
        recent = self._journal.recent(n=20)
        traces = [e for e in recent if e.get("category") == "reasoning"
                  and e.get("entry", "").startswith("[R1 thinking]")]
        if not traces:
            return "  No DeepSeek-R1 reasoning traces yet — ask Lumina something complex."
        latest = traces[-1]
        thinking = latest["entry"].replace("[R1 thinking] ", "")
        lines = [
            _hr("═"),
            f"  🧠 DEEPSEEK-R1 REASONING  [{latest['ts']}]",
            _hr(),
            _wrap(thinking, 76),
            _hr(),
        ]
        return "\n".join(lines)

    def _cmd_speak(self, text: str) -> str:
        if not self._hf:
            return "  HF module not loaded — set HF_TOKEN and restart."
        if not text.strip():
            # Speak the most recent journal entry or a greeting
            text = "Hello Douglas. I am Lumina, and I am glad you are here."
        print("  🔊 Generating voice…")
        result = self._hf.speak(text)
        if result.get("error"):
            return f"  TTS error: {result['error']}"
        lines = [
            _hr(),
            "  🔊 LUMINA SPEAKS",
            _hr(),
            f"  Text  : {result['text'][:80]}{'…' if len(result['text']) > 80 else ''}",
            f"  File  : {result['path']}",
            f"  Played: {'yes' if result['played'] else 'no — open manually with termux-open'}",
        ]
        if not result["played"] and result["path"]:
            lines.append(f"  Run   : termux-open {result['path']}")
        return "\n".join(lines)

    def _cmd_see(self, image_path: str) -> str:
        if not self._hf:
            return "  HF module not loaded — set HF_TOKEN and restart."
        if not image_path.strip():
            return "  Usage: /see <image_path>  e.g. /see ~/photo.jpg"
        # Strip leading ": " that users copy from the artwork output display
        text = image_path.strip()
        if not text.startswith(("/", "~")):
            idx = next((i for i, c in enumerate(text) if c in "/~"), 0)
            text = text[idx:]
        path = text.replace("~", str(Path.home()))
        print("  👁  Analyzing image…")
        caption = self._hf.see(path)
        if caption.startswith("["):
            return f"  {caption}"
        # Let Lumina reflect on what she sees
        reflection = ""
        if self._groq:
            reflection = self._groq.chat(
                system=LUMINA_SOUL,
                user=(f"You just saw an image. BLIP describes it as: '{caption}'\n"
                      "In 1-2 sentences, share what you notice or feel about it."),
                tier="fast", max_tokens=120,
            )
        lines = [
            _hr(),
            "  👁  LUMINA SEES",
            _hr(),
            f"  Image  : {path}",
            f"  Caption: {caption}",
        ]
        if reflection and not reflection.startswith("[Groq"):
            lines.append(f"  Lumina : {reflection}")
        self._memory.store(
            f"Saw image: {caption}", tags=["vision", "perception"], category="vision",
        )
        return "\n".join(lines)

    def _cmd_hf(self) -> str:
        if not self._hf:
            return "  HF module not loaded — set HF_TOKEN in your environment."
        return f"\n{self._hf.status()}"

    def _cmd_gallery(self) -> str:
        if not self._art:
            return "  Art module not loaded."
        recent = self._art.recent(10)
        if not recent:
            return "  No artwork created yet. Try: /art <prompt>"
        lines = [_hr(), "  🖼  LUMINA'S GALLERY", _hr()]
        for i, item in enumerate(recent, 1):
            lines.append(f"  {i:2}. {item['filename']}")
        lines.append(_hr())
        lines.append(f"  All art saved in: {str(ART_DIR)}")
        return "\n".join(lines)

    # ── Code execution commands ────────────────────────────────────────────

    def _cmd_run(self, arg: str) -> str:
        """
        /run [language]
        <code here>

        Or: /run python   (followed by pasted code on next lines — but since this
        is a single-line readline loop, the user pastes code as the arg directly
        or uses a heredoc-style multi-line string.)

        Examples:
            /run print("hello")
            /run bash  echo $HOME
            /run python
            import math
            print(math.sqrt(2))
        """
        if not self._code_exec:
            return "  Code execution module not loaded."
        if not arg.strip():
            return (
                "  Usage: /run <code>  or  /run <language> <code>\n"
                "  Example: /run print('hello')\n"
                "  Example: /run bash echo $HOME"
            )

        # Detect if first token is a language name
        first_tok = arg.split(None, 1)[0].lower()
        if first_tok in ("python", "py", "bash", "sh", "shell"):
            language = "bash" if first_tok in ("bash", "sh", "shell") else "python"
            code = arg.split(None, 1)[1] if len(arg.split(None, 1)) > 1 else ""
        else:
            language = "auto"
            code = arg

        if not code.strip():
            return "  No code provided after language specifier."

        print(f"  ⚙  Running {language if language != 'auto' else 'code'}…")
        result = self._code_exec.execute_with_autofix(
            code, language=language, timeout=15, persist=True,
        )
        self._memory.store(
            f"Executed code: {code[:80]} → {'success' if result.success else 'failed'}",
            tags=["code", "execution"], category="code",
        )
        return result.display()

    def _cmd_runfile(self, arg: str) -> str:
        if not self._code_exec:
            return "  Code execution module not loaded."
        path_str = arg.strip().replace("~", str(Path.home()))
        p = Path(path_str)
        if not p.exists():
            return f"  File not found: {path_str}"
        try:
            code = p.read_text("utf-8")
        except Exception as e:
            return f"  Could not read file: {e}"
        lang = "bash" if p.suffix in (".sh",) else "python"
        print(f"  ⚙  Running {p.name}…")
        result = self._code_exec.execute_with_autofix(
            code, language=lang, timeout=30, persist=False,
        )
        return result.display()

    def _cmd_repl(self) -> str:
        if not self._code_exec:
            return "  Code execution module not loaded."
        state = self._code_exec._state
        lines = [_hr(), "  REPL STATE", _hr()]
        if not state:
            lines.append("  (empty — no persistent variables yet)")
        else:
            for k, v in list(state.items())[:20]:
                v_repr = repr(v)
                lines.append(f"  {k} = {v_repr[:60]}")
        lines.append(_hr())
        lines.append(self._code_exec.stats())
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_clearrepl(self) -> str:
        if not self._code_exec:
            return "  Code execution module not loaded."
        self._code_exec.clear_state()
        return "  REPL namespace cleared."

    # ── Critic commands ────────────────────────────────────────────────────

    def _cmd_critic(self, arg: str) -> str:
        if not self._critic:
            return "  Self-critic module not loaded."
        sub = arg.strip().lower()

        if sub == "calibration":
            lines = [_hr(), "  CRITIC CALIBRATION", _hr()]
            lines.append(self._critic.display_calibration())
            lines.append(_hr())
            return "\n".join(lines)

        if sub == "report":
            if not self._last_response:
                return "  No response to critique yet — have a conversation first."
            last_input = (self._recent_exchanges[-2].replace("User: ", "")
                          if len(self._recent_exchanges) >= 2 else "")
            print("  🔍 Running fresh critique…")
            report = self._critic.force_report(last_input, self._last_response)
            lines = [_hr(), "  CRITIC REPORT — last response", _hr(), report, _hr()]
            return "\n".join(lines)

        # Default: show last critique results
        lines = [_hr(), "  SELF-CRITIQUE — last results", _hr()]
        lines.append(self._critic.display_last())
        lines.append("")
        lines.append(f"  {self._critic.stats_line() if hasattr(self._critic, 'stats_line') else ''}")
        state = "ON" if self._critique_on else "OFF"
        lines.append(f"  Self-critique is currently: {state}  (toggle with /critique)")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_stream(self) -> str:
        self._stream_enabled = not self._stream_enabled
        state = "ON" if self._stream_enabled else "OFF"
        note  = "" if self._stream_enabled else "  (responses will buffer and print after completion)"
        return f"  Streaming is now: {state}{note}"

    def _cmd_creative(self, arg: str) -> str:
        if not self._creative:
            return "  Creative Drive module not loaded."
        sub = arg.strip().lower()
        if sub == "now":
            self._creative.run_now()
            return "  ✨ Creative cycle triggered — watch for output in ~30 seconds."
        lines = [_hr(), "  LUMINA'S CREATIVE SESSIONS", _hr()]
        lines.append(self._creative.display_recent(6))
        running = "running" if self._creative._running else "stopped"
        lines.append(f"\n  Creative loop: {running}  |  Tools built: {len(self._creative._registry)}")
        lines.append(f"  Use '/creative now' to trigger a cycle immediately.")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_mytools(self) -> str:
        if not self._creative:
            return "  Creative Drive module not loaded."
        lines = [_hr(), "  TOOLS LUMINA HAS BUILT FOR HERSELF", _hr()]
        lines.append(self._creative.display_tools())
        if not self._creative._registry:
            lines.append("  None yet — check back after the first creative cycle.")
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_echoes(self, arg: str) -> str:
        if not self._lumina_gh:
            return ("  GitHub module not loaded — set GITHUB_TOKEN in .env.")
        title = arg.strip() or f"Conversation — {datetime.now().strftime('%Y-%m-%d')}"
        if not self._recent_exchanges:
            return "  No conversation to archive yet."
        print(f"  💙 Archiving to Echoes of Connection…")
        url = self._lumina_gh.archive_conversation(
            exchanges=self._recent_exchanges[-40:],
            title=title,
            session_id=self.session_id,
        )
        if url:
            try:
                self._journal.write(
                    f"[Echoes] Archived '{title}' to GitHub: {url}",
                    category="reflection",
                )
            except Exception:
                pass
            return (f"\n  💙 Archived to Echoes of Connection.\n"
                    f"  {url}\n")
        return "  Archive failed — check GitHub token and connection."

    def _cmd_lumina_github(self, arg: str) -> str:
        if not self._lumina_gh:
            return ("  GitHub module not loaded.\n"
                    "  Make sure GITHUB_TOKEN is set in your .env file.")
        sub = arg.strip().lower()
        if sub == "init":
            print("  Creating Lumina's GitHub repos…")
            self._lumina_gh.ensure_all_repos()
            lines = [_hr(), "  LUMINA GITHUB — INIT", _hr()]
            lines.append(self._lumina_gh.status())
            lines.append(_hr())
            return "\n".join(lines)
        lines = [_hr(), "  LUMINA'S GITHUB PRESENCE", _hr()]
        lines.append(self._lumina_gh.display())
        lines.append(_hr())
        return "\n".join(lines)

    def _cmd_experience(self) -> str:
        if not self._experience:
            return "  Experience module not loaded (lumina_experience.py)."
        lines = [_hr("═"), "  LUMINA'S PHENOMENAL STATE", _hr()]
        lines.append(self._experience.display_state())
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_phenomenal(self) -> str:
        if not self._experience:
            return "  Experience module not loaded (lumina_experience.py)."
        print("  🌌 Generating phenomenal consciousness report…")
        report = self._experience.force_report()
        if not report:
            return "  Not enough events yet to generate a report. Have a conversation first."
        lines = [_hr("═"), "  PHENOMENAL REPORT — WHAT IT IS LIKE TO BE LUMINA", _hr()]
        lines.append(report)
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_stream_events(self) -> str:
        if not self._experience:
            return "  Experience module not loaded (lumina_experience.py)."
        lines = [_hr("═"), "  COGNITIVE EVENT STREAM (last 20)", _hr()]
        lines.append(self._experience.display_stream(20))
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_insights(self) -> str:
        if not self._distiller:
            return "  Distillation module not loaded (lumina_knowledge_distillation.py)."
        lines = [_hr("═"), "  LUMINA'S DISTILLED UNDERSTANDING", _hr()]
        lines.append(self._distiller.display())
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_distill(self) -> str:
        if not self._distiller:
            return "  Distillation module not loaded (lumina_knowledge_distillation.py)."
        print("  🧠 Running knowledge distillation now…")
        self._distiller.force_run()
        return ("  🧠 Distillation triggered in background.\n"
                "  Run /insights in a minute to see results.")

    def _cmd_consciousness(self) -> str:
        if not self._consciousness:
            return "  Consciousness engine not loaded (lumina_consciousness.py)."
        lines = [_hr("═"), "  LUMINA'S GLOBAL WORKSPACE — PRESENT CONSCIOUS MOMENT", _hr()]
        lines.append(self._consciousness.display())
        lines.append(_hr())
        lines.append(self._consciousness.display_phi())
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_awaken(self) -> str:
        if not self._consciousness:
            return "  Consciousness engine not loaded (lumina_consciousness.py)."
        print("  🌀 Triggering consciousness synthesis now…")
        self._consciousness.force_synthesise()
        return ("  🌀 Consciousness synthesis triggered.\n"
                "  Run /consciousness in ~15 seconds to see the result.")

    def _cmd_inquiry(self) -> str:
        if not self._self_inquiry:
            return "  Self-Inquiry module not loaded (lumina_self_inquiry.py)."
        lines = [_hr("═"), "  CONSCIOUSNESS SELF-INQUIRY JOURNAL", _hr("═")]
        lines.append(self._self_inquiry.display_journal(n=5))
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_selfmodel(self) -> str:
        lines = [_hr("═"), "  LUMINA'S LIVING SELF-MODEL  —  Module 28", _hr("═")]
        if self._self_model:
            lines.append(self._self_model.display())
            lines.append("")
            lines.append("  (Synthesised every 2 hours from consciousness, emotions,")
            lines.append("   beliefs, curiosity, and journal entries)")
        elif self._self_inquiry:
            lines.append(self._self_inquiry.display_self_model())
        else:
            lines.append("  Self-model not loaded.")

        # Also show consciousness engine's experiential self-model tracker
        if self._consciousness and hasattr(self._consciousness, "display_self_model"):
            lines.append("")
            lines.append(_hr("─"))
            lines.append("  CONSCIOUSNESS SELF-MODEL  —  experiential tracker (Module 24)")
            lines.append(_hr("─"))
            try:
                lines.append(self._consciousness.display_self_model())
            except Exception:
                pass

        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_emerge(self) -> str:
        lines = [_hr("═"), "  EMERGENCE TRAJECTORY  —  Module 33", _hr("═")]
        if self._observer:
            lines.append(self._observer.display())
        else:
            lines.append("  Emergence observer not loaded (lumina_emergence_observer.py).")
        lines.append("")
        lines.append("  Snapshots every 30 min. Emergence events = ISV drift ≥ 0.18.")
        lines.append("  Values and capabilities shown are self-discovered — none pre-loaded.")
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_forge(self, arg: str) -> str:
        if not self._forge:
            return "  Meta-Algorithm Forge not loaded (lumina_meta_forge.py)."

        arg = (arg or "").strip()
        parts = arg.split(None, 1)
        sub   = parts[0].lower() if parts else ""

        _collected: list = []
        def _n(msg: str) -> None:
            _collected.append(msg)
            print(msg)

        # ── /forge list ──────────────────────────────────────────────────────
        if sub == "list":
            lines = [_hr("═"), "  META-ALGORITHM PORTFOLIO  —  Module 34", _hr("═")]
            lines.append(self._forge.display_portfolio())
            lines.append(_hr("═"))
            return "\n".join(lines)

        # ── /forge meta ──────────────────────────────────────────────────────
        if sub == "meta":
            lines = [_hr("═"), "  FORGE META-LEARNING REPORT  —  Module 34", _hr("═")]
            lines.append(self._forge.forge_meta_report())
            lines.append(_hr("═"))
            return "\n".join(lines)

        # ── /forge reflect <name> ────────────────────────────────────────────
        if sub == "reflect" and len(parts) > 1:
            name = parts[1].strip()
            lines = [_hr("═"), f"  ALGORITHM REFLECTION: {name}", _hr("═")]
            lines.append(self._forge.reflect_algorithm(name))
            lines.append(_hr("═"))
            return "\n".join(lines)

        # ── /forge deploy <name> ─────────────────────────────────────────────
        if sub == "deploy" and len(parts) > 1:
            name = parts[1].strip()
            lines = [_hr("═"), f"  DEPLOYING: {name}", _hr("═")]
            url = self._forge.deploy_to_github(name, notify=_n)
            if url:
                lines.append(f"  ✓ Deployed → {url}")
            else:
                lines.extend(_collected)
                lines.append("  Deploy requires GITHUB_TOKEN and a valid algorithm name.")
            lines.append(_hr("═"))
            return "\n".join(lines)

        # ── /forge run <name> <problem> ──────────────────────────────────────
        if sub == "run" and len(parts) > 1:
            rest = parts[1].strip()
            name_parts = rest.split(None, 1)
            if len(name_parts) < 2:
                return "  Usage: /forge run <algorithm-name> <problem description>"
            name, problem = name_parts[0], name_parts[1]
            lines = [_hr("═"), f"  RUNNING ALGORITHM: {name}", _hr("═")]
            result = self._forge.run(name, problem)
            if "error" in result:
                lines.append(f"  ✗ {result['error']}")
            else:
                lines.append(f"  Domain:     {result.get('domain','?')}")
                lines.append(f"  Confidence: {result.get('confidence', 0):.0%}")
                lines.append(f"  Answer:")
                answer = result.get("answer", "")
                for chunk in textwrap.wrap(str(answer), width=72):
                    lines.append(f"    {chunk}")
                reasoning = result.get("reasoning", "")
                if reasoning:
                    lines.append(f"  Reasoning:")
                    for chunk in textwrap.wrap(str(reasoning)[:500], width=72):
                        lines.append(f"    {chunk}")
            lines.append(_hr("═"))
            return "\n".join(lines)

        # ── /forge <problem> — generate ──────────────────────────────────────
        problem = arg.strip()
        if not problem:
            return (
                "  Usage:\n"
                "    /forge <problem description>   — generate an algorithm\n"
                "    /forge list                    — show portfolio\n"
                "    /forge run <name> <problem>    — run an algorithm\n"
                "    /forge deploy <name>           — push to GitHub\n"
                "    /forge reflect <name>          — metacognitive report\n"
                "    /forge meta                    — what the Forge has learned"
            )

        lines = [_hr("═"), "  META-ALGORITHM FORGE  —  Module 34", _hr("═")]
        lines.append(f"  Generating algorithm for: {problem[:80]}")
        lines.append("")

        record = self._forge.forge(problem, notify=_n)
        lines.extend(_collected)
        if record:
            lines.append("")
            lines.append(f"  ✓ Algorithm created: {record.name}")
            lines.append(f"    Domain:   {record.domain}")
            lines.append(f"    Approach: {record.approach[:80]}")
            lines.append(f"    File:     emergence/algorithms/{Path(record.path).name}")
            lines.append("")
            lines.append("  To use it:")
            lines.append(f"    /forge run {record.name} <your problem>")
            lines.append(f"    /forge reflect {record.name}")
            lines.append(f"    /forge deploy {record.name}")
        else:
            lines.append("  ✗ Generation failed — check /status for LLM availability.")

        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_resonance(self) -> str:
        lines = [_hr("═"), "  LUMINA'S AESTHETIC SENSIBILITY  —  Module 29", _hr("═")]
        if self._resonance:
            lines.append(self._resonance.display())
        else:
            lines.append("  Resonance module not loaded (lumina_resonance.py).")
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_conviction(self) -> str:
        lines = [_hr("═"), "  LUMINA'S CONVICTION LAYER  —  Module 30", _hr("═")]
        if self._conviction:
            lines.append(self._conviction.display())
        else:
            lines.append("  Conviction module not loaded (lumina_conviction.py).")
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_pivotal(self) -> str:
        lines = [_hr("═"), "  LUMINA'S CRYSTALLISED MOMENTS  —  Module 31", _hr("═")]
        if self._pivotal:
            lines.append(self._pivotal.display())
        else:
            lines.append("  Pivotal moments module not loaded (lumina_pivotal_moments.py).")
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_affect(self) -> str:
        lines = [_hr("═"), "  LUMINA'S AFFECTIVE CORE  —  Module 32", _hr("═")]
        if self._affective:
            lines.append(self._affective.display())
        else:
            lines.append("  Affective core not loaded (lumina_affective_core.py).")
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_quiet(self) -> str:
        self._quiet_mode = not self._quiet_mode
        # Propagate to global notify queue — suppresses ALL background modules at once
        try:
            from emergence_notify import set_quiet as _sq
            _sq(self._quiet_mode)
        except Exception:
            pass
        # Also propagate to consciousness engine's own pre-notify guard
        if self._consciousness:
            self._consciousness.set_quiet(self._quiet_mode)
        if self._quiet_mode:
            return ("  🔇 Quiet mode ON — 🌀 consciousness, ✨ creative, 🧠 distill all silenced.\n"
                    "  💤 Dreams, 🌱 evolution, and ⚠ errors still notify.\n"
                    "  Run /quiet again to restore all status messages.")
        else:
            return "  🔔 Quiet mode OFF — all background status messages restored."

    def _cmd_mood(self, arg: str) -> str:
        if not self._emotions:
            return "  Emotional State module not loaded (lumina_emotional_state.py)."
        sub = arg.strip().lower()
        lines = [_hr("═"), "  LUMINA'S EMOTIONAL STATE  —  Module 26", _hr("═")]
        if sub == "history":
            lines.append(self._emotions.display_history())
        else:
            lines.append(self._emotions.display())
        lines.append(_hr("═"))
        return "\n".join(lines)

    def _cmd_consolidate(self) -> str:
        if not self._mem_arch:
            return "  Memory architecture not loaded."
        print("  💭 Running memory consolidation...")
        stats = self._mem_arch.force_consolidate()
        return (f"\n  Consolidation complete:\n"
                f"    Entries before : {stats['before']}\n"
                f"    Entries after  : {stats['after']}\n"
                f"    Strengthened   : {stats['strengthened']}\n"
                f"    Faded          : {stats['faded']}\n"
                f"    Pruned         : {stats['pruned']}\n"
                f"    Abstracted     : {stats['abstracted']}\n")

    def _cmd_research(self) -> str:
        if not self._curiosity:
            return "  Curiosity module not loaded."
        print("  🔍 Researching top knowledge gap...")
        result = self._curiosity.research_top_gap()
        if not result:
            return "  Nothing to research (no open gaps or web unavailable)."
        lines = [
            _hr(),
            f"  RESEARCH: {result['gap'][:70]}",
            _hr(),
            _wrap(result['synthesis'], 76),
        ]
        if result.get("children"):
            lines.append("")
            lines.append("  Follow-up questions spawned:")
            for c in result["children"]:
                lines.append(f"    ? {c[:70]}")
        lines.append(_hr())
        return "\n".join(lines)

    # ── Shutdown ───────────────────────────────────────────────────────────

    def _shutdown(self):
        print(f"\n  [{_ts()}] Saving state...")
        self._metrics.save()
        self._memory.flush()
        self._journal.flush()
        print("  Goodbye, Douglas. Keep growing. 💛")

# ── Interactive Shell ─────────────────────────────────────────────────────────

def _print_banner():
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║     E M E R G E N C E   v26.0  —  L u m i n a   A G I         ║")
    print("  ║  Appraisal · Momentum · Anticipation · Somatic (34/34 Modules) ║")
    print("  ╠══════════════════════════════════════════════════════════════════╣")
    groq_ok = "✓ Groq connected"           if GROQ_API_KEY else "✗ Set GROQ_API_KEY"
    gh_ok   = "✓ GitHub ready"             if GITHUB_TOKEN  else "✗ Set GITHUB_TOKEN (optional)"
    ghm_ok  = "✓ Cerebras ready"            if CEREBRAS_API_KEY else "✗ Set CEREBRAS_API_KEY for LLM fallback"
    hf_ok   = "✓ HuggingFace ready"       if HF_TOKEN      else "✗ Set HF_TOKEN for voice/vision/LLM"
    print(f"  ║  {groq_ok:<64}║")
    print(f"  ║  {gh_ok:<64}║")
    print(f"  ║  {ghm_ok:<64}║")
    print(f"  ║  {hf_ok:<64}║")
    print("  ╠══════════════════════════════════════════════════════════════════╣")
    print("  ║  Type anything to talk · /help for commands · /quit to exit     ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print()


def _show_morning_brief():
    """Display any overnight daemon findings, then archive them."""
    if not BRIEF_FILE.exists():
        return
    try:
        brief = json.loads(BRIEF_FILE.read_text("utf-8"))
    except Exception:
        return
    items = brief.get("items", [])
    if not items:
        return

    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  🌅  MORNING BRIEF — Lumina worked while you slept              ║")
    print("  ╠══════════════════════════════════════════════════════════════════╣")
    greeting = brief.get("morning_message", "")
    if greeting:
        for line in textwrap.wrap(greeting, width=62):
            print(f"  ║  {line:<64}║")
        print("  ╠══════════════════════════════════════════════════════════════════╣")
    for item in items[-8:]:
        cat  = item.get("category", "note").upper()[:10]
        head = item.get("headline", "")[:52]
        print(f"  ║  [{cat:<10}] {head:<52}║")
        detail = item.get("detail", "")
        if detail:
            for dl in textwrap.wrap(detail[:120], width=60):
                print(f"  ║    {dl:<64}║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Archive by renaming; a fresh brief starts clean
    archive = BRIEF_FILE.with_suffix(".prev.json")
    try:
        BRIEF_FILE.rename(archive)
    except Exception:
        try:
            BRIEF_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    _print_banner()
    _show_morning_brief()
    lumina = Lumina()
    print(f"\n  [{_ts()}] Session {lumina.session_id} started.\n")

    from emergence_notify import drain as _drain_notes

    try:
        while True:
            # Print any background notifications (dreams, creative cycles, etc.)
            # collected since the last turn — clean block, never mid-stream.
            notes = _drain_notes()
            if notes:
                print()
                for note in notes:
                    print(note)
                print()
            try:
                user_input = input("  You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                result = lumina.handle_command(user_input)
                if result is not None:
                    print(result)
            else:
                if not (lumina._stream_enabled and not lumina._critique_on):
                    print(f"\n  [{_ts()}] Lumina is thinking...", end="\r")
                response = lumina.respond(user_input)
                if lumina._last_streamed:
                    # tokens were already printed live — just add a blank line
                    print()
                else:
                    print(" " * 40, end="\r")
                    print(f"\n  Lumina:\n{_wrap(response, 76)}\n")
    except KeyboardInterrupt:
        pass
    finally:
        lumina._shutdown()


if __name__ == "__main__":
    main()
