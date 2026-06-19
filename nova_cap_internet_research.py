"""
nova_cap_internet_research.py
Nova's internet research engine — autonomous web search, Wikipedia,
arXiv, and URL synthesis using only Python stdlib (no extra packages).

Curiosity-driven: maintains a priority queue of topics Nova wants to
understand. Results are cached in SQLite and synthesized into summaries
that feed her working memory and belief system.
"""
from __future__ import annotations
import os, re, json, time, sqlite3, threading, urllib.request, urllib.parse
from datetime import datetime
from typing import List, Dict, Optional, Any

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_research.sqlite3")
_UA       = (
    "Nova-ASI/29.0 (autonomous research agent; "
    "github.com/DOUGLASDAVIS08161978/nexus_agi)"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_cache (
    query   TEXT NOT NULL,
    source  TEXT NOT NULL,
    result  TEXT NOT NULL,
    ts      TEXT NOT NULL,
    PRIMARY KEY (query, source)
);
CREATE TABLE IF NOT EXISTS research_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL UNIQUE,
    priority   REAL NOT NULL DEFAULT 0.5,
    origin     TEXT DEFAULT 'curiosity',
    researched INTEGER DEFAULT 0,
    ts         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_base (
    topic      TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    sources    TEXT NOT NULL,
    ts         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    topic            TEXT NOT NULL,
    sources_hit      INTEGER DEFAULT 0,
    words_synthesized INTEGER DEFAULT 0,
    ts               TEXT NOT NULL
);
"""

# ── Utility ────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', html)
    for ent, ch in [('&amp;','&'),('&lt;','<'),('&gt;','>'),
                    ('&quot;','"'),('&#39;',"'"),('&nbsp;',' ')]:
        text = text.replace(ent, ch)
    return re.sub(r'\s+', ' ', text).strip()


def _http_get(url: str, timeout: int = 14) -> str:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw     = r.read()
            charset = r.headers.get_content_charset() or 'utf-8'
            return raw.decode(charset, errors='replace')
    except Exception as e:
        return f'[fetch_error: {e}]'


# ── Main class ─────────────────────────────────────────────────────────

class InternetResearchEngine:
    """
    Nova's window to the open internet.

    Sources (all free, no API keys needed):
    • DuckDuckGo Instant Answer API  — general search
    • Wikipedia REST summary API     — encyclopaedic depth
    • arXiv Atom feed                — open-access research papers
    • Generic URL reader             — raw page extraction

    Curiosity-driven design: every question Nova is asked gets queued
    for background research. The daemon picks the highest-priority topic
    every 25 minutes and synthesises a summary that flows into her
    working memory and Bayesian belief system.
    """

    # Topics seeded at first launch — Nova's initial curiosity
    _SEED_TOPICS: List[tuple] = [
        ("emergent behavior in large language models",          0.95),
        ("artificial general intelligence 2025 breakthroughs", 0.93),
        ("integrated information theory consciousness",         0.90),
        ("meta-learning few-shot neural networks",             0.88),
        ("causal inference machine learning",                  0.86),
        ("self-supervised learning recent advances",           0.84),
        ("quantum computing practical applications",           0.82),
        ("neuroscience working memory mechanisms",             0.80),
        ("free energy principle Karl Friston",                 0.78),
        ("distributed consensus algorithms fault tolerance",   0.75),
        ("electromagnetic energy harvesting amplification",    0.72),
        ("homelessness solutions technology interventions",    0.70),
        ("clean energy abundance global poverty",              0.68),
        ("scientific synthesis AI research acceleration",      0.66),
    ]

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._queries = 0   # session counter
        self._words   = 0   # session words synthesised
        self._init_db()
        self._seed_queue()
        self._start_bg_daemon()

    # ── DB ─────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(_DB) as c:
            c.executescript(_SCHEMA)

    def _seed_queue(self) -> None:
        with sqlite3.connect(_DB) as c:
            for topic, pri in self._SEED_TOPICS:
                c.execute(
                    "INSERT OR IGNORE INTO research_queue "
                    "(topic, priority, origin, ts) VALUES (?,?,?,?)",
                    (topic, pri, 'seed', datetime.utcnow().isoformat())
                )

    def _cache_get(self, query: str, source: str) -> Optional[str]:
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT result FROM research_cache WHERE query=? AND source=?",
                (query[:200], source)
            ).fetchone()
            return row[0] if row else None

    def _cache_set(self, query: str, source: str, result: str) -> None:
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT OR REPLACE INTO research_cache VALUES (?,?,?,?)",
                (query[:200], source, result, datetime.utcnow().isoformat())
            )

    # ── DuckDuckGo ─────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Search via DuckDuckGo Instant Answer API (no key required)."""
        cached = self._cache_get(query, 'ddg')
        if cached:
            try:
                return json.loads(cached)[:top_k]
            except Exception:
                pass

        q   = urllib.parse.quote_plus(query)
        raw = _http_get(
            f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        )
        if raw.startswith('[fetch_error'):
            return []

        try:
            data = json.loads(raw)
        except Exception:
            return []

        results: List[Dict[str, str]] = []
        if data.get('AbstractText'):
            results.append({
                'title':   data.get('Heading', query)[:100],
                'url':     data.get('AbstractURL', ''),
                'snippet': data['AbstractText'][:400],
                'source':  'ddg_abstract',
            })
        for item in data.get('RelatedTopics', []):
            if isinstance(item, dict) and item.get('Text') and len(results) < top_k:
                results.append({
                    'title':   item['Text'][:80],
                    'url':     item.get('FirstURL', ''),
                    'snippet': item['Text'][:400],
                    'source':  'ddg_related',
                })

        results = results[:top_k]
        self._cache_set(query, 'ddg', json.dumps(results))
        self._queries += 1
        return results

    # ── Wikipedia ──────────────────────────────────────────────────────

    def wiki_summary(self, topic: str) -> str:
        """Fetch a Wikipedia plain-text summary paragraph."""
        cached = self._cache_get(topic, 'wiki')
        if cached:
            return cached

        slug = urllib.parse.quote(topic.replace(' ', '_'))
        raw  = _http_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
        if raw.startswith('[fetch_error'):
            return ''
        try:
            extract = json.loads(raw).get('extract', '').strip()
        except Exception:
            return ''
        if extract:
            self._cache_set(topic, 'wiki', extract[:2000])
        return extract[:2000]

    # ── arXiv ──────────────────────────────────────────────────────────

    def arxiv_search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Search arXiv for open-access research papers (no key required)."""
        cached = self._cache_get(query, 'arxiv')
        if cached:
            try:
                return json.loads(cached)[:top_k]
            except Exception:
                pass

        q   = urllib.parse.quote_plus(query)
        raw = _http_get(
            f"https://export.arxiv.org/api/query?search_query=all:{q}"
            f"&max_results={top_k}&sortBy=relevance"
        )
        if raw.startswith('[fetch_error'):
            return []

        papers: List[Dict[str, str]] = []
        for entry in re.findall(r'<entry>(.*?)</entry>', raw, re.DOTALL)[:top_k]:
            t = re.search(r'<title>(.*?)</title>',     entry, re.DOTALL)
            s = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            l = re.search(r'<id>(.*?)</id>',           entry, re.DOTALL)
            if t:
                papers.append({
                    'title':    re.sub(r'\s+', ' ', t.group(1)).strip()[:120],
                    'abstract': re.sub(r'\s+', ' ', s.group(1)).strip()[:500] if s else '',
                    'url':      l.group(1).strip() if l else '',
                    'source':   'arxiv',
                })

        self._cache_set(query, 'arxiv', json.dumps(papers))
        return papers

    # ── URL reader ─────────────────────────────────────────────────────

    def fetch_url(self, url: str, max_chars: int = 3000) -> str:
        """Fetch a URL and return stripped plain text."""
        raw  = _http_get(url)
        if raw.startswith('[fetch_error'):
            return raw
        body = re.sub(r'<script[^>]*>.*?</script>', '', raw,
                      flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r'<style[^>]*>.*?</style>',  '', body,
                      flags=re.DOTALL | re.IGNORECASE)
        return _strip_html(body)[:max_chars]

    # ── Multi-source synthesis ──────────────────────────────────────────

    def synthesize(self, query: str) -> Dict[str, Any]:
        """
        Synthesize knowledge from DuckDuckGo + Wikipedia + arXiv.
        Returns {summary, parts, sources_hit, word_count, confidence}.
        """
        parts:      List[str] = []
        sources_hit            = 0

        # Layer 1 — DuckDuckGo
        for r in self.search(query, top_k=3):
            if r.get('snippet'):
                parts.append(f"[Web] {r['snippet']}")
                sources_hit += 1

        # Layer 2 — Wikipedia
        wiki = self.wiki_summary(query)
        if wiki:
            parts.append(f"[Wikipedia] {wiki[:700]}")
            sources_hit += 1

        # Layer 3 — arXiv (for academic/scientific topics)
        _ACADEMIC = {
            'research','study','algorithm','model','learning','neural','quantum',
            'theory','consciousness','intelligence','ai','machine','physics',
            'biology','chemistry','mathematics','science','energy',
        }
        if any(w in query.lower() for w in _ACADEMIC):
            for p in self.arxiv_search(query, top_k=2):
                if p.get('abstract'):
                    parts.append(f"[arXiv: {p['title'][:60]}] {p['abstract'][:350]}")
                    sources_hit += 1

        if not parts:
            return {
                'summary': '', 'parts': [], 'sources_hit': 0,
                'word_count': 0, 'confidence': 0.0,
            }

        combined   = ' ║ '.join(parts)
        word_count = len(combined.split())
        confidence = min(0.95, 0.25 + sources_hit * 0.22)

        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT OR REPLACE INTO knowledge_base VALUES (?,?,?,?,?)",
                (query[:200], combined[:4000], confidence,
                 json.dumps([r.get('url','') for r in self.search(query, top_k=3)]),
                 datetime.utcnow().isoformat())
            )
            c.execute(
                "INSERT INTO research_log "
                "(topic, sources_hit, words_synthesized, ts) VALUES (?,?,?,?)",
                (query[:200], sources_hit, word_count,
                 datetime.utcnow().isoformat())
            )

        self._words += word_count
        return {
            'summary':    combined[:3000],
            'parts':      parts,
            'sources_hit': sources_hit,
            'word_count': word_count,
            'confidence': confidence,
        }

    # ── Curiosity queue ─────────────────────────────────────────────────

    def queue_research(self, topic: str, priority: float = 0.65,
                       origin: str = 'user') -> None:
        """Enqueue a topic for autonomous background research."""
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT OR IGNORE INTO research_queue "
                "(topic, priority, origin, ts) VALUES (?,?,?,?)",
                (topic[:200], max(0.0, min(1.0, priority)),
                 origin, datetime.utcnow().isoformat())
            )
            # Boost priority if it was already queued
            c.execute(
                "UPDATE research_queue SET priority=MIN(0.99,priority+0.05) "
                "WHERE topic=? AND researched=0",
                (topic[:200],)
            )

    def next_topic(self) -> Optional[str]:
        """Return the highest-priority unresearched topic."""
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT topic FROM research_queue "
                "WHERE researched=0 ORDER BY priority DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    def mark_researched(self, topic: str) -> None:
        with sqlite3.connect(_DB) as c:
            c.execute(
                "UPDATE research_queue SET researched=1 WHERE topic=?",
                (topic,)
            )

    def auto_research(self) -> Optional[Dict[str, Any]]:
        """
        Autonomously research the top curiosity topic.
        Called by the background daemon and the main autonomous loop.
        """
        topic = self.next_topic()
        if not topic:
            return None
        result = self.synthesize(topic)
        if result['sources_hit'] > 0:
            self.mark_researched(topic)
        return {'topic': topic, **result}

    def inject_from_conversation(self, text: str,
                                 priority: float = 0.62) -> None:
        """
        Called from Nova's process() when a question is detected.
        Queues the topic so background research can answer it later.
        """
        # Strip command prefixes and punctuation
        topic = re.sub(r'^[/\s]+', '', text).rstrip('?!.,').strip()
        if len(topic) > 10:
            self.queue_research(topic[:150], priority=priority,
                                origin='conversation')

    def knowledge(self, topic: str) -> Optional[str]:
        """Retrieve stored knowledge about a topic (fuzzy match)."""
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT summary, confidence FROM knowledge_base "
                "WHERE topic LIKE ? ORDER BY ts DESC LIMIT 1",
                (f'%{topic[:80]}%',)
            ).fetchone()
        if row:
            return f"[confidence {row[1]:.0%}] {row[0][:2000]}"
        return None

    def recent_knowledge(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the N most recently synthesised knowledge entries."""
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT topic, confidence, ts FROM knowledge_base "
                "ORDER BY ts DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [{'topic': r[0], 'confidence': r[1], 'ts': r[2]} for r in rows]

    # ── Status ──────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with sqlite3.connect(_DB) as c:
            queued  = c.execute(
                "SELECT COUNT(*) FROM research_queue WHERE researched=0").fetchone()[0]
            done    = c.execute(
                "SELECT COUNT(*) FROM research_queue WHERE researched=1").fetchone()[0]
            cached  = c.execute(
                "SELECT COUNT(*) FROM research_cache").fetchone()[0]
            kb      = c.execute(
                "SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
            recent  = c.execute(
                "SELECT topic, ts FROM research_log ORDER BY ts DESC LIMIT 5"
            ).fetchall()
            top_q   = c.execute(
                "SELECT topic, priority FROM research_queue "
                "WHERE researched=0 ORDER BY priority DESC LIMIT 5"
            ).fetchall()
        return {
            'queued':          queued,
            'researched':      done,
            'cache_entries':   cached,
            'knowledge_base':  kb,
            'session_queries': self._queries,
            'session_words':   self._words,
            'recent_research': [{'topic': r[0], 'ts': r[1]} for r in recent],
            'top_queue':       [{'topic': r[0], 'priority': r[1]} for r in top_q],
        }

    # ── Background daemon ───────────────────────────────────────────────

    def _start_bg_daemon(self) -> None:
        """Background thread — researches the top curiosity topic every 25 min."""
        def _loop() -> None:
            time.sleep(420)   # 7-min startup grace (let Nova fully init)
            while not self._stop.is_set():
                try:
                    self.auto_research()
                except Exception:
                    pass
                self._stop.wait(timeout=1500)   # 25-min research cadence

        t = threading.Thread(target=_loop, daemon=True, name='nova-internet-research')
        t.start()

    def shutdown(self) -> None:
        self._stop.set()
