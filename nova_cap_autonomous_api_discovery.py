"""
Nova ASI — Autonomous API Discovery
Proposed autonomously via /evolve
"""

"""
APIExplorer — Autonomous API Discovery Engine for Nova.
Fetches OpenAPI specs, indexes endpoints with TF-IDF salience scoring,
calls endpoints with Bayesian confidence tracking and EMA success rates.
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫
"""

import os
import json
import sqlite3
import time
import re
import math
import statistics
import threading
import hashlib
import urllib.request
import urllib.parse
import urllib.error
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

class APIExplorer:
    """Autonomous API discovery, indexing, and intelligent endpoint invocation."""

    _DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_explorer.db")
    _CAPACITY = 200
    _DECAY = 0.0005
    _EMA_ALPHA = 0.15

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._endpoints: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._base_url: str = ""
        self._call_history: List[Tuple[float, float]] = []   # (predicted_success, actual)
        self._ema_success: float = 0.5
        self._total_calls: int = 0
        self._idf_cache: Dict[str, float] = {}
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._init_db()
        self._load_from_db()

    # ── DB bootstrap ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS endpoints (
                    name TEXT PRIMARY KEY,
                    method TEXT, path TEXT, summary TEXT,
                    params TEXT, base_url TEXT,
                    importance REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    last_ts REAL,
                    success_ema REAL DEFAULT 0.5
                )""")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS call_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, ts REAL, status_code INTEGER,
                    latency_ms REAL, confidence REAL, success INTEGER
                )""")

    def _load_from_db(self) -> None:
        cur = self._conn.execute("SELECT * FROM endpoints")
        for row in cur.fetchall():
            name, method, path, summary, params_json, base_url, imp, acc, ts, sema = row
            self._endpoints[name] = {
                "method": method, "path": path, "summary": summary,
                "params": json.loads(params_json or "{}"),
                "base_url": base_url,
                "importance": imp, "access_count": acc,
                "last_ts": ts or time.time(),
                "success_ema": sema
            }

    # ── Decay & salience helpers ───────────────────────────────────────────────

    def _decayed_importance(self, name: str) -> float:
        ep = self._endpoints[name]
        elapsed = time.time() - ep["last_ts"]
        return ep["importance"] * math.exp(-self._DECAY * elapsed)

    def _tfidf_score(self, name: str, query_tokens: List[str]) -> float:
        ep = self._endpoints.get(name, {})
        doc = f"{ep.get('path','')} {ep.get('summary','')} {ep.get('method','')}".lower().split()
        if not doc or not query_tokens:
            return 0.0
        N = max(len(self._endpoints), 1)
        score = 0.0
        for t in query_tokens:
            tf = doc.count(t) / len(doc)
            df = self._idf_cache.get(t, sum(1 for e in self._endpoints.values()
                                            if t in f"{e.get('path','')} {e.get('summary','')}".lower()))
            self._idf_cache[t] = df
            idf = math.log(N / (df + 1))
            score += tf * idf
        return score

    def _rebuild_idf(self) -> None:
        self._idf_cache.clear()

    # ── Public API ────────────────────────────────────────────────────────────

    def discover(self, base_url: str) -> Dict[str, Any]:
        """Returns dict of discovered endpoint count and confidence after fetching /openapi.json."""
        base_url = base_url.rstrip("/")
        spec_url = f"{base_url}/openapi.json"
        try:
            with urllib.request.urlopen(spec_url, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                spec: Dict = json.loads(raw)
        except urllib.error.URLError as exc:
            return {"ok": False, "error": str(exc), "endpoints": 0, "confidence": 0.0}
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"JSON parse: {exc}", "endpoints": 0, "confidence": 0.0}

        paths: Dict = spec.get("paths", {})
        discovered = 0
        with self._lock:
            self._base_url = base_url
            for path, methods in paths.items():
                for method, info in methods.items():
                    if method.upper() not in ("GET","POST","PUT","DELETE","PATCH"):
                        continue
                    op_id = info.get("operationId") or f"{method.upper()}_{path}"
                    name = re.sub(r"[^a-zA-Z0-9_]", "_", op_id)
                    params = {p["name"]: p.get("schema", {})
                              for p in info.get("parameters", []) if "name" in p}
                    ep = {
                        "method": method.upper(), "path": path,
                        "summary": info.get("summary", "")[:120],
                        "params": params, "base_url": base_url,
                        "importance": 1.0, "access_count": 0,
                        "last_ts": time.time(), "success_ema": 0.5
                    }
                    self._endpoints[name] = ep
                    self._conn.execute("""
                        INSERT OR REPLACE INTO endpoints
                        (name,method,path,summary,params,base_url,importance,access_count,last_ts,success_ema)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (name, ep["method"], ep["path"], ep["summary"],
                         json.dumps(ep["params"]), base_url,
                         ep["importance"], 0, ep["last_ts"], 0.5))
                    discovered += 1
            self._conn.commit()
            self._rebuild_idf()
            self._lru_evict()

        confidence = min(1.0, discovered / max(len(paths) * 2, 1))
        return {"ok": True, "endpoints": discovered, "base_url": base_url,
                "confidence": round(confidence, 4), "spec_title": spec.get("info", {}).get("title", "")}

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """Returns list of dicts with name, method, path, importance, success_ema for all known endpoints."""
        now = time.time()
        with self._lock:
            result = []
            for name, ep in self._endpoints.items():
                elapsed = now - ep["last_ts"]
                decayed = ep["importance"] * math.exp(-self._DECAY * elapsed)
                result.append({
                    "name": name,
                    "method": ep["method"],
                    "path": ep["path"],
                    "summary": ep["summary"],
                    "importance": round(decayed, 4),
                    "access_count": ep["access_count"],
                    "success_ema": round(ep["success_ema"], 4)
                })
            result.sort(key=lambda x: x["importance"], reverse=True)
        return result

    def call(self, endpoint_name: str, **params: Any) -> Dict[str, Any]:
        """Returns HTTP response dict with status, body, latency_ms, confidence, and Bayesian success update."""
        with self._lock:
            ep = self._endpoints.get(endpoint_name)
        if ep is None:
            return {"ok": False, "error": f"Unknown endpoint: {endpoint_name}",
                    "confidence": 0.0}

        prior = ep["success_ema"]
        url = ep["base_url"] + ep["path"]
        method = ep["method"]
        confidence = prior

        # Build request
        encoded: Optional[bytes] = None
        if method == "GET" and params:
            url += "?" + urllib.parse.urlencode(params)
        elif params:
            encoded = json.dumps(params).encode("utf-8")

        req = urllib.request.Request(url, data=encoded, method=method)
        if encoded:
            req.add_header("Content-Type", "application/json")

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body_raw = resp.read().decode("utf-8", errors="replace")
                status = resp.status
                latency_ms = (time.time() - t0) * 1000
                try:
                    body = json.loads(body_raw)
                except json.JSONDecodeError:
                    body = body_raw
                success = 1 if 200 <= status < 300 else 0
        except urllib.error.HTTPError as exc:
            status = exc.code
            latency_ms = (time.time() - t0) * 1000
            body = str(exc.reason)
            success = 0
        except urllib.error.URLError as exc:
            return {"ok": False, "error": str(exc), "confidence": confidence}

        # Bayesian posterior update on success probability
        likelihood_success = 0.9 if success else 0.1
        posterior = (likelihood_success * prior) / (
            likelihood_success * prior + (1 - likelihood_success) * (1 - prior) + 1e-12)

        # EMA update on global success rate
        self._ema_success = self._EMA_ALPHA * success + (1 - self._EMA_ALPHA) * self._ema_success
        self._total_calls += 1
        self._call_history.append((confidence, float(success)))
        if len(self._call_history) > 200:
            self._call_history.pop(0)

        with self._lock:
            ep = self._endpoints[endpoint_name]
            ep["success_ema"] = posterior
            ep["access_count"] += 1
            ep["importance"] = self._decayed_importance(endpoint_name) * (1 + 0.1 * ep["access_count"])
            ep["last_ts"] = time.time()
            self._conn.execute("""
                UPDATE endpoints SET importance=?,access_count=?,last_ts=?,success_ema=? WHERE name=?""",
                (ep["importance"], ep["access_count"], ep["last_ts"], posterior, endpoint_name))
            self._conn.execute("""
                INSERT INTO call_log (name,ts,status_code,latency_ms,confidence,success)
                VALUES (?,?,?,?,?,?)""",
                (endpoint_name, time.time(), status, latency_ms, confidence, success))
            self._conn.commit()

        return {
            "ok": success == 1, "status": status, "body": body,
            "latency_ms": round(latency_ms, 2),
            "confidence": round(posterior, 4),
            "prior_confidence": round(prior, 4)
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Returns ranked list of endpoints matching query using TF-IDF scoring."""
        tokens = re.sub(r"[^a-z0-9 ]", "", query.lower()).split()
        with self._lock:
            scored = [(name, self._tfidf_score(name, tokens)) for name in self._endpoints]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for name, score in scored[:10]:
            ep = self._endpoints.get(name, {})
            results.append({"name": name, "score": round(score, 4),
                             "method": ep.get("method"), "path": ep.get("path")})
        return results

    def status(self) -> Dict[str, Any]:
        """Returns health dict with numeric keys for ConsciousnessIntegrator Φ computation."""
        history = self._call_history[-50:] if self._call_history else []
        if len(history) >= 2:
            actuals = [a for _, a in history]
            preds = [p for p, _ in history]
            mae = statistics.mean(abs(p - a) for p, a in zip(preds, actuals))
            calibration_error = round(mae, 4)
            try:
                acc_std = statistics.stdev(actuals)
            except statistics.StatisticsError:
                acc_std = 0.0
        else:
            calibration_error = 1.0
            acc_std = 0.0

        with self._lock:
            items = len(self._endpoints)
        return {
            "items": items,
            "confidence": round(self._ema_success, 4),
            "accuracy": round(1.0 - calibration_error, 4),
            "calibration_error": calibration_error,
            "total_calls": self._total_calls,
            "active": min(items, self._CAPACITY),
            "quality": round(self._ema_success * (1 - calibration_error), 4),
            "acc_std": round(acc_std, 4)
        }

    def consolidate(self) -> Dict[str, Any]:
        """Returns summary after persisting all in-memory endpoint states to SQLite."""
        with self._lock:
            count = 0
            for name, ep in self._endpoints.items():
                self._conn.execute("""
                    UPDATE endpoints SET importance=?,access_count=?,last_ts=?,success_ema=? WHERE name=?""",
                    (ep["importance"], ep["access_count"], ep["last_ts"], ep["success_ema"], name))
                count += 1
            self._conn.commit()
        return {"consolidated": count, "ts": datetime.utcnow().isoformat()}

    def forget_least_important(self, n: int = 10) -> List[str]:
        """Returns list of endpoint names evicted by LRU+importance scoring."""
        return self._lru_evict(force_n=n)

    # ── Internal eviction ─────────────────────────────────────────────────────

    def _lru_evict(self, force_n: int = 0) -> List[str]:
        evicted: List[str] = []
        over = max(0, len(self._endpoints) - self._CAPACITY)
        n = max(over, force_n)
        if n <= 0:
            return evicted
        scored = sorted(
            self._endpoints.keys(),
            key=lambda nm: self._decayed_importance(nm)
        )
        for name in scored[:n]:
            del self._
