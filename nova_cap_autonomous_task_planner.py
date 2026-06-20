"""
Nova ASI — Autonomous Task Planner
Proposed autonomously via /evolve
"""

"""
Nova TaskPlanner — Bayesian working-memory-backed autonomous task planner.
Pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3, threading, time, math, statistics, json, os, hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

class TaskPlanner:
    """Autonomous goal-directed task planner with Bayesian decay and LRU working memory."""

    _DECAY = 0.0005          # importance decay rate per second
    _CAPACITY = 200          # max working-memory slots
    _CYCLE_SEC = 30          # auto_cycle interval
    _CONF_FLOOR = 0.30       # replan threshold

    def __init__(self) -> None:
        self._lock = threading.Lock()
        db_path = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'task_planner.db')
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._build_schema()
        # in-memory LRU working memory: key -> {value, importance, ts, access_count}
        self._wm: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cycles: int = 0
        self._history: List[Tuple[float, float]] = []   # (predicted_progress, actual_progress)
        self._ema_pred: float = 0.0
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ── schema ────────────────────────────────────────────────────────────────
    def _build_schema(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY, goal TEXT, created REAL, confidence REAL DEFAULT 1.0
            );
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT, idx INTEGER, description TEXT,
                done INTEGER DEFAULT 0, completed_at REAL DEFAULT NULL,
                weight REAL DEFAULT 1.0
            );
        """)
        self._conn.commit()

    # ── working-memory helpers ─────────────────────────────────────────────
    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Stores item in LRU working memory with Bayesian importance; returns None."""
        with self._lock:
            if key in self._wm:
                self._wm.move_to_end(key)
                self._wm[key]['access_count'] += 1
                self._wm[key]['importance'] = min(1.0, importance + 0.05 * self._wm[key]['access_count'])
            else:
                self._wm[key] = {'value': value, 'importance': importance,
                                 'ts': time.time(), 'access_count': 0}
                self._wm.move_to_end(key)
            if len(self._wm) > self._CAPACITY:
                self.forget_least_important()

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieves decayed item from working memory; returns value or None."""
        with self._lock:
            if key not in self._wm:
                return None
            item = self._wm[key]
            elapsed = time.time() - item['ts']
            item['importance'] *= math.exp(-self._DECAY * elapsed)
            item['importance'] = min(1.0, item['importance'] * (1 + 0.1 * item['access_count']))
            item['ts'] = time.time()
            item['access_count'] += 1
            self._wm.move_to_end(key)
            return item['value']

    def consolidate(self) -> Dict[str, Any]:
        """Decays all items and evicts below-threshold entries; returns summary dict."""
        evicted = 0
        with self._lock:
            now = time.time()
            dead = [k for k, v in self._wm.items()
                    if v['importance'] * math.exp(-self._DECAY * (now - v['ts'])) < 0.01]
            for k in dead:
                del self._wm[k]; evicted += 1
        return {'evicted': evicted, 'remaining': len(self._wm)}

    def capacity_used(self) -> float:
        """Returns fraction (0-1) of working-memory capacity currently occupied."""
        with self._lock:
            return len(self._wm) / self._CAPACITY

    def forget_least_important(self) -> Optional[str]:
        """Evicts lowest-importance item from working memory; returns evicted key."""
        if not self._wm:
            return None
        now = time.time()
        worst = min(self._wm, key=lambda k: self._wm[k]['importance'] *
                    math.exp(-self._DECAY * (now - self._wm[k]['ts'])))
        del self._wm[worst]
        return worst

    # ── core planner ──────────────────────────────────────────────────────────
    def plan(self, goal: str, steps_list: List[str]) -> str:
        """Creates a new plan with ordered steps; returns plan_id string."""
        plan_id = hashlib.md5(f"{goal}{time.time()}".encode()).hexdigest()[:12]
        c = self._conn.cursor()
        with self._lock:
            c.execute("INSERT INTO plans VALUES (?,?,?,?)",
                      (plan_id, goal, time.time(), 1.0))
            for i, s in enumerate(steps_list):
                c.execute("INSERT INTO steps (plan_id,idx,description,weight) VALUES (?,?,?,?)",
                          (plan_id, i, s, 1.0))
            self._conn.commit()
        self.store(f'plan:{plan_id}', {'goal': goal, 'steps': len(steps_list)}, importance=0.9)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            HGP().add_goal(f"Complete plan '{goal}' ({len(steps_list)} steps)", priority=7)
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning('task_planner', 'plan_creation', 1.0, True)
        except Exception:
            pass
        return plan_id

    def complete(self, plan_id: str, idx: int) -> Dict[str, Any]:
        """Marks step idx done in plan; returns updated progress dict with confidence."""
        c = self._conn.cursor()
        with self._lock:
            c.execute("UPDATE steps SET done=1, completed_at=? WHERE plan_id=? AND idx=?",
                      (time.time(), plan_id, idx))
            self._conn.commit()
        prog = self.progress(plan_id)
        # Bayesian posterior update: prior=conf, likelihood=0.9 if step done cleanly
        c.execute("SELECT confidence FROM plans WHERE id=?", (plan_id,))
        row = c.fetchone()
        if row:
            prior = row[0]; likelihood = 0.9
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-12)
            with self._lock:
                c.execute("UPDATE plans SET confidence=? WHERE id=?", (posterior, plan_id))
                self._conn.commit()
            prog['confidence'] = round(posterior, 4)
        actual = prog['percent'] / 100.0
        self._history.append((self._ema_pred, actual))
        self._ema_pred = 0.15 * actual + 0.85 * self._ema_pred
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning('task_planner', 'step_complete', prog.get('confidence', 0.9), True)
        except Exception:
            pass
        return prog

    def progress(self, plan_id: str) -> Dict[str, Any]:
        """Returns percent done, step counts, confidence, and calibration error for plan."""
        c = self._conn.cursor()
        c.execute("SELECT COUNT(*), SUM(done) FROM steps WHERE plan_id=?", (plan_id,))
        total, done = c.fetchone()
        total = total or 0; done = done or 0
        pct = round((done / total * 100) if total else 0.0, 2)
        c.execute("SELECT confidence FROM plans WHERE id=?", (plan_id,))
        row = c.fetchone(); conf = round(row[0], 4) if row else 1.0
        mae = 0.0
        if len(self._history) >= 2:
            mae = round(statistics.mean(abs(p - a) for p, a in self._history[-50:]), 4)
        # z-score anomaly on progress
        if len(self._history) >= 5:
            vals = [a for _, a in self._history[-20:]]
            mu = statistics.mean(vals); sigma = statistics.stdev(vals) + 1e-9
            z = round((pct / 100.0 - mu) / sigma, 3)
        else:
            z = 0.0
        return {'plan_id': plan_id, 'percent': pct, 'done': done,
                'total': total, 'confidence': conf,
                'calibration_mae': mae, 'anomaly_z': z,
                'cycles': self._cycles, 'active': 1, 'pending': total - done,
                'entropy': round(-conf * math.log2(conf + 1e-12), 4)}

    def next_step(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Returns first incomplete step dict or None if all steps done."""
        c = self._conn.cursor()
        c.execute("SELECT idx, description FROM steps WHERE plan_id=? AND done=0 ORDER BY idx LIMIT 1",
                  (plan_id,))
        row = c.fetchone()
        if not row:
            return None
        idx, desc = row
        # long-horizon confidence: product of remaining step confidences
        c.execute("SELECT COUNT(*) FROM steps WHERE plan_id=? AND done=0", (plan_id,))
        remaining = c.fetchone()[0]
        c.execute("SELECT confidence FROM plans WHERE id=?", (plan_id,))
        pr = c.fetchone(); plan_conf = pr[0] if pr else 1.0
        horizon_conf = round(plan_conf ** max(remaining, 1), 4)
        replan = horizon_conf < self._CONF_FLOOR
        cached = self.retrieve(f'step:{plan_id}:{idx}')
        if cached:
            return {**cached, 'horizon_conf': horizon_conf, 'replan': replan}
        result = {'idx': idx, 'description': desc,
                  'horizon_conf': horizon_conf, 'replan': replan}
        self.store(f'step:{plan_id}:{idx}', result, importance=0.8)
        return result

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        c = self._conn.cursor()
        c.execute("SELECT COUNT(*) FROM plans")
        items = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM steps WHERE done=0")
        pending = c.fetchone()[0]
        c.execute("SELECT AVG(confidence) FROM plans")
        avg_conf = c.fetchone()[0] or 1.0
        mae = 0.0
        if len(self._history) >= 2:
            mae = round(statistics.mean(abs(p - a) for p, a in self._history[-50:]), 4)
        return {'items': items, 'pending': pending, 'confidence': round(avg_conf, 4),
                'accuracy': round(1.0 - mae, 4), 'cycles': self._cycles,
                'active': 1, 'capacity': round(self.capacity_used(), 4),
                'entropy': round(-avg_conf * math.log2(avg_conf + 1e-12), 4)}

    # ── autonomous loop ────────────────────────────────────────────────────────
    def _auto_loop(self) -> None:
        while True:
            time.sleep(self._CYCLE_SEC)
            try:
                self._cycles += 1
                self.consolidate()
                c = self._conn.cursor()
                c.execute("SELECT id FROM plans")
                for (pid,) in c.fetchall():
                    prog = self.progress(pid)
                    if prog['replan'] if 'replan' in prog else False:
                        try:
                            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
                            HGP().add_goal(f"Replan low-confidence plan {pid}", priority=8)
                        except Exception:
                            pass
                try:
                    from MetacognitiveMonitor import MetacognitiveMonitor as MCM
                    MCM().log_reasoning('task_planner', 'auto_cycle', 0.85, True)
                except Exception:
                    pass
            except Exception:
                pass

# Usage: obj = TaskPlanner() | plan_id = obj.plan("Launch product", ["Research","Build","Test","Deploy"]) | obj.complete(plan_id, 0) | obj.progress(plan_id) | obj.next_step(plan_id)
