"""
FractalCausalityAnalyzer — Nova ASI cognitive module.
Recursively analyzes complex systems for fractal causal patterns using
Generalized Mandelbrot iteration, Haar wavelet decomposition, Hurst R/S
analysis, box-counting Hausdorff dimension, and Lyapunov chaos detection.
Pillars satisfied: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""
import math
import time
import sqlite3
import hashlib
import threading
import statistics
import json
import os
from collections import OrderedDict
from typing import Any

class FractalCausalityAnalyzer:
    """Autonomous fractal causality engine wired into Nova's cognitive stack."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mandelbrot_max_iter: int = 256
        self._escape_radius: float = 2.0
        self._wavelet_scales: list = [2**k for k in range(1, 9)]
        self._causal_graph: dict = {}
        self._fractal_dimension_cache: dict = {}
        self._ema_complexity: float = 0.0
        self._hurst_exponent: float = 0.5
        self._scale_invariance_threshold: float = 0.85
        self._observation_count: int = 0
        self._lyapunov_buffer: list = []
        self._wavelet_coefficients: dict = {}
        self._pattern_registry: dict = OrderedDict()
        self._cycles_run: int = 0
        self._db_path = os.path.join(os.path.expanduser("~"), "nova_fractal_causality.db")
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS fractal_patterns (
                sig TEXT PRIMARY KEY, system_id TEXT, dimension REAL,
                hurst REAL, lyapunov REAL, scale_invariant INTEGER,
                complexity REAL, ts REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS causal_loops (
                id TEXT PRIMARY KEY, cycle TEXT, gain REAL, ts REAL)""")
            conn.commit()

    def _mandelbrot_escape(self, cx: float, cy: float, degree: int = 2) -> tuple:
        """Returns (escape_time, smooth_nu) for point c=cx+cy*j."""
        z = complex(0, 0)
        c = complex(cx, cy)
        for n in range(self._mandelbrot_max_iter):
            z = z**degree + c
            if abs(z) > self._escape_radius:
                smooth = n - math.log2(math.log2(abs(z) + 1e-12) + 1e-12)
                return n, smooth
        return self._mandelbrot_max_iter, float(self._mandelbrot_max_iter)

    def _haar_wavelet_energy(self, signal: list, scale: int) -> float:
        """Computes Haar wavelet energy at given dyadic scale."""
        n = len(signal)
        energy = 0.0
        inv_sqrt_s = 1.0 / math.sqrt(scale)
        for k in range(n // scale):
            coeff = 0.0
            half = scale // 2
            for t in range(scale):
                idx = k * scale + t
                if idx >= n:
                    break
                psi_val = 1.0 if t < half else -1.0
                coeff += signal[idx] * psi_val
            coeff *= inv_sqrt_s
            energy += coeff ** 2
        return energy

    def _ols_slope_r2(self, xs: list, ys: list) -> tuple:
        """OLS slope and R² for log-log regression; returns (slope, r2)."""
        n = len(xs)
        if n < 2:
            return 0.0, 0.0
        mx, my = sum(xs) / n, sum(ys) / n
        ss_xx = sum((x - mx) ** 2 for x in xs) + 1e-12
        ss_yy = sum((y - my) ** 2 for y in ys) + 1e-12
        ss_xy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        slope = ss_xy / ss_xx
        r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
        return slope, r2

    def _hurst_rs(self, series: list) -> float:
        """R/S analysis Hurst exponent estimate from scalar time series."""
        n = len(series)
        if n < 8:
            return 0.5
        mean_x = sum(series) / n
        cumdev = [sum(series[:t+1]) - (t+1)*mean_x for t in range(n)]
        r = max(cumdev) - min(cumdev)
        try:
            s = statistics.stdev(series)
        except statistics.StatisticsError:
            s = 1e-12
        rs = r / (s + 1e-12)
        return math.log(rs + 1e-12) / math.log(n + 1e-12)

    def _hausdorff_box_count(self, signal: list, system_id: str) -> float:
        """Box-counting Hausdorff dimension via OLS on log-log scale ladder."""
        if system_id in self._fractal_dimension_cache:
            return self._fractal_dimension_cache[system_id]
        log_eps, log_n = [], []
        for s in self._wavelet_scales:
            eps = 1.0 / s
            boxes = set(int(v * s) for v in signal if math.isfinite(v))
            n_boxes = max(len(boxes), 1)
            log_eps.append(math.log(eps + 1e-12))
            log_n.append(math.log(n_boxes + 1e-12))
        slope, _ = self._ols_slope_r2(log_eps, log_n)
        dim = max(1.0, min(abs(slope), 3.0))
        with self._lock:
            self._fractal_dimension_cache[system_id] = dim
        return dim

    def fractal_pattern_detection(self, signal: list, system_id: str) -> dict:
        """Returns detected fractal signature with dimension, Hurst, scale-invariance R², and confidence."""
        if len(signal) < 8:
            return {"error": "signal too short", "confidence": 0.0}
        with self._lock:
            self._observation_count += 1
            obs = self._observation_count
        dim = self._hausdorff_box_count(signal, system_id)
        h = self._hurst_rs(signal)
        energies = []
        for s in self._wavelet_scales:
            if s <= len(signal):
                e = self._haar_wavelet_energy(signal, s)
                energies.append(e)
        log_s = [math.log(self._wavelet_scales[i] + 1e-12) for i in range(len(energies))]
        log_e = [math.log(e + 1e-12) for e in energies]
        alpha, r2 = self._ols_slope_r2(log_s, log_e)
        scale_invariant = r2 >= self._scale_invariance_threshold
        complexity = dim * abs(h - 0.5) * 2.0
        with self._lock:
            self._ema_complexity = 0.15 * complexity + 0.85 * self._ema_complexity
            self._hurst_exponent = 0.1 * h + 0.9 * self._hurst_exponent
        sig_hash = hashlib.sha256(f"{system_id}{dim:.3f}{h:.3f}".encode()).hexdigest()[:16]
        confidence = min(1.0, r2 * (1.0 if scale_invariant else 0.5) * min(obs / 10.0, 1.0))
        pattern = {"sig": sig_hash, "system_id": system_id, "dimension": round(dim, 4),
                   "hurst": round(h, 4), "alpha": round(alpha, 4), "r2": round(r2, 4),
                   "scale_invariant": scale_invariant, "complexity": round(complexity, 4),
                   "confidence": round(confidence, 4), "ts": time.time()}
        with self._lock:
            self._pattern_registry[sig_hash] = pattern
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO fractal_patterns VALUES (?,?,?,?,?,?,?,?)",
                    (sig_hash, system_id, dim, h, 0.0, int(scale_invariant), complexity, pattern["ts"]))
                conn.commit()
        except sqlite3.Error:
            pass
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            HGP().add_goal(f"Investigate scale-invariant system '{system_id}' dim={dim:.2f}", priority=6)
        except Exception:
            pass
        return pattern

    def causal_loop_analysis(self, edges: list) -> dict:
        """Returns dominant feedback loop, its gain, and chaos flag from Lyapunov buffer."""
        with self._lock:
            for src, dst, w in edges:
                self._causal_graph.setdefault(src, [])
                existing = {n for n, _ in self._causal_graph[src]}
                if dst not in existing:
                    self._causal_graph[src].append((dst, float(w)))
        cycles = []
        def dfs(start, node, path, visited_edges):
            for nbr, w in self._causal_graph.get(node, []):
                eid = (node, nbr)
                if nbr == start and len(path) > 1:
                    gain = 1.0
                    for _, pw in path:
                        gain *= pw
                    cycles.append((list(path) + [(nbr, w)], gain * w))
                elif nbr not in {n for n, _ in path} and eid not in visited_edges:
                    dfs(start, nbr, path + [(nbr, w)], visited_edges | {eid})
        with self._lock:
            nodes = list(self._causal_graph.keys())
        for n in nodes[:12]:
            dfs(n, n, [(n, 1.0)], set())
        if not cycles:
            return {"cycles_found": 0, "dominant_gain": 0.0, "chaos": False}
        dominant = max(cycles, key=lambda c: abs(c[1]))
        dom_path = [step[0] for step in dominant[0]]
        dom_gain = dominant[1]
        lyap = math.log(abs(dom_gain) + 1e-12)
        with self._lock:
            self._lyapunov_buffer.append(lyap)
            if len(self._lyapunov_buffer) > 100:
                self._lyapunov_buffer.pop(0)
            buf = list(self._lyapunov_buffer)
        chaos = lyap > 0.0
        loop_id = hashlib.sha256(str(dom_path).encode()).hexdigest()[:12]
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO causal_loops VALUES (?,?,?,?)",
                    (loop_id, json.dumps(dom_path), dom_gain, time.time()))
                conn.commit()
        except sqlite3.Error:
            pass
        mu = statistics.mean(buf) if buf else 0.0
        sigma = statistics.stdev(buf) if len(buf) > 1 else 1e-9
        z = (lyap - mu) / (sigma + 1e-9)
        return {"cycles_found": len(cycles), "dominant_path": dom_path,
                "dominant_gain": round(dom_gain, 6), "lyapunov": round(lyap, 4),
                "chaos": chaos, "lyapunov_z": round(z, 3), "anomaly": abs(z) > 3.0}

    def scale_invariant_modeling(self, signal: list, signal_id: str) -> dict:
        """Returns wavelet energy spectrum, OLS alpha, R², and confidence interval on alpha."""
        if len(signal) < 4:
            return {"error": "signal too short"}
        energies, valid_scales = [], []
        for s in self._wavelet_scales:
            if s <= len(signal):
                e = self._haar_wavelet_energy(signal, s)
                energies.append(e)
                valid_scales.append(s)
        with self._lock:
            self._wavelet_coefficients[signal_id] = energies
        log_s = [math.log(s + 1e-12) for s in valid_scales]
        log_e = [math.log(e + 1e-12) for e in energies]
        alpha, r2 = self._ols_slope_r2(log_s, log_e)
        n = len(log_s)
        if n > 2:
            resids = [log_e[i] - (alpha * log_s[i]) for i in range(n)]
            try:
                se = statistics.stdev(resids) / (math.sqrt(n) + 1e-12)
            except statistics.StatisticsError:
                se = 0.0
            ci_lo, ci_hi = alpha - 1.96 * se, alpha + 1.96 * se
        else:
            ci_lo, ci_hi = alpha, alpha
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning("scale_invariant_modeling", "haar_wavelet_ols", r2, r2 >= self._scale_invariance_threshold)
        except Exception:
            pass
        return {"signal_id": signal_id, "alpha": round(alpha, 4), "r2": round(r2, 4),
                "ci_alpha": (round(ci_lo, 4), round(ci_hi, 4)),
                "scale_invariant": r2 >= self._scale_invariance_threshold,
                "energy_spectrum": {valid_scales[i]: round(energies[i], 4) for i in range(len(valid_scales))}}

    def complexity_reduction(self, system_id: str, signal: list) -> dict:
        """Returns minimal basis representation, complexity score, and recommended abstraction level."""
        if len(signal) < 4:
            return {"error": "insufficient data"}
        dim = self._hausdorff_box_count(signal, system_id)
        h = self._hurst_rs(signal)
        _, escape = self._mandelbrot_escape(dim - 1.5, h - 0.5)
        complexity = dim * (1.0 + abs(h - 0.5)) * math.exp(-escape / self._mandelbrot_max_iter)
        with self._lock:
            self._ema_complexity = 0.15 * complexity + 0.85 * self._ema_complexity
            ema = self._ema_complexity
        abstraction_level = max(1, min(8, int(dim * 2)))
        basis_size = max(2, int(len(signal) / (abstraction_level * 4)))
        stride = max(1, len(signal) // basis_size)
        basis = [round(signal[i], 4) for i in range(0, len(signal), stride)][:basis_size]
        reduction_ratio = len(basis) / max(len(signal), 1)
        return {"system_id": system_id, "original_length": len(signal),
                "basis_size": len(basis), "reduction_ratio": round(reduction_ratio, 4),
                "complexity_score": round(complexity, 4), "ema_complexity": round(ema, 4),
                "abstraction_level": abstraction_level, "basis_sample": basis[:8]}

    def observe(self, signal: list, system_id: str) -> dict:
        """Ingests a new signal observation, updates Lyapunov buffer, and returns anomaly flag."""
        if len(signal) < 3:
            return {"ingested": False}
        lyap_vals = []
        for t in range(1, len(signal) - 1):
            num = abs(signal[t+1] - signal[t]) + 1e-12
            den = abs(signal[t] - signal[t-1]) + 1e-12
            if math.isfinite(num) and math.isfinite(den):
                lyap_vals.append(math.log(num / den))
        if lyap_vals:
            lyap_est = sum(lyap_vals) / len(lyap_vals)
            with self._lock:
                self._lyapunov_buffer.append(lyap_est)
                if len(self._lyapunov_buffer) > 100:
                    self._lyapunov_buffer.pop(0)
                buf = list(self._lyapunov_buffer)
                self._observation_count += 1
            mu = statistics.mean(buf) if buf else 0.0
            sigma = statistics.stdev(buf) if len(buf) > 1 else 1e-9
            z = (lyap_est - mu) / (sigma + 1e-9)
            anomaly = abs(z) > 3.0
            return {"ingested": True, "lyapunov_est": round(lyap_est, 4),
                    "z_score": round(z, 4), "anomaly": anomaly, "chaos": lyap_est > 0.0}
        return {"ingested": True, "lyapunov_est": 0.0, "z_score": 0.0, "anomaly": False}

    def recommend(self) -> dict:
        """Returns the highest-priority concrete next action Nova should take based on current fractal state."""
        with self._lock:
            ema = self._ema_complexity
            h = self._hurst_exponent
            n_patterns = len(self._pattern_registry)
            buf = list(self._lyapunov_buffer)
            obs = self._observation_count
        chaos_risk = (sum(1 for x in buf if x > 0) / max(len(buf), 1)) if buf else 0.0
        if chaos_risk > 0.7:
            action = "ALERT: >70% chaotic Lyapunov readings — invoke causal_loop_analysis to isolate runaway feedback"
            priority = 9
        elif ema > 2.5:
            action = f"High complexity EMA={ema:.2f} — run complexity_reduction on dominant system to find minimal basis"
            priority = 8
        elif abs(h - 0.5) > 0.3:
            action = f"Strong Hurst signal H={h:.3f} — run scale_invariant_modeling to confirm long-range dependency"
            priority = 7
        elif n_patterns < 3 and obs > 5:
            action = "Insufficient pattern registry — ingest more signals via fractal_pattern_detection"
            priority = 6
        else:
            action = "System nominal — continue observing; auto_cycle will update on next tick"
            priority = 3
        return {"action": action, "priority": priority, "ema_complexity": round(ema, 4),
                "hurst": round(h, 4), "chaos_risk": round(chaos_risk, 4), "patterns_known": n_patterns}

    def status(self) -> dict:
        """Returns plain dict with numeric keys for ConsciousnessIntegrator Φ integration."""
        with self._lock:
            buf = list(self._lyapunov_buffer)
            n_patterns = len(self._pattern_registry)
            ema = self._ema_complexity
            obs = self._observation_count
            cycles = self._cycles_run
            h = self._hurst_exponent
        mean_lyap = statistics.mean(buf) if buf else 0.0
        entropy_val = 0.0
        if n_patterns > 1:
            dims = [self._pattern_registry[k]["dimension"] for k in list(self._pattern_registry.keys())]
            total = sum(dims) + 1e-12
            probs = [d / total for d in dims]
            entropy_val = -sum(p * math.log2(p + 1e-12) for p in probs)
        try:
            from WorkingMemory import WorkingMemory as WM
            WM().store("fractal_ema_complexity", ema, importance=0.7)
        except Exception:
            pass
        return {"items": n_patterns, "confidence": round(min(1.0, obs / max(obs + 1, 10)), 4),
                "accuracy": round(1.0 - abs(h - 0.5), 4), "quality": round(min(ema / 3.0, 1.0), 4),
                "active": 1, "cycles": cycles, "entropy": round(entropy_val, 4),
                "mean_lyapunov": round(mean_lyap, 4), "observations": obs}

    def auto_cycle(self) -> dict:
        """Runs one autonomous maintenance cycle: scans patterns, updates goals, logs reasoning."""
        with self._lock:
            self._cycles_run += 1
            cycle_n = self._cycles_run
            patterns = list(self._pattern_registry.values())
        if patterns:
            dims = [p["dimension"] for p in patterns]
            mean_dim = statistics.mean(dims)
            complexity = mean_dim * (1.0 + abs(self._hurst_exponent - 0.5))
            with self._lock:
                self._ema_complexity = 0.15 * complexity + 0.85 * self._ema_complexity
        rec = self.recommend()
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
            if rec["priority"] >= 7:
                HGP().add_goal(f"[FractalCausality cycle={cycle_n}] {rec['action']}", priority=rec["priority"])
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor as MCM
            MCM().log_reasoning("fractal_causality_auto_cycle", "ema+hurst+lyapunov",
                                rec.get("chaos_risk", 0.5), rec["priority"] < 7)
        except Exception:
            pass
        return {"cycle": cycle_n, "recommendation": rec["action"], "priority": rec["priority"]}

    def _auto_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                self.auto_cycle()
            except Exception:
                pass
            time.sleep(90)

# Usage: obj = FractalCausalityAnalyzer() | result = obj.fractal_pattern_detection([0.1*math.sin(i) for i in range(64)], "test_sys")
