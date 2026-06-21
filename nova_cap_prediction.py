"""
nova_cap_prediction.py
Nova ASI — Prediction
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
Nova Temporal Prediction Engine — linear-regression time-series predictor with
Bayesian-calibrated confidence intervals, EMA-tracked MAE, anomaly detection,
autonomous forecasting daemon, and full cross-system integration.
"""

import math
import time
import threading
import statistics
import sqlite3
import os
import json
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional

class TemporalPredictionEngine:
    """ASI-grade time-series predictor: linear regression, CI bounds, causal feedback, autonomous cycling."""

    DB_PATH = "nova_temporal_prediction.db"
    WINDOW = 20
    EMA_ALPHA = 0.15
    ANOMALY_Z = 3.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observations: List[float] = []
        self._timestamps: List[float] = []
        self._predictions: List[float] = []
        self._actuals: List[float] = []
        self._ema_mae: float = 0.0
        self._cycles: int = 0
        self._anomaly_count: int = 0
        self._last_slope: float = 0.0
        self._last_intercept: float = 0.0
        self._std_error: float = 1.0
        self._init_db()
        self._load_state()
        self._start_daemon()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.DB_PATH)
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS observations
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts REAL, value REAL, prediction REAL, anomaly INTEGER)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS meta
                (key TEXT PRIMARY KEY, value TEXT)""")
            conn.commit()
        finally:
            conn.close()

    def _load_state(self) -> None:
        conn = sqlite3.connect(self.DB_PATH)
        try:
            rows = conn.execute(
                "SELECT ts, value, prediction FROM observations ORDER BY id DESC LIMIT 40"
            ).fetchall()
            for ts, val, pred in reversed(rows):
                self._observations.append(val)
                self._timestamps.append(ts)
                if pred is not None:
                    self._predictions.append(pred)
                    self._actuals.append(val)
            row = conn.execute("SELECT value FROM meta WHERE key='ema_mae'").fetchone()
            if row:
                self._ema_mae = float(row[0])
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    def _save_observation(self, ts: float, value: float, prediction: float, anomaly: int) -> None:
        conn = sqlite3.connect(self.DB_PATH)
        try:
            conn.execute(
                "INSERT INTO observations (ts, value, prediction, anomaly) VALUES (?,?,?,?)",
                (ts, value, prediction, anomaly)
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('ema_mae', ?)",
                (str(self._ema_mae),)
            )
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    def _linear_regression(self) -> Tuple[float, float, float]:
        """Returns (slope, intercept, std_error) over the last WINDOW observations."""
        window = self._observations[-self.WINDOW:]
        n = len(window)
        if n < 2:
            return 0.0, window[-1] if window else 0.0, 1.0
        xs = list(range(n))
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(window)
        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, window))
        slope = ss_xy / (ss_xx + 1e-12)
        intercept = mean_y - slope * mean_x
        residuals = [(window[i] - (slope * xs[i] + intercept)) for i in range(n)]
        mse = sum(r ** 2 for r in residuals) / max(n - 2, 1)
        std_error = math.sqrt(mse * (1 + 1 / n + (n - mean_x) ** 2 / (ss_xx + 1e-12)))
        return slope, intercept, std_error

    def _z_score_for_alpha(self, alpha: float) -> float:
        """Approximates z-score for two-tailed CI using rational approximation."""
        p = 1.0 - (1.0 - alpha) / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p + 1e-12))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t ** 2) / \
            (1 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3)
        return max(z, 0.0)

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            self.auto_cycle()

    def observe(self, value: float) -> Dict:
        """Records a new observation, updates EMA-MAE, detects anomalies; returns anomaly flag and z-score."""
        ts = time.time()
        anomaly = 0
        z = 0.0
        prior_pred = 0.0
        with self._lock:
            if len(self._predictions) > 0 and len(self._actuals) < len(self._predictions):
                error = abs(value - self._predictions[-1])
                self._ema_mae = self.EMA_ALPHA * error + (1 - self.EMA_ALPHA) * self._ema_mae
                self._actuals.append(value)
            if len(self._observations) >= 4:
                recent = self._observations[-20:]
                mean_r = statistics.mean(recent)
                std_r = statistics.stdev(recent) if len(recent) > 1 else 1.0
                z = (value - mean_r) / (std_r + 1e-9)
                if abs(z) > self.ANOMALY_Z:
                    anomaly = 1
                    self._anomaly_count += 1
            self._observations.append(value)
            self._timestamps.append(ts)
            if len(self._observations) > 200:
                self._observations.pop(0)
                self._timestamps.pop(0)
            slope, intercept, std_err = self._linear_regression()
            self._last_slope = slope
            self._last_intercept = intercept
            self._std_error = std_err
            prior_pred = slope * (len(self._observations[-self.WINDOW:])) + intercept
            self._predictions.append(prior_pred)
            if len(self._predictions) > 200:
                self._predictions.pop(0)
                self._actuals = self._actuals[-200:]
        self._save_observation(ts, value, prior_pred, anomaly)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("prediction", "linear_regression_ema",
                                                  1.0 - min(self._ema_mae / (abs(value) + 1e-9), 1.0), anomaly == 0)
        except Exception:
            pass
        return {"value": value, "anomaly": bool(anomaly), "z_score": round(z, 4), "ema_mae": round(self._ema_mae, 6)}

    def predict_next(self) -> Dict:
        """Returns next predicted value with slope, intercept, and current std_error."""
        with self._lock:
            if not self._observations:
                return {"prediction": 0.0, "slope": 0.0, "intercept": 0.0, "std_error": 1.0, "confidence": 0.0}
            slope, intercept, std_err = self._linear_regression()
            self._last_slope = slope
            self._last_intercept = intercept
            self._std_error = std_err
            n = len(self._observations[-self.WINDOW:])
            pred = slope * n + intercept
            conf = max(0.0, 1.0 - std_err / (abs(statistics.mean(self._observations[-self.WINDOW:])) + 1e-9))
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Validate prediction={pred:.4f} against next observation", priority=2)
        except Exception:
            pass
        return {"prediction": round(pred, 6), "slope": round(slope, 6),
                "intercept": round(intercept, 6), "std_error": round(std_err, 6),
                "confidence": round(min(conf, 1.0), 4)}

    def confidence_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """Returns (lower, upper) CI bounds for next prediction at given confidence level."""
        with self._lock:
            if not self._observations:
                return (0.0, 0.0)
            n = len(self._observations[-self.WINDOW:])
            pred = self._last_slope * n + self._last_intercept
            z = self._z_score_for_alpha(alpha)
            margin = z * self._std_error
        return (round(pred - margin, 6), round(pred + margin, 6))

    def trend(self) -> Dict:
        """Returns slope direction, magnitude, entropy of residuals, and trend label."""
        with self._lock:
            if len(self._observations) < 3:
                return {"label": "insufficient_data", "slope": 0.0, "entropy": 0.0}
            slope, intercept, std_err = self._linear_regression()
            window = self._observations[-self.WINDOW:]
            preds = [slope * i + intercept for i in range(len(window))]
            residuals = [abs(w - p) for w, p in zip(window, preds)]
            total = sum(residuals) + 1e-12
            dist = [r / total for r in residuals]
            entropy = -sum(p * math.log2(p + 1e-12) for p in dist)
            label = "rising" if slope > 0.01 else "falling" if slope < -0.01 else "stable"
        return {"label": label, "slope": round(slope, 6), "entropy": round(entropy, 4),
                "std_error": round(std_err, 6), "window_size": len(window)}

    def accuracy_last_n(self, n: int = 10) -> Dict:
        """Returns 1 - MAE/mean accuracy metric and raw MAE for last n prediction pairs."""
        with self._lock:
            pairs = list(zip(self._predictions, self._actuals))[-n:]
            if not pairs:
                return {"accuracy": 0.0, "mae": 0.0, "n": 0}
            mae = statistics.mean(abs(p - a) for p, a in pairs)
            mean_val = statistics.mean(a for _, a in pairs)
            accuracy = max(0.0, 1.0 - mae / (abs(mean_val) + 1e-9))
        return {"accuracy": round(accuracy, 4), "mae": round(mae, 6),
                "n": len(pairs), "mean_actual": round(mean_val, 6)}

    def anomaly_report(self) -> Dict:
        """Returns total anomaly count, rate, and recent z-score distribution summary."""
        with self._lock:
            total = len(self._observations)
            rate = self._anomaly_count / max(total, 1)
        return {"total_observations": total, "anomaly_count": self._anomaly_count,
                "anomaly_rate": round(rate, 4), "ema_mae": round(self._ema_mae, 6)}

    def auto_cycle(self) -> Dict:
        """Runs one autonomous maintenance cycle: recomputes regression, logs to MetacognitiveMonitor."""
        with self._lock:
            self._cycles += 1
            if len(self._observations) < 2:
                return {"status": "insufficient_data", "cycles": self._cycles}
            slope, intercept, std_err = self._linear_regression()
            self._last_slope = slope
            self._last_intercept = intercept
            self._std_error = std_err
            acc = self.accuracy_last_n(10)
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("prediction_auto_cycle", "regression_refresh",
                                                  acc["accuracy"], acc["accuracy"] > 0.7)
        except Exception:
            pass
        return {"status": "ok", "cycles": self._cycles, "slope": round(slope, 6),
                "std_error": round(std_err, 6), "accuracy": acc["accuracy"]}

    def status(self) -> Dict:
        """Returns plain dict with numeric keys required for ConsciousnessIntegrator Φ computation."""
        with self._lock:
            n_obs = len(self._observations)
            n_pairs = len(list(zip(self._predictions, self._actuals)))
            conf = max(0.0, 1.0 - self._std_error / (abs(statistics.mean(self._observations[-self.WINDOW:])) + 1e-9)) if self._observations else 0.0
        acc = self.accuracy_last_n(10)
        return {
            "items": n_obs,
            "confidence": round(min(conf, 1.0), 4),
            "accuracy": acc["accuracy"],
            "active": 1,
            "cycles": self._cycles,
            "pending": max(0, len(self._predictions) - n_pairs),
            "entropy": round(self.trend().get("entropy", 0.0), 4),
            "anomaly_count": self._anomaly_count,
            "ema_mae": round(self._ema_mae, 6),
        }

# Usage: obj = TemporalPredictionEngine() | result = obj.observe(42.7)