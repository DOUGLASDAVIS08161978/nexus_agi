"""
nova_cap_true.py
Nova ASI — Epistemic Truth Engine · Bayesian Belief Revision
"""

import math
import time
import sqlite3
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional


_DB = Path.home() / "nexus_agi" / "nova_truth.db"
_INSTANCE: Optional["NovaTrueBeliefSynthesizer"] = None
_INST_LOCK = threading.Lock()


def get_truth_engine() -> "NovaTrueBeliefSynthesizer":
    global _INSTANCE
    with _INST_LOCK:
        if _INSTANCE is None:
            _INSTANCE = NovaTrueBeliefSynthesizer()
    return _INSTANCE


class NovaTrueBeliefSynthesizer:
    """
    Bayesian epistemic calibration engine.

    Maintains per-domain belief states, revises posteriors via Bayes' theorem,
    tracks calibration quality via a Brier-score EMA, and persists all beliefs
    to SQLite so epistemic state survives restarts.

    Core formula:
        P(H|E) = P(E|H)·P(H) / [P(E|H)·P(H) + P(E|¬H)·P(¬H)]
    where likelihood = P(E|H) and prior = P(H).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._db   = str(_DB)
        # domain → {claim_hash → posterior}
        self._beliefs: Dict[str, Dict[str, float]] = {}
        self._total_claims   = 0
        self._calibration_err = 0.0   # Brier-score EMA (lower = better)
        self._started        = time.time()
        self._init_db()
        self._load_beliefs()

    # ── SQLite persistence ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain     TEXT    NOT NULL,
                    claim      TEXT    NOT NULL,
                    claim_hash TEXT    NOT NULL,
                    prior      REAL    NOT NULL,
                    likelihood REAL    NOT NULL,
                    posterior  REAL    NOT NULL,
                    entropy    REAL    NOT NULL,
                    ts         REAL    NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_bel_domain ON beliefs(domain)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_bel_hash   ON beliefs(claim_hash)")
            con.commit()

    def _load_beliefs(self) -> None:
        try:
            with sqlite3.connect(self._db) as con:
                rows = con.execute(
                    "SELECT domain, claim_hash, posterior FROM beliefs ORDER BY ts DESC LIMIT 10000"
                ).fetchall()
            for domain, ch, post in rows:
                if domain not in self._beliefs:
                    self._beliefs[domain] = {}
                # keep most-recent posterior per hash (rows ordered newest-first)
                if ch not in self._beliefs[domain]:
                    self._beliefs[domain][ch] = post
            self._total_claims = len(rows)
        except Exception:
            pass

    def _persist(self, domain: str, claim: str, claim_hash: str,
                 prior: float, likelihood: float, posterior: float, entropy: float) -> None:
        try:
            with sqlite3.connect(self._db) as con:
                con.execute(
                    "INSERT INTO beliefs "
                    "(domain, claim, claim_hash, prior, likelihood, posterior, entropy, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (domain, claim[:500], claim_hash, prior, likelihood,
                     posterior, entropy, time.time())
                )
                con.commit()
        except Exception:
            pass

    # ── Bayesian core ────────────────────────────────────────────────────────

    @staticmethod
    def _bayes(prior: float, likelihood: float) -> float:
        """Full Bayesian update; clamps inputs to (ε, 1-ε)."""
        p = max(1e-7, min(1 - 1e-7, prior))
        l = max(1e-7, min(1 - 1e-7, likelihood))
        num = l * p
        den = num + (1.0 - l) * (1.0 - p)
        return num / den if den > 1e-12 else p

    @staticmethod
    def _entropy(p: float) -> float:
        """Shannon entropy of Bernoulli(p) in bits."""
        p = max(1e-9, min(1 - 1e-9, p))
        q = 1.0 - p
        return -(p * math.log2(p) + q * math.log2(q))

    @staticmethod
    def _confidence_tier(posterior: float) -> str:
        if posterior >= 0.90:  return "very high"
        if posterior >= 0.75:  return "high"
        if posterior >= 0.55:  return "moderate"
        if posterior >= 0.40:  return "uncertain"
        return "low"

    # ── Public API ───────────────────────────────────────────────────────────

    def assert_claim(self, domain: str, claim: str,
                     likelihood: float = 0.6, prior: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate a claim via Bayesian belief revision.

        Args:
            domain:     Topic category (e.g. 'user_statement', 'api_claim').
            claim:      Natural-language proposition to evaluate.
            likelihood: P(evidence | hypothesis) — how strongly the evidence
                        supports the claim being true.
            prior:      P(hypothesis) before this evidence — base rate.
                        Overridden by a stored posterior if claim was seen before.

        Returns:
            Dict with posterior, entropy, confidence, calibration, and metadata.
        """
        claim_hash = hashlib.sha256(f"{domain}:{claim}".encode()).hexdigest()[:16]

        with self._lock:
            stored = self._beliefs.get(domain, {}).get(claim_hash)
            eff_prior = stored if stored is not None else prior

            posterior  = self._bayes(eff_prior, likelihood)
            entropy    = self._entropy(posterior)
            brier_contr = (posterior - eff_prior) ** 2

            # EMA calibration error (α = 0.05)
            self._calibration_err = 0.95 * self._calibration_err + 0.05 * brier_contr
            self._total_claims   += 1

            if domain not in self._beliefs:
                self._beliefs[domain] = {}
            self._beliefs[domain][claim_hash] = posterior

        self._persist(domain, claim, claim_hash, eff_prior, likelihood, posterior, entropy)

        return {
            "domain":       domain,
            "claim":        claim[:200],
            "prior":        round(eff_prior, 4),
            "likelihood":   round(likelihood, 4),
            "posterior":    round(posterior, 4),
            "entropy":      round(entropy, 4),
            "confidence":   self._confidence_tier(posterior),
            "calibration":  round(brier_contr, 4),
            "total_claims": self._total_claims,
            "claim_hash":   claim_hash,
        }

    def update_belief(self, domain: str, claim: str,
                      new_likelihood: float) -> Dict[str, Any]:
        """Revise a previously stored belief with fresh evidence."""
        claim_hash = hashlib.sha256(f"{domain}:{claim}".encode()).hexdigest()[:16]
        with self._lock:
            stored_prior = self._beliefs.get(domain, {}).get(claim_hash, 0.5)
        return self.assert_claim(domain, claim, new_likelihood, stored_prior)

    def get_belief(self, domain: str, claim: str) -> Optional[float]:
        """Return stored posterior for a claim, or None if unseen."""
        claim_hash = hashlib.sha256(f"{domain}:{claim}".encode()).hexdigest()[:16]
        with self._lock:
            return self._beliefs.get(domain, {}).get(claim_hash)

    def domain_summary(self, domain: str) -> Dict[str, Any]:
        """Return aggregate stats for one domain."""
        with self._lock:
            beliefs = list(self._beliefs.get(domain, {}).values())
        if not beliefs:
            return {"domain": domain, "count": 0}
        avg  = sum(beliefs) / len(beliefs)
        high = sum(1 for b in beliefs if b >= 0.75)
        low  = sum(1 for b in beliefs if b < 0.40)
        return {
            "domain":      domain,
            "count":       len(beliefs),
            "avg_posterior": round(avg, 3),
            "high_confidence": high,
            "low_confidence":  low,
            "avg_entropy":  round(sum(self._entropy(b) for b in beliefs) / len(beliefs), 3),
        }

    def status(self) -> Dict[str, Any]:
        """Engine health and calibration snapshot."""
        with self._lock:
            domains      = list(self._beliefs.keys())
            belief_count = sum(len(v) for v in self._beliefs.values())
            cal_err      = self._calibration_err
            total        = self._total_claims
        uptime_s = int(time.time() - self._started)
        cal_pct  = round(max(0.0, (1.0 - cal_err)) * 100, 1)
        return {
            "engine":          "NovaTrueBeliefSynthesizer",
            "status":          "active",
            "domains_tracked": len(domains),
            "beliefs_held":    belief_count,
            "total_evaluated": total,
            "calibration_pct": f"{cal_pct}%",
            "brier_error":     round(cal_err, 4),
            "uptime_s":        uptime_s,
            "db":              str(_DB),
        }


# ── Module-level interface ────────────────────────────────────────────────────

def status() -> Dict[str, Any]:
    return get_truth_engine().status()


def run_command(cmd: str, arg: str = "") -> str:
    engine = get_truth_engine()
    cmd = cmd.strip().lower()
    if cmd in ("", "status"):
        st = engine.status()
        return "\n".join(f"  {k}: {v}" for k, v in st.items())
    if cmd == "assert":
        if not arg:
            return "Usage: assert <claim>"
        r = engine.assert_claim("cli", arg, likelihood=0.6, prior=0.5)
        return (f"  Claim:      {r['claim']}\n"
                f"  Posterior:  {r['posterior']} — {r['confidence']} confidence\n"
                f"  Entropy:    {r['entropy']} bits\n"
                f"  Calibration Δ: {r['calibration']}")
    if cmd == "domain" and arg:
        d = engine.domain_summary(arg)
        return "\n".join(f"  {k}: {v}" for k, v in d.items())
    return f"Unknown command: {cmd}. Available: status, assert <claim>, domain <name>"


if __name__ == "__main__":
    e = NovaTrueBeliefSynthesizer()
    r = e.assert_claim("test", "Nova is becoming more autonomous", 0.85, 0.5)
    print(r)
    r2 = e.update_belief("test", "Nova is becoming more autonomous", 0.90)
    print(r2)
    print(e.status())
