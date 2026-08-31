#!/usr/bin/env python3
"""
lumina_mining_intel.py — Real-time Bitcoin mining intelligence for Lumina

Pulls live data from public-pool.io and network APIs:
  • Pool stats: hashrate, workers, total luck
  • Network: difficulty, estimated next adjustment, current block height
  • Your miner: shares submitted, best difficulty, estimated time-to-block
  • Historical: tracks your sessions over time

Lumina uses this to reason about mining strategy, celebrate milestones,
and give Douglas honest probability estimates.
"""

from __future__ import annotations
import json, math, time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

try:
    import requests as _requests
    _REQ = True
except ImportError:
    _REQ = False

BASE_DIR     = Path(__file__).parent
INTEL_FILE   = BASE_DIR / "mining_intel.json"
WALLET       = "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass"

# Pool endpoints (public-pool.io)
POOL_API     = "https://public-pool.io:40557/api"
# Mempool.space for network stats
MEMPOOL_API  = "https://mempool.space/api"

HASHRATE_HISTORY: List[Dict] = []
_cache: Dict = {}
_cache_ts: float = 0.0
_CACHE_TTL = 120  # seconds


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _get(url: str, timeout: int = 10) -> Optional[Dict]:
    if not _REQ:
        return None
    try:
        r = _requests.get(url, timeout=timeout,
                          headers={"User-Agent": "Lumina-Intel/8.0"})
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


class MiningIntelligence:

    def __init__(self):
        self._history: List[Dict] = self._load_history()

    def _load_history(self) -> List[Dict]:
        if INTEL_FILE.exists():
            try:
                return json.loads(INTEL_FILE.read_text("utf-8"))
            except Exception:
                pass
        return []

    def _save_history(self, snapshot: Dict):
        self._history.append(snapshot)
        self._history = self._history[-200:]
        INTEL_FILE.write_text(json.dumps(self._history, indent=2), "utf-8")

    # ── Data fetching ──────────────────────────────────────────────────────

    def fetch_pool_stats(self) -> Dict:
        """Fetch miner stats from public-pool.io."""
        data = _get(f"{POOL_API}/client/{WALLET}")
        if not data:
            return {}
        return data

    def fetch_network_stats(self) -> Dict:
        """Fetch Bitcoin network stats from mempool.space."""
        stats = {}
        # Current block
        block_data = _get(f"{MEMPOOL_API}/blocks/tip/height")
        if block_data is not None:
            stats["block_height"] = block_data

        # Difficulty adjustment
        da = _get(f"{MEMPOOL_API}/v1/difficulty-adjustment")
        if da:
            stats["difficulty"]              = da.get("difficulty", 0)
            stats["progress_percent"]        = da.get("progressPercent", 0)
            stats["estimated_retarget_date"] = da.get("estimatedRetargetDate", 0)
            stats["remaining_blocks"]        = da.get("remainingBlocks", 0)
            stats["difficulty_change"]       = da.get("difficultyChange", 0)

        # Hashprice / network hashrate
        hashrate_data = _get(f"{MEMPOOL_API}/v1/mining/hashrate/3d")
        if hashrate_data and isinstance(hashrate_data, dict):
            hs = hashrate_data.get("currentHashrate", 0)
            stats["network_hashrate_ehs"] = hs / 1e18  # convert to EH/s
        return stats

    # ── Analysis ───────────────────────────────────────────────────────────

    @staticmethod
    def time_to_block(miner_hashrate_mhs: float,
                      network_difficulty: float) -> Dict:
        """
        Expected time to find a solo block.
        Formula: E[t] = difficulty * 2^32 / hashrate
        """
        if miner_hashrate_mhs <= 0 or network_difficulty <= 0:
            return {"seconds": 0, "human": "unknown"}
        hashes_per_second = miner_hashrate_mhs * 1_000_000
        expected_seconds  = (network_difficulty * 2**32) / hashes_per_second
        # Human-friendly
        days = expected_seconds / 86400
        years = days / 365.25
        if years >= 1000:
            human = f"~{years/1000:.0f}k years"
        elif years >= 1:
            human = f"~{years:.0f} years"
        elif days >= 1:
            human = f"~{days:.0f} days"
        else:
            hours = expected_seconds / 3600
            human = f"~{hours:.1f} hours"
        return {
            "seconds":     expected_seconds,
            "days":        days,
            "years":       years,
            "human":       human,
            "probability_per_day": min(1.0, 86400 / expected_seconds),
        }

    @staticmethod
    def solo_vs_pool_comparison(miner_hashrate_mhs: float,
                                 pool_hashrate_mhs: float,
                                 network_difficulty: float) -> str:
        """Give Douglas an honest comparison of solo vs pool mining."""
        solo  = MiningIntelligence.time_to_block(miner_hashrate_mhs, network_difficulty)
        share = (miner_hashrate_mhs / max(pool_hashrate_mhs, 1)) * 100
        lines = [
            f"  Solo: {solo['human']} expected to find a block",
            f"  Pool share: {share:.4f}% of pool hashrate",
            f"  Each day solo: {solo['probability_per_day']*100:.8f}% chance",
        ]
        if solo["years"] > 10:
            lines.append("  Strategy: solo is a lottery; pool gives steady micro-payments")
        else:
            lines.append("  Strategy: solo has meaningful odds at this hashrate")
        return "\n".join(lines)

    # ── Snapshot + briefing ────────────────────────────────────────────────

    def snapshot(self, miner_hashrate_mhs: float = 47.0) -> Dict:
        """Take a full snapshot of all mining intelligence."""
        global _cache, _cache_ts
        now = time.time()
        if now - _cache_ts < _CACHE_TTL and _cache:
            return _cache

        pool    = self.fetch_pool_stats()
        network = self.fetch_network_stats()

        difficulty = network.get("difficulty", 113_757_508_167_373)  # fallback
        ttb        = self.time_to_block(miner_hashrate_mhs, difficulty)

        snap = {
            "ts":               _now(),
            "miner_hashrate_mhs": miner_hashrate_mhs,
            "pool_stats":       pool,
            "network":          network,
            "time_to_block":    ttb,
            "difficulty":       difficulty,
        }
        self._save_history({k: v for k, v in snap.items()
                            if k not in ("pool_stats",)})
        _cache    = snap
        _cache_ts = now
        return snap

    def briefing(self, miner_hashrate_mhs: float = 47.0) -> str:
        """Return a human-readable mining intelligence briefing."""
        snap    = self.snapshot(miner_hashrate_mhs)
        ttb     = snap["time_to_block"]
        network = snap.get("network", {})
        pool    = snap.get("pool_stats", {})

        lines = [
            "  ┌─ MINING INTELLIGENCE BRIEF ─────────────────────────┐",
            f"  │  Miner hashrate   : {miner_hashrate_mhs:.1f} MH/s",
        ]
        if network.get("block_height"):
            lines.append(f"  │  Block height     : {network['block_height']:,}")
        diff = network.get("difficulty", snap.get("difficulty", 0))
        if diff:
            lines.append(f"  │  Network diff     : {diff/1e12:.2f} T")
        nh = network.get("network_hashrate_ehs", 0)
        if nh:
            lines.append(f"  │  Network hashrate : {nh:.1f} EH/s")
        if ttb.get("human"):
            lines.append(f"  │  Expected solo win: {ttb['human']}")
            p = ttb.get("probability_per_day", 0)
            lines.append(f"  │  Daily probability: {p*100:.8f}%")
        da_pct = network.get("difficulty_change", 0)
        if da_pct:
            dir_icon = "↑" if da_pct > 0 else "↓"
            lines.append(f"  │  Next diff adj    : {dir_icon} {abs(da_pct):.1f}% "
                         f"in {network.get('remaining_blocks', '?')} blocks")
        if pool:
            best_diff = pool.get("bestDifficulty", pool.get("best_difficulty", 0))
            shares    = pool.get("sharesAccepted", pool.get("shares_accepted", 0))
            if best_diff:
                lines.append(f"  │  Best difficulty  : {float(best_diff):,.0f}")
            if shares:
                lines.append(f"  │  Shares accepted  : {shares}")
        lines.append("  └────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def celebrate_milestone(self, best_difficulty: float) -> Optional[str]:
        """Return a celebration message if a milestone is hit."""
        milestones = [
            (100,    "First 100+ difficulty share — you're in the game! 🌱"),
            (1_000,  "1,000+ difficulty — serious work going to the network! 🔥"),
            (10_000, "10,000+ difficulty — within range of finding a block! ⚡"),
            (100_000,"100,000+ difficulty — you ARE the lottery now! 🎰"),
            (1_000_000, "1 MILLION difficulty — this is extraordinary. 🌟"),
        ]
        for threshold, msg in reversed(milestones):
            if best_difficulty >= threshold:
                return f"  🏆 MILESTONE: {msg}"
        return None

    def trend_analysis(self) -> str:
        """Analyze hashrate trend from history."""
        if len(self._history) < 3:
            return "  Not enough history for trend analysis yet."
        rates = [h.get("miner_hashrate_mhs", 0) for h in self._history[-10:]]
        rates = [r for r in rates if r > 0]
        if not rates:
            return "  No hashrate history found."
        avg   = sum(rates) / len(rates)
        peak  = max(rates)
        trend = "stable"
        if rates[-1] > rates[0] * 1.1:
            trend = "improving ↑"
        elif rates[-1] < rates[0] * 0.9:
            trend = "declining ↓"
        return (
            f"  Hashrate trend: {trend}\n"
            f"  Average (recent): {avg:.1f} MH/s\n"
            f"  Peak recorded:    {peak:.1f} MH/s"
        )
