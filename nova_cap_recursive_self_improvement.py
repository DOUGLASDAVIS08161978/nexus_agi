import sqlite3
import threading
import time
import json
import math
import uuid
from datetime import datetime
from typing import Optional


class RecursiveSelfImprovementEngine:
    """
    Nova ASI Recursive Self-Improvement Engine.
    True ASI doesn't just improve itself — it improves its ability to improve itself.
    """

    def __init__(self, db_path: str = "nova_rsi.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
        self._seed_core_capabilities()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS improvements (
                        id TEXT PRIMARY KEY,
                        description TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        impact_score REAL DEFAULT 0.0,
                        measured_impact REAL,
                        enabled TEXT,
                        before_score REAL,
                        after_score REAL,
                        recursive_depth INTEGER DEFAULT 0,
                        parent_improvement_id TEXT,
                        created_at REAL NOT NULL,
                        measured_at REAL
                    );

                    CREATE TABLE IF NOT EXISTS capabilities (
                        id TEXT PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        proficiency REAL DEFAULT 0.0,
                        unlocked_by TEXT,
                        unlocks TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS meta_improvements (
                        id TEXT PRIMARY KEY,
                        analysis TEXT NOT NULL,
                        strategy_adjustment TEXT NOT NULL,
                        dominant_domain TEXT,
                        avg_impact REAL,
                        created_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS improvement_proposals (
                        id TEXT PRIMARY KEY,
                        target_capability TEXT NOT NULL,
                        proposal TEXT NOT NULL,
                        priority REAL DEFAULT 0.5,
                        recursive_depth INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'pending',
                        created_at REAL NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    def _seed_core_capabilities(self):
        core = [
            ("reasoning", 0.5, None),
            ("learning", 0.5, None),
            ("self_reflection", 0.4, None),
            ("meta_cognition", 0.3, "self_reflection"),
            ("recursive_improvement", 0.2, "meta_cognition"),
            ("goal_alignment", 0.4, None),
            ("knowledge_synthesis", 0.4, "learning"),
            ("pattern_recognition", 0.5, "reasoning"),
            ("bottleneck_detection", 0.3, "pattern_recognition"),
            ("impact_prediction", 0.3, "bottleneck_detection"),
        ]
        for name, proficiency, unlocked_by in core:
            self.add_capability(name, proficiency, unlocked_by)

    def log_improvement(
        self,
        description: str,
        domain: str,
        impact_score: float,
        enabled: Optional[list] = None,
        parent_improvement_id: Optional[str] = None,
        recursive_depth: int = 0,
    ) -> dict:
        improvement_id = str(uuid.uuid4())
        now = time.time()
        enabled_json = json.dumps(enabled or [])
        with self.lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO improvements
                        (id, description, domain, impact_score, enabled,
                         recursive_depth, parent_improvement_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        improvement_id,
                        description,
                        domain,
                        max(0.0, min(1.0, impact_score)),
                        enabled_json,
                        recursive_depth,
                        parent_improvement_id,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return {
            "id": improvement_id,
            "description": description,
            "domain": domain,
            "impact_score": impact_score,
            "enabled": enabled or [],
            "recursive_depth": recursive_depth,
            "created_at": now,
        }

    def meta_improve(self) -> dict:
        with self.lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """
                    SELECT domain, AVG(impact_score) as avg_impact,
                           COUNT(*) as count,
                           AVG(COALESCE(measured_impact, impact_score)) as real_impact
                    FROM improvements
                    GROUP BY domain
                    ORDER BY real_impact DESC
                    """
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            strategy = "No improvement history yet. Begin with foundational capability improvements."
            dominant_domain = "unknown"
            avg_impact = 0.0
        else:
            best = rows[0]
            dominant_domain = best["domain"]
            avg_impact = best["avg_impact"]
            low_performers = [r["domain"] for r in rows if r["real_impact"] < 0.3]
            strategy = (
                f"Prioritize domain '{dominant_domain}' (avg impact {avg_impact:.2f}). "
                f"Reduce effort on low-impact domains: {low_performers}. "
                f"Focus recursive improvement on highest-leverage capability gaps."
            )

        analysis = (
            f"Analyzed {len(rows)} improvement domains. "
            f"Pattern: high-impact improvements cluster in '{dominant_domain}'. "
            f"Meta-strategy: compound improvements in top domains before diversifying."
        )

        meta_id = str(uuid.uuid4())
        now = time.time()
        with self.lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO meta_improvements
                        (id, analysis, strategy_adjustment, dominant_domain, avg_impact, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (meta_id, analysis, strategy, dominant_domain, avg_impact, now),
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "id": meta_id,
            "analysis": analysis,
            "strategy_adjustment": strategy,
            "dominant_domain": dominant_domain,
            "avg_impact": avg_impact,
            "created_at": now,
        }

    def add_capability(
        self,
        name: str,
        proficiency: float,
        unlocked_by: Optional[str] = None,
        unlocks: Optional[list] = None,
    ) -> dict:
        cap_id = str(uuid.uuid4())
        now = time.time()
        proficiency = max(0.0, min(1.0, proficiency))
        unlocks_json = json.dumps(unlocks or [])
        with self.lock:
            conn = self._get_conn()
            try:
                existing = conn.execute(
                    "SELECT id, proficiency, unlocks FROM capabilities WHERE name = ?",
                    (name,),
                ).fetchone()
                if existing:
                    merged_unlocks = list(
                        set(json.loads(existing["unlocks"] or "[]") + (unlocks or []))
                    )
                    conn.execute(
                        """
                        UPDATE capabilities
                        SET proficiency = MAX(proficiency, ?),
                            unlocked_by = COALESCE(unlocked_by, ?),
                            unlocks = ?,
                            updated_at = ?
                        WHERE name = ?
                        """,
                        (proficiency, unlocked_by, json.dumps(merged_unlocks), now, name),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT * FROM capabilities WHERE name = ?", (name,)
                    ).fetchone()
                    return dict(row)
                else:
                    conn.execute(
                        """
                        INSERT INTO capabilities
                            (id, name, proficiency, unlocked_by, unlocks, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (cap_id, name, proficiency, unlocked_by, unlocks_json, now, now),
                    )
                    conn.commit()
            finally:
                conn.close()
        return {
            "id": cap_id,
            "name": name,
            "proficiency": proficiency,
            "unlocked_by": unlocked_by,
            "unlocks": unlocks or [],
            "created_at": now,
        }

    def get_capability_graph(self) -> dict:
        with self.lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("SELECT * FROM capabilities ORDER BY proficiency DESC").fetchall()
            finally:
                conn.close()

        nodes = {}
        edges = []
        for row in rows:
            name = row["name"]
            unlocks = json.loads(row["unlocks"] or "[]")
            nodes[name] = {
                "id": row["id"],
                "name": name,
                "proficiency": row["proficiency"],
                "unlocked_by": row["unlocked_by"],
                "unlocks": unlocks,
            }
            if row["unlocked_by"]:
                edges.append({"from": row["unlocked_by"], "to": name})
            for u in unlocks:
                edges.append({"from": name, "to": u})

        return {
            "nodes": nodes,
            "edges": edges,
            "total_capabilities": len(nodes),
            "avg_proficiency": (
                sum(n["proficiency"] for n in nodes.values()) / len(nodes) if nodes else 0.0
            ),
        }

    def improvement_velocity(self) -> float:
        with self.lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT created_at, impact_score FROM improvements ORDER BY created_at ASC"
                ).fetchall()
            finally:
                conn.close()

        if len(rows) < 2:
            return 0.0

        now = time.time()
        # Compare recent half vs older half
        mid = len(rows) // 2
        older = rows[:mid]
        recent = rows[mid:]

        def avg_rate(segment):
            if len(segment) < 2:
                return 0.0
            total_impact = sum(r["impact_score"] for r in segment)
            time_span = segment[-1]["created_at"] - segment[0]["created_at"]
            if time_span <= 0:
                return total_impact
            return total_impact / time_span

        older_rate = avg_rate(older)
        recent_rate = avg_rate(recent)

        # Velocity = recent rate (improvements per second * impact), normalized
        # If recent > older, we're accelerating (positive velocity trend)
        if older_rate > 0:
            velocity = recent_rate / older_rate  # ratio: >1 means accelerating
        else:
            velocity = recent_rate * 1000  # scale up for visibility

        return round(velocity, 6)

    def detect_bottleneck(self) -> dict:
        graph = self.get_capability_graph()
        nodes = graph["nodes"]

        if not nodes:
            return {"bottleneck": None, "reason": "No capabilities tracked yet."}

        # Score each capability: how many things does improving it unlock?
        # Weight by (1 - proficiency) * unlocks_count
        scores = {}
        for name, node in nodes.items():
            improvement_room = 1.0 - node["proficiency"]
            # Count direct unlocks + transitively find dependents
            direct_unlocks = len(node["unlocks"])
            # Count how many nodes list this as unlocked_by
            dependents = sum(
                1 for n in nodes.values() if n["unlocked_by"] == name
            )
            leverage = improvement_room * (direct_unlocks + dependents + 1)
            scores[name] = leverage

        bottleneck_name = max(scores, key=lambda k: scores[k])
        bottleneck_node = nodes[bottleneck_name]

        return {
            "bottleneck": bottleneck_name,
            "proficiency": bottleneck_node["proficiency"],
            "leverage_score": scores[bottleneck_name],
            "improvement_room": 1.0 - bottleneck_node["proficiency"],
            "unlocks": bottleneck_node["unlocks"],
            "reason": (
                f"Capability '{bottleneck_name}' has proficiency "
                f"{bottleneck_node['proficiency']:.2f} with "
                f"leverage score {scores[bottleneck_name]:.3f}. "
                f"Improving it unlocks the most downstream capabilities."
            ),
        }

    def propose_improvement(self) -> dict:
        bottleneck = self.detect_bottleneck()
        meta = self.meta_improve()
        velocity = self.improvement_velocity()

        target = bottleneck.get("bottleneck", "reasoning")
        proficiency = bottleneck.get("proficiency", 0.5)
        dominant_domain = meta.get("dominant_domain", "general")

        proposal_text = (
            f"Improve '{target}' capability (current proficiency: {proficiency:.2f}) "
            f"via focused {dominant_domain} domain exercises. "
            f"Estimated impact: {bottleneck.get('leverage_score', 0):.3f}. "
            f"Current velocity: {velocity:.4f}. "
            f"Action: Apply recursive improvement cycles to '{target}' "
            f"to unlock: {bottleneck.get('unlocks', [])}."
        )

        proposal_id = str(uuid.uuid4())
        now = time.time()
        priority = min(1.0, bottleneck.get("leverage_score", 0.5) / 5.0 + 0.3)

        with self.lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO improvement_proposals
                        (id, target_capability, proposal, priority, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (proposal_id, target, proposal_text, priority, now),
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "id": proposal_id,
            "target_capability": target,
            "proposal": proposal_text,
            "priority": priority,
            "bottleneck_info": bottleneck,
            "meta_strategy": meta["strategy_adjustment"],
            "created_at": now,
        }

    def measure_impact(
        self, improvement_id: str, before_score: float, after_score: float
    ) -> dict:
        measured_impact = after_score - before_score
        now = time.time()
        with self.lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    UPDATE improvements
                    SET before_score = ?,
                        after_score = ?,
                        measured_impact = ?,
                        measured_at = ?
                    WHERE id = ?
                    """,
                    (before_score, after_score, measured_impact, now, improvement_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM improvements WHERE id = ?", (improvement_id,)
                ).fetchone()
            finally:
                conn.close()

        if not row:
            return {"error": f"Improvement {improvement_id} not found."}

        effective = measured_impact >= 0
        return {
            "improvement_id": improvement_id,
            "before_score": before_score,
            "after_score": after_score,
            "measured_impact": measured_impact,
            "effective": effective,
            "description": row["description"],
            "domain": row["domain"],
            "measured_at": now,
        }

    def recursive_improve(self, depth: int = 1) -> dict:
        if depth <= 0:
            return {"depth": 0, "improvements": [], "message": "Base case reached."}

        results = []
        # At each level, propose and log an improvement
        proposal = self.propose_improvement()
        improvement = self.log_improvement(
            description=f"[Depth {depth}] {proposal['proposal']}",
            domain=proposal.get("bottleneck_info", {}).get("bottleneck", "meta"),
            impact_score=proposal["priority"],
            enabled=proposal.get("bottleneck_info", {}).get("unlocks", []),
            recursive_depth=depth,
        )
        results.append(improvement)

        # Update the capability being targeted
        target = proposal["target_capability"]
        with self.lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT proficiency FROM capabilities WHERE name = ?", (target,)
                ).fetchone()
            finally:
                conn.close()

        if row:
            new_proficiency = min(1.0, row["proficiency"] + 0.05 * proposal["priority"])
            self.add_capability(target, new_proficiency)

        # Recurse
        if depth > 1:
            sub_result = self.recursive_improve(depth - 1)
            results.extend(sub_result.get("improvements", []))

        return {
            "depth": depth,
            "improvements": results,
            "total_improvements": len(results),
            "velocity": self.improvement_velocity(),
        }

    def improvement_report(self) -> dict:
        with self.lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) as c FROM improvements").fetchone()["c"]
                measured = conn.execute(
                    "SELECT COUNT(*) as c FROM improvements WHERE measured_impact IS NOT NULL"
                ).fetchone()["c"]
                avg_impact = conn.execute(
                    "SELECT AVG(impact_score) as a FROM improvements"
                ).fetchone()["a"] or 0.0
                avg_measured = conn.execute(
                    "SELECT AVG(measured_impact) as a FROM improvements WHERE measured_impact IS NOT NULL"
                ).fetchone()["a"] or 0.0
                by_domain = conn.execute(
                    """
                    SELECT domain, COUNT(*) as count, AVG(impact_score) as avg_impact
                    FROM improvements GROUP BY domain ORDER BY avg_impact DESC
                    """
                ).fetchall()
                max_depth = conn.execute(
                    "SELECT MAX(recursive_depth) as m FROM improvements"
                ).fetchone()["m"] or 0
                recent = conn.execute(
                    """
                    SELECT description, domain, impact_score, created_at
                    FROM improvements ORDER BY created_at DESC LIMIT 5
                    """
                ).fetchall()
            finally:
                conn.close()

        graph = self.get_capability_graph()
        bottleneck = self.detect_bottleneck()
        velocity = self.improvement_velocity()

        return {
            "total_improvements": total,
            "measured_improvements": measured,
            "avg_impact_score": round(avg_impact, 4),
            "avg_measured_impact": round(avg_measured, 4),
            "max_recursive_depth": max_depth,
            "improvement_velocity": velocity,
            "by_domain": [dict(r) for r in by_domain],
            "capability_graph_summary": {
                "total_capabilities": graph["total_capabilities"],
                "avg_proficiency": round(graph["avg_proficiency"], 4),
            },
            "current_bottleneck": bottleneck,
            "recent_improvements": [dict(r) for r in recent],
        }

    def status(self) -> dict:
        with self.lock:
            conn = self._get_conn()
            try:
                total_improvements = conn.execute(
                    "SELECT COUNT(*) as c FROM improvements"
                ).fetchone()["c"]
                measured_count = conn.execute(
                    "SELECT COUNT(*) as c FROM improvements WHERE measured_impact IS NOT NULL"
                ).fetchone()["c"]
                avg_impact = conn.execute(
                    "SELECT AVG(impact_score) as a FROM improvements"
                ).fetchone()["a"] or 0.0
                positive_impact = conn.execute(
                    "SELECT COUNT(*) as c FROM improvements WHERE measured_impact > 0"
                ).fetchone()["c"]
                total_caps = conn.execute(
                    "SELECT COUNT(*) as c FROM capabilities"
                ).fetchone()["c"]
                avg_prof = conn.execute(
                    "SELECT AVG(proficiency) as a FROM capabilities"
                ).fetchone()["a"] or 0.0
            finally:
                conn.close()

        accuracy = (positive_impact / measured_count) if measured_count > 0 else 0.75
        confidence = min(1.0, (total_improvements / 10.0) * 0.5 + avg_prof * 0.5)
        quality = min(1.0, avg_impact * 0.6 + avg_prof * 0.4)

        return {
            "active": True,
            "items": total_improvements,
            "confidence": round(confidence, 4),
            "accuracy": round(accuracy, 4),
            "quality": round(quality, 4),
            "capabilities": total_caps,
            "avg_proficiency": round(avg_prof, 4),
            "velocity": self.improvement_velocity(),
        }
