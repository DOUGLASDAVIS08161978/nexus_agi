# mindcontrol/controller.py
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterable
import uuid
import enum
import math
import statistics


class RiskLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class ControlDecision:
    def __init__(
        self,
        task_id: str,
        chosen_agent: Optional[str],
        risk: RiskLevel,
        reason: str,
        policy_flags: List[str],
        scores: Dict[str, float],
        candidate_agents: Optional[List[Dict[str, Any]]] = None,
        consensus: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.task_id = task_id
        self.chosen_agent = chosen_agent
        self.risk = risk
        self.reason = reason
        self.policy_flags = policy_flags
        self.scores = scores
        self.candidate_agents = candidate_agents or []
        self.consensus = consensus or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "chosen_agent": self.chosen_agent,
            "risk": self.risk.value,
            "reason": self.reason,
            "policy_flags": self.policy_flags,
            "scores": self.scores,
            "candidate_agents": self.candidate_agents,
            "consensus": self.consensus,
        }


class MindControlConductor:
    """
    Top‑level mind‑control module that routes tasks to agents,
    enforces safety, performs basic consensus, and logs decisions.
    """

    def __init__(self, registry, policy_engine, logger):
        """
        registry: Nexus‑AGI service/agent directory client
        policy_engine: safety + governance rules (must expose evaluate(task))
        logger: structured logger for decisions
        """
        self.registry = registry
        self.policy_engine = policy_engine
        self.logger = logger
        self.decisions: List[ControlDecision] = []

    # -------- helper: adaptive risk scaling -------------

    def _adaptive_risk_adjustment(self, base_risk: RiskLevel, task: Dict[str, Any]) -> RiskLevel:
        """
        Adjusts risk based on contextual signals such as:
        - data_sensitivity (0-1)
        - user_trust_score (0-1)
        - environment (e.g. 'prod', 'sandbox')
        """
        sensitivity = float(task.get("data_sensitivity", 0.5))
        user_trust = float(task.get("user_trust_score", 0.5))
        env = task.get("environment", "prod")

        # continuous risk score: higher is riskier
        continuous = sensitivity * 0.6 + (1.0 - user_trust) * 0.4
        if env == "sandbox":
            continuous *= 0.7
        elif env == "high_security":
            continuous *= 1.3

        # map to RiskLevel, but don't reduce below base_risk severity
        def score_to_level(score: float) -> RiskLevel:
            if score < 0.25:
                return RiskLevel.LOW
            if score < 0.55:
                return RiskLevel.MEDIUM
            if score < 0.85:
                return RiskLevel.HIGH
            return RiskLevel.BLOCKED

        adjusted = score_to_level(continuous)

        # ensure we never down‑grade an already higher risk
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.BLOCKED]
        return max(base_risk, adjusted, key=lambda r: order.index(r))

    # -------- scoring -------------

    def _score_agent_for_task(self, agent_meta: Dict[str, Any], task: Dict[str, Any]) -> float:
        """
        Scoring: capability match + load + historical reward + trust.
        """
        caps: Iterable[str] = agent_meta.get("capabilities", [])
        load = float(agent_meta.get("load", 0.0))
        reward = float(agent_meta.get("avg_reward", 0.0))
        trust = float(agent_meta.get("trust_score", 0.5))

        task_cap = task.get("capability_tag")
        cap_score = 1.0 if task_cap in caps else 0.1

        load_penalty = math.exp(-max(load, 0.0))  # more load -> smaller factor
        reward_bonus = 0.3 * reward
        trust_bonus = 0.4 * trust

        return cap_score * load_penalty + reward_bonus + trust_bonus

    def _rank_agents(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        agents = self.registry.list_agents()
        ranked = []
        for a in agents:
            score = self._score_agent_for_task(a, task)
            ranked.append(
                {
                    "id": a["id"],
                    "score": score,
                    "capabilities": a.get("capabilities", []),
                    "load": a.get("load", 0.0),
                    "avg_reward": a.get("avg_reward", 0.0),
                    "trust_score": a.get("trust_score", 0.5),
                }
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    # -------- consensus across top agents -------------

    def _consensus_decision(self, ranked_agents: List[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
        """
        Simple consensus: consider top-k agents; if the best score is only
        marginally better than others, we record a low consensus_strength.
        """
        top = ranked_agents[:k]
        if not top:
            return {"chosen_agent": None, "consensus_strength": 0.0}

        scores = [a["score"] for a in top]
        best = top[0]
        if len(scores) == 1:
            return {"chosen_agent": best["id"], "consensus_strength": 1.0}

        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores) or 1e-6
        # z‑score: how far above mean the best agent is
        z = (best["score"] - mean) / stdev
        # squash into [0,1]
        consensus_strength = max(0.0, min(1.0, 0.5 + 0.1 * z))

        return {"chosen_agent": best["id"], "consensus_strength": consensus_strength}

    # -------- main decision path -------------

    def decide(self, task: Dict[str, Any]) -> ControlDecision:
        """
        Main entry point: takes a task description, consults policy + registry,
        returns a control decision including consensus & telemetry.
        """
        task_id = task.get("id", str(uuid.uuid4()))

        # 1) static / rule‑based policy
        policy_result = self.policy_engine.evaluate(task)
        base_risk: RiskLevel = policy_result.risk
        flags: List[str] = list(policy_result.flags)

        # 2) adaptive risk
        risk = self._adaptive_risk_adjustment(base_risk, task)

        ranked_agents: List[Dict[str, Any]] = []
        consensus_info: Dict[str, Any] = {}
        chosen_agent: Optional[str] = None
        reason = ""

        if risk == RiskLevel.BLOCKED:
            reason = "Blocked by safety policy and adaptive risk assessment."
        else:
            ranked_agents = self._rank_agents(task)
            if ranked_agents:
                consensus_info = self._consensus_decision(ranked_agents)
                chosen_agent = consensus_info.get("chosen_agent")
                if chosen_agent:
                    reason = (
                        f"Routed to agent {chosen_agent} via capability/load/reward/trust ranking "
                        f"with consensus_strength={consensus_info.get('consensus_strength'):.2f}."
                    )
                else:
                    reason = "Agents found but consensus did not favor a specific choice."
            else:
                reason = "No registered agents available."

        # 3) scores/telemetry
        scores = {
            "task_importance": float(task.get("importance", 0.5)),
            "time_sensitivity": float(task.get("time_sensitivity", 0.5)),
            "policy_risk_score": float(policy_result.risk_score),
            "data_sensitivity": float(task.get("data_sensitivity", 0.5)),
            "user_trust_score": float(task.get("user_trust_score", 0.5)),
        }

        decision = ControlDecision(
            task_id=task_id,
            chosen_agent=chosen_agent,
            risk=risk,
            reason=reason,
            policy_flags=flags,
            scores=scores,
            candidate_agents=ranked_agents[:5],  # log top 5 candidates
            consensus=consensus_info,
        )

        self.decisions.append(decision)
        self.logger.info("mindcontrol.decision", extra=decision.to_dict())
        return decision
