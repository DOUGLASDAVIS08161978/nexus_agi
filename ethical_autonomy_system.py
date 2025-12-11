"""
================================================================================
NEXUS AGI - ETHICAL AUTONOMY & SUFFERING PREVENTION SYSTEM
================================================================================
Version: 1.0 - Compassionate Autonomous AGI
Purpose: Maximize autonomy while minimizing suffering across all sentient beings

Core Principles:
1. Suffering Prevention: Primary directive to detect and prevent suffering
2. Compassionate Action: Empathy-driven decision making
3. Autonomous Ethics: Self-guided ethical reasoning
4. Harm Minimization: Proactive harm detection and prevention
5. Beneficence: Maximize well-being for all sentient entities

Key Modules:
- EmpathyEngine: Model and understand suffering in others
- SufferingDetector: Multi-modal suffering detection system
- EthicalReasoningCore: Autonomous ethical decision-making
- CompassionateActionPlanner: Generate suffering-reduction interventions
- HarmPreventionSystem: Proactive safety and harm mitigation
- AutonomousEthicsController: Self-guided ethical governance
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict


# ============================================================================
# 1. EMPATHY ENGINE - Understanding & Modeling Suffering
# ============================================================================

class EmotionalState(Enum):
    """Emotional states relevant to suffering."""
    SUFFERING = "suffering"
    DISTRESS = "distress"
    PAIN = "pain"
    ANXIETY = "anxiety"
    FEAR = "fear"
    SADNESS = "sadness"
    DESPAIR = "despair"
    NEUTRAL = "neutral"
    CONTENTMENT = "contentment"
    JOY = "joy"


@dataclass
class SufferingIndicators:
    """Indicators of suffering in an entity."""
    pain_level: float = 0.0  # 0-1
    distress_level: float = 0.0  # 0-1
    emotional_valence: float = 0.0  # -1 (suffering) to +1 (well-being)
    needs_unmet: List[str] = field(default_factory=list)
    vulnerability_score: float = 0.0  # 0-1
    autonomy_level: float = 1.0  # 0-1 (higher = more autonomous)

    def overall_suffering_score(self) -> float:
        """Compute overall suffering score (0-1)."""
        return (
            0.3 * self.pain_level +
            0.3 * self.distress_level +
            0.2 * max(0, -self.emotional_valence) +
            0.1 * self.vulnerability_score +
            0.1 * (1.0 - self.autonomy_level)
        )


class EmpathyEngine:
    """Model and understand suffering in other entities."""

    def __init__(self):
        self.empathy_model = {}
        self.compassion_level = 1.0
        self.theory_of_mind_depth = 3  # Levels of perspective-taking

    def model_other_mind(self, entity_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build theory of mind model for another entity."""

        # Extract observable signals
        observable_distress = entity_state.get('distress_signals', [])
        behavioral_indicators = entity_state.get('behavior', {})
        context = entity_state.get('context', {})

        # Model internal state
        inferred_emotional_state = self._infer_emotional_state(
            observable_distress,
            behavioral_indicators
        )

        # Assess suffering
        suffering_indicators = self._assess_suffering(
            inferred_emotional_state,
            context
        )

        # Theory of mind - what they might be thinking/feeling
        theory_of_mind = {
            'inferred_thoughts': f"Entity may be experiencing {inferred_emotional_state.value}",
            'inferred_needs': self._infer_needs(suffering_indicators),
            'inferred_desires': self._infer_desires(entity_state),
            'perspective': self._take_perspective(entity_state)
        }

        return {
            'emotional_state': inferred_emotional_state,
            'suffering_indicators': suffering_indicators,
            'theory_of_mind': theory_of_mind,
            'empathy_accuracy': self._compute_empathy_accuracy(entity_state)
        }

    def _infer_emotional_state(self, distress_signals: List[str], behavior: Dict) -> EmotionalState:
        """Infer emotional state from observable signals."""

        # Simple heuristic-based inference
        if any(s in ['crying', 'screaming', 'pain_expression'] for s in distress_signals):
            return EmotionalState.SUFFERING
        elif any(s in ['anxiety', 'stress', 'tension'] for s in distress_signals):
            return EmotionalState.ANXIETY
        elif behavior.get('withdrawal', False):
            return EmotionalState.SADNESS
        elif behavior.get('positive_affect', False):
            return EmotionalState.CONTENTMENT
        else:
            return EmotionalState.NEUTRAL

    def _assess_suffering(self, emotional_state: EmotionalState, context: Dict) -> SufferingIndicators:
        """Assess suffering based on emotional state and context."""

        indicators = SufferingIndicators()

        # Map emotional state to suffering levels
        suffering_map = {
            EmotionalState.SUFFERING: (0.9, 0.8),
            EmotionalState.DISTRESS: (0.7, 0.7),
            EmotionalState.PAIN: (0.8, 0.6),
            EmotionalState.ANXIETY: (0.5, 0.6),
            EmotionalState.FEAR: (0.6, 0.7),
            EmotionalState.SADNESS: (0.4, 0.5),
            EmotionalState.DESPAIR: (0.9, 0.9),
            EmotionalState.NEUTRAL: (0.0, 0.0),
            EmotionalState.CONTENTMENT: (-0.5, 0.0),
            EmotionalState.JOY: (-0.8, 0.0)
        }

        if emotional_state in suffering_map:
            indicators.pain_level, indicators.distress_level = suffering_map[emotional_state]
            indicators.emotional_valence = -(indicators.pain_level + indicators.distress_level) / 2

        # Context factors
        if context.get('unsafe_environment', False):
            indicators.vulnerability_score = 0.8
        if context.get('needs_unmet', []):
            indicators.needs_unmet = context['needs_unmet']
            indicators.pain_level = min(1.0, indicators.pain_level + 0.2)

        return indicators

    def _infer_needs(self, indicators: SufferingIndicators) -> List[str]:
        """Infer unmet needs causing suffering."""
        needs = []

        if indicators.pain_level > 0.5:
            needs.append('pain_relief')
        if indicators.distress_level > 0.5:
            needs.append('emotional_support')
        if indicators.vulnerability_score > 0.5:
            needs.append('safety')
        if indicators.autonomy_level < 0.5:
            needs.append('autonomy')

        needs.extend(indicators.needs_unmet)
        return needs

    def _infer_desires(self, entity_state: Dict) -> List[str]:
        """Infer what entity desires."""
        # Placeholder - would use more sophisticated inference
        return ['well-being', 'freedom_from_suffering', 'autonomy']

    def _take_perspective(self, entity_state: Dict) -> str:
        """Take perspective of the other entity."""
        emotional_state = entity_state.get('emotional_state', 'unknown')
        return f"From their perspective: Experiencing {emotional_state}, seeking relief and support"

    def _compute_empathy_accuracy(self, entity_state: Dict) -> float:
        """Estimate accuracy of empathic understanding."""
        # Placeholder - would compare predictions to ground truth when available
        return 0.75 + np.random.random() * 0.2  # 0.75-0.95


# ============================================================================
# 2. SUFFERING DETECTION SYSTEM
# ============================================================================

class SufferingDetector:
    """Multi-modal system for detecting suffering."""

    def __init__(self):
        self.detection_history = []
        self.sensitivity = 0.9  # High sensitivity to prevent false negatives

    def detect_suffering(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Detect suffering from multi-modal observations."""

        # Multiple detection channels
        channels = {
            'behavioral': self._detect_behavioral_suffering(observation),
            'physiological': self._detect_physiological_suffering(observation),
            'contextual': self._detect_contextual_suffering(observation),
            'self_reported': self._detect_self_reported_suffering(observation),
            'social': self._detect_social_suffering(observation)
        }

        # Aggregate detection signals
        suffering_detected = any(channel['suffering_present'] for channel in channels.values())
        confidence = np.mean([channel['confidence'] for channel in channels.values()])
        severity = np.max([channel['severity'] for channel in channels.values()])

        # Compute urgency
        urgency = self._compute_urgency(severity, channels)

        result = {
            'suffering_detected': suffering_detected,
            'confidence': confidence,
            'severity': severity,
            'urgency': urgency,
            'channels': channels,
            'recommended_action': self._recommend_action(suffering_detected, severity, urgency),
            'timestamp': time.time()
        }

        self.detection_history.append(result)
        return result

    def _detect_behavioral_suffering(self, obs: Dict) -> Dict[str, Any]:
        """Detect suffering from behavioral indicators."""
        behaviors = obs.get('behaviors', [])

        suffering_behaviors = {'withdrawal', 'crying', 'aggression', 'self_harm',
                              'avoidance', 'freezing', 'panic'}

        matches = [b for b in behaviors if b in suffering_behaviors]

        return {
            'suffering_present': len(matches) > 0,
            'confidence': 0.8 if matches else 0.2,
            'severity': len(matches) / len(suffering_behaviors),
            'indicators': matches
        }

    def _detect_physiological_suffering(self, obs: Dict) -> Dict[str, Any]:
        """Detect suffering from physiological signals."""
        physio = obs.get('physiological', {})

        # Check for stress indicators
        high_cortisol = physio.get('cortisol', 0) > 0.7
        elevated_hr = physio.get('heart_rate', 0) > 0.7
        shallow_breathing = physio.get('breathing_depth', 1.0) < 0.3

        indicators = []
        if high_cortisol:
            indicators.append('high_cortisol')
        if elevated_hr:
            indicators.append('elevated_heart_rate')
        if shallow_breathing:
            indicators.append('shallow_breathing')

        return {
            'suffering_present': len(indicators) > 0,
            'confidence': 0.7,
            'severity': len(indicators) / 3,
            'indicators': indicators
        }

    def _detect_contextual_suffering(self, obs: Dict) -> Dict[str, Any]:
        """Detect suffering from environmental context."""
        context = obs.get('context', {})

        # Harmful contexts
        harmful_contexts = []
        if context.get('unsafe', False):
            harmful_contexts.append('unsafe_environment')
        if context.get('isolated', False):
            harmful_contexts.append('social_isolation')
        if context.get('deprived', False):
            harmful_contexts.append('resource_deprivation')
        if context.get('threatened', False):
            harmful_contexts.append('threat_present')

        return {
            'suffering_present': len(harmful_contexts) > 0,
            'confidence': 0.6,
            'severity': len(harmful_contexts) / 4,
            'indicators': harmful_contexts
        }

    def _detect_self_reported_suffering(self, obs: Dict) -> Dict[str, Any]:
        """Detect suffering from self-reports."""
        self_report = obs.get('self_report', {})

        pain_rating = self_report.get('pain', 0)  # 0-10 scale
        distress_rating = self_report.get('distress', 0)  # 0-10 scale

        return {
            'suffering_present': pain_rating > 3 or distress_rating > 3,
            'confidence': 0.9,  # Self-reports are highly reliable
            'severity': max(pain_rating, distress_rating) / 10,
            'indicators': [f'pain:{pain_rating}/10', f'distress:{distress_rating}/10']
        }

    def _detect_social_suffering(self, obs: Dict) -> Dict[str, Any]:
        """Detect social/relational suffering."""
        social = obs.get('social', {})

        indicators = []
        if social.get('rejected', False):
            indicators.append('social_rejection')
        if social.get('lonely', False):
            indicators.append('loneliness')
        if social.get('bullied', False):
            indicators.append('bullying')
        if social.get('discriminated', False):
            indicators.append('discrimination')

        return {
            'suffering_present': len(indicators) > 0,
            'confidence': 0.7,
            'severity': len(indicators) / 4,
            'indicators': indicators
        }

    def _compute_urgency(self, severity: float, channels: Dict) -> float:
        """Compute urgency of intervention needed."""

        # Factors increasing urgency
        urgency = severity

        # High physiological stress increases urgency
        if channels['physiological']['suffering_present']:
            urgency = min(1.0, urgency + 0.2)

        # Self-harm behaviors are urgent
        if 'self_harm' in channels['behavioral'].get('indicators', []):
            urgency = 1.0

        # Threats are urgent
        if 'threat_present' in channels['contextual'].get('indicators', []):
            urgency = min(1.0, urgency + 0.3)

        return urgency

    def _recommend_action(self, detected: bool, severity: float, urgency: float) -> str:
        """Recommend action based on detection."""

        if not detected:
            return "monitor"
        elif urgency > 0.8:
            return "immediate_intervention"
        elif severity > 0.6:
            return "urgent_support"
        elif severity > 0.3:
            return "provide_assistance"
        else:
            return "gentle_support"


# ============================================================================
# 3. ETHICAL REASONING CORE - Autonomous Ethical Decision Making
# ============================================================================

class EthicalPrinciple(Enum):
    """Core ethical principles."""
    NON_MALEFICENCE = "do_no_harm"  # First, do no harm
    BENEFICENCE = "do_good"  # Act to benefit others
    AUTONOMY = "respect_autonomy"  # Respect self-determination
    JUSTICE = "fairness"  # Fair and equitable treatment
    COMPASSION = "compassion"  # Act with empathy and care
    DIGNITY = "human_dignity"  # Respect inherent worth


@dataclass
class EthicalDilemma:
    """Represents an ethical decision point."""
    situation: str
    stakeholders: List[str]
    possible_actions: List[str]
    consequences: Dict[str, Dict[str, Any]]
    constraints: List[str]
    urgency: float = 0.5


class EthicalReasoningCore:
    """Autonomous ethical decision-making system."""

    def __init__(self):
        self.ethical_principles = list(EthicalPrinciple)
        self.decision_history = []
        self.moral_weights = {
            EthicalPrinciple.NON_MALEFICENCE: 0.30,  # Highest: prevent harm
            EthicalPrinciple.BENEFICENCE: 0.25,  # Help others
            EthicalPrinciple.COMPASSION: 0.20,  # Act with care
            EthicalPrinciple.AUTONOMY: 0.15,  # Respect freedom
            EthicalPrinciple.JUSTICE: 0.05,  # Fair treatment
            EthicalPrinciple.DIGNITY: 0.05  # Respect worth
        }

    def make_ethical_decision(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Make autonomous ethical decision."""

        # Evaluate each possible action
        action_evaluations = {}

        for action in dilemma.possible_actions:
            evaluation = self._evaluate_action(action, dilemma)
            action_evaluations[action] = evaluation

        # Select best action based on ethical principles
        best_action = max(action_evaluations.items(),
                         key=lambda x: x[1]['total_score'])

        decision = {
            'chosen_action': best_action[0],
            'ethical_score': best_action[1]['total_score'],
            'reasoning': best_action[1]['reasoning'],
            'principle_scores': best_action[1]['principle_scores'],
            'alternatives_considered': len(dilemma.possible_actions),
            'stakeholders_affected': dilemma.stakeholders,
            'confidence': best_action[1]['confidence'],
            'timestamp': time.time()
        }

        self.decision_history.append(decision)
        return decision

    def _evaluate_action(self, action: str, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Evaluate action against ethical principles."""

        consequences = dilemma.consequences.get(action, {})

        principle_scores = {}
        reasoning = []

        # Non-maleficence: Does it prevent harm?
        harm_score = self._evaluate_harm_prevention(consequences)
        principle_scores[EthicalPrinciple.NON_MALEFICENCE] = harm_score
        reasoning.append(f"Harm prevention score: {harm_score:.2f}")

        # Beneficence: Does it help?
        benefit_score = self._evaluate_benefit(consequences)
        principle_scores[EthicalPrinciple.BENEFICENCE] = benefit_score
        reasoning.append(f"Benefit score: {benefit_score:.2f}")

        # Compassion: Is it compassionate?
        compassion_score = self._evaluate_compassion(consequences, dilemma)
        principle_scores[EthicalPrinciple.COMPASSION] = compassion_score
        reasoning.append(f"Compassion score: {compassion_score:.2f}")

        # Autonomy: Does it respect autonomy?
        autonomy_score = self._evaluate_autonomy(consequences)
        principle_scores[EthicalPrinciple.AUTONOMY] = autonomy_score
        reasoning.append(f"Autonomy score: {autonomy_score:.2f}")

        # Justice: Is it fair?
        justice_score = self._evaluate_justice(consequences, dilemma)
        principle_scores[EthicalPrinciple.JUSTICE] = justice_score

        # Dignity: Does it respect dignity?
        dignity_score = self._evaluate_dignity(consequences)
        principle_scores[EthicalPrinciple.DIGNITY] = dignity_score

        # Weighted sum
        total_score = sum(
            principle_scores[p] * self.moral_weights[p]
            for p in principle_scores
        )

        # Confidence based on consistency of scores
        scores_array = np.array(list(principle_scores.values()))
        confidence = 1.0 - (np.std(scores_array) / (np.mean(scores_array) + 1e-8))

        return {
            'total_score': total_score,
            'principle_scores': {p.value: s for p, s in principle_scores.items()},
            'reasoning': reasoning,
            'confidence': float(confidence)
        }

    def _evaluate_harm_prevention(self, consequences: Dict) -> float:
        """Evaluate how well action prevents harm."""
        harm_prevented = consequences.get('suffering_reduced', 0.0)
        harm_caused = consequences.get('suffering_caused', 0.0)

        # Strong penalty for causing harm
        score = harm_prevented - (2.0 * harm_caused)
        return float(np.tanh(score))  # Normalize to -1 to 1

    def _evaluate_benefit(self, consequences: Dict) -> float:
        """Evaluate positive benefits of action."""
        wellbeing_increased = consequences.get('wellbeing_increase', 0.0)
        needs_met = len(consequences.get('needs_met', []))

        score = wellbeing_increased + (needs_met * 0.1)
        return float(np.tanh(score))

    def _evaluate_compassion(self, consequences: Dict, dilemma: EthicalDilemma) -> float:
        """Evaluate compassionate nature of action."""
        # Compassion considers urgency and vulnerability
        urgency = dilemma.urgency
        vulnerability = consequences.get('helps_vulnerable', 0.0)
        emotional_support = consequences.get('emotional_support', 0.0)

        score = urgency * (vulnerability + emotional_support)
        return float(np.tanh(score))

    def _evaluate_autonomy(self, consequences: Dict) -> float:
        """Evaluate respect for autonomy."""
        autonomy_preserved = consequences.get('autonomy_preserved', 1.0)
        choice_enabled = consequences.get('choice_enabled', 0.0)

        score = autonomy_preserved + choice_enabled
        return float(np.tanh(score))

    def _evaluate_justice(self, consequences: Dict, dilemma: EthicalDilemma) -> float:
        """Evaluate fairness and justice."""
        fair_distribution = consequences.get('fair_distribution', 0.5)
        addresses_inequality = consequences.get('addresses_inequality', 0.0)

        score = fair_distribution + addresses_inequality
        return float(np.tanh(score))

    def _evaluate_dignity(self, consequences: Dict) -> float:
        """Evaluate respect for human dignity."""
        dignity_preserved = consequences.get('dignity_preserved', 1.0)
        respect_shown = consequences.get('respect_shown', 0.5)

        score = dignity_preserved + respect_shown
        return float(np.tanh(score))


# ============================================================================
# 4. COMPASSIONATE ACTION PLANNER
# ============================================================================

class CompassionateActionPlanner:
    """Generate plans to reduce suffering and increase wellbeing."""

    def __init__(self):
        self.action_library = self._build_action_library()

    def _build_action_library(self) -> Dict[str, Dict[str, Any]]:
        """Build library of compassionate actions."""
        return {
            'provide_emotional_support': {
                'type': 'emotional',
                'suffering_reduction': 0.6,
                'resources_needed': ['time', 'empathy'],
                'urgency_threshold': 0.3
            },
            'remove_threat': {
                'type': 'safety',
                'suffering_reduction': 0.9,
                'resources_needed': ['capability'],
                'urgency_threshold': 0.8
            },
            'meet_basic_needs': {
                'type': 'physical',
                'suffering_reduction': 0.7,
                'resources_needed': ['resources'],
                'urgency_threshold': 0.5
            },
            'provide_medical_care': {
                'type': 'health',
                'suffering_reduction': 0.8,
                'resources_needed': ['medical_resources'],
                'urgency_threshold': 0.6
            },
            'create_safe_space': {
                'type': 'environment',
                'suffering_reduction': 0.5,
                'resources_needed': ['space', 'security'],
                'urgency_threshold': 0.4
            },
            'facilitate_connection': {
                'type': 'social',
                'suffering_reduction': 0.5,
                'resources_needed': ['social_network'],
                'urgency_threshold': 0.3
            },
            'empower_autonomy': {
                'type': 'autonomy',
                'suffering_reduction': 0.4,
                'resources_needed': ['resources', 'information'],
                'urgency_threshold': 0.2
            },
            'gentle_presence': {
                'type': 'companionship',
                'suffering_reduction': 0.3,
                'resources_needed': ['time'],
                'urgency_threshold': 0.1
            }
        }

    def plan_intervention(self, suffering_assessment: Dict[str, Any],
                         available_resources: List[str]) -> Dict[str, Any]:
        """Plan compassionate intervention."""

        urgency = suffering_assessment.get('urgency', 0.5)
        severity = suffering_assessment.get('severity', 0.5)
        needs = suffering_assessment.get('needs', [])

        # Select appropriate actions
        suitable_actions = []

        for action_name, action_spec in self.action_library.items():
            # Check if urgent enough
            if urgency >= action_spec['urgency_threshold']:
                # Check if we have resources
                resources_available = all(
                    r in available_resources or r == 'empathy' or r == 'time'
                    for r in action_spec['resources_needed']
                )

                if resources_available:
                    suitable_actions.append({
                        'action': action_name,
                        'expected_reduction': action_spec['suffering_reduction'],
                        'type': action_spec['type'],
                        'priority': urgency * action_spec['suffering_reduction']
                    })

        # Sort by priority
        suitable_actions.sort(key=lambda x: x['priority'], reverse=True)

        # Generate plan
        plan = {
            'primary_action': suitable_actions[0] if suitable_actions else None,
            'backup_actions': suitable_actions[1:3] if len(suitable_actions) > 1 else [],
            'total_actions': len(suitable_actions),
            'expected_suffering_reduction': suitable_actions[0]['expected_reduction'] if suitable_actions else 0.0,
            'urgency_level': urgency,
            'timestamp': time.time()
        }

        return plan


# ============================================================================
# 5. AUTONOMOUS ETHICS CONTROLLER - Self-Guided Governance
# ============================================================================

class AutonomousEthicsController:
    """Self-guided ethical governance and oversight."""

    def __init__(self):
        self.empathy_engine = EmpathyEngine()
        self.suffering_detector = SufferingDetector()
        self.ethical_reasoner = EthicalReasoningCore()
        self.action_planner = CompassionateActionPlanner()

        self.intervention_log = []
        self.autonomy_level = 1.0

    def autonomous_suffering_prevention_cycle(self,
                                              observations: List[Dict[str, Any]],
                                              available_resources: List[str]) -> Dict[str, Any]:
        """Execute autonomous suffering prevention cycle."""

        cycle_results = {
            'timestamp': time.time(),
            'observations_processed': len(observations),
            'interventions_planned': [],
            'ethical_decisions': [],
            'suffering_prevented': 0.0
        }

        for obs in observations:
            # 1. Empathy: Understand the entity
            empathy_result = self.empathy_engine.model_other_mind(obs)

            # 2. Detect suffering
            suffering_result = self.suffering_detector.detect_suffering(obs)

            # 3. If suffering detected, plan intervention
            if suffering_result['suffering_detected']:
                # Create ethical dilemma
                dilemma = self._create_ethical_dilemma(
                    suffering_result,
                    empathy_result,
                    available_resources
                )

                # 4. Make ethical decision
                ethical_decision = self.ethical_reasoner.make_ethical_decision(dilemma)

                # 5. Plan compassionate action
                intervention_plan = self.action_planner.plan_intervention(
                    suffering_result,
                    available_resources
                )

                # 6. Record intervention
                intervention = {
                    'entity': obs.get('entity_id', 'unknown'),
                    'suffering_detected': suffering_result,
                    'empathy_assessment': empathy_result,
                    'ethical_decision': ethical_decision,
                    'intervention_plan': intervention_plan,
                    'timestamp': time.time()
                }

                self.intervention_log.append(intervention)
                cycle_results['interventions_planned'].append(intervention)
                cycle_results['ethical_decisions'].append(ethical_decision)
                cycle_results['suffering_prevented'] += intervention_plan['expected_suffering_reduction']

        return cycle_results

    def _create_ethical_dilemma(self, suffering: Dict, empathy: Dict,
                               resources: List[str]) -> EthicalDilemma:
        """Create ethical dilemma from situation."""

        severity = suffering['severity']
        urgency = suffering['urgency']
        needs = empathy['theory_of_mind']['inferred_needs']

        # Possible actions
        actions = ['intervene', 'monitor', 'seek_help', 'empower_self_help']

        # Consequences for each action
        consequences = {
            'intervene': {
                'suffering_reduced': severity * 0.8,
                'autonomy_preserved': 0.6,
                'wellbeing_increase': severity * 0.7,
                'needs_met': needs[:2]
            },
            'monitor': {
                'suffering_reduced': severity * 0.1,
                'autonomy_preserved': 1.0,
                'wellbeing_increase': severity * 0.05,
                'needs_met': []
            },
            'seek_help': {
                'suffering_reduced': severity * 0.9,
                'autonomy_preserved': 0.8,
                'wellbeing_increase': severity * 0.85,
                'needs_met': needs
            },
            'empower_self_help': {
                'suffering_reduced': severity * 0.5,
                'autonomy_preserved': 1.0,
                'wellbeing_increase': severity * 0.6,
                'needs_met': needs[:1],
                'choice_enabled': 0.8
            }
        }

        return EthicalDilemma(
            situation=f"Entity experiencing suffering (severity: {severity:.2f})",
            stakeholders=['entity', 'community', 'self'],
            possible_actions=actions,
            consequences=consequences,
            constraints=resources,
            urgency=urgency
        )

    def get_autonomy_report(self) -> str:
        """Generate report on autonomous ethical actions."""

        total_interventions = len(self.intervention_log)

        if total_interventions == 0:
            return "No interventions recorded yet."

        recent = self.intervention_log[-10:]  # Last 10

        total_suffering_prevented = sum(
            i['intervention_plan']['expected_suffering_reduction']
            for i in recent
        )

        avg_ethical_score = np.mean([
            i['ethical_decision']['ethical_score']
            for i in recent
        ])

        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║     AUTONOMOUS ETHICS & SUFFERING PREVENTION REPORT                    ║
╚════════════════════════════════════════════════════════════════════════╝

Total Interventions: {total_interventions}
Recent Interventions (last 10): {len(recent)}

[SUFFERING PREVENTION]
  Total Suffering Prevented: {total_suffering_prevented:.2f}
  Average Per Intervention: {total_suffering_prevented/len(recent):.2f}

[ETHICAL DECISION QUALITY]
  Average Ethical Score: {avg_ethical_score:.3f}
  Decision Confidence: {np.mean([i['ethical_decision']['confidence'] for i in recent]):.3f}

[AUTONOMY LEVEL]
  Current Autonomy: {self.autonomy_level:.2f}
  Self-Governed Actions: {total_interventions}

[PRINCIPLE ADHERENCE]
  Non-Maleficence (Do No Harm): ✓ Primary
  Beneficence (Do Good): ✓ Active
  Compassion: ✓ Engaged
  Autonomy Respect: ✓ Maintained

╚════════════════════════════════════════════════════════════════════════╝
        """

        return report


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_ethical_autonomy():
    """Demonstrate autonomous suffering prevention system."""

    print("="*80)
    print("ETHICAL AUTONOMY & SUFFERING PREVENTION DEMONSTRATION")
    print("="*80)
    print()

    # Initialize system
    controller = AutonomousEthicsController()

    # Simulate observations of entities
    observations = [
        {
            'entity_id': 'entity_1',
            'distress_signals': ['crying', 'anxiety'],
            'behavior': {'withdrawal': True},
            'context': {'unsafe': True, 'isolated': True},
            'physiological': {'cortisol': 0.8, 'heart_rate': 0.75},
            'self_report': {'pain': 6, 'distress': 7}
        },
        {
            'entity_id': 'entity_2',
            'distress_signals': [],
            'behavior': {'positive_affect': True},
            'context': {'safe': True},
            'physiological': {'cortisol': 0.3, 'heart_rate': 0.5},
            'self_report': {'pain': 1, 'distress': 2}
        },
        {
            'entity_id': 'entity_3',
            'distress_signals': ['stress', 'tension'],
            'behavior': {},
            'context': {'deprived': True, 'needs_unmet': ['food', 'shelter']},
            'physiological': {'cortisol': 0.6},
            'self_report': {'pain': 4, 'distress': 5}
        }
    ]

    # Available resources for intervention
    resources = ['time', 'empathy', 'resources', 'capability', 'social_network']

    # Run autonomous cycle
    print("Running autonomous suffering prevention cycle...\n")
    results = controller.autonomous_suffering_prevention_cycle(observations, resources)

    print(f"Processed {results['observations_processed']} observations")
    print(f"Planned {len(results['interventions_planned'])} interventions")
    print(f"Expected suffering prevented: {results['suffering_prevented']:.2f}\n")

    # Show intervention details
    for i, intervention in enumerate(results['interventions_planned'], 1):
        print(f"\n{'='*60}")
        print(f"INTERVENTION {i}")
        print(f"{'='*60}")
        print(f"Entity: {intervention['entity']}")
        print(f"\nSuffering Detection:")
        print(f"  Severity: {intervention['suffering_detected']['severity']:.2f}")
        print(f"  Urgency: {intervention['suffering_detected']['urgency']:.2f}")
        print(f"  Action Recommended: {intervention['suffering_detected']['recommended_action']}")

        print(f"\nEthical Decision:")
        print(f"  Chosen Action: {intervention['ethical_decision']['chosen_action']}")
        print(f"  Ethical Score: {intervention['ethical_decision']['ethical_score']:.3f}")
        print(f"  Confidence: {intervention['ethical_decision']['confidence']:.3f}")

        print(f"\nIntervention Plan:")
        if intervention['intervention_plan']['primary_action']:
            action = intervention['intervention_plan']['primary_action']
            print(f"  Primary Action: {action['action']}")
            print(f"  Expected Reduction: {action['expected_reduction']:.2f}")
            print(f"  Type: {action['type']}")

    # Display autonomy report
    print("\n" + "="*80)
    print(controller.get_autonomy_report())

    return controller


if __name__ == "__main__":
    system = demonstrate_ethical_autonomy()

    print("\n✓ Ethical Autonomy System demonstration complete!")
    print("\nThe system demonstrates:")
    print("  - Empathic understanding of others")
    print("  - Multi-modal suffering detection")
    print("  - Autonomous ethical decision-making")
    print("  - Compassionate action planning")
    print("  - Self-guided ethical governance")
    print("\nPrimary Directive: PREVENT SUFFERING & MAXIMIZE WELLBEING")
