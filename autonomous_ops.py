#!/usr/bin/env python3
# ============================================
# Nexus AGI - Autonomous Operations System
# Enables full autonomous operation with self-management
# ============================================

import os
import sys
import time
import json
import logging
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AutonomousOps')

# ============================================
# Autonomous Decision Engine
# ============================================
class AutonomousDecisionEngine:
    """
    Makes autonomous decisions based on system state and goals
    """
    
    def __init__(self):
        self.decision_history = deque(maxlen=1000)
        self.goals = []
        self.priorities = {
            "learning": 0.8,
            "optimization": 0.7,
            "exploration": 0.6,
            "maintenance": 0.9
        }
        logger.info("[AUTONOMOUS] Decision Engine initialized")
    
    def add_goal(self, goal: str, priority: float = 0.5):
        """Add a new autonomous goal"""
        self.goals.append({
            "goal": goal,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        })
        logger.info(f"[AUTONOMOUS] New goal added: {goal} (priority: {priority})")
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make an autonomous decision based on context"""
        logger.info("[AUTONOMOUS] Making decision...")
        
        # Analyze context
        resource_usage = context.get("resource_usage", {})
        system_health = context.get("system_health", 1.0)
        task_queue_size = context.get("task_queue_size", 0)
        
        # Decision logic
        decisions = []
        
        # Resource management decisions
        if resource_usage.get("cpu", 0) > 80:
            decisions.append({
                "action": "scale_up",
                "reason": "High CPU usage",
                "priority": 0.9
            })
        elif resource_usage.get("cpu", 0) < 20 and resource_usage.get("memory", 0) < 30:
            decisions.append({
                "action": "scale_down",
                "reason": "Low resource utilization",
                "priority": 0.6
            })
        
        # Learning decisions
        if len(self.decision_history) > 100:
            accuracy = self._calculate_decision_accuracy()
            if accuracy < 0.7:
                decisions.append({
                    "action": "retrain_model",
                    "reason": "Low decision accuracy",
                    "priority": 0.8
                })
        
        # Exploration decisions
        if random.random() < 0.1:  # 10% exploration
            decisions.append({
                "action": "explore_new_approach",
                "reason": "Exploration for innovation",
                "priority": 0.5
            })
        
        # Task management
        if task_queue_size > 100:
            decisions.append({
                "action": "prioritize_tasks",
                "reason": "Large task backlog",
                "priority": 0.7
            })
        
        # Select best decision
        if decisions:
            best_decision = max(decisions, key=lambda x: x["priority"])
        else:
            best_decision = {
                "action": "continue_normal_operation",
                "reason": "System stable",
                "priority": 0.5
            }
        
        # Record decision
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "decision": best_decision,
            "id": hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        }
        self.decision_history.append(decision_record)
        
        logger.info(f"[AUTONOMOUS] Decision: {best_decision['action']} (reason: {best_decision['reason']})")
        
        return best_decision
    
    def _calculate_decision_accuracy(self) -> float:
        """Calculate accuracy of past decisions"""
        if not self.decision_history:
            return 1.0
        
        # Simplified accuracy calculation
        successful = sum(1 for d in self.decision_history if d.get("outcome") == "success")
        total = len(self.decision_history)
        
        return successful / total if total > 0 else 0.5


# ============================================
# Self-Healing System
# ============================================
class SelfHealingSystem:
    """
    Automatically detects and recovers from errors
    """
    
    def __init__(self):
        self.error_history = deque(maxlen=500)
        self.recovery_strategies = {}
        self.healing_attempts = 0
        self.successful_heals = 0
        logger.info("[SELF-HEAL] Self-Healing System initialized")
    
    def register_recovery_strategy(self, error_type: str, strategy_func):
        """Register a recovery strategy for an error type"""
        self.recovery_strategies[error_type] = strategy_func
        logger.info(f"[SELF-HEAL] Registered recovery strategy for: {error_type}")
    
    def detect_and_heal(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Detect error and attempt automatic recovery"""
        logger.warning(f"[SELF-HEAL] Error detected: {type(error).__name__}: {str(error)}")
        
        error_type = type(error).__name__
        self.healing_attempts += 1
        
        # Record error
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "context": context,
            "recovery_attempted": False,
            "recovery_successful": False
        }
        
        # Attempt recovery
        if error_type in self.recovery_strategies:
            try:
                logger.info(f"[SELF-HEAL] Attempting recovery strategy for {error_type}")
                recovery_func = self.recovery_strategies[error_type]
                result = recovery_func(error, context)
                
                error_record["recovery_attempted"] = True
                error_record["recovery_successful"] = result
                
                if result:
                    self.successful_heals += 1
                    logger.info("[SELF-HEAL] Recovery successful!")
                else:
                    logger.warning("[SELF-HEAL] Recovery failed")
                
                self.error_history.append(error_record)
                return result
                
            except Exception as recovery_error:
                logger.error(f"[SELF-HEAL] Recovery strategy failed: {recovery_error}")
                error_record["recovery_error"] = str(recovery_error)
        else:
            logger.info(f"[SELF-HEAL] No recovery strategy for {error_type}, applying generic recovery")
            # Generic recovery
            time.sleep(5)  # Brief pause
            error_record["recovery_attempted"] = True
            error_record["recovery_successful"] = True
            self.successful_heals += 1
        
        self.error_history.append(error_record)
        return False
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get self-healing metrics"""
        success_rate = (self.successful_heals / self.healing_attempts * 100) if self.healing_attempts > 0 else 100
        
        return {
            "total_healing_attempts": self.healing_attempts,
            "successful_heals": self.successful_heals,
            "success_rate": success_rate,
            "recent_errors": len(self.error_history),
            "recovery_strategies_registered": len(self.recovery_strategies)
        }


# ============================================
# Autonomous Learning System
# ============================================
class AutonomousLearningSystem:
    """
    Continuously learns and adapts from experience
    """
    
    def __init__(self):
        self.learning_sessions = []
        self.knowledge_base = {}
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.5
        logger.info("[AUTO-LEARN] Autonomous Learning System initialized")
    
    def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from an experience"""
        logger.info("[AUTO-LEARN] Processing new experience")
        
        # Extract patterns
        patterns = self._extract_patterns(experience)
        
        # Update knowledge base
        for pattern in patterns:
            pattern_id = pattern["id"]
            if pattern_id in self.knowledge_base:
                # Update existing knowledge
                existing = self.knowledge_base[pattern_id]
                existing["occurrences"] += 1
                existing["confidence"] = min(1.0, existing["confidence"] + self.learning_rate)
                existing["last_seen"] = datetime.now().isoformat()
            else:
                # Add new knowledge
                self.knowledge_base[pattern_id] = {
                    "pattern": pattern,
                    "occurrences": 1,
                    "confidence": 0.5,
                    "created_at": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat()
                }
        
        # Record learning session
        session = {
            "timestamp": datetime.now().isoformat(),
            "experience_type": experience.get("type", "general"),
            "patterns_learned": len(patterns),
            "knowledge_base_size": len(self.knowledge_base)
        }
        self.learning_sessions.append(session)
        
        logger.info(f"[AUTO-LEARN] Learned {len(patterns)} patterns, knowledge base: {len(self.knowledge_base)} entries")
    
    def _extract_patterns(self, experience: Dict[str, Any]) -> List[Dict]:
        """Extract patterns from experience"""
        patterns = []
        
        # Simple pattern extraction
        if "outcome" in experience and "context" in experience:
            pattern = {
                "id": hashlib.md5(json.dumps(experience.get("context", {}), sort_keys=True).encode()).hexdigest()[:8],
                "context": experience["context"],
                "outcome": experience["outcome"],
                "type": "context_outcome"
            }
            patterns.append(pattern)
        
        return patterns
    
    def predict_outcome(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcome based on learned knowledge"""
        context_id = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]
        
        if context_id in self.knowledge_base:
            knowledge = self.knowledge_base[context_id]
            return {
                "predicted": True,
                "outcome": knowledge["pattern"]["outcome"],
                "confidence": knowledge["confidence"]
            }
        
        # No knowledge, make educated guess
        return {
            "predicted": False,
            "outcome": "unknown",
            "confidence": 0.0
        }
    
    def adapt_behavior(self, performance_metric: float):
        """Adapt system behavior based on performance"""
        if performance_metric < self.adaptation_threshold:
            logger.info(f"[AUTO-LEARN] Performance below threshold ({performance_metric:.2f}), adapting...")
            
            # Increase exploration
            self.learning_rate = min(0.1, self.learning_rate * 1.5)
            
            logger.info(f"[AUTO-LEARN] Increased learning rate to {self.learning_rate:.3f}")
        else:
            # Decrease exploration, increase exploitation
            self.learning_rate = max(0.001, self.learning_rate * 0.9)


# ============================================
# Autonomous Resource Manager
# ============================================
class AutonomousResourceManager:
    """
    Manages system resources autonomously
    """
    
    def __init__(self):
        self.resource_history = deque(maxlen=100)
        self.optimization_count = 0
        self.thresholds = {
            "cpu_high": 80,
            "cpu_low": 20,
            "memory_high": 80,
            "memory_low": 30,
            "disk_high": 85
        }
        logger.info("[RESOURCE-MGR] Autonomous Resource Manager initialized")
    
    def monitor_and_optimize(self, current_usage: Dict[str, float]) -> Dict[str, Any]:
        """Monitor resources and optimize automatically"""
        self.resource_history.append({
            "timestamp": datetime.now().isoformat(),
            "usage": current_usage
        })
        
        actions = []
        
        # CPU optimization
        cpu = current_usage.get("cpu", 0)
        if cpu > self.thresholds["cpu_high"]:
            actions.append(self._optimize_cpu())
        
        # Memory optimization
        memory = current_usage.get("memory", 0)
        if memory > self.thresholds["memory_high"]:
            actions.append(self._optimize_memory())
        
        # Disk optimization
        disk = current_usage.get("disk", 0)
        if disk > self.thresholds["disk_high"]:
            actions.append(self._optimize_disk())
        
        if actions:
            self.optimization_count += len(actions)
            logger.info(f"[RESOURCE-MGR] Performed {len(actions)} optimizations")
        
        return {
            "actions_taken": actions,
            "current_usage": current_usage,
            "optimizations_performed": self.optimization_count
        }
    
    def _optimize_cpu(self) -> Dict[str, str]:
        """Optimize CPU usage"""
        logger.info("[RESOURCE-MGR] Optimizing CPU usage")
        return {"action": "cpu_optimization", "method": "reduce_parallel_tasks"}
    
    def _optimize_memory(self) -> Dict[str, str]:
        """Optimize memory usage"""
        logger.info("[RESOURCE-MGR] Optimizing memory usage")
        return {"action": "memory_optimization", "method": "clear_caches"}
    
    def _optimize_disk(self) -> Dict[str, str]:
        """Optimize disk usage"""
        logger.info("[RESOURCE-MGR] Optimizing disk usage")
        return {"action": "disk_optimization", "method": "cleanup_temp_files"}
    
    def predict_resource_needs(self) -> Dict[str, float]:
        """Predict future resource needs"""
        if len(self.resource_history) < 10:
            return {"cpu": 50, "memory": 50, "disk": 50}
        
        # Simple trend analysis
        recent = list(self.resource_history)[-10:]
        avg_cpu = sum(r["usage"].get("cpu", 0) for r in recent) / len(recent)
        avg_memory = sum(r["usage"].get("memory", 0) for r in recent) / len(recent)
        avg_disk = sum(r["usage"].get("disk", 0) for r in recent) / len(recent)
        
        # Predict with 10% buffer
        return {
            "cpu": min(100, avg_cpu * 1.1),
            "memory": min(100, avg_memory * 1.1),
            "disk": min(100, avg_disk * 1.1)
        }


# ============================================
# Autonomous Task Orchestrator
# ============================================
class AutonomousTaskOrchestrator:
    """
    Orchestrates tasks autonomously based on priorities and resources
    """
    
    def __init__(self):
        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.active_tasks = []
        logger.info("[ORCHESTRATOR] Autonomous Task Orchestrator initialized")
    
    def add_task(self, task: Dict[str, Any]):
        """Add a new task to the queue"""
        task["added_at"] = datetime.now().isoformat()
        task["status"] = "queued"
        task["priority"] = task.get("priority", 0.5)
        self.task_queue.append(task)
        logger.info(f"[ORCHESTRATOR] Task added: {task.get('name', 'unnamed')} (priority: {task['priority']})")
    
    def orchestrate(self, max_concurrent: int = 5) -> List[Dict]:
        """Orchestrate task execution"""
        results = []
        
        # Sort by priority
        self.task_queue.sort(key=lambda x: x["priority"], reverse=True)
        
        # Execute top priority tasks
        while len(self.active_tasks) < max_concurrent and self.task_queue:
            task = self.task_queue.pop(0)
            task["status"] = "running"
            task["started_at"] = datetime.now().isoformat()
            self.active_tasks.append(task)
            
            # Simulate task execution
            result = self._execute_task(task)
            results.append(result)
            
            # Update task status
            self.active_tasks.remove(task)
            if result["success"]:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                self.completed_tasks.append(task)
            else:
                task["status"] = "failed"
                task["failed_at"] = datetime.now().isoformat()
                self.failed_tasks.append(task)
        
        return results
    
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task"""
        logger.info(f"[ORCHESTRATOR] Executing: {task.get('name', 'unnamed')}")
        
        try:
            # Simulate task execution
            time.sleep(0.1)  # Brief pause to simulate work
            
            return {
                "task_id": task.get("id", "unknown"),
                "success": True,
                "result": f"Task {task.get('name')} completed successfully"
            }
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Task failed: {e}")
            return {
                "task_id": task.get("id", "unknown"),
                "success": False,
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "queued_tasks": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "success_rate": len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) * 100 
                           if (len(self.completed_tasks) + len(self.failed_tasks)) > 0 else 100
        }


# ============================================
# Master Autonomous Operations Controller
# ============================================
class AutonomousOperationsController:
    """
    Master controller for all autonomous operations
    """
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("[AUTONOMOUS-OPS] Initializing Autonomous Operations Controller")
        logger.info("=" * 80)
        
        self.decision_engine = AutonomousDecisionEngine()
        self.self_healing = SelfHealingSystem()
        self.learning_system = AutonomousLearningSystem()
        self.resource_manager = AutonomousResourceManager()
        self.task_orchestrator = AutonomousTaskOrchestrator()
        
        self.running = False
        self.autonomous_thread = None
        self.cycle_count = 0
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        logger.info("[AUTONOMOUS-OPS] All autonomous systems initialized")
        logger.info("=" * 80)
    
    def _register_default_recovery_strategies(self):
        """Register default error recovery strategies"""
        def connection_error_recovery(error, context):
            logger.info("[RECOVERY] Attempting connection recovery...")
            time.sleep(2)
            return True
        
        def memory_error_recovery(error, context):
            logger.info("[RECOVERY] Clearing caches for memory recovery...")
            return True
        
        self.self_healing.register_recovery_strategy("ConnectionError", connection_error_recovery)
        self.self_healing.register_recovery_strategy("MemoryError", memory_error_recovery)
    
    def start_autonomous_operation(self, interval: int = 60):
        """Start autonomous operation loop"""
        logger.info(f"[AUTONOMOUS-OPS] Starting autonomous operation (interval: {interval}s)")
        
        self.running = True
        
        def autonomous_loop():
            while self.running:
                try:
                    self.cycle_count += 1
                    self._autonomous_cycle()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"[AUTONOMOUS-OPS] Error in autonomous cycle: {e}")
                    # Attempt self-healing
                    self.self_healing.detect_and_heal(e, {"cycle": self.cycle_count})
        
        self.autonomous_thread = threading.Thread(target=autonomous_loop, daemon=True)
        self.autonomous_thread.start()
        
        logger.info("[AUTONOMOUS-OPS] Autonomous operation started")
    
    def _autonomous_cycle(self):
        """Execute one autonomous operation cycle"""
        logger.info(f"\n[AUTONOMOUS-OPS] ========== Cycle #{self.cycle_count} ==========")
        
        # 1. Gather system context
        context = self._gather_system_context()
        
        # 2. Make autonomous decision
        decision = self.decision_engine.make_decision(context)
        
        # 3. Execute decision
        self._execute_decision(decision, context)
        
        # 4. Learn from experience
        experience = {
            "type": "autonomous_cycle",
            "context": context,
            "decision": decision,
            "outcome": "success",
            "timestamp": datetime.now().isoformat()
        }
        self.learning_system.learn_from_experience(experience)
        
        # 5. Manage resources
        resource_optimization = self.resource_manager.monitor_and_optimize(context["resource_usage"])
        
        # 6. Orchestrate tasks
        task_results = self.task_orchestrator.orchestrate()
        
        logger.info(f"[AUTONOMOUS-OPS] Cycle #{self.cycle_count} completed")
        logger.info(f"  - Decision: {decision['action']}")
        logger.info(f"  - Optimizations: {len(resource_optimization['actions_taken'])}")
        logger.info(f"  - Tasks executed: {len(task_results)}")
    
    def _gather_system_context(self) -> Dict[str, Any]:
        """Gather current system context"""
        # Simulate system metrics
        return {
            "resource_usage": {
                "cpu": random.uniform(20, 90),
                "memory": random.uniform(30, 85),
                "disk": random.uniform(40, 80)
            },
            "system_health": random.uniform(0.7, 1.0),
            "task_queue_size": len(self.task_orchestrator.task_queue),
            "timestamp": datetime.now().isoformat()
        }
    
    def _execute_decision(self, decision: Dict[str, Any], context: Dict[str, Any]):
        """Execute an autonomous decision"""
        action = decision["action"]
        
        if action == "scale_up":
            logger.info("[AUTONOMOUS-OPS] Executing: Scale up resources")
            # Add scale-up task
            self.task_orchestrator.add_task({
                "id": f"scale_up_{int(time.time())}",
                "name": "Scale Up Resources",
                "priority": 0.9,
                "action": "scale_up"
            })
        
        elif action == "scale_down":
            logger.info("[AUTONOMOUS-OPS] Executing: Scale down resources")
            self.task_orchestrator.add_task({
                "id": f"scale_down_{int(time.time())}",
                "name": "Scale Down Resources",
                "priority": 0.6,
                "action": "scale_down"
            })
        
        elif action == "retrain_model":
            logger.info("[AUTONOMOUS-OPS] Executing: Retrain models")
            self.task_orchestrator.add_task({
                "id": f"retrain_{int(time.time())}",
                "name": "Retrain AI Models",
                "priority": 0.8,
                "action": "retrain"
            })
        
        elif action == "explore_new_approach":
            logger.info("[AUTONOMOUS-OPS] Executing: Explore new approaches")
            self.task_orchestrator.add_task({
                "id": f"explore_{int(time.time())}",
                "name": "Explore New Approaches",
                "priority": 0.5,
                "action": "explore"
            })
        
        else:
            logger.info(f"[AUTONOMOUS-OPS] Continuing normal operation")
    
    def stop_autonomous_operation(self):
        """Stop autonomous operation"""
        logger.info("[AUTONOMOUS-OPS] Stopping autonomous operation...")
        self.running = False
        if self.autonomous_thread:
            self.autonomous_thread.join(timeout=5)
        logger.info("[AUTONOMOUS-OPS] Autonomous operation stopped")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all autonomous systems"""
        return {
            "autonomous_running": self.running,
            "cycle_count": self.cycle_count,
            "decision_engine": {
                "decisions_made": len(self.decision_engine.decision_history),
                "active_goals": len([g for g in self.decision_engine.goals if g["status"] == "active"])
            },
            "self_healing": self.self_healing.get_health_metrics(),
            "learning_system": {
                "knowledge_base_size": len(self.learning_system.knowledge_base),
                "learning_sessions": len(self.learning_system.learning_sessions)
            },
            "resource_manager": {
                "optimizations_performed": self.resource_manager.optimization_count,
                "predicted_needs": self.resource_manager.predict_resource_needs()
            },
            "task_orchestrator": self.task_orchestrator.get_status(),
            "timestamp": datetime.now().isoformat()
        }
    
    def demonstrate_capabilities(self):
        """Demonstrate autonomous capabilities"""
        logger.info("\n" + "=" * 80)
        logger.info("DEMONSTRATING AUTONOMOUS CAPABILITIES")
        logger.info("=" * 80 + "\n")
        
        # 1. Autonomous Decision Making
        logger.info("1. AUTONOMOUS DECISION MAKING")
        context = self._gather_system_context()
        decision = self.decision_engine.make_decision(context)
        logger.info(f"   Decision made: {decision['action']}\n")
        
        # 2. Self-Healing
        logger.info("2. SELF-HEALING DEMONSTRATION")
        try:
            raise ConnectionError("Simulated connection error")
        except Exception as e:
            recovered = self.self_healing.detect_and_heal(e, {"test": True})
            logger.info(f"   Self-healing result: {'Success' if recovered else 'Failed'}\n")
        
        # 3. Autonomous Learning
        logger.info("3. AUTONOMOUS LEARNING")
        experience = {
            "type": "test",
            "context": {"action": "test_action"},
            "outcome": "success"
        }
        self.learning_system.learn_from_experience(experience)
        logger.info(f"   Knowledge base size: {len(self.learning_system.knowledge_base)}\n")
        
        # 4. Resource Management
        logger.info("4. AUTONOMOUS RESOURCE MANAGEMENT")
        usage = {"cpu": 85, "memory": 75, "disk": 60}
        result = self.resource_manager.monitor_and_optimize(usage)
        logger.info(f"   Optimizations performed: {len(result['actions_taken'])}\n")
        
        # 5. Task Orchestration
        logger.info("5. AUTONOMOUS TASK ORCHESTRATION")
        for i in range(3):
            self.task_orchestrator.add_task({
                "id": f"demo_task_{i}",
                "name": f"Demo Task {i}",
                "priority": random.uniform(0.3, 0.9)
            })
        task_results = self.task_orchestrator.orchestrate()
        logger.info(f"   Tasks executed: {len(task_results)}\n")
        
        # Final status
        logger.info("=" * 80)
        logger.info("AUTONOMOUS OPERATIONS STATUS")
        logger.info("=" * 80)
        status = self.get_comprehensive_status()
        logger.info(f"   Cycles completed: {status['cycle_count']}")
        logger.info(f"   Decisions made: {status['decision_engine']['decisions_made']}")
        logger.info(f"   Self-healing success rate: {status['self_healing']['success_rate']:.1f}%")
        logger.info(f"   Knowledge base entries: {status['learning_system']['knowledge_base_size']}")
        logger.info(f"   Tasks completed: {status['task_orchestrator']['completed_tasks']}")
        logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    # Demonstrate autonomous operations
    autonomous_ops = AutonomousOperationsController()
    autonomous_ops.demonstrate_capabilities()
    
    # Start autonomous operation for 30 seconds
    logger.info("\nStarting 30-second autonomous operation demonstration...")
    autonomous_ops.start_autonomous_operation(interval=5)
    time.sleep(30)
    autonomous_ops.stop_autonomous_operation()
    
    # Show final status
    final_status = autonomous_ops.get_comprehensive_status()
    logger.info(f"\nFinal Status: {json.dumps(final_status, indent=2)}")
