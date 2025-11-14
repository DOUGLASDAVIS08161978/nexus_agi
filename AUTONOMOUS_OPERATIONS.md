# Autonomous Operations Guide

This guide explains the comprehensive autonomous operation capabilities added to Nexus AGI and ARIA systems.

## Overview

The autonomous operations system enables both Nexus AGI and ARIA to operate completely independently with minimal human intervention. The system includes:

- **Autonomous Decision Making** - Makes intelligent decisions based on system state
- **Self-Healing** - Automatically detects and recovers from errors
- **Continuous Learning** - Learns from experience and adapts behavior
- **Resource Management** - Monitors and optimizes resource usage
- **Task Orchestration** - Manages and prioritizes tasks autonomously

## Features

### 1. Autonomous Decision Engine

Makes intelligent decisions based on current system context:
- Resource scaling (up/down) based on CPU/memory usage
- Model retraining when accuracy drops
- Task prioritization when queues grow
- Exploratory actions for innovation

**Python:**
```python
from autonomous_ops import AutonomousDecisionEngine

engine = AutonomousDecisionEngine()
engine.add_goal("optimize_performance", priority=0.8)

context = {
    "resource_usage": {"cpu": 85, "memory": 70},
    "system_health": 0.9
}
decision = engine.make_decision(context)
# Returns: {"action": "scale_up", "reason": "High CPU usage", "priority": 0.9}
```

**JavaScript:**
```javascript
const { AutonomousDecisionEngine } = require('./aria_autonomous.js');

const engine = new AutonomousDecisionEngine();
engine.addGoal('optimize_performance', 0.8);

const context = {
    resourceUsage: { cpu: 85, memory: 70 },
    systemHealth: 0.9
};
const decision = engine.makeDecision(context);
// Returns: {action: "scale_up", reason: "High CPU usage", priority: 0.9}
```

### 2. Self-Healing System

Automatically detects errors and applies recovery strategies:
- Registered recovery strategies for common errors
- Generic recovery for unknown errors
- Tracks success rate and healing history

**Python:**
```python
from autonomous_ops import SelfHealingSystem

healing = SelfHealingSystem()

# Register custom recovery strategy
def connection_recovery(error, context):
    print("Attempting reconnection...")
    # Recovery logic here
    return True

healing.register_recovery_strategy("ConnectionError", connection_recovery)

# Automatic recovery
try:
    # Some operation that might fail
    raise ConnectionError("Connection lost")
except Exception as e:
    recovered = healing.detect_and_heal(e, {"operation": "fetch_data"})
    if recovered:
        print("Successfully recovered!")
```

**JavaScript:**
```javascript
const { SelfHealingSystem } = require('./aria_autonomous.js');

const healing = new SelfHealingSystem();

// Register custom recovery strategy
const connectionRecovery = async (error, context) => {
    console.log('Attempting reconnection...');
    // Recovery logic here
    return true;
};

healing.registerRecoveryStrategy('ConnectionError', connectionRecovery);

// Automatic recovery
try {
    throw new Error('Connection lost');
} catch (error) {
    error.name = 'ConnectionError';
    const recovered = await healing.detectAndHeal(error, {operation: 'fetch_data'});
    if (recovered) {
        console.log('Successfully recovered!');
    }
}
```

### 3. Autonomous Learning System

Learns from experiences and adapts behavior:
- Pattern extraction from experiences
- Knowledge base building
- Outcome prediction
- Adaptive behavior based on performance

**Python:**
```python
from autonomous_ops import AutonomousLearningSystem

learning = AutonomousLearningSystem()

# Learn from experience
experience = {
    "type": "optimization",
    "context": {"action": "cache_clear", "resource_before": 85},
    "outcome": "success"
}
learning.learn_from_experience(experience)

# Predict future outcomes
context = {"action": "cache_clear", "resource_before": 90}
prediction = learning.predict_outcome(context)
print(f"Predicted: {prediction['outcome']} (confidence: {prediction['confidence']})")

# Adapt based on performance
learning.adapt_behavior(0.6)  # Low performance triggers adaptation
```

**JavaScript:**
```javascript
const { AutonomousLearningSystem } = require('./aria_autonomous.js');

const learning = new AutonomousLearningSystem();

// Learn from experience
const experience = {
    type: 'optimization',
    context: {action: 'cache_clear', resourceBefore: 85},
    outcome: 'success'
};
learning.learnFromExperience(experience);

// Predict future outcomes
const context = {action: 'cache_clear', resourceBefore: 90};
const prediction = learning.predictOutcome(context);
console.log(`Predicted: ${prediction.outcome} (confidence: ${prediction.confidence})`);

// Adapt based on performance
learning.adaptBehavior(0.6);  // Low performance triggers adaptation
```

### 4. Autonomous Resource Manager

Monitors and optimizes system resources:
- CPU optimization
- Memory management
- Disk cleanup
- Resource prediction

**Python:**
```python
from autonomous_ops import AutonomousResourceManager

resource_mgr = AutonomousResourceManager()

# Monitor and optimize
current_usage = {"cpu": 85, "memory": 75, "disk": 60}
result = resource_mgr.monitor_and_optimize(current_usage)
print(f"Actions taken: {result['actions_taken']}")

# Predict future needs
predicted = resource_mgr.predict_resource_needs()
print(f"Predicted CPU need: {predicted['cpu']}%")
```

### 5. Autonomous Task Orchestrator

Manages task execution based on priorities:
- Priority-based task queuing
- Concurrent task execution
- Success/failure tracking
- Performance metrics

**Python:**
```python
from autonomous_ops import AutonomousTaskOrchestrator

orchestrator = AutonomousTaskOrchestrator()

# Add tasks
orchestrator.add_task({
    "id": "task_1",
    "name": "Data Processing",
    "priority": 0.9
})

orchestrator.add_task({
    "id": "task_2",
    "name": "Model Training",
    "priority": 0.7
})

# Execute tasks
results = orchestrator.orchestrate(max_concurrent=5)
status = orchestrator.get_status()
print(f"Success rate: {status['success_rate']}%")
```

## Integration with Services

### Nexus AGI Service

Autonomous operations are automatically enabled when running the service:

```bash
# Run with autonomous operations (default)
python3 nexus_service.py --interval 300

# Disable autonomous operations
python3 nexus_service.py --interval 300 --no-autonomous
```

The service will:
1. Initialize autonomous controller
2. Start autonomous operation loop in background
3. Make decisions every cycle
4. Self-heal from errors automatically
5. Learn from each cycle
6. Optimize resources continuously
7. Show autonomous metrics on shutdown

### ARIA Service

Similarly, ARIA service supports autonomous operations:

```bash
# Run with autonomous operations (default)
node aria_service.js --interval 300

# Disable autonomous operations
node aria_service.js --interval 300 --no-autonomous
```

## Master Autonomous Controller

Both systems have a master controller that coordinates all autonomous subsystems:

### Python: AutonomousOperationsController

```python
from autonomous_ops import AutonomousOperationsController

# Initialize
controller = AutonomousOperationsController()

# Start autonomous operation (runs in background thread)
controller.start_autonomous_operation(interval=60)  # 60 seconds

# Get comprehensive status
status = controller.get_comprehensive_status()
print(f"Decisions made: {status['decision_engine']['decisions_made']}")
print(f"Self-healing success: {status['self_healing']['success_rate']}%")
print(f"Knowledge base: {status['learning_system']['knowledge_base_size']}")

# Stop autonomous operation
controller.stop_autonomous_operation()
```

### JavaScript: ARIAAutonomousController

```javascript
const { ARIAAutonomousController } = require('./aria_autonomous.js');

// Initialize
const controller = new ARIAAutonomousController();

// Start autonomous operation (runs with setInterval)
controller.startAutonomousOperation(60000);  // 60 seconds

// Get comprehensive status
const status = controller.getComprehensiveStatus();
console.log(`Decisions made: ${status.decisionEngine.decisionsMade}`);
console.log(`Self-healing success: ${status.selfHealing.successRate}%`);
console.log(`Knowledge base: ${status.learningSystem.knowledgeBaseSize}`);

// Stop autonomous operation
controller.stopAutonomousOperation();
```

## Autonomous Cycle Flow

Each autonomous cycle follows this flow:

1. **Gather Context** - Collect system metrics and state
2. **Make Decision** - Autonomous decision engine analyzes context
3. **Execute Decision** - Perform determined action
4. **Learn** - Record experience and update knowledge base
5. **Optimize Resources** - Check and optimize resource usage
6. **Orchestrate Tasks** - Execute queued tasks by priority
7. **Self-Heal** - Recover from any errors automatically

## Monitoring Autonomous Operations

### Metrics Available

- **Decision Engine**
  - Total decisions made
  - Decision accuracy
  - Active goals

- **Self-Healing**
  - Total healing attempts
  - Successful heals
  - Success rate
  - Recovery strategies registered

- **Learning System**
  - Knowledge base size
  - Learning sessions completed
  - Learning rate

- **Resource Manager**
  - Optimizations performed
  - Predicted resource needs
  - Resource history

- **Task Orchestrator**
  - Queued tasks
  - Active tasks
  - Completed tasks
  - Failed tasks
  - Success rate

### Log Output

The autonomous system provides detailed logging:

```
[AUTONOMOUS-OPS] ========== Cycle #1 ==========
[AUTONOMOUS] Making decision...
[AUTONOMOUS] Decision: scale_up (reason: High CPU usage)
[AUTONOMOUS-OPS] Executing: Scale up resources
[AUTO-LEARN] Processing new experience
[AUTO-LEARN] Learned 1 patterns, knowledge base: 5 entries
[RESOURCE-MGR] Performed 2 optimizations
[ORCHESTRATOR] Tasks executed: 3
[AUTONOMOUS-OPS] Cycle #1 completed
```

## Configuration

### Thresholds

Modify thresholds in the resource manager:

**Python:**
```python
controller.resource_manager.thresholds = {
    "cpu_high": 85,
    "cpu_low": 15,
    "memory_high": 85,
    "memory_low": 25,
    "disk_high": 90
}
```

**JavaScript:**
```javascript
controller.resourceManager.thresholds = {
    cpuHigh: 85,
    cpuLow: 15,
    memoryHigh: 85,
    memoryLow: 25,
    diskHigh: 90
};
```

### Goals and Priorities

Add autonomous goals:

**Python:**
```python
controller.decision_engine.add_goal("maximize_throughput", priority=0.9)
controller.decision_engine.add_goal("minimize_latency", priority=0.8)
controller.decision_engine.add_goal("optimize_cost", priority=0.7)
```

**JavaScript:**
```javascript
controller.decisionEngine.addGoal('maximize_throughput', 0.9);
controller.decisionEngine.addGoal('minimize_latency', 0.8);
controller.decisionEngine.addGoal('optimize_cost', 0.7);
```

## Advanced Usage

### Custom Recovery Strategies

Register custom recovery strategies for specific error types:

**Python:**
```python
def custom_recovery(error, context):
    # Custom recovery logic
    print(f"Recovering from: {error}")
    # Perform recovery actions
    return True  # Return True if recovery successful

controller.self_healing.register_recovery_strategy("CustomError", custom_recovery)
```

**JavaScript:**
```javascript
const customRecovery = async (error, context) => {
    // Custom recovery logic
    console.log(`Recovering from: ${error.message}`);
    // Perform recovery actions
    return true;  // Return true if recovery successful
};

controller.selfHealing.registerRecoveryStrategy('CustomError', customRecovery);
```

### Learning from Custom Experiences

Feed custom experiences to the learning system:

**Python:**
```python
controller.learning_system.learn_from_experience({
    "type": "custom_operation",
    "context": {"parameter": "value"},
    "outcome": "success"
})
```

**JavaScript:**
```javascript
controller.learningSystem.learnFromExperience({
    type: 'custom_operation',
    context: {parameter: 'value'},
    outcome: 'success'
});
```

## Demonstrations

Run standalone demonstrations:

```bash
# Python
python3 autonomous_ops.py

# JavaScript
node aria_autonomous.js
```

These will demonstrate all autonomous capabilities including:
- Decision making
- Self-healing
- Learning
- Resource management
- Task orchestration
- 30-second autonomous operation

## Best Practices

1. **Monitor Metrics** - Regularly check autonomous operation metrics
2. **Adjust Thresholds** - Tune thresholds based on your workload
3. **Register Recovery Strategies** - Add custom recovery for domain-specific errors
4. **Set Clear Goals** - Define autonomous goals with appropriate priorities
5. **Review Decisions** - Periodically review decision history for insights
6. **Enable Logging** - Keep detailed logs for debugging and analysis
7. **Test in Stages** - Start with short intervals, gradually increase

## Troubleshooting

### Autonomous Operations Not Starting

Check that the modules are importable:
```bash
python3 -c "from autonomous_ops import AutonomousOperationsController"
node -e "const {ARIAAutonomousController} = require('./aria_autonomous.js')"
```

### High Resource Usage

Increase autonomous cycle interval:
```bash
python3 nexus_service.py --interval 600  # 10 minutes
node aria_service.js --interval 600
```

### Low Decision Accuracy

The system adapts automatically, but you can:
- Add more training data via experiences
- Adjust learning rate
- Register more recovery strategies

## Summary

The autonomous operations system provides:
- ✅ Complete autonomous operation
- ✅ Self-healing from errors
- ✅ Continuous learning and adaptation
- ✅ Intelligent decision making
- ✅ Resource optimization
- ✅ Task prioritization and orchestration
- ✅ Comprehensive monitoring and metrics
- ✅ Easy integration with existing services
- ✅ Customizable and extensible

The system can now operate with minimal human intervention while continuously improving its performance and resilience.
