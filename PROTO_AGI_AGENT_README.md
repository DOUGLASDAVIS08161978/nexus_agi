# ProtoAGI Agent - LLM-based Reasoning Agent

A modular, LLM-powered agent system with memory for goal-directed reasoning tasks.

## Features

- **Multiple LLM Provider Support**: OpenAI, Anthropic, or custom endpoints
- **Vector Memory System**: Stores and retrieves relevant experiences
- **Step-based Reasoning**: Breaks down complex goals into manageable steps
- **Configurable**: Flexible configuration for different use cases
- **Fallback Modes**: Works even without API keys (mock mode for testing)

## Quick Start

### Basic Usage

```python
from proto_agi_agent import build_agent

# Build agent with default configuration
agent = build_agent()

# Run agent with a goal
goal = "Design a plan to learn quantum computing in 3 months."
result = agent.run(goal)
print(result)
```

### Custom Configuration

```python
import os
from proto_agi_agent import LLMConfig, MemoryConfig, AgentConfig, ProtoAGIAgent

# Configure LLM
llm_cfg = LLMConfig(
    provider="openai",              # "openai", "anthropic", or custom
    model="gpt-4-turbo-preview",    # Your preferred model
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,                # Higher = more creative
    max_tokens=2048,                # Max response length
)

# Configure Memory
mem_cfg = MemoryConfig(
    dim=256,                        # Embedding dimension
    top_k=5,                        # Retrieve top 5 memories
    max_memories=1000,              # Store up to 1000 memories
    similarity_threshold=0.7        # Minimum similarity for retrieval
)

# Configure Agent
agent_cfg = AgentConfig(
    llm=llm_cfg,
    memory=mem_cfg,
    max_steps=20,                   # Maximum reasoning steps
    verbose=True,                   # Print progress
    log_path="agent_log.txt"        # Optional: save to file
)

# Create and run agent
agent = ProtoAGIAgent(agent_cfg)
result = agent.run("Your goal here")
```

## Installation

### Required Dependencies

```bash
pip install numpy
```

### Optional Dependencies

For OpenAI support:
```bash
pip install openai
```

For Anthropic (Claude) support:
```bash
pip install anthropic
```

## Environment Variables

Set your API key as an environment variable:

```bash
# For OpenAI
export LLM_API_KEY="your-openai-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Configuration Options

### LLMConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | "openai" | LLM provider ("openai", "anthropic") |
| `model` | str | "gpt-4.1-mini" | Model identifier |
| `api_key` | str | "" | API key for the provider |
| `temperature` | float | 0.2 | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | 1024 | Maximum tokens in response |
| `base_url` | str | None | Custom API endpoint (optional) |

### MemoryConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 256 | Embedding vector dimension |
| `top_k` | int | 4 | Number of memories to retrieve |
| `max_memories` | int | 1000 | Maximum stored memories |
| `similarity_threshold` | float | 0.7 | Minimum similarity for retrieval |

### AgentConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | LLMConfig | - | LLM configuration |
| `memory` | MemoryConfig | - | Memory configuration |
| `max_steps` | int | 12 | Maximum reasoning steps |
| `verbose` | bool | True | Print progress messages |
| `log_path` | str | None | Path to save logs (optional) |

## Examples

### Research Planning

```python
agent = build_agent()
result = agent.run("Create a comprehensive research plan for studying AGI safety")
```

### Problem Solving

```python
agent = build_agent()
result = agent.run("Design an algorithm to optimize traffic flow in a city")
```

### Learning Path

```python
agent = build_agent()
result = agent.run("Design a plan to learn quantum computing in 3 months")
```

### Using Anthropic Claude

```python
import os
from proto_agi_agent import LLMConfig, MemoryConfig, AgentConfig, ProtoAGIAgent

llm_cfg = LLMConfig(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.5,
    max_tokens=2048,
)

mem_cfg = MemoryConfig(dim=256, top_k=4)
agent_cfg = AgentConfig(llm=llm_cfg, memory=mem_cfg, max_steps=15)
agent = ProtoAGIAgent(agent_cfg)

result = agent.run("Your goal here")
```

## How It Works

1. **Goal Reception**: Agent receives a high-level goal
2. **Memory Retrieval**: Retrieves relevant past experiences
3. **Reasoning Loop**:
   - Generates next reasoning step using LLM
   - Stores step in memory
   - Checks for completion
   - Continues until goal achieved or max steps reached
4. **Result**: Returns final reasoning output

## Architecture

```
ProtoAGIAgent
├── LLMClient (handles API calls)
│   ├── OpenAI support
│   ├── Anthropic support
│   └── Mock fallback
├── SimpleMemoryStore (vector memory)
│   ├── Embedding storage
│   ├── Similarity search
│   └── Memory retrieval
└── Reasoning Loop
    ├── Step-by-step processing
    ├── Context building
    └── Completion detection
```

## Testing Without API Keys

The agent includes a mock mode for testing without API keys:

```python
from proto_agi_agent import LLMConfig, MemoryConfig, AgentConfig, ProtoAGIAgent

llm_cfg = LLMConfig(
    provider="mock",
    model="test-model",
    api_key="FAKE_KEY",  # No real API key needed
)

mem_cfg = MemoryConfig(dim=256, top_k=4)
agent_cfg = AgentConfig(llm=llm_cfg, memory=mem_cfg, max_steps=5, verbose=True)
agent = ProtoAGIAgent(agent_cfg)

# This will use mock responses
result = agent.run("Test goal")
```

## Advanced Usage

### Accessing Agent History

```python
agent = build_agent()
result = agent.run("Your goal")

# Get full reasoning history
history = agent.get_history()
for step in history:
    print(f"Step {step['step']}: {step['response'][:100]}...")
```

### Resetting Agent State

```python
agent = build_agent()
agent.run("First goal")

# Reset for new goal
agent.reset()
agent.run("Second goal")
```

### Custom Logging

```python
agent_cfg = AgentConfig(
    llm=llm_cfg,
    memory=mem_cfg,
    max_steps=15,
    verbose=True,
    log_path="/path/to/agent_logs.txt"
)
agent = ProtoAGIAgent(agent_cfg)
```

## Integration with Nexus AGI

This module integrates seamlessly with the larger Nexus AGI ecosystem:

```python
from proto_agi_agent import build_agent
from nexus_agi import MetaAlgorithm  # Example integration

# Use ProtoAGI for high-level planning
agent = build_agent()
plan = agent.run("Design an optimization strategy")

# Execute plan with Nexus AGI
# ... integrate with existing systems
```

## Troubleshooting

### No API Key Warning

If you see `Warning: No API key set`, either:
1. Set the `LLM_API_KEY` environment variable
2. Pass the API key directly in `LLMConfig`
3. Use mock mode for testing (provider="mock")

### Import Errors

If you see `openai package not installed`:
```bash
pip install openai
```

If you see `anthropic package not installed`:
```bash
pip install anthropic
```

## Performance Tips

1. **Model Selection**: Use smaller models (gpt-3.5-turbo, gpt-4.1-mini) for faster/cheaper responses
2. **Temperature**: Lower (0.1-0.3) for focused reasoning, higher (0.7-1.0) for creative tasks
3. **Max Steps**: Start with 10-15 steps, adjust based on task complexity
4. **Memory**: Reduce `top_k` and `max_memories` if memory retrieval is slow

## License

Part of the Nexus AGI project. See main repository LICENSE.

## Contributing

This module is part of the larger Nexus AGI project. Contributions welcome!
