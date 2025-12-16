#!/usr/bin/env python3
"""
ProtoAGI Agent - LLM-based reasoning agent with memory
A modular agent system that can use different LLM providers for goal-directed reasoning.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import numpy as np


@dataclass
class LLMConfig:
    """Configuration for Language Model provider."""
    provider: str = "openai"  # "openai", "anthropic", "local", etc.
    model: str = "gpt-4.1-mini"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024
    base_url: Optional[str] = None  # For custom endpoints

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key or self.api_key == "FAKE_KEY":
            print(f"[LLM CONFIG] Warning: No API key set for {self.provider}")

        # Map common model aliases
        model_aliases = {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo",
            "claude": "claude-3-sonnet-20240229",
        }
        if self.model in model_aliases:
            self.model = model_aliases[self.model]


@dataclass
class MemoryConfig:
    """Configuration for agent memory system."""
    dim: int = 256  # Embedding dimension
    top_k: int = 4  # Number of memories to retrieve
    max_memories: int = 1000  # Maximum stored memories
    similarity_threshold: float = 0.7  # Minimum similarity for retrieval


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    llm: LLMConfig
    memory: MemoryConfig
    max_steps: int = 12
    verbose: bool = True
    log_path: Optional[str] = None


class SimpleMemoryStore:
    """Simple vector memory store for agent experiences."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memories: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

    def add(self, text: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """Store a memory with its embedding."""
        if len(self.memories) >= self.config.max_memories:
            # Remove oldest memory
            self.memories.pop(0)
            self.embeddings.pop(0)

        self.memories.append({
            "text": text,
            "metadata": metadata or {},
            "timestamp": len(self.memories)
        })
        self.embeddings.append(embedding)

    def retrieve(self, query_embedding: np.ndarray, k: Optional[int] = None) -> List[Dict]:
        """Retrieve top-k most similar memories."""
        if not self.embeddings:
            return []

        k = k or self.config.top_k

        # Compute cosine similarities
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append(sim)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Return memories above threshold
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.config.similarity_threshold:
                results.append({
                    **self.memories[idx],
                    "similarity": similarities[idx]
                })

        return results

    def clear(self):
        """Clear all memories."""
        self.memories.clear()
        self.embeddings.clear()


class LLMClient:
    """Unified client for different LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.config.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
                print(f"[LLM] Initialized OpenAI client with model {self.config.model}")
            except ImportError:
                print("[LLM] Warning: openai package not installed. Install with: pip install openai")

        elif self.config.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.config.api_key)
                print(f"[LLM] Initialized Anthropic client with model {self.config.model}")
            except ImportError:
                print("[LLM] Warning: anthropic package not installed. Install with: pip install anthropic")

        else:
            print(f"[LLM] Warning: Unknown provider '{self.config.provider}'. Using mock client.")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the LLM."""
        if not self.client:
            return self._mock_generate(messages)

        try:
            if self.config.provider == "openai":
                return self._openai_generate(messages)
            elif self.config.provider == "anthropic":
                return self._anthropic_generate(messages)
            else:
                return self._mock_generate(messages)
        except Exception as e:
            print(f"[LLM] Error generating response: {e}")
            return self._mock_generate(messages)

    def _openai_generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

    def _anthropic_generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate using Anthropic API."""
        # Extract system message if present
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        kwargs = {
            "model": self.config.model,
            "messages": filtered_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def _mock_generate(self, messages: List[Dict[str, str]]) -> str:
        """Mock generation for testing without API."""
        last_message = messages[-1]["content"] if messages else ""
        return f"[MOCK RESPONSE] Analyzed: '{last_message[:50]}...' → Next step: Continue reasoning."

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text. Uses simple hash-based embedding if API unavailable."""
        try:
            if self.config.provider == "openai" and self.client:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"[LLM] Could not get embedding from API: {e}, using fallback")

        # Fallback: simple hash-based embedding
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str, dim: int = 256) -> np.ndarray:
        """Create a deterministic embedding from text hash."""
        # Simple but deterministic embedding for testing
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(dim)
        return embedding / np.linalg.norm(embedding)


class ProtoAGIAgent:
    """
    A prototype AGI agent that uses LLM-based reasoning with memory.

    The agent follows a step-based reasoning loop:
    1. Understand the goal
    2. Retrieve relevant memories
    3. Plan next action
    4. Execute reasoning step
    5. Store results in memory
    6. Repeat until goal is achieved or max steps reached
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.memory = SimpleMemoryStore(config.memory)
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

    def run(self, goal: str) -> str:
        """Execute the agent to achieve the given goal."""
        self._log(f"\n{'='*60}")
        self._log(f"ProtoAGI Agent started")
        self._log(f"Goal: {goal}")
        self._log(f"{'='*60}\n")

        # Initialize conversation
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Goal: {goal}\n\nBegin reasoning step-by-step to achieve this goal."}
        ]

        result = ""
        for step in range(self.config.max_steps):
            self.step_count = step + 1
            self._log(f"\n--- Step {self.step_count}/{self.config.max_steps} ---")

            # Retrieve relevant memories
            if step > 0:
                query_emb = self.llm.get_embedding(goal)
                memories = self.memory.retrieve(query_emb)
                if memories:
                    memory_context = "\n".join([f"- {m['text']}" for m in memories[:3]])
                    messages.append({
                        "role": "system",
                        "content": f"Relevant memories:\n{memory_context}"
                    })

            # Generate next reasoning step
            response = self.llm.generate(messages)
            self._log(f"Response: {response[:200]}...")

            # Store in memory
            response_emb = self.llm.get_embedding(response)
            self.memory.add(
                text=response[:500],  # Store first 500 chars
                embedding=response_emb,
                metadata={"step": step, "goal": goal}
            )

            # Add to conversation
            messages.append({"role": "assistant", "content": response})

            # Track history
            self.history.append({
                "step": self.step_count,
                "response": response
            })

            # Check for completion
            if self._is_complete(response, goal):
                self._log("\n[COMPLETE] Goal appears to be achieved!")
                result = response
                break

            # Prepare next step
            messages.append({
                "role": "user",
                "content": "Continue to the next reasoning step."
            })

            result = response

        self._log(f"\n{'='*60}")
        self._log(f"Agent completed after {self.step_count} steps")
        self._log(f"{'='*60}\n")

        return result

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are a ProtoAGI reasoning agent. Your task is to:

1. Break down complex goals into step-by-step reasoning
2. Think through each step carefully and logically
3. Build upon previous steps to make progress
4. Provide concrete, actionable insights
5. Conclude when the goal is sufficiently addressed

Be concise but thorough. Focus on quality reasoning over quantity."""

    def _is_complete(self, response: str, goal: str) -> bool:
        """Check if the response indicates completion."""
        completion_indicators = [
            "complete", "finished", "achieved", "accomplished",
            "final plan", "final step", "conclusion"
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in completion_indicators)

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)

        if self.config.log_path:
            with open(self.config.log_path, 'a') as f:
                f.write(message + '\n')

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full history of reasoning steps."""
        return self.history

    def reset(self):
        """Reset the agent state."""
        self.step_count = 0
        self.history.clear()
        self.memory.clear()


def build_agent() -> ProtoAGIAgent:
    """Build a ProtoAGI agent with default configuration."""
    llm_cfg = LLMConfig(
        provider="openai",          # or "anthropic", etc.
        model="gpt-4.1-mini",       # change to your model
        api_key=os.getenv("LLM_API_KEY", "FAKE_KEY"),
        temperature=0.2,
        max_tokens=1024,
    )
    mem_cfg = MemoryConfig(dim=256, top_k=4)
    agent_cfg = AgentConfig(llm=llm_cfg, memory=mem_cfg, max_steps=12)
    return ProtoAGIAgent(agent_cfg)


if __name__ == "__main__":
    # Example usage
    agent = build_agent()
    goal = "Design a plan to learn quantum computing in 3 months."
    result = agent.run(goal)
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(result)
