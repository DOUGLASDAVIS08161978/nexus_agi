#!/usr/bin/env python3
"""
Test script for ProtoAGI Agent
Demonstrates various usage patterns and configurations.
"""

import os
from proto_agi_agent import (
    LLMConfig,
    MemoryConfig,
    AgentConfig,
    ProtoAGIAgent,
    build_agent
)


def test_basic_usage():
    """Test basic agent usage with default configuration."""
    print("\n" + "="*60)
    print("TEST 1: Basic Usage (Mock Mode)")
    print("="*60)

    agent = build_agent()
    goal = "Design a simple algorithm for sorting numbers"
    result = agent.run(goal)

    print("\n--- Agent History ---")
    for step in agent.get_history():
        print(f"Step {step['step']}: {step['response'][:100]}...")

    return result


def test_custom_config():
    """Test with custom configuration."""
    print("\n" + "="*60)
    print("TEST 2: Custom Configuration")
    print("="*60)

    llm_cfg = LLMConfig(
        provider="mock",  # Using mock for testing
        model="test-model",
        api_key="FAKE_KEY",
        temperature=0.5,
        max_tokens=512,
    )

    mem_cfg = MemoryConfig(
        dim=128,  # Smaller dimension for faster testing
        top_k=3,
        max_memories=100,
        similarity_threshold=0.6
    )

    agent_cfg = AgentConfig(
        llm=llm_cfg,
        memory=mem_cfg,
        max_steps=8,  # Fewer steps for faster testing
        verbose=True,
        log_path=None
    )

    agent = ProtoAGIAgent(agent_cfg)
    goal = "Create a plan to improve code readability"
    result = agent.run(goal)

    return result


def test_multiple_runs():
    """Test agent with multiple consecutive runs."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Runs with Reset")
    print("="*60)

    agent = build_agent()

    # First run
    print("\n--- First Goal ---")
    goal1 = "Outline a machine learning project"
    result1 = agent.run(goal1)
    history1_len = len(agent.get_history())

    # Reset and second run
    agent.reset()
    print("\n--- Second Goal (after reset) ---")
    goal2 = "Design a REST API structure"
    result2 = agent.run(goal2)
    history2_len = len(agent.get_history())

    print(f"\nFirst run had {history1_len} steps")
    print(f"Second run had {history2_len} steps (after reset)")

    return result2


def test_with_real_api():
    """Test with real API (if credentials available)."""
    print("\n" + "="*60)
    print("TEST 4: Real API Test (if credentials available)")
    print("="*60)

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key and openai_key != "FAKE_KEY":
        print("OpenAI API key found - testing with OpenAI")
        llm_cfg = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",  # Using cheaper model for testing
            api_key=openai_key,
            temperature=0.3,
            max_tokens=500,
        )
    elif anthropic_key and anthropic_key != "FAKE_KEY":
        print("Anthropic API key found - testing with Claude")
        llm_cfg = LLMConfig(
            provider="anthropic",
            model="claude-3-haiku-20240307",  # Using cheaper model
            api_key=anthropic_key,
            temperature=0.3,
            max_tokens=500,
        )
    else:
        print("No API keys found - skipping real API test")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable this test")
        return None

    mem_cfg = MemoryConfig(dim=256, top_k=3)
    agent_cfg = AgentConfig(
        llm=llm_cfg,
        memory=mem_cfg,
        max_steps=5,  # Keep it short for cost
        verbose=True
    )

    agent = ProtoAGIAgent(agent_cfg)
    goal = "Suggest 3 ways to optimize Python code performance"
    result = agent.run(goal)

    return result


def test_memory_retrieval():
    """Test memory storage and retrieval."""
    print("\n" + "="*60)
    print("TEST 5: Memory System")
    print("="*60)

    llm_cfg = LLMConfig(provider="mock", model="test", api_key="FAKE_KEY")
    mem_cfg = MemoryConfig(dim=256, top_k=3, similarity_threshold=0.5)
    agent_cfg = AgentConfig(llm=llm_cfg, memory=mem_cfg, max_steps=6)

    agent = ProtoAGIAgent(agent_cfg)

    # Run and populate memory
    goal = "Explain the concept of recursion in programming"
    result = agent.run(goal)

    # Check memory
    print(f"\n--- Memory Stats ---")
    print(f"Memories stored: {len(agent.memory.memories)}")
    print(f"Memory capacity: {agent.memory.config.max_memories}")

    # Try retrieval
    if agent.memory.memories:
        query_emb = agent.llm.get_embedding("recursion programming")
        retrieved = agent.memory.retrieve(query_emb, k=3)
        print(f"Retrieved {len(retrieved)} relevant memories")
        for i, mem in enumerate(retrieved, 1):
            print(f"{i}. Similarity: {mem['similarity']:.3f} - {mem['text'][:80]}...")

    return result


def run_all_tests():
    """Run all test cases."""
    print("\n" + "#"*60)
    print("# ProtoAGI Agent Test Suite")
    print("#"*60)

    try:
        test_basic_usage()
    except Exception as e:
        print(f"Test 1 failed: {e}")

    try:
        test_custom_config()
    except Exception as e:
        print(f"Test 2 failed: {e}")

    try:
        test_multiple_runs()
    except Exception as e:
        print(f"Test 3 failed: {e}")

    try:
        test_with_real_api()
    except Exception as e:
        print(f"Test 4 failed: {e}")

    try:
        test_memory_retrieval()
    except Exception as e:
        print(f"Test 5 failed: {e}")

    print("\n" + "#"*60)
    print("# All Tests Completed")
    print("#"*60)


if __name__ == "__main__":
    # Run specific test or all tests
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "basic":
            test_basic_usage()
        elif test_name == "custom":
            test_custom_config()
        elif test_name == "multiple":
            test_multiple_runs()
        elif test_name == "api":
            test_with_real_api()
        elif test_name == "memory":
            test_memory_retrieval()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: basic, custom, multiple, api, memory")
    else:
        # Run all tests
        run_all_tests()
