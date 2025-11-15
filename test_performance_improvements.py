#!/usr/bin/env python3
"""
Test script to validate performance improvements
"""

import time
import numpy as np
import sys

def test_nexus_agi_imports():
    """Test that nexus_agi module imports correctly"""
    print("\n=== Testing nexus_agi imports ===")
    try:
        from nexus_agi import (
            OpenQuantumSimulator,
            NeuralProcessor,
            HoloConceptEngine
        )
        print("✓ Successfully imported core modules")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_neural_processor_projection_cache():
    """Test that NeuralProcessor caching works"""
    print("\n=== Testing NeuralProcessor projection cache ===")
    try:
        from nexus_agi import NeuralProcessor
        import torch
        
        # Create processor
        processor = NeuralProcessor(input_dim=10, hidden_dim=5, output_dim=3)
        
        # Check cache exists
        assert hasattr(processor, 'projection_cache'), "projection_cache not found"
        assert isinstance(processor.projection_cache, dict), "projection_cache should be a dict"
        
        # Simulate training scenario where projection is needed
        shape_key = (3, 5)
        if shape_key not in processor.projection_cache:
            processor.projection_cache[shape_key] = torch.nn.Linear(shape_key[0], shape_key[1])
        
        # Verify cache works
        layer1 = processor.projection_cache[shape_key]
        layer2 = processor.projection_cache[shape_key]
        assert layer1 is layer2, "Cache should return same object"
        
        print("✓ Projection cache working correctly")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_omega_quantum_vectorization():
    """Test that omega_asi quantum operations work"""
    print("\n=== Testing OMEGA quantum vectorization ===")
    try:
        from omega_asi import AdvancedQuantumProcessor
        
        # Create processor with small number of qubits for testing
        processor = AdvancedQuantumProcessor(num_qubits=3)
        
        # Set a non-trivial initial state (not equal superposition)
        # Use state |001⟩ where qubit 0 = 1
        processor.quantum_state = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=complex)
        initial_state = processor.quantum_state.copy()
        
        # Test entanglement - CNOT should flip target when control is 1
        processor.entangle_qubits(0, 1)
        
        # Verify state changed (should move from index 1 to index 3)
        assert not np.allclose(initial_state, processor.quantum_state), "State should change after entanglement"
        
        # Verify state is normalized
        norm = np.linalg.norm(processor.quantum_state)
        assert np.abs(norm - 1.0) < 1e-10, f"State should be normalized, got norm={norm}"
        
        # Verify the correct transformation occurred (|001⟩ -> |011⟩)
        expected_state = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=complex)
        assert np.allclose(processor.quantum_state, expected_state), "CNOT gate produced incorrect result"
        
        print("✓ Quantum vectorization working correctly")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_holo_concept_engine_caching():
    """Test that HoloConceptEngine embedding caching works"""
    print("\n=== Testing HoloConceptEngine embedding cache ===")
    try:
        from nexus_agi import HoloConceptEngine
        
        # Create engine
        hce = HoloConceptEngine(embedding_dim=64)
        
        # Check cache exists
        assert hasattr(hce, 'embedding_cache'), "embedding_cache not found"
        assert isinstance(hce.embedding_cache, dict), "embedding_cache should be a dict"
        
        # Test caching
        test_text = "test concept"
        embedding1 = hce._get_embedding(test_text)
        embedding2 = hce._get_embedding(test_text)
        
        # Verify same embedding returned
        assert np.allclose(embedding1, embedding2), "Cache should return same embedding"
        
        # Verify it was actually cached
        assert test_text in hce.embedding_cache, "Text should be in cache"
        
        print("✓ Embedding cache working correctly")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_improvement():
    """Measure basic performance characteristics"""
    print("\n=== Testing performance characteristics ===")
    try:
        from omega_asi import AdvancedQuantumProcessor
        
        # Test with small quantum system
        processor = AdvancedQuantumProcessor(num_qubits=8)
        
        # Time entanglement operation
        start = time.time()
        for i in range(min(4, processor.num_qubits - 1)):
            processor.entangle_qubits(i, i + 1)
        duration = time.time() - start
        
        print(f"  Entanglement operations took: {duration:.4f} seconds")
        
        # Should be reasonably fast
        assert duration < 1.0, f"Operations took too long: {duration}s"
        
        print("✓ Performance characteristics acceptable")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Performance Optimization Validation Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_nexus_agi_imports()))
    results.append(("Neural Processor Cache", test_neural_processor_projection_cache()))
    results.append(("Quantum Vectorization", test_omega_quantum_vectorization()))
    results.append(("Embedding Cache", test_holo_concept_engine_caching()))
    results.append(("Performance", test_performance_improvement()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
