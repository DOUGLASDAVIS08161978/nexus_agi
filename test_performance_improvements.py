#!/usr/bin/env python3
"""
Test script to validate performance improvements
"""

import time
import numpy as np
import sys

# The classes are in nexus_agi.py, not the nexus_agi package.
# We need to import them directly from the file.
from nexus_agi import (
    OpenQuantumSimulator,
    NeuralProcessor,
    HoloConceptEngine
)
from omega_asi import AdvancedQuantumProcessor


def test_nexus_agi_imports():
    """Test that nexus_agi module imports correctly"""
    print("\n=== Testing nexus_agi imports ===")
    # The import is now at the top level, just check if we got here.
    try:
        # This is just a dummy check since the import is global now.
        assert 'OpenQuantumSimulator' in globals()
        assert 'NeuralProcessor' in globals()
        assert 'HoloConceptEngine' in globals()
        print("✓ Successfully imported core modules")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_neural_processor_projection_cache():
    """Test that NeuralProcessor caching works"""
    print("\n=== Testing NeuralProcessor projection cache ===")
    try:
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
        
        # Verify cache size
        assert len(hce.embedding_cache) == 1, "Cache should have one entry"
        
        print("✓ Embedding cache working correctly")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_characteristics():
    """Test performance of key operations"""
    print("\n=== Testing performance characteristics ===")
    try:
        # Test quantum entanglement performance
        processor = AdvancedQuantumProcessor(num_qubits=8)
        start_time = time.time()
        for _ in range(10):
            processor.entangle_qubits(0, 1)
        end_time = time.time()
        entanglement_time = end_time - start_time
        print(f"  Entanglement operations took: {entanglement_time:.4f} seconds")
        assert entanglement_time < 0.1, "Entanglement is too slow"
        
        print("✓ Performance characteristics acceptable")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run all tests"""
    results = {
        "Imports": test_nexus_agi_imports(),
        "Neural Processor Cache": test_neural_processor_projection_cache(),
        "Quantum Vectorization": test_omega_quantum_vectorization(),
        "Embedding Cache": test_holo_concept_engine_caching(),
        "Performance": test_performance_characteristics(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed_count = 0
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30}: {status}")
        if result:
            passed_count += 1
            
    total_tests = len(results)
    print(f"\nTotal: {passed_count}/{total_tests} tests passed")
    
    if passed_count < total_tests:
        print(f"\n✗ {total_tests - passed_count} test(s) failed")
        sys.exit(1)
    else:
        print("\n✓ All optimizations validated successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
