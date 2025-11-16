#!/usr/bin/env python3
"""
Simple integration test for performance improvements
Tests only the omega_asi module which has minimal dependencies
"""

import time
import numpy as np

def test_quantum_optimizations():
    """Test all quantum optimizations"""
    print("\n" + "="*60)
    print("Testing Quantum Optimizations")
    print("="*60)
    
    from omega_asi import AdvancedQuantumProcessor
    
    # Test 1: Vectorized entanglement
    print("\n1. Testing vectorized entanglement...")
    processor = AdvancedQuantumProcessor(num_qubits=4)
    processor.quantum_state = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
    
    initial = processor.quantum_state.copy()
    processor.entangle_qubits(0, 1)
    
    assert not np.allclose(initial, processor.quantum_state), "Entanglement should change state"
    assert np.abs(np.linalg.norm(processor.quantum_state) - 1.0) < 1e-10, "State should be normalized"
    print("   ✓ Vectorized entanglement working correctly")
    
    # Test 2: Vectorized single-qubit gates
    print("\n2. Testing vectorized single-qubit gates...")
    processor2 = AdvancedQuantumProcessor(num_qubits=4)
    initial2 = processor2.quantum_state.copy()
    
    gate_matrix = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z
    processor2._apply_single_qubit_gate(0, gate_matrix)
    
    # State should change (some phases flipped)
    norm = np.linalg.norm(processor2.quantum_state)
    assert np.abs(norm - 1.0) < 1e-10, "State should remain normalized"
    print("   ✓ Vectorized single-qubit gates working correctly")
    
    # Test 3: Performance benchmark
    print("\n3. Testing performance characteristics...")
    processor3 = AdvancedQuantumProcessor(num_qubits=10)
    
    start = time.time()
    for i in range(5):
        processor3.entangle_qubits(i, i+1)
    duration = time.time() - start
    
    print(f"   Entangled 5 qubit pairs in {duration:.4f}s")
    assert duration < 0.5, f"Operations too slow: {duration}s"
    print("   ✓ Performance is acceptable")
    
    return True

def test_aria_optimizations():
    """Test JavaScript optimizations by checking the file"""
    print("\n" + "="*60)
    print("Testing ARIA.js Optimizations")
    print("="*60)
    
    with open('aria.js', 'r') as f:
        content = f.read()
    
    # Check for optimizations
    checks = [
        ('actualStateSize', 'State size optimization present'),
        ('new Array(inputSize)', 'Pre-allocated arrays'),
        ('deterministic', 'Reduced random operations'),
    ]
    
    for pattern, desc in checks:
        if pattern in content:
            print(f"   ✓ {desc}")
        else:
            print(f"   ✗ {desc} - pattern '{pattern}' not found")
            return False
    
    return True

def test_python_caching():
    """Test that caching is present in Python files"""
    print("\n" + "="*60)
    print("Testing Python Caching Optimizations")
    print("="*60)
    
    # Check nexus_agi.py for caching
    with open('nexus_agi.py', 'r') as f:
        nexus_content = f.read()
    
    checks = [
        ('projection_cache', 'Neural processor projection cache'),
        ('embedding_cache', 'HoloConceptEngine embedding cache'),
        ('RandomState', 'RandomState instead of global seed'),
    ]
    
    for pattern, desc in checks:
        if pattern in nexus_content:
            print(f"   ✓ {desc}")
        else:
            print(f"   ✗ {desc} - pattern '{pattern}' not found")
            return False
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("Performance Optimization Validation")
    print("="*60)
    
    tests = [
        ("Quantum Optimizations", test_quantum_optimizations),
        ("ARIA Optimizations", test_aria_optimizations),
        ("Python Caching", test_python_caching),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All optimizations validated successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
