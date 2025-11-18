#!/usr/bin/env python3
# ============================================
# Test Suite for Enhanced Qiskit Quantum Computing Module
# ============================================

import unittest
import numpy as np
from quantum_qiskit_enhanced import EnhancedQiskitQuantumProcessor, QuantumResult


class TestEnhancedQiskitQuantumProcessor(unittest.TestCase):
    """
    Comprehensive test suite for the Enhanced Qiskit Quantum Processor
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.qp = EnhancedQiskitQuantumProcessor(num_qubits=8, shots=1000)
        print("\n" + "=" * 80)
        print("🧪 QUANTUM COMPUTING MODULE TEST SUITE")
        print("=" * 80)

    def test_01_bell_state_phi_plus(self):
        """Test Bell state Φ+ creation"""
        print("\n[Test 1] Bell State Φ+ (|00⟩ + |11⟩)/√2")
        result = self.qp.create_bell_state("phi_plus")

        # Verify result structure
        self.assertIsInstance(result, QuantumResult)
        self.assertEqual(result.circuit_name, "Bell State (phi_plus)")
        self.assertIsNotNone(result.statevector)
        self.assertIsNotNone(result.measurement_counts)

        # Check statevector
        sv = result.statevector
        self.assertAlmostEqual(abs(sv[0]) ** 2, 0.5, places=2)  # |00⟩ probability
        self.assertAlmostEqual(abs(sv[3]) ** 2, 0.5, places=2)  # |11⟩ probability

        # Check measurements show only |00⟩ and |11⟩
        counts = result.measurement_counts
        total_counts = sum(counts.values())
        self.assertEqual(total_counts, 1000)

        # Should have roughly equal |00⟩ and |11⟩ (within statistical variance)
        expected_outcomes = {"00", "11"}
        measured_outcomes = set(counts.keys())
        self.assertEqual(measured_outcomes, expected_outcomes)

        print(f"  ✅ Statevector: {sv}")
        print(f"  ✅ Measurements: {counts}")
        print(f"  ✅ Test passed!")

    def test_02_bell_states_all_types(self):
        """Test all 4 Bell state types"""
        print("\n[Test 2] All Bell State Types")
        bell_types = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]

        for bell_type in bell_types:
            result = self.qp.create_bell_state(bell_type)
            self.assertIsInstance(result, QuantumResult)
            self.assertEqual(result.fidelity, 1.0)
            print(f"  ✅ {bell_type}: {result.measurement_counts}")

        print(f"  ✅ All Bell states created successfully!")

    def test_03_ghz_state(self):
        """Test GHZ state creation"""
        print("\n[Test 3] GHZ State (Multi-qubit Entanglement)")
        result = self.qp.create_ghz_state(num_qubits=3)

        # Check statevector for GHZ state: (|000⟩ + |111⟩)/√2
        sv = result.statevector
        self.assertAlmostEqual(abs(sv[0]) ** 2, 0.5, places=2)  # |000⟩
        self.assertAlmostEqual(abs(sv[7]) ** 2, 0.5, places=2)  # |111⟩

        # Verify measurements
        counts = result.measurement_counts
        expected_outcomes = {"000", "111"}
        measured_outcomes = set(counts.keys())
        self.assertEqual(measured_outcomes, expected_outcomes)

        print(f"  ✅ Statevector amplitudes correct")
        print(f"  ✅ Measurements: {counts}")
        print(f"  ✅ Test passed!")

    def test_04_quantum_teleportation(self):
        """Test quantum teleportation protocol"""
        print("\n[Test 4] Quantum Teleportation Protocol")
        result = self.qp.quantum_teleportation(theta=np.pi / 4, phi=0)

        self.assertIsInstance(result, QuantumResult)
        self.assertEqual(result.metadata["protocol"], "quantum_teleportation")
        self.assertEqual(result.metadata["num_qubits"], 3)

        # Verify measurements exist
        counts = result.measurement_counts
        self.assertGreater(len(counts), 0)

        print(f"  ✅ Protocol executed successfully")
        print(f"  ✅ Measurement outcomes: {len(counts)} states observed")
        print(f"  ✅ Test passed!")

    def test_05_grovers_search(self):
        """Test Grover's search algorithm"""
        print("\n[Test 5] Grover's Search Algorithm")
        marked_item = 5
        result = self.qp.grovers_search(marked_item=marked_item, num_qubits=3)

        # Verify algorithm ran correctly
        self.assertEqual(result.metadata["marked_item"], marked_item)
        self.assertEqual(result.metadata["marked_binary"], "101")

        # Check success probability (should be high, >80%)
        success_prob = result.metadata["success_probability"]
        self.assertGreater(success_prob, 0.80)

        # Verify marked item was found most frequently
        counts = result.measurement_counts
        max_count_state = max(counts, key=counts.get)
        self.assertEqual(max_count_state, "101")

        print(f"  ✅ Success probability: {success_prob:.2%}")
        print(f"  ✅ Marked item found: {max_count_state}")
        print(f"  ✅ Test passed!")

    def test_06_quantum_fourier_transform(self):
        """Test Quantum Fourier Transform"""
        print("\n[Test 6] Quantum Fourier Transform")
        result = self.qp.quantum_fourier_transform(input_state=[1, 0, 1, 0])

        self.assertEqual(result.metadata["algorithm"], "qft")
        self.assertEqual(result.metadata["num_qubits"], 4)
        self.assertIsNotNone(result.statevector)

        # Verify QFT properties: all states should have equal probability
        sv = result.statevector
        state_size = len(sv)

        # Each amplitude should have roughly equal magnitude
        expected_prob = 1.0 / state_size
        for amp in sv:
            prob = abs(amp) ** 2
            self.assertAlmostEqual(prob, expected_prob, places=2)

        print(f"  ✅ QFT applied successfully")
        print(f"  ✅ Equal superposition verified")
        print(f"  ✅ Test passed!")

    def test_07_quantum_error_correction(self):
        """Test quantum error correction"""
        print("\n[Test 7] Quantum Error Correction")
        result = self.qp.quantum_error_correction(error_qubit=1)

        self.assertEqual(result.metadata["code"], "3-qubit bit flip")
        self.assertTrue(result.metadata["error_introduced"])
        self.assertEqual(result.metadata["error_qubit"], 1)

        # Verify measurements
        counts = result.measurement_counts
        self.assertGreater(len(counts), 0)

        print(f"  ✅ Error correction protocol executed")
        print(f"  ✅ Measurements: {counts}")
        print(f"  ✅ Test passed!")

    def test_08_quantum_supremacy_demo(self):
        """Test quantum supremacy demonstration"""
        print("\n[Test 8] Quantum Supremacy Demonstration")
        result = self.qp.quantum_supremacy_demo(num_qubits=6, depth=5)

        self.assertEqual(result.metadata["num_qubits"], 6)
        self.assertEqual(result.metadata["depth"], 5)
        self.assertEqual(result.metadata["hilbert_space_size"], 2**6)

        # Verify many different outcomes (hard to simulate classically)
        counts = result.measurement_counts
        unique_states = len(counts)
        self.assertGreater(unique_states, 10)  # Should see many different states

        print(f"  ✅ Random circuit executed")
        print(f"  ✅ Unique states observed: {unique_states}")
        print(f"  ✅ Hilbert space size: {result.metadata['hilbert_space_size']}")
        print(f"  ✅ Test passed!")

    def test_09_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        print("\n[Test 9] Statistics Tracking")
        stats = self.qp.get_statistics()

        self.assertIn("total_experiments", stats)
        self.assertIn("total_execution_time", stats)
        self.assertIn("average_execution_time", stats)

        # Should have run at least the tests above
        self.assertGreaterEqual(stats["total_experiments"], 8)

        print(f"  ✅ Total experiments: {stats['total_experiments']}")
        print(f"  ✅ Total time: {stats['total_execution_time']:.4f}s")
        print(f"  ✅ Average time: {stats['average_execution_time']:.4f}s")
        print(f"  ✅ Test passed!")

    def test_10_result_history(self):
        """Test that results are stored in history"""
        print("\n[Test 10] Result History")

        initial_count = len(self.qp.results_history)

        # Run a new experiment
        self.qp.create_bell_state("phi_plus")

        # Check history increased
        new_count = len(self.qp.results_history)
        self.assertEqual(new_count, initial_count + 1)

        # Verify last result
        last_result = self.qp.results_history[-1]
        self.assertIsInstance(last_result, QuantumResult)

        print(f"  ✅ History size: {new_count}")
        print(f"  ✅ Last result: {last_result.circuit_name}")
        print(f"  ✅ Test passed!")

    def test_11_measurement_probabilities(self):
        """Test that measurement probabilities sum to 1"""
        print("\n[Test 11] Measurement Probability Conservation")
        result = self.qp.create_bell_state("phi_plus")

        counts = result.measurement_counts
        total = sum(counts.values())

        # Should equal the number of shots
        self.assertEqual(total, 1000)

        # Probabilities should sum to 1
        probs = [count / total for count in counts.values()]
        prob_sum = sum(probs)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

        print(f"  ✅ Total measurements: {total}")
        print(f"  ✅ Probability sum: {prob_sum}")
        print(f"  ✅ Test passed!")

    def test_12_statevector_normalization(self):
        """Test that statevectors are properly normalized"""
        print("\n[Test 12] Statevector Normalization")
        result = self.qp.create_ghz_state(num_qubits=4)

        sv = result.statevector
        norm_squared = sum(abs(amp) ** 2 for amp in sv)

        # Should be normalized to 1
        self.assertAlmostEqual(norm_squared, 1.0, places=5)

        print(f"  ✅ Statevector norm: {np.sqrt(norm_squared)}")
        print(f"  ✅ Normalization verified")
        print(f"  ✅ Test passed!")


class TestQuantumIntegration(unittest.TestCase):
    """Test the integration with Nexus AGI"""

    def test_integration_import(self):
        """Test that integration module can be imported"""
        print("\n[Integration Test] Module Import")
        try:
            from nexus_qiskit_integration import NexusQiskitIntegration

            print("  ✅ Integration module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import integration module: {e}")


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedQiskitQuantumProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✨ ALL TESTS PASSED! ✨")
    else:
        print("\n❌ SOME TESTS FAILED")

    print("=" * 80 + "\n")

    return result


if __name__ == "__main__":
    run_tests()
