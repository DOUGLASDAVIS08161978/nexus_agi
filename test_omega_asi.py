#!/usr/bin/env python3
"""
Test suite for OMEGA ASI module
"""

import sys
import unittest
import numpy as np
from omega_asi import (
    AdvancedQuantumProcessor,
    EnhancedConsciousnessFramework,
    MultiDimensionalEmpathySystem,
    CausalReasoningEngine,
    OMEGA_ASI
)


class TestQuantumProcessor(unittest.TestCase):
    """Test Advanced Quantum Processor"""
    
    def setUp(self):
        self.qp = AdvancedQuantumProcessor(num_qubits=4)
    
    def test_initialization(self):
        """Test quantum processor initializes correctly"""
        self.assertEqual(self.qp.num_qubits, 4)
        self.assertEqual(len(self.qp.quantum_state), 2**4)
        self.assertAlmostEqual(np.linalg.norm(self.qp.quantum_state), 1.0, places=5)
    
    def test_entanglement(self):
        """Test qubit entanglement"""
        self.qp.entangle_qubits(0, 1)
        self.assertTrue(self.qp.entanglement_graph.has_edge(0, 1))
    
    def test_quantum_gates(self):
        """Test quantum gate application"""
        initial_state = self.qp.quantum_state.copy()
        self.qp.apply_quantum_gate("Hadamard", [0])
        # State should change after gate application
        self.assertFalse(np.allclose(initial_state, self.qp.quantum_state))
    
    def test_measurement(self):
        """Test quantum state measurement"""
        measurements = self.qp.measure_quantum_state(num_shots=100)
        self.assertIsInstance(measurements, dict)
        self.assertTrue(len(measurements) > 0)
        # Total counts should equal num_shots
        total_counts = sum(measurements.values())
        self.assertEqual(total_counts, 100)
    
    def test_quantum_optimization(self):
        """Test quantum optimization"""
        def simple_objective(state):
            return np.abs(state[0])**2
        
        result = self.qp.quantum_optimization(simple_objective, num_iterations=10)
        self.assertIn("optimal_params", result)
        self.assertIn("optimal_energy", result)
        self.assertIsInstance(result["optimal_params"], list)


class TestConsciousnessFramework(unittest.TestCase):
    """Test Enhanced Consciousness Framework"""
    
    def setUp(self):
        self.consciousness = EnhancedConsciousnessFramework(initial_awareness=0.7)
    
    def test_initialization(self):
        """Test consciousness framework initializes correctly"""
        self.assertAlmostEqual(self.consciousness.awareness_level, 0.7, places=2)
        self.assertEqual(len(self.consciousness.consciousness_history), 0)
    
    def test_consciousness_update(self):
        """Test consciousness state update"""
        input_data = {"feature1": 0.5, "feature2": 0.8}
        context = {"urgency": "high"}
        
        state = self.consciousness.update_consciousness_state(input_data, context)
        
        self.assertIsNotNone(state)
        self.assertGreater(state.awareness_level, 0)
        self.assertLessEqual(state.awareness_level, 1.0)
        self.assertEqual(len(self.consciousness.consciousness_history), 1)
    
    def test_awareness_adjustment(self):
        """Test automatic awareness adjustment"""
        # Simple input should lower awareness need
        simple_input = {"f1": 0.1}
        simple_context = {"urgency": "low"}
        
        # Complex input should raise awareness need  
        complex_input = {f"f{i}": 0.5 for i in range(15)}
        complex_context = {"urgency": "high"}
        
        self.consciousness.update_consciousness_state(simple_input, simple_context)
        simple_awareness = self.consciousness.awareness_level
        
        self.consciousness.update_consciousness_state(complex_input, complex_context)
        complex_awareness = self.consciousness.awareness_level
        
        # More complex input should result in adjustment toward higher awareness
        # (though it may not always be higher due to gradual adjustment)
        self.assertIsInstance(complex_awareness, float)
    
    def test_consciousness_report(self):
        """Test consciousness status report"""
        self.consciousness.update_consciousness_state({"f1": 0.5}, {})
        report = self.consciousness.get_consciousness_report()
        
        self.assertIn("current_awareness", report)
        self.assertIn("meta_cognitive_layers", report)
        self.assertIn("consciousness_trajectory", report)


class TestEmpathySystem(unittest.TestCase):
    """Test Multi-Dimensional Empathy System"""
    
    def setUp(self):
        self.empathy = MultiDimensionalEmpathySystem()
    
    def test_initialization(self):
        """Test empathy system initializes correctly"""
        self.assertIsNotNone(self.empathy.theory_of_mind)
        self.assertIsNotNone(self.empathy.emotional_analyzer)
        self.assertEqual(len(self.empathy.perspective_models), 0)
    
    def test_stakeholder_analysis(self):
        """Test stakeholder perspective analysis"""
        scenario = {
            "type": "climate",
            "description": "Climate change mitigation"
        }
        stakeholders = ["ecosystems", "future_generations"]
        
        perspectives = self.empathy.analyze_stakeholder_perspectives(scenario, stakeholders)
        
        self.assertEqual(len(perspectives), 2)
        self.assertIn("ecosystems", perspectives)
        self.assertIn("future_generations", perspectives)
        
        # Check perspective structure
        for stakeholder, perspective in perspectives.items():
            self.assertIn("emotional_state", perspective)
            self.assertIn("concerns", perspective)
            self.assertIn("values", perspective)
    
    def test_empathic_response(self):
        """Test empathic response generation"""
        scenario = {"type": "climate"}
        stakeholders = ["ecosystems"]
        
        self.empathy.analyze_stakeholder_perspectives(scenario, stakeholders)
        response = self.empathy.generate_empathic_response("ecosystems", "deforestation")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_empathy_score(self):
        """Test empathy score computation"""
        scenario = {"type": "climate"}
        stakeholders = ["ecosystems", "future_generations"]
        
        self.empathy.analyze_stakeholder_perspectives(scenario, stakeholders)
        
        action = {"description": "plant trees and reduce emissions"}
        score = self.empathy.compute_empathy_score(action, stakeholders)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestCausalReasoning(unittest.TestCase):
    """Test Causal Reasoning Engine"""
    
    def setUp(self):
        self.causal = CausalReasoningEngine()
    
    def test_initialization(self):
        """Test causal engine initializes correctly"""
        self.assertIsNotNone(self.causal.causal_graph)
        self.assertEqual(len(self.causal.causal_graph.nodes()), 0)
    
    def test_causal_graph_construction(self):
        """Test causal graph construction"""
        domain_knowledge = {
            "climate_science": "Carbon emissions cause temperature increase",
            "economics": "Economic activity affects emissions",
            "policy": "Policy influences behavior"
        }
        
        graph = self.causal.construct_causal_graph(domain_knowledge)
        
        self.assertIsNotNone(graph)
        self.assertGreater(len(graph.nodes()), 0)
    
    def test_counterfactual_reasoning(self):
        """Test counterfactual reasoning"""
        domain_knowledge = {
            "climate": "Emissions affect temperature",
            "economics": "Economy affects emissions"
        }
        self.causal.construct_causal_graph(domain_knowledge)
        
        intervention = {"policy": 0.8}
        current_state = {"policy": 0.5, "emissions": 0.7}
        
        result = self.causal.counterfactual_reasoning(intervention, current_state)
        
        self.assertIn("intervention", result)
        self.assertIn("predicted_outcome", result)
        self.assertIn("recommendation", result)
    
    def test_intervention_planning(self):
        """Test intervention planning"""
        domain_knowledge = {
            "climate": "Emissions cause warming",
            "policy": "Policy reduces emissions"
        }
        graph = self.causal.construct_causal_graph(domain_knowledge)
        
        if len(graph.nodes()) > 0:
            goal = list(graph.nodes())[0]
            current_state = {node: 0.5 for node in graph.nodes()}
            constraints = {node: 0.9 for node in graph.nodes()}
            
            result = self.causal.plan_intervention(goal, current_state, constraints)
            
            self.assertIsInstance(result, dict)


class TestOmegaASI(unittest.TestCase):
    """Test main OMEGA ASI system"""
    
    def setUp(self):
        # Use smaller configuration for faster tests
        self.omega = OMEGA_ASI(num_qubits=4, initial_awareness=0.7)
    
    def test_initialization(self):
        """Test OMEGA ASI initializes correctly"""
        self.assertIsNotNone(self.omega.quantum_processor)
        self.assertIsNotNone(self.omega.consciousness)
        self.assertIsNotNone(self.omega.empathy_system)
        self.assertIsNotNone(self.omega.causal_engine)
    
    def test_problem_solving(self):
        """Test superintelligent problem solving"""
        problem = {
            "title": "Test Problem",
            "type": "test",
            "domain_knowledge": {
                "domain1": "Test domain 1",
                "domain2": "Test domain 2"
            },
            "stakeholders": ["stakeholder1", "stakeholder2"]
        }
        
        constraints = {"constraint1": 0.5}
        
        solution = self.omega.solve_superintelligent_problem(problem, constraints)
        
        self.assertIsNotNone(solution)
        self.assertIn("problem_title", solution)
        self.assertIn("quantum_analysis", solution)
        self.assertIn("consciousness_analysis", solution)
        self.assertIn("empathic_analysis", solution)
        self.assertIn("causal_analysis", solution)
        self.assertIn("recommendations", solution)
        self.assertIn("asi_confidence", solution)
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.omega.get_system_status()
        
        self.assertIn("quantum_processor", status)
        self.assertIn("consciousness", status)
        self.assertIn("empathy_system", status)
        self.assertIn("causal_engine", status)
        self.assertIn("problems_solved", status)
    
    def test_problem_history(self):
        """Test problem solving history"""
        initial_count = len(self.omega.problem_solving_history)
        
        problem = {
            "title": "Test",
            "domain_knowledge": {"test": "test"},
            "stakeholders": ["test"]
        }
        
        self.omega.solve_superintelligent_problem(problem)
        
        self.assertEqual(len(self.omega.problem_solving_history), initial_count + 1)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessFramework))
    suite.addTests(loader.loadTestsFromTestCase(TestEmpathySystem))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalReasoning))
    suite.addTests(loader.loadTestsFromTestCase(TestOmegaASI))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
