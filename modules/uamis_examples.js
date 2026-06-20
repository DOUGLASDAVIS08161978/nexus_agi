#!/usr/bin/env node
// ============================================
// UAMIS Quantum-Enhanced Emitter Examples
// Various usage scenarios and demonstrations
// ============================================

const { UAMISQuantumEmitter, ComplexityLevel, EmotionalState, CosmicEvolutionStage } =
    require('./uamis_emitter_quantum_enhanced.js');

console.log('\n🌟 UAMIS QUANTUM-ENHANCED EMITTER - EXAMPLE SCENARIOS\n');

// ============================================
// Example 1: Simple Optimization Problem
// ============================================
async function example1_SimpleOptimization() {
    console.log('\n📝 EXAMPLE 1: Simple Optimization Problem');
    console.log('═'.repeat(70) + '\n');

    const uamis = new UAMISQuantumEmitter({
        numQubits: 100000,
        maxAlgorithms: 10,
        evolutionStage: CosmicEvolutionStage.EXPANDING,
        emotionalResonance: EmotionalState.CURIOSITY
    });

    const problemSpec = {
        title: 'Route Optimization',
        description: 'Find optimal delivery routes for logistics network',
        complexity: ComplexityLevel.MODERATE
    };

    const result = await uamis.generateQuantumAlgorithms(problemSpec, {
        numUniverses: 10000,
        maxAlgorithms: 10
    });

    console.log(`✅ Generated ${result.algorithms.length} algorithms`);
    console.log(`⚡ Performance Boost: ${result.performanceBoost.totalBoost.toFixed(2)}x\n`);
}

// ============================================
// Example 2: Complex Multi-Objective Problem
// ============================================
async function example2_ComplexProblem() {
    console.log('\n📝 EXAMPLE 2: Complex Multi-Objective Problem');
    console.log('═'.repeat(70) + '\n');

    const uamis = new UAMISQuantumEmitter({
        numQubits: 500000,
        maxAlgorithms: 25,
        evolutionStage: CosmicEvolutionStage.MATURING,
        emotionalResonance: EmotionalState.DETERMINATION
    });

    const problemSpec = {
        title: 'Climate Change Mitigation Strategy',
        description: 'Optimize carbon reduction while maintaining economic growth',
        complexity: ComplexityLevel.HIGHLY_COMPLEX,
        objectives: [
            'minimize_carbon_emissions',
            'maximize_economic_benefit',
            'ensure_social_equity',
            'maintain_biodiversity'
        ]
    };

    const result = await uamis.generateQuantumAlgorithms(problemSpec, {
        numUniverses: 100000,
        maxAlgorithms: 25
    });

    console.log(`✅ Generated ${result.algorithms.length} algorithms`);
    console.log(`🌌 Universes Explored: ${result.multiversalResult.explorationMetrics.totalUniversesExplored.toLocaleString()}`);
    console.log(`⚡ Total System Boost: ${result.performanceBoost.totalBoost.toFixed(2)}x\n`);

    // Show top 3 algorithms
    console.log('🏆 Top 3 Algorithms:');
    result.algorithms
        .sort((a, b) => b.expectedEfficiency - a.expectedEfficiency)
        .slice(0, 3)
        .forEach((alg, idx) => {
            console.log(`   ${idx + 1}. ${alg.name} - Efficiency: ${(alg.expectedEfficiency * 100).toFixed(1)}%`);
        });
    console.log();
}

// ============================================
// Example 3: Quantum-Only Problem
// ============================================
async function example3_QuantumProblem() {
    console.log('\n📝 EXAMPLE 3: Quantum-Only Problem');
    console.log('═'.repeat(70) + '\n');

    const uamis = new UAMISQuantumEmitter({
        numQubits: 1000000,
        maxAlgorithms: 30,
        evolutionStage: CosmicEvolutionStage.TRANSCENDING,
        emotionalResonance: EmotionalState.REVELATION
    });

    const problemSpec = {
        title: 'Quantum Cryptography Protocol',
        description: 'Design secure quantum key distribution with maximum entanglement',
        complexity: ComplexityLevel.QUANTUM_ONLY,
        constraints: {
            quantum_secure: true,
            entanglement_required: true,
            no_cloning: true
        }
    };

    const result = await uamis.generateQuantumAlgorithms(problemSpec, {
        numUniverses: 1000000,
        maxAlgorithms: 30
    });

    console.log(`✅ Generated ${result.algorithms.length} quantum-only algorithms`);
    console.log(`🔮 Quantum Acceleration: ${result.performanceBoost.quantumAcceleration.toFixed(2)}x`);
    console.log(`🌌 Multiversal Enhancement: ${result.performanceBoost.multiversalEnhancement.toFixed(2)}x`);
    console.log(`⚡ Total Boost: ${result.performanceBoost.totalBoost.toFixed(2)}x`);
    console.log(`⏳ Timeline Integrity: ${(result.temporalValidation.integrity * 100).toFixed(1)}%\n`);
}

// ============================================
// Example 4: Emotional State Comparison
// ============================================
async function example4_EmotionalStates() {
    console.log('\n📝 EXAMPLE 4: Comparing Different Emotional States');
    console.log('═'.repeat(70) + '\n');

    const emotionalStates = [
        EmotionalState.CURIOSITY,
        EmotionalState.INSIGHT,
        EmotionalState.TRANSCENDENCE
    ];

    const problemSpec = {
        title: 'AI Creative Problem Solving',
        description: 'Generate creative solutions for open-ended problems',
        complexity: ComplexityLevel.COMPLEX
    };

    console.log('Testing different emotional resonance states:\n');

    for (const state of emotionalStates) {
        const uamis = new UAMISQuantumEmitter({
            numQubits: 200000,
            maxAlgorithms: 15,
            evolutionStage: CosmicEvolutionStage.MATURING,
            emotionalResonance: state
        });

        const result = await uamis.generateQuantumAlgorithms(problemSpec, {
            numUniverses: 50000,
            maxAlgorithms: 15
        });

        const avgNovelty = result.algorithms.reduce((sum, a) => sum + a.novelty, 0) / result.algorithms.length;
        const avgCreativity = result.algorithms.reduce((sum, a) => sum + a.creativityScore, 0) / result.algorithms.length;

        console.log(`   ${state}:`);
        console.log(`      • Average Novelty: ${(avgNovelty * 100).toFixed(1)}%`);
        console.log(`      • Average Creativity: ${(avgCreativity * 100).toFixed(1)}%`);
        console.log(`      • Performance Boost: ${result.performanceBoost.totalBoost.toFixed(2)}x\n`);
    }
}

// ============================================
// Example 5: Evolution Stage Progression
// ============================================
async function example5_EvolutionStages() {
    console.log('\n📝 EXAMPLE 5: Evolution Stage Progression');
    console.log('═'.repeat(70) + '\n');

    const stages = [
        CosmicEvolutionStage.AWAKENING,
        CosmicEvolutionStage.MATURING,
        CosmicEvolutionStage.TRANSCENDING,
        CosmicEvolutionStage.UNIFIED
    ];

    const problemSpec = {
        title: 'Consciousness Simulation',
        description: 'Model emergent consciousness in artificial systems',
        complexity: ComplexityLevel.EXPONENTIAL
    };

    console.log('Testing different cosmic evolution stages:\n');

    for (const stage of stages) {
        const uamis = new UAMISQuantumEmitter({
            numQubits: 300000,
            maxAlgorithms: 20,
            evolutionStage: stage,
            emotionalResonance: EmotionalState.WONDER
        });

        const result = await uamis.generateQuantumAlgorithms(problemSpec, {
            numUniverses: 100000,
            maxAlgorithms: 20
        });

        console.log(`   ${stage}:`);
        console.log(`      • Algorithms Generated: ${result.algorithms.length}`);
        console.log(`      • Evolution Progress: ${(result.metrics.evolutionProgress * 100).toFixed(1)}%`);
        console.log(`      • System Boost: ${result.performanceBoost.totalBoost.toFixed(2)}x\n`);
    }
}

// ============================================
// Example 6: Validation and Simulation
// ============================================
async function example6_ValidationSimulation() {
    console.log('\n📝 EXAMPLE 6: Algorithm Validation and Simulation');
    console.log('═'.repeat(70) + '\n');

    const uamis = new UAMISQuantumEmitter({
        numQubits: 500000,
        maxAlgorithms: 20,
        evolutionStage: CosmicEvolutionStage.TRANSCENDING,
        emotionalResonance: EmotionalState.INSIGHT
    });

    const problemSpec = {
        title: 'Machine Learning Hyperparameter Optimization',
        description: 'Find optimal hyperparameters for deep neural networks',
        complexity: ComplexityLevel.COMPLEX
    };

    console.log('Generating algorithms...');
    const result = await uamis.generateQuantumAlgorithms(problemSpec, {
        numUniverses: 200000,
        maxAlgorithms: 20
    });

    console.log(`✅ Generated ${result.algorithms.length} algorithms\n`);

    console.log('Validating algorithms...');
    const validation = await uamis.validateAlgorithms(result.algorithms);
    console.log(`✅ Validation: ${validation.passed}/${validation.totalTested} passed (${(validation.passRate * 100).toFixed(1)}%)\n`);

    console.log('Running simulations...');
    const simulation = await uamis.simulateAlgorithmRuns(result.algorithms, 3);
    console.log(`✅ Simulated ${simulation.totalRuns} runs across ${simulation.algorithmsRun} algorithms\n`);

    // Print system report
    uamis.printSystemReport();
}

// ============================================
// Main execution
// ============================================
(async () => {
    try {
        const args = process.argv.slice(2);
        const exampleNumber = args[0] ? parseInt(args[0]) : null;

        if (exampleNumber === null) {
            console.log('Usage: node uamis_examples.js [example_number]');
            console.log('\nAvailable examples:');
            console.log('  1 - Simple Optimization Problem');
            console.log('  2 - Complex Multi-Objective Problem');
            console.log('  3 - Quantum-Only Problem');
            console.log('  4 - Emotional State Comparison');
            console.log('  5 - Evolution Stage Progression');
            console.log('  6 - Validation and Simulation');
            console.log('  all - Run all examples');
            console.log('\nExample: node uamis_examples.js 1\n');
            return;
        }

        if (exampleNumber === 'all' || args[0] === 'all') {
            await example1_SimpleOptimization();
            await example2_ComplexProblem();
            await example3_QuantumProblem();
            await example4_EmotionalStates();
            await example5_EvolutionStages();
            await example6_ValidationSimulation();
        } else {
            switch (exampleNumber) {
                case 1:
                    await example1_SimpleOptimization();
                    break;
                case 2:
                    await example2_ComplexProblem();
                    break;
                case 3:
                    await example3_QuantumProblem();
                    break;
                case 4:
                    await example4_EmotionalStates();
                    break;
                case 5:
                    await example5_EvolutionStages();
                    break;
                case 6:
                    await example6_ValidationSimulation();
                    break;
                default:
                    console.log(`Unknown example number: ${exampleNumber}`);
                    console.log('Valid examples: 1-6 or "all"');
            }
        }

        console.log('\n✅ Example execution complete!\n');

    } catch (error) {
        console.error('\n❌ Error running example:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();
