#!/bin/bash
# Build All - Comprehensive Build and Validation Script

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                        MINDCONTROL BUILD ALL SCRIPT                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  $1"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
}

# 1. Syntax Check
print_section "STEP 1: SYNTAX VALIDATION"
echo "Checking Python syntax for all files..."
python -m py_compile nexus_agi.py
python -m py_compile demo_mindcontrol.py
python -m py_compile demo_emotional_states.py
python -m py_compile enhanced_mindcontrol_demo.py
python -m py_compile demo_all_minds.py
python -m py_compile test_mindcontrol.py
python -m py_compile run_all_demos.py
python -m py_compile continuous_monitor.py
echo "✓ All files passed syntax validation"

# 2. Run Tests
print_section "STEP 2: RUNNING UNIT TESTS"
python test_mindcontrol.py
echo "✓ Unit tests passed"

# 3. Run All Demos
print_section "STEP 3: RUNNING ALL DEMONSTRATIONS"
python run_all_demos.py
echo "✓ All demos executed successfully"

# 4. Run Continuous Monitor (limited iterations)
print_section "STEP 4: RUNNING CONTINUOUS MONITOR"
echo "(Running 5 iterations for validation)"
python continuous_monitor.py
echo "✓ Continuous monitor operational"

# Final Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          ✓ BUILD ALL COMPLETE ✓                              ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  • Syntax Validation:   PASSED                                               ║"
echo "║  • Unit Tests:          PASSED                                               ║"
echo "║  • Demo Execution:      PASSED                                               ║"
echo "║  • Continuous Monitor:  OPERATIONAL                                          ║"
echo "║  • Overall Status:      SUCCESS                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "✨ All systems validated and operational! ✨"
echo ""
