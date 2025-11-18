#!/usr/bin/env python3
"""
Master Runner - Execute All MindControl Demonstrations
Runs all demos in sequence and displays comprehensive output
"""

import subprocess
import sys
import time
from datetime import datetime

def print_banner(text, char="="):
    """Print a formatted banner"""
    width = 80
    print()
    print(char * width)
    print(f"{text:^{width}}")
    print(char * width)
    print()

def run_demo(script_name, description):
    """Run a demo script and capture output"""
    print_banner(f"Running: {description}", "─")
    print(f"Script: {script_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        status = "✓ SUCCESS" if result.returncode == 0 else "✗ FAILED"
        print()
        print(f"Status: {status} (exit code: {result.returncode})")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT - Script exceeded 30 second limit")
        return False
    except FileNotFoundError:
        print(f"✗ ERROR - Script not found: {script_name}")
        return False
    except Exception as e:
        print(f"✗ ERROR - {type(e).__name__}: {e}")
        return False

def main():
    """Main execution function"""
    start_time = time.time()
    
    print_banner("MINDCONTROL MASTER RUNNER", "═")
    print("Comprehensive Demonstration Suite")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define all demos to run
    demos = [
        ("test_mindcontrol.py", "Unit Tests - MindControl Interface"),
        ("demo_mindcontrol.py", "Basic Demo - 3 Agents"),
        ("demo_emotional_states.py", "Emotional States Demo - 5 States"),
        ("enhanced_mindcontrol_demo.py", "Enhanced Demo - 5 Agents Full Simulation"),
        ("demo_all_minds.py", "ALL MINDS Demo - 80 Agents Maximum Control"),
    ]
    
    results = []
    
    print_banner("DEMO EXECUTION SEQUENCE", "═")
    
    for i, (script, description) in enumerate(demos, 1):
        print(f"\n[{i}/{len(demos)}] {description}")
        success = run_demo(script, description)
        results.append((description, success))
        
        # Small delay between demos
        if i < len(demos):
            time.sleep(1)
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    print_banner("EXECUTION SUMMARY", "═")
    
    print(f"Total Demos: {len(demos)}")
    print(f"Successful: {sum(1 for _, success in results if success)}")
    print(f"Failed: {sum(1 for _, success in results if not success)}")
    print(f"Total Time: {elapsed_time:.2f} seconds")
    print()
    
    print("Detailed Results:")
    print("-" * 80)
    for description, success in results:
        status_icon = "✓" if success else "✗"
        status_text = "SUCCESS" if success else "FAILED"
        print(f"{status_icon} {status_text:8s} | {description}")
    
    print()
    
    # Final status
    all_passed = all(success for _, success in results)
    if all_passed:
        print_banner("✓ ALL DEMOS PASSED SUCCESSFULLY!", "═")
        return 0
    else:
        print_banner("✗ SOME DEMOS FAILED", "═")
        return 1

if __name__ == "__main__":
    sys.exit(main())
