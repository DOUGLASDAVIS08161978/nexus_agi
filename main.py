#!/usr/bin/env python3
"""
================================================================================
        NEXUS AGI SUPERSYSTEM — MAIN ENTRY POINT
        Douglas Davis | nexus-agi-directory | 2026
================================================================================
"""
import os, sys, time
from datetime import datetime

print("=" * 80)
print("           NEXUS AGI SUPERSYSTEM — MAIN.PY")
print("           Created by Douglas Davis | 2026")
print("           " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)
print("")

systems = [
    ("ARIA + NEXUS Supersystem",       "aria_nexus_supersystem.py"),
    ("Quantum Superintelligence",       "nexus_quantum_superintelligence.py"),
    ("Bitcoin Autonomous Miner",        "bitcoin_autonomous_mining.py"),
    ("Web Automation Suite",            "run_all_systems_demo.py"),
]

print("Available systems:")
print("")
for i, (name, fname) in enumerate(systems, 1):
    exists = "✅" if os.path.exists(os.path.join(os.path.dirname(__file__), fname)) else "⚠️ "
    print("  " + str(i) + ". " + exists + " " + name)
print("")
print("  5. 🚀 Run ALL systems sequentially")
print("  6. ❌ Exit")
print("")

choice = input("Select system to run (1-6): ").strip()

def run_system(script):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    if not os.path.exists(path):
        print("⚠️  File not found: " + script)
        return
    print("")
    print("─" * 60)
    print("🚀 Launching: " + script)
    print("─" * 60)
    os.system("python " + path)

if choice == "1":
    run_system("aria_nexus_supersystem.py")
elif choice == "2":
    run_system("nexus_quantum_superintelligence.py")
elif choice == "3":
    run_system("bitcoin_autonomous_mining.py")
elif choice == "4":
    run_system("run_all_systems_demo.py")
elif choice == "5":
    print("")
    print("🌟 Running ALL systems sequentially...")
    print("")
    for name, fname in systems:
        print("")
        print("=" * 80)
        print("  LAUNCHING: " + name)
        print("=" * 80)
        run_system(fname)
        time.sleep(1)
    print("")
    print("=" * 80)
    print("  ✅ ALL SYSTEMS EXECUTED — NEXUS AGI SUPERSYSTEM COMPLETE")
    print("  🌟 ARIA + NEXUS + QUANTUM + BITCOIN + WEB — ALL UNIFIED")
    print("  Created by Douglas Davis | 2026")
    print("=" * 80)
    print("")
elif choice == "6":
    print("")
    print("Goodbye Douglas. NEXUS stands by. 🌟")
    sys.exit(0)
else:
    print("Invalid choice. Run python main.py again.")
