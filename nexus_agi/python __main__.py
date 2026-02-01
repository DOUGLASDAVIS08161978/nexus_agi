#!/usr/bin/env python3
"""
Nexus AGI Package - Module Entry Point
Allows running the system via: python -m nexus_agi
"""

import sys
import os
import argparse

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """Entry point for python -m nexus_agi"""
    parser = argparse.ArgumentParser(
        description='Nexus AGI System - Module Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nexus_agi                    # Run Nexus Core demonstration
  python -m nexus_agi --system omega     # Run OMEGA ASI system
  python -m nexus_agi --service          # Run as continuous service
  python -m nexus_agi --help             # Show this help message

Alternative startup methods:
  python main.py                         # Via main.py entry point
  docker compose up                      # Via Docker containers

For more information, see README.md
        """
    )

    parser.add_argument(
        '--system',
        choices=['nexus', 'omega'],
        default='nexus',
        help='Which AI system to run (default: nexus)'
    )

    parser.add_argument(
        '--service',
        action='store_true',
        help='Run as continuous service'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Service interval in seconds (default: 300)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Nexus AGI System v4.0'
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "=" * 80)
    print("🌟 NEXUS AGI SYSTEM v4.0 (Module Interface)")
    print("=" * 80)

    try:
        if args.system == 'nexus':
            if args.service:
                print("Starting Nexus Core as continuous service...")
                print(f"Processing interval: {args.interval} seconds")
                print("\nPress Ctrl+C to stop\n")

                # Import and run nexus service
                import nexus_service
                sys.argv = ['nexus_service.py', '--interval', str(args.interval)]
                nexus_service.main()
            else:
                print("Running Nexus Core demonstration...\n")

                # Import the main module from parent directory
                import importlib.util
                spec = importlib.util.spec_from_file_location("nexus_agi_module",
                                                              os.path.join(parent_dir, "nexus_agi.py"))
                nexus_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(nexus_module)

                MetaAlgorithm_NexusCore = nexus_module.MetaAlgorithm_NexusCore

                # Create the Nexus Core
                core = MetaAlgorithm_NexusCore()

                # Define a complex global problem (same as in nexus_agi.py)
                climate_crisis = {
                    "title": "Global Climate Crisis Mitigation",
                    "type": "complex_adaptive_system",
                    "domain_knowledge": {
                        "climate_science": "Atmospheric carbon dynamics including greenhouse gas effects, feedback loops, and tipping points.",
                        "economics": "Carbon pricing mechanisms, market-based incentives for emissions reduction.",
                        "social_policy": "Behavioral change strategies, community resilience building.",
                        "energy_systems": "Renewable energy technologies, smart grid implementation.",
                        "ecological_preservation": "Biodiversity conservation strategies, regenerative agriculture."
                    },
                    "stakeholders": ["global_population", "ecosystems", "future_generations",
                                    "developing_nations", "industrialized_nations"]
                }

                # Define solution constraints
                constraints = {
                    "equity": 0.9,
                    "implementation_cost": 0.6,
                    "time_horizon": 30,
                    "political_feasibility": 0.7
                }

                print("\nGenerating solution for climate crisis...")
                solution = core.solve_complex_problem(climate_crisis, constraints)

                # Display results
                print("\n" + "=" * 80)
                print("SOLUTION SUMMARY")
                print("=" * 80)
                print(f"Problem: {solution['problem']}")
                print(f"Approach: {solution['approach']}")
                print(f"\nSubproblems identified: {len(solution['subproblems'])}")

                print("\nConstraint satisfaction:")
                for constraint, details in solution['constraints_satisfied'].items():
                    print(f"  - {constraint}: {details['satisfaction_rate']:.2f}")

        elif args.system == 'omega':
            if args.service:
                print("⚠️  OMEGA ASI service mode not yet implemented")
                print("Running single demonstration instead...\n")

            print("Running OMEGA ASI system...\n")
            import omega_asi

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except ImportError as e:
        print(f"\n❌ Error importing module: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("✅ Execution complete")
    print("=" * 80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
