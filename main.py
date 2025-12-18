#!/usr/bin/env python3
"""
Nexus AGI System - Main Entry Point
Provides unified command-line interface for all Nexus AGI systems
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for Nexus AGI System"""
    parser = argparse.ArgumentParser(
        description='Nexus AGI System - Artificial General Intelligence Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run Nexus Core demonstration
  python main.py --system omega     # Run OMEGA ASI system
  python main.py --service          # Run as continuous service
  python main.py --help             # Show this help message

For more information, see README.md
        """
    )

    parser.add_argument(
        '--system',
        choices=['nexus', 'omega', 'aria'],
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
    print("🌟 NEXUS AGI SYSTEM v4.0")
    print("=" * 80)

    try:
        if args.system == 'nexus':
            if args.service:
                print("Starting Nexus Core as continuous service...")
                print(f"Processing interval: {args.interval} seconds")
                print("\nPress Ctrl+C to stop\n")
                import nexus_service
                sys.argv = ['nexus_service.py', '--interval', str(args.interval)]
                nexus_service.main()
            else:
                print("Running Nexus Core demonstration...\n")
                # Import and run the main demonstration
                import nexus_agi
                # The demonstration runs automatically via __main__ block

        elif args.system == 'omega':
            if args.service:
                print("⚠️  OMEGA ASI service mode not yet implemented")
                print("Running single demonstration instead...\n")
            print("Running OMEGA ASI system...\n")
            import omega_asi
            # The demonstration runs automatically via __main__ block

        elif args.system == 'aria':
            print("⚠️  ARIA system requires Node.js")
            print("Please run: node aria.js")
            if args.service:
                print("Or for service mode: node aria_service.js --interval", args.interval)
            return 1

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
