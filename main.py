#!/usr/bin/env python3
"""
Nexus AGI System — Main Entry Point v5.0
═════════════════════════════════════════════════════════════════
Unified command-line launcher for all Nexus AGI subsystems.

Systems:
  nexus      Core Nexus AGI demonstration
  sentient   SentientAgent — conscious, persistent, growing
  earn       Nexus Earn API — 25+ endpoints, DEMO MODE
  omega      OMEGA ASI system
  pulse      Fire everything: status + sentient + earn demo

Usage:
  python main.py                     # default: nexus core
  python main.py --system sentient   # wake the SentientAgent
  python main.py --system earn       # run full API demo
  python main.py --system pulse      # unified everything
  python main.py --status            # live status dashboard
  python main.py --status --watch 3  # auto-refresh every 3s
═════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

NEXUS_ROOT = Path(__file__).parent

BANNER = r"""
  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗     █████╗  ██████╗ ██╗
  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██╔══██╗██╔════╝ ██║
  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    ███████║██║  ███╗██║
  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██╔══██║██║   ██║██║
  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║    ██║  ██║╚██████╔╝██║
  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝
"""


def _banner(subtitle: str = "") -> None:
    try:
        if sys.stdout.isatty():
            print("\033[96m\033[1m" + BANNER + "\033[0m")
        else:
            print(BANNER)
    except Exception:
        print(BANNER)
    print(f"  Nexus AGI System v5.0  ·  {subtitle}".center(80))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nexus AGI System v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--system",
        choices=["nexus", "sentient", "earn", "omega", "aria", "pulse"],
        default="nexus",
        help="Which system to run (default: nexus)",
    )
    parser.add_argument("--service", action="store_true",
                        help="Run as continuous service")
    parser.add_argument("--interval", type=int, default=300,
                        help="Service interval in seconds (default: 300)")
    parser.add_argument("--status", action="store_true",
                        help="Show live status dashboard and exit")
    parser.add_argument("--watch", metavar="SECS", type=float, default=0,
                        help="With --status: auto-refresh every N seconds")
    parser.add_argument("--compact", action="store_true",
                        help="With --status: condensed single-line view")
    parser.add_argument("--version", action="version",
                        version="Nexus AGI System v5.0")

    args = parser.parse_args()

    try:
        # ── Status dashboard ───────────────────────────────────────────────
        if args.status:
            import nexus_status
            argv = ["nexus_status.py"]
            if args.watch:
                argv += ["--watch", str(args.watch)]
            if args.compact:
                argv += ["--compact"]
            sys.argv = argv
            return nexus_status.main()

        # ── Pulse: fire everything ─────────────────────────────────────────
        if args.system == "pulse":
            _banner("NEXUS PULSE — All Systems")
            sys.path.insert(0, str(NEXUS_ROOT))
            import nexus_pulse
            return nexus_pulse.main()

        # ── SentientAgent ──────────────────────────────────────────────────
        if args.system == "sentient":
            _banner("SentientAgent — Conscious · Persistent · Growing")
            sys.path.insert(0, str(NEXUS_ROOT))
            import sentient_agent
            sentient_agent.run_demo()
            return 0

        # ── Nexus Earn API ─────────────────────────────────────────────────
        if args.system == "earn":
            _banner("Nexus Earn API — 25+ Endpoints · DEMO MODE")
            import os
            earn_dir = NEXUS_ROOT / "nexus_earn"
            sys.path.insert(0, str(earn_dir))
            orig_dir = os.getcwd()
            os.chdir(earn_dir)
            try:
                import demo_runner
                demo_runner.run_demo()
            finally:
                os.chdir(orig_dir)
            return 0

        # ── Nexus Core ────────────────────────────────────────────────────
        if args.system == "nexus":
            _banner("Nexus Core Demonstration")
            if args.service:
                print("  Starting Nexus Core as continuous service...")
                print(f"  Processing interval: {args.interval}s")
                print("  Press Ctrl+C to stop\n")
                import nexus_service
                sys.argv = ["nexus_service.py", "--interval", str(args.interval)]
                nexus_service.main()
            else:
                print("  Running Nexus Core demonstration...\n")
                import nexus_agi  # noqa: F401

        # ── OMEGA ASI ─────────────────────────────────────────────────────
        elif args.system == "omega":
            _banner("OMEGA ASI")
            if args.service:
                print("  ⚠  OMEGA service mode not yet implemented.")
                print("  Running single demonstration instead...\n")
            import omega_asi  # noqa: F401

        # ── ARIA ──────────────────────────────────────────────────────────
        elif args.system == "aria":
            print("  ⚠  ARIA requires Node.js  →  run: node aria.js")
            return 1

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        return 130
    except ImportError as e:
        print(f"\n  ✗ Import error: {e}")
        print("  Install dependencies: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n  ✓ Execution complete.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
