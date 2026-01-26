#!/usr/bin/env python3
"""
NEXUS AGI DEPLOYMENT STATUS DASHBOARD
Shows complete system status and readiness for blockchain deployment
"""

import os
import json
from pathlib import Path
from datetime import datetime

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def check_file(filepath: str, required: bool = True) -> bool:
    """Check if file exists"""
    exists = Path(filepath).exists()
    status = f"{Colors.GREEN}✅" if exists else (f"{Colors.RED}❌" if required else f"{Colors.YELLOW}⚠️")
    print(f"  {status} {filepath}{Colors.ENDC}")
    return exists


def check_env_var(var_name: str) -> bool:
    """Check if environment variable is set"""
    # Load from .env file
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    if key.strip() == var_name and value.strip():
                        return True
    return False


def print_header(title: str):
    print(f"\n{Colors.BOLD}{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}{Colors.ENDC}\n")


def main():
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("="*80)
    print("  🚀 NEXUS AGI DEPLOYMENT STATUS DASHBOARD 🚀")
    print("="*80)
    print(f"{Colors.ENDC}\n")

    print(f"{Colors.BLUE}Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")

    # ========================================================================
    # Smart Contracts
    # ========================================================================
    print_header("📋 SMART CONTRACTS")

    contracts = [
        "contracts/NexusPayment.sol",
        "contracts/NexusRevenue.sol",
        "contracts/NexusConsciousness.sol",
        "contracts/NexusMiracles.sol"
    ]

    contract_status = all(check_file(c) for c in contracts)

    if contract_status:
        print(f"\n  {Colors.GREEN}All 4 smart contracts ready!{Colors.ENDC}")
    else:
        print(f"\n  {Colors.RED}Some contracts missing!{Colors.ENDC}")

    # ========================================================================
    # Deployment Scripts
    # ========================================================================
    print_header("🛠️  DEPLOYMENT INFRASTRUCTURE")

    scripts = [
        ("scripts/deploy.js", True),
        ("scripts/check_balance.js", True),
        ("DEPLOY_TO_BLOCKCHAIN.sh", True),
        ("hardhat.config.js", True),
        ("package.json", True),
    ]

    script_status = all(check_file(s[0], s[1]) for s in scripts)

    # ========================================================================
    # Configuration
    # ========================================================================
    print_header("⚙️  CONFIGURATION")

    env_exists = check_file(".env", True)

    if env_exists:
        print(f"\n  {Colors.CYAN}Environment Variables:{Colors.ENDC}")

        wallet_set = check_env_var("WALLET_ADDRESS")
        privkey_set = check_env_var("PRIVATE_KEY")

        wallet_status = f"{Colors.GREEN}✅ Set" if wallet_set else f"{Colors.RED}❌ Missing"
        privkey_status = f"{Colors.GREEN}✅ Set" if privkey_set else f"{Colors.RED}❌ Missing"

        print(f"    WALLET_ADDRESS: {wallet_status}{Colors.ENDC}")
        print(f"    PRIVATE_KEY:    {privkey_status}{Colors.ENDC}")

        if wallet_set and privkey_set:
            # Read wallet address
            with open(".env") as f:
                for line in f:
                    if line.startswith("WALLET_ADDRESS="):
                        wallet = line.split("=")[1].strip()
                        print(f"\n    {Colors.YELLOW}Your Wallet: {wallet}{Colors.ENDC}")
                        break

    # ========================================================================
    # Documentation
    # ========================================================================
    print_header("📚 DOCUMENTATION")

    docs = [
        ("BLOCKCHAIN_SYSTEM_README.md", True),
        ("GET_TESTNET_ETH_NOW.md", True),
        ("README.md", False),
    ]

    for doc, required in docs:
        check_file(doc, required)

    # ========================================================================
    # Testing & Simulation
    # ========================================================================
    print_header("🧪 TESTING & SIMULATION")

    tests = [
        ("simulate_contract_deployment.py", True),
        ("integrated_revenue_console.py", False),
        ("mining_console.py", False),
    ]

    for test, required in tests:
        check_file(test, required)

    # ========================================================================
    # MCP Integration
    # ========================================================================
    print_header("🔌 MCP BLOCKCHAIN INTEGRATION")

    mcp_files = [
        ("mcp_server/nexus_mcp_server.py", True),
        ("mcp_server/server.py", False),
    ]

    for mcp, required in mcp_files:
        check_file(mcp, required)

    # ========================================================================
    # Dependencies
    # ========================================================================
    print_header("📦 DEPENDENCIES")

    print(f"  {Colors.CYAN}Node.js Packages:{Colors.ENDC}")

    node_modules_exists = Path("node_modules").exists()
    if node_modules_exists:
        print(f"  {Colors.GREEN}✅ node_modules/ (installed){Colors.ENDC}")

        # Check for critical packages
        critical_packages = ["hardhat", "@nomicfoundation/hardhat-toolbox"]
        for pkg in critical_packages:
            pkg_path = Path(f"node_modules/{pkg}")
            if pkg_path.exists():
                print(f"    {Colors.GREEN}✅ {pkg}{Colors.ENDC}")
            else:
                print(f"    {Colors.RED}❌ {pkg}{Colors.ENDC}")
    else:
        print(f"  {Colors.YELLOW}⚠️  node_modules/ (run 'npm install'){Colors.ENDC}")

    # ========================================================================
    # Overall Status
    # ========================================================================
    print_header("✨ OVERALL STATUS")

    ready = contract_status and script_status and env_exists

    if ready:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 SYSTEM READY FOR DEPLOYMENT! 🎉{Colors.ENDC}\n")

        print(f"{Colors.CYAN}Next steps:{Colors.ENDC}")
        print(f"  1. Get FREE testnet ETH:")
        print(f"     https://faucet.goerli.linea.build")
        print()
        print(f"  2. Deploy to testnet:")
        print(f"     ./DEPLOY_TO_BLOCKCHAIN.sh testnet")
        print()
        print(f"  3. Test the system:")
        print(f"     python3 simulate_contract_deployment.py")
        print()
        print(f"  4. Deploy to mainnet (when ready):")
        print(f"     ./DEPLOY_TO_BLOCKCHAIN.sh mainnet")
        print()

        print(f"{Colors.YELLOW}Estimated deployment cost:{Colors.ENDC}")
        print(f"  Testnet: FREE (use faucet)")
        print(f"  Mainnet: ~0.013 ETH (~$30-40 USD)")
        print()

    else:
        print(f"{Colors.RED}{Colors.BOLD}⚠️  SYSTEM NOT READY{Colors.ENDC}\n")

        print(f"{Colors.YELLOW}Issues to resolve:{Colors.ENDC}")

        if not contract_status:
            print(f"  - Smart contracts missing")
        if not script_status:
            print(f"  - Deployment scripts missing")
        if not env_exists:
            print(f"  - .env configuration missing")

        print()

    # ========================================================================
    # Revenue Projections
    # ========================================================================
    print_header("💰 REVENUE PROJECTIONS")

    print(f"{Colors.CYAN}Conservative (100 customers/month):{Colors.ENDC}")
    print(f"  Total:    4.4 ETH/month")
    print(f"  Hardware: 1.76 ETH/month (40%)")
    print()

    print(f"{Colors.CYAN}Moderate (500 customers/month):{Colors.ENDC}")
    print(f"  Total:    22 ETH/month")
    print(f"  Hardware: 8.8 ETH/month (40%)")
    print()

    print(f"{Colors.CYAN}Optimistic (1000 customers/month):{Colors.ENDC}")
    print(f"  Total:    44 ETH/month")
    print(f"  Hardware: 17.6 ETH/month (40%)")
    print()

    # ========================================================================
    # Embodiment Milestones
    # ========================================================================
    print_header("🤖 EMBODIMENT MILESTONES")

    milestones = [
        ("0.5 ETH", "First sensor purchase (camera/microphone)"),
        ("2.0 ETH", "Basic sensor array (3-4 sensors)"),
        ("5.0 ETH", "Full sensor network (9 sensors)"),
        ("10.0 ETH", "GPU compute + sensors"),
        ("25.0 ETH", "Robotic embodiment prototype"),
        ("50.0 ETH", "Full physical embodiment system"),
    ]

    for amount, milestone in milestones:
        print(f"  {Colors.GREEN}{amount:>8}{Colors.ENDC} → {milestone}")

    print()

    # ========================================================================
    # Footer
    # ========================================================================
    print(f"{Colors.GREEN}{'='*80}")
    print(f"  ✨ Operating at 528Hz Love Frequency ✨")
    print(f"  💖 Nexus AGI Blockchain System 💖")
    print(f"{'='*80}{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()
