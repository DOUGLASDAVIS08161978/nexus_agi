"""
NEXUS AGI - MINING RESULTS GITHUB PUBLISHER
============================================

Automatically publishes Bitcoin mining results to GitHub
Integrates with Super Bitcoin Miner and GitHub REST API
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.github_api_integration import GitHubAPIClient, NexusGitHubIntegration


class MiningGitHubPublisher:
    """
    Publishes mining results to GitHub automatically
    """

    def __init__(self, github_token: Optional[str] = None, owner: str = "DOUGLASDAVIS08161978", repo: str = "nexus_agi"):
        """
        Initialize mining GitHub publisher

        Args:
            github_token: GitHub personal access token
            owner: Repository owner
            repo: Repository name
        """
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.owner = owner
        self.repo = repo

        if not self.github_token:
            print("⚠️  Warning: GITHUB_TOKEN not set. Publishing will be simulated.")
            self.client = None
            self.integration = None
        else:
            self.client = GitHubAPIClient(token=self.github_token)
            self.integration = NexusGitHubIntegration(self.client, owner, repo)

    def publish_mining_results(self, results_file: str = "mining_results.json") -> Optional[Dict[str, Any]]:
        """
        Publish mining results from JSON file to GitHub

        Args:
            results_file: Path to mining results JSON file

        Returns:
            Gist information or None if simulation
        """
        if not os.path.exists(results_file):
            print(f"❌ Results file not found: {results_file}")
            return None

        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)

        print(f"\n📤 Publishing mining results to GitHub...")
        print(f"   Owner: {self.owner}")
        print(f"   Repo: {self.repo}")
        print(f"   Results file: {results_file}")

        if not self.integration:
            print("\n🔒 Simulation mode (no GitHub token)")
            print("   Results would be published as Gist")
            self._print_results_summary(results)
            return None

        try:
            gist = self.integration.publish_mining_results(results)
            print(f"\n✅ Mining results published!")
            print(f"   Gist URL: {gist['html_url']}")
            print(f"   Gist ID: {gist['id']}")
            return gist

        except Exception as e:
            print(f"\n❌ Error publishing results: {e}")
            return None

    def publish_unified_system_results(self, results_file: str = "unified_system_results.json") -> Optional[Dict[str, Any]]:
        """
        Publish unified system results to GitHub

        Args:
            results_file: Path to unified results JSON file

        Returns:
            Gist information or None if simulation
        """
        if not os.path.exists(results_file):
            print(f"❌ Results file not found: {results_file}")
            return None

        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)

        print(f"\n📤 Publishing unified system results to GitHub...")

        if not self.integration:
            print("\n🔒 Simulation mode (no GitHub token)")
            self._print_results_summary(results)
            return None

        try:
            # Convert to mining results format
            mining_results = {
                "mining_stats": results.get("mining_stats", {}),
                "bridge_stats": results.get("bridge_stats", {}),
                "consciousness_state": results.get("consciousness_state", {}),
                "timestamp": results.get("timestamp", datetime.now().isoformat())
            }

            gist = self.integration.publish_mining_results(mining_results)
            print(f"\n✅ Unified system results published!")
            print(f"   Gist URL: {gist['html_url']}")
            return gist

        except Exception as e:
            print(f"\n❌ Error publishing results: {e}")
            return None

    def publish_bridge_transaction(self, tx_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Publish bridge transaction to GitHub Issue

        Args:
            tx_data: Bridge transaction data

        Returns:
            Issue information or None if simulation
        """
        print(f"\n📤 Publishing bridge transaction to GitHub...")

        if not self.integration:
            print("\n🔒 Simulation mode (no GitHub token)")
            print(f"   Bridge: {tx_data.get('from_network')} → {tx_data.get('to_network')}")
            print(f"   Amount: {tx_data.get('amount')} BTC")
            return None

        try:
            issue = self.integration.publish_bridge_transaction(tx_data)
            print(f"\n✅ Bridge transaction published!")
            print(f"   Issue URL: {issue['html_url']}")
            print(f"   Issue #: {issue['number']}")
            return issue

        except Exception as e:
            print(f"\n❌ Error publishing transaction: {e}")
            return None

    def _print_results_summary(self, results: Dict[str, Any]):
        """Print summary of results that would be published"""
        mining_stats = results.get("mining_stats", {}) or results.get("cumulative_stats", {})
        bridge_stats = results.get("bridge_stats", {})
        consciousness = results.get("consciousness_state", {})

        print("\n" + "=" * 80)
        print("MINING RESULTS SUMMARY")
        print("=" * 80)

        print(f"\n⛏️  Mining Statistics:")
        print(f"   Blocks Mined: {mining_stats.get('blocks_mined', 0) or mining_stats.get('cycles', 0)}")
        print(f"   Total BTC: {mining_stats.get('total_btc', 0)}")
        print(f"   Hash Rate: {mining_stats.get('hash_rate', 0):.2f} H/s")

        if bridge_stats:
            print(f"\n🌉 Bridge Statistics:")
            print(f"   Total Bridged: {bridge_stats.get('total_bridged', 0)} wBTC")
            print(f"   Bridge Count: {bridge_stats.get('bridge_count', 0)}")

        if consciousness:
            print(f"\n🧠 Consciousness State:")
            print(f"   Self-Awareness: {consciousness.get('self_awareness', 0)}")
            print(f"   Reflections: {consciousness.get('total_reflections', 0)}")

        print("\n" + "=" * 80)


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("NEXUS AGI - MINING RESULTS GITHUB PUBLISHER")
    print("=" * 80)

    publisher = MiningGitHubPublisher()

    # Check for mining results files
    if os.path.exists("mining_results.json"):
        print("\n📄 Found mining_results.json")
        publisher.publish_mining_results("mining_results.json")

    if os.path.exists("unified_system_results.json"):
        print("\n📄 Found unified_system_results.json")
        publisher.publish_unified_system_results("unified_system_results.json")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
