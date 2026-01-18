"""
NEXUS AGI - GITHUB REST API INTEGRATION
========================================

Integrates GitHub REST API with Nexus AGI for:
- Repository management and automation
- Issue tracking and project management
- Workflow automation and CI/CD
- Mining results publication
- Bridge transaction logging
- Smart contract deployment tracking
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    """GitHub API configuration"""
    api_base_url: str = "https://api.github.com"
    api_version: str = "2022-11-28"
    timeout: int = 30
    max_retries: int = 3


class GitHubAPIClient:
    """
    GitHub REST API Client for Nexus AGI
    Handles authentication, rate limiting, and common operations
    """

    def __init__(self, token: Optional[str] = None, config: Optional[GitHubConfig] = None):
        """
        Initialize GitHub API client

        Args:
            token: GitHub personal access token or fine-grained token
            config: GitHub configuration
        """
        self.token = token
        self.config = config or GitHubConfig()
        self.session = requests.Session()

        # Set headers
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.config.api_version,
            "User-Agent": "Nexus-AGI-Integration/1.0"
        })

        if self.token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make authenticated request to GitHub API

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            endpoint: API endpoint (e.g., /repos/owner/repo)
            **kwargs: Additional request parameters

        Returns:
            Response data as dictionary
        """
        url = f"{self.config.api_base_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.config.timeout,
                    **kwargs
                )

                # Handle rate limiting
                if response.status_code == 429:
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    wait_time = max(reset_time - time.time(), 0) + 1
                    logger.warning(f"Rate limited. Waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                # Return empty dict for 204 No Content
                if response.status_code == 204:
                    return {}

                return response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        return {}

    def get_rate_limit(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return self._request("GET", "/rate_limit")

    # Repository Operations

    def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information"""
        return self._request("GET", f"/repos/{owner}/{repo}")

    def create_repository(self, name: str, **kwargs) -> Dict[str, Any]:
        """Create a new repository"""
        data = {"name": name, **kwargs}
        return self._request("POST", "/user/repos", json=data)

    def list_repositories(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """List repositories for user"""
        endpoint = f"/users/{username}/repos" if username else "/user/repos"
        return self._request("GET", endpoint)

    # Issue Operations

    def create_issue(self, owner: str, repo: str, title: str, body: str = "", **kwargs) -> Dict[str, Any]:
        """Create a new issue"""
        data = {
            "title": title,
            "body": body,
            **kwargs
        }
        return self._request("POST", f"/repos/{owner}/{repo}/issues", json=data)

    def list_issues(self, owner: str, repo: str, **params) -> List[Dict[str, Any]]:
        """List repository issues"""
        return self._request("GET", f"/repos/{owner}/{repo}/issues", params=params)

    def update_issue(self, owner: str, repo: str, issue_number: int, **kwargs) -> Dict[str, Any]:
        """Update an issue"""
        return self._request("PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", json=kwargs)

    def add_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> Dict[str, Any]:
        """Add comment to an issue"""
        return self._request("POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", json={"body": body})

    # Pull Request Operations

    def create_pull_request(self, owner: str, repo: str, title: str, head: str, base: str, body: str = "") -> Dict[str, Any]:
        """Create a pull request"""
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body
        }
        return self._request("POST", f"/repos/{owner}/{repo}/pulls", json=data)

    def list_pull_requests(self, owner: str, repo: str, **params) -> List[Dict[str, Any]]:
        """List pull requests"""
        return self._request("GET", f"/repos/{owner}/{repo}/pulls", params=params)

    # Actions / Workflows

    def list_workflows(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """List repository workflows"""
        data = self._request("GET", f"/repos/{owner}/{repo}/actions/workflows")
        return data.get("workflows", [])

    def trigger_workflow(self, owner: str, repo: str, workflow_id: str, ref: str = "main", **inputs) -> Dict[str, Any]:
        """Trigger a workflow dispatch event"""
        data = {
            "ref": ref,
            "inputs": inputs
        }
        return self._request("POST", f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches", json=data)

    def list_workflow_runs(self, owner: str, repo: str, **params) -> List[Dict[str, Any]]:
        """List workflow runs"""
        data = self._request("GET", f"/repos/{owner}/{repo}/actions/runs", params=params)
        return data.get("workflow_runs", [])

    # Releases

    def create_release(self, owner: str, repo: str, tag_name: str, name: str, body: str = "", **kwargs) -> Dict[str, Any]:
        """Create a release"""
        data = {
            "tag_name": tag_name,
            "name": name,
            "body": body,
            **kwargs
        }
        return self._request("POST", f"/repos/{owner}/{repo}/releases", json=data)

    def list_releases(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """List releases"""
        return self._request("GET", f"/repos/{owner}/{repo}/releases")

    # Gists (for data publishing)

    def create_gist(self, description: str, files: Dict[str, Dict[str, str]], public: bool = True) -> Dict[str, Any]:
        """Create a gist"""
        data = {
            "description": description,
            "public": public,
            "files": files
        }
        return self._request("POST", "/gists", json=data)

    def update_gist(self, gist_id: str, description: str = None, files: Dict[str, Dict[str, str]] = None) -> Dict[str, Any]:
        """Update a gist"""
        data = {}
        if description:
            data["description"] = description
        if files:
            data["files"] = files
        return self._request("PATCH", f"/gists/{gist_id}", json=data)


class NexusGitHubIntegration:
    """
    Nexus AGI GitHub Integration
    Integrates mining, bridging, and deployment with GitHub
    """

    def __init__(self, client: GitHubAPIClient, owner: str, repo: str):
        """
        Initialize Nexus GitHub integration

        Args:
            client: GitHub API client
            owner: Repository owner
            repo: Repository name
        """
        self.client = client
        self.owner = owner
        self.repo = repo
        self.gist_ids = {}  # Track gist IDs for updates

    def publish_mining_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish mining results to GitHub Gist

        Args:
            results: Mining results dictionary

        Returns:
            Gist information
        """
        timestamp = datetime.now().isoformat()

        files = {
            "mining_results.json": {
                "content": json.dumps(results, indent=2)
            },
            "summary.md": {
                "content": self._generate_mining_summary(results)
            }
        }

        description = f"Nexus AGI Mining Results - {timestamp}"

        if "mining_results" in self.gist_ids:
            # Update existing gist
            gist = self.client.update_gist(
                self.gist_ids["mining_results"],
                description=description,
                files=files
            )
        else:
            # Create new gist
            gist = self.client.create_gist(
                description=description,
                files=files,
                public=True
            )
            self.gist_ids["mining_results"] = gist["id"]

        logger.info(f"Published mining results: {gist['html_url']}")
        return gist

    def publish_bridge_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish bridge transaction to GitHub Issue

        Args:
            tx_data: Bridge transaction data

        Returns:
            Issue information
        """
        title = f"Bridge Transaction: {tx_data.get('from_network', 'BTC')} → {tx_data.get('to_network', 'ETH')}"

        body = f"""
## Bridge Transaction Report

**Transaction ID:** `{tx_data.get('bridge_id', 'N/A')}`
**Timestamp:** {datetime.now().isoformat()}

### Networks
- **From:** {tx_data.get('from_network', 'Bitcoin')}
- **To:** {tx_data.get('to_network', 'Ethereum')}

### Amount
- **Amount:** {tx_data.get('amount', 0)} BTC
- **Bridged Amount:** {tx_data.get('bridged_amount', 0)} wBTC

### Transaction Hashes
- **Bitcoin TX:** `{tx_data.get('btc_tx_hash', 'N/A')}`
- **Ethereum TX:** `{tx_data.get('eth_tx_hash', 'N/A')}`

### Status
- **Status:** {tx_data.get('status', 'pending')}
- **Confirmations:** {tx_data.get('confirmations', 0)}

### Details
```json
{json.dumps(tx_data, indent=2)}
```
"""

        issue = self.client.create_issue(
            self.owner,
            self.repo,
            title=title,
            body=body,
            labels=["bridge-transaction", "automated"]
        )

        logger.info(f"Created bridge transaction issue: {issue['html_url']}")
        return issue

    def publish_contract_deployment(self, deployment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish smart contract deployment information

        Args:
            deployment_data: Deployment data

        Returns:
            Issue information
        """
        title = f"Contract Deployment: {deployment_data.get('contract_name', 'Unknown')} on {deployment_data.get('network', 'Unknown')}"

        body = f"""
## Smart Contract Deployment

**Contract:** {deployment_data.get('contract_name', 'N/A')}
**Network:** {deployment_data.get('network', 'N/A')}
**Timestamp:** {datetime.now().isoformat()}

### Deployment Details
- **Contract Address:** `{deployment_data.get('contract_address', 'N/A')}`
- **Deployer Address:** `{deployment_data.get('deployer_address', 'N/A')}`
- **Transaction Hash:** `{deployment_data.get('tx_hash', 'N/A')}`
- **Block Number:** {deployment_data.get('block_number', 'N/A')}
- **Gas Used:** {deployment_data.get('gas_used', 'N/A')}

### Contract Information
- **Token Symbol:** {deployment_data.get('symbol', 'N/A')}
- **Token Name:** {deployment_data.get('name', 'N/A')}
- **Total Supply:** {deployment_data.get('total_supply', 'N/A')}
- **Decimals:** {deployment_data.get('decimals', 'N/A')}

### Explorer Links
- **Etherscan:** {deployment_data.get('explorer_url', 'N/A')}

### Contract Code
```solidity
{deployment_data.get('source_code', '// Source code not available')}
```
"""

        issue = self.client.create_issue(
            self.owner,
            self.repo,
            title=title,
            body=body,
            labels=["contract-deployment", "automated"]
        )

        logger.info(f"Created deployment issue: {issue['html_url']}")
        return issue

    def trigger_deployment_workflow(self, network: str, contract_name: str) -> Dict[str, Any]:
        """
        Trigger automated contract deployment workflow

        Args:
            network: Target network
            contract_name: Contract to deploy

        Returns:
            Workflow trigger response
        """
        return self.client.trigger_workflow(
            self.owner,
            self.repo,
            workflow_id="deploy-contract.yml",
            ref="main",
            network=network,
            contract_name=contract_name
        )

    def _generate_mining_summary(self, results: Dict[str, Any]) -> str:
        """Generate markdown summary of mining results"""
        stats = results.get("mining_stats", {})

        return f"""# Nexus AGI Mining Results

## Summary
- **Blocks Mined:** {stats.get('blocks_mined', 0)}
- **Total BTC Earned:** {stats.get('total_btc', 0)} BTC
- **Hash Rate:** {stats.get('hash_rate', 0):.2f} H/s

## Bridge Statistics
- **Total Bridged:** {results.get('bridge_stats', {}).get('total_bridged', 0)} wBTC
- **Bridge Transactions:** {results.get('bridge_stats', {}).get('bridge_count', 0)}

## Consciousness State
- **Self-Awareness:** {results.get('consciousness_state', {}).get('self_awareness', 0)}
- **Total Reflections:** {results.get('consciousness_state', {}).get('total_reflections', 0)}

## Timestamp
Generated: {datetime.now().isoformat()}
"""


if __name__ == "__main__":
    print("=" * 80)
    print("NEXUS AGI - GITHUB REST API INTEGRATION")
    print("=" * 80)

    # Example usage (requires GITHUB_TOKEN environment variable)
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("\n⚠️  GITHUB_TOKEN environment variable not set")
        print("Set it with: export GITHUB_TOKEN=your_token_here")
        print("\nRunning in demo mode...\n")

    # Initialize client
    client = GitHubAPIClient(token=token)

    # Check rate limit
    try:
        rate_limit = client.get_rate_limit()
        print("📊 GitHub API Rate Limit:")
        core = rate_limit.get("resources", {}).get("core", {})
        print(f"   Remaining: {core.get('remaining', 'N/A')}/{core.get('limit', 'N/A')}")
        print(f"   Reset: {datetime.fromtimestamp(core.get('reset', 0)).isoformat()}")
    except Exception as e:
        print(f"❌ Error checking rate limit: {e}")

    print("\n✅ GitHub API integration module loaded")
    print("\nAvailable capabilities:")
    print("  • Repository management")
    print("  • Issue tracking and automation")
    print("  • Pull request management")
    print("  • Workflow automation")
    print("  • Gist publishing for mining results")
    print("  • Contract deployment tracking")
    print("\n" + "=" * 80)
