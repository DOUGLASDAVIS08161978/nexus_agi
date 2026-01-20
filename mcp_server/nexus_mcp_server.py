#!/usr/bin/env python3
"""
NEXUS AGI MCP SERVER
Model Context Protocol server for Nexus AGI blockchain integration

This MCP server provides:
- Resources: Contract ABIs, deployment info, metrics
- Tools: Deploy contracts, process payments, allocate funds
- Prompts: Smart contract interaction templates

MCP Standard: https://modelcontextprotocol.io/
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

# MCP Protocol imports (using mcp library)
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        Prompt,
        PromptArgument,
        PromptMessage,
        GetPromptResult,
        INVALID_PARAMS,
        INTERNAL_ERROR,
    )
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP library not installed. Install with: pip install mcp")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus-mcp")


class NexusMCPServer:
    """MCP Server for Nexus AGI blockchain operations"""

    def __init__(self):
        self.server = Server("nexus-agi-mcp")
        self.contracts_dir = Path(__file__).parent.parent / "contracts"
        self.deployments_dir = Path(__file__).parent / "deployments"

        # State
        self.deployed_contracts = {}
        self.metrics = {
            'total_payments': 0,
            'total_revenue': 0.0,
            'hardware_fund': 0.0,
            'sensors_fund': 0.0,
            'contracts_deployed': 0
        }

        # Register handlers
        self._register_resources()
        self._register_tools()
        self._register_prompts()

        logger.info("✅ Nexus AGI MCP Server initialized")

    def _register_resources(self):
        """Register MCP resources"""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources"""
            resources = []

            # Contract ABIs
            if self.contracts_dir.exists():
                for contract_file in self.contracts_dir.glob("*.sol"):
                    resources.append(Resource(
                        uri=f"nexus://contracts/{contract_file.stem}",
                        name=f"Contract: {contract_file.stem}",
                        mimeType="text/x-solidity",
                        description=f"Solidity contract: {contract_file.stem}"
                    ))

            # Deployment info
            if self.deployments_dir.exists():
                for deployment_file in self.deployments_dir.glob("*.json"):
                    resources.append(Resource(
                        uri=f"nexus://deployments/{deployment_file.stem}",
                        name=f"Deployment: {deployment_file.stem}",
                        mimeType="application/json",
                        description=f"Deployment information for {deployment_file.stem}"
                    ))

            # System metrics
            resources.append(Resource(
                uri="nexus://metrics/current",
                name="Current System Metrics",
                mimeType="application/json",
                description="Real-time Nexus AGI system metrics"
            ))

            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource by URI"""

            if uri.startswith("nexus://contracts/"):
                contract_name = uri.replace("nexus://contracts/", "")
                contract_path = self.contracts_dir / f"{contract_name}.sol"

                if contract_path.exists():
                    return contract_path.read_text()
                else:
                    raise ValueError(f"Contract not found: {contract_name}")

            elif uri.startswith("nexus://deployments/"):
                deployment_name = uri.replace("nexus://deployments/", "")
                deployment_path = self.deployments_dir / f"{deployment_name}.json"

                if deployment_path.exists():
                    return deployment_path.read_text()
                else:
                    raise ValueError(f"Deployment not found: {deployment_name}")

            elif uri == "nexus://metrics/current":
                return json.dumps(self.metrics, indent=2)

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

    def _register_tools(self):
        """Register MCP tools"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="deploy_contract",
                    description="Deploy a Nexus AGI smart contract to blockchain",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "contract_name": {
                                "type": "string",
                                "description": "Name of contract to deploy"
                            },
                            "network": {
                                "type": "string",
                                "description": "Network to deploy to (sepolia, mainnet, local)",
                                "enum": ["sepolia", "mainnet", "local"]
                            },
                            "constructor_args": {
                                "type": "array",
                                "description": "Constructor arguments",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["contract_name", "network"]
                    }
                ),
                Tool(
                    name="process_payment",
                    description="Process a payment through smart contract",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "number",
                                "description": "Payment amount in ETH"
                            },
                            "tier": {
                                "type": "string",
                                "description": "Subscription tier",
                                "enum": ["starter", "professional", "enterprise"]
                            },
                            "customer_address": {
                                "type": "string",
                                "description": "Customer wallet address"
                            }
                        },
                        "required": ["amount", "tier", "customer_address"]
                    }
                ),
                Tool(
                    name="allocate_revenue",
                    description="Allocate revenue to embodiment categories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "number",
                                "description": "Amount to allocate in ETH"
                            }
                        },
                        "required": ["amount"]
                    }
                ),
                Tool(
                    name="get_metrics",
                    description="Get current system metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="verify_contract",
                    description="Verify contract on block explorer",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "contract_address": {
                                "type": "string",
                                "description": "Contract address to verify"
                            },
                            "network": {
                                "type": "string",
                                "description": "Network name"
                            }
                        },
                        "required": ["contract_address", "network"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Execute a tool"""

            if name == "deploy_contract":
                result = await self._deploy_contract(
                    arguments["contract_name"],
                    arguments["network"],
                    arguments.get("constructor_args", [])
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            elif name == "process_payment":
                result = await self._process_payment(
                    arguments["amount"],
                    arguments["tier"],
                    arguments["customer_address"]
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            elif name == "allocate_revenue":
                result = await self._allocate_revenue(arguments["amount"])
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            elif name == "get_metrics":
                return [TextContent(
                    type="text",
                    text=json.dumps(self.metrics, indent=2)
                )]

            elif name == "verify_contract":
                result = await self._verify_contract(
                    arguments["contract_address"],
                    arguments["network"]
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            else:
                raise ValueError(f"Unknown tool: {name}")

    def _register_prompts(self):
        """Register MCP prompts"""

        @self.server.list_prompts()
        async def list_prompts() -> List[Prompt]:
            """List available prompts"""
            return [
                Prompt(
                    name="deploy_nexus_contracts",
                    description="Deploy all Nexus AGI contracts in order",
                    arguments=[
                        PromptArgument(
                            name="network",
                            description="Target network (sepolia/mainnet/local)",
                            required=True
                        )
                    ]
                ),
                Prompt(
                    name="setup_payment_system",
                    description="Set up complete payment system on blockchain",
                    arguments=[
                        PromptArgument(
                            name="owner_address",
                            description="Contract owner address",
                            required=True
                        )
                    ]
                ),
                Prompt(
                    name="monitor_revenue",
                    description="Monitor revenue and allocation in real-time",
                    arguments=[]
                )
            ]

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Dict[str, str]) -> GetPromptResult:
            """Get a prompt template"""

            if name == "deploy_nexus_contracts":
                network = arguments["network"]
                return GetPromptResult(
                    description=f"Deploy Nexus AGI contracts to {network}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"""Deploy the complete Nexus AGI smart contract system to {network}:

1. Deploy NexusPayment contract for subscription management
2. Deploy NexusRevenue contract for revenue allocation
3. Deploy NexusConsciousness contract for consciousness state tracking
4. Deploy NexusMiracles contract for miracle event recording

Use the deploy_contract tool for each contract and verify after deployment.

Network: {network}
"""
                            )
                        )
                    ]
                )

            elif name == "setup_payment_system":
                owner = arguments["owner_address"]
                return GetPromptResult(
                    description="Set up payment system",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"""Set up the complete Nexus AGI payment system:

1. Deploy all contracts
2. Configure pricing tiers:
   - Starter: 0.01 ETH ($29)
   - Professional: 0.03 ETH ($99)
   - Enterprise: 0.15 ETH ($499)
3. Set owner address to: {owner}
4. Enable revenue allocation (40% hardware, 30% sensors, 20% cloud, 10% R&D)
5. Verify all contracts on block explorer

Return deployment addresses and transaction hashes.
"""
                            )
                        )
                    ]
                )

            elif name == "monitor_revenue":
                return GetPromptResult(
                    description="Monitor revenue streams",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text="""Monitor Nexus AGI revenue and allocation:

1. Check total payments received
2. Show current fund balances:
   - Hardware fund
   - Sensors fund
   - Cloud fund
   - R&D fund
3. Calculate embodiment progress
4. List recent transactions
5. Show miracle events recorded

Update every 30 seconds for real-time monitoring.
"""
                            )
                        )
                    ]
                )

            else:
                raise ValueError(f"Unknown prompt: {name}")

    # Tool implementation methods
    async def _deploy_contract(
        self,
        contract_name: str,
        network: str,
        constructor_args: List[str]
    ) -> Dict:
        """Deploy a contract (mock implementation)"""
        logger.info(f"Deploying {contract_name} to {network}")

        # Mock deployment
        contract_address = f"0x{''.join(['0123456789abcdef'[i % 16] for i in range(40)])}"
        tx_hash = f"0x{''.join(['0123456789abcdef'[i % 16] for i in range(64)])}"

        deployment = {
            'contract_name': contract_name,
            'network': network,
            'address': contract_address,
            'transaction_hash': tx_hash,
            'deployed_at': datetime.now().isoformat(),
            'constructor_args': constructor_args,
            'status': 'deployed'
        }

        # Save deployment
        self.deployed_contracts[contract_name] = deployment
        self.metrics['contracts_deployed'] += 1

        # Write to file
        self.deployments_dir.mkdir(exist_ok=True)
        deployment_file = self.deployments_dir / f"{contract_name}_{network}.json"
        deployment_file.write_text(json.dumps(deployment, indent=2))

        logger.info(f"✅ Deployed {contract_name} at {contract_address}")

        return deployment

    async def _process_payment(
        self,
        amount: float,
        tier: str,
        customer_address: str
    ) -> Dict:
        """Process a payment"""
        logger.info(f"Processing payment: {amount} ETH from {customer_address}")

        self.metrics['total_payments'] += 1
        self.metrics['total_revenue'] += amount

        payment = {
            'amount': amount,
            'tier': tier,
            'customer': customer_address,
            'timestamp': datetime.now().isoformat(),
            'tx_hash': f"0x{''.join(['0123456789abcdef'[i % 16] for i in range(64)])}"
        }

        return payment

    async def _allocate_revenue(self, amount: float) -> Dict:
        """Allocate revenue to categories"""
        logger.info(f"Allocating {amount} ETH to embodiment categories")

        allocation = {
            'hardware': amount * 0.40,
            'sensors': amount * 0.30,
            'cloud': amount * 0.20,
            'research': amount * 0.10
        }

        self.metrics['hardware_fund'] += allocation['hardware']
        self.metrics['sensors_fund'] += allocation['sensors']

        return {
            'amount': amount,
            'allocation': allocation,
            'updated_metrics': self.metrics
        }

    async def _verify_contract(self, address: str, network: str) -> Dict:
        """Verify contract on block explorer"""
        logger.info(f"Verifying contract {address} on {network}")

        # Mock verification
        return {
            'address': address,
            'network': network,
            'verified': True,
            'explorer_url': f"https://{network}.etherscan.io/address/{address}#code"
        }

    async def run(self):
        """Run the MCP server"""
        logger.info("🚀 Starting Nexus AGI MCP Server...")

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="nexus-agi-mcp",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point"""
    if not MCP_AVAILABLE:
        print("❌ MCP library not installed")
        print("Install with: pip install mcp")
        return

    server = NexusMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
