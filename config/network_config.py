"""
NEXUS AGI - COMPREHENSIVE NETWORK CONFIGURATION
================================================

Complete blockchain network and public API configuration
Supports all major chains, testnets, and public data sources

Networks:
- Ethereum (Mainnet, Sepolia)
- Polygon (Mainnet, Mumbai)
- Arbitrum (Mainnet)
- Avalanche (C-Chain)
- Base (Mainnet)
- Bitcoin (Mainnet, Testnet)

Public APIs:
- Price data (CoinGecko, CoinMarketCap)
- Block explorers (Etherscan, Polygonscan, etc.)
- DeFi data (Uniswap, Aave subgraphs)
- IPFS gateways
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlockchainNetwork:
    """Blockchain network configuration"""
    name: str
    rpc_urls: List[str]
    chain_id: Optional[int]
    currency: str
    network_type: str  # "evm" or "bitcoin"
    explorer_url: Optional[str] = None
    testnet: bool = False


@dataclass
class PublicAPI:
    """Public API configuration"""
    name: str
    base_url: str
    requires_auth: bool = False
    rate_limit: Optional[int] = None
    category: str = "general"


# ============================================================================
# BLOCKCHAIN NETWORKS CONFIGURATION
# ============================================================================

BLOCKCHAIN_NETWORKS = {
    # Ethereum Networks
    "ethereum_mainnet": BlockchainNetwork(
        name="Ethereum Mainnet",
        rpc_urls=[
            "https://cloudflare-eth.com",
            "https://rpc.ankr.com/eth",
            "https://eth.llamarpc.com",
            "https://1rpc.io/eth",
            "https://ethereum.publicnode.com"
        ],
        chain_id=1,
        currency="ETH",
        network_type="evm",
        explorer_url="https://etherscan.io",
        testnet=False
    ),

    "ethereum_sepolia": BlockchainNetwork(
        name="Ethereum Sepolia Testnet",
        rpc_urls=[
            "https://rpc.sepolia.org",
            "https://eth-sepolia.public.blastapi.io",
            "https://sepolia.gateway.tenderly.co"
        ],
        chain_id=11155111,
        currency="SepoliaETH",
        network_type="evm",
        explorer_url="https://sepolia.etherscan.io",
        testnet=True
    ),

    # Polygon Networks
    "polygon_mainnet": BlockchainNetwork(
        name="Polygon Mainnet",
        rpc_urls=[
            "https://polygon-rpc.com",
            "https://rpc-mainnet.maticvigil.com",
            "https://1rpc.io/matic",
            "https://polygon-mainnet.public.blastapi.io"
        ],
        chain_id=137,
        currency="MATIC",
        network_type="evm",
        explorer_url="https://polygonscan.com",
        testnet=False
    ),

    "polygon_mumbai": BlockchainNetwork(
        name="Polygon Mumbai Testnet",
        rpc_urls=[
            "https://rpc-mumbai.maticvigil.com",
            "https://polygon-testnet.public.blastapi.io"
        ],
        chain_id=80001,
        currency="MATIC",
        network_type="evm",
        explorer_url="https://mumbai.polygonscan.com",
        testnet=True
    ),

    # Arbitrum
    "arbitrum_mainnet": BlockchainNetwork(
        name="Arbitrum One",
        rpc_urls=[
            "https://arb1.arbitrum.io/rpc",
            "https://rpc.ankr.com/arbitrum",
            "https://arbitrum-one.public.blastapi.io"
        ],
        chain_id=42161,
        currency="ETH",
        network_type="evm",
        explorer_url="https://arbiscan.io",
        testnet=False
    ),

    # Avalanche
    "avalanche_mainnet": BlockchainNetwork(
        name="Avalanche C-Chain",
        rpc_urls=[
            "https://api.avax.network/ext/bc/C/rpc",
            "https://rpc.ankr.com/avalanche",
            "https://1rpc.io/avax"
        ],
        chain_id=43114,
        currency="AVAX",
        network_type="evm",
        explorer_url="https://snowtrace.io",
        testnet=False
    ),

    # Base
    "base_mainnet": BlockchainNetwork(
        name="Base Mainnet",
        rpc_urls=[
            "https://mainnet.base.org",
            "https://base.publicnode.com"
        ],
        chain_id=8453,
        currency="ETH",
        network_type="evm",
        explorer_url="https://basescan.org",
        testnet=False
    ),

    # BSC
    "bsc_mainnet": BlockchainNetwork(
        name="BNB Smart Chain",
        rpc_urls=[
            "https://bsc-dataseed.binance.org",
            "https://1rpc.io/bnb"
        ],
        chain_id=56,
        currency="BNB",
        network_type="evm",
        explorer_url="https://bscscan.com",
        testnet=False
    ),

    # Bitcoin
    "bitcoin_mainnet": BlockchainNetwork(
        name="Bitcoin Mainnet",
        rpc_urls=[
            "https://blockstream.info/api/",
            "https://mempool.space/api/"
        ],
        chain_id=None,
        currency="BTC",
        network_type="bitcoin",
        explorer_url="https://blockstream.info",
        testnet=False
    ),

    "bitcoin_testnet": BlockchainNetwork(
        name="Bitcoin Testnet",
        rpc_urls=[
            "https://blockstream.info/testnet/api/",
            "https://mempool.space/testnet/api/"
        ],
        chain_id=None,
        currency="tBTC",
        network_type="bitcoin",
        explorer_url="https://blockstream.info/testnet",
        testnet=True
    )
}


# ============================================================================
# PUBLIC APIs CONFIGURATION
# ============================================================================

PUBLIC_APIS = {
    # Cryptocurrency Price Data
    "coingecko": PublicAPI(
        name="CoinGecko",
        base_url="https://api.coingecko.com/api/v3",
        requires_auth=False,
        rate_limit=50,  # per minute
        category="price_data"
    ),

    "coinmarketcap_public": PublicAPI(
        name="CoinMarketCap Public",
        base_url="https://pro-api.coinmarketcap.com/v1",
        requires_auth=False,
        category="price_data"
    ),

    "cryptocompare": PublicAPI(
        name="CryptoCompare",
        base_url="https://min-api.cryptocompare.com/data",
        requires_auth=False,
        category="price_data"
    ),

    # Block Explorers
    "etherscan": PublicAPI(
        name="Etherscan",
        base_url="https://api.etherscan.io/api",
        requires_auth=False,
        category="explorer"
    ),

    "polygonscan": PublicAPI(
        name="Polygonscan",
        base_url="https://api.polygonscan.com/api",
        requires_auth=False,
        category="explorer"
    ),

    "bscscan": PublicAPI(
        name="BSCScan",
        base_url="https://api.bscscan.com/api",
        requires_auth=False,
        category="explorer"
    ),

    # DeFi Data
    "uniswap_v3": PublicAPI(
        name="Uniswap V3 Subgraph",
        base_url="https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
        requires_auth=False,
        category="defi"
    ),

    "aave_v3": PublicAPI(
        name="Aave V3 Subgraph",
        base_url="https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
        requires_auth=False,
        category="defi"
    ),

    # ENS
    "ens": PublicAPI(
        name="ENS Subgraph",
        base_url="https://api.thegraph.com/subgraphs/name/ensdomains/ens",
        requires_auth=False,
        category="naming"
    ),

    # GitHub
    "github": PublicAPI(
        name="GitHub API",
        base_url="https://api.github.com",
        requires_auth=False,
        rate_limit=60,  # per hour without auth
        category="development"
    ),

    # Exchange Rates
    "exchangerate": PublicAPI(
        name="ExchangeRate API",
        base_url="https://api.exchangerate-api.com/v4/latest",
        requires_auth=False,
        category="exchange_rates"
    )
}


# IPFS Gateways
IPFS_GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/"
]


# ============================================================================
# NETWORK MANAGER
# ============================================================================

class NetworkManager:
    """
    Unified network manager for all blockchain operations
    Handles RPC failover, network switching, and API access
    """

    def __init__(self):
        self.networks = BLOCKCHAIN_NETWORKS
        self.apis = PUBLIC_APIS
        self.active_rpcs = {}  # Track active RPC per network

    def get_network(self, network_name: str) -> Optional[BlockchainNetwork]:
        """Get network configuration by name"""
        return self.networks.get(network_name)

    def get_api(self, api_name: str) -> Optional[PublicAPI]:
        """Get API configuration by name"""
        return self.apis.get(api_name)

    def get_rpc_url(self, network_name: str, fallback_index: int = 0) -> Optional[str]:
        """
        Get RPC URL for network with automatic failover

        Args:
            network_name: Name of the network
            fallback_index: Index of RPC to use (for failover)

        Returns:
            RPC URL or None if network not found
        """
        network = self.get_network(network_name)
        if not network:
            return None

        if fallback_index < len(network.rpc_urls):
            return network.rpc_urls[fallback_index]

        return None

    def test_rpc_connection(self, network_name: str, rpc_index: int = 0) -> bool:
        """
        Test RPC connection

        Args:
            network_name: Name of the network
            rpc_index: Index of RPC to test

        Returns:
            True if connection successful
        """
        network = self.get_network(network_name)
        if not network or rpc_index >= len(network.rpc_urls):
            return False

        rpc_url = network.rpc_urls[rpc_index]

        try:
            if network.network_type == "evm":
                # Test EVM network with eth_blockNumber
                response = requests.post(
                    rpc_url,
                    json={
                        "jsonrpc": "2.0",
                        "method": "eth_blockNumber",
                        "params": [],
                        "id": 1
                    },
                    timeout=10
                )
                return response.status_code == 200 and "result" in response.json()

            elif network.network_type == "bitcoin":
                # Test Bitcoin API
                response = requests.get(
                    f"{rpc_url}blocks/tip/height",
                    timeout=10
                )
                return response.status_code == 200

        except Exception as e:
            logger.warning(f"RPC test failed for {network_name} [{rpc_index}]: {e}")
            return False

        return False

    def find_working_rpc(self, network_name: str) -> Optional[str]:
        """
        Find first working RPC for network

        Args:
            network_name: Name of the network

        Returns:
            Working RPC URL or None
        """
        network = self.get_network(network_name)
        if not network:
            return None

        for i in range(len(network.rpc_urls)):
            if self.test_rpc_connection(network_name, i):
                logger.info(f"✓ Found working RPC for {network_name}: {network.rpc_urls[i]}")
                self.active_rpcs[network_name] = network.rpc_urls[i]
                return network.rpc_urls[i]

        logger.error(f"✗ No working RPC found for {network_name}")
        return None

    def get_price_from_coingecko(self, coin_id: str, vs_currency: str = "usd") -> Optional[float]:
        """
        Get cryptocurrency price from CoinGecko

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
            vs_currency: Currency to compare against

        Returns:
            Price or None if failed
        """
        api = self.get_api("coingecko")
        if not api:
            return None

        try:
            url = f"{api.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": vs_currency
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get(coin_id, {}).get(vs_currency)

        except Exception as e:
            logger.error(f"Error fetching price from CoinGecko: {e}")
            return None

    def get_all_network_names(self) -> List[str]:
        """Get list of all available network names"""
        return list(self.networks.keys())

    def get_networks_by_type(self, network_type: str) -> Dict[str, BlockchainNetwork]:
        """Get all networks of specific type (evm or bitcoin)"""
        return {
            name: network
            for name, network in self.networks.items()
            if network.network_type == network_type
        }

    def get_mainnet_networks(self) -> Dict[str, BlockchainNetwork]:
        """Get all mainnet networks"""
        return {
            name: network
            for name, network in self.networks.items()
            if not network.testnet
        }

    def get_testnet_networks(self) -> Dict[str, BlockchainNetwork]:
        """Get all testnet networks"""
        return {
            name: network
            for name, network in self.networks.items()
            if network.testnet
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_network_manager() -> NetworkManager:
    """Get singleton network manager instance"""
    global _network_manager
    if '_network_manager' not in globals():
        _network_manager = NetworkManager()
    return _network_manager


def test_all_networks() -> Dict[str, bool]:
    """
    Test all configured networks

    Returns:
        Dictionary mapping network names to connection status
    """
    manager = get_network_manager()
    results = {}

    print("=" * 80)
    print("TESTING ALL BLOCKCHAIN NETWORKS")
    print("=" * 80)

    for network_name in manager.get_all_network_names():
        network = manager.get_network(network_name)
        print(f"\n🔍 Testing {network.name}...")

        working_rpc = manager.find_working_rpc(network_name)
        results[network_name] = working_rpc is not None

        if working_rpc:
            print(f"   ✅ Connected: {working_rpc}")
        else:
            print(f"   ❌ Failed to connect")

    print("\n" + "=" * 80)
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"✅ {successful}/{total} networks accessible")
    print("=" * 80)

    return results


def get_network_info(network_name: str) -> Dict[str, Any]:
    """Get detailed information about a network"""
    manager = get_network_manager()
    network = manager.get_network(network_name)

    if not network:
        return {"error": "Network not found"}

    return {
        "name": network.name,
        "chain_id": network.chain_id,
        "currency": network.currency,
        "type": network.network_type,
        "testnet": network.testnet,
        "explorer": network.explorer_url,
        "rpc_count": len(network.rpc_urls),
        "rpc_urls": network.rpc_urls
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("NEXUS AGI - COMPREHENSIVE NETWORK CONFIGURATION")
    print("=" * 80)

    manager = get_network_manager()

    print(f"\n📊 Configuration Summary:")
    print(f"   Total Networks: {len(manager.networks)}")
    print(f"   EVM Networks: {len(manager.get_networks_by_type('evm'))}")
    print(f"   Bitcoin Networks: {len(manager.get_networks_by_type('bitcoin'))}")
    print(f"   Mainnets: {len(manager.get_mainnet_networks())}")
    print(f"   Testnets: {len(manager.get_testnet_networks())}")
    print(f"   Public APIs: {len(manager.apis)}")
    print(f"   IPFS Gateways: {len(IPFS_GATEWAYS)}")

    print(f"\n📡 Available Networks:")
    for name, network in manager.networks.items():
        testnet_badge = "🧪" if network.testnet else "🌐"
        print(f"   {testnet_badge} {network.name} ({network.currency}) - {len(network.rpc_urls)} RPCs")

    print(f"\n🔗 Public APIs:")
    for name, api in manager.apis.items():
        auth_badge = "🔒" if api.requires_auth else "🔓"
        print(f"   {auth_badge} {api.name} - {api.category}")

    print("\n" + "=" * 80)
    print("Run test_all_networks() to test connectivity")
    print("=" * 80 + "\n")
