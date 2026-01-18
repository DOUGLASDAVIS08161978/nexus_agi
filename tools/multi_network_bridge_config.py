"""
NEXUS AGI - MULTI-NETWORK BRIDGE CONFIGURATION
===============================================

Comprehensive bridge configuration supporting:
- Ethereum L2s: Arbitrum, Optimism, Base
- Alternative L1s: Avalanche, BNB Chain
- Cross-chain bridges for Bitcoin and wrapped assets

Integrated with Nexus AGI Directory for intelligent routing
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Network configuration"""
    name: str
    chain_id: int
    network_type: str  # "ethereum", "l2", "alternative"
    rpc_urls: List[str]
    block_explorer: str
    native_token: str
    average_block_time: int  # in seconds


@dataclass
class TokenContract:
    """Token contract configuration"""
    symbol: str
    name: str
    decimals: int
    addresses: Dict[str, str]  # network_name -> contract_address


@dataclass
class BridgeRoute:
    """Bridge route configuration"""
    from_network: str
    to_network: str
    token: str
    bridge_contract: str
    operator_type: str
    estimated_time: int  # in minutes
    estimated_fee_usd: float


# ============================================================================
# NETWORK CONFIGURATIONS
# ============================================================================

NETWORKS = {
    # Ethereum Networks
    "ethereum": NetworkConfig(
        name="Ethereum Mainnet",
        chain_id=1,
        network_type="ethereum",
        rpc_urls=[
            "https://eth.llamarpc.com",
            "https://rpc.ankr.com/eth",
            "https://ethereum.publicnode.com"
        ],
        block_explorer="https://etherscan.io",
        native_token="ETH",
        average_block_time=12
    ),
    "sepolia": NetworkConfig(
        name="Ethereum Sepolia Testnet",
        chain_id=11155111,
        network_type="ethereum",
        rpc_urls=[
            "https://rpc.sepolia.org",
            "https://rpc2.sepolia.org",
            "https://ethereum-sepolia.publicnode.com"
        ],
        block_explorer="https://sepolia.etherscan.io",
        native_token="ETH",
        average_block_time=12
    ),

    # Layer 2 Networks - Arbitrum
    "arbitrum": NetworkConfig(
        name="Arbitrum One",
        chain_id=42161,
        network_type="l2",
        rpc_urls=[
            "https://arb1.arbitrum.io/rpc",
            "https://arbitrum.llamarpc.com",
            "https://rpc.ankr.com/arbitrum"
        ],
        block_explorer="https://arbiscan.io",
        native_token="ETH",
        average_block_time=1
    ),
    "arbitrum-sepolia": NetworkConfig(
        name="Arbitrum Sepolia Testnet",
        chain_id=421614,
        network_type="l2",
        rpc_urls=[
            "https://sepolia-rollup.arbitrum.io/rpc",
            "https://arbitrum-sepolia.publicnode.com"
        ],
        block_explorer="https://sepolia.arbiscan.io",
        native_token="ETH",
        average_block_time=1
    ),

    # Layer 2 Networks - Optimism
    "optimism": NetworkConfig(
        name="Optimism Mainnet",
        chain_id=10,
        network_type="l2",
        rpc_urls=[
            "https://mainnet.optimism.io",
            "https://optimism.llamarpc.com",
            "https://rpc.ankr.com/optimism"
        ],
        block_explorer="https://optimistic.etherscan.io",
        native_token="ETH",
        average_block_time=2
    ),
    "optimism-sepolia": NetworkConfig(
        name="Optimism Sepolia Testnet",
        chain_id=11155420,
        network_type="l2",
        rpc_urls=[
            "https://sepolia.optimism.io",
            "https://optimism-sepolia.publicnode.com"
        ],
        block_explorer="https://sepolia-optimism.etherscan.io",
        native_token="ETH",
        average_block_time=2
    ),

    # Layer 2 Networks - Base
    "base": NetworkConfig(
        name="Base Mainnet",
        chain_id=8453,
        network_type="l2",
        rpc_urls=[
            "https://mainnet.base.org",
            "https://base.llamarpc.com",
            "https://rpc.ankr.com/base"
        ],
        block_explorer="https://basescan.org",
        native_token="ETH",
        average_block_time=2
    ),
    "base-sepolia": NetworkConfig(
        name="Base Sepolia Testnet",
        chain_id=84532,
        network_type="l2",
        rpc_urls=[
            "https://sepolia.base.org",
            "https://base-sepolia.publicnode.com"
        ],
        block_explorer="https://sepolia.basescan.org",
        native_token="ETH",
        average_block_time=2
    ),

    # Alternative L1 - Polygon
    "polygon": NetworkConfig(
        name="Polygon Mainnet",
        chain_id=137,
        network_type="alternative",
        rpc_urls=[
            "https://polygon-rpc.com",
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com"
        ],
        block_explorer="https://polygonscan.com",
        native_token="MATIC",
        average_block_time=2
    ),

    # Alternative L1 - Avalanche
    "avalanche": NetworkConfig(
        name="Avalanche C-Chain",
        chain_id=43114,
        network_type="alternative",
        rpc_urls=[
            "https://api.avax.network/ext/bc/C/rpc",
            "https://avalanche.public-rpc.com",
            "https://rpc.ankr.com/avalanche"
        ],
        block_explorer="https://snowtrace.io",
        native_token="AVAX",
        average_block_time=2
    ),
    "avalanche-fuji": NetworkConfig(
        name="Avalanche Fuji Testnet",
        chain_id=43113,
        network_type="alternative",
        rpc_urls=[
            "https://api.avax-test.network/ext/bc/C/rpc",
            "https://avalanche-fuji-c-chain.publicnode.com"
        ],
        block_explorer="https://testnet.snowtrace.io",
        native_token="AVAX",
        average_block_time=2
    ),

    # Alternative L1 - BNB Chain
    "bsc": NetworkConfig(
        name="BNB Smart Chain",
        chain_id=56,
        network_type="alternative",
        rpc_urls=[
            "https://bsc-dataseed.binance.org",
            "https://bsc.publicnode.com",
            "https://rpc.ankr.com/bsc"
        ],
        block_explorer="https://bscscan.com",
        native_token="BNB",
        average_block_time=3
    ),
    "bsc-testnet": NetworkConfig(
        name="BNB Smart Chain Testnet",
        chain_id=97,
        network_type="alternative",
        rpc_urls=[
            "https://data-seed-prebsc-1-s1.binance.org:8545",
            "https://bsc-testnet.publicnode.com"
        ],
        block_explorer="https://testnet.bscscan.com",
        native_token="BNB",
        average_block_time=3
    )
}


# ============================================================================
# TOKEN CONFIGURATIONS
# ============================================================================

TOKENS = {
    "WBTC": TokenContract(
        symbol="WBTC",
        name="Wrapped Bitcoin",
        decimals=8,
        addresses={
            "ethereum": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "arbitrum": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
            "optimism": "0x68f180fcCe6836688e9084f035309E29Bf0A2095",
            "base": "0x0555E30da8f98308EdB960aa94C0Db47230d2B9c",
            "polygon": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
            "avalanche": "0x50b7545627a5162F82A992c33b87aDc75187B218",
            "bsc": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c"
        }
    ),
    "USDC": TokenContract(
        symbol="USDC",
        name="USD Coin",
        decimals=6,
        addresses={
            "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "arbitrum": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            "optimism": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
            "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "polygon": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
            "avalanche": "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
            "bsc": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d"
        }
    ),
    "USDT": TokenContract(
        symbol="USDT",
        name="Tether USD",
        decimals=6,
        addresses={
            "ethereum": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "arbitrum": "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
            "optimism": "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",
            "base": "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2",
            "polygon": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
            "avalanche": "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7",
            "bsc": "0x55d398326f99059fF775485246999027B3197955"
        }
    )
}


# ============================================================================
# BRIDGE ROUTES
# ============================================================================

BRIDGE_ROUTES = [
    # Bitcoin to Ethereum Networks
    BridgeRoute(
        from_network="bitcoin",
        to_network="ethereum",
        token="WBTC",
        bridge_contract="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        operator_type="wbtc_dao",
        estimated_time=60,
        estimated_fee_usd=50.0
    ),

    # Ethereum to Layer 2s
    BridgeRoute(
        from_network="ethereum",
        to_network="arbitrum",
        token="WBTC",
        bridge_contract="0x0000000000000000000000000000000000000000",  # Arbitrum Bridge
        operator_type="optimistic_rollup",
        estimated_time=10,
        estimated_fee_usd=5.0
    ),
    BridgeRoute(
        from_network="ethereum",
        to_network="optimism",
        token="WBTC",
        bridge_contract="0x0000000000000000000000000000000000000000",  # Optimism Bridge
        operator_type="optimistic_rollup",
        estimated_time=20,
        estimated_fee_usd=5.0
    ),
    BridgeRoute(
        from_network="ethereum",
        to_network="base",
        token="WBTC",
        bridge_contract="0x0000000000000000000000000000000000000000",  # Base Bridge
        operator_type="optimistic_rollup",
        estimated_time=20,
        estimated_fee_usd=3.0
    ),
    BridgeRoute(
        from_network="ethereum",
        to_network="polygon",
        token="WBTC",
        bridge_contract="0xA0c68C638235ee32657e8f720a23ceC1bFc77C77",  # Polygon PoS Bridge
        operator_type="pos_bridge",
        estimated_time=30,
        estimated_fee_usd=10.0
    ),

    # Layer 2 to Layer 2 (via Ethereum)
    BridgeRoute(
        from_network="arbitrum",
        to_network="optimism",
        token="WBTC",
        bridge_contract="0x0000000000000000000000000000000000000000",
        operator_type="l2_to_l2",
        estimated_time=30,
        estimated_fee_usd=10.0
    ),
    BridgeRoute(
        from_network="arbitrum",
        to_network="base",
        token="WBTC",
        bridge_contract="0x0000000000000000000000000000000000000000",
        operator_type="l2_to_l2",
        estimated_time=30,
        estimated_fee_usd=8.0
    ),

    # Alternative L1 Bridges
    BridgeRoute(
        from_network="ethereum",
        to_network="avalanche",
        token="WBTC",
        bridge_contract="0x0000000000000000000000000000000000000000",  # Avalanche Bridge
        operator_type="external_bridge",
        estimated_time=15,
        estimated_fee_usd=20.0
    ),
    BridgeRoute(
        from_network="ethereum",
        to_network="bsc",
        token="WBTC",
        bridge_contract="0x0000000000000000000000000000000000000000",  # BSC Bridge
        operator_type="external_bridge",
        estimated_time=10,
        estimated_fee_usd=5.0
    ),

    # Bitcoin to Layer 2s (via Ethereum)
    BridgeRoute(
        from_network="bitcoin",
        to_network="arbitrum",
        token="WBTC",
        bridge_contract="0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
        operator_type="multi_hop",
        estimated_time=70,
        estimated_fee_usd=55.0
    ),
    BridgeRoute(
        from_network="bitcoin",
        to_network="optimism",
        token="WBTC",
        bridge_contract="0x68f180fcCe6836688e9084f035309E29Bf0A2095",
        operator_type="multi_hop",
        estimated_time=80,
        estimated_fee_usd=55.0
    ),
    BridgeRoute(
        from_network="bitcoin",
        to_network="base",
        token="WBTC",
        bridge_contract="0x0555E30da8f98308EdB960aa94C0Db47230d2B9c",
        operator_type="multi_hop",
        estimated_time=80,
        estimated_fee_usd=53.0
    ),
    BridgeRoute(
        from_network="bitcoin",
        to_network="polygon",
        token="WBTC",
        bridge_contract="0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
        operator_type="multi_hop",
        estimated_time=90,
        estimated_fee_usd=60.0
    ),
    BridgeRoute(
        from_network="bitcoin",
        to_network="avalanche",
        token="WBTC",
        bridge_contract="0x50b7545627a5162F82A992c33b87aDc75187B218",
        operator_type="multi_hop",
        estimated_time=75,
        estimated_fee_usd=70.0
    ),
    BridgeRoute(
        from_network="bitcoin",
        to_network="bsc",
        token="WBTC",
        bridge_contract="0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c",
        operator_type="multi_hop",
        estimated_time=70,
        estimated_fee_usd=55.0
    )
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_network(network_name: str) -> NetworkConfig:
    """Get network configuration by name"""
    if network_name not in NETWORKS:
        raise ValueError(f"Unknown network: {network_name}")
    return NETWORKS[network_name]


def get_token(token_symbol: str) -> TokenContract:
    """Get token configuration by symbol"""
    if token_symbol not in TOKENS:
        raise ValueError(f"Unknown token: {token_symbol}")
    return TOKENS[token_symbol]


def get_token_address(token_symbol: str, network_name: str) -> str:
    """Get token contract address for specific network"""
    token = get_token(token_symbol)
    if network_name not in token.addresses:
        raise ValueError(f"Token {token_symbol} not available on {network_name}")
    return token.addresses[network_name]


def find_bridge_route(from_network: str, to_network: str, token: str = "WBTC") -> BridgeRoute:
    """Find bridge route between networks"""
    for route in BRIDGE_ROUTES:
        if (route.from_network == from_network and
            route.to_network == to_network and
            route.token == token):
            return route

    raise ValueError(f"No bridge route found: {from_network} -> {to_network} ({token})")


def get_all_supported_networks() -> List[str]:
    """Get list of all supported network names"""
    return list(NETWORKS.keys())


def get_networks_supporting_token(token_symbol: str) -> List[str]:
    """Get list of networks that support a specific token"""
    token = get_token(token_symbol)
    return list(token.addresses.keys())


def estimate_bridge_cost(from_network: str, to_network: str, token: str = "WBTC") -> Dict[str, Any]:
    """Estimate bridge cost and time"""
    route = find_bridge_route(from_network, to_network, token)

    return {
        "from_network": from_network,
        "to_network": to_network,
        "token": token,
        "estimated_time_minutes": route.estimated_time,
        "estimated_fee_usd": route.estimated_fee_usd,
        "operator_type": route.operator_type,
        "bridge_contract": route.bridge_contract
    }


if __name__ == "__main__":
    # Display network support matrix
    print("=" * 80)
    print("NEXUS AGI - MULTI-NETWORK BRIDGE CONFIGURATION")
    print("=" * 80)

    print("\n📡 Supported Networks:")
    for network_name, network in NETWORKS.items():
        print(f"  • {network.name} (Chain ID: {network.chain_id})")
        print(f"    Type: {network.network_type}, Native: {network.native_token}")

    print("\n💰 Supported Tokens:")
    for token_symbol, token in TOKENS.items():
        networks = list(token.addresses.keys())
        print(f"  • {token.name} ({token_symbol})")
        print(f"    Available on: {', '.join(networks)}")

    print("\n🌉 Bridge Routes:")
    print(f"  Total routes configured: {len(BRIDGE_ROUTES)}")

    # Show sample routes
    print("\n  Sample Bitcoin bridge routes:")
    btc_routes = [r for r in BRIDGE_ROUTES if r.from_network == "bitcoin"]
    for route in btc_routes:
        print(f"    Bitcoin → {route.to_network.capitalize()}")
        print(f"      Time: ~{route.estimated_time} min, Fee: ~${route.estimated_fee_usd}")

    print("\n  Sample L2 routes:")
    l2_routes = [r for r in BRIDGE_ROUTES if r.from_network == "ethereum" and
                 NETWORKS[r.to_network].network_type == "l2"]
    for route in l2_routes:
        print(f"    Ethereum → {route.to_network.capitalize()}")
        print(f"      Time: ~{route.estimated_time} min, Fee: ~${route.estimated_fee_usd}")

    print("\n" + "=" * 80)
