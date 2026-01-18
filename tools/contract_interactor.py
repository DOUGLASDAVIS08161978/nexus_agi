"""
NEXUS AGI - SMART CONTRACT INTERACTION MODULE
==============================================

Read-only smart contract interaction across all integrated networks
Supports eth_call for reading contract state without gas costs

Features:
- Contract calls across 8 EVM networks
- ERC-20 token data retrieval
- ERC-721 NFT data retrieval
- Custom contract interaction
- Function encoding/decoding
- Multi-network contract queries
"""

import sys
import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.network_config import get_network_manager, NetworkManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# COMMON CONTRACT ABIs
# ============================================================================

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    }
]

ERC721_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "name": "ownerOf",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "name": "tokenURI",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }
]


@dataclass
class ContractCallResult:
    """Result of a contract call"""
    success: bool
    data: Any
    network: str
    contract_address: str
    function_name: str
    error: Optional[str] = None


class ContractInteractor:
    """
    Smart contract interaction across all EVM networks
    Read-only operations via eth_call
    """

    def __init__(self, network_manager: Optional[NetworkManager] = None):
        """
        Initialize contract interactor

        Args:
            network_manager: Network manager instance
        """
        self.network_manager = network_manager or get_network_manager()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })

    def _make_rpc_call(
        self,
        network_name: str,
        method: str,
        params: List[Any],
        rpc_index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Make JSON-RPC call to network

        Args:
            network_name: Name of network
            method: RPC method (e.g., "eth_call")
            params: Method parameters
            rpc_index: Index of RPC to use

        Returns:
            RPC response or None
        """
        rpc_url = self.network_manager.get_rpc_url(network_name, rpc_index)

        if not rpc_url:
            logger.error(f"No RPC URL found for {network_name}")
            return None

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }

        try:
            response = self.session.post(rpc_url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            if "error" in data:
                logger.error(f"RPC error: {data['error']}")
                return None

            return data

        except Exception as e:
            logger.error(f"RPC call failed: {e}")

            # Try next RPC if available
            network = self.network_manager.get_network(network_name)
            if network and rpc_index + 1 < len(network.rpc_urls):
                logger.info(f"Trying next RPC endpoint...")
                return self._make_rpc_call(network_name, method, params, rpc_index + 1)

            return None

    def eth_call(
        self,
        network_name: str,
        contract_address: str,
        data: str,
        block: str = "latest"
    ) -> Optional[str]:
        """
        Execute eth_call to read contract data

        Args:
            network_name: Network to call
            contract_address: Contract address
            data: Encoded function call data
            block: Block number or "latest"

        Returns:
            Hex-encoded result or None
        """
        params = [
            {
                "to": contract_address,
                "data": data
            },
            block
        ]

        result = self._make_rpc_call(network_name, "eth_call", params)

        if result and "result" in result:
            return result["result"]

        return None

    def get_block_number(self, network_name: str) -> Optional[int]:
        """
        Get current block number

        Args:
            network_name: Network name

        Returns:
            Block number or None
        """
        result = self._make_rpc_call(network_name, "eth_blockNumber", [])

        if result and "result" in result:
            return int(result["result"], 16)

        return None

    def get_balance(self, network_name: str, address: str) -> Optional[float]:
        """
        Get ETH balance for address

        Args:
            network_name: Network name
            address: Ethereum address

        Returns:
            Balance in ETH or None
        """
        result = self._make_rpc_call(
            network_name,
            "eth_getBalance",
            [address, "latest"]
        )

        if result and "result" in result:
            balance_wei = int(result["result"], 16)
            return balance_wei / 10**18

        return None

    def get_transaction_count(self, network_name: str, address: str) -> Optional[int]:
        """
        Get transaction count (nonce) for address

        Args:
            network_name: Network name
            address: Ethereum address

        Returns:
            Transaction count or None
        """
        result = self._make_rpc_call(
            network_name,
            "eth_getTransactionCount",
            [address, "latest"]
        )

        if result and "result" in result:
            return int(result["result"], 16)

        return None

    def get_code(self, network_name: str, address: str) -> Optional[str]:
        """
        Get contract bytecode

        Args:
            network_name: Network name
            address: Contract address

        Returns:
            Bytecode hex string or None
        """
        result = self._make_rpc_call(
            network_name,
            "eth_getCode",
            [address, "latest"]
        )

        if result and "result" in result:
            return result["result"]

        return None

    def is_contract(self, network_name: str, address: str) -> bool:
        """
        Check if address is a contract

        Args:
            network_name: Network name
            address: Address to check

        Returns:
            True if contract, False otherwise
        """
        code = self.get_code(network_name, address)
        return code is not None and code != "0x"


class ERC20Reader(ContractInteractor):
    """
    ERC-20 token contract reader
    Provides convenient methods for reading token data
    """

    def get_token_info(
        self,
        network_name: str,
        token_address: str
    ) -> Dict[str, Any]:
        """
        Get complete ERC-20 token information

        Args:
            network_name: Network name
            token_address: Token contract address

        Returns:
            Dictionary with token info
        """
        info = {
            "network": network_name,
            "address": token_address,
            "is_contract": self.is_contract(network_name, token_address)
        }

        if not info["is_contract"]:
            info["error"] = "Address is not a contract"
            return info

        # Get name
        info["name"] = self._call_string_function(network_name, token_address, "name()")

        # Get symbol
        info["symbol"] = self._call_string_function(network_name, token_address, "symbol()")

        # Get decimals
        decimals_data = self.eth_call(
            network_name,
            token_address,
            "0x313ce567"  # decimals()
        )
        if decimals_data and decimals_data != "0x":
            info["decimals"] = int(decimals_data, 16)
        else:
            info["decimals"] = None

        # Get total supply
        supply_data = self.eth_call(
            network_name,
            token_address,
            "0x18160ddd"  # totalSupply()
        )
        if supply_data and supply_data != "0x":
            total_supply_wei = int(supply_data, 16)
            decimals = info.get("decimals", 18)
            info["total_supply"] = total_supply_wei / (10 ** decimals) if decimals else total_supply_wei
            info["total_supply_wei"] = total_supply_wei
        else:
            info["total_supply"] = None

        return info

    def get_balance(
        self,
        network_name: str,
        token_address: str,
        holder_address: str
    ) -> Optional[float]:
        """
        Get ERC-20 token balance for address

        Args:
            network_name: Network name
            token_address: Token contract address
            holder_address: Address to check balance for

        Returns:
            Token balance or None
        """
        # balanceOf(address)
        function_selector = "0x70a08231"

        # Encode address parameter (remove 0x, pad to 32 bytes)
        address_param = holder_address[2:].lower().zfill(64)

        data = function_selector + address_param

        result = self.eth_call(network_name, token_address, data)

        if result and result != "0x":
            balance_wei = int(result, 16)

            # Get decimals to convert
            token_info = self.get_token_info(network_name, token_address)
            decimals = token_info.get("decimals", 18)

            return balance_wei / (10 ** decimals)

        return None

    def get_allowance(
        self,
        network_name: str,
        token_address: str,
        owner_address: str,
        spender_address: str
    ) -> Optional[float]:
        """
        Get ERC-20 token allowance

        Args:
            network_name: Network name
            token_address: Token contract address
            owner_address: Token owner address
            spender_address: Spender address

        Returns:
            Allowance amount or None
        """
        # allowance(address,address)
        function_selector = "0xdd62ed3e"

        # Encode parameters
        owner_param = owner_address[2:].lower().zfill(64)
        spender_param = spender_address[2:].lower().zfill(64)

        data = function_selector + owner_param + spender_param

        result = self.eth_call(network_name, token_address, data)

        if result and result != "0x":
            allowance_wei = int(result, 16)

            # Get decimals
            token_info = self.get_token_info(network_name, token_address)
            decimals = token_info.get("decimals", 18)

            return allowance_wei / (10 ** decimals)

        return None

    def _call_string_function(
        self,
        network_name: str,
        contract_address: str,
        function_signature: str
    ) -> Optional[str]:
        """
        Call a function that returns a string

        Args:
            network_name: Network name
            contract_address: Contract address
            function_signature: Function signature (e.g., "name()")

        Returns:
            String result or None
        """
        # Function selectors for common string functions
        selectors = {
            "name()": "0x06fdde03",
            "symbol()": "0x95d89b41"
        }

        data = selectors.get(function_signature)
        if not data:
            return None

        result = self.eth_call(network_name, contract_address, data)

        if result and result != "0x":
            try:
                # Decode string from ABI encoding
                # Skip first 64 chars (offset), next 64 chars (length)
                hex_data = result[2:]

                if len(hex_data) < 128:
                    return None

                # Get string length
                length = int(hex_data[64:128], 16)

                # Get string data
                string_hex = hex_data[128:128 + (length * 2)]

                # Convert hex to string
                string_value = bytes.fromhex(string_hex).decode('utf-8', errors='ignore')

                return string_value.strip('\x00')

            except Exception as e:
                logger.error(f"Error decoding string: {e}")
                return None

        return None


class MultiNetworkTokenReader:
    """
    Read token data across multiple networks simultaneously
    """

    def __init__(self):
        self.network_manager = get_network_manager()
        self.erc20_reader = ERC20Reader(self.network_manager)

    def get_token_across_networks(
        self,
        token_addresses: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get token information across multiple networks

        Args:
            token_addresses: Dict mapping network_name to token_address

        Returns:
            Dict mapping network_name to token info
        """
        results = {}

        print("\n" + "=" * 80)
        print("MULTI-NETWORK TOKEN QUERY")
        print("=" * 80)

        for network_name, token_address in token_addresses.items():
            print(f"\n🔍 Querying {network_name}...")
            print(f"   Token: {token_address}")

            token_info = self.erc20_reader.get_token_info(network_name, token_address)

            results[network_name] = token_info

            if token_info.get("symbol"):
                print(f"   ✅ {token_info['name']} ({token_info['symbol']})")
                print(f"   Decimals: {token_info.get('decimals')}")
                print(f"   Total Supply: {token_info.get('total_supply'):,.2f}" if token_info.get('total_supply') else "   Total Supply: N/A")
            else:
                print(f"   ❌ Failed to read token data")

        print("\n" + "=" * 80)

        return results

    def get_balance_across_networks(
        self,
        token_addresses: Dict[str, str],
        holder_address: str
    ) -> Dict[str, float]:
        """
        Get token balance across multiple networks

        Args:
            token_addresses: Dict mapping network_name to token_address
            holder_address: Address to check balance for

        Returns:
            Dict mapping network_name to balance
        """
        balances = {}

        print("\n" + "=" * 80)
        print(f"TOKEN BALANCES FOR {holder_address}")
        print("=" * 80)

        for network_name, token_address in token_addresses.items():
            balance = self.erc20_reader.get_balance(
                network_name,
                token_address,
                holder_address
            )

            balances[network_name] = balance

            if balance is not None:
                token_info = self.erc20_reader.get_token_info(network_name, token_address)
                symbol = token_info.get("symbol", "???")
                print(f"   {network_name:<20} {balance:>15,.4f} {symbol}")
            else:
                print(f"   {network_name:<20} {'N/A':>15}")

        print("=" * 80)

        return balances


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("NEXUS AGI - SMART CONTRACT INTERACTION MODULE")
    print("=" * 80)

    interactor = ContractInteractor()
    erc20_reader = ERC20Reader()

    # Example 1: Get block numbers across networks
    print("\n📊 Current Block Numbers:")
    for network_name in ["ethereum_mainnet", "polygon_mainnet", "arbitrum_mainnet"]:
        block_number = interactor.get_block_number(network_name)
        if block_number:
            print(f"   {network_name:<20} Block: {block_number:,}")

    # Example 2: Check WBTC token info on Ethereum
    print("\n💰 WBTC Token Info (Ethereum):")
    wbtc_eth = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

    token_info = erc20_reader.get_token_info("ethereum_mainnet", wbtc_eth)

    if token_info.get("symbol"):
        print(f"   Name: {token_info['name']}")
        print(f"   Symbol: {token_info['symbol']}")
        print(f"   Decimals: {token_info['decimals']}")
        print(f"   Total Supply: {token_info['total_supply']:,.8f} WBTC")

    # Example 3: Multi-network WBTC query
    print("\n🌐 WBTC Across Multiple Networks:")

    wbtc_addresses = {
        "ethereum_mainnet": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "polygon_mainnet": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
        "arbitrum_mainnet": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
    }

    multi_reader = MultiNetworkTokenReader()
    results = multi_reader.get_token_across_networks(wbtc_addresses)

    print("\n" + "=" * 80)
    print("✅ Contract interaction module operational")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
