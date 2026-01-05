#!/usr/bin/env python3
"""
Curl-based RPC Client for Ethereum
Uses curl commands to bypass network restrictions
"""

import json
import subprocess
import time
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurlRPCClient:
    """
    Ethereum RPC client using curl commands
    Bypasses Python network restrictions by using system curl
    """

    def __init__(self, rpc_url: str):
        """
        Initialize curl-based RPC client

        Args:
            rpc_url: Ethereum RPC endpoint URL
        """
        self.rpc_url = rpc_url
        self.request_id = 0

    def _execute_curl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute curl command to make RPC request

        Args:
            data: JSON-RPC request data

        Returns:
            JSON-RPC response
        """
        self.request_id += 1
        data['id'] = self.request_id

        json_data = json.dumps(data)

        # Build curl command
        curl_cmd = [
            'curl',
            '-s',  # Silent mode
            '-X', 'POST',
            '-H', 'Content-Type: application/json',
            '--data', json_data,
            '--connect-timeout', '30',
            '--max-time', '60',
            self.rpc_url
        ]

        try:
            logger.debug(f"Executing curl: {' '.join(curl_cmd)}")
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=65
            )

            if result.returncode != 0:
                logger.error(f"Curl failed with code {result.returncode}: {result.stderr}")
                return {"error": f"Curl command failed: {result.stderr}"}

            response = json.loads(result.stdout)
            return response

        except subprocess.TimeoutExpired:
            logger.error("Curl request timeout")
            return {"error": "Request timeout"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {result.stdout}")
            return {"error": f"Invalid JSON response: {e}"}
        except Exception as e:
            logger.error(f"Curl execution error: {e}")
            return {"error": str(e)}

    def eth_blockNumber(self) -> Optional[int]:
        """Get latest block number"""
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": []
        })

        if 'error' in response:
            logger.error(f"eth_blockNumber error: {response['error']}")
            return None

        if 'result' in response:
            return int(response['result'], 16)

        return None

    def eth_chainId(self) -> Optional[int]:
        """Get chain ID"""
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_chainId",
            "params": []
        })

        if 'error' in response:
            return None

        if 'result' in response:
            return int(response['result'], 16)

        return None

    def eth_getBalance(self, address: str, block: str = "latest") -> Optional[int]:
        """
        Get balance in wei

        Args:
            address: Ethereum address
            block: Block parameter (default: "latest")

        Returns:
            Balance in wei
        """
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, block]
        })

        if 'error' in response:
            logger.error(f"eth_getBalance error: {response['error']}")
            return None

        if 'result' in response:
            return int(response['result'], 16)

        return None

    def eth_gasPrice(self) -> Optional[int]:
        """Get current gas price in wei"""
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_gasPrice",
            "params": []
        })

        if 'error' in response:
            logger.error(f"eth_gasPrice error: {response['error']}")
            return None

        if 'result' in response:
            return int(response['result'], 16)

        return None

    def eth_getTransactionCount(self, address: str, block: str = "latest") -> Optional[int]:
        """
        Get transaction count (nonce)

        Args:
            address: Ethereum address
            block: Block parameter (default: "latest")

        Returns:
            Transaction count
        """
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_getTransactionCount",
            "params": [address, block]
        })

        if 'error' in response:
            logger.error(f"eth_getTransactionCount error: {response['error']}")
            return None

        if 'result' in response:
            return int(response['result'], 16)

        return None

    def eth_estimateGas(self, transaction: Dict[str, str]) -> Optional[int]:
        """
        Estimate gas for transaction

        Args:
            transaction: Transaction object

        Returns:
            Estimated gas
        """
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_estimateGas",
            "params": [transaction]
        })

        if 'error' in response:
            logger.warning(f"eth_estimateGas error: {response['error']}")
            return 21000  # Default gas for simple transfer

        if 'result' in response:
            return int(response['result'], 16)

        return 21000

    def eth_sendRawTransaction(self, signed_tx: str) -> Optional[str]:
        """
        Send raw signed transaction

        Args:
            signed_tx: Signed transaction hex string

        Returns:
            Transaction hash
        """
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_sendRawTransaction",
            "params": [signed_tx]
        })

        if 'error' in response:
            logger.error(f"eth_sendRawTransaction error: {response['error']}")
            return None

        if 'result' in response:
            return response['result']

        return None

    def eth_getTransactionByHash(self, tx_hash: str) -> Optional[Dict]:
        """
        Get transaction by hash

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction object
        """
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_getTransactionByHash",
            "params": [tx_hash]
        })

        if 'error' in response:
            logger.error(f"eth_getTransactionByHash error: {response['error']}")
            return None

        return response.get('result')

    def eth_getTransactionReceipt(self, tx_hash: str) -> Optional[Dict]:
        """
        Get transaction receipt

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction receipt
        """
        response = self._execute_curl({
            "jsonrpc": "2.0",
            "method": "eth_getTransactionReceipt",
            "params": [tx_hash]
        })

        if 'error' in response:
            return None

        return response.get('result')


class CurlEthereumConnector:
    """
    High-level Ethereum connector using curl
    Drop-in replacement for Web3-based connector
    """

    def __init__(self, network: str = "mainnet"):
        """
        Initialize curl-based Ethereum connector

        Args:
            network: 'mainnet', 'sepolia', 'goerli'
        """
        self.network = network
        self.providers = self._get_providers()
        self.rpc_client = None
        self.connected = False

        # Try to connect
        self._connect()

    def _get_providers(self) -> List[str]:
        """Get RPC provider URLs for network"""
        providers = {
            "mainnet": [
                "https://eth.llamarpc.com",
                "https://rpc.ankr.com/eth",
                "https://ethereum.publicnode.com",
                "https://1rpc.io/eth",
                "https://eth.drpc.org",
                "https://rpc.flashbots.net",
                "https://cloudflare-eth.com"
            ],
            "sepolia": [
                "https://rpc.sepolia.org",
                "https://rpc2.sepolia.org",
                "https://ethereum-sepolia.publicnode.com"
            ],
            "goerli": [
                "https://goerli.infura.io/v3/9aa3d95b3bc440fa88ea12eaa4456161",
                "https://rpc.ankr.com/eth_goerli"
            ]
        }
        return providers.get(self.network, providers["mainnet"])

    def _connect(self) -> bool:
        """Try to connect to an RPC provider"""
        for provider in self.providers:
            try:
                logger.info(f"Trying to connect to {provider}...")
                client = CurlRPCClient(provider)

                # Test connection
                block_number = client.eth_blockNumber()
                if block_number is not None:
                    self.rpc_client = client
                    self.connected = True
                    logger.info(f"✓ Connected to {provider}")
                    logger.info(f"✓ Latest block: {block_number}")
                    return True

            except Exception as e:
                logger.warning(f"Failed to connect to {provider}: {e}")
                continue

        logger.error("Failed to connect to any Ethereum provider")
        return False

    def get_balance(self, address: str) -> Decimal:
        """Get ETH balance for address"""
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        balance_wei = self.rpc_client.eth_getBalance(address)
        if balance_wei is None:
            raise Exception("Failed to get balance")

        # Convert wei to ETH
        balance_eth = Decimal(balance_wei) / Decimal(10**18)
        return balance_eth

    def get_gas_price(self) -> int:
        """Get current gas price with buffer"""
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        gas_price = self.rpc_client.eth_gasPrice()
        if gas_price is None:
            # Fallback to 50 Gwei
            return 50 * 10**9

        # Add 20% buffer
        return int(gas_price * 1.2)

    def send_raw_transaction(self, signed_tx_hex: str) -> str:
        """Send raw signed transaction"""
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        tx_hash = self.rpc_client.eth_sendRawTransaction(signed_tx_hex)
        if tx_hash is None:
            raise Exception("Failed to send transaction")

        return tx_hash

    def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction receipt"""
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        return self.rpc_client.eth_getTransactionReceipt(tx_hash)

    def wait_for_confirmation(self, tx_hash: str, confirmations: int = 12, timeout: int = 600) -> bool:
        """Wait for transaction confirmation"""
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        logger.info(f"Waiting for {confirmations} confirmations...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                receipt = self.rpc_client.eth_getTransactionReceipt(tx_hash)

                if receipt:
                    current_block = self.rpc_client.eth_blockNumber()
                    tx_block = int(receipt['blockNumber'], 16)
                    current_confirmations = current_block - tx_block

                    logger.info(f"Confirmations: {current_confirmations}/{confirmations}")

                    if current_confirmations >= confirmations:
                        status = int(receipt.get('status', '0x0'), 16)
                        if status == 1:
                            logger.info(f"✓ Transaction confirmed in block {tx_block}")
                            return True
                        else:
                            logger.error("✗ Transaction failed")
                            return False

                time.sleep(15)

            except Exception as e:
                logger.warning(f"Error checking confirmation: {e}")
                time.sleep(15)

        logger.warning(f"Timeout waiting for confirmation after {timeout}s")
        return False


if __name__ == "__main__":
    # Test curl-based connector
    print("Testing Curl-based Ethereum Connector...\n")

    connector = CurlEthereumConnector(network="mainnet")

    if connector.connected:
        print(f"✓ Connected to Ethereum {connector.network}")

        # Test balance check
        test_address = "0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d"
        print(f"\nChecking balance for {test_address}...")

        balance = connector.get_balance(test_address)
        print(f"Balance: {balance} ETH")

        # Test gas price
        gas_price = connector.get_gas_price()
        gas_price_gwei = gas_price / 10**9
        print(f"Gas Price: {gas_price_gwei:.2f} Gwei")
    else:
        print("✗ Failed to connect to Ethereum network")
